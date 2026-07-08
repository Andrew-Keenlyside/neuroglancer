/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  BoundedByteCache,
  createCrossChunkLinksCaches,
  decodeCrossChunkLinkCell,
  decodeVlenBytesSingleItem,
  readCrossChunkLinks,
  readCrossChunkLinksForChunk,
  readCrossChunkLinksForOwnedChunks,
  type CrossChunkLinksListResult,
} from "#src/datasource/zarr-vectors/cross_chunk_links.js";

const TEXT_ENC = new TextEncoder();

function jsonBytes(obj: unknown): Uint8Array {
  return TEXT_ENC.encode(JSON.stringify(obj));
}

function hexBytes(hex: string): Uint8Array {
  const clean = hex.trim();
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < out.length; ++i) {
    out[i] = parseInt(clean.substr(i * 2, 2), 16);
  }
  return out;
}

/**
 * Raw (pre-vlen-bytes, pre-zstd) 2-record payload for link_width=2:
 * record 0: ci=[0,1] vi=[7,3]; record 1: ci=[1,0] vi=[2,9].
 * Built + verified against numcodecs.vlen.VLenBytes / zstandard in Python
 * (see the ingest-side session notes) — this exact hex is the ground truth.
 */
const RAW_CELL_PAYLOAD_HEX =
  "000107000000000000000300000000000000010002000000000000000900000000000000";

/**
 * The same payload wrapped in a numcodecs vlen-bytes single-item frame
 * (4-byte item-count=1, 4-byte length, payload), matching what
 * decodeVlenBytesSingleItem expects as input (i.e. already zstd-decoded).
 */
const VLEN_WRAPPED_HEX =
  "0100000024000000" + RAW_CELL_PAYLOAD_HEX;

/**
 * The vlen-wrapped bytes above, zstd-compressed (level 0) — the exact bytes
 * a real shard cell would contain on disk.  Verified round-trip in Python.
 */
const ZSTD_CELL_HEX =
  "28b52ffd202c150100d0010000002400000000010700030001000200090000000000000003140003042701";

/**
 * A minimal single-cell shard (shard_shape = [1,1,1,1,1,1], numCells=1,
 * index_location="end"): [compressed cell][offset=0,length u64 LE][4-byte
 * checksum placeholder — this reader does not verify crc32c].
 */
function buildMinimalShard(compressedCell: Uint8Array): Uint8Array {
  const offset = 0;
  const length = compressedCell.byteLength;
  const out = new Uint8Array(compressedCell.byteLength + 16 + 4);
  out.set(compressedCell, 0);
  const dv = new DataView(out.buffer);
  dv.setBigUint64(compressedCell.byteLength, BigInt(offset), true);
  dv.setBigUint64(compressedCell.byteLength + 8, BigInt(length), true);
  // Trailing 4 bytes (checksum) left as zero — unverified by this reader.
  return out;
}

describe("decodeVlenBytesSingleItem", () => {
  it("decodes a real numcodecs vlen-bytes single-item frame", () => {
    const wrapped = hexBytes(VLEN_WRAPPED_HEX);
    const payload = decodeVlenBytesSingleItem(wrapped);
    expect(Array.from(payload)).toEqual(Array.from(hexBytes(RAW_CELL_PAYLOAD_HEX)));
  });

  it("throws when item count is not 1", () => {
    const buf = new Uint8Array(12);
    new DataView(buf.buffer).setUint32(0, 2, true); // n_items = 2
    expect(() => decodeVlenBytesSingleItem(buf)).toThrow(/exactly 1/);
  });

  it("throws on truncated buffer", () => {
    const buf = new Uint8Array(8);
    const dv = new DataView(buf.buffer);
    dv.setUint32(0, 1, true); // n_items = 1
    dv.setUint32(4, 100, true); // declared length 100, but buffer ends here
    expect(() => decodeVlenBytesSingleItem(buf)).toThrow(/truncated/);
  });
});

describe("decodeCrossChunkLinkCell", () => {
  it("decodes a 2-record, link_width=2 cell against its K=2 sorted chunks", () => {
    const payload = hexBytes(RAW_CELL_PAYLOAD_HEX);
    const sortedChunks = [
      [0, 0, 0],
      [1, 0, 0],
    ];
    const records = decodeCrossChunkLinkCell(payload, sortedChunks, 2);
    expect(records).toHaveLength(2);
    // record 0: ci=[0,1] vi=[7,3] -> endpoint0 at sortedChunks[0]:7, endpoint1 at sortedChunks[1]:3
    expect(records[0].endpoints[0]).toEqual({ chunkCoords: [0, 0, 0], vertexIndex: 7 });
    expect(records[0].endpoints[1]).toEqual({ chunkCoords: [1, 0, 0], vertexIndex: 3 });
    // record 1: ci=[1,0] vi=[2,9] -> endpoint0 at sortedChunks[1]:2, endpoint1 at sortedChunks[0]:9
    expect(records[1].endpoints[0]).toEqual({ chunkCoords: [1, 0, 0], vertexIndex: 2 });
    expect(records[1].endpoints[1]).toEqual({ chunkCoords: [0, 0, 0], vertexIndex: 9 });
  });

  it("throws when payload length is not a multiple of one record", () => {
    const payload = new Uint8Array(5);
    expect(() =>
      decodeCrossChunkLinkCell(payload, [[0], [1]], 2),
    ).toThrow(/not a multiple of one record/);
  });

  it("throws when a ci value is out of range for K", () => {
    // link_width=1: 1 uint8 ci + 1 int64 vi = 9 bytes; ci=5 is out of range for K=1.
    const buf = new Uint8Array(9);
    buf[0] = 5;
    expect(() => decodeCrossChunkLinkCell(buf, [[0, 0, 0]], 1)).toThrow(
      /out of range/,
    );
  });
});

describe("readCrossChunkLinks (v0.8 sharded layout)", () => {
  function makeStore(opts: {
    groupAttrs?: any;
    kMetaByK?: Record<number, any>;
    shardsByPath?: Record<string, Uint8Array>;
    listByPrefix?: Record<string, CrossChunkLinksListResult>;
  }) {
    const reads: string[] = [];
    const kvStoreRead = async (
      subpath: string,
    ): Promise<Uint8Array | undefined> => {
      reads.push(subpath);
      if (subpath === "cross_chunk_links/0/zarr.json") {
        return opts.groupAttrs === undefined
          ? undefined
          : jsonBytes({ attributes: opts.groupAttrs });
      }
      const kMatch = subpath.match(/^cross_chunk_links\/0\/k(\d+)\/zarr\.json$/);
      if (kMatch !== null) {
        const K = Number(kMatch[1]);
        const meta = opts.kMetaByK?.[K];
        return meta === undefined ? undefined : jsonBytes(meta);
      }
      return opts.shardsByPath?.[subpath];
    };
    const kvStoreList = async (
      prefix: string,
    ): Promise<CrossChunkLinksListResult> => {
      return opts.listByPrefix?.[prefix] ?? { directories: [], files: [] };
    };
    return { kvStoreRead, kvStoreList, reads };
  }

  const K2_META = {
    attributes: {
      zv_array: "cross_chunk_links_k",
      level_delta: 0,
      K: 2,
      sid_ndim: 3,
      link_width: 2,
      shard_shape: [1, 1, 1, 1, 1, 1],
      chunk_origin: [0, 0, 0],
    },
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: [1, 1, 1, 1, 1, 1],
          codecs: [{ name: "vlen-bytes" }, { name: "zstd" }],
          index_codecs: [{ name: "bytes" }, { name: "crc32c" }],
          index_location: "end",
        },
      },
    ],
  };

  it("returns undefined when the group is absent", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({});
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    expect(table).toBeUndefined();
  });

  it("throws on a genuine pre-0.8 single-blob layout ('num_links' present)", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: { link_width: 2, sid_ndim: 3, num_links: 5 }, // v0.7 tell-tale
    });
    await expect(
      readCrossChunkLinks(
        { kvStoreRead, kvStoreList },
        new AbortController().signal,
      ),
    ).rejects.toThrow(/pre-0.8 single-blob/);
  });

  it("throws on an explicitly unsupported layout value", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: { link_width: 2, sid_ndim: 3, layout: "something_else" },
    });
    await expect(
      readCrossChunkLinks(
        { kvStoreRead, kvStoreList },
        new AbortController().signal,
      ),
    ).rejects.toThrow(/unsupported cross_chunk_links layout/);
  });

  it("tolerates a missing 'layout' field (writers that predate the fix)", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: { link_width: 2, sid_ndim: 3 }, // no layout, no num_links
    });
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    expect(table?.records).toEqual([]);
  });

  it("returns an empty table when no kK arrays exist (all lazily absent)", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: { link_width: 2, sid_ndim: 3, layout: "sharded_v1" },
    });
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    expect(table?.records).toEqual([]);
    expect(table?.linkWidth).toBe(2);
    expect(table?.sidNdim).toBe(3);
  });

  it("decodes one populated shard end-to-end (real vlen-bytes+zstd bytes)", async () => {
    const compressedCell = hexBytes(ZSTD_CELL_HEX);
    const shard = buildMinimalShard(compressedCell);
    const shardPath = "cross_chunk_links/0/k2/c/0/0/0/1/0/0";
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: { link_width: 2, sid_ndim: 3, layout: "sharded_v1" },
      kMetaByK: { 2: K2_META },
      shardsByPath: { [shardPath]: shard },
      listByPrefix: {
        "cross_chunk_links/0/k2/c/": { directories: ["0"], files: [] },
        "cross_chunk_links/0/k2/c/0/": { directories: ["0"], files: [] },
        "cross_chunk_links/0/k2/c/0/0/": { directories: ["0"], files: [] },
        "cross_chunk_links/0/k2/c/0/0/0/": { directories: ["1"], files: [] },
        "cross_chunk_links/0/k2/c/0/0/0/1/": { directories: ["0"], files: [] },
        // Last level: the leaf shard file reported as a "file" at this prefix.
        "cross_chunk_links/0/k2/c/0/0/0/1/0/": { directories: [], files: ["0"] },
      },
    });
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    expect(table).toBeDefined();
    expect(table!.linkWidth).toBe(2);
    expect(table!.sidNdim).toBe(3);
    expect(table!.records).toHaveLength(2);
    // sortedChunks for cell coord (0,0,0, 1,0,0) = [(0,0,0), (1,0,0)].
    expect(table!.records[0].endpoints[0]).toEqual({
      chunkCoords: [0, 0, 0],
      vertexIndex: 7,
    });
    expect(table!.records[0].endpoints[1]).toEqual({
      chunkCoords: [1, 0, 0],
      vertexIndex: 3,
    });
    expect(table!.records[1].endpoints[0]).toEqual({
      chunkCoords: [1, 0, 0],
      vertexIndex: 2,
    });
    expect(table!.records[1].endpoints[1]).toEqual({
      chunkCoords: [0, 0, 0],
      vertexIndex: 9,
    });
  });

  it("applies chunk_origin when resolving cell coordinates", async () => {
    const compressedCell = hexBytes(ZSTD_CELL_HEX);
    const shard = buildMinimalShard(compressedCell);
    const originMeta = {
      ...K2_META,
      attributes: { ...K2_META.attributes, chunk_origin: [-2, 0, 0] },
    };
    const shardPath = "cross_chunk_links/0/k2/c/0/0/0/1/0/0";
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: { link_width: 2, sid_ndim: 3, layout: "sharded_v1" },
      kMetaByK: { 2: originMeta },
      shardsByPath: { [shardPath]: shard },
      listByPrefix: {
        "cross_chunk_links/0/k2/c/": { directories: ["0"], files: [] },
        "cross_chunk_links/0/k2/c/0/": { directories: ["0"], files: [] },
        "cross_chunk_links/0/k2/c/0/0/": { directories: ["0"], files: [] },
        "cross_chunk_links/0/k2/c/0/0/0/": { directories: ["1"], files: [] },
        "cross_chunk_links/0/k2/c/0/0/0/1/": { directories: ["0"], files: [] },
        "cross_chunk_links/0/k2/c/0/0/0/1/0/": { directories: [], files: ["0"] },
      },
    });
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    // Cell coord (0,0,0, 1,0,0) + origin (-2,0,0) per K-group -> (-2,0,0), (-1,0,0).
    expect(table!.records[0].endpoints[0].chunkCoords).toEqual([-2, 0, 0]);
    expect(table!.records[0].endpoints[1].chunkCoords).toEqual([-1, 0, 0]);
  });

  it("throws when a kK array is populated but no kvStoreList is provided", async () => {
    const { kvStoreRead } = makeStore({
      groupAttrs: { link_width: 2, sid_ndim: 3, layout: "sharded_v1" },
      kMetaByK: { 2: K2_META },
    });
    await expect(
      readCrossChunkLinks({ kvStoreRead }, new AbortController().signal),
    ).rejects.toThrow(/requires a kvStoreList callback/);
  });

  it("throws on invalid link_width / sid_ndim", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: { link_width: 0, sid_ndim: 3, layout: "sharded_v1" },
    });
    await expect(
      readCrossChunkLinks(
        { kvStoreRead, kvStoreList },
        new AbortController().signal,
      ),
    ).rejects.toThrow(/link_width/);
  });

  it("caps concurrent shard-list requests when the tree is wide", async () => {
    // K=1 (ndim = sid_ndim*K = 3) so the root listing's children are NOT
    // the last level — each of them triggers its own recursive
    // ``kvStoreList`` call, producing a genuine concurrent burst.
    const K1_META = {
      attributes: {
        zv_array: "cross_chunk_links_k",
        level_delta: 0,
        K: 1,
        sid_ndim: 3,
        link_width: 1,
        shard_shape: [1, 1, 1],
        chunk_origin: [0, 0, 0],
      },
      codecs: [
        {
          name: "sharding_indexed",
          configuration: {
            chunk_shape: [1, 1, 1],
            codecs: [{ name: "vlen-bytes" }, { name: "zstd" }],
            index_codecs: [{ name: "bytes" }, { name: "crc32c" }],
            index_location: "end",
          },
        },
      ],
    };

    const WIDTH = 30;
    const rootDirs = Array.from({ length: WIDTH }, (_, i) => String(i));
    const listByPrefix: Record<string, CrossChunkLinksListResult> = {
      "cross_chunk_links/0/k1/c/": { directories: rootDirs, files: [] },
    };
    for (const name of rootDirs) {
      listByPrefix[`cross_chunk_links/0/k1/c/${name}/`] = {
        directories: [],
        files: [],
      };
    }

    const { kvStoreRead } = makeStore({
      groupAttrs: { link_width: 1, sid_ndim: 3, layout: "sharded_v1" },
      kMetaByK: { 1: K1_META },
    });

    let active = 0;
    let maxActive = 0;
    const kvStoreList = async (
      prefix: string,
    ): Promise<CrossChunkLinksListResult> => {
      active++;
      maxActive = Math.max(maxActive, active);
      await new Promise((resolve) => setTimeout(resolve, 0));
      active--;
      return listByPrefix[prefix] ?? { directories: [], files: [] };
    };

    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    expect(table?.records).toEqual([]);
    // Sanity: the burst really happened (more requests than the cap)...
    expect(WIDTH).toBeGreaterThan(8);
    // ...but concurrency never exceeded the limiter's cap.
    expect(maxActive).toBeLessThanOrEqual(8);
  });

  it("respects the delta argument when forming subpaths", async () => {
    const reads: string[] = [];
    const kvStoreRead = async (
      subpath: string,
    ): Promise<Uint8Array | undefined> => {
      reads.push(subpath);
      if (subpath === "cross_chunk_links/-1/zarr.json") {
        return jsonBytes({
          attributes: { link_width: 2, sid_ndim: 3, layout: "sharded_v1" },
        });
      }
      return undefined;
    };
    const kvStoreList = async (): Promise<CrossChunkLinksListResult> => ({
      directories: [],
      files: [],
    });
    await readCrossChunkLinks(
      { kvStoreRead, kvStoreList, delta: -1 },
      new AbortController().signal,
    );
    expect(reads[0]).toBe("cross_chunk_links/-1/zarr.json");
  });
});

describe("readCrossChunkLinksForChunk", () => {
  const K2_META = {
    attributes: {
      zv_array: "cross_chunk_links_k",
      level_delta: 0,
      K: 2,
      sid_ndim: 3,
      link_width: 2,
      shard_shape: [1, 1, 1, 1, 1, 1],
      chunk_origin: [0, 0, 0],
    },
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: [1, 1, 1, 1, 1, 1],
          codecs: [{ name: "vlen-bytes" }, { name: "zstd" }],
          index_codecs: [{ name: "bytes" }, { name: "crc32c" }],
          index_location: "end",
        },
      },
    ],
  };

  // Two shards (one cell each, per shard_shape=[1,1,1,1,1,1]): one pairs
  // chunk (0,0,0) with (1,0,0), the other pairs (0,0,0) with (2,0,0) --
  // both reuse the same real zstd-compressed 2-record payload, since the
  // filtering logic under test only cares about *which chunks* a cell
  // resolves to, not its record content.
  function makeTwoShardStore() {
    const reads: string[] = [];
    const lists: string[] = [];
    const shard = buildMinimalShard(hexBytes(ZSTD_CELL_HEX));
    const shardsByPath: Record<string, Uint8Array> = {
      "cross_chunk_links/0/k2/c/0/0/0/1/0/0": shard,
      "cross_chunk_links/0/k2/c/0/0/0/2/0/0": shard,
    };
    const listByPrefix: Record<string, CrossChunkLinksListResult> = {
      "cross_chunk_links/0/k2/c/": { directories: ["0"], files: [] },
      "cross_chunk_links/0/k2/c/0/": { directories: ["0"], files: [] },
      "cross_chunk_links/0/k2/c/0/0/": { directories: ["0"], files: [] },
      "cross_chunk_links/0/k2/c/0/0/0/": {
        directories: ["1", "2"],
        files: [],
      },
      "cross_chunk_links/0/k2/c/0/0/0/1/": { directories: ["0"], files: [] },
      "cross_chunk_links/0/k2/c/0/0/0/1/0/": { directories: [], files: ["0"] },
      "cross_chunk_links/0/k2/c/0/0/0/2/": { directories: ["0"], files: [] },
      "cross_chunk_links/0/k2/c/0/0/0/2/0/": { directories: [], files: ["0"] },
    };
    const kvStoreRead = async (
      subpath: string,
    ): Promise<Uint8Array | undefined> => {
      reads.push(subpath);
      if (subpath === "cross_chunk_links/0/zarr.json") {
        return jsonBytes({
          attributes: { link_width: 2, sid_ndim: 3, layout: "sharded_v1" },
        });
      }
      if (subpath === "cross_chunk_links/0/k2/zarr.json") {
        return jsonBytes(K2_META);
      }
      return shardsByPath[subpath];
    };
    const kvStoreList = async (
      prefix: string,
    ): Promise<CrossChunkLinksListResult> => {
      lists.push(prefix);
      return listByPrefix[prefix] ?? { directories: [], files: [] };
    };
    return { kvStoreRead, kvStoreList, reads, lists };
  }

  it("decodes only the records touching the target chunk", async () => {
    const { kvStoreRead, kvStoreList } = makeTwoShardStore();
    const caches = createCrossChunkLinksCaches();
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      new AbortController().signal,
    );
    expect(table!.records).toHaveLength(2);
    for (const record of table!.records) {
      expect(record.endpoints.map((e) => e.chunkCoords)).toContainEqual([
        1, 0, 0,
      ]);
    }
  });

  it("returns records from the other shard for a different target chunk", async () => {
    const { kvStoreRead, kvStoreList } = makeTwoShardStore();
    const caches = createCrossChunkLinksCaches();
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [2, 0, 0],
      caches,
      new AbortController().signal,
    );
    expect(table!.records).toHaveLength(2);
    for (const record of table!.records) {
      expect(record.endpoints.map((e) => e.chunkCoords)).toContainEqual([
        2, 0, 0,
      ]);
    }
  });

  it("returns an empty table for a chunk with no incident records", async () => {
    const { kvStoreRead, kvStoreList } = makeTwoShardStore();
    const caches = createCrossChunkLinksCaches();
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [99, 99, 99],
      caches,
      new AbortController().signal,
    );
    expect(table!.records).toEqual([]);
  });

  it("returns records from every shard touching a chunk shared by multiple cells", async () => {
    const { kvStoreRead, kvStoreList } = makeTwoShardStore();
    const caches = createCrossChunkLinksCaches();
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [0, 0, 0],
      caches,
      new AbortController().signal,
    );
    // (0,0,0) is one endpoint of every record in both shards.
    expect(table!.records).toHaveLength(4);
  });

  it("caches shard bytes across queries for different target chunks", async () => {
    const { kvStoreRead, kvStoreList, reads } = makeTwoShardStore();
    const caches = createCrossChunkLinksCaches();
    const signal = new AbortController().signal;

    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      signal,
    );
    const shardReadsAfterFirst = reads.filter((r) => r.includes("/c/")).length;
    expect(shardReadsAfterFirst).toBeGreaterThan(0);

    // Second query, same target chunk: shard bytes for the (already
    // fetched) matching shard must not be re-read.
    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      signal,
    );
    expect(reads.filter((r) => r.includes("/c/")).length).toBe(
      shardReadsAfterFirst,
    );
  });

  it("repeat queries for the same target chunk reuse the cached discovery walk", async () => {
    const { kvStoreRead, kvStoreList, lists } = makeTwoShardStore();
    const caches = createCrossChunkLinksCaches();
    const signal = new AbortController().signal;

    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      signal,
    );
    const listsAfterFirst = lists.length;
    expect(listsAfterFirst).toBeGreaterThan(0);

    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      signal,
    );
    expect(lists.length).toBe(listsAfterFirst);
  });

  it("prunes discovery to subtrees that could contain the target chunk", async () => {
    // Target (1,0,0) can only ever appear in the "1/0/0" branch off the
    // shared (0,0,0) prefix — the "2/0/0" branch (holding chunk (2,0,0),
    // not (1,0,0), in both K-slots) must never be listed at all.
    const { kvStoreRead, kvStoreList, lists } = makeTwoShardStore();
    const caches = createCrossChunkLinksCaches();
    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      new AbortController().signal,
    );
    expect(lists).not.toContain("cross_chunk_links/0/k2/c/0/0/0/2/");
    expect(lists).not.toContain("cross_chunk_links/0/k2/c/0/0/0/2/0/");
    expect(lists).toContain("cross_chunk_links/0/k2/c/0/0/0/1/");
  });

  it("concurrent queries for the same target chunk share one discovery walk", async () => {
    // Regression test for a cache-stampede: before results are cached,
    // several concurrent download() calls for the same chunk must not
    // each independently kick off their own listPopulatedShardCoords walk.
    const { kvStoreRead, kvStoreList, lists } = makeTwoShardStore();
    const caches = createCrossChunkLinksCaches();
    const signal = new AbortController().signal;

    await Promise.all([
      readCrossChunkLinksForChunk(
        { kvStoreRead, kvStoreList },
        [1, 0, 0],
        caches,
        signal,
      ),
      readCrossChunkLinksForChunk(
        { kvStoreRead, kvStoreList },
        [1, 0, 0],
        caches,
        signal,
      ),
      readCrossChunkLinksForChunk(
        { kvStoreRead, kvStoreList },
        [1, 0, 0],
        caches,
        signal,
      ),
    ]);
    const uniqueLists = new Set(lists);
    expect(lists.length).toBe(uniqueLists.size);
  });

  it("matches readCrossChunkLinks's full-table result filtered to one chunk", async () => {
    // Cross-check against the whole-table reader: for a given target
    // chunk, the two entry points must agree exactly.
    const { kvStoreRead, kvStoreList } = makeTwoShardStore();
    const signal = new AbortController().signal;

    const fullTable = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      signal,
    );
    const expected = fullTable!.records.filter((r) =>
      r.endpoints.some(
        (e) => e.chunkCoords[0] === 1 && e.chunkCoords[1] === 0 &&
          e.chunkCoords[2] === 0,
      ),
    );

    const caches = createCrossChunkLinksCaches();
    const targeted = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      signal,
    );
    expect(targeted!.records).toEqual(expected);
  });
});

describe("BoundedByteCache", () => {
  it("returns undefined for a missing key", () => {
    const cache = new BoundedByteCache(1000);
    expect(cache.get("missing")).toBeUndefined();
  });

  it("returns a stored value", () => {
    const cache = new BoundedByteCache(1000);
    const value = new Uint8Array([1, 2, 3]);
    cache.set("a", value);
    expect(cache.get("a")).toBe(value);
  });

  it("evicts the least-recently-used entry once the byte budget is exceeded", () => {
    const cache = new BoundedByteCache(10);
    cache.set("a", new Uint8Array(4));
    cache.set("b", new Uint8Array(4));
    cache.set("c", new Uint8Array(4)); // total would be 12 > 10 -> evict "a"
    expect(cache.get("a")).toBeUndefined();
    expect(cache.get("b")).toBeDefined();
    expect(cache.get("c")).toBeDefined();
  });

  it("treats a get() as a recency touch, protecting it from eviction", () => {
    const cache = new BoundedByteCache(10);
    cache.set("a", new Uint8Array(4));
    cache.set("b", new Uint8Array(4));
    cache.get("a"); // "a" is now more recently used than "b"
    cache.set("c", new Uint8Array(4)); // evicts "b", not "a"
    expect(cache.get("a")).toBeDefined();
    expect(cache.get("b")).toBeUndefined();
    expect(cache.get("c")).toBeDefined();
  });
});

describe("readCrossChunkLinksForOwnedChunks (go-direct)", () => {
  const K2_META = {
    attributes: {
      zv_array: "cross_chunk_links_k",
      level_delta: 0,
      K: 2,
      sid_ndim: 3,
      link_width: 2,
      shard_shape: [1, 1, 1, 1, 1, 1],
      chunk_origin: [0, 0, 0],
    },
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: [1, 1, 1, 1, 1, 1],
          codecs: [{ name: "vlen-bytes" }, { name: "zstd" }],
          index_codecs: [{ name: "bytes" }, { name: "crc32c" }],
          index_location: "end",
        },
      },
    ],
  };

  // The end-to-end fixture's populated cell is at grid (0,0,0, 1,0,0) =
  // chunks [(0,0,0), (1,0,0)], holding two records.
  const SHARD_PATH = "cross_chunk_links/0/k2/c/0/0/0/1/0/0";

  function makeStore() {
    const reads: string[] = [];
    const listPrefixes: string[] = [];
    const shard = buildMinimalShard(hexBytes(ZSTD_CELL_HEX));
    const kvStoreRead = async (subpath: string) => {
      reads.push(subpath);
      if (subpath === "cross_chunk_links/0/zarr.json") {
        return jsonBytes({
          attributes: { link_width: 2, sid_ndim: 3, layout: "sharded_v1" },
        });
      }
      if (subpath === "cross_chunk_links/0/k1/zarr.json") return undefined;
      if (subpath === "cross_chunk_links/0/k2/zarr.json") {
        return jsonBytes(K2_META);
      }
      if (subpath === SHARD_PATH) return shard;
      return undefined;
    };
    const kvStoreList = async (prefix: string): Promise<CrossChunkLinksListResult> => {
      listPrefixes.push(prefix); // must never be called by go-direct
      return { directories: [], files: [] };
    };
    return { kvStoreRead, kvStoreList, reads, listPrefixes };
  }

  it("reads the owned pair's cell directly, with NO directory listing", async () => {
    const { kvStoreRead, kvStoreList, listPrefixes } = makeStore();
    const table = await readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreList },
      [
        [0, 0, 0],
        [1, 0, 0],
      ],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table).toBeDefined();
    expect(table!.records).toHaveLength(2);
    expect(table!.records[0].endpoints[0]).toEqual({
      chunkCoords: [0, 0, 0],
      vertexIndex: 7,
    });
    // Go-direct computes the shard coordinate from the owned pair — it must
    // never walk the chunk-key directory tree.
    expect(listPrefixes).toEqual([]);
  });

  it("skips cells whose partner chunk is not owned (no read, empty result)", async () => {
    const { kvStoreRead, kvStoreList, reads } = makeStore();
    // Only (0,0,0) owned: with K=2 there is no owned-owned pair, so no
    // candidate cell -> no shard read at all.
    const table = await readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreList },
      [[0, 0, 0]],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table!.records).toEqual([]);
    expect(reads).not.toContain(SHARD_PATH);
  });

  it("caches the header across queries (no repeated zarr.json / k1 404)", async () => {
    const { kvStoreRead, kvStoreList, reads } = makeStore();
    const caches = createCrossChunkLinksCaches();
    const owned = [
      [0, 0, 0],
      [1, 0, 0],
    ];
    const signal = new AbortController().signal;
    await readCrossChunkLinksForOwnedChunks({ kvStoreRead, kvStoreList }, owned, caches, signal);
    await readCrossChunkLinksForOwnedChunks({ kvStoreRead, kvStoreList }, owned, caches, signal);
    // group + k1 (404) + k2 zarr.json each fetched exactly once across both queries.
    expect(reads.filter((r) => r === "cross_chunk_links/0/zarr.json")).toHaveLength(1);
    expect(reads.filter((r) => r === "cross_chunk_links/0/k1/zarr.json")).toHaveLength(1);
    expect(reads.filter((r) => r === "cross_chunk_links/0/k2/zarr.json")).toHaveLength(1);
  });
});
