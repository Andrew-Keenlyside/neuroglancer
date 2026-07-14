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
  decodeRaggedBlobInt64Records,
  lehmerDecode,
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
 * Raw "inline-header ragged blob" (``encode_ragged_blob``) for two
 * link_width=2 records: record 0 = [perm_idx=0, vi=[7,3]] (identity);
 * record 1 = [perm_idx=1, vi=[2,9]] (the L=2 swap permutation). Built +
 * verified against Python's own encoder/`zstandard` (see the ingest-side
 * session notes) — this exact hex is the ground truth.
 */
const RAW_BLOB_HEX =
  "020000000000000000000000000000001800000000000000000000000000000007000000000000000300000000000000010000000000000002000000000000000900000000000000";

/**
 * The same blob, zstd-compressed (level 0) — the exact bytes a real cell
 * file would contain on disk (CCL cells are NOT vlen-bytes-framed, unlike
 * the per-chunk-array shard cells in `sharded_array.ts`).
 */
const ZSTD_CELL_HEX =
  "28b52ffd2048f50000a0020018000700030001000200090000000000000006500200638c419603";

describe("decodeRaggedBlobInt64Records", () => {
  it("decodes a real ragged blob into fixed-stride records", () => {
    const rows = decodeRaggedBlobInt64Records(hexBytes(RAW_BLOB_HEX), 3);
    expect(rows).toHaveLength(2);
    expect(Array.from(rows[0])).toEqual([0n, 7n, 3n]);
    expect(Array.from(rows[1])).toEqual([1n, 2n, 9n]);
  });

  it("returns an empty array for a too-short buffer", () => {
    expect(decodeRaggedBlobInt64Records(new Uint8Array(4), 3)).toEqual([]);
  });

  it("returns an empty array when the record count is zero", () => {
    const buf = new Uint8Array(8);
    new DataView(buf.buffer).setBigInt64(0, 0n, true);
    expect(decodeRaggedBlobInt64Records(buf, 3)).toEqual([]);
  });

  it("throws when a record's byte span doesn't match ncols", () => {
    // k=1, offsets=[0], data is only 8 bytes (1 int64) but ncols=3 expects 24.
    const buf = new Uint8Array(24);
    const dv = new DataView(buf.buffer);
    dv.setBigInt64(0, 1n, true); // k=1
    dv.setBigInt64(8, 0n, true); // offsets[0]=0
    dv.setBigInt64(16, 42n, true); // 8 bytes of "data"
    expect(() => decodeRaggedBlobInt64Records(buf, 3)).toThrow(/is 8 bytes/);
  });
});

describe("lehmerDecode", () => {
  it("decodes the identity permutation (code 0) for any L", () => {
    expect(lehmerDecode(0, 1)).toEqual([0]);
    expect(lehmerDecode(0, 2)).toEqual([0, 1]);
    expect(lehmerDecode(0, 3)).toEqual([0, 1, 2]);
  });

  it("decodes the swap permutation for L=2", () => {
    expect(lehmerDecode(1, 2)).toEqual([1, 0]);
  });

  it("decodes every permutation of L=3 uniquely", () => {
    const seen = new Set<string>();
    for (let code = 0; code < 6; ++code) {
      const perm = lehmerDecode(code, 3);
      expect(perm.slice().sort()).toEqual([0, 1, 2]);
      seen.add(perm.join(","));
    }
    expect(seen.size).toBe(6);
  });

  it("throws when code is out of range", () => {
    expect(() => lehmerDecode(2, 2)).toThrow(/out of range/);
    expect(() => lehmerDecode(-1, 2)).toThrow(/out of range/);
  });
});

describe("decodeCrossChunkLinkCell", () => {
  const cellChunks = [
    [0, 0, 0],
    [1, 0, 0],
  ];

  it("decodes a 2-record cell, recovering original endpoint order via perm_idx", () => {
    const payload = hexBytes(RAW_BLOB_HEX);
    const records = decodeCrossChunkLinkCell(payload, cellChunks, 2);
    expect(records).toHaveLength(2);
    // record 0: perm_idx=0 (identity) -> endpoints in cell order.
    expect(records[0].endpoints[0]).toEqual({ chunkCoords: [0, 0, 0], vertexIndex: 7 });
    expect(records[0].endpoints[1]).toEqual({ chunkCoords: [1, 0, 0], vertexIndex: 3 });
    // record 1: perm_idx=1 (swap) -> endpoints[1]=sorted[0], endpoints[0]=sorted[1].
    expect(records[1].endpoints[0]).toEqual({ chunkCoords: [1, 0, 0], vertexIndex: 9 });
    expect(records[1].endpoints[1]).toEqual({ chunkCoords: [0, 0, 0], vertexIndex: 2 });
  });

  it("throws when cellChunks length does not match linkWidth", () => {
    expect(() =>
      decodeCrossChunkLinkCell(new Uint8Array(0), [[0, 0, 0]], 2),
    ).toThrow(/expected link_width=2/);
  });
});

describe("readCrossChunkLinks (0.8.1 flat-cell-key layout)", () => {
  function makeStore(opts: {
    groupAttrs?: any;
    cellsByKey?: Record<string, Uint8Array>;
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
      const cellMatch = subpath.match(/^cross_chunk_links\/0\/([^/]+)\/c\/0$/);
      if (cellMatch !== null) {
        return opts.cellsByKey?.[cellMatch[1]];
      }
      return undefined;
    };
    const kvStoreList = async (
      prefix: string,
    ): Promise<CrossChunkLinksListResult> => {
      return opts.listByPrefix?.[prefix] ?? { directories: [], files: [] };
    };
    return { kvStoreRead, kvStoreList, reads };
  }

  const GROUP_ATTRS = { link_width: 2, sid_ndim: 3, directed: true, store: "canonical" };

  it("returns undefined when the group is absent", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({});
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    expect(table).toBeUndefined();
  });

  it("returns an empty table when no cells exist", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: GROUP_ATTRS,
      listByPrefix: { "cross_chunk_links/0/": { directories: [], files: [] } },
    });
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    expect(table?.records).toEqual([]);
    expect(table?.linkWidth).toBe(2);
    expect(table?.sidNdim).toBe(3);
  });

  it("decodes cells across the level, real zstd-compressed bytes", async () => {
    const cellBytes = hexBytes(ZSTD_CELL_HEX);
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: GROUP_ATTRS,
      cellsByKey: { "0.0.0.1.0.0": cellBytes },
      listByPrefix: {
        "cross_chunk_links/0/": { directories: ["0.0.0.1.0.0"], files: [] },
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
    expect(table!.records[0].endpoints[0]).toEqual({
      chunkCoords: [0, 0, 0],
      vertexIndex: 7,
    });
    expect(table!.records[0].endpoints[1]).toEqual({
      chunkCoords: [1, 0, 0],
      vertexIndex: 3,
    });
  });

  it("throws on invalid link_width / sid_ndim", async () => {
    const { kvStoreRead, kvStoreList } = makeStore({
      groupAttrs: { link_width: 0, sid_ndim: 3 },
    });
    await expect(
      readCrossChunkLinks(
        { kvStoreRead, kvStoreList },
        new AbortController().signal,
      ),
    ).rejects.toThrow(/link_width/);
  });

  it("throws when the level is populated but no kvStoreList is provided", async () => {
    const { kvStoreRead } = makeStore({ groupAttrs: GROUP_ATTRS });
    await expect(
      readCrossChunkLinks({ kvStoreRead }, new AbortController().signal),
    ).rejects.toThrow(/requires a kvStoreList callback/);
  });

  it("respects the delta argument when forming subpaths", async () => {
    const reads: string[] = [];
    const kvStoreRead = async (
      subpath: string,
    ): Promise<Uint8Array | undefined> => {
      reads.push(subpath);
      if (subpath === "cross_chunk_links/-1/zarr.json") {
        return jsonBytes({ attributes: GROUP_ATTRS });
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
  const GROUP_ATTRS = { link_width: 2, sid_ndim: 3, directed: true, store: "canonical" };

  // Two cells (each a real zstd-compressed 2-record payload — content
  // doesn't matter for the filtering logic under test, only which
  // chunks each cell's key resolves to): one pairs (0,0,0) with (1,0,0),
  // the other pairs (0,0,0) with (2,0,0).
  function makeTwoCellStore() {
    const reads: string[] = [];
    const lists: string[] = [];
    const cellBytes = hexBytes(ZSTD_CELL_HEX);
    const cellsByKey: Record<string, Uint8Array> = {
      "0.0.0.1.0.0": cellBytes,
      "0.0.0.2.0.0": cellBytes,
    };
    const kvStoreRead = async (
      subpath: string,
    ): Promise<Uint8Array | undefined> => {
      reads.push(subpath);
      if (subpath === "cross_chunk_links/0/zarr.json") {
        return jsonBytes({ attributes: GROUP_ATTRS });
      }
      const cellMatch = subpath.match(/^cross_chunk_links\/0\/([^/]+)\/c\/0$/);
      if (cellMatch !== null) return cellsByKey[cellMatch[1]];
      return undefined;
    };
    const kvStoreList = async (
      prefix: string,
    ): Promise<CrossChunkLinksListResult> => {
      lists.push(prefix);
      if (prefix === "cross_chunk_links/0/") {
        return { directories: Object.keys(cellsByKey), files: [] };
      }
      return { directories: [], files: [] };
    };
    return { kvStoreRead, kvStoreList, reads, lists };
  }

  it("decodes only the records touching the target chunk", async () => {
    const { kvStoreRead, kvStoreList } = makeTwoCellStore();
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

  it("returns records from the other cell for a different target chunk", async () => {
    const { kvStoreRead, kvStoreList } = makeTwoCellStore();
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
    const { kvStoreRead, kvStoreList } = makeTwoCellStore();
    const caches = createCrossChunkLinksCaches();
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [99, 99, 99],
      caches,
      new AbortController().signal,
    );
    expect(table!.records).toEqual([]);
  });

  it("returns records from every cell touching a chunk shared by multiple cells", async () => {
    const { kvStoreRead, kvStoreList } = makeTwoCellStore();
    const caches = createCrossChunkLinksCaches();
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [0, 0, 0],
      caches,
      new AbortController().signal,
    );
    // (0,0,0) is one endpoint of every record in both cells.
    expect(table!.records).toHaveLength(4);
  });

  it("caches cell bytes across queries for different target chunks", async () => {
    const { kvStoreRead, kvStoreList, reads } = makeTwoCellStore();
    const caches = createCrossChunkLinksCaches();
    const signal = new AbortController().signal;

    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      signal,
    );
    const cellReadsAfterFirst = reads.filter((r) => r.includes("/c/0")).length;
    expect(cellReadsAfterFirst).toBeGreaterThan(0);

    // Second query, same target chunk: already-fetched cell bytes must
    // not be re-read.
    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      signal,
    );
    expect(reads.filter((r) => r.includes("/c/0")).length).toBe(
      cellReadsAfterFirst,
    );
  });

  it("lists the level's cell keys at most once across repeat queries", async () => {
    const { kvStoreRead, kvStoreList, lists } = makeTwoCellStore();
    const caches = createCrossChunkLinksCaches();
    const signal = new AbortController().signal;

    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [1, 0, 0],
      caches,
      signal,
    );
    expect(lists).toHaveLength(1);

    await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      [2, 0, 0],
      caches,
      signal,
    );
    // A different target chunk at the same level reuses the cached
    // listing — no additional kvStoreList call.
    expect(lists).toHaveLength(1);
  });

  it("concurrent queries for the same level share one listing call", async () => {
    const { kvStoreRead, kvStoreList, lists } = makeTwoCellStore();
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
        [2, 0, 0],
        caches,
        signal,
      ),
    ]);
    expect(lists).toHaveLength(1);
  });

  it("matches readCrossChunkLinks's full-table result filtered to one chunk", async () => {
    const { kvStoreRead, kvStoreList } = makeTwoCellStore();
    const signal = new AbortController().signal;

    const fullTable = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      signal,
    );
    const expected = fullTable!.records.filter((r) =>
      r.endpoints.some(
        (e) =>
          e.chunkCoords[0] === 1 && e.chunkCoords[1] === 0 && e.chunkCoords[2] === 0,
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
  const cellBytes = hexBytes(ZSTD_CELL_HEX);

  function makeStore(opts: { directed: boolean; cellKey: string }) {
    const reads: string[] = [];
    const listPrefixes: string[] = [];
    const kvStoreRead = async (subpath: string) => {
      reads.push(subpath);
      if (subpath === "cross_chunk_links/0/zarr.json") {
        return jsonBytes({
          attributes: {
            link_width: 2,
            sid_ndim: 3,
            directed: opts.directed,
            store: "canonical",
          },
        });
      }
      if (subpath === `cross_chunk_links/0/${opts.cellKey}/c/0`) return cellBytes;
      return undefined;
    };
    const kvStoreList = async (
      prefix: string,
    ): Promise<CrossChunkLinksListResult> => {
      listPrefixes.push(prefix); // must never be called by go-direct
      return { directories: [], files: [] };
    };
    return { kvStoreRead, kvStoreList, reads, listPrefixes };
  }

  it("reads the owned pair's cell directly (undirected, canonical order), with NO directory listing", async () => {
    const { kvStoreRead, kvStoreList, listPrefixes } = makeStore({
      directed: false,
      cellKey: "0.0.0.1.0.0",
    });
    const table = await readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreList },
      [
        [1, 0, 0],
        [0, 0, 0],
      ],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table).toBeDefined();
    expect(table!.records).toHaveLength(2);
    // Go-direct computes the cell key from the owned pair directly — it
    // must never walk the level's cell-key listing.
    expect(listPrefixes).toEqual([]);
  });

  it("tries every ordering for a directed family (writer keys by literal input order)", async () => {
    // Writer chose (1,0,0) before (0,0,0) — the reader can't predict
    // this, so it must try both orderings of the owned pair.
    const { kvStoreRead, kvStoreList, listPrefixes } = makeStore({
      directed: true,
      cellKey: "1.0.0.0.0.0",
    });
    const table = await readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreList },
      [
        [0, 0, 0],
        [1, 0, 0],
      ],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table!.records).toHaveLength(2);
    expect(listPrefixes).toEqual([]);
  });

  it("skips when there is no owned-owned pair (no read, empty result)", async () => {
    const { kvStoreRead, kvStoreList, reads } = makeStore({
      directed: true,
      cellKey: "0.0.0.1.0.0",
    });
    // Only one owned chunk: with link_width=2 there is no owned-owned
    // pair, so no candidate cell -> no cell read at all.
    const table = await readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreList },
      [[0, 0, 0]],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table!.records).toEqual([]);
    expect(reads.some((r) => r.endsWith("/c/0"))).toBe(false);
  });

  it("caches the header across queries (no repeated zarr.json read)", async () => {
    const { kvStoreRead, kvStoreList, reads } = makeStore({
      directed: true,
      cellKey: "0.0.0.1.0.0",
    });
    const caches = createCrossChunkLinksCaches();
    const owned = [
      [0, 0, 0],
      [1, 0, 0],
    ];
    const signal = new AbortController().signal;
    await readCrossChunkLinksForOwnedChunks({ kvStoreRead, kvStoreList }, owned, caches, signal);
    await readCrossChunkLinksForOwnedChunks({ kvStoreRead, kvStoreList }, owned, caches, signal);
    expect(reads.filter((r) => r === "cross_chunk_links/0/zarr.json")).toHaveLength(1);
  });

  it("throws for a store='duplicate' family (unsupported by go-direct)", async () => {
    const kvStoreRead = async (subpath: string) => {
      if (subpath === "cross_chunk_links/0/zarr.json") {
        return jsonBytes({
          attributes: { link_width: 2, sid_ndim: 3, directed: true, store: "duplicate" },
        });
      }
      return undefined;
    };
    await expect(
      readCrossChunkLinksForOwnedChunks(
        { kvStoreRead },
        [
          [0, 0, 0],
          [1, 0, 0],
        ],
        createCrossChunkLinksCaches(),
        new AbortController().signal,
      ),
    ).rejects.toThrow(/only supports store="canonical"/);
  });
});

// ---------------------------------------------------------------------------
// packed_sharded layout
// ---------------------------------------------------------------------------

/**
 * REAL on-disk fixture generated by zarr-vectors-py's own writer via
 * `write_cross_chunk_links(..., layout="packed_sharded")` for a directed
 * link_width=2 family with sid_ndim=3 and these four records (see the
 * flat FLATCELL_HEX fixtures below — the SAME records, written flat):
 *   0: ((0,0,0),1) -> ((1,0,0),2)
 *   1: ((1,0,0),9) -> ((0,0,0),8)
 *   2: ((2,0,0),8) -> ((0,0,0),3)
 *   3: ((5,0,0),4) -> ((5,1,0),5)
 * The writer sorts the cell keys, so flat array index i holds cell
 * PACKED_CELL_KEYS[i]. This is the exact `cross_chunk_links/0/c/0` shard
 * file (single shard: shard shape [512] > array length 4), zstd+vlen
 * framed per element with a trailing shard index region + crc32c.
 */
const PACKED_CELL_KEYS = [
  "0.0.0.1.0.0",
  "1.0.0.0.0.0",
  "2.0.0.0.0.0",
  "5.0.0.5.1.0",
];

/**
 * The real per-cell flat blobs (zstd of the ragged blob) written by the
 * SAME records in the default flat layout — used to build an equivalent
 * flat store so packed reads can be asserted identical to flat reads.
 */
const FLATCELL_HEX: Record<string, string> = {
  "0.0.0.1.0.0":
    "28b52ffd2028a5000060010001000200000000000000020060c00ac002",
  "1.0.0.0.0.0":
    "28b52ffd2028a5000060010009000800000000000000020060c00ac002",
  "2.0.0.0.0.0":
    "28b52ffd2028a5000060010008000300000000000000020060c00ac002",
  "5.0.0.5.1.0":
    "28b52ffd2028a5000060010004000500000000000000020060c00ac002",
};

const PACKED_SHARD_HEX =
  "28b52ffd2030d500009001000000280000000100020000000000000002003b09a0002428b52ffd2030e50000a00100000028000000010009000800000000000000020060c00a400228b52ffd2030e50000a00100000028000000010008000300000000000000020060c00a400228b52ffd2030e50000a00100000028000000010004000500000000000000020060c00a40020000000000000000230000000000000023000000000000002500000000000000480000000000000025000000000000006d000000000000002500000000000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff626e35de";

/**
 * Build a mock kvstore over the packed fixture. `kvStoreReadRange` slices
 * the single shard file per byteRange (real byte-range reads);
 * `kvStoreRead` serves the array `zarr.json` and (for the whole-shard
 * fallback test) the whole shard blob. `kvStoreList` throws — packed
 * reads must never list.
 */
function makePackedStore(opts?: { delta?: number; withReadRange?: boolean }) {
  const delta = opts?.delta ?? 0;
  const withReadRange = opts?.withReadRange ?? true;
  const base = `cross_chunk_links/${delta}`;
  const shard = hexBytes(PACKED_SHARD_HEX);
  const meta = {
    shape: [PACKED_CELL_KEYS.length],
    data_type: "variable_length_bytes",
    chunk_grid: { name: "regular", configuration: { chunk_shape: [512] } },
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: [1],
          codecs: [
            { name: "vlen-bytes", configuration: {} },
            { name: "zstd", configuration: { level: 0, checksum: false } },
          ],
          index_codecs: [
            { name: "bytes", configuration: { endian: "little" } },
            { name: "crc32c" },
          ],
          index_location: "end",
        },
      },
    ],
    attributes: {
      sid_ndim: 3,
      link_width: 2,
      directed: true,
      store: "canonical",
      layout: "packed_sharded",
      cell_keys: PACKED_CELL_KEYS,
    },
    zarr_format: 3,
    node_type: "array",
  };
  const shardPath = `${base}/c/0`;
  const reads: string[] = [];
  const kvStoreRead = async (
    subpath: string,
  ): Promise<Uint8Array | undefined> => {
    reads.push(subpath);
    if (subpath === `${base}/zarr.json`) return jsonBytes(meta);
    if (subpath === shardPath) return shard;
    return undefined;
  };
  const rangeReads: string[] = [];
  const kvStoreReadRange = async (
    subpath: string,
    byteRange: { offset: number; length: number } | { suffixLength: number },
    _signal: AbortSignal,
  ): Promise<Uint8Array | undefined> => {
    rangeReads.push(subpath);
    if (subpath !== shardPath) return undefined;
    if ("suffixLength" in byteRange) {
      return shard.subarray(shard.byteLength - byteRange.suffixLength);
    }
    return shard.subarray(byteRange.offset, byteRange.offset + byteRange.length);
  };
  const kvStoreList = async (): Promise<CrossChunkLinksListResult> => {
    throw new Error("packed layout must not list the store");
  };
  return {
    kvStoreRead,
    kvStoreList,
    ...(withReadRange ? { kvStoreReadRange } : {}),
    reads,
    rangeReads,
  };
}

/** Build the equivalent FLAT store for the same records (sorted cell order matches packed). */
function makeFlatEquivStore(delta = 0) {
  const base = `cross_chunk_links/${delta}`;
  const kvStoreRead = async (
    subpath: string,
  ): Promise<Uint8Array | undefined> => {
    if (subpath === `${base}/zarr.json`) {
      return jsonBytes({
        attributes: {
          sid_ndim: 3,
          link_width: 2,
          directed: true,
          store: "canonical",
        },
      });
    }
    const m = subpath.match(
      new RegExp(`^${base.replace(/[-]/g, "\\-")}/([^/]+)/c/0$`),
    );
    if (m !== null) {
      const hex = FLATCELL_HEX[m[1]];
      return hex === undefined ? undefined : hexBytes(hex);
    }
    return undefined;
  };
  const kvStoreList = async (
    prefix: string,
  ): Promise<CrossChunkLinksListResult> => {
    if (prefix === `${base}/`) {
      return { directories: [...PACKED_CELL_KEYS], files: [] };
    }
    return { directories: [], files: [] };
  };
  return { kvStoreRead, kvStoreList };
}

const EXPECTED_RECORDS = [
  { endpoints: [{ chunkCoords: [0, 0, 0], vertexIndex: 1 }, { chunkCoords: [1, 0, 0], vertexIndex: 2 }] },
  { endpoints: [{ chunkCoords: [1, 0, 0], vertexIndex: 9 }, { chunkCoords: [0, 0, 0], vertexIndex: 8 }] },
  { endpoints: [{ chunkCoords: [2, 0, 0], vertexIndex: 8 }, { chunkCoords: [0, 0, 0], vertexIndex: 3 }] },
  { endpoints: [{ chunkCoords: [5, 0, 0], vertexIndex: 4 }, { chunkCoords: [5, 1, 0], vertexIndex: 5 }] },
];

describe("readCrossChunkLinks (packed_sharded layout)", () => {
  it("decodes the whole packed table (real shard bytes, byte-range reads)", async () => {
    const { kvStoreRead, kvStoreReadRange, kvStoreList } = makePackedStore();
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreReadRange, kvStoreList },
      new AbortController().signal,
    );
    expect(table).toBeDefined();
    expect(table!.linkWidth).toBe(2);
    expect(table!.sidNdim).toBe(3);
    expect(table!.records).toEqual(EXPECTED_RECORDS);
  });

  it("returns records IDENTICAL to the flat layout for the same data", async () => {
    const packed = makePackedStore();
    const flat = makeFlatEquivStore();
    const signal = new AbortController().signal;
    const packedTable = await readCrossChunkLinks(
      {
        kvStoreRead: packed.kvStoreRead,
        kvStoreReadRange: packed.kvStoreReadRange,
        kvStoreList: packed.kvStoreList,
      },
      signal,
    );
    const flatTable = await readCrossChunkLinks(
      { kvStoreRead: flat.kvStoreRead, kvStoreList: flat.kvStoreList },
      signal,
    );
    expect(packedTable).toEqual(flatTable);
  });

  it("reads correctly via the whole-shard fallback when no kvStoreReadRange is given", async () => {
    const { kvStoreRead, kvStoreList } = makePackedStore({
      withReadRange: false,
    });
    const table = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList },
      new AbortController().signal,
    );
    expect(table!.records).toEqual(EXPECTED_RECORDS);
  });

  it("dispatches per-family: mixed store (delta 0 packed, delta -1 flat)", async () => {
    const packed = makePackedStore({ delta: 0 });
    const flat = makeFlatEquivStore(-1);
    const signal = new AbortController().signal;
    // One combined kvStoreRead / list serving both families.
    const kvStoreRead = async (subpath: string) =>
      (await packed.kvStoreRead(subpath)) ?? (await flat.kvStoreRead(subpath));
    const kvStoreList = async (prefix: string) => flat.kvStoreList(prefix);
    const packedTable = await readCrossChunkLinks(
      {
        kvStoreRead,
        kvStoreReadRange: packed.kvStoreReadRange,
        kvStoreList,
        delta: 0,
      },
      signal,
    );
    const flatTable = await readCrossChunkLinks(
      { kvStoreRead, kvStoreList, delta: -1 },
      signal,
    );
    expect(packedTable!.records).toEqual(EXPECTED_RECORDS);
    expect(flatTable!.records).toEqual(EXPECTED_RECORDS);
  });
});

describe("readCrossChunkLinksForOwnedChunks (packed_sharded layout)", () => {
  async function ownedPacked(
    ownedChunks: number[][],
    caches = createCrossChunkLinksCaches(),
  ) {
    const { kvStoreRead, kvStoreReadRange, kvStoreList } = makePackedStore();
    return readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreReadRange, kvStoreList },
      ownedChunks,
      caches,
      new AbortController().signal,
    );
  }
  async function ownedFlat(ownedChunks: number[][]) {
    const { kvStoreRead, kvStoreList } = makeFlatEquivStore();
    return readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreList },
      ownedChunks,
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
  }

  it("matches flat for an owned pair present as two directed orderings", async () => {
    const owned = [
      [1, 0, 0],
      [0, 0, 0],
    ];
    const packed = await ownedPacked(owned);
    const flat = await ownedFlat(owned);
    expect(packed).toEqual(flat);
    // (0,0,0)<->(1,0,0) has cells 0.0.0.1.0.0 and 1.0.0.0.0.0 -> 2 records.
    expect(packed!.records).toHaveLength(2);
  });

  it("matches flat with a mix of present and absent candidate cells", async () => {
    const owned = [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
    ];
    const packed = await ownedPacked(owned);
    const flat = await ownedFlat(owned);
    expect(packed).toEqual(flat);
    // Present owned-owned cells: 0.0.0.1.0.0, 1.0.0.0.0.0, 2.0.0.0.0.0.
    expect(packed!.records).toHaveLength(3);
  });

  it("matches flat (empty) when no candidate cell is present", async () => {
    const owned = [
      [9, 0, 0],
      [0, 0, 0],
    ];
    const packed = await ownedPacked(owned);
    const flat = await ownedFlat(owned);
    expect(packed).toEqual(flat);
    expect(packed!.records).toEqual([]);
  });

  it("never lists the store (go-direct) and reuses the cell-key index map", async () => {
    const { kvStoreRead, kvStoreReadRange, kvStoreList } = makePackedStore();
    const caches = createCrossChunkLinksCaches();
    const owned = [
      [0, 0, 0],
      [1, 0, 0],
    ];
    const signal = new AbortController().signal;
    await readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreReadRange, kvStoreList },
      owned,
      caches,
      signal,
    );
    await readCrossChunkLinksForOwnedChunks(
      { kvStoreRead, kvStoreReadRange, kvStoreList },
      owned,
      caches,
      signal,
    );
    // The cell-key index map is built once and cached on the caches struct.
    expect(caches.cellKeyIndexMaps.size).toBe(1);
    expect(caches.cellKeyIndexMaps.get("cross_chunk_links/0")!.size).toBe(4);
  });
});
