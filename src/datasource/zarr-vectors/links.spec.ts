/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { describe, expect, it } from "vitest";
import {
  createCrossChunkLinksCaches,
  decodeLinkCell,
  decodeRaggedBlobRows,
  readCrossChunkLinksForChunk,
} from "#src/datasource/zarr-vectors/links.js";

// --- fixture builders reproducing the writer's on-disk framing ---------------

/** `encode_ragged_blob`: [int64 k][int64 groupByteOffset × k][int64 data...]. */
function raggedBlob(groups: number[][][]): Uint8Array {
  const flat: number[] = [];
  const byteOffsets: number[] = [];
  for (const group of groups) {
    byteOffsets.push(flat.length * 8);
    for (const row of group) for (const v of row) flat.push(v);
  }
  const k = groups.length;
  const buf = new Uint8Array(8 * (1 + k) + flat.length * 8);
  const dv = new DataView(buf.buffer);
  dv.setBigInt64(0, BigInt(k), true);
  for (let i = 0; i < k; ++i)
    dv.setBigInt64(8 + 8 * i, BigInt(byteOffsets[i]), true);
  const base = 8 * (1 + k);
  for (let i = 0; i < flat.length; ++i)
    dv.setBigInt64(base + 8 * i, BigInt(flat[i]), true);
  return buf;
}

/** numcodecs vlen-bytes container holding exactly one item. */
function vlenBytesSingle(item: Uint8Array): Uint8Array {
  const buf = new Uint8Array(8 + item.byteLength);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, 1, true);
  dv.setUint32(4, item.byteLength, true);
  buf.set(item, 8);
  return buf;
}

/** A cell as stored: vlen-bytes(ragged_blob(rows as one group)). */
function cell(rows: number[][]): Uint8Array {
  return vlenBytesSingle(raggedBlob([rows]));
}

const enc = (obj: unknown) => new TextEncoder().encode(JSON.stringify(obj));

describe("decodeRaggedBlobRows", () => {
  it("decodes a single group of width-2 rows", () => {
    // The real HCP1065 cell 0/links/0/0.0.+1/c/0/0/0: one edge [29, 63].
    const rows = decodeRaggedBlobRows(raggedBlob([[[29, 63]]]), 2);
    expect(rows).toEqual([[29, 63]]);
  });

  it("flattens multiple groups", () => {
    const rows = decodeRaggedBlobRows(
      raggedBlob([
        [
          [1, 2],
          [3, 4],
        ],
        [[5, 6]],
      ]),
      2,
    );
    expect(rows).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });

  it("returns [] for an empty blob (k = 0)", () => {
    expect(decodeRaggedBlobRows(raggedBlob([]), 2)).toEqual([]);
    expect(decodeRaggedBlobRows(new Uint8Array(0), 2)).toEqual([]);
  });

  it("throws when a group is not a whole number of rows", () => {
    // 3 int64 with ncols=2 -> not divisible.
    expect(() => decodeRaggedBlobRows(raggedBlob([[[1, 2, 3]]]), 2)).toThrow();
  });
});

describe("decodeLinkCell", () => {
  it("places endpoint chunks at source + offset (the real +z edge)", () => {
    const records = decodeLinkCell(
      cell([[29, 63]]),
      [0, 0, 0], // source chunk
      [[0, 0, 1]], // offset from the path (0.0.+1)
      /*linkWidth=*/ 2,
      /*hasPerm=*/ false,
    );
    expect(records).toEqual([
      {
        endpoints: [
          { chunkCoords: [0, 0, 0], vertexIndex: 29 }, // endpoint 0 = source
          { chunkCoords: [0, 0, 1], vertexIndex: 63 }, // endpoint 1 = source + offset
        ],
      },
    ]);
  });

  it("uses a negative offset correctly", () => {
    const [record] = decodeLinkCell(
      cell([[7, 8]]),
      [2, 2, 2],
      [[-1, 0, 0]],
      2,
      false,
    );
    expect(record.endpoints[1].chunkCoords).toEqual([1, 2, 2]);
  });

  it("skips the perm_idx column when has_perm is set", () => {
    // Row is [perm_idx, vi_0, vi_1]; only the vi columns become endpoints.
    const [record] = decodeLinkCell(
      cell([[5, 11, 22]]),
      [0, 0, 0],
      [[0, 1, 0]],
      2,
      true,
    );
    expect(record.endpoints.map((e) => e.vertexIndex)).toEqual([11, 22]);
    expect(record.endpoints[1].chunkCoords).toEqual([0, 1, 0]);
  });

  it("rejects a zstd-compressed cell with a clear error", () => {
    const zstd = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, 0, 0, 0, 0]);
    expect(() =>
      decodeLinkCell(zstd, [0, 0, 0], [[0, 0, 1]], 2, false),
    ).toThrow(/zstd/);
  });
});

// --- reader against an in-memory store, GET-only (no lister) ------------------

function makeStore(entries: Record<string, Uint8Array>) {
  return async (subpath: string): Promise<Uint8Array | undefined> =>
    entries[subpath];
}

const FAMILY = enc({
  attributes: {
    zv_array: "links_family",
    link_width: 2,
    sid_ndim: 3,
    directed: true,
    store: "canonical",
    level_delta: 0,
  },
});

const PLUS_Z_ARRAY = enc({
  attributes: {
    zv_array: "links",
    offsets: [[0, 0, 1]],
    has_perm: false,
    link_width: 2,
    nonempty_chunks: ["0.0.0"], // only chunk (0,0,0) has +z edges
  },
});

describe("readCrossChunkLinksForChunk (GET-only fallback)", () => {
  const store = makeStore({
    "links/0/zarr.json": FAMILY,
    "links/0/0.0.+1/zarr.json": PLUS_Z_ARRAY,
    "links/0/0.0.+1/c/0/0/0": cell([[29, 63]]),
  });
  const options = { kvStoreRead: store }; // no kvStoreList -> bounded probe

  it("finds a source-side edge and locates its neighbour endpoint", async () => {
    const table = await readCrossChunkLinksForChunk(
      options,
      [0, 0, 0],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table).toBeDefined();
    expect(table!.linkWidth).toBe(2);
    expect(table!.records).toEqual([
      {
        endpoints: [
          { chunkCoords: [0, 0, 0], vertexIndex: 29 },
          { chunkCoords: [0, 0, 1], vertexIndex: 63 },
        ],
      },
    ]);
  });

  it("returns the SAME bridge for the neighbour chunk via the mirror (target) cell", async () => {
    // Chunk (0,0,1) is the successor. Its own cell is empty, but the record is
    // filed under (0,0,1) - (0,0,1) = (0,0,0), so the target-side read finds it.
    const table = await readCrossChunkLinksForChunk(
      options,
      [0, 0, 1],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table!.records).toHaveLength(1);
    expect(table!.records[0].endpoints[1].chunkCoords).toEqual([0, 0, 1]);
  });

  it("skips the GET when nonempty_chunks excludes the cell", async () => {
    // A chunk with no incident edges either way returns an empty table (not
    // undefined): the family exists, this chunk just has no links.
    const table = await readCrossChunkLinksForChunk(
      options,
      [5, 5, 5],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table).toBeDefined();
    expect(table!.records).toEqual([]);
  });

  it("returns undefined when the links family is absent", async () => {
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead: makeStore({}) },
      [0, 0, 0],
      createCrossChunkLinksCaches(),
      new AbortController().signal,
    );
    expect(table).toBeUndefined();
  });

  it("throws when the family group is present but malformed (unreadable, not absent)", async () => {
    const bad = makeStore({
      "links/0/zarr.json": enc({ attributes: { zv_array: "not_links" } }),
    });
    await expect(
      readCrossChunkLinksForChunk(
        { kvStoreRead: bad },
        [0, 0, 0],
        createCrossChunkLinksCaches(),
        new AbortController().signal,
      ),
    ).rejects.toThrow();
  });
});
