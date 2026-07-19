/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
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

import { afterEach, describe, expect, it, vi } from "vitest";
import { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import type { RoiFilterableChunk } from "#src/datasource/zarr-vectors/roi_filter_backend.js";
import {
  RoiFilterServiceClient,
  encodeChunkGeometry,
} from "#src/datasource/zarr-vectors/roi_filter_service.js";

/**
 * The dissection payload crosses a language boundary, so these tests pin the
 * exact bytes. `GOLDEN_CHUNK_HEX` is asserted here and decoded by
 * `python/tests/tractography_service_test.py`; if either side's idea of the
 * layout drifts, one of the two fails.
 */

/** A fragment index of contiguous ranges, the streamline norm. */
function rangeFragmentIndex(ranges: { start: number; count: number }[]) {
  const numFragments = ranges.length;
  const bitmapBytes = (numFragments + 7) >> 3;
  const bitmap = new Uint8Array(bitmapBytes);
  for (let f = 0; f < numFragments; ++f) bitmap[f >> 3] |= 1 << (f & 7);
  const rangeTable = new BigInt64Array(numFragments * 2);
  for (let f = 0; f < numFragments; ++f) {
    rangeTable[f * 2] = BigInt(ranges[f].start);
    rangeTable[f * 2 + 1] = BigInt(ranges[f].count);
  }
  return new FragmentIndex(
    numFragments,
    bitmap,
    rangeTable,
    new Uint32Array([0]),
    new BigInt64Array(0),
  );
}

/** Per-vertex `uvec2` segment column from a per-vertex id list. */
function segmentColumn(ids: bigint[]): Uint32Array {
  const out = new Uint32Array(ids.length * 2);
  for (let i = 0; i < ids.length; ++i) {
    out[i * 2] = Number(ids[i] & 0xffffffffn);
    out[i * 2 + 1] = Number(ids[i] >> 32n);
  }
  return out;
}

/**
 * Two tracts and a trailing ghost vertex.
 *
 * Vertices 0-2 are object 7, vertices 3-4 are object 9, vertex 5 is a ghost —
 * it exists in `positions` but belongs to no fragment, exactly as
 * `appendGhostVertices` leaves it.
 */
function sampleChunk(): RoiFilterableChunk {
  const positions = Float32Array.from([
    0,
    0,
    0,
    1,
    0,
    0,
    2,
    0,
    0, // object 7
    10,
    0,
    0,
    11,
    0,
    0, // object 9
    99,
    99,
    99, // ghost
  ]);
  return {
    rank: 3,
    numVertices: 6,
    positions,
    segmentIds: segmentColumn([7n, 7n, 7n, 9n, 9n, 0n]),
    fragmentIndex: rangeFragmentIndex([
      { start: 0, count: 3 },
      { start: 3, count: 2 },
    ]),
  };
}

const GOLDEN_CHUNK_HEX =
  "030000000200000005000000" + // rank 3, 2 fragments, 5 vertices
  "00000000" + // reserved
  "07000000000000000900000000000000" + // rowIds: 7, 9
  "000000000300000005000000" + // rowOffsets: 0, 3, 5
  "000000000000000000000000" + // (0, 0, 0)
  "0000803f0000000000000000" + // (1, 0, 0)
  "000000400000000000000000" + // (2, 0, 0)
  "000020410000000000000000" + // (10, 0, 0)
  "000030410000000000000000"; // (11, 0, 0) — 104 bytes, already 8-aligned

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

describe("encodeChunkGeometry", () => {
  it("matches the golden bytes the Python decoder is tested against", () => {
    expect(toHex(encodeChunkGeometry(sampleChunk()))).toBe(GOLDEN_CHUNK_HEX);
  });

  it("gathers each fragment's vertices contiguously and drops ghosts", () => {
    const bytes = encodeChunkGeometry(sampleChunk());
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(0, true)).toBe(3); // rank
    expect(view.getUint32(4, true)).toBe(2); // fragments
    // 5 vertices, not 6: the ghost belongs to no fragment and is not sent.
    expect(view.getUint32(8, true)).toBe(5);

    const rowIds = new BigUint64Array(bytes.buffer, bytes.byteOffset + 16, 2);
    expect([...rowIds]).toEqual([7n, 9n]);

    const rowOffsets = new Uint32Array(bytes.buffer, bytes.byteOffset + 32, 3);
    expect([...rowOffsets]).toEqual([0, 3, 5]);

    const positions = new Float32Array(bytes.buffer, bytes.byteOffset + 44, 15);
    expect([...positions]).toEqual([
      0, 0, 0, 1, 0, 0, 2, 0, 0, 10, 0, 0, 11, 0, 0,
    ]);
  });

  it("skips empty fragments, as computeChunkCrossings does", () => {
    const chunk = sampleChunk();
    const withEmpty: RoiFilterableChunk = {
      ...chunk,
      fragmentIndex: rangeFragmentIndex([
        { start: 0, count: 3 },
        { start: 3, count: 0 }, // empty
        { start: 3, count: 2 },
      ]),
    };
    const bytes = encodeChunkGeometry(withEmpty);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(4, true)).toBe(2);
    expect(view.getUint32(8, true)).toBe(5);
  });

  it("emits an empty chunk when there is no segment column", () => {
    const chunk = sampleChunk();
    const bytes = encodeChunkGeometry({ ...chunk, segmentIds: undefined });
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(4, true)).toBe(0);
    expect(view.getUint32(8, true)).toBe(0);
  });

  it("pads every blob to 8 bytes so the next one's uint64s stay aligned", () => {
    for (const counts of [[1], [1, 1], [3, 2], [2, 2, 2], [5]]) {
      const ranges: { start: number; count: number }[] = [];
      let start = 0;
      for (const c of counts) {
        ranges.push({ start, count: c });
        start += c;
      }
      const total = start;
      const chunk: RoiFilterableChunk = {
        rank: 3,
        numVertices: total,
        positions: new Float32Array(total * 3),
        segmentIds: segmentColumn(
          Array.from({ length: total }, (_, i) => BigInt(i)),
        ),
        fragmentIndex: rangeFragmentIndex(ranges),
      };
      expect(encodeChunkGeometry(chunk).byteLength % 8).toBe(0);
    }
  });

  it("does not detach the chunk's buffers", () => {
    // Transferring rather than copying would break serializeSkeletonChunkData
    // and every later refilter.
    const chunk = sampleChunk();
    encodeChunkGeometry(chunk);
    expect(chunk.positions.byteLength).toBeGreaterThan(0);
    expect(chunk.positions[0]).toBe(0);
    expect(chunk.segmentIds!.byteLength).toBeGreaterThan(0);
  });
});

/** An empty but well-formed response body. */
function emptyResponseBody(): ArrayBuffer {
  const buffer = new ArrayBuffer(16);
  const header = new Uint32Array(buffer);
  header[0] = 0x52464c54; // 'RFLT'
  return buffer;
}

function chunkEntry(key: string) {
  return { key, data: sampleChunk() };
}

describe("RoiFilterServiceClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("falls back rather than rejecting when the service errors", async () => {
    // The service worker answers 503 once the browser has recycled it, and
    // while Python is still starting. Rejecting here would leave the caller
    // retrying forever with the working in-worker path sitting unused.
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    let calls = 0;
    vi.stubGlobal("fetch", async () => {
      ++calls;
      return new Response(null, { status: 503 });
    });
    const client = new RoiFilterServiceClient("s");
    await expect(client.compute([], [])).resolves.toBeUndefined();
    expect(calls).toBe(1);
  });

  it("stops asking after a run of failures", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    let calls = 0;
    vi.stubGlobal("fetch", async () => {
      ++calls;
      return new Response(null, { status: 500 });
    });
    const client = new RoiFilterServiceClient("s");
    for (let i = 0; i < 6; ++i) {
      expect(await client.compute([], [])).toBeUndefined();
    }
    expect(client.isUnavailable).toBe(true);
    // Latched after the third; the remaining calls never hit the network.
    expect(calls).toBe(3);
  });

  it("recovers from an isolated failure", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    let n = 0;
    vi.stubGlobal("fetch", async () => {
      if (++n === 1) return new Response(null, { status: 503 });
      return new Response(emptyResponseBody(), { status: 200 });
    });
    const client = new RoiFilterServiceClient("s");
    expect(await client.compute([], [])).toBeUndefined();
    expect(await client.compute([], [])).toBeDefined();
    expect(client.isUnavailable).toBe(false);
  });

  it("latches immediately when there is no service at all", async () => {
    vi.stubGlobal("fetch", async () => {
      throw new TypeError("Failed to fetch");
    });
    const client = new RoiFilterServiceClient("s");
    expect(await client.compute([], [])).toBeUndefined();
    expect(client.isUnavailable).toBe(true);
  });

  it("uploads a chunk's geometry once, then only the regions", async () => {
    const bodies: number[] = [];
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      bodies.push((init.body as Uint8Array).byteLength);
      return new Response(emptyResponseBody(), { status: 200 });
    });
    const client = new RoiFilterServiceClient("s");
    const chunks = [chunkEntry("c0"), chunkEntry("c1")];
    await client.compute(chunks, []);
    await client.compute(chunks, []);
    expect(bodies[0]).toBeGreaterThan(0);
    expect(bodies[1]).toBe(0); // the drag case: regions only
  });

  it("re-uploads everything after the service reports geometry missing", async () => {
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      const params = JSON.parse(decodeURIComponent(url.split("?p=")[1]));
      if (params.uploads.length === 0) {
        return new Response(JSON.stringify({ missing: ["c0"] }), {
          status: 409,
        });
      }
      void init;
      return new Response(emptyResponseBody(), { status: 200 });
    });
    const client = new RoiFilterServiceClient("s");
    const chunks = [chunkEntry("c0")];
    await client.compute(chunks, []);
    // Second call would normally send no geometry; the 409 forces a re-upload
    // and the retry must succeed rather than surfacing the error.
    expect(await client.compute(chunks, [])).toBeDefined();
  });

  it("retracts aged-out chunks so the service can release them", async () => {
    const drops: string[][] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      const params = JSON.parse(decodeURIComponent(url.split("?p=")[1]));
      drops.push(params.drop);
      return new Response(emptyResponseBody(), { status: 200 });
    });
    const client = new RoiFilterServiceClient("s");
    // A sliding window, as panning produces. Well past the cap.
    for (let step = 0; step < 600; ++step) {
      await client.compute([chunkEntry(`c${step}`)], []);
    }
    const totalDropped = drops.reduce((n, d) => n + d.length, 0);
    expect(totalDropped).toBeGreaterThan(0);
    // Held set stays bounded rather than growing with distance panned.
    const held = 600 - totalDropped;
    expect(held).toBeLessThanOrEqual(512);
  });

  it("drops nothing while the chunk set is unchanged", async () => {
    // Dragging a region must not evict, or the service rebuilds its combined
    // index every frame.
    const drops: string[][] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      const params = JSON.parse(decodeURIComponent(url.split("?p=")[1]));
      drops.push(params.drop);
      return new Response(emptyResponseBody(), { status: 200 });
    });
    const client = new RoiFilterServiceClient("s");
    const chunks = [chunkEntry("c0"), chunkEntry("c1")];
    for (let i = 0; i < 30; ++i) await client.compute(chunks, []);
    expect(drops.every((d) => d.length === 0)).toBe(true);
  });
});
