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

/**
 * Client for the Python ROI dissection service.
 *
 * The streamline filter's geometry -- does this tract cross this region --
 * evaluates in Python/WASM rather than here. Everything around it is unchanged:
 * the render-layer backend still decides which pyramid level to evaluate, which
 * chunks are resident, and when a recompute is warranted; it just delegates the
 * arithmetic.
 *
 * The transport is a plain `fetch` from this (the chunk backend) worker. In the
 * Pyodide build a Service Worker intercepts `/neuroglancer/...` and routes it
 * straight to the Python worker, so the request never touches the main thread
 * and never touches viewer state. Where there is no such service -- an ordinary
 * Neuroglancer build -- {@link RoiFilterServiceClient.compute} reports itself
 * unavailable and the caller falls back to the TypeScript implementation in
 * `roi_filter_backend.ts`, so the feature degrades to its previous behaviour
 * rather than disappearing.
 *
 * **Geometry is uploaded once per chunk.** A decoded chunk is immutable for the
 * life of its key (the pyramid level is part of that key), so the client tracks
 * what it has sent and thereafter transmits only the region definitions. This is
 * what makes dragging a region cheap: a few hundred bytes per frame instead of
 * a couple of megabytes.
 *
 * The byte layout is specified in `python/neuroglancer/tractography/wire.py`.
 * If either side changes, both change together.
 */

import type { Roi, RoiGroupConfig, RoiShape } from "#src/datasource/zarr-vectors/roi.js";
import type { RoiFilterableChunk } from "#src/datasource/zarr-vectors/roi_filter_backend.js";

/** Endpoint prefix; matched by the Pyodide service worker's `shouldIntercept`. */
const ROI_FILTER_PATH = "/neuroglancer/roi_filter";

const RESPONSE_MAGIC = 0x52464c54; // 'RFLT'
const CHUNK_HEADER_BYTES = 16;
const RESPONSE_HEADER_BYTES = 16;

/**
 * Consecutive service failures tolerated before falling back permanently.
 * Above one so a single hiccup does not cost the session's acceleration.
 */
const MAX_CONSECUTIVE_FAILURES = 3;

/**
 * How many chunks' geometry the service is asked to hold, per layer.
 *
 * The viewer evicts chunks it no longer needs, but nothing tells the service
 * that, so without a cap a long pan would pin every chunk ever resident in the
 * Python heap. A few hundred comfortably covers a fully-resident coarse level
 * -- the intended operating point -- so eviction only begins once the view has
 * genuinely moved on.
 */
const MAX_CACHED_CHUNKS = 512;

/** One chunk of the batch being evaluated, with the key its geometry is cached under. */
export interface RoiFilterChunkEntry {
  readonly key: string;
  readonly data: RoiFilterableChunk;
}

/** What the service returns; the same shape `computeGroupedPassingSet` produces. */
export interface RoiFilterResult {
  passing: Set<bigint>;
  colorById: Map<bigint, number>;
  highDetail: Set<bigint>;
}

function shapeToJson(shape: RoiShape): any {
  switch (shape.kind) {
    case "ellipsoid":
      return {
        kind: "ellipsoid",
        center: Array.from(shape.center),
        radii: Array.from(shape.radii),
      };
    case "box":
      return {
        kind: "box",
        lower: Array.from(shape.lower),
        upper: Array.from(shape.upper),
      };
    case "halfspace":
      return {
        kind: "halfspace",
        origin: Array.from(shape.origin),
        normal: Array.from(shape.normal),
      };
  }
}

function roiToJson(roi: Roi): any {
  return {
    shape: shapeToJson(roi.shape),
    predicate: roi.predicate,
    operator: roi.operator,
  };
}

function groupToJson(group: RoiGroupConfig): any {
  return {
    rois: group.rois.map(roiToJson),
    colorPacked: group.colorPacked,
    visible: group.visible,
    highDetail: group.highDetail,
  };
}

/** Read the uint64 segment id of vertex `v` from the `uvec2` segment column. */
function segmentIdOf(segmentIds: Uint32Array, v: number): bigint {
  const lo = BigInt(segmentIds[v * 2] >>> 0);
  const hi = BigInt(segmentIds[v * 2 + 1] >>> 0);
  return (hi << 32n) | lo;
}

/**
 * Serialise one chunk's geometry, gathered per fragment.
 *
 * The vertices of each fragment are emitted contiguously, in fragment order,
 * rather than the chunk's raw vertex array. Two reasons, both load-bearing:
 *
 * - The buffers must be **copied**, never transferred. `serializeSkeletonChunkData`
 *   copies positions into the message it sends the frontend and only transfers
 *   `indices.buffer`, which is exactly why a retained `roiFilterableChunk` stays
 *   valid afterwards. Detaching these here would break both that and every later
 *   refilter. Since a copy is unavoidable, gathering during it is free.
 * - Gathering drops the chunk's ghost vertices. Those are appended by
 *   `appendGhostVertices` for rendering continuity across chunk borders and
 *   belong to no fragment, so testing them would invent crossings the previous
 *   implementation never reported.
 *
 * Empty fragments are skipped, matching `computeChunkCrossings`.
 */
export function encodeChunkGeometry(chunk: RoiFilterableChunk): Uint8Array {
  const { positions, segmentIds, fragmentIndex, rank } = chunk;
  if (segmentIds === undefined) {
    // No segment column: geometry cannot be attributed to an object. Emit an
    // empty chunk rather than failing, matching `computeChunkCrossings`.
    return encodeEmptyChunk(rank);
  }
  const numFragments = fragmentIndex.numFragments;

  // Pass 1: which fragments contribute, and how many vertices in total.
  const kept: Uint32Array[] = [];
  const keptIds: bigint[] = [];
  let numVertices = 0;
  for (let f = 0; f < numFragments; ++f) {
    const indices = fragmentIndex.indices(f);
    if (indices.length === 0) continue;
    kept.push(indices);
    // All vertices of a fragment belong to one object; read the id once.
    keptIds.push(segmentIdOf(segmentIds, indices[0]));
    numVertices += indices.length;
  }

  const f32PerVertex = rank;
  const rowIdBytes = kept.length * 8;
  const rowOffsetBytes = (kept.length + 1) * 4;
  const positionBytes = numVertices * f32PerVertex * 4;
  const unpadded =
    CHUNK_HEADER_BYTES + rowIdBytes + rowOffsetBytes + positionBytes;
  const buffer = new ArrayBuffer((unpadded + 7) & ~7);

  const header = new Uint32Array(buffer, 0, 4);
  header[0] = rank;
  header[1] = kept.length;
  header[2] = numVertices;
  header[3] = 0;

  let offset = CHUNK_HEADER_BYTES;
  const rowIds = new BigUint64Array(buffer, offset, kept.length);
  offset += rowIdBytes;
  const rowOffsets = new Uint32Array(buffer, offset, kept.length + 1);
  offset += rowOffsetBytes;
  const out = new Float32Array(buffer, offset, numVertices * f32PerVertex);

  // Pass 2: gather.
  let vertex = 0;
  for (let i = 0; i < kept.length; ++i) {
    const indices = kept[i];
    rowIds[i] = keptIds[i];
    rowOffsets[i] = vertex;
    for (let j = 0; j < indices.length; ++j) {
      const src = indices[j] * rank;
      const dst = (vertex + j) * rank;
      for (let d = 0; d < rank; ++d) out[dst + d] = positions[src + d];
    }
    vertex += indices.length;
  }
  rowOffsets[kept.length] = vertex;

  return new Uint8Array(buffer);
}

function encodeEmptyChunk(rank: number): Uint8Array {
  const buffer = new ArrayBuffer(CHUNK_HEADER_BYTES + 4 + 4); // header + 1 offset, padded
  const header = new Uint32Array(buffer, 0, 4);
  header[0] = rank;
  header[1] = 0;
  header[2] = 0;
  header[3] = 0;
  return new Uint8Array(buffer);
}

function decodeResponse(buffer: ArrayBuffer): RoiFilterResult {
  const view = new DataView(buffer);
  if (view.byteLength < RESPONSE_HEADER_BYTES) {
    throw new Error(`roi_filter response too short: ${view.byteLength}`);
  }
  const magic = view.getUint32(0, true);
  if (magic !== RESPONSE_MAGIC) {
    throw new Error(
      `roi_filter response magic 0x${magic.toString(16)}, expected 0x${RESPONSE_MAGIC.toString(16)}`,
    );
  }
  const numPassing = view.getUint32(4, true);
  const numHighDetail = view.getUint32(8, true);
  const numColors = view.getUint32(12, true);

  let offset = RESPONSE_HEADER_BYTES;
  const passingRaw = new BigUint64Array(buffer, offset, numPassing);
  offset += numPassing * 8;
  const highDetailRaw = new BigUint64Array(buffer, offset, numHighDetail);
  offset += numHighDetail * 8;
  const colorIds = new BigUint64Array(buffer, offset, numColors);
  offset += numColors * 8;
  const colorValues = new Uint32Array(buffer, offset, numColors);
  offset += numColors * 4;
  if (offset > view.byteLength) {
    throw new Error("roi_filter response truncated");
  }

  const passing = new Set<bigint>();
  for (const id of passingRaw) passing.add(id);
  const highDetail = new Set<bigint>();
  for (const id of highDetailRaw) highDetail.add(id);
  const colorById = new Map<bigint, number>();
  for (let i = 0; i < numColors; ++i) {
    colorById.set(colorIds[i], colorValues[i]);
  }
  return { passing, colorById, highDetail };
}

/**
 * Talks to the Python service, remembering what geometry it has already sent.
 *
 * One instance per render-layer backend; `scope` separates their caches on the
 * Python side.
 */
export class RoiFilterServiceClient {
  /**
   * Chunk keys whose geometry the service is believed to hold, in
   * least-recently-used order (a `Set` preserves insertion order).
   */
  private uploaded = new Set<string>();
  /**
   * Set once the service has failed persistently. From then on the caller uses
   * the TypeScript path for the rest of the session.
   */
  private unavailable = false;
  /** Consecutive failed requests; reset by any success. */
  private consecutiveFailures = 0;

  constructor(public scope: string) {}

  get isUnavailable(): boolean {
    return this.unavailable;
  }

  /**
   * Evaluate `groups` against `chunks`.
   *
   * **Never rejects.** Resolves to `undefined` whenever the service could not
   * answer, for any reason, which is the caller's signal to compute the same
   * thing locally. The service is an accelerator over a path that already
   * works, not a dependency: a filter that stops updating because a worker was
   * recycled would be a worse outcome than one that quietly costs more.
   *
   * The Pyodide deployment makes that a live concern rather than a theoretical
   * one -- the service worker answers 503 once the browser has recycled it and
   * its port to the Python worker is gone, and again while Python is still
   * starting up.
   */
  async compute(
    chunks: readonly RoiFilterChunkEntry[],
    groups: readonly RoiGroupConfig[],
  ): Promise<RoiFilterResult | undefined> {
    if (this.unavailable) return undefined;
    try {
      return this.noteSuccess(
        await this.request(chunks, groups, /*forceFullUpload=*/ false),
      );
    } catch (e) {
      if (e instanceof RoiFilterGeometryMissing) {
        // The service lost our geometry (its worker restarted). Re-upload
        // everything once rather than filtering against a partial set.
        this.uploaded.clear();
        try {
          return this.noteSuccess(
            await this.request(chunks, groups, /*forceFullUpload=*/ true),
          );
        } catch (retryError) {
          return this.noteFailure(retryError);
        }
      }
      return this.noteFailure(e);
    }
  }

  private noteSuccess(
    result: RoiFilterResult | undefined,
  ): RoiFilterResult | undefined {
    if (result !== undefined) this.consecutiveFailures = 0;
    return result;
  }

  private noteFailure(e: unknown): undefined {
    // A single failure is treated as transient -- this recompute falls back and
    // the next one tries again. A run of them says the service is not coming
    // back, and retrying costs a full geometry upload each time.
    if (++this.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
      this.unavailable = true;
      console.error(
        `ROI dissection service failed ${this.consecutiveFailures} times in a row; ` +
          "evaluating in the worker for the rest of this session.",
        e,
      );
    } else {
      console.warn(
        "ROI dissection service request failed; evaluating in the worker instead.",
        e,
      );
    }
    return undefined;
  }

  private async request(
    chunks: readonly RoiFilterChunkEntry[],
    groups: readonly RoiGroupConfig[],
    forceFullUpload: boolean,
  ): Promise<RoiFilterResult | undefined> {
    const uploads: { key: string }[] = [];
    const blobs: Uint8Array[] = [];
    let totalBytes = 0;
    for (const entry of chunks) {
      if (!forceFullUpload && this.uploaded.has(entry.key)) continue;
      const blob = encodeChunkGeometry(entry.data);
      uploads.push({ key: entry.key });
      blobs.push(blob);
      totalBytes += blob.byteLength;
    }
    // Touch this batch as most-recently-used, then retract whatever has aged
    // out beyond the cap. Only keys absent from the current batch are dropped,
    // so a region drag -- same chunks, new regions -- sends an empty `drop` and
    // leaves the service's combined index intact, which is what keeps a drag
    // cheap.
    const batch = new Set(chunks.map((c) => c.key));
    for (const key of batch) {
      if (this.uploaded.delete(key)) this.uploaded.add(key);
    }
    // Counted against what this request is about to add, not just what is
    // already held, or the cache settles one chunk above the cap forever.
    const incoming = uploads.reduce(
      (n, u) => n + (this.uploaded.has(u.key) ? 0 : 1),
      0,
    );
    const drop: string[] = [];
    for (const key of this.uploaded) {
      if (this.uploaded.size + incoming - drop.length <= MAX_CACHED_CHUNKS) {
        break;
      }
      // Never retract a chunk this request needs, even if that means briefly
      // exceeding the cap: the resident set is not negotiable.
      if (!batch.has(key)) drop.push(key);
    }

    const params = {
      scope: this.scope,
      chunkKeys: chunks.map((c) => c.key),
      uploads,
      drop,
      groups: groups.map(groupToJson),
    };

    const body = new Uint8Array(totalBytes);
    let offset = 0;
    for (const blob of blobs) {
      body.set(blob, offset);
      offset += blob.byteLength;
    }

    let response: Response;
    try {
      response = await fetch(
        `${ROI_FILTER_PATH}/${encodeURIComponent(this.scope)}?p=${encodeURIComponent(
          JSON.stringify(params),
        )}`,
        { method: "POST", body },
      );
    } catch {
      // No interceptor at all (ordinary build, or the service worker is not
      // controlling this worker yet). Fall back for the rest of the session.
      this.unavailable = true;
      return undefined;
    }

    if (response.status === 404) {
      this.unavailable = true;
      return undefined;
    }
    if (response.status === 409) {
      throw new RoiFilterGeometryMissing();
    }
    if (!response.ok) {
      throw new Error(
        `roi_filter service returned ${response.status}: ${await response.text()}`,
      );
    }

    // Committed only once the service has accepted them, so a failed request
    // leaves the two caches agreeing about what is held.
    for (const key of drop) this.uploaded.delete(key);
    for (const upload of uploads) this.uploaded.add(upload.key);

    return decodeResponse(await response.arrayBuffer());
  }
}

/** The service does not hold geometry we believed it had cached. */
class RoiFilterGeometryMissing extends Error {
  constructor() {
    super("roi_filter service is missing cached geometry");
  }
}
