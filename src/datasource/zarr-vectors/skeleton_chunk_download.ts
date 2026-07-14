/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Worker-side orchestrator: fetch the byte blobs for one spatial chunk
 * of a zarr-vectors skeleton / polyline / streamline store and decode
 * them into a `SkeletonChunk` ready for upload to the render layer.
 *
 * This module is deliberately decoupled from neuroglancer's SharedObject
 * / RPC scaffolding so it can be unit-tested with a mock kvstore, AND
 * from the on-disk container format (native-sharded per-chunk arrays,
 * see `sharded_array.ts`) — callers supply `readArrayChunk` /
 * `readArrayChunkScoped`, which know how to resolve one chunk's payload
 * for a given array name; this module just names the arrays it needs and
 * decodes their bytes.
 */

import { decodeFragments } from "#src/datasource/zarr-vectors/fragment_index.js";
import {
  buildSkeletonChunk,
  type AttributeTypedArray,
  type GhostVertexRecord,
  type LinksConvention,
  type SkeletonChunk,
  type SkeletonGeometryKind,
} from "#src/datasource/zarr-vectors/skeleton_chunk.js";

/** Supported on-disk integer dtype for `links/0/<chunk>`. */
export type LinkDtype =
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "int64";

/** Supported on-disk dtype for a per-vertex attribute. */
export type AttributeDtype =
  | "float32"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32";

/** Fetch + decode one per-chunk array's whole payload for `chunkCoords`, or `undefined` if absent/empty. */
export type ReadArrayChunk = (
  arrayPath: string,
  chunkCoords: readonly number[],
  signal: AbortSignal,
) => Promise<Uint8Array | undefined>;

/**
 * Fetch a byte sub-range WITHIN one chunk's raw (uncompressed) payload —
 * `innerByteRange` addresses the decoded element bytes directly (e.g.
 * `rowIndex * bytesPerElement`). Only valid for arrays whose cells are
 * uncompressed (`vertices`/`vertex_attributes/<name>` when the writer
 * stamped `cell_codec: "raw"` — see `sharded_array.ts`); callers should
 * have already gated on that before passing this in.
 */
export type ReadArrayChunkScoped = (
  arrayPath: string,
  chunkCoords: readonly number[],
  innerByteRange: { offset: number; length: number },
  signal: AbortSignal,
) => Promise<Uint8Array | undefined>;

/**
 * Inputs the orchestrator needs to download a chunk.
 *
 * `readArrayChunk` resolves to `undefined` for a missing/empty chunk
 * (sparse chunk presence) — the orchestrator interprets that as "no data
 * here" and returns an empty `SkeletonChunk` when even the vertex array
 * is absent.
 */
export interface SkeletonChunkDownloadOptions {
  /** Spatial chunk grid coordinates. */
  readonly chunkCoords: readonly number[];
  /** Rank of the position vectors (== sid_ndim). */
  readonly rank: number;
  /**
   * On-disk dtype of `links/0/<chunk>` (per its array's `zarr.json`
   * `attributes.dtype`); used to reinterpret link bytes correctly. Not
   * consulted for `implicit_sequential` stores (which don't have a
   * `links/0` array).
   */
  readonly linkDtype: LinkDtype;
  /** Per-vertex attribute names, in the order the render layer expects. */
  readonly attributeNames: readonly string[];
  /** Per-vertex attribute dtypes, parallel to `attributeNames`. */
  readonly attributeDtypes: readonly AttributeDtype[];
  /** How vertex-to-vertex edges are encoded for this geometry type. */
  readonly linksConvention: LinksConvention;
  /** Geometry kind (drives whether per-vertex tangents are precomputed). */
  readonly geometryKind: SkeletonGeometryKind;
  /**
   * Whether `fragment_attributes/segment_id` chunks are present in this
   * level.  When false the fetch is skipped entirely (avoids 404 noise for
   * stores that never wrote per-fragment segment ids, e.g. streamline stores).
   * Defaults to true for backward-compatibility with callers that don't set it.
   */
  readonly hasFragmentSegmentIds?: boolean;
  readonly readArrayChunk: ReadArrayChunk;
}

/** Number of bytes per element of an attribute / link dtype. */
const BYTES_PER_ELEMENT: Record<LinkDtype | AttributeDtype, number> = {
  float32: 4,
  uint8: 1,
  uint16: 2,
  uint32: 4,
  int8: 1,
  int16: 2,
  int32: 4,
  int64: 8,
};

const ATTRIBUTE_CTORS: Record<
  AttributeDtype,
  | Float32ArrayConstructor
  | Uint8ArrayConstructor
  | Uint16ArrayConstructor
  | Uint32ArrayConstructor
  | Int8ArrayConstructor
  | Int16ArrayConstructor
  | Int32ArrayConstructor
> = {
  float32: Float32Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
};

/**
 * Reinterpret a byte blob as a typed array of the given dtype.  Returns
 * a possibly-aligned view (zero-copy when the source buffer is aligned
 * to the element size) or a copy when alignment forbids the in-place
 * view.  Throws if the byte length is not a multiple of the dtype size.
 */
function reinterpretBytes(
  bytes: Uint8Array,
  dtype: AttributeDtype,
  expectedElements: number,
): AttributeTypedArray {
  const elementSize = BYTES_PER_ELEMENT[dtype];
  const expectedBytes = expectedElements * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors chunk: dtype=${dtype} expected ${expectedBytes} bytes ` +
        `(${expectedElements} elements), got ${bytes.byteLength}`,
    );
  }
  const Ctor = ATTRIBUTE_CTORS[dtype];
  if (bytes.byteOffset % elementSize === 0) {
    return new (Ctor as any)(bytes.buffer, bytes.byteOffset, expectedElements);
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new (Ctor as any)(copy.buffer, 0, expectedElements);
}

/** Widen any integer dtype's link buffer to chunk-local `Uint32Array`. */
function reinterpretLinkBytes(
  bytes: Uint8Array,
  dtype: LinkDtype,
  numEdges: number,
  linkWidth: number,
): Uint32Array {
  const elementSize = BYTES_PER_ELEMENT[dtype];
  const expectedBytes = numEdges * linkWidth * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors links chunk: dtype=${dtype} link_width=${linkWidth} ` +
        `expected ${expectedBytes} bytes (${numEdges} edges), got ${bytes.byteLength}`,
    );
  }
  const total = numEdges * linkWidth;
  const out = new Uint32Array(total);
  if (dtype === "int64") {
    const aligned =
      bytes.byteOffset % 8 === 0
        ? new BigInt64Array(bytes.buffer, bytes.byteOffset, total)
        : (() => {
            const copy = new Uint8Array(bytes.byteLength);
            copy.set(bytes);
            return new BigInt64Array(copy.buffer, 0, total);
          })();
    for (let i = 0; i < total; ++i) out[i] = Number(aligned[i]);
    return out;
  }
  const arr = reinterpretBytes(bytes, dtype, total);
  // Copy across to Uint32Array — narrow uints widen losslessly,
  // signed types may carry negative values (rejected upstream by writer).
  for (let i = 0; i < total; ++i) out[i] = arr[i];
  return out;
}

/**
 * Download and decode one spatial chunk into a `SkeletonChunk`.
 *
 * Reads (via `options.readArrayChunk`):
 *   - `vertices`               — required for non-empty chunks.
 *   - `vertex_fragments`       — required for non-empty chunks.
 *   - `links/0`                — required unless `linksConvention === "implicit_sequential"`.
 *   - `vertex_attributes/<name>` — one per declared attribute name.
 *   - `fragment_attributes/segment_id` — when `hasFragmentSegmentIds`.
 *
 * Returns `undefined` when the chunk's `vertices` payload is absent (the
 * canonical "this chunk has no data" signal).  Throws on any partially-
 * present chunk (vertices but missing fragments, or fragments but
 * missing edges in the explicit-edge cases).
 */
export async function downloadSkeletonChunk(
  options: SkeletonChunkDownloadOptions,
  signal: AbortSignal,
): Promise<SkeletonChunk | undefined> {
  const {
    chunkCoords,
    rank,
    linkDtype,
    attributeNames,
    attributeDtypes,
    linksConvention,
    geometryKind,
    readArrayChunk,
  } = options;
  if (attributeNames.length !== attributeDtypes.length) {
    throw new Error(
      `downloadSkeletonChunk: attributeNames (${attributeNames.length}) ` +
        `and attributeDtypes (${attributeDtypes.length}) length mismatch`,
    );
  }

  // 1. Vertices — required.
  const vertexBytes = await readArrayChunk("vertices", chunkCoords, signal);
  if (vertexBytes === undefined || vertexBytes.byteLength === 0) {
    return undefined;
  }
  const bytesPerVertex = rank * 4; // float32
  if (vertexBytes.byteLength % bytesPerVertex !== 0) {
    throw new Error(
      `zarr-vectors vertices/${chunkCoords.join(".")}: ${vertexBytes.byteLength} bytes ` +
        `not a multiple of ${bytesPerVertex} (rank=${rank} * float32)`,
    );
  }
  const numVertices = vertexBytes.byteLength / bytesPerVertex;
  const positions = reinterpretBytes(
    vertexBytes,
    "float32",
    numVertices * rank,
  ) as Float32Array;

  // 2. Fragment index — required.
  const fragmentBytes = await readArrayChunk(
    "vertex_fragments",
    chunkCoords,
    signal,
  );
  if (fragmentBytes === undefined) {
    throw new Error(
      `zarr-vectors chunk ${chunkCoords.join(".")} has vertices but vertex_fragments is missing`,
    );
  }
  const fragmentIndex = decodeFragments(fragmentBytes);

  // 3. Explicit edges (links/0/<chunk>) — required for explicit /
  // implicit_sequential_with_branches, absent for pure implicit_sequential.
  let explicitEdges: Uint32Array | undefined;
  if (
    linksConvention === "explicit" ||
    linksConvention === "implicit_sequential_with_branches"
  ) {
    const linkBytes = await readArrayChunk("links/0", chunkCoords, signal);
    if (linkBytes === undefined || linkBytes.byteLength === 0) {
      // implicit_sequential_with_branches with no explicit branches in
      // this chunk is legitimate (a leaf-only sub-skeleton).
      explicitEdges = new Uint32Array(0);
    } else {
      const elementSize = BYTES_PER_ELEMENT[linkDtype];
      const totalElements = linkBytes.byteLength / elementSize;
      if (totalElements % 2 !== 0) {
        throw new Error(
          `zarr-vectors links/0/${chunkCoords.join(".")}: ${totalElements} elements is ` +
            `not a multiple of link_width=2`,
        );
      }
      const numEdges = totalElements / 2;
      explicitEdges = reinterpretLinkBytes(linkBytes, linkDtype, numEdges, 2);
    }
  }

  // 4. Per-vertex attributes — one fetch per declared attribute name.
  // A missing per-chunk attribute blob is tolerated and degrades to a
  // zero-filled array of the declared dtype.  Two situations trigger
  // this in practice:
  //
  //   - The writer's pyramid coarsening doesn't propagate
  //     `vertex_attributes/<name>` to higher levels; coarser levels
  //     have vertices but no attribute arrays.
  //   - Future writers may emit attributes sparsely (per-chunk
  //     opt-in).
  //
  // The user-visible effect is `prop_<name>()` evaluating to 0 inside
  // the shader for chunks without that attribute.  This matches how
  // the spatially-indexed skeleton shader handles "this segment has no
  // value" elsewhere and avoids cascading layer failures from a
  // single missing optional blob.
  // Fetched concurrently with `fragment_attributes/segment_id` below (5):
  // both are independent per-chunk reads, and awaiting them sequentially
  // would tack a full extra round trip onto every chunk's download — this
  // sits directly on the critical path for how quickly a chunk newly
  // scrolled into view becomes pickable, which visibly stalled continuous
  // multi-segment drag-selection once `hasFragmentSegmentIds` (5) became
  // true for every pyramid level instead of only some.
  const numFragments = fragmentIndex.numFragments;
  const segFragBytesPromise: Promise<Uint8Array | undefined> =
    (options.hasFragmentSegmentIds ?? true)
      ? readArrayChunk("fragment_attributes/segment_id", chunkCoords, signal)
      : Promise.resolve(undefined);
  const [vertexAttributes, segFragBytes] = await Promise.all([
    Promise.all(
      attributeNames.map(async (name, i) => {
        const bytes = await readArrayChunk(
          `vertex_attributes/${name}`,
          chunkCoords,
          signal,
        );
        if (bytes === undefined) {
          return reinterpretBytes(
            new Uint8Array(numVertices * BYTES_PER_ELEMENT[attributeDtypes[i]]),
            attributeDtypes[i],
            numVertices,
          );
        }
        return reinterpretBytes(bytes, attributeDtypes[i], numVertices);
      }),
    ),
    segFragBytesPromise,
  ]);

  // 5. Per-fragment segment_id → synthesised per-vertex "segment" column.
  // The writer stores one uint64 flywire id per fragment under
  // `fragment_attributes/segment_id`.  We expand it to a per-vertex
  // column of the FULL uint64, stored interleaved as two uint32
  // components `[lo, hi, lo, hi, …]` (a `uvec2` on the GPU).  The render
  // layer hashes the full id with the same `segmentColorHash` as the flat
  // segmentation, so dense fragments colour identically to that segment's
  // voxels, and a picked fragment surfaces the global id.  Colours/ids stay
  // consistent across chunks and pyramid levels because the id is intrinsic
  // to the segment.  When the blob is absent or mis-sized (older stores, or
  // streamline/polyline geometries that never wrote it), fall back to the
  // fragment's index within the chunk: still distinct per fragment, just not
  // unified across chunks (`[f, 0]`).
  let fragSegIds: BigUint64Array | undefined;
  if (
    segFragBytes !== undefined &&
    segFragBytes.byteLength >= numFragments * 8
  ) {
    fragSegIds = new BigUint64Array(
      segFragBytes.buffer.slice(
        segFragBytes.byteOffset,
        segFragBytes.byteOffset + numFragments * 8,
      ),
    );
  }
  // Two uint32 per vertex: [lo, hi].
  const segmentIds = new Uint32Array(numVertices * 2);
  for (let f = 0; f < numFragments; ++f) {
    let lo: number;
    let hi: number;
    if (fragSegIds !== undefined) {
      const id = fragSegIds[f];
      lo = Number(id & 0xffffffffn) >>> 0;
      hi = Number((id >> 32n) & 0xffffffffn) >>> 0;
    } else {
      lo = f;
      hi = 0;
    }
    const idxs = fragmentIndex.indices(f);
    for (let k = 0; k < idxs.length; ++k) {
      const v = idxs[k];
      segmentIds[v * 2] = lo;
      segmentIds[v * 2 + 1] = hi;
    }
  }

  return buildSkeletonChunk({
    rank,
    positions,
    fragmentIndex,
    explicitEdges,
    linksConvention,
    geometryKind,
    vertexAttributes,
    segmentIds,
  });
}

/**
 * Fragment-scoped counterpart of {@link downloadSkeletonChunk}: decode a
 * chunk but fetch, over the network, ONLY the vertices (and per-vertex
 * attributes) of the fragments named by `restrictToFragments`, via
 * `readArrayChunkScoped` byte-range reads keyed off the (small, whole-
 * read) `vertex_fragments` index — instead of downloading the whole
 * chunk's vertex payload just to discard everything but one selected
 * object's fragments.
 *
 * This is the pass-2 (object-keyed) path: a selected streamline owns only
 * a handful of a chunk's (often thousands of) fragments, so the whole-
 * chunk read is mostly waste.  Pass-1 (spatial) still uses
 * {@link downloadSkeletonChunk} — it renders every fragment in the chunk.
 *
 * Correctness is by construction: the returned `SkeletonChunk` is
 * identical to what {@link downloadSkeletonChunk} would produce EXCEPT
 * that vertices/attributes outside the owned fragments' covering span are
 * left zero-filled. `filterChunkByFragments` (the required next step)
 * reads only owned-fragment vertices, so the zero-filled remainder is
 * dropped and the filtered output matches the whole-chunk path exactly.
 * (CPU/memory still scale with the chunk's vertex count — only the
 * network read is scoped; the win is I/O, which is the bottleneck for
 * remote stores.)
 *
 * Scoped to `implicit_sequential` (streamline/polyline): it reads no
 * `links/0` array (edges are synthesised from fragment ranges), so no
 * whole-array edge read is needed. Other conventions keep the whole-chunk
 * path.
 *
 * Returns `undefined` for an absent chunk (no `vertex_fragments`), same
 * as {@link downloadSkeletonChunk}'s empty signal. Throws if a range read
 * returns an unexpected length (a store that silently ignored the range —
 * the caller should have gated on offset-read support first).
 */
export async function downloadSkeletonChunkScoped(
  options: SkeletonChunkDownloadOptions & {
    readonly restrictToFragments: Uint32Array;
    readonly readArrayChunkScoped: ReadArrayChunkScoped;
  },
  signal: AbortSignal,
): Promise<SkeletonChunk | undefined> {
  const {
    chunkCoords,
    rank,
    attributeNames,
    attributeDtypes,
    linksConvention,
    geometryKind,
    readArrayChunk,
    readArrayChunkScoped,
    restrictToFragments,
  } = options;
  if (attributeNames.length !== attributeDtypes.length) {
    throw new Error(
      `downloadSkeletonChunkScoped: attributeNames (${attributeNames.length}) ` +
        `and attributeDtypes (${attributeDtypes.length}) length mismatch`,
    );
  }
  if (linksConvention !== "implicit_sequential") {
    throw new Error(
      `downloadSkeletonChunkScoped: only supports implicit_sequential, ` +
        `got ${linksConvention}`,
    );
  }

  // 1. Fragment index — read whole (small); needed to size the chunk and
  // locate each owned fragment's contiguous vertex range. Its absence is
  // the canonical "chunk has no data" signal (mirrors downloadSkeletonChunk).
  const fragmentBytes = await readArrayChunk(
    "vertex_fragments",
    chunkCoords,
    signal,
  );
  if (fragmentBytes === undefined) return undefined;
  const fragmentIndex = decodeFragments(fragmentBytes);

  // True chunk vertex count = highest vertex index referenced by any
  // fragment + 1 (every vertex belongs to a fragment). Lets us allocate a
  // full-length positions array without a separate size read, so
  // fragment-index-referenced indices stay in-bounds during decode.
  let numVertices = 0;
  for (let f = 0; f < fragmentIndex.numFragments; ++f) {
    if (fragmentIndex.isRange(f)) {
      const { start, count } = fragmentIndex.range(f);
      if (count > 0) numVertices = Math.max(numVertices, start + count);
    } else {
      const idx = fragmentIndex.indices(f);
      for (let k = 0; k < idx.length; ++k) {
        numVertices = Math.max(numVertices, idx[k] + 1);
      }
    }
  }
  if (numVertices === 0) return undefined;

  // Covering [minIdx, maxIdx] over all owned fragments' vertex indices.
  // For streamlines (contiguous range fragments, usually one per chunk)
  // this is tight; scattered explicit fragments widen it (still correct,
  // just less savings).
  let minIdx = Number.POSITIVE_INFINITY;
  let maxIdx = -1;
  for (let i = 0; i < restrictToFragments.length; ++i) {
    const f = restrictToFragments[i];
    if (f >= fragmentIndex.numFragments) continue;
    if (fragmentIndex.isRange(f)) {
      const { start, count } = fragmentIndex.range(f);
      if (count > 0) {
        minIdx = Math.min(minIdx, start);
        maxIdx = Math.max(maxIdx, start + count - 1);
      }
    } else {
      const idx = fragmentIndex.indices(f);
      for (let k = 0; k < idx.length; ++k) {
        minIdx = Math.min(minIdx, idx[k]);
        maxIdx = Math.max(maxIdx, idx[k]);
      }
    }
  }

  const bytesPerVertex = rank * 4; // float32
  const positions = new Float32Array(numVertices * rank);
  if (maxIdx >= 0) {
    const spanVertices = maxIdx - minIdx + 1;
    const offset = minIdx * bytesPerVertex;
    const length = spanVertices * bytesPerVertex;
    const spanBytes = await readArrayChunkScoped(
      "vertices",
      chunkCoords,
      { offset, length },
      signal,
    );
    // Owned fragments exist (maxIdx >= 0) so their vertices must too;
    // an absent or wrong-length range read means the store didn't honor
    // the request — throw so the caller falls back to a whole-chunk read
    // rather than silently dropping this block's geometry.
    if (spanBytes === undefined || spanBytes.byteLength !== length) {
      throw new Error(
        `zarr-vectors vertices/${chunkCoords.join(".")}: scoped range read returned ` +
          `${spanBytes === undefined ? "no data" : `${spanBytes.byteLength} bytes`}, ` +
          `expected ${length}`,
      );
    }
    const spanFloats = reinterpretBytes(
      spanBytes,
      "float32",
      spanVertices * rank,
    ) as Float32Array;
    positions.set(spanFloats, minIdx * rank);
  }

  // Per-vertex attributes — same covering span. A missing attribute blob
  // (undefined) zero-fills, exactly as downloadSkeletonChunk does.
  const vertexAttributes = await Promise.all(
    attributeNames.map(async (name, i) => {
      const dtype = attributeDtypes[i];
      const elementSize = BYTES_PER_ELEMENT[dtype];
      const arr = reinterpretBytes(
        new Uint8Array(numVertices * elementSize),
        dtype,
        numVertices,
      );
      if (maxIdx >= 0) {
        const spanElements = maxIdx - minIdx + 1;
        const spanBytes = await readArrayChunkScoped(
          `vertex_attributes/${name}`,
          chunkCoords,
          { offset: minIdx * elementSize, length: spanElements * elementSize },
          signal,
        );
        if (
          spanBytes !== undefined &&
          spanBytes.byteLength === spanElements * elementSize
        ) {
          const span = reinterpretBytes(spanBytes, dtype, spanElements);
          (arr as unknown as { set: (a: ArrayLike<number>, o: number) => void }).set(
            span as unknown as ArrayLike<number>,
            minIdx,
          );
        }
      }
      return arr;
    }),
  );

  // Per-fragment segment_id → per-vertex [lo, hi] column. Read whole (it's
  // one small uint64 per fragment); identical expansion to downloadSkeletonChunk.
  const numFragments = fragmentIndex.numFragments;
  const segFragBytes = (options.hasFragmentSegmentIds ?? true)
    ? await readArrayChunk("fragment_attributes/segment_id", chunkCoords, signal)
    : undefined;
  let fragSegIds: BigUint64Array | undefined;
  if (segFragBytes !== undefined && segFragBytes.byteLength >= numFragments * 8) {
    fragSegIds = new BigUint64Array(
      segFragBytes.buffer.slice(
        segFragBytes.byteOffset,
        segFragBytes.byteOffset + numFragments * 8,
      ),
    );
  }
  const segmentIds = new Uint32Array(numVertices * 2);
  for (let f = 0; f < numFragments; ++f) {
    let lo: number;
    let hi: number;
    if (fragSegIds !== undefined) {
      const id = fragSegIds[f];
      lo = Number(id & 0xffffffffn) >>> 0;
      hi = Number((id >> 32n) & 0xffffffffn) >>> 0;
    } else {
      lo = f;
      hi = 0;
    }
    const idxs = fragmentIndex.indices(f);
    for (let k = 0; k < idxs.length; ++k) {
      const v = idxs[k];
      segmentIds[v * 2] = lo;
      segmentIds[v * 2 + 1] = hi;
    }
  }

  return buildSkeletonChunk({
    rank,
    positions,
    fragmentIndex,
    linksConvention,
    geometryKind,
    vertexAttributes,
    segmentIds,
  });
}

/**
 * One request for a ghost vertex.  `hostLocalVertex` identifies the
 * endpoint in the current chunk; `neighborChunkCoords` + `neighborLocalVertex`
 * identify the source vertex in a different chunk to copy into the host.
 */
export interface GhostVertexRequest {
  readonly hostLocalVertex: number;
  readonly neighborChunkCoords: readonly number[];
  readonly neighborLocalVertex: number;
  /**
   * True when the neighbor's vertex precedes the host in the
   * streamline's walk order.  Forwarded onto the resulting
   * `GhostVertexRecord` so `appendGhostVertices` can flip the
   * synthesised ghost-tangent sign accordingly.  Defaults to `false`
   * (ghost is the successor) when callers don't specify it.
   */
  readonly isGhostPredecessor?: boolean;
}

/**
 * Slice a single float32×rank vertex out of a chunk's whole `vertices`
 * payload.  Returns `undefined` when the requested index is out of range
 * — caller drops the ghost in that case (avoids dangling references on
 * sparse / writer-inconsistent stores).
 */
function sliceVertexFromBytes(
  bytes: Uint8Array,
  vertexIndex: number,
  rank: number,
): Float32Array | undefined {
  const bytesPerVertex = rank * 4;
  const offset = vertexIndex * bytesPerVertex;
  if (vertexIndex < 0 || offset + bytesPerVertex > bytes.byteLength) {
    return undefined;
  }
  // Reinterpret a `rank`-element float32 slice.  Subarray gives a
  // zero-copy view; reinterpretBytes handles alignment.
  return reinterpretBytes(
    bytes.subarray(offset, offset + bytesPerVertex),
    "float32",
    rank,
  ) as Float32Array;
}

/**
 * Slice a single attribute element from a chunk's whole
 * `vertex_attributes/<name>` payload, packaged as a length-1 typed-array
 * of the declared dtype. Returns `undefined` when the requested index is
 * out of range.
 */
function sliceAttributeFromBytes(
  bytes: Uint8Array,
  vertexIndex: number,
  dtype: AttributeDtype,
): AttributeTypedArray | undefined {
  const elementSize = BYTES_PER_ELEMENT[dtype];
  const offset = vertexIndex * elementSize;
  if (vertexIndex < 0 || offset + elementSize > bytes.byteLength) {
    return undefined;
  }
  return reinterpretBytes(
    bytes.subarray(offset, offset + elementSize),
    dtype,
    1,
  );
}

/**
 * Fetch + slice one ghost vertex per request, grouping by
 * `neighborChunkCoords` so each unique neighbor's `vertices` and per-
 * attribute payloads are fetched exactly once.  Subsequent fetches for
 * the same key are served from the kvstore/shard-index cache (and when
 * the neighbor loads as its own render chunk, every byte is already
 * cached — the "prefetch" reorders work rather than adding net traffic).
 *
 * Requests whose neighbor's `vertices` payload is absent are silently
 * dropped (sparse chunk presence; we never emit a dangling reference).
 * Requests whose `vertex_attributes/<name>` payload is absent get a
 * zero-filled value for that attribute — same rule the per-chunk
 * download applies for pyramid levels that don't propagate attributes.
 */
export async function fetchGhostVertices(
  requests: readonly GhostVertexRequest[],
  options: {
    readonly rank: number;
    readonly attributeNames: readonly string[];
    readonly attributeDtypes: readonly AttributeDtype[];
    readonly readArrayChunk: ReadArrayChunk;
  },
  signal: AbortSignal,
): Promise<GhostVertexRecord[]> {
  const { rank, attributeNames, attributeDtypes, readArrayChunk } = options;
  if (requests.length === 0) return [];

  // 1. Group by neighbor chunk coords — one fetch per unique key per array.
  const byKeyString = new Map<string, readonly number[]>();
  for (const r of requests) {
    byKeyString.set(r.neighborChunkCoords.join(","), r.neighborChunkCoords);
  }

  // 2. Fetch positions + each attribute for each unique key in parallel.
  type NeighborBlobs = {
    positions: Uint8Array | undefined;
    attrs: Array<Uint8Array | undefined>;
  };
  const byKey = new Map<string, NeighborBlobs>();
  await Promise.all(
    Array.from(byKeyString.entries()).map(async ([keyString, coords]) => {
      const [positions, ...attrs] = await Promise.all([
        readArrayChunk("vertices", coords, signal),
        ...attributeNames.map((name) =>
          readArrayChunk(`vertex_attributes/${name}`, coords, signal),
        ),
      ]);
      byKey.set(keyString, { positions, attrs });
    }),
  );

  // 3. Slice each request's element.  Drop requests whose neighbor
  // positions blob is absent (sparse chunk) or whose vertex index is
  // out of range — these would otherwise create dangling bridge edges.
  const out: GhostVertexRecord[] = [];
  for (const req of requests) {
    const blobs = byKey.get(req.neighborChunkCoords.join(","));
    if (blobs === undefined || blobs.positions === undefined) continue;
    const position = sliceVertexFromBytes(
      blobs.positions,
      req.neighborLocalVertex,
      rank,
    );
    if (position === undefined) continue;
    const attributes: AttributeTypedArray[] = [];
    for (let i = 0; i < attributeNames.length; ++i) {
      const bytes = blobs.attrs[i];
      const sliced =
        bytes === undefined
          ? undefined
          : sliceAttributeFromBytes(
              bytes,
              req.neighborLocalVertex,
              attributeDtypes[i],
            );
      if (sliced === undefined) {
        // Zero-fill missing attribute — mirrors `downloadSkeletonChunk`
        // behavior for chunk-local attributes (pyramid levels without
        // `vertex_attributes/<name>`).
        attributes.push(
          reinterpretBytes(
            new Uint8Array(BYTES_PER_ELEMENT[attributeDtypes[i]]),
            attributeDtypes[i],
            1,
          ),
        );
      } else {
        attributes.push(sliced);
      }
    }
    out.push({
      position,
      attributes,
      bridgeFromLocalVertex: req.hostLocalVertex,
      isGhostPredecessor: req.isGhostPredecessor ?? false,
    });
  }
  return out;
}
