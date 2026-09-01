/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Dropping the objects a memory budget did not admit, from a decoded pass-1
 * chunk.
 *
 * This is the step that turns {@link ObjectAdmission} from an arithmetic answer
 * into fewer bytes on the GPU. It runs on the FULLY assembled chunk — after
 * cross-chunk bridge insertion — for one reason: a ghost vertex inherits its
 * host endpoint's segment id (`appendGhostVertices`), so filtering by object
 * here keeps or drops a bridge together with the tract it belongs to, and no
 * separate reasoning about bridges is needed. Filtering earlier would leave the
 * cross-chunk link table's vertex indices pointing into a numbering that no
 * longer exists.
 */

import { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import type {
  AttributeTypedArray,
  SkeletonChunk,
} from "#src/datasource/zarr-vectors/geometry_chunk.js";

/**
 * Decides whether one object's geometry is drawn, from the two halves of its
 * uint64 segment id.
 */
export type ObjectPredicate = (idLow: number, idHigh: number) => boolean;

/**
 * `chunk` with every non-admitted object's geometry removed.
 *
 * Returns the input untouched when nothing would be dropped, so the common
 * "everything is admitted" case costs one pass and no allocation, and stores
 * without a segment column (no way to attribute geometry to an object) are
 * unaffected.
 *
 * Edges and faces survive only with all of their endpoints, so no dangling
 * index can reach the GPU. Because admission is decided per object and a
 * fragment belongs to exactly one object, that condition never actually splits
 * a primitive — it is enforced anyway rather than assumed.
 */
export function filterChunkByAdmittedObjects(
  chunk: SkeletonChunk,
  admits: ObjectPredicate,
): SkeletonChunk {
  const { segmentIds, numVertices } = chunk;
  // `segmentIdsAreGlobal` is the load-bearing check, not `segmentIds !== undefined`.
  // The decoder ALWAYS populates a segment column for an object-model kind, but
  // substitutes the fragment's index WITHIN THE CHUNK when
  // `fragment_attributes/segment_id` is missing or short. Admitting on those
  // would give the same tract a different id in every cell it crosses, so a
  // rank cut would keep it here and drop it next door — shattering every tract
  // at every chunk boundary rather than thinning them evenly.
  if (
    segmentIds === undefined ||
    chunk.segmentIdsAreGlobal !== true ||
    numVertices === 0
  ) {
    return chunk;
  }

  // One decision per OBJECT, not per vertex: a tract is ~200 vertices, and the
  // predicate hashes.
  const decisions = new Map<string, boolean>();
  const keep = new Uint8Array(numVertices);
  let numKept = 0;
  for (let v = 0; v < numVertices; ++v) {
    const low = segmentIds[v * 2] >>> 0;
    const high = segmentIds[v * 2 + 1] >>> 0;
    const key = `${low}:${high}`;
    let decision = decisions.get(key);
    if (decision === undefined) {
      decision = admits(low, high);
      decisions.set(key, decision);
    }
    if (decision) {
      keep[v] = 1;
      ++numKept;
    }
  }
  if (numKept === numVertices) return chunk;

  const { rank, positions, edges, faces, tangents, vertexAttributes } = chunk;
  const remap = new Int32Array(numVertices).fill(-1);
  const kept = new Uint32Array(numKept);
  let out = 0;
  for (let v = 0; v < numVertices; ++v) {
    if (keep[v] === 0) continue;
    remap[v] = out;
    kept[out] = v;
    ++out;
  }

  const newPositions = new Float32Array(numKept * rank);
  for (let i = 0; i < numKept; ++i) {
    const v = kept[i];
    for (let d = 0; d < rank; ++d) {
      newPositions[i * rank + d] = positions[v * rank + d];
    }
  }

  const newSegmentIds = new Uint32Array(numKept * 2);
  for (let i = 0; i < numKept; ++i) {
    const v = kept[i];
    newSegmentIds[i * 2] = segmentIds[v * 2];
    newSegmentIds[i * 2 + 1] = segmentIds[v * 2 + 1];
  }

  let newTangents: Float32Array | undefined;
  if (tangents !== undefined) {
    newTangents = new Float32Array(numKept * 3);
    for (let i = 0; i < numKept; ++i) {
      const v = kept[i];
      newTangents[i * 3] = tangents[v * 3];
      newTangents[i * 3 + 1] = tangents[v * 3 + 1];
      newTangents[i * 3 + 2] = tangents[v * 3 + 2];
    }
  }

  const newAttributes: AttributeTypedArray[] = vertexAttributes.map((src) => {
    const Ctor = src.constructor as new (n: number) => AttributeTypedArray;
    const dst = new Ctor(numKept);
    for (let i = 0; i < numKept; ++i) dst[i] = src[kept[i]] as never;
    return dst;
  });

  return {
    rank,
    numVertices: numKept,
    positions: newPositions,
    ...filterPrimitives(edges, faces, keep, remap),
    tangents: newTangents,
    vertexAttributes: newAttributes,
    segmentIds: newSegmentIds,
    segmentIdsAreGlobal: true,
    fragmentIndex: remapFragmentIndex(chunk.fragmentIndex, remap),
  };
}

/** Edges and faces restricted to primitives all of whose corners survived. */
function filterPrimitives(
  edges: Uint32Array,
  faces: Uint32Array | undefined,
  keep: Uint8Array,
  remap: Int32Array,
): {
  numEdges: number;
  edges: Uint32Array;
  faces?: Uint32Array;
  numFaces?: number;
} {
  const keptEdges: number[] = [];
  for (let e = 0; e + 1 < edges.length; e += 2) {
    const a = edges[e];
    const b = edges[e + 1];
    if (keep[a] === 1 && keep[b] === 1) {
      keptEdges.push(remap[a], remap[b]);
    }
  }
  const newEdges = Uint32Array.from(keptEdges);
  if (faces === undefined) {
    return { numEdges: newEdges.length >> 1, edges: newEdges };
  }
  const keptFaces: number[] = [];
  for (let f = 0; f + 2 < faces.length; f += 3) {
    const a = faces[f];
    const b = faces[f + 1];
    const c = faces[f + 2];
    if (keep[a] === 1 && keep[b] === 1 && keep[c] === 1) {
      keptFaces.push(remap[a], remap[b], remap[c]);
    }
  }
  const newFaces = Uint32Array.from(keptFaces);
  return {
    numEdges: newEdges.length >> 1,
    edges: newEdges,
    faces: newFaces,
    numFaces: newFaces.length / 3,
  };
}

/**
 * The fragment index rebuilt over the surviving vertices.
 *
 * Range fragments are preserved as ranges wherever their remapped rows stay
 * contiguous, which — because admission keeps or drops whole objects and the
 * remap is order-preserving — is very nearly always. That matters for memory,
 * not elegance: this index is retained for the ROI filter and charged to the
 * system-memory budget, and an all-explicit rebuild costs 8 bytes per VERTEX
 * against 16 bytes per FRAGMENT for a range. On a level-0 cell of ~500k
 * vertices that is megabytes per chunk, hundreds across the volume.
 *
 * Fragments losing every row are dropped rather than kept empty, so the ROI
 * filter never walks an object that is not drawn.
 */
function remapFragmentIndex(
  index: FragmentIndex,
  remap: Int32Array,
): FragmentIndex {
  const isRange: boolean[] = [];
  const ranges: bigint[] = [];
  const offsets: number[] = [0];
  const rows: bigint[] = [];
  for (let f = 0; f < index.numFragments; ++f) {
    const source = index.indices(f);
    const kept: number[] = [];
    for (let i = 0; i < source.length; ++i) {
      const to = remap[source[i]];
      if (to >= 0) kept.push(to);
    }
    if (kept.length === 0) continue;
    let contiguous = true;
    for (let i = 1; i < kept.length; ++i) {
      if (kept[i] !== kept[i - 1] + 1) {
        contiguous = false;
        break;
      }
    }
    if (contiguous) {
      isRange.push(true);
      ranges.push(BigInt(kept[0]), BigInt(kept.length));
    } else {
      isRange.push(false);
      for (const row of kept) rows.push(BigInt(row));
      offsets.push(rows.length);
    }
  }
  const numFragments = isRange.length;
  // The bitmap marks which fragments are ranges; `range()` indexes the range
  // table by popcount prefix, so the table must be in fragment order — which is
  // the order it was pushed in.
  const bitmap = new Uint8Array((numFragments + 7) >> 3);
  for (let f = 0; f < numFragments; ++f) {
    if (isRange[f]) bitmap[f >> 3] |= 1 << (f & 7);
  }
  return new FragmentIndex(
    numFragments,
    bitmap,
    BigInt64Array.from(ranges),
    Uint32Array.from(offsets),
    BigInt64Array.from(rows),
  );
}
