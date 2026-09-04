/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * How many whole objects of an object-sparsity pyramid to load, and what to do
 * with the ones that do not fit.
 *
 * Three steps of one decision, previously three modules: how many objects each
 * level holds, which of them a memory budget admits, and dropping the rest from
 * a decoded pass-1 chunk. They are only ever used together.
 */

import { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import type {
  AttributeTypedArray,
  SkeletonChunk,
} from "#src/datasource/zarr-vectors/geometry_chunk.js";

// ---------------------------------------------------------------- pyramid_objects

/**
 * Deriving how many *objects* each level of an object-sparsity pyramid holds.
 *
 * Split out of the datasource entry point because everything that budgets memory
 * for a tractogram divides by this number, and getting it wrong is silent: a
 * plausible-looking count flows all the way to the renderer and simply makes the
 * wrong amount of geometry load.
 */

/** The per-level metadata this derivation reads, finest-first. */
export interface ZarrVectorsLevelObjectMeta {
  /** ``zarr_vectors_level.vertex_count``. */
  vertexCount: number | undefined;
  /** ``zarr_vectors_level.object_sparsity``, relative to the parent level. */
  objectSparsity: number | undefined;
  /** ``zarr_vectors_level.inherited_num_objects``. */
  numObjects: number | undefined;
}

/**
 * Live object count per level, **finest-first**, or `undefined` per level where
 * it cannot be determined.
 *
 * This is the detail axis of an object-sparsity pyramid: every level covers the
 * whole volume with the same `chunk_shape`, and coarser levels simply hold fewer
 * *complete* objects. It cannot be read off `object_index/manifests`, whose
 * length is the object-id space rather than the live count — with
 * ``preserves_object_ids`` a dropped object keeps its slot and stores an empty
 * manifest, so every level reports the same shape.
 *
 * So it is derived: the finest level's count scaled by the running product of
 * ``object_sparsity`` down the chain.
 *
 * Getting that base count right is the whole difficulty.
 * ``inherited_num_objects`` is defined as the OID-space size inherited from the
 * parent level and is explicitly *absent on standalone levels* — which level 0
 * always is. Reading it only at level 0 therefore fails on every real pyramid,
 * and the ``vertex_count`` fallback then reports one "object" per vertex, ~200x
 * too many. That is not a harmless approximation:
 * {@link bytesPerObjectFromLevelCounts} detects it by the
 * `vertexCount === objectCount` degeneracy and declines to answer, so everything
 * downstream that needs a per-object cost stops working entirely.
 *
 * **Level 1's value is the one to use.** Its parent is level 0, so its
 * ``inherited_num_objects`` is level 0's object count under either reading of
 * the field — the preserved-id OID space, or a plain parent count.
 */
export function computePerLevelObjectCount(
  perLevelMeta: ReadonlyArray<ZarrVectorsLevelObjectMeta>,
): (number | undefined)[] {
  const n = perLevelMeta.length;
  const base = perLevelMeta[0]?.numObjects ?? perLevelMeta[1]?.numObjects;
  if (base !== undefined) {
    const out: (number | undefined)[] = [];
    let cumulative = 1;
    let ok = true;
    for (let k = 0; k < n; ++k) {
      const s = perLevelMeta[k].objectSparsity;
      if (s === undefined) {
        ok = false;
        break;
      }
      // Level 0's sparsity is 1.0; each later level's is relative to its parent.
      cumulative *= s;
      out.push(Math.max(1, base * cumulative));
    }
    if (ok) return out;
  }
  // No level declares the field. Vertex counts track object counts closely while
  // vertices-per-object stays roughly constant, but callers needing a TRUE object
  // count must treat `vertexCount === objectCount` as "unknown" — which is
  // exactly what this fallback produces.
  return perLevelMeta.map((m) => m.vertexCount);
}

/**
 * Per-level multiplier that spreads a pyramid the chunk shape cannot separate,
 * **finest-first**. All `1` when the counts are unavailable or already
 * reflected in how the chunks grow.
 *
 * A zarr-vectors pyramid does not have to coarsen its grid: this writer keeps
 * ONE `chunk_shape` for every level and thins the contents instead, so all five
 * levels of a whole-brain point cloud declare the same 0.5 mm spacing. That is
 * the only signal the render layer's level picker reads
 * (`findClosestSpatialSkeletonGridLevelBySpacing`), so without a correction it
 * sees five ties, takes the first, and the resolution histogram collapses to a
 * single bin -- a pyramid that exists on disk and is invisible to the UI.
 *
 * The correction restates thinning as coarsening: a level holding 1/8 as much
 * behaves like a grid twice as wide, hence the cube root. `objectCounts` is the
 * right numerator where the store has an object model, because that is what a
 * tractogram's levels actually differ in; `vertexCounts` is the same signal one
 * tier down and the only one a point cloud has, so it stands in when any object
 * count is missing rather than letting the whole correction lapse.
 *
 * `spacings` (finest-first, one scalar per level) is subtracted out: a writer
 * that DOES grow its chunks has already expressed part of the drop, and only
 * the shortfall is made up -- never less than 1, so this can coarsen a level's
 * apparent spacing but never sharpen it.
 */
export function computePyramidDensityScales(
  objectCounts: readonly (number | undefined)[],
  vertexCounts: readonly (number | undefined)[],
  spacings: readonly number[],
): number[] {
  const n = spacings.length;
  const ones = new Array<number>(n).fill(1);
  const counts =
    objectCounts.length === n && objectCounts.every((c) => c !== undefined)
      ? objectCounts
      : vertexCounts;
  if (counts.length !== n) return ones;
  // Anchor on the densest level; levels are finest-first, but derive it rather
  // than assume, so an unordered store still behaves.
  let finestCount: number | undefined;
  for (const c of counts) {
    if (c === undefined) return ones; // partial metadata: don't half-apply
    if (finestCount === undefined || c > finestCount) finestCount = c;
  }
  if (finestCount === undefined || finestCount <= 0) return ones;
  const finestSpacing = Math.min(...spacings);
  if (!(finestSpacing > 0)) return ones;
  return counts.map((c, k) => {
    const densityFactor = Math.cbrt(finestCount! / c!);
    const chunkGrowth = spacings[k] / finestSpacing;
    // Only make up the shortfall the chunk shape does not already express.
    return Math.max(1, densityFactor / chunkGrowth);
  });
}

// ---------------------------------------------------------------- object_admission

/**
 * Choosing WHICH whole objects to load, when a whole pyramid level will not fit.
 *
 * An object-sparsity pyramid gives coarse control over how many complete objects
 * are drawn — but only at its rungs, which are typically an order of magnitude
 * apart. A tractogram whose levels hold 503k / 50k / 5k tracts offers nothing at
 * all between 50k and 503k, so a budget that could hold 105k tracts draws 50k and
 * leaves the rest of the memory idle. That is not a tuning problem; it is the
 * whole reason such a viewer appears to "stop" at a fixed level.
 *
 * This module fills the gap by admitting a *subset* of a level's objects. It
 * relies on one property of the format, verified for real stores: with
 * ``preserves_object_ids`` the levels are strictly **nested** subsets of one id
 * space. Every object therefore has a well-defined **depth** — the coarsest level
 * that still contains it — and any prefix of the ordering "by depth descending,
 * then by a stable rank" is a valid, globally-spread, non-duplicating set.
 *
 * The admitted set is expressed as two numbers rather than a list of ids, so it
 * costs nothing to ship to the worker and nothing to re-evaluate per fragment.
 *
 * Level indices here are **store levels, finest-first**: level 0 is the finest
 * (every object), level `N-1` the coarsest. `{depth >= k}` is exactly the set of
 * objects present at level `k`.
 */

/** Absent from every level; distinguishable from a real depth of 0. */
export const OBJECT_DEPTH_ABSENT = 255;

/**
 * The admitted set: load `loadLevel`'s chunks, and within them keep every object
 * that also survives into a coarser level, plus `fraction` of the objects that
 * appear for the first time at this level.
 */
export interface ObjectAdmission {
  /** Store level whose chunks are fetched, finest-first. */
  loadLevel: number;
  /** Share of the objects new at `loadLevel` to keep, in [0, 1]. */
  fraction: number;
}

/** Admits everything, for stores with no usable per-level object metadata. */
export const ADMIT_ALL: ObjectAdmission = { loadLevel: 0, fraction: 1 };

/**
 * A stable, uniform rank in [0, 1) for an object id.
 *
 * Stability is what makes residency converge: the admitted set has to be the
 * same set frame after frame, or raising the budget would evict and refetch a
 * different sample every time instead of adding to the one already loaded.
 * Uniformity is what makes it spread across the volume — a contiguous run of ids
 * is very likely a contiguous run of the file, which for a tractogram is one
 * bundle or one seed region.
 *
 * `lowbias32` finalizer: strong avalanche, so neighbouring ids land far apart.
 */
export function objectRank(id: number): number {
  let x = id >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

/**
 * Per-object depth from per-level membership.
 *
 * `perLevelVertexCounts` is finest-first; entry `k` holds one vertex count per
 * object at level `k`, with **0 meaning absent** (the on-disk sentinel mapped
 * down). Depth is the largest level index at which the object is present.
 *
 * Returns {@link OBJECT_DEPTH_ABSENT} for ids present nowhere.
 */
export function objectDepths(
  perLevelVertexCounts: ReadonlyArray<Uint32Array>,
): Uint8Array {
  const numObjects = perLevelVertexCounts[0]?.length ?? 0;
  const depths = new Uint8Array(numObjects).fill(OBJECT_DEPTH_ABSENT);
  for (let k = 0; k < perLevelVertexCounts.length; ++k) {
    const counts = perLevelVertexCounts[k];
    for (let id = 0; id < numObjects; ++id) {
      // Levels are nested, so a later (coarser) hit always supersedes.
      if (counts[id] !== 0) depths[id] = k;
    }
  }
  return depths;
}

/**
 * Whether the levels really are nested — every level's members present in every
 * finer level.
 *
 * The depth model is only meaningful under nesting. A store that violates it
 * must fall back to whole-level selection rather than silently drop geometry,
 * so this is checked once at load rather than assumed.
 */
export function levelsAreNested(
  perLevelVertexCounts: ReadonlyArray<Uint32Array>,
): boolean {
  for (let k = 1; k < perLevelVertexCounts.length; ++k) {
    const coarser = perLevelVertexCounts[k];
    const finer = perLevelVertexCounts[k - 1];
    if (coarser.length !== finer.length) return false;
    for (let id = 0; id < coarser.length; ++id) {
      if (coarser[id] !== 0 && finer[id] === 0) return false;
    }
  }
  return true;
}

/** Total bytes of every object present at level `k`. */
function levelCostBytes(counts: Uint32Array, bytesPerVertex: number): number {
  let total = 0;
  for (let id = 0; id < counts.length; ++id) total += counts[id];
  return total * bytesPerVertex;
}

/**
 * Bytes the objects surviving into `depthAtLeast` cost when read from level
 * `atLevel`'s geometry.
 *
 * The distinction matters: once a finer level is being loaded, even the coarse
 * backbone is drawn from that finer level's arrays, so it must be costed there.
 */
function retainedCostBytes(
  counts: Uint32Array,
  depths: Uint8Array,
  depthAtLeast: number,
  bytesPerVertex: number,
): number {
  let total = 0;
  for (let id = 0; id < counts.length; ++id) {
    if (depths[id] >= depthAtLeast) total += counts[id];
  }
  return total * bytesPerVertex;
}

/**
 * The `(rank, vertexCount)` pairs of every object new at `depth`.
 *
 * Extracted once so the cutoff search below can sweep them repeatedly without
 * re-walking the whole id space each time.
 */
function newAtDepth(
  counts: Uint32Array,
  depths: Uint8Array,
  depth: number,
): { ranks: Float64Array; counts: Uint32Array; totalVertices: number } {
  let n = 0;
  for (let id = 0; id < counts.length; ++id) if (depths[id] === depth) ++n;
  const ranks = new Float64Array(n);
  const out = new Uint32Array(n);
  let i = 0;
  let totalVertices = 0;
  for (let id = 0; id < counts.length; ++id) {
    if (depths[id] !== depth) continue;
    ranks[i] = objectRank(id);
    out[i] = counts[id];
    totalVertices += counts[id];
    ++i;
  }
  return { ranks, counts: out, totalVertices };
}

/** Vertices selected by `rank < cutoff`. */
function verticesUnderCutoff(
  ranks: Float64Array,
  counts: Uint32Array,
  cutoff: number,
): number {
  let total = 0;
  for (let i = 0; i < ranks.length; ++i) {
    if (ranks[i] < cutoff) total += counts[i];
  }
  return total;
}

/** Iterations of the cutoff search; 40 halvings resolve a rank exactly. */
const CUTOFF_SEARCH_STEPS = 40;

/**
 * The largest `cutoff` whose selection fits `budgetVertices`.
 *
 * Found by bisection on the cutoff rather than by scaling the mean, because the
 * ranks of one depth class are not guaranteed to be spread over [0, 1). If a
 * writer happened to choose survivors by something correlated with our hash,
 * a whole depth class can occupy a narrow band of ranks, and a proportional
 * estimate would then admit all of it or none of it. Bisection reads the actual
 * distribution and is correct for any of them.
 */
function cutoffForBudget(
  ranks: Float64Array,
  counts: Uint32Array,
  budgetVertices: number,
): number {
  if (!(budgetVertices > 0) || ranks.length === 0) return 0;
  let lo = 0;
  let hi = 1;
  for (let step = 0; step < CUTOFF_SEARCH_STEPS; ++step) {
    const mid = (lo + hi) / 2;
    if (verticesUnderCutoff(ranks, counts, mid) <= budgetVertices) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/**
 * The largest set of whole objects a byte budget affords.
 *
 * Walks the pyramid from finest to coarsest for the finest level that fits
 * *entirely*; the gap below that rung is then filled by admitting a fraction of
 * the objects one level finer. The result always describes complete objects,
 * never a spatial crop — which is the point: a tract clipped at a chunk boundary
 * has lost the shape that made it worth drawing.
 *
 * `perLevelVertexCounts` is finest-first with 0 for absent, matching
 * {@link objectDepths}.
 */
export function admissionForBudget(
  perLevelVertexCounts: ReadonlyArray<Uint32Array>,
  depths: Uint8Array,
  bytesPerVertex: number,
  budgetBytes: number,
): ObjectAdmission {
  const numLevels = perLevelVertexCounts.length;
  if (numLevels === 0 || !(bytesPerVertex > 0)) return ADMIT_ALL;
  const coarsest = numLevels - 1;
  // A budget of nothing still draws the coarsest level whole rather than an
  // empty screen -- the same "sparse beats blank" rule the pyramid selector
  // follows. `fraction: 0` here would have admitted literally no object, which
  // is a blank view reported as a successful answer.
  if (!(budgetBytes > 0)) return { loadLevel: coarsest, fraction: 1 };

  /**
   * Load `loadLevel`, keeping everything that survives to `fullDepth` plus a
   * share of what is new at `loadLevel`.
   */
  const refine = (loadLevel: number, fullDepth: number): ObjectAdmission => {
    const counts = perLevelVertexCounts[loadLevel];
    const baseCost = retainedCostBytes(
      counts,
      depths,
      fullDepth,
      bytesPerVertex,
    );
    if (baseCost >= budgetBytes && fullDepth <= coarsest) {
      // The backbone alone overruns once re-costed at this finer level, so a
      // partial finer level would be worse than the coarser level drawn whole.
      return { loadLevel: fullDepth, fraction: 1 };
    }
    const {
      ranks,
      counts: newCounts,
      totalVertices,
    } = newAtDepth(counts, depths, loadLevel);
    if (totalVertices === 0) return { loadLevel, fraction: 0 };
    const remainingVertices =
      Math.max(0, budgetBytes - baseCost) / bytesPerVertex;
    if (remainingVertices >= totalVertices) return { loadLevel, fraction: 1 };
    return {
      loadLevel,
      fraction: cutoffForBudget(ranks, newCounts, remainingVertices),
    };
  };

  // Costs fall monotonically towards the coarse end, so the first level that
  // fits walking finest-first is the finest that fits.
  for (let k = 0; k < numLevels; ++k) {
    const full = levelCostBytes(perLevelVertexCounts[k], bytesPerVertex);
    if (full <= budgetBytes) {
      return k === 0 ? ADMIT_ALL : refine(k - 1, k);
    }
  }
  // Not even the coarsest level fits whole; take what of it we can.
  return refine(coarsest, coarsest + 1);
}

/**
 * Whether an object is drawn, given the level whose chunk it came out of.
 *
 * `depth > loadLevel` is the coarse backbone — always kept, so the picture never
 * loses a tract that a coarser level would have shown. `depth === loadLevel` is
 * what this level newly offers, rationed by rank. `depth < loadLevel` cannot
 * occur for an object genuinely present in that level's chunk, and is refused.
 */
export function admitsObject(
  id: number,
  depth: number,
  admission: ObjectAdmission,
): boolean {
  const { loadLevel, fraction } = admission;
  if (depth === OBJECT_DEPTH_ABSENT) return false;
  if (depth > loadLevel) return true;
  if (depth < loadLevel) return false;
  if (fraction >= 1) return true;
  if (!(fraction > 0)) return false;
  return objectRank(id) < fraction;
}

// ---------------------------------------------------------------- object_filter

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
