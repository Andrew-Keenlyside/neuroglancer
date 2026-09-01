/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

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
