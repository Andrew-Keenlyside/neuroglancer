/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

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
