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

/**
 * Pure level-selection helpers for object-keyed multiscale skeleton sources
 * (`MultiscaleSkeletonSource`/`MultiscaleSkeletonRenderLayerBackend`,
 * `src/skeleton/{backend,frontend}.ts`). Unlike the spatially-indexed
 * pass-1 pyramid, an object-keyed source picks ONE level for a whole
 * selected object (not per spatial region), so these operate on a flat
 * per-object level list, indexed `0` (finest) .. `N-1` (coarsest), rather
 * than a spatial grid.
 */

import { quantizeSpacingForArbitration } from "#src/skeleton/screen_size.js";

/**
 * Picks the level whose static chunk spacing is closest to
 * `desiredSpacingRaw` (after quantizing to the nearest quarter-octave, the
 * same anti-thrashing step pass-1's grid-anchor arbitration uses — see
 * `quantizeSpacingForArbitration`). `levelSpacings` is finest-first (index
 * `0` = level 0). Ties favor the finer (lower-index) level.
 */
export function pickTargetLevelByScreenSize(
  levelSpacings: ArrayLike<number>,
  desiredSpacingRaw: number,
): number {
  const desiredSpacing = quantizeSpacingForArbitration(desiredSpacingRaw);
  let bestIndex = 0;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (let i = 0; i < levelSpacings.length; ++i) {
    const delta = Math.abs(levelSpacings[i] - desiredSpacing);
    if (delta < bestDelta) {
      bestDelta = delta;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Walks from `target` toward level 0 (finer) until a level where the
 * object is actually present is found. Deliberately one-directional:
 * coarsening only ever *drops* objects via sparsity thresholds, so
 * absence at `target` guarantees absence at every level coarser than
 * `target` too — there is never a valid reason to search coarser-than
 * -`target` for *presence*. (This is why
 * `selectSpatiallyIndexedSkeletonEntriesByGridWithFallback`'s bidirectional
 * fallback, used by the spatial pass-1 path, is not reused here — its
 * coarser-fallback branch would be semantically wrong for this question.)
 *
 * Returns `undefined` only if the object isn't present at level 0 either
 * (i.e. `presentLevels` has no `true` entry at or below `target`) — for a
 * store where object IDs are preserved identically across levels (the
 * common streamline/skeleton case), level 0 is always present, so this
 * always terminates with a value in practice.
 */
export function pickFinestPresentLevelAtOrBelow(
  presentLevels: ArrayLike<boolean>,
  target: number,
): number | undefined {
  const clampedTarget = Math.min(
    Math.max(target, 0),
    presentLevels.length - 1,
  );
  for (let level = clampedTarget; level >= 0; --level) {
    if (presentLevels[level]) return level;
  }
  return undefined;
}

/**
 * Draw-time "which resident level do I show right now" rule — the
 * no-blink mechanism. Searches from `targetActualLevel` (the finest level
 * ever fetched, per `pickFinestPresentLevelAtOrBelow`) coarser-ward
 * through present levels only, returning the first one `isReady` reports
 * as GPU-resident. Never returns a level finer than `targetActualLevel`
 * (finer levels are never requested in the first place). Mirrors
 * precomputed multiscale meshes' `getMultiscaleChunksToDraw`'s "fall back
 * to the nearest ready ancestor," adapted from an octree to a flat level
 * list — call this fresh every frame purely from current chunk state
 * (never cache the result across frames), so a level transitioning to
 * GPU-resident between frames is picked up immediately with no explicit
 * unload/swap step.
 */
export function pickReadyLevelToDraw(
  presentLevels: ArrayLike<boolean>,
  targetActualLevel: number,
  isReady: (level: number) => boolean,
): number | undefined {
  for (let level = targetActualLevel; level < presentLevels.length; ++level) {
    if (presentLevels[level] && isReady(level)) return level;
  }
  return undefined;
}

/**
 * Picks the target level from a *real-world* resolution target — the
 * object-keyed counterpart of pass-1's meters-based
 * `skeletonGridResolutionTarget3d` arbitration, shared with it via the
 * same control ("Resolution (skeleton grid 3D)") rather than the generic
 * mesh `renderScaleTarget` quality dial. Deliberately has no dependency
 * on camera projection/position: an object-keyed source has no natural
 * "screen footprint" the way a mesh fragment or spatial chunk does (it's
 * a single object, not a spatially-partitioned region), and computing one
 * from a single representative point (e.g. a bounding-box centroid) is
 * fragile — it breaks down (returns `Infinity`/`NaN`) whenever that point
 * crosses the near clip plane, which happens easily while zooming in on
 * a single selected object. Working purely in real-world units sidesteps
 * that entirely, and — since pass-1's "auto" mode already keeps
 * `skeletonGridResolutionTarget3d` in sync with the current camera zoom
 * when it's active — still tracks zoom in the common case where pass-1
 * and pass-2 are both enabled.
 */
export function pickTargetLevelByRealWorldSpacing(
  levelSpacings: ArrayLike<number>,
  metersPerUnit: number,
  targetSpacingMeters: number,
): number {
  const levelSpacingsMeters = Array.from(
    levelSpacings,
    (s) => s * metersPerUnit,
  );
  return pickTargetLevelByScreenSize(levelSpacingsMeters, targetSpacingMeters);
}
