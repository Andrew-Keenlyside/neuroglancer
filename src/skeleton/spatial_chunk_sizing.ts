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

import type {
  SpatialSkeletonBounds,
  SpatialSkeletonVector,
} from "#src/skeleton/api.js";

const DEFAULT_SPATIALLY_INDEXED_SKELETON_MAX_CHUNKS = 64;
const DEFAULT_SPATIALLY_INDEXED_SKELETON_MIN_CHUNK_SIZE = 1;

export type SpatiallyIndexedSkeletonChunkSize = number[];
export type SpatialSkeletonGridSize = { x: number; y: number; z: number };
export type SpatialSkeletonGridLevel = {
  size: SpatialSkeletonGridSize;
  lod: number;
  /**
   * How many objects this level holds, when the source can say. For a
   * tractogram this is the streamline count, which is what "how big is this
   * level" means to a user -- a pyramid that drops from ~503k tracts to ~50
   * is describing sparsity, not chunk geometry.
   */
  objectCount?: number;
};

export interface DefaultSpatiallyIndexedSkeletonChunkSizeOptions {
  maxChunks?: number;
  minChunkSize?: number;
}

export function getSpatialSkeletonGridSpacing(size: SpatialSkeletonGridSize) {
  return Math.min(size.x, size.y, size.z);
}

export function sortSpatialSkeletonGridSizes(
  gridSizes: readonly SpatialSkeletonGridSize[],
): SpatialSkeletonGridSize[] {
  return [...gridSizes].sort(
    (a, b) =>
      getSpatialSkeletonGridSpacing(b) - getSpatialSkeletonGridSpacing(a),
  );
}

/**
 * Build the level list, **preserving the caller's order**: coarsest first.
 *
 * `lod` is nothing but the normalised position in this list, so the order is
 * the whole of the level structure — `size` only supplies a physical scale for
 * the render-scale widget and for matching a resolution target.
 *
 * The order is therefore the caller's to declare, and it deliberately is not
 * re-derived here by sorting on `size`. A datasource already knows its
 * pyramid: for zarr-vectors it is the `multiscales` directory order, and the
 * `gridIndex` each level carries is assigned straight from that order. Sorting
 * by spacing silently assumed the two would agree, which holds only while
 * spacing is what separates the levels. An object-sparsity pyramid keeps one
 * chunk_shape at every level and drops whole objects instead: every level then
 * reports the same spacing, the sort degenerates to a no-op, and the resulting
 * positions disagree with the `gridIndex` values — so the wrong level loads.
 *
 * Callers that genuinely rank by spacing can pass
 * `sortSpatialSkeletonGridSizes(sizes)` and get the previous behaviour.
 */
export function buildSpatialSkeletonGridLevels(
  gridSizes: readonly SpatialSkeletonGridSize[],
  objectCounts?: readonly (number | undefined)[],
): SpatialSkeletonGridLevel[] {
  if (gridSizes.length === 0) return [];
  const lastIndex = gridSizes.length - 1;
  return gridSizes.map((size, index) => {
    const objectCount = objectCounts?.[index];
    return {
      size,
      lod: lastIndex === 0 ? 0 : index / lastIndex,
      ...(Number.isFinite(objectCount) && (objectCount as number) > 0
        ? { objectCount: objectCount as number }
        : {}),
    };
  });
}

/**
 * Index of the finest level a memory budget can afford: the last level whose
 * estimated footprint fits, given `costsBytes` parallel to the level list and
 * therefore ascending (coarsest first).
 *
 * This is the "load as much as fits" rule, and it is deliberately not the same
 * question the camera-driven resolution target answers. That target asks how
 * much detail the screen can show, which is the right question for a pyramid
 * whose levels differ in *resolution*. Where levels instead differ in *how
 * many complete objects* they hold, detail-per-pixel says nothing about
 * whether the level will fit: on a whole-brain tractogram the camera target
 * lands on the finest level and asks for ~10^8 vertices, which the budget
 * cannot hold and the renderer cannot survive.
 *
 * Returns 0 -- the coarsest level -- when nothing fits, since showing the
 * sparsest available data beats showing none. A caller with no budget
 * information should not call this at all.
 */
/**
 * Where a spatially-indexed pyramid's spare memory goes.
 *
 * `LOCAL` is the standard neuroglancer answer: spend it near the camera, so
 * detail tracks where the user is looking. Right when a level's chunks are
 * independently meaningful — an image tile, a mesh fragment — because a chunk
 * that arrives is a finished picture of its own cell.
 *
 * `OBJECT` is the answer a tractogram needs. Its objects are long: one
 * streamline crosses many cells, so a spatially-local subset of chunks yields
 * streamlines chopped off at the boundary of whatever did not fit — the shape
 * of the fibre, which is the whole point, is exactly what gets lost. This mode
 * spends the same memory on WHOLE objects instead, admitting a chosen subset of
 * them across the entire volume; see `zarr-vectors/object_admission.ts`.
 */
export enum SpatialSkeletonDetailFocus {
  LOCAL = 0,
  OBJECT = 1,
}

/**
 * The grid spacing a per-cell memory budget can sustain, or `undefined` when it
 * cannot be determined.
 *
 * This is what LOCAL detail focus should aim at, and it is deliberately NOT a
 * screen-resolution quantity. Detail-per-pixel self-limits for an image pyramid
 * — one texel per pixel and you are done — but the levels of an object-sparsity
 * pyramid differ in how many whole objects they hold, and more objects are
 * always more useful. A pixel-derived target therefore asks for the finest level
 * at every zoom, and the level stops responding to the camera entirely.
 *
 * What does vary usefully with zoom is **how many cells are in view**: a
 * whole-volume view must spread the budget over every cell and can afford only a
 * sparse level in each, while a view of one region divides the same budget among
 * a handful and can afford the finest. So the budget is divided by the visible
 * cell count and the finest level whose per-cell cost fits is chosen.
 *
 * Both `levelSpacingsMeters` and `perCellCostBytes` are **coarsest-first**, so a
 * larger index is finer. Returns the coarsest level's spacing when nothing fits
 * — sparse beats blank — and `undefined` when the inputs cannot support an
 * answer, which leaves the caller on its previous behaviour.
 */
export function targetSpacingForCellBudget(
  levelSpacingsMeters: readonly number[],
  perCellCostBytes: readonly number[],
  visibleCells: number,
  budgetBytes: number,
): number | undefined {
  const n = levelSpacingsMeters.length;
  if (
    n === 0 ||
    perCellCostBytes.length !== n ||
    !Number.isFinite(budgetBytes) ||
    budgetBytes <= 0 ||
    !Number.isFinite(visibleCells) ||
    visibleCells <= 0
  ) {
    return undefined;
  }
  const perCellBudget = budgetBytes / visibleCells;
  let best = -1;
  for (let k = 0; k < n; ++k) {
    const cost = perCellCostBytes[k];
    // An unknown cost is not evidence that the level fits.
    if (!Number.isFinite(cost) || cost <= 0) continue;
    if (cost <= perCellBudget) best = k;
  }
  if (best >= 0) return levelSpacingsMeters[best];
  // Nothing fits; fall back to the coarsest level that has a cost at all.
  for (let k = 0; k < n; ++k) {
    if (Number.isFinite(perCellCostBytes[k]) && perCellCostBytes[k] > 0) {
      return levelSpacingsMeters[k];
    }
  }
  return undefined;
}

export function selectSpatialSkeletonGridLevelByBudget(
  costsBytes: readonly number[],
  budgetBytes: number,
): number {
  if (costsBytes.length === 0) return 0;
  let best = 0;
  for (let i = 0; i < costsBytes.length; ++i) {
    const cost = costsBytes[i];
    // An unknown cost is not evidence that the level fits.
    if (!Number.isFinite(cost)) continue;
    if (cost <= budgetBytes) best = i;
  }
  return best;
}

function validateFiniteOptions(
  options: DefaultSpatiallyIndexedSkeletonChunkSizeOptions,
) {
  if (
    options.minChunkSize !== undefined &&
    !Number.isFinite(options.minChunkSize)
  ) {
    throw new Error("Spatially indexed skeleton minChunkSize must be finite.");
  }
  if (options.maxChunks !== undefined && !Number.isFinite(options.maxChunks)) {
    throw new Error("Spatially indexed skeleton maxChunks must be finite.");
  }
}

function validateFiniteVector(vector: SpatialSkeletonVector, label: string) {
  for (let i = 0; i < vector.length; ++i) {
    const value = Number(vector[i]);
    if (!Number.isFinite(value)) {
      throw new Error(
        `Spatially indexed skeleton bounds must be finite, but ${label}[${i}] is ${value}.`,
      );
    }
  }
}

function validateFiniteBounds(bounds: SpatialSkeletonBounds) {
  if (bounds.lowerBounds.length !== bounds.upperBounds.length) {
    throw new Error(
      "Spatially indexed skeleton lower and upper bounds must have matching ranks.",
    );
  }
  if (bounds.lowerBounds.length === 0) {
    throw new Error("Spatially indexed skeleton bounds must have rank > 0.");
  }
  validateFiniteVector(bounds.lowerBounds, "lowerBounds");
  validateFiniteVector(bounds.upperBounds, "upperBounds");
}

function getChunkCoverageForChunkSize(
  extents: readonly number[],
  chunkSize: number,
) {
  return extents.reduce((product, extent) => {
    const axisChunks = extent <= 0 ? 1 : Math.ceil(extent / chunkSize);
    return product * axisChunks;
  }, 1);
}

export function getDefaultSpatiallyIndexedSkeletonChunkSize(
  bounds: SpatialSkeletonBounds,
  options: DefaultSpatiallyIndexedSkeletonChunkSizeOptions = {},
): SpatiallyIndexedSkeletonChunkSize {
  validateFiniteOptions(options);
  validateFiniteBounds(bounds);
  const minChunkSize = Math.max(
    DEFAULT_SPATIALLY_INDEXED_SKELETON_MIN_CHUNK_SIZE,
    Math.ceil(
      options.minChunkSize ?? DEFAULT_SPATIALLY_INDEXED_SKELETON_MIN_CHUNK_SIZE,
    ),
  );
  const maxChunks = Math.max(
    1,
    Math.floor(
      options.maxChunks ?? DEFAULT_SPATIALLY_INDEXED_SKELETON_MAX_CHUNKS,
    ),
  );
  const extents = Array.from(bounds.lowerBounds, (lowerBound, index) =>
    Math.max(0, Number(bounds.upperBounds[index]) - Number(lowerBound)),
  );
  const maxExtent = Math.max(...extents);

  if (!(maxExtent > 0)) {
    return extents.map(() => minChunkSize);
  }

  // Choose the smallest isotropic chunk size that keeps the full bounding box
  // coverage within the requested chunk budget.
  let low = minChunkSize;
  let high = Math.max(minChunkSize, Math.ceil(maxExtent));
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (getChunkCoverageForChunkSize(extents, mid) <= maxChunks) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }

  return extents.map(() => low);
}
