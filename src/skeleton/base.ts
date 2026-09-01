/**
 * @license
 * Copyright 2016 Google Inc.
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

import type { DataType } from "#src/util/data_type.js";

export const SKELETON_LAYER_RPC_ID = "skeleton/SkeletonLayer";

export const SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID =
  "skeleton/SpatiallyIndexedSkeletonRenderLayer";
export const SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID =
  "skeleton/SpatiallyIndexedSkeletonRenderLayer.updateSources";
/**
 * On-demand request the tract Export tab makes: evaluate the ticked ROI groups
 * over the currently-resident chunks and return each group's passing object ids.
 * A promise RPC (request/response), unlike the fire-and-forget passing-set
 * recompute -- the tab needs the ids in hand before it can build the export job.
 */
export const SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_ROI_EXPORT_IDS_RPC_ID =
  "skeleton/SpatiallyIndexedSkeletonRenderLayer.roiExportIds";
/**
 * On-demand request the Filter tab makes before it can offer a per-vertex
 * attribute as a filter: the observed range of each named attribute over the
 * same resident chunks the filter itself folds.
 *
 * The values live only in the worker (the frontend holds them as opaque packed
 * texture bytes), and the range has to come from the data because the format
 * declares none. Measuring it over exactly the fold's chunks is also what keeps
 * a slider's ends meaningful: they bound what the filter can currently select.
 */
export const SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_VERTEX_ATTR_STATS_RPC_ID =
  "skeleton/SpatiallyIndexedSkeletonRenderLayer.vertexAttrStats";

/**
 * The observed distribution of one per-vertex attribute over the resident
 * chunks: enough to draw a range control for it and to decide whether it is a
 * flag rather than a measurement.
 */
export interface VertexAttrStats {
  readonly name: string;
  /** How many resident vertices carried a finite value. */
  readonly count: number;
  readonly min: number;
  readonly max: number;
  /** Whether every observed value was a whole number. */
  readonly integral: boolean;
  /** How many distinct values were seen, capped (see the backend's LIMIT). */
  readonly distinct: number;
}

export interface VertexAttributeInfo {
  dataType: DataType;
  numComponents: number;
}

/**
 * Closes a spatially-indexed skeleton chunk key.
 *
 * A key is the grid position and nothing else, but it needs to END somewhere:
 * cell keys are prefix-matched for targeted invalidation
 * (`invalidateCacheKeyPrefixes`), and without a terminator the key of cell
 * `1,2,3` is a prefix of cell `1,2,30`, so editing one cell would invalidate a
 * neighbour it never touched. One constant character makes the match exact.
 *
 * Constant, deliberately: this slot used to hold the normalised pyramid level,
 * which changed on every camera move and orphaned the entire resident set each
 * time it did.
 */
export const SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR = "|";

/**
 * Most cells a whole-volume enumeration will visit before giving up.
 *
 * A guard, not a budget: the enumeration exists for object-focused loading,
 * whose residency is sized so the entire level fits, and a real store's spatial
 * index is a few hundred cells. A grid orders of magnitude larger than that is
 * a store this mode cannot serve, and walking it every priority pass would
 * stall the worker rather than fail visibly.
 */
export const MAX_SPATIAL_SKELETON_VOLUME_CELLS = 65536;

/**
 * Visit EVERY cell of a spatial index, whatever the camera is looking at.
 *
 * The frustum walk (`forEachVisibleVolumetricChunk`) is the right enumeration
 * for geometry whose chunks are independently meaningful — you want what is on
 * screen and nothing more. It is the wrong one for whole objects: a streamline
 * spans the volume, so tying its residency to the view means the parts of it
 * outside the frustum are evicted as the camera moves, and the object that was
 * supposed to be loaded whole is quietly reduced to the piece you happen to be
 * looking at.
 *
 * Returns the number of cells visited, or `-1` if the grid exceeds
 * {@link MAX_SPATIAL_SKELETON_VOLUME_CELLS} (in which case the callback is
 * never invoked, so a caller can fall back rather than half-enumerate).
 */
export function forEachSpatialSkeletonVolumeCell(
  lowerChunkBound: ArrayLike<number>,
  upperChunkBound: ArrayLike<number>,
  callback: (positionInChunks: Float32Array) => void,
): number {
  let total = 1;
  for (let i = 0; i < 3; ++i) {
    const extent = Math.max(0, upperChunkBound[i] - lowerChunkBound[i]);
    if (!Number.isFinite(extent)) return -1;
    total *= extent;
    if (total > MAX_SPATIAL_SKELETON_VOLUME_CELLS) return -1;
  }
  if (total === 0) return 0;
  const position = new Float32Array(3);
  let visited = 0;
  for (let z = lowerChunkBound[2]; z < upperChunkBound[2]; ++z) {
    for (let y = lowerChunkBound[1]; y < upperChunkBound[1]; ++y) {
      for (let x = lowerChunkBound[0]; x < upperChunkBound[0]; ++x) {
        position[0] = x;
        position[1] = y;
        position[2] = z;
        callback(position);
        ++visited;
      }
    }
  }
  return visited;
}
