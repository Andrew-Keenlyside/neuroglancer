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
 * Reads a parcellation (a discrete segmentation layer) into a dense
 * {@link RoiLabelField} so the tractography ROI filter can, in the worker, ask
 * "what anatomical label sits at this streamline vertex?" for a
 * {@link LabelMaskShape} region.
 *
 * Two things are produced:
 *
 *  - {@link readParcellationLabels} enumerates the layer's label ids, names and
 *    colours from its segment-property map, for a UI checklist of labels the
 *    user can include/exclude.
 *  - {@link buildRoiLabelField} fetches every chunk of a chosen resolution of
 *    the layer's volume and assembles a dense uint32 label grid together with
 *    the model→voxel transform that indexes it. This is the heavy, one-shot
 *    build that happens on the frontend when a label-mask ROI is configured;
 *    the resulting {@link RoiLabelField} is structured-clone-shipped to the
 *    worker, where `makeLabelSampler` (in {@link ./roi.js}) turns it into the
 *    per-vertex reader.
 *
 * The chunk-fetching machinery mirrors `src/python_integration/volume.ts`,
 * which streams a volume layer chunk-by-chunk for the Python integration; the
 * transform derivation reuses the same `render_coordinate_transform.ts`
 * helpers the slice-view renderer uses.
 *
 * ## Frame assumption: tract model space == global x,y,z
 *
 * The tract ROI system assumes the streamline model frame is the viewer's
 * global x,y,z (mm) frame — geometric ROIs, tract vertices and this label field
 * are all expressed there. So a model-space point `(x, y, z)` handed to the
 * sampler is placed into the viewer's global position vector (at the global
 * dimensions named "x"/"y"/"z", falling back to global dims 0/1/2) and then run
 * through the parcellation layer's own global→voxel transform. This holds only
 * while the tractography demo pins the tract frame to the global frame; if a
 * layer introduced a non-identity model→global transform for the tracts, that
 * transform would have to be composed in here.
 *
 * The field's grid axes are the chosen source's chunk dimensions 0, 1, 2 (its
 * spatial dimensions — a single trailing channel dimension, if any, is ignored;
 * parcellations are single-channel). `data` is laid out x-fastest over those
 * three axes, and `modelToVoxel` maps a model-space point straight to voxel
 * coordinates that index `data` (the lower voxel bound is folded in, so voxel 0
 * is the first stored voxel).
 */

import type { RoiLabelField } from "#src/datasource/zarr-vectors/roi.js";
// Type-only: used purely for annotations here, so this edge is erased at runtime
// and does not form an import cycle with index.ts (which imports buildRoiLabelField).
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  getChunkPositionFromCombinedGlobalLocalPositions,
  getChunkTransformParameters,
} from "#src/render_coordinate_transform.js";
import type { SliceViewSingleResolutionSource } from "#src/sliceview/frontend.js";
import type {
  VolumeChunk,
  VolumeChunkSource,
} from "#src/sliceview/volume/frontend.js";
import { SliceViewVolumeRenderLayer } from "#src/sliceview/volume/renderlayer.js";
import { valueOrThrow } from "#src/util/error.js";
import * as matrix from "#src/util/matrix.js";

/** Default cap on the dense grid size, in voxels, when choosing a resolution. */
const DEFAULT_MAX_VOXELS = 40_000_000;

/** Number of chunk fetches allowed in flight at once while filling the grid. */
const FETCH_CONCURRENCY = 8;

/** One parcellation label, for a UI include/exclude checklist. */
export interface ParcellationLabel {
  /** Label id (segment id, assumed < 2^32 for parcellations). */
  readonly id: number;
  /** Display name from the segment-property LUT, or `""` if there is none. */
  readonly name: string;
  /** Packed `R | G<<8 | B<<16` colour, if the LUT declares one for this id. */
  readonly colorPacked?: number;
}

/**
 * Enumerate every label id + name + colour from the layer's segment-property
 * map, index-aligned across the map's `ids`, `label` and `rgb` columns.
 *
 * Returns `[]` when the layer has no inline segment-property map loaded yet (so
 * a UI can render an empty, not-yet-ready checklist rather than throwing).
 */
export function readParcellationLabels(
  layer: SegmentationUserLayer,
): ParcellationLabel[] {
  const pre = layer.displayState.segmentPropertyMap.value;
  const inlineProperties = pre?.segmentPropertyMap.inlineProperties;
  if (pre === undefined || inlineProperties === undefined) return [];
  const { ids } = inlineProperties;
  const labelValues = pre.labels?.values;
  const colorValues = pre.colors?.values;
  const result: ParcellationLabel[] = [];
  for (let i = 0; i < ids.length; ++i) {
    const packed = colorValues?.[i];
    result.push({
      id: Number(ids[i]),
      name: labelValues?.[i] ?? "",
      // `-1` is the "no colour for this id" sentinel in the rgb column.
      ...(packed !== undefined && packed >= 0 ? { colorPacked: packed } : {}),
    });
  }
  return result;
}

/**
 * Maps a model-space point to continuous voxel coordinates by sampling three
 * basis directions rather than picking the affine apart by hand.
 *
 * `globalToVoxel(x, y, z)` maps a model-space point to continuous voxel
 * coordinates in the source's chunk frame (before the lower-bound offset). The
 * returned row-major 4×4 folds in `lo` (the field's lower voxel bound) so that
 * `M * [x, y, z, 1]` yields voxel coordinates that index `data` directly:
 *
 *   column c (c in x,y,z) = basis_c − origin     (the linear part)
 *   translation column    = origin − lo          (origin shifted into the grid)
 *
 * Kept as an exported pure helper so the (fiddly) affine algebra can be unit
 * tested against a mock `globalToVoxel` without standing up a volume source.
 */
export function buildModelToVoxelAffine(
  globalToVoxel: (
    x: number,
    y: number,
    z: number,
  ) => readonly [number, number, number],
  lo: readonly [number, number, number],
): Float32Array {
  const o = globalToVoxel(0, 0, 0);
  const ex = globalToVoxel(1, 0, 0);
  const ey = globalToVoxel(0, 1, 0);
  const ez = globalToVoxel(0, 0, 1);
  const m = new Float32Array(16);
  for (let r = 0; r < 3; ++r) {
    m[r * 4 + 0] = ex[r] - o[r];
    m[r * 4 + 1] = ey[r] - o[r];
    m[r * 4 + 2] = ez[r] - o[r];
    m[r * 4 + 3] = o[r] - lo[r];
  }
  m[15] = 1;
  return m;
}

/**
 * Fetch every chunk of a chosen resolution of `layer`'s volume and assemble a
 * dense {@link RoiLabelField}.
 *
 * `globalDimensionNames` are the viewer's global coordinate-space dimension
 * names (e.g. `["x", "y", "z"]`), used to place a model-space `(x, y, z)` into
 * the global position vector before it is mapped to voxels.
 *
 * Returns `undefined` when the layer is not ready, or does not have exactly one
 * {@link SliceViewVolumeRenderLayer}, or has no usable (rank ≥ 3, non-empty)
 * source — the caller then leaves the label field unset and the label-mask ROI
 * momentarily matches nothing rather than matching wrongly.
 */
export async function buildRoiLabelField(
  layer: SegmentationUserLayer,
  globalDimensionNames: readonly string[],
  options?: { signal?: AbortSignal; maxVoxels?: number },
): Promise<RoiLabelField | undefined> {
  const maxVoxels = options?.maxVoxels ?? DEFAULT_MAX_VOXELS;
  const signal = options?.signal;

  // `isReady` is a getter on UserLayer, not a method.
  if (!layer.isReady) return undefined;

  const renderLayers = layer.renderLayers.filter(
    (renderLayer) => renderLayer instanceof SliceViewVolumeRenderLayer,
  ) as SliceViewVolumeRenderLayer[];
  if (renderLayers.length !== 1) return undefined;
  const renderLayer = renderLayers[0];

  const transform = valueOrThrow(renderLayer.transform.value);
  const { multiscaleSource } = renderLayer;

  const sources = multiscaleSource.getSources({
    discreteValues: true,
    displayRank: multiscaleSource.rank,
    modelChannelDimensionIndices: [],
    multiscaleToViewTransform: matrix.createIdentity(
      Float32Array,
      multiscaleSource.rank,
    ),
  });

  // `getSources` returns `[alternative orientation][scale]`; flatten and choose
  // by voxel count so the exact nesting order does not matter (alternatives at
  // one scale carry identical voxel counts — same data, different chunk layout).
  const flat: SliceViewSingleResolutionSource<VolumeChunkSource>[] = [];
  for (const alternatives of sources) {
    for (const source of alternatives) flat.push(source);
  }
  if (flat.length === 0) return undefined;

  const spatialVoxelCount = (
    source: SliceViewSingleResolutionSource<VolumeChunkSource>,
  ): number => {
    const { spec } = source.chunkSource;
    if (spec.rank < 3) return Number.POSITIVE_INFINITY;
    let n = 1;
    for (let d = 0; d < 3; ++d) {
      n *= Math.max(
        0,
        Math.ceil(spec.upperVoxelBound[d]) - Math.floor(spec.lowerVoxelBound[d]),
      );
    }
    return n;
  };

  // Prefer the finest scale (largest voxel count) that fits the budget; if none
  // fits, fall back to the coarsest (smallest voxel count).
  let single: SliceViewSingleResolutionSource<VolumeChunkSource> | undefined;
  let chosenCount = -1;
  for (const source of flat) {
    const n = spatialVoxelCount(source);
    if (n <= maxVoxels && n > chosenCount) {
      single = source;
      chosenCount = n;
    }
  }
  if (single === undefined) {
    let smallest = Number.POSITIVE_INFINITY;
    for (const source of flat) {
      const n = spatialVoxelCount(source);
      if (n < smallest) {
        smallest = n;
        single = source;
      }
    }
  }
  if (single === undefined) return undefined;

  const { spec } = single.chunkSource;
  const { rank } = spec;
  if (rank < 3) return undefined;

  // --- Build the model→voxel mapping (basis sampling). ---------------------
  const chunkTransform = getChunkTransformParameters(
    transform,
    single.chunkToMultiscaleTransform,
  );
  const layerRank = chunkTransform.layerRank;
  const combined = chunkTransform.combinedGlobalLocalToChunkTransform;
  const combinedRank = chunkTransform.combinedGlobalLocalRank;
  // Split the combined [global, local] input space so the translation column of
  // `combined` lands where `getChunkPositionFromCombinedGlobalLocalPositions`
  // expects it: this requires globalRank + localRank === combinedRank.
  const globalRank = Math.min(globalDimensionNames.length, combinedRank);
  const localRank = combinedRank - globalRank;

  const globalIndexFor = (name: string, fallback: number): number => {
    const idx = globalDimensionNames.indexOf(name);
    if (idx >= 0 && idx < globalRank) return idx;
    return globalRank > 0 ? Math.min(fallback, globalRank - 1) : 0;
  };
  const xi = globalIndexFor("x", 0);
  const yi = globalIndexFor("y", 1);
  const zi = globalIndexFor("z", 2);

  const globalPos = new Float32Array(globalRank);
  const localPos = new Float32Array(localRank);
  const chunkPos = new Float32Array(layerRank);
  const globalToVoxel = (
    x: number,
    y: number,
    z: number,
  ): readonly [number, number, number] => {
    globalPos.fill(0);
    if (globalRank > 0) {
      globalPos[xi] = x;
      globalPos[yi] = y;
      globalPos[zi] = z;
    }
    getChunkPositionFromCombinedGlobalLocalPositions(
      chunkPos,
      globalPos,
      localPos,
      layerRank,
      combined,
    );
    return [chunkPos[0], chunkPos[1], chunkPos[2]];
  };

  const lo: [number, number, number] = [
    Math.floor(spec.lowerVoxelBound[0]),
    Math.floor(spec.lowerVoxelBound[1]),
    Math.floor(spec.lowerVoxelBound[2]),
  ];
  const dims: [number, number, number] = [
    Math.ceil(spec.upperVoxelBound[0]) - lo[0],
    Math.ceil(spec.upperVoxelBound[1]) - lo[1],
    Math.ceil(spec.upperVoxelBound[2]) - lo[2],
  ];
  if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) return undefined;

  const modelToVoxel = buildModelToVoxelAffine(globalToVoxel, lo);

  // --- Fetch every chunk and fill the dense grid. --------------------------
  const data = new Uint32Array(dims[0] * dims[1] * dims[2]);
  const chunkDataSize = spec.chunkDataSize;
  const fillValue = Number(spec.fillValue) >>> 0;

  // Valid chunk grid positions lie in [lowerChunkBound, upperChunkBound) per
  // spatial dim; non-spatial (e.g. channel) dims are pinned to their single
  // valid chunk index. The voxel origin of chunk grid position g along a
  // spatial dim d is g * chunkDataSize[d] (the frontend's baseVoxelOffset does
  // not enter this mapping — see backend.ts:computeChunkBounds).
  const gridBase = new Float32Array(rank);
  for (let d = 0; d < rank; ++d) {
    gridBase[d] = Math.floor(spec.lowerChunkBound[d]);
  }
  const cLo = [
    Math.floor(spec.lowerChunkBound[0]),
    Math.floor(spec.lowerChunkBound[1]),
    Math.floor(spec.lowerChunkBound[2]),
  ];
  const cHi = [
    Math.ceil(spec.upperChunkBound[0]),
    Math.ceil(spec.upperChunkBound[1]),
    Math.ceil(spec.upperChunkBound[2]),
  ];

  const chunkGridPositions: Float32Array[] = [];
  for (let gz = cLo[2]; gz < cHi[2]; ++gz) {
    for (let gy = cLo[1]; gy < cHi[1]; ++gy) {
      for (let gx = cLo[0]; gx < cHi[0]; ++gx) {
        const g = Float32Array.from(gridBase);
        g[0] = gx;
        g[1] = gy;
        g[2] = gz;
        chunkGridPositions.push(g);
      }
    }
  }

  const fillChunk = (chunkGridPosition: Float32Array): Promise<void> =>
    single!.chunkSource.fetchChunk(
      chunkGridPosition,
      (chunk) => {
        const volumeChunk = chunk as VolumeChunk;
        const chunkData = (volumeChunk as unknown as { data: unknown }).data;
        // A null-data (fill) chunk with a zero fill value contributes nothing;
        // the grid is already zero-initialised.
        if (chunkData === null && fillValue === 0) return;
        const cds = volumeChunk.chunkDataSize;
        const ox = chunkGridPosition[0] * chunkDataSize[0];
        const oy = chunkGridPosition[1] * chunkDataSize[1];
        const oz = chunkGridPosition[2] * chunkDataSize[2];
        // Reused per-voxel position (spatial dims in 0..2; any extra dims stay
        // 0 -> channel 0). `getValueAt` is layout-safe across chunk formats.
        const dataPosition = new Uint32Array(rank);
        for (let iz = 0; iz < cds[2]; ++iz) {
          const vz = oz + iz - lo[2];
          if (vz < 0 || vz >= dims[2]) continue;
          dataPosition[2] = iz;
          for (let iy = 0; iy < cds[1]; ++iy) {
            const vy = oy + iy - lo[1];
            if (vy < 0 || vy >= dims[1]) continue;
            dataPosition[1] = iy;
            const rowBase = dims[0] * (vy + dims[1] * vz);
            for (let ix = 0; ix < cds[0]; ++ix) {
              const vx = ox + ix - lo[0];
              if (vx < 0 || vx >= dims[0]) continue;
              dataPosition[0] = ix;
              const value =
                chunkData === null
                  ? fillValue
                  : Number(volumeChunk.getValueAt(dataPosition));
              data[vx + rowBase] = value;
            }
          }
        }
      },
      { signal },
    );

  for (let i = 0; i < chunkGridPositions.length; i += FETCH_CONCURRENCY) {
    if (signal?.aborted) {
      throw signal.reason instanceof Error
        ? signal.reason
        : new Error("buildRoiLabelField aborted");
    }
    await Promise.all(
      chunkGridPositions.slice(i, i + FETCH_CONCURRENCY).map(fillChunk),
    );
  }

  return { data, dims, modelToVoxel };
}
