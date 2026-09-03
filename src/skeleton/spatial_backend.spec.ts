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

import { describe, expect, it } from "vitest";

import {
  getSpatiallyIndexedSkeletonChunkPriority,
  getSpatiallyIndexedSkeletonRenderPriority,
  selectRoiBackgroundScales,
  SpatiallyIndexedSkeletonChunkRequestOwner,
} from "#src/skeleton/spatial_backend.js";
import {
  BASE_PRIORITY,
  SCALE_PRIORITY_MULTIPLIER,
} from "#src/sliceview/backend.js";

describe("skeleton/spatial_backend chunk priority", () => {
  it("uses the standard chunk-origin distance rule for spatial skeleton chunks", () => {
    expect(
      getSpatiallyIndexedSkeletonChunkPriority(
        Float32Array.of(3, 4, 0),
        Float32Array.of(2, 5, 1),
        Float32Array.of(1, 0, 0),
      ),
    ).toBeCloseTo(-Math.sqrt(17));
  });

  it("prioritizes chunks nearer the view center ahead of farther chunks", () => {
    const localCenter = Float32Array.of(10, 20, 30);
    const chunkSize = Float32Array.of(4, 4, 8);
    const nearChunk = Float32Array.of(2, 5, 4);
    const farChunk = Float32Array.of(5, 1, 0);

    expect(
      getSpatiallyIndexedSkeletonChunkPriority(
        localCenter,
        chunkSize,
        nearChunk,
      ),
    ).toBeGreaterThan(
      getSpatiallyIndexedSkeletonChunkPriority(
        localCenter,
        chunkSize,
        farChunk,
      ),
    );
  });

  it("matches equivalent volume-rendering chunks' priority exactly (no boost)", () => {
    // Spatially-indexed skeleton chunks must compete for the shared chunk
    // memory budget on equal footing with other VISIBLE-tier sources
    // (volume rendering, annotations, meshes) — differentiated only by
    // genuine distance-to-view relevance, never by an unconditional boost.
    // A boost here previously let every skeleton chunk, however
    // irrelevant, outrank and evict every other layer's required chunks.
    const basePriority = BASE_PRIORITY;
    const scaleIndex = 2;
    const localCenter = Float32Array.of(10, 20, 30);
    const chunkSize = Float32Array.of(4, 4, 8);
    const positionInChunks = Float32Array.of(2, 5, 4);
    const distancePriority = getSpatiallyIndexedSkeletonChunkPriority(
      localCenter,
      chunkSize,
      positionInChunks,
    );
    const equivalentVolumeRenderingPriority =
      basePriority + SCALE_PRIORITY_MULTIPLIER * scaleIndex + distancePriority;

    for (const owner of [
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_2D,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D,
    ]) {
      expect(owner).not.toBe(SpatiallyIndexedSkeletonChunkRequestOwner.NONE);
      const skeletonPriority = getSpatiallyIndexedSkeletonRenderPriority(
        basePriority,
        scaleIndex,
        localCenter,
        chunkSize,
        positionInChunks,
      );

      expect(skeletonPriority).toBeCloseTo(equivalentVolumeRenderingPriority);
    }
  });

  it("confines the ROI filter's chunk requests to the background level", () => {
    // The dissection is a selection WITHIN what the background already draws.
    // Requesting the ROI regions at every level pinned finer levels resident,
    // which then won the evaluation-level choice, so simply having a filter
    // made the layer fetch and fold geometry the view never drew.
    const scales = [0, 1, 2, 3].map(
      (gridIndex) => ({ source: { parameters: { gridIndex } } }) as any,
    );
    expect(
      selectRoiBackgroundScales(scales, 2).map(
        (s: any) => s.source.parameters.gridIndex,
      ),
    ).toEqual([2]);
  });

  it("picks exactly one background level when the selected one is absent", () => {
    // Same fallback ordering the draw path uses, so both sides land on the
    // same substitute rather than the filter and the view disagreeing.
    const scales = [1, 3].map(
      (gridIndex) => ({ source: { parameters: { gridIndex } } }) as any,
    );
    expect(selectRoiBackgroundScales(scales, 2)).toHaveLength(1);
  });

  it("keeps every scale for a source that publishes no grid indices", () => {
    // No level selection to speak of there, so nothing to narrow to.
    const scales = [{ source: {} }, { source: {} }] as any[];
    expect(selectRoiBackgroundScales(scales, 0)).toHaveLength(2);
    expect(selectRoiBackgroundScales([], 0)).toEqual([]);
  });

  it("ranks every fill-level chunk below every background chunk", () => {
    // The whole partial-load scheme rests on this: the fill level is requested
    // alongside the background level and must only ever get what the background
    // leaves. `scaleIndex` counts from the finest level, so the fill level --
    // one FINER than the background -- sits a whole SCALE_PRIORITY_MULTIPLIER
    // below it, which no distance term can close. Were the two comparable, a
    // near fill chunk would evict a far background chunk and punch a hole in
    // the level that is supposed to cover everything.
    const basePriority = BASE_PRIORITY;
    const backgroundScaleIndex = 2;
    const fillScaleIndex = backgroundScaleIndex - 1;
    const localCenter = Float32Array.of(10, 20, 30);
    const chunkSize = Float32Array.of(4, 4, 8);
    const nearestFill = Float32Array.of(2, 5, 4);
    const farthestBackground = Float32Array.of(1000, 1000, 1000);

    expect(
      getSpatiallyIndexedSkeletonRenderPriority(
        basePriority,
        fillScaleIndex,
        localCenter,
        chunkSize,
        nearestFill,
      ),
    ).toBeLessThan(
      getSpatiallyIndexedSkeletonRenderPriority(
        basePriority,
        backgroundScaleIndex,
        localCenter,
        chunkSize,
        farthestBackground,
      ),
    );
  });

  it("keeps spatial skeleton chunks ordered by distance", () => {
    const basePriority = BASE_PRIORITY;
    const scaleIndex = 1;
    const localCenter = Float32Array.of(10, 20, 30);
    const chunkSize = Float32Array.of(4, 4, 8);
    const nearChunk = Float32Array.of(2, 5, 4);
    const farChunk = Float32Array.of(5, 1, 0);

    expect(
      getSpatiallyIndexedSkeletonRenderPriority(
        basePriority,
        scaleIndex,
        localCenter,
        chunkSize,
        nearChunk,
      ),
    ).toBeGreaterThan(
      getSpatiallyIndexedSkeletonRenderPriority(
        basePriority,
        scaleIndex,
        localCenter,
        chunkSize,
        farChunk,
      ),
    );
  });
});
