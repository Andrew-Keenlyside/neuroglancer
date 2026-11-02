/**
 * @license
 * Copyright 2019 Google Inc.
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

import { TrackableValue } from "#src/trackable_value.js";
import { makeVerifyNumberInInterval } from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";
import { VisibilityPriorityAggregator } from "#src/visibility_priority/frontend.js";

export const numRenderScaleHistogramBins = 40;
export const renderScaleHistogramBinSize = 0.5;
export const renderScaleHistogramOrigin = -4;

export function getRenderScaleHistogramOffset(
  renderScale: number,
  origin: number = renderScaleHistogramOrigin,
  binSize: number = renderScaleHistogramBinSize,
): number {
  return (Math.log2(renderScale) - origin) / binSize;
}

export function getRenderScaleFromHistogramOffset(
  offset: number,
  origin: number = renderScaleHistogramOrigin,
  binSize: number = renderScaleHistogramBinSize,
): number {
  return 2 ** (offset * binSize + origin);
}

export function trackableRenderScaleTarget(
  initialValue: number,
  scaleOrigin: number = 2 ** renderScaleHistogramOrigin,
  scaleMax?: number,
) {
  if (scaleMax === undefined) {
    scaleMax =
      2 **
        Math.round(
          renderScaleHistogramBinSize * numRenderScaleHistogramBins +
            renderScaleHistogramOrigin,
        ) -
      1;
  }
  const verifyNumberInInterval = makeVerifyNumberInInterval(
    scaleOrigin,
    scaleMax,
  );
  return new TrackableValue<number>(initialValue, verifyNumberInInterval);
}

export class RenderScaleHistogram {
  visibility = new VisibilityPriorityAggregator();
  changed = new NullarySignal();

  constructor(
    readonly logScaleOrigin: number = renderScaleHistogramOrigin,
    readonly numBins: number = numRenderScaleHistogramBins,
    readonly binSize: number = renderScaleHistogramBinSize,
  ) {
    this.value = new Uint32Array(numBins * this.numHistogramRows * 2);
  }

  /**
   * Frame number corresponding to the current histogram.
   */
  frameNumber = -1;

  /**
   * Maps from spatial scale (nanometers) to histogram row index in the range
   * `[0, spatialScales.size)`.
   */
  spatialScales = new Map<number, number>();

  /**
   * Current number of rows allocated for the histogram.
   */
  numHistogramRows = 1;

  /**
   * Histogram value array, stores both present and not-present counts.
   */
  value: Uint32Array;

  /**
   * Number of chunks that are indication only (not present in the data).
   */
  fakeChunkCount = 0;

  /**
   * Every level (spatial scale) declared by the layer or seen in a frame,
   * kept across frames, so a widget can lay out all of them, not only those
   * drawn now.
   */
  levels = new Set<number>();

  /**
   * The size of a screen pixel in spatial-scale units (spatial scale over
   * render scale), from the last finite entry: where the target render
   * scale falls among the levels.
   */
  pixelSize: number | undefined;

  /** Sets the layer's levels (spatial scales), for example every scale of a multiscale volume. */
  declareLevels(spatialScales: Iterable<number>) {
    const levels = new Set(
      Array.from(spatialScales).filter((s) => Number.isFinite(s)),
    );
    if (
      levels.size === this.levels.size &&
      Array.from(levels).every((s) => this.levels.has(s))
    ) {
      return;
    }
    this.levels = levels;
    this.changed.dispatch();
  }

  begin(frameNumber: number) {
    if (frameNumber !== this.frameNumber) {
      this.value.fill(0);
      this.frameNumber = frameNumber;
      this.spatialScales.clear();
      this.fakeChunkCount = 0;
      this.changed.dispatch();
    }
  }

  /**
   * Adds a count to the histogram.
   *
   * @param spatialScale Spatial resolution of data in nanometers.
   * @param renderScale Rendered scale of data in screen pixels.
   * @param presentCount Number of present chunks.
   * @param notPresentCount Number of desired but not-present chunks.
   * @param renderOnly If true, indicates that the added bar is for display only, and is not linked
   *     to actual chunk loading stats. Defaults to false.
   */
  add(
    spatialScale: number,
    renderScale: number,
    presentCount: number,
    notPresentCount: number,
    renderOnly = false,
  ) {
    let { spatialScales, numHistogramRows, value } = this;
    if (
      !renderOnly &&
      Number.isFinite(spatialScale) &&
      Number.isFinite(renderScale) &&
      renderScale > 0
    ) {
      this.levels.add(spatialScale);
      this.pixelSize = spatialScale / renderScale;
    }
    let spatialScaleIndex = spatialScales.get(spatialScale);
    if (spatialScaleIndex === undefined) {
      spatialScaleIndex = spatialScales.size;
      spatialScales.set(spatialScale, spatialScaleIndex);
    }
    if (spatialScaleIndex >= numHistogramRows) {
      this.numHistogramRows = numHistogramRows *= 2;
      const newValue = new Uint32Array(numHistogramRows * this.numBins * 2);
      newValue.set(value);
      this.value = value = newValue;
    }
    const index =
      spatialScaleIndex * this.numBins * 2 +
      Math.min(
        Math.max(
          0,
          Math.round(
            getRenderScaleHistogramOffset(
              renderScale,
              this.logScaleOrigin,
              this.binSize,
            ),
          ),
        ),
        this.numBins - 1,
      );
    value[index] += presentCount;
    value[index + this.numBins] += notPresentCount;
    if (renderOnly) {
      this.fakeChunkCount = this.fakeChunkCount + notPresentCount;
    }
  }
}

/** The middle of a voxel's three sizes: the spatial scale the slice views give a level. */
export function medianVoxelSize(size: ArrayLike<number>): number {
  const [a, b, c] = [size[0], size[1], size[2]];
  return Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));
}
