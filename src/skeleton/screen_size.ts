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
 * On-screen-size primitives shared by every skeleton LOD-selection path
 * (spatially-indexed pass-1's grid-anchor arbitration in `backend.ts`, and
 * the object-keyed multiscale selection in `multiscale_object_selection.ts`).
 * Extracted from `backend.ts`, where these were originally private/inlined,
 * so both paths compute "how big does this world-space size look on screen"
 * the same way instead of maintaining two copies.
 */

export function getChunkSpacing(size: Float32Array): number {
  return Math.max(Math.min(size[0], size[1], size[2]), 1e-6);
}

export function computePhysicalUnitsPerScreenPixelAtPoint(
  modelViewProjection: Float32Array,
  viewportWidth: number,
  viewportHeight: number,
  worldPoint: Float32Array,
  displayDimensionScales?: Float64Array,
): number {
  const m = modelViewProjection;
  const m00 = m[0],
    m10 = m[1];
  const m01 = m[4],
    m11 = m[5];
  const m02 = m[8],
    m12 = m[9];
  const m30 = m[3],
    m31 = m[7],
    m32 = m[11],
    m33 = m[15];
  const w =
    m30 * worldPoint[0] + m31 * worldPoint[1] + m32 * worldPoint[2] + m33;
  if (!Number.isFinite(w) || w <= 0) return Number.POSITIVE_INFINITY;

  const sx =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 0 &&
    Number.isFinite(displayDimensionScales[0]) &&
    displayDimensionScales[0] > 0
      ? displayDimensionScales[0]
      : 1;
  const sy =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 1 &&
    Number.isFinite(displayDimensionScales[1]) &&
    displayDimensionScales[1] > 0
      ? displayDimensionScales[1]
      : sx;
  const sz =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 2 &&
    Number.isFinite(displayDimensionScales[2]) &&
    displayDimensionScales[2] > 0
      ? displayDimensionScales[2]
      : sy;

  const xScale = Math.sqrt(
    ((m00 / sx) * viewportWidth) ** 2 + ((m10 / sx) * viewportHeight) ** 2,
  );
  const yScale = Math.sqrt(
    ((m01 / sy) * viewportWidth) ** 2 + ((m11 / sy) * viewportHeight) ** 2,
  );
  const zScale = Math.sqrt(
    ((m02 / sz) * viewportWidth) ** 2 + ((m12 / sz) * viewportHeight) ** 2,
  );
  const scaleFactor = Math.max(xScale, yScale, zScale);
  if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  return w / scaleFactor;
}

export function getMetersPerUnit(projectionParameters: {
  displayDimensionRenderInfo?: { displayDimensionScales?: Float64Array };
}): number {
  const ddScales =
    projectionParameters.displayDimensionRenderInfo?.displayDimensionScales;
  if (ddScales === undefined || ddScales.length === 0) {
    return 1;
  }
  let metersPerUnit = Infinity;
  for (let i = 0; i < ddScales.length; ++i) {
    const s = ddScales[i];
    if (Number.isFinite(s) && s > 0) {
      metersPerUnit = Math.min(metersPerUnit, s);
    }
  }
  return Number.isFinite(metersPerUnit) ? metersPerUnit : 1;
}

export function quantizeSpacingForArbitration(spacing: number): number {
  const clamped = Math.max(spacing, 1e-12);
  const log2Spacing = Math.log2(clamped);
  const quantizedLog = Math.round(log2Spacing * 4) / 4;
  return 2 ** quantizedLog;
}
