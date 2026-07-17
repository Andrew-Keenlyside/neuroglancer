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
 * @file Bridge from neuroglancer annotations to {@link roi.ts} ROI geometry.
 *
 * ROIs are authored as ordinary ellipsoid / bounding-box annotations (free
 * placement, editing, and URL round-trip). This turns one annotation into a
 * {@link RoiShape} and maps it from the annotation layer's coordinate space
 * into the tract layer's model space, so the streamline test runs in the same
 * coordinates as the vertex positions.
 *
 * Kept free of any backend/render dependency: the frontend builds a plain
 * `Roi[]` here and transmits *that* (not annotation objects) to the worker, so
 * the per-chunk filter imports only the pure geometry in `roi.ts`.
 */

import type { Annotation } from "#src/annotation/index.js";
import { AnnotationType } from "#src/annotation/index.js";
import type {
  Roi,
  RoiOperator,
  RoiPredicate,
  RoiShape,
} from "#src/datasource/zarr-vectors/roi.js";

/**
 * A per-axis affine `tract[i] = scales[i] * annotation[i] + offsets[i]` mapping
 * annotation coordinates to the tract layer's model space.
 *
 * Only diagonal (per-axis scale + translation) transforms are representable,
 * because a {@link RoiShape} is axis-aligned: a rotation between the two spaces
 * would shear an ellipsoid off its axes, which this shape family cannot express.
 * Aligned neuroglancer layers differ only by unit scale and translation, which
 * this covers; a rotated pairing is rejected upstream, not silently mis-mapped.
 */
export interface RoiAxisTransform {
  readonly scales: Float64Array;
  readonly offsets: Float64Array;
}

/** Identity transform of the given rank. */
export function identityRoiTransform(rank: number): RoiAxisTransform {
  return {
    scales: new Float64Array(rank).fill(1),
    offsets: new Float64Array(rank),
  };
}

function applyAxis(
  value: number,
  axis: number,
  transform: RoiAxisTransform,
): number {
  return transform.scales[axis] * value + transform.offsets[axis];
}

/**
 * Convert one annotation to an {@link RoiShape} in tract model space, or
 * `undefined` for an annotation type that is not a region (line, point,
 * polyline).
 *
 * - Ellipsoid → axis-aligned ellipsoid: `center` maps through the affine;
 *   `radii` scale by `|scales|` (per axis, sign-independent — a radius is a
 *   magnitude).
 * - Bounding box → box with `lower`/`upper` the per-axis min/max of the two
 *   mapped corners (the affine may flip an axis if a scale is negative, so
 *   re-derive min/max rather than assuming corner order survives).
 */
export function annotationToRoiShape(
  annotation: Annotation,
  transform: RoiAxisTransform,
): RoiShape | undefined {
  switch (annotation.type) {
    case AnnotationType.ELLIPSOID: {
      const rank = annotation.center.length;
      const center = new Float32Array(rank);
      const radii = new Float32Array(rank);
      for (let i = 0; i < rank; ++i) {
        center[i] = applyAxis(annotation.center[i], i, transform);
        radii[i] = Math.abs(transform.scales[i]) * annotation.radii[i];
      }
      return { kind: "ellipsoid", center, radii };
    }
    case AnnotationType.AXIS_ALIGNED_BOUNDING_BOX: {
      const rank = annotation.pointA.length;
      const lower = new Float32Array(rank);
      const upper = new Float32Array(rank);
      for (let i = 0; i < rank; ++i) {
        const a = applyAxis(annotation.pointA[i], i, transform);
        const b = applyAxis(annotation.pointB[i], i, transform);
        lower[i] = Math.min(a, b);
        upper[i] = Math.max(a, b);
      }
      return { kind: "box", lower, upper };
    }
    default:
      return undefined;
  }
}

/** One entry in the ordered ROI filter list: an annotation plus its role. */
export interface RoiFilterEntry {
  readonly annotation: Annotation;
  readonly predicate: RoiPredicate;
  readonly operator: RoiOperator;
}

/**
 * Assemble the ordered `Roi[]` the filter evaluates, dropping entries whose
 * annotation is not a region. Order is preserved: it is the whole of the
 * left-fold syntax (see {@link streamlinePassesRois}).
 */
export function buildRoiList(
  entries: readonly RoiFilterEntry[],
  transform: RoiAxisTransform,
): Roi[] {
  const rois: Roi[] = [];
  for (const entry of entries) {
    const shape = annotationToRoiShape(entry.annotation, transform);
    if (shape === undefined) continue;
    rois.push({
      shape,
      predicate: entry.predicate,
      operator: entry.operator,
    });
  }
  return rois;
}
