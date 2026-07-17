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
import { AnnotationType } from "#src/annotation/index.js";
import type {
  Annotation,
  AxisAlignedBoundingBox,
  Ellipsoid,
  Line,
} from "#src/annotation/index.js";
import { RoiOperator, RoiPredicate } from "#src/datasource/zarr-vectors/roi.js";
import {
  annotationToRoiShape,
  buildRoiList,
  identityRoiTransform,
  type RoiAxisTransform,
} from "#src/datasource/zarr-vectors/roi_filter.js";

const ID = identityRoiTransform(3);

function ellipsoid(center: number[], radii: number[]): Ellipsoid {
  return {
    type: AnnotationType.ELLIPSOID,
    id: "e",
    center: Float32Array.from(center),
    radii: Float32Array.from(radii),
    properties: [],
  };
}
function box(pointA: number[], pointB: number[]): AxisAlignedBoundingBox {
  return {
    type: AnnotationType.AXIS_ALIGNED_BOUNDING_BOX,
    id: "b",
    pointA: Float32Array.from(pointA),
    pointB: Float32Array.from(pointB),
    properties: [],
  };
}
function line(): Line {
  return {
    type: AnnotationType.LINE,
    id: "l",
    pointA: Float32Array.from([0, 0, 0]),
    pointB: Float32Array.from([1, 1, 1]),
    properties: [],
  };
}

describe("annotationToRoiShape — ellipsoid", () => {
  it("passes centre and radii through under identity", () => {
    const shape = annotationToRoiShape(ellipsoid([1, 2, 3], [4, 5, 6]), ID);
    expect(shape).toEqual({
      kind: "ellipsoid",
      center: Float32Array.from([1, 2, 3]),
      radii: Float32Array.from([4, 5, 6]),
    });
  });

  it("maps centre through the affine and scales radii by |scale|", () => {
    const t: RoiAxisTransform = {
      scales: Float64Array.from([2, 2, 2]),
      offsets: Float64Array.from([10, 0, -5]),
    };
    const shape = annotationToRoiShape(ellipsoid([1, 1, 1], [3, 3, 3]), t);
    // center: 2*1+offset ; radii: |2|*3
    expect(Array.from((shape as any).center)).toEqual([12, 2, -3]);
    expect(Array.from((shape as any).radii)).toEqual([6, 6, 6]);
  });

  it("keeps radii positive under a negative axis scale", () => {
    const t: RoiAxisTransform = {
      scales: Float64Array.from([-1, 1, 1]),
      offsets: Float64Array.from([0, 0, 0]),
    };
    const shape = annotationToRoiShape(ellipsoid([0, 0, 0], [4, 4, 4]), t);
    expect((shape as any).radii[0]).toBe(4); // not -4
  });
});

describe("annotationToRoiShape — bounding box", () => {
  it("derives lower/upper as per-axis min/max regardless of corner order", () => {
    const shape = annotationToRoiShape(box([5, 0, 9], [1, 8, 2]), ID);
    expect(Array.from((shape as any).lower)).toEqual([1, 0, 2]);
    expect(Array.from((shape as any).upper)).toEqual([5, 8, 2 < 9 ? 9 : 2]);
  });

  it("re-derives min/max after a negative-scale axis flip", () => {
    // scale -1 on x flips the corners' order, so lower/upper must be recomputed.
    const t: RoiAxisTransform = {
      scales: Float64Array.from([-1, 1, 1]),
      offsets: Float64Array.from([0, 0, 0]),
    };
    const shape = annotationToRoiShape(box([1, 0, 0], [5, 1, 1]), t);
    // x: -1*1=-1 and -1*5=-5 -> lower -5, upper -1
    expect((shape as any).lower[0]).toBe(-5);
    expect((shape as any).upper[0]).toBe(-1);
  });
});

describe("annotationToRoiShape — non-region annotations", () => {
  it("returns undefined for a line", () => {
    expect(annotationToRoiShape(line(), ID)).toBeUndefined();
  });
});

describe("buildRoiList", () => {
  it("preserves order and carries predicate/operator", () => {
    const entries = [
      {
        annotation: ellipsoid([0, 0, 0], [1, 1, 1]),
        predicate: RoiPredicate.ANY_SEGMENT,
        operator: RoiOperator.AND,
      },
      {
        annotation: box([0, 0, 0], [1, 1, 1]),
        predicate: RoiPredicate.ANY_VERTEX,
        operator: RoiOperator.ANDNOT,
      },
    ];
    const rois = buildRoiList(entries, ID);
    expect(rois.map((r) => r.shape.kind)).toEqual(["ellipsoid", "box"]);
    expect(rois[1].predicate).toBe(RoiPredicate.ANY_VERTEX);
    expect(rois[1].operator).toBe(RoiOperator.ANDNOT);
  });

  it("drops non-region entries but keeps order of the rest", () => {
    const entries = [
      {
        annotation: ellipsoid([0, 0, 0], [1, 1, 1]),
        predicate: RoiPredicate.ANY_SEGMENT,
        operator: RoiOperator.AND,
      },
      {
        annotation: line() as Annotation,
        predicate: RoiPredicate.ANY_SEGMENT,
        operator: RoiOperator.AND,
      },
      {
        annotation: box([0, 0, 0], [1, 1, 1]),
        predicate: RoiPredicate.ANY_SEGMENT,
        operator: RoiOperator.OR,
      },
    ];
    const rois = buildRoiList(entries, ID);
    expect(rois.map((r) => r.shape.kind)).toEqual(["ellipsoid", "box"]);
  });
});
