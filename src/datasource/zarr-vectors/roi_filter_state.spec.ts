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
  RoiOperator,
  RoiPredicate,
  type Roi,
} from "#src/datasource/zarr-vectors/roi.js";
import { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";

const ellipsoid: Roi = {
  shape: {
    kind: "ellipsoid",
    center: Float32Array.from([1, 2, 3]),
    radii: Float32Array.from([4, 5, 6]),
  },
  predicate: RoiPredicate.ANY_SEGMENT,
  operator: RoiOperator.AND,
};
const box: Roi = {
  shape: {
    kind: "box",
    lower: Float32Array.from([0, 0, 0]),
    upper: Float32Array.from([9, 9, 9]),
  },
  predicate: RoiPredicate.ANY_VERTEX,
  operator: RoiOperator.ANDNOT,
};

describe("RoiFilterState", () => {
  it("is empty and out of the URL by default", () => {
    const s = new RoiFilterState();
    expect(s.active).toBe(false);
    expect(s.rois).toEqual([]);
    expect(s.toJSON()).toBeUndefined();
  });

  it("round-trips ROIs, active, and ghostAlpha through JSON", () => {
    const s = new RoiFilterState();
    s.setRois([ellipsoid, box]);
    s.active = true;
    s.ghostAlpha = 0.25;
    const json = s.toJSON();
    expect(json.active).toBe(true);
    expect(json.ghostAlpha).toBe(0.25);
    expect(json.rois).toHaveLength(2);

    const restored = new RoiFilterState();
    restored.restoreState(json);
    expect(restored.active).toBe(true);
    expect(restored.ghostAlpha).toBe(0.25);
    expect(restored.rois).toHaveLength(2);
    // Geometry survives with the right kinds and values.
    const [e, b] = restored.rois;
    expect(e.shape.kind).toBe("ellipsoid");
    expect(Array.from((e.shape as any).radii)).toEqual([4, 5, 6]);
    expect(e.predicate).toBe(RoiPredicate.ANY_SEGMENT);
    expect(e.operator).toBe(RoiOperator.AND);
    expect(b.shape.kind).toBe("box");
    expect(b.predicate).toBe(RoiPredicate.ANY_VERTEX);
    expect(b.operator).toBe(RoiOperator.ANDNOT);
  });

  it("preserves ROI order through the round-trip", () => {
    const s = new RoiFilterState();
    s.setRois([box, ellipsoid]); // box first this time
    const restored = new RoiFilterState();
    restored.restoreState(s.toJSON());
    expect(restored.rois.map((r) => r.shape.kind)).toEqual([
      "box",
      "ellipsoid",
    ]);
  });

  it("omits ghostAlpha from JSON when at its default", () => {
    const s = new RoiFilterState();
    s.setRois([ellipsoid]);
    const json = s.toJSON();
    expect("ghostAlpha" in json).toBe(false);
  });

  it("dispatches changed on mutation and clamps ghostAlpha", () => {
    const s = new RoiFilterState();
    let count = 0;
    s.changed.add(() => ++count);
    s.active = true;
    s.setRois([ellipsoid]);
    s.ghostAlpha = 5; // clamps to 1
    expect(s.ghostAlpha).toBe(1);
    expect(count).toBe(3);
    // Setting the same value again does not re-dispatch.
    s.active = true;
    expect(count).toBe(3);
  });

  it("rejects an unknown predicate/operator/shape on restore", () => {
    const s = new RoiFilterState();
    expect(() =>
      s.restoreState({
        rois: [{ shape: shapeJson(), predicate: "nope", operator: "and" }],
      }),
    ).toThrow();
    expect(() =>
      s.restoreState({
        rois: [
          {
            shape: { type: "wedge" },
            predicate: "any_segment",
            operator: "and",
          },
        ],
      }),
    ).toThrow();
  });

  it("reset returns to default and clears ROIs", () => {
    const s = new RoiFilterState();
    s.setRois([ellipsoid]);
    s.active = true;
    s.colorByGroup = true;
    s.reset();
    expect(s.active).toBe(false);
    expect(s.colorByGroup).toBe(false);
    expect(s.rois).toEqual([]);
    expect(s.toJSON()).toBeUndefined();
  });

  it("round-trips colorByGroup", () => {
    const s = new RoiFilterState();
    s.setRois([ellipsoid]);
    s.colorByGroup = true;
    const restored = new RoiFilterState();
    restored.restoreState(s.toJSON());
    expect(restored.colorByGroup).toBe(true);
  });
});

describe("RoiFilterState mutators", () => {
  it("addRoi appends and returns the new index", () => {
    const s = new RoiFilterState();
    expect(s.addRoi(ellipsoid)).toBe(0);
    expect(s.addRoi(box)).toBe(1);
    expect(s.rois.map((r) => r.shape.kind)).toEqual(["ellipsoid", "box"]);
  });

  it("updateRoi replaces fields of one entry, out-of-range is a no-op", () => {
    const s = new RoiFilterState();
    s.setRois([ellipsoid, box]);
    s.updateRoi(1, { operator: RoiOperator.OR });
    expect(s.rois[1].operator).toBe(RoiOperator.OR);
    expect(s.rois[1].shape.kind).toBe("box"); // shape untouched
    s.updateRoi(9, { operator: RoiOperator.AND });
    expect(s.rois).toHaveLength(2);
  });

  it("removeRoi drops one entry; out-of-range is a no-op", () => {
    const s = new RoiFilterState();
    s.setRois([ellipsoid, box]);
    s.removeRoi(0);
    expect(s.rois.map((r) => r.shape.kind)).toEqual(["box"]);
    s.removeRoi(5);
    expect(s.rois).toHaveLength(1);
  });

  it("moveRoi reorders; identity/out-of-range moves are no-ops", () => {
    const s = new RoiFilterState();
    const c: Roi = { ...box, predicate: RoiPredicate.EITHER_ENDPOINT };
    s.setRois([ellipsoid, box, c]);
    s.moveRoi(2, 0); // c to front
    expect(s.rois.map((r) => r.predicate)).toEqual([
      RoiPredicate.EITHER_ENDPOINT,
      RoiPredicate.ANY_SEGMENT,
      RoiPredicate.ANY_VERTEX,
    ]);
    s.moveRoi(1, 1); // no-op
    s.moveRoi(0, 9); // out of range no-op
    expect(s.rois).toHaveLength(3);
  });

  it("each mutator dispatches changed", () => {
    const s = new RoiFilterState();
    let count = 0;
    s.changed.add(() => ++count);
    s.addRoi(ellipsoid);
    s.addRoi(box);
    s.updateRoi(0, { operator: RoiOperator.OR });
    s.moveRoi(0, 1);
    s.removeRoi(0);
    expect(count).toBe(5);
  });
});

function shapeJson() {
  return { type: "ellipsoid", center: [0, 0, 0], radii: [1, 1, 1] };
}
