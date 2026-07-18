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
import { vec3 } from "#src/util/geom.js";

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

/** Count `changed` dispatches while running `fn`. */
function countChanges(s: RoiFilterState, fn: () => void): number {
  let n = 0;
  const unregister = s.changed.add(() => {
    n++;
  });
  fn();
  unregister();
  return n;
}

describe("RoiFilterState groups", () => {
  it("is empty, colour-by-group on, and out of the URL by default", () => {
    const s = new RoiFilterState();
    expect(s.active).toBe(false);
    expect(s.colorByGroup).toBe(true);
    expect(s.groups).toEqual([]);
    expect(s.hasVisibleRois()).toBe(false);
    expect(s.toJSON()).toBeUndefined();
  });

  it("adds groups with distinct ids, default names, palette colours", () => {
    const s = new RoiFilterState();
    const a = s.addGroup();
    const b = s.addGroup();
    expect(a).not.toBe(b);
    expect(s.groups.map((g) => g.name)).toEqual(["Group 1", "Group 2"]);
    expect(s.groups[0].visible).toBe(true);
    // Distinct palette colours for the first two groups.
    expect(Array.from(s.groups[0].color)).not.toEqual(
      Array.from(s.groups[1].color),
    );
  });

  it("binds ROIs to a group and reports hasVisibleRois", () => {
    const s = new RoiFilterState();
    const g = s.addGroup();
    expect(s.hasVisibleRois()).toBe(false);
    s.addRoi(g, ellipsoid);
    expect(s.groups[0].rois).toEqual([ellipsoid]);
    expect(s.hasVisibleRois()).toBe(true);
    // A hidden group's ROIs do not count.
    s.updateGroup(g, { visible: false });
    expect(s.hasVisibleRois()).toBe(false);
  });

  it("addRoi to a missing group is a no-op returning -1", () => {
    const s = new RoiFilterState();
    expect(countChanges(s, () => expect(s.addRoi(999, ellipsoid)).toBe(-1))).toBe(
      0,
    );
  });

  it("updates group name/colour/visibility", () => {
    const s = new RoiFilterState();
    const g = s.addGroup();
    s.updateGroup(g, { name: "Motor", visible: false });
    expect(s.groups[0].name).toBe("Motor");
    expect(s.groups[0].visible).toBe(false);
  });

  it("removes a group and reorders groups", () => {
    const s = new RoiFilterState();
    const a = s.addGroup();
    s.addGroup();
    s.moveGroup(0, 1);
    expect(s.groups[1].id).toBe(a);
    s.removeGroup(a);
    expect(s.groups.map((g) => g.id)).not.toContain(a);
  });

  it("updates, removes, and reorders ROIs within a group", () => {
    const s = new RoiFilterState();
    const g = s.addGroup();
    s.addRoi(g, ellipsoid);
    s.addRoi(g, box);
    s.updateRoi(g, 0, { operator: RoiOperator.OR });
    expect(s.groups[0].rois[0].operator).toBe(RoiOperator.OR);
    s.moveRoi(g, 0, 1);
    expect(s.groups[0].rois[1].operator).toBe(RoiOperator.OR);
    s.removeRoi(g, 0);
    expect(s.groups[0].rois).toHaveLength(1);
  });

  it("every mutator reassigns arrays (new identity) and dispatches once", () => {
    const s = new RoiFilterState();
    const g = s.addGroup();
    s.addRoi(g, ellipsoid);
    const before = s.groups;
    expect(countChanges(s, () => s.updateRoi(g, 0, { predicate: RoiPredicate.ANY_VERTEX }))).toBe(1);
    expect(s.groups).not.toBe(before); // new array reference
  });
});

describe("RoiFilterState serialization", () => {
  it("round-trips groups (name, colour, visibility, ROIs) + config through JSON", () => {
    const s = new RoiFilterState();
    const g = s.addGroup();
    s.updateGroup(g, {
      name: "Motor",
      color: vec3.fromValues(1, 0, 0),
      visible: false,
      opacity: 0.4,
      highDetail: true,
    });
    s.addRoi(g, ellipsoid);
    s.addRoi(g, box);
    s.active = true;
    s.ghostAlpha = 0.25;
    s.colorByGroup = false;
    s.hideOverlays2d = true;

    const json = s.toJSON();
    expect(json.hideOverlays2d).toBe(true);
    expect(json.groups).toHaveLength(1);
    expect(json.groups[0].name).toBe("Motor");
    expect(json.groups[0].color).toBe("#ff0000");
    expect(json.groups[0].visible).toBe(false);
    expect(json.groups[0].opacity).toBe(0.4);
    expect(json.groups[0].highDetail).toBe(true);
    expect(json.active).toBe(true);
    expect(json.ghostAlpha).toBe(0.25);
    expect(json.colorByGroup).toBe(false);

    const restored = new RoiFilterState();
    restored.restoreState(json);
    expect(restored.groups[0].name).toBe("Motor");
    expect(Array.from(restored.groups[0].color)).toEqual([1, 0, 0]);
    expect(restored.groups[0].visible).toBe(false);
    expect(restored.groups[0].opacity).toBe(0.4);
    expect(restored.groups[0].highDetail).toBe(true);
    expect(restored.groups[0].rois).toHaveLength(2);
    expect(restored.active).toBe(true);
    expect(restored.ghostAlpha).toBe(0.25);
    expect(restored.colorByGroup).toBe(false);
    expect(restored.hideOverlays2d).toBe(true);
  });

  it("defaults new-group opacity to 1 and highDetail to false, omitted from JSON", () => {
    const s = new RoiFilterState();
    s.addGroup();
    expect(s.groups[0].opacity).toBe(1);
    expect(s.groups[0].highDetail).toBe(false);
    const json = s.toJSON();
    expect("opacity" in json.groups[0]).toBe(false);
    expect("highDetail" in json.groups[0]).toBe(false);
  });

  it("omits ghostAlpha/colorByGroup from JSON at their defaults", () => {
    const s = new RoiFilterState();
    s.addGroup();
    const json = s.toJSON();
    expect("ghostAlpha" in json).toBe(false);
    expect("colorByGroup" in json).toBe(false); // default true
  });

  it("migrates an old flat `rois` list into one default group", () => {
    const s = new RoiFilterState();
    s.restoreState({
      rois: [
        {
          shape: { type: "ellipsoid", center: [1, 2, 3], radii: [4, 5, 6] },
          predicate: "any_segment",
          operator: "and",
        },
      ],
      active: true,
    });
    expect(s.groups).toHaveLength(1);
    expect(s.groups[0].name).toBe("Group 1");
    expect(s.groups[0].visible).toBe(true);
    expect(s.groups[0].rois).toHaveLength(1);
    expect(s.active).toBe(true);
  });

  it("reset returns to the empty default and re-runs from a clean id counter", () => {
    const s = new RoiFilterState();
    s.addGroup();
    s.active = true;
    s.reset();
    expect(s.groups).toEqual([]);
    expect(s.active).toBe(false);
    expect(s.colorByGroup).toBe(true);
    // A fresh group after reset is "Group 1" again.
    s.addGroup();
    expect(s.groups[0].name).toBe("Group 1");
  });
});

const f32 = (...xs: number[]) => Float32Array.from(xs);

describe("insertGroup", () => {
  it("carries a group's contents across states intact", () => {
    // Separating a group to its own layer, and merging it back, both move a
    // group between two independent RoiFilterStates. Nothing about it may be
    // lost in transit except the id, which is only unique within one state.
    const source = new RoiFilterState();
    const id = source.addGroup();
    source.updateGroup(id, {
      name: "Corticospinal",
      color: vec3.fromValues(0.25, 0.5, 0.75),
      visible: false,
      opacity: 0.4,
    });
    source.addRoi(id, {
      shape: { kind: "ellipsoid", center: f32(1, 2, 3), radii: f32(4, 5, 6) },
      predicate: RoiPredicate.ANY_VERTEX,
      operator: RoiOperator.ANDNOT,
    });
    source.addRoi(id, {
      shape: { kind: "box", lower: f32(0, 0, 0), upper: f32(9, 9, 9) },
      predicate: RoiPredicate.ANY_SEGMENT,
      operator: RoiOperator.OR,
    });
    const original = source.groups[0];

    const target = new RoiFilterState();
    const newId = target.insertGroup(original);

    const moved = target.groups[0];
    expect(target.groups).toHaveLength(1);
    expect(moved.id).toBe(newId);
    expect(moved.name).toBe("Corticospinal");
    expect(moved.visible).toBe(false);
    expect(moved.opacity).toBe(0.4);
    expect(Array.from(moved.color)).toEqual(Array.from(original.color));
    expect(moved.rois).toEqual(original.rois);

    // The two states are now independent: the source still holds its copy
    // until the caller removes it, and editing one must not touch the other.
    source.removeGroup(id);
    expect(source.groups).toHaveLength(0);
    expect(target.groups).toHaveLength(1);
    expect(target.groups[0].rois).toHaveLength(2);
  });

  it("survives a round trip through JSON", () => {
    // Separating goes via toJSON/restoreState (the new layer is built from a
    // spec), so a moved group must also survive serialisation.
    const source = new RoiFilterState();
    const id = source.addGroup();
    source.updateGroup(id, { name: "Arcuate", opacity: 0.6 });
    source.addRoi(id, {
      shape: { kind: "box", lower: f32(-1, -2, -3), upper: f32(1, 2, 3) },
      predicate: RoiPredicate.ANY_SEGMENT,
      operator: RoiOperator.AND,
    });

    const carrier = new RoiFilterState();
    carrier.insertGroup(source.groups[0]);
    carrier.active = true;
    carrier.ghostAlpha = 0;

    const restored = new RoiFilterState();
    restored.restoreState(carrier.toJSON());

    expect(restored.groups).toHaveLength(1);
    expect(restored.groups[0].name).toBe("Arcuate");
    expect(restored.groups[0].opacity).toBeCloseTo(0.6);
    expect(restored.groups[0].rois).toHaveLength(1);
    expect(restored.ghostAlpha).toBe(0);
    expect(restored.active).toBe(true);
  });

  it("assigns ids that do not collide with existing groups", () => {
    const target = new RoiFilterState();
    const existing = target.addGroup();
    const source = new RoiFilterState();
    source.addGroup();
    const moved = target.insertGroup(source.groups[0]);
    expect(moved).not.toBe(existing);
    expect(new Set(target.groups.map((g) => g.id)).size).toBe(2);
  });
});
