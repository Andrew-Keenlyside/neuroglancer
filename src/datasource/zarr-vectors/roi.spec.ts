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

import { describe, expect, it } from "vitest";
import {
  combineRoiVerdicts,
  RoiOperator,
  RoiPredicate,
  streamlinePassesRoi,
  roiRegionBounds,
  streamlinePassesRois,
  type Roi,
  type RoiShape,
  type StreamlineRef,
} from "#src/datasource/zarr-vectors/roi.js";

const f32 = (...xs: number[]) => Float32Array.from(xs);

/** A rank-3 streamline from a flat list of xyz triples. */
function line(...xyz: number[]): StreamlineRef {
  return {
    positions: Float32Array.from(xyz),
    start: 0,
    count: xyz.length / 3,
    rank: 3,
  };
}

const sphere = (cx: number, cy: number, cz: number, r: number): RoiShape => ({
  kind: "ellipsoid",
  center: f32(cx, cy, cz),
  radii: f32(r, r, r),
});

const passes = (
  s: StreamlineRef,
  shape: RoiShape,
  p = RoiPredicate.ANY_SEGMENT,
) => streamlinePassesRoi(s, shape, p);

describe("streamlinePassesRoi — ellipsoid", () => {
  it("passes a track with a vertex inside the sphere", () => {
    expect(passes(line(0, 0, 0, 10, 0, 0, 20, 0, 0), sphere(10, 0, 0, 2))).toBe(
      true,
    );
  });

  it("rejects a track that stays outside", () => {
    expect(passes(line(0, 0, 0, 10, 0, 0), sphere(0, 50, 0, 2))).toBe(false);
  });

  it("catches a track that crosses BETWEEN vertices (the leap-across case)", () => {
    // The motivating case for ANY_SEGMENT: the track passes straight through
    // the sphere, but every vertex is well outside it. TrackVis's ANY_VERTEX
    // test misses this; testing the polyline does not.
    const track = line(-10, 0, 0, 10, 0, 0);
    const roi = sphere(0, 0, 0, 1);
    expect(passes(track, roi, RoiPredicate.ANY_SEGMENT)).toBe(true);
    expect(passes(track, roi, RoiPredicate.ANY_VERTEX)).toBe(false);
  });

  it("respects each radius independently (anisotropic ellipsoid)", () => {
    const shape: RoiShape = {
      kind: "ellipsoid",
      center: f32(0, 0, 0),
      radii: f32(10, 1, 1),
    };
    // Inside the long axis, outside the short one.
    expect(passes(line(5, 5, 0, 6, 5, 0), shape)).toBe(false);
    expect(passes(line(5, 0.5, 0, 6, 0.5, 0), shape)).toBe(true);
  });

  it("treats the surface as inside (closed region)", () => {
    expect(
      passes(
        line(2, 0, 0, 2, 0, 0),
        sphere(0, 0, 0, 2),
        RoiPredicate.ANY_VERTEX,
      ),
    ).toBe(true);
  });

  it("rejects a degenerate ellipsoid rather than dividing by zero", () => {
    const shape: RoiShape = {
      kind: "ellipsoid",
      center: f32(0, 0, 0),
      radii: f32(1, 1, 0),
    };
    expect(passes(line(0, 0, 0, 1, 0, 0), shape)).toBe(false);
  });
});

describe("streamlinePassesRoi — box", () => {
  const box: RoiShape = {
    kind: "box",
    lower: f32(-1, -1, -1),
    upper: f32(1, 1, 1),
  };

  it("passes a track crossing the box between vertices", () => {
    expect(passes(line(-10, 0, 0, 10, 0, 0), box)).toBe(true);
  });

  it("rejects a track passing beside the box", () => {
    expect(passes(line(-10, 5, 0, 10, 5, 0), box)).toBe(false);
  });

  it("handles a segment parallel to a slab", () => {
    // Constant z, inside the z-slab: must not be discarded by the parallel case.
    expect(passes(line(-10, 0, 0.5, 10, 0, 0.5), box)).toBe(true);
    // Constant z, outside the z-slab.
    expect(passes(line(-10, 0, 9, 10, 0, 9), box)).toBe(false);
  });

  it("rejects a segment that stops short of the box", () => {
    expect(passes(line(-10, 0, 0, -5, 0, 0), box)).toBe(false);
  });

  it("passes a track wholly inside", () => {
    expect(passes(line(-0.5, 0, 0, 0.5, 0, 0), box)).toBe(true);
  });
});

describe("streamlinePassesRoi — halfspace", () => {
  // Keep x >= 0.
  const half: RoiShape = {
    kind: "halfspace",
    origin: f32(0, 0, 0),
    normal: f32(1, 0, 0),
  };

  it("passes a track that crosses the plane", () => {
    expect(passes(line(-5, 0, 0, 5, 0, 0), half)).toBe(true);
  });

  it("passes a track wholly on the positive side", () => {
    expect(passes(line(1, 0, 0, 5, 0, 0), half)).toBe(true);
  });

  it("rejects a track wholly on the negative side", () => {
    expect(passes(line(-5, 0, 0, -1, 0, 0), half)).toBe(false);
  });

  it("does not require a unit normal", () => {
    const scaled: RoiShape = {
      kind: "halfspace",
      origin: f32(0, 0, 0),
      normal: f32(7, 0, 0),
    };
    expect(passes(line(1, 0, 0, 5, 0, 0), scaled)).toBe(true);
    expect(passes(line(-5, 0, 0, -1, 0, 0), scaled)).toBe(false);
  });
});

describe("streamlinePassesRoi — predicates", () => {
  // Enters the sphere at the origin only in the middle of its run.
  const track = line(-10, 0, 0, 0, 0, 0, 10, 0, 0);
  const roi = sphere(0, 0, 0, 1);

  it("EITHER_ENDPOINT selects tracts terminating in the region", () => {
    expect(passes(track, roi, RoiPredicate.EITHER_ENDPOINT)).toBe(false);
    const ending = line(-10, 0, 0, -5, 0, 0, 0, 0, 0);
    expect(passes(ending, roi, RoiPredicate.EITHER_ENDPOINT)).toBe(true);
  });

  it("BOTH_ENDPOINTS requires each end inside", () => {
    const oneEnd = line(0, 0, 0, 10, 0, 0);
    expect(passes(oneEnd, roi, RoiPredicate.BOTH_ENDPOINTS)).toBe(false);
    const bothEnds = line(0.5, 0, 0, 5, 0, 0, -0.5, 0, 0);
    expect(passes(bothEnds, roi, RoiPredicate.BOTH_ENDPOINTS)).toBe(true);
  });

  it("ANY_SEGMENT falls back to the vertex for a single-vertex track", () => {
    const dot = line(0, 0, 0);
    expect(passes(dot, roi, RoiPredicate.ANY_SEGMENT)).toBe(true);
    expect(passes(line(9, 9, 9), roi, RoiPredicate.ANY_SEGMENT)).toBe(false);
  });

  it("rejects an empty streamline", () => {
    const empty: StreamlineRef = {
      positions: new Float32Array(0),
      start: 0,
      count: 0,
      rank: 3,
    };
    expect(passes(empty, roi)).toBe(false);
  });
});

describe("streamlinePassesRoi — addressing", () => {
  it("reads only the requested run out of a shared buffer", () => {
    // Two tracks packed back to back; the second is inside the sphere.
    const positions = Float32Array.from([
      50,
      50,
      50,
      51,
      50,
      50, // track 0, far away
      0,
      0,
      0,
      1,
      0,
      0, // track 1, at the origin
    ]);
    const roi = sphere(0, 0, 0, 2);
    expect(
      streamlinePassesRoi(
        { positions, start: 0, count: 2, rank: 3 },
        roi,
        RoiPredicate.ANY_SEGMENT,
      ),
    ).toBe(false);
    expect(
      streamlinePassesRoi(
        { positions, start: 2, count: 2, rank: 3 },
        roi,
        RoiPredicate.ANY_SEGMENT,
      ),
    ).toBe(true);
  });
});

describe("streamlinePassesRois — composition", () => {
  const atOrigin = sphere(0, 0, 0, 1);
  const atTen = sphere(10, 0, 0, 1);
  const roi = (shape: RoiShape, operator: RoiOperator): Roi => ({
    shape,
    operator,
    predicate: RoiPredicate.ANY_SEGMENT,
  });

  it("passes everything when there are no regions", () => {
    expect(streamlinePassesRois(line(99, 99, 99), [])).toBe(true);
  });

  it("seeds the verdict from the first region, treating Or as Include", () => {
    const through = line(-5, 0, 0, 5, 0, 0);
    // Both seed to "passes this region": OR is degenerate at index 0
    // (`false || x === x`), so it must not act as a third seeding mode.
    for (const op of [RoiOperator.AND, RoiOperator.OR]) {
      expect(streamlinePassesRois(through, [roi(atOrigin, op)])).toBe(true);
    }
  });

  it("seeds from a leading ANDNOT, so a dissection can be pure exclusion", () => {
    // "Everything except tracts crossing the origin" -- expressible without a
    // leading include-everything region.
    const rois = [roi(atOrigin, RoiOperator.ANDNOT)];
    expect(streamlinePassesRois(line(-5, 0, 0, 5, 0, 0), rois)).toBe(false);
    expect(streamlinePassesRois(line(9, 0, 0, 11, 0, 0), rois)).toBe(true);
  });

  it("folds subsequent regions into a leading exclusion", () => {
    // NOT through the origin, AND through x=10.
    const rois = [
      roi(atOrigin, RoiOperator.ANDNOT),
      roi(atTen, RoiOperator.AND),
    ];
    expect(streamlinePassesRois(line(9, 0, 0, 11, 0, 0), rois)).toBe(true);
    expect(streamlinePassesRois(line(-5, 0, 0, 15, 0, 0), rois)).toBe(false);
    expect(streamlinePassesRois(line(0, 50, 0, 1, 50, 0), rois)).toBe(false);
  });

  it("AND requires passing both regions", () => {
    const both = line(-5, 0, 0, 15, 0, 0); // crosses origin and x=10
    const onlyOrigin = line(-5, 0, 0, 1, 0, 0);
    const rois = [roi(atOrigin, RoiOperator.AND), roi(atTen, RoiOperator.AND)];
    expect(streamlinePassesRois(both, rois)).toBe(true);
    expect(streamlinePassesRois(onlyOrigin, rois)).toBe(false);
  });

  it("OR accepts either region", () => {
    const rois = [roi(atOrigin, RoiOperator.AND), roi(atTen, RoiOperator.OR)];
    expect(streamlinePassesRois(line(-5, 0, 0, 1, 0, 0), rois)).toBe(true);
    expect(streamlinePassesRois(line(9, 0, 0, 11, 0, 0), rois)).toBe(true);
    expect(streamlinePassesRois(line(0, 50, 0, 10, 50, 0), rois)).toBe(false);
  });

  it("ANDNOT excludes tracks passing the region", () => {
    // Through the origin but NOT through x=10.
    const rois = [
      roi(atOrigin, RoiOperator.AND),
      roi(atTen, RoiOperator.ANDNOT),
    ];
    expect(streamlinePassesRois(line(-5, 0, 0, 1, 0, 0), rois)).toBe(true);
    expect(streamlinePassesRois(line(-5, 0, 0, 15, 0, 0), rois)).toBe(false);
  });

  it("folds left, so order changes the result", () => {
    const through = line(-5, 0, 0, 15, 0, 0); // passes both regions
    // (origin AND NOT ten) -> false;  (ten ANDNOT-ed first then OR) differs.
    const a = [roi(atOrigin, RoiOperator.AND), roi(atTen, RoiOperator.ANDNOT)];
    const b = [roi(atTen, RoiOperator.AND), roi(atOrigin, RoiOperator.OR)];
    expect(streamlinePassesRois(through, a)).toBe(false);
    expect(streamlinePassesRois(through, b)).toBe(true);
  });

  it("short-circuits later regions once the verdict is settled", () => {
    // Records whether a region was evaluated at all, so the skip is observable
    // rather than merely assumed from an unchanged verdict.
    const evaluated: string[] = [];
    const spy = (name: string, shape: RoiShape): RoiShape =>
      new Proxy(shape, {
        get(target, prop, receiver) {
          if (prop === "kind") evaluated.push(name);
          return Reflect.get(target, prop, receiver);
        },
      }) as RoiShape;

    // Fails the first region, so an AND can never recover: skip the second.
    evaluated.length = 0;
    streamlinePassesRois(line(0, 50, 0, 1, 50, 0), [
      roi(atOrigin, RoiOperator.AND),
      roi(spy("second", atTen), RoiOperator.AND),
    ]);
    expect(evaluated).not.toContain("second");

    // Passes the first, so an OR is already satisfied: skip the second.
    evaluated.length = 0;
    streamlinePassesRois(line(-5, 0, 0, 1, 0, 0), [
      roi(atOrigin, RoiOperator.AND),
      roi(spy("second", atTen), RoiOperator.OR),
    ]);
    expect(evaluated).not.toContain("second");

    // But an ANDNOT after a passing region must still be evaluated.
    evaluated.length = 0;
    streamlinePassesRois(line(-5, 0, 0, 1, 0, 0), [
      roi(atOrigin, RoiOperator.AND),
      roi(spy("second", atTen), RoiOperator.ANDNOT),
    ]);
    expect(evaluated).toContain("second");
  });

  it("mixes predicates across regions", () => {
    const rois: Roi[] = [
      {
        shape: atOrigin,
        operator: RoiOperator.AND,
        predicate: RoiPredicate.ANY_SEGMENT,
      },
      {
        shape: atTen,
        operator: RoiOperator.AND,
        predicate: RoiPredicate.EITHER_ENDPOINT,
      },
    ];
    // Crosses the origin and terminates at x=10.
    expect(streamlinePassesRois(line(-5, 0, 0, 5, 0, 0, 10, 0, 0), rois)).toBe(
      true,
    );
    // Crosses both but terminates beyond x=10.
    expect(streamlinePassesRois(line(-5, 0, 0, 15, 0, 0), rois)).toBe(false);
  });
});

describe("combineRoiVerdicts — agrees with streamlinePassesRois (no drift)", () => {
  const atOrigin = sphere(0, 0, 0, 1);
  const atTen = sphere(10, 0, 0, 1);
  const roi = (shape: RoiShape, operator: RoiOperator): Roi => ({
    shape,
    operator,
    predicate: RoiPredicate.ANY_SEGMENT,
  });

  const cases: Roi[][] = [
    [],
    [roi(atOrigin, RoiOperator.AND)],
    [roi(atOrigin, RoiOperator.AND), roi(atTen, RoiOperator.AND)],
    [roi(atOrigin, RoiOperator.AND), roi(atTen, RoiOperator.OR)],
    [roi(atOrigin, RoiOperator.AND), roi(atTen, RoiOperator.ANDNOT)],
    // Seed-position operators. The first region seeds the fold rather than
    // folding into it, so these are the cases that catch one fold being
    // updated without the other -- with only AND at index 0 above, changing
    // `streamlinePassesRois` alone would leave this suite green while the
    // per-chunk filter (which uses `combineRoiVerdicts`) kept the old
    // behaviour.
    [roi(atOrigin, RoiOperator.OR)],
    [roi(atOrigin, RoiOperator.ANDNOT)],
    [roi(atOrigin, RoiOperator.ANDNOT), roi(atTen, RoiOperator.AND)],
    [roi(atOrigin, RoiOperator.ANDNOT), roi(atTen, RoiOperator.OR)],
    [roi(atOrigin, RoiOperator.ANDNOT), roi(atTen, RoiOperator.ANDNOT)],
  ];
  const streams: StreamlineRef[] = [
    line(-5, 0, 0, 5, 0, 0), // crosses origin only
    line(5, 0, 0, 15, 0, 0), // crosses ten only
    line(-5, 0, 0, 15, 0, 0), // crosses both
    line(0, 5, 0, 0, 15, 0), // crosses neither
  ];

  it("folding a per-region crossing vector matches the geometry fold", () => {
    for (const rois of cases) {
      for (const s of streams) {
        const crossed = rois.map((r) =>
          streamlinePassesRoi(s, r.shape, r.predicate),
        );
        expect(combineRoiVerdicts(crossed, rois)).toBe(
          streamlinePassesRois(s, rois),
        );
      }
    }
  });

  it("OR-ing crossings across fragments recovers a spanning AND", () => {
    // The whole point: a streamline that crosses A in one fragment and B in
    // another passes `A AND B`, even though no single fragment crosses both.
    const rois = [roi(atOrigin, RoiOperator.AND), roi(atTen, RoiOperator.AND)];
    const fragmentA = line(-5, 0, 0, 5, 0, 0); // crosses origin, not ten
    const fragmentB = line(5, 0, 0, 15, 0, 0); // crosses ten, not origin
    const crossedA = rois.map((r) =>
      streamlinePassesRoi(fragmentA, r.shape, r.predicate),
    );
    const crossedB = rois.map((r) =>
      streamlinePassesRoi(fragmentB, r.shape, r.predicate),
    );
    const merged = crossedA.map((v, i) => v || crossedB[i]);
    expect(combineRoiVerdicts(merged, rois)).toBe(true);
    // Neither fragment alone passes the AND.
    expect(combineRoiVerdicts(crossedA, rois)).toBe(false);
    expect(combineRoiVerdicts(crossedB, rois)).toBe(false);
  });
});

describe("roiRegionBounds", () => {
  const f = (...xs: number[]) => Float32Array.from(xs);
  const mk = (shape: RoiShape, operator = RoiOperator.AND): Roi => ({
    shape,
    operator,
    predicate: RoiPredicate.ANY_SEGMENT,
  });

  it("encloses an ellipsoid", () => {
    const b = roiRegionBounds([
      mk({ kind: "ellipsoid", center: f(10, 10, 10), radii: f(2, 3, 4) }),
    ])!;
    expect(Array.from(b.lower)).toEqual([8, 7, 6]);
    expect(Array.from(b.upper)).toEqual([12, 13, 14]);
  });

  it("normalises an inverted box", () => {
    const b = roiRegionBounds([
      mk({ kind: "box", lower: f(9, 9, 9), upper: f(1, 1, 1) }),
    ])!;
    expect(Array.from(b.lower)).toEqual([1, 1, 1]);
    expect(Array.from(b.upper)).toEqual([9, 9, 9]);
  });

  it("unions several regions", () => {
    const b = roiRegionBounds([
      mk({ kind: "box", lower: f(0, 0, 0), upper: f(1, 1, 1) }),
      mk({ kind: "box", lower: f(5, 5, 5), upper: f(7, 7, 7) }),
    ])!;
    expect(Array.from(b.lower)).toEqual([0, 0, 0]);
    expect(Array.from(b.upper)).toEqual([7, 7, 7]);
  });

  it("includes exclusion regions", () => {
    // Unlike the exporter's bounds, which ask which tracts can pass: deciding
    // that a tract crosses an exclusion needs that region's geometry resident.
    const b = roiRegionBounds([
      mk({ kind: "box", lower: f(0, 0, 0), upper: f(1, 1, 1) }),
      mk(
        { kind: "box", lower: f(50, 50, 50), upper: f(60, 60, 60) },
        RoiOperator.ANDNOT,
      ),
    ])!;
    expect(Array.from(b.upper)).toEqual([60, 60, 60]);
  });

  it("is undefined for a halfspace, which has no finite extent", () => {
    expect(
      roiRegionBounds([
        mk({ kind: "halfspace", origin: f(0, 0, 0), normal: f(0, 0, 1) }),
      ]),
    ).toBeUndefined();
  });

  it("is undefined when any region is unbounded", () => {
    expect(
      roiRegionBounds([
        mk({ kind: "box", lower: f(0, 0, 0), upper: f(1, 1, 1) }),
        mk({ kind: "halfspace", origin: f(0, 0, 0), normal: f(1, 0, 0) }),
      ]),
    ).toBeUndefined();
  });

  it("is undefined for an empty list", () => {
    expect(roiRegionBounds([])).toBeUndefined();
  });
});
