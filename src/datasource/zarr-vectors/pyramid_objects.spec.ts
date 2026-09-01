/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";

import {
  computePerLevelObjectCount,
  computePyramidDensityScales,
} from "#src/datasource/zarr-vectors/pyramid_objects.js";

/** Finest-first level metadata, shaped like a real per-object pyramid. */
function level(
  vertexCount: number | undefined,
  objectSparsity: number | undefined,
  numObjects?: number,
) {
  return { vertexCount, objectSparsity, numObjects };
}

describe("computePerLevelObjectCount", () => {
  // hcp1065_whole_brain: five levels, sparsity 0.1 per step, and -- as the
  // format requires of a standalone level -- NO `inherited_num_objects` on
  // level 0. Measured live counts are 503085 / 50308 / 5031 / 503 / 50.
  const hcp = [
    level(102_067_827, 1.0),
    level(10_182_539, 0.1, 503_085),
    level(1_000_555, 0.1, 503_085),
    level(100_582, 0.1, 503_085),
    level(10_688, 0.1, 503_085),
  ];

  it("takes the base from level 1 when level 0 is standalone", () => {
    // The real-store case. Level 0 declares no `inherited_num_objects` because
    // it has no parent; level 1's names level 0's count.
    const counts = computePerLevelObjectCount(hcp).map((c) => Math.round(c!));
    expect(counts).toEqual([503_085, 50_309, 5_031, 503, 50]);
  });

  it("never degenerates to one object per vertex on that store", () => {
    // The regression this exists for: falling through to `vertexCount` makes
    // objectCount === vertexCount, which `bytesPerObjectFromLevelCounts` reads
    // as "unknown" and refuses -- disabling every per-object memory budget.
    const counts = computePerLevelObjectCount(hcp);
    for (let k = 0; k < hcp.length; ++k) {
      expect(counts[k]).not.toBe(hcp[k].vertexCount);
    }
  });

  it("prefers level 0's own count when it has one", () => {
    const counts = computePerLevelObjectCount([
      level(1000, 1.0, 800),
      level(100, 0.5, 999_999),
    ]);
    expect(counts).toEqual([800, 400]);
  });

  it("compounds sparsity down the chain, not against level 0 each time", () => {
    const counts = computePerLevelObjectCount([
      level(1000, 1.0),
      level(500, 0.5, 1000),
      level(250, 0.5, 1000),
      level(125, 0.5, 1000),
    ]);
    expect(counts).toEqual([1000, 500, 250, 125]);
  });

  it("floors at one object rather than reporting a fraction", () => {
    const counts = computePerLevelObjectCount([
      level(10, 1.0),
      level(1, 0.001, 10),
    ]);
    expect(counts).toEqual([10, 1]);
  });

  it("falls back to vertex counts when no level declares the field", () => {
    const counts = computePerLevelObjectCount([
      level(1000, 1.0),
      level(100, 0.1),
    ]);
    expect(counts).toEqual([1000, 100]);
  });

  it("falls back when the sparsity chain is incomplete", () => {
    // A partial chain must not be half-applied: a missing sparsity anywhere
    // would silently scale every level below it by the wrong factor.
    const counts = computePerLevelObjectCount([
      level(1000, 1.0),
      level(100, undefined, 900),
      level(10, 0.1),
    ]);
    expect(counts).toEqual([1000, 100, 10]);
  });

  it("handles a single-level store", () => {
    expect(computePerLevelObjectCount([level(1000, 1.0)])).toEqual([1000]);
    expect(computePerLevelObjectCount([])).toEqual([]);
  });
});

describe("computePyramidDensityScales", () => {
  it("spreads a point-cloud pyramid whose levels share one chunk_shape", () => {
    // The ABC MERFISH conversion: five levels, ~8x thinning each, all on the
    // same 0.5 mm grid. Without the correction every level reports 0.5 and the
    // picker cannot tell them apart.
    const vertexCounts = [3739946, 467493, 58436, 7304, 913];
    const scales = computePyramidDensityScales(
      new Array<number | undefined>(5).fill(undefined),
      vertexCounts,
      new Array<number>(5).fill(0.5),
    );
    // 8x fewer points reads as a grid 2x coarser.
    expect(scales[0]).toBeCloseTo(1, 2);
    expect(scales[1]).toBeCloseTo(2, 1);
    expect(scales[2]).toBeCloseTo(4, 1);
    expect(scales[3]).toBeCloseTo(8, 1);
    expect(scales[4]).toBeCloseTo(16, 0);
    // The point of the exercise: the effective spacings are all distinct.
    const effective = scales.map((s) => s * 0.5);
    expect(new Set(effective).size).toBe(5);
  });

  it("prefers object counts where the store has an object model", () => {
    // A tractogram: objects are the detail axis, and vertex counts would give a
    // different (here: no) correction. Objects must win.
    const scales = computePyramidDensityScales(
      [8000, 1000],
      [100, 100],
      [1, 1],
    );
    expect(scales[0]).toBeCloseTo(1, 6);
    expect(scales[1]).toBeCloseTo(2, 6);
  });

  it("does not double-count a pyramid that already coarsens its grid", () => {
    // 8x thinning with 2x chunk growth is fully expressed by the spacing.
    const scales = computePyramidDensityScales([8000, 1000], [], [1, 2]);
    expect(scales).toEqual([1, 1]);
  });

  it("makes no correction when a level's count is missing", () => {
    expect(
      computePyramidDensityScales([undefined, 1], [undefined, 1], [1, 1]),
    ).toEqual([1, 1]);
  });
});
