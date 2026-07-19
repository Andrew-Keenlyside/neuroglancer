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
  buildSpatialSkeletonGridLevels,
  getDefaultSpatiallyIndexedSkeletonChunkSize,
  getSpatialSkeletonGridSpacing,
  bytesPerObjectFromLevelCounts,
  objectBudgetFromBytes,
  selectSpatialSkeletonGridLevelByBudget,
  sortSpatialSkeletonGridSizes,
  type SpatialSkeletonGridSize,
} from "#src/skeleton/spatial_chunk_sizing.js";

describe("skeleton/spatial_chunk_sizing", () => {
  it("derives an isotropic chunk size that stays within the default chunk budget", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        lowerBounds: [5, 6, 7],
        upperBounds: [25, 66, 127],
      }),
    ).toEqual([15, 15, 15]);
  });

  it("handles elongated bounds while keeping the chunk size isotropic", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        lowerBounds: [0, 0, 0],
        upperBounds: [1000, 10, 10],
      }),
    ).toEqual([16, 16, 16]);
  });

  it("returns the minimum chunk size for tiny bounds", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        lowerBounds: [0, 0, 0],
        upperBounds: [2, 2, 2],
      }),
    ).toEqual([1, 1, 1]);
  });

  it("returns a chunk-size array with the same rank as the bounds", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        lowerBounds: [0, 0, 0, 0],
        upperBounds: [16, 32, 48, 2],
      }),
    ).toEqual([8, 8, 8, 8]);
  });

  it("supports overriding the chunk budget", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize(
        {
          lowerBounds: [0, 0, 0],
          upperBounds: [100, 100, 100],
        },
        { maxChunks: 8 },
      ),
    ).toEqual([50, 50, 50]);
  });

  it("rejects NaN bounds", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        lowerBounds: [Number.NaN, 0, 0],
        upperBounds: [10, 10, 10],
      }),
    ).toThrow(/bounds must be finite/i);
  });

  it("rejects infinite bounds", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        lowerBounds: [0, 0, 0],
        upperBounds: [Number.POSITIVE_INFINITY, 10, 10],
      }),
    ).toThrow(/bounds must be finite/i);
  });

  it("rejects mismatched lower/upper bound ranks", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        lowerBounds: [0, 0],
        upperBounds: [10, 10, 10],
      }),
    ).toThrow(/matching ranks/i);
  });

  it("rejects NaN minChunkSize", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize(
        {
          lowerBounds: [0, 0, 0],
          upperBounds: [10, 10, 10],
        },
        { minChunkSize: Number.NaN },
      ),
    ).toThrow(/minChunkSize must be finite/i);
  });

  it("rejects infinite maxChunks", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize(
        {
          lowerBounds: [0, 0, 0],
          upperBounds: [10, 10, 10],
        },
        { maxChunks: Number.POSITIVE_INFINITY },
      ),
    ).toThrow(/maxChunks must be finite/i);
  });
});

const size = (x: number, y = x, z = x): SpatialSkeletonGridSize => ({
  x,
  y,
  z,
});

describe("getSpatialSkeletonGridSpacing", () => {
  it("is the smallest axis", () => {
    expect(getSpatialSkeletonGridSpacing(size(30, 20, 40))).toBe(20);
  });
});

describe("sortSpatialSkeletonGridSizes", () => {
  it("orders coarsest first without mutating the input", () => {
    const input = [size(1), size(100), size(10)];
    expect(sortSpatialSkeletonGridSizes(input).map((s) => s.x)).toEqual([
      100, 10, 1,
    ]);
    expect(input.map((s) => s.x)).toEqual([1, 100, 10]);
  });
});

describe("buildSpatialSkeletonGridLevels", () => {
  it("keeps the caller's order rather than deriving one", () => {
    // Deliberately not sorted by spacing: the caller declares the pyramid, so
    // the order it gives is the order of the levels.
    const levels = buildSpatialSkeletonGridLevels([
      size(1),
      size(100),
      size(10),
    ]);
    expect(levels.map((l) => l.size.x)).toEqual([1, 100, 10]);
  });

  it("assigns lod as the normalised position, coarsest = 0", () => {
    const levels = buildSpatialSkeletonGridLevels([
      size(100),
      size(10),
      size(1),
    ]);
    expect(levels.map((l) => l.lod)).toEqual([0, 0.5, 1]);
  });

  it("gives a lone level lod 0 rather than dividing by zero", () => {
    expect(buildSpatialSkeletonGridLevels([size(5)])).toEqual([
      { size: size(5), lod: 0 },
    ]);
  });

  it("returns nothing for no levels", () => {
    expect(buildSpatialSkeletonGridLevels([])).toEqual([]);
  });

  it("separates levels that share a spacing, which a sort cannot", () => {
    // An object-sparsity pyramid: one chunk_shape at every level, differing
    // only in how many objects each holds. Sorting by spacing is a no-op
    // here, so the declared order is the only thing that can order them --
    // and it must, because each level's gridIndex is assigned from that
    // same order.
    const levels = buildSpatialSkeletonGridLevels([
      size(31),
      size(31),
      size(31),
    ]);
    expect(levels.map((l) => l.lod)).toEqual([0, 0.5, 1]);
  });

  it("composes with the sort for callers that do rank by spacing", () => {
    const levels = buildSpatialSkeletonGridLevels(
      sortSpatialSkeletonGridSizes([size(1), size(100), size(10)]),
    );
    expect(levels.map((l) => l.size.x)).toEqual([100, 10, 1]);
    expect(levels.map((l) => l.lod)).toEqual([0, 0.5, 1]);
  });
});

describe("selectSpatialSkeletonGridLevelByBudget", () => {
  // Costs are parallel to the level list: coarsest (cheapest) first.
  const costs = [0.12e6, 1.2e6, 12e6, 122e6, 1224e6]; // ~HCP1065, bytes

  it("takes the finest level that fits", () => {
    expect(selectSpatialSkeletonGridLevelByBudget(costs, 1e9)).toBe(3);
  });

  it("takes the finest level outright when everything fits", () => {
    expect(selectSpatialSkeletonGridLevelByBudget(costs, 1e12)).toBe(4);
  });

  it("falls back to the coarsest rather than showing nothing", () => {
    // Even the sparsest level is over budget: sparse data beats no data.
    expect(selectSpatialSkeletonGridLevelByBudget(costs, 1)).toBe(0);
  });

  it("treats a level exactly on budget as affordable", () => {
    expect(selectSpatialSkeletonGridLevelByBudget([10, 20, 30], 20)).toBe(1);
  });

  it("does not assume an unknown cost fits", () => {
    // A level whose footprint could not be estimated must not be selected on
    // the strength of not having a number.
    expect(
      selectSpatialSkeletonGridLevelByBudget([10, Number.NaN, 30], 1e9),
    ).toBe(2);
    expect(selectSpatialSkeletonGridLevelByBudget([10, Number.NaN], 1e9)).toBe(
      0,
    );
  });

  it("returns 0 for no levels", () => {
    expect(selectSpatialSkeletonGridLevelByBudget([], 1e9)).toBe(0);
  });

  it("picks the finest affordable level even if costs are not monotonic", () => {
    // Ordering is the caller's contract; do not silently re-rank.
    expect(selectSpatialSkeletonGridLevelByBudget([10, 5000, 20], 100)).toBe(2);
  });
});

describe("buildSpatialSkeletonGridLevels — object counts", () => {
  const sizes = [
    { x: 8, y: 8, z: 8 },
    { x: 4, y: 4, z: 4 },
    { x: 2, y: 2, z: 2 },
  ];

  it("attaches per-level object counts positionally", () => {
    const levels = buildSpatialSkeletonGridLevels(sizes, [50, 5000, 503000]);
    expect(levels.map((l) => l.objectCount)).toEqual([50, 5000, 503000]);
  });

  it("omits the count where the writer did not stamp one", () => {
    const levels = buildSpatialSkeletonGridLevels(sizes, [
      50,
      undefined,
      503000,
    ]);
    expect(levels[0].objectCount).toBe(50);
    expect("objectCount" in levels[1]).toBe(false);
    expect(levels[2].objectCount).toBe(503000);
  });

  it("omits zero and non-finite counts, which would size a bar as real", () => {
    const levels = buildSpatialSkeletonGridLevels(sizes, [0, Number.NaN, 7]);
    expect("objectCount" in levels[0]).toBe(false);
    expect("objectCount" in levels[1]).toBe(false);
    expect(levels[2].objectCount).toBe(7);
  });

  it("is unchanged when no counts are supplied", () => {
    const levels = buildSpatialSkeletonGridLevels(sizes);
    expect(levels).toHaveLength(3);
    expect(levels.every((l) => l.objectCount === undefined)).toBe(true);
    expect(levels.map((l) => l.lod)).toEqual([0, 0.5, 1]);
  });
});

describe("bytesPerObjectFromLevelCounts", () => {
  it("scales vertices-per-object by the per-vertex cost", () => {
    // 200 vertices per object at 40 bytes each.
    expect(bytesPerObjectFromLevelCounts(103369, 503, 40)).toBeCloseTo(
      (103369 / 503) * 40,
    );
  });

  it("declines when the counts are equal", () => {
    // A pyramid without per-level object counts substitutes the vertex counts,
    // so the ratio degenerates to 1.0 vertex per object -- a budget wrong by two
    // orders of magnitude on a tractogram. Better to decline than to fabricate.
    expect(bytesPerObjectFromLevelCounts(5000, 5000, 40)).toBeUndefined();
  });

  it("declines on missing, zero or non-finite inputs", () => {
    expect(bytesPerObjectFromLevelCounts(undefined, 10, 40)).toBeUndefined();
    expect(bytesPerObjectFromLevelCounts(1000, undefined, 40)).toBeUndefined();
    expect(bytesPerObjectFromLevelCounts(1000, 0, 40)).toBeUndefined();
    expect(bytesPerObjectFromLevelCounts(0, 10, 40)).toBeUndefined();
    expect(bytesPerObjectFromLevelCounts(1000, 10, 0)).toBeUndefined();
    expect(bytesPerObjectFromLevelCounts(Number.NaN, 10, 40)).toBeUndefined();
  });
});

describe("objectBudgetFromBytes", () => {
  it("floors to whole objects", () => {
    expect(objectBudgetFromBytes(1000, 300)).toBe(3);
  });

  it("is 0 when it cannot afford one", () => {
    expect(objectBudgetFromBytes(100, 300)).toBe(0);
  });

  it("is 0 rather than negative or infinite on bad input", () => {
    expect(objectBudgetFromBytes(-5, 300)).toBe(0);
    expect(objectBudgetFromBytes(1000, 0)).toBe(0);
    expect(objectBudgetFromBytes(Number.NaN, 300)).toBe(0);
  });

  it("converts a real half-share of a 1 GB pool", () => {
    // 500 MB at ~8.1 kB per tract: a few tens of thousands of streamlines,
    // which is the order the default manual budget assumed.
    const perObject = bytesPerObjectFromLevelCounts(103369, 503, 40)!;
    const budget = objectBudgetFromBytes(0.5e9, perObject);
    expect(budget).toBeGreaterThan(10_000);
    expect(budget).toBeLessThan(200_000);
  });
});
