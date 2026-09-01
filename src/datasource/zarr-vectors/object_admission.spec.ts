/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";

import {
  ADMIT_ALL,
  admissionForBudget,
  admitsObject,
  levelsAreNested,
  OBJECT_DEPTH_ABSENT,
  objectDepths,
  objectRank,
} from "#src/datasource/zarr-vectors/object_admission.js";

/**
 * A writer's own survivor selection, deliberately INDEPENDENT of
 * {@link objectRank}. A real store picks survivors from its own seeded RNG, so
 * a fixture that reused the reader's hash would be testing a coincidence.
 */
function writerRank(id: number): number {
  let x = (id ^ 0x9e3779b9) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x85ebca6b) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

/**
 * A nested object-sparsity pyramid shaped like a real tractogram: `sparsity`
 * survivors per step, a constant vertex count per object, and every coarser
 * level a strict subset of the finer one.
 */
function makePyramid(
  numObjects: number,
  numLevels: number,
  sparsity: number,
  verticesPerObject = 200,
  rank: (id: number) => number = writerRank,
) {
  const levels: Uint32Array[] = [];
  let cutoff = 1;
  for (let k = 0; k < numLevels; ++k) {
    const counts = new Uint32Array(numObjects);
    for (let id = 0; id < numObjects; ++id) {
      if (rank(id) < cutoff) counts[id] = verticesPerObject;
    }
    levels.push(counts);
    cutoff *= sparsity;
  }
  return levels;
}

const BPV = 48;

describe("objectRank", () => {
  it("is stable", () => {
    expect(objectRank(12345)).toBe(objectRank(12345));
  });

  it("is uniform enough to sample a population evenly", () => {
    const buckets = new Array(10).fill(0);
    for (let id = 0; id < 100_000; ++id) {
      buckets[Math.min(9, Math.floor(objectRank(id) * 10))]++;
    }
    for (const b of buckets) {
      expect(b).toBeGreaterThan(9_000);
      expect(b).toBeLessThan(11_000);
    }
  });

  it("scatters neighbouring ids", () => {
    // Consecutive ids are consecutive in the file, which for a tractogram means
    // one bundle or seed region. A weak hash would sample one lobe of the brain.
    const consecutive = [0, 1, 2, 3, 4, 5, 6, 7].map(objectRank);
    const spread = Math.max(...consecutive) - Math.min(...consecutive);
    expect(spread).toBeGreaterThan(0.5);
  });

  it("stays in [0, 1)", () => {
    for (const id of [0, 1, 7, 1023, 65_535, 503_084, 4_294_967_295]) {
      expect(objectRank(id)).toBeGreaterThanOrEqual(0);
      expect(objectRank(id)).toBeLessThan(1);
    }
  });
});

describe("objectDepths / levelsAreNested", () => {
  it("reports the coarsest level holding each object", () => {
    //             id: 0  1  2  3
    const levels = [
      Uint32Array.of(5, 5, 5, 5), // level 0: all
      Uint32Array.of(5, 5, 0, 0), // level 1
      Uint32Array.of(5, 0, 0, 0), // level 2 (coarsest)
    ];
    expect(Array.from(objectDepths(levels))).toEqual([2, 1, 0, 0]);
  });

  it("marks an object present nowhere", () => {
    const levels = [Uint32Array.of(5, 0), Uint32Array.of(5, 0)];
    expect(objectDepths(levels)[1]).toBe(OBJECT_DEPTH_ABSENT);
  });

  it("accepts a nested pyramid and rejects a broken one", () => {
    expect(
      levelsAreNested([
        Uint32Array.of(5, 5, 5),
        Uint32Array.of(5, 5, 0),
        Uint32Array.of(5, 0, 0),
      ]),
    ).toBe(true);
    // Object 2 exists at the coarse level but not the fine one — the depth
    // model would silently drop it, so this must be detected at load.
    expect(
      levelsAreNested([Uint32Array.of(5, 5, 0), Uint32Array.of(5, 0, 5)]),
    ).toBe(false);
  });
});

describe("admissionForBudget", () => {
  // 50k objects, 4 levels at 10x sparsity: 50000 / 5000 / 500 / 50.
  const levels = makePyramid(50_000, 4, 0.1);
  const depths = objectDepths(levels);
  const cost = (n: number) => n * 200 * BPV; // n objects, 200 vtx each

  it("takes the whole finest level when everything fits", () => {
    expect(admissionForBudget(levels, depths, BPV, cost(60_000))).toEqual(
      ADMIT_ALL,
    );
  });

  it("fills the gap between two rungs instead of dropping to the coarser one", () => {
    // The regression this module exists for. A budget for ~20k objects used to
    // draw the 5k rung and leave three quarters of the memory unused.
    const { loadLevel, fraction } = admissionForBudget(
      levels,
      depths,
      BPV,
      cost(20_000),
    );
    expect(loadLevel).toBe(0);
    const admitted = countAdmitted(levels, depths, { loadLevel, fraction });
    expect(admitted).toBeGreaterThan(19_000);
    expect(admitted).toBeLessThan(21_000);
  });

  it("keeps the coarse backbone whole while rationing the new objects", () => {
    const admission = admissionForBudget(levels, depths, BPV, cost(20_000));
    // Everything the 5k rung would have shown is still shown.
    for (let id = 0; id < 50_000; ++id) {
      if (depths[id] >= 1) {
        expect(admitsObject(id, depths[id], admission)).toBe(true);
      }
    }
  });

  it("spends a bigger budget on proportionally more objects", () => {
    const small = admissionForBudget(levels, depths, BPV, cost(10_000));
    const large = admissionForBudget(levels, depths, BPV, cost(30_000));
    expect(countAdmitted(levels, depths, large)).toBeGreaterThan(
      countAdmitted(levels, depths, small) * 2,
    );
  });

  it("grows the admitted set monotonically, never reshuffling it", () => {
    // Residency follows this set: a bigger budget must ADD to what is already
    // loaded, or every budget change would evict and refetch a different sample.
    const small = admissionForBudget(levels, depths, BPV, cost(8_000));
    const large = admissionForBudget(levels, depths, BPV, cost(25_000));
    expect(large.loadLevel).toBe(small.loadLevel);
    for (let id = 0; id < 50_000; ++id) {
      if (admitsObject(id, depths[id], small)) {
        expect(admitsObject(id, depths[id], large)).toBe(true);
      }
    }
  });

  it("takes a rung whole when it happens to fit exactly", () => {
    const admission = admissionForBudget(levels, depths, BPV, cost(5_000));
    expect(countAdmitted(levels, depths, admission)).toBe(5_000);
  });

  it("draws part of the coarsest level when even that will not fit", () => {
    const wholeCoarsest = countLive(levels[3]);
    const admission = admissionForBudget(levels, depths, BPV, cost(20));
    expect(admission.loadLevel).toBe(3);
    const admitted = countAdmitted(levels, depths, admission);
    expect(admitted).toBeGreaterThan(0);
    expect(admitted).toBeLessThan(wholeCoarsest);
    expect(admitted).toBeLessThanOrEqual(20);
  });

  it("is correct even when a depth class occupies a narrow rank band", () => {
    // The degenerate case the bisection exists for: a writer whose survivor
    // choice happens to track the reader's hash leaves every object of one
    // depth crowded into a sliver of [0, 1). A proportional estimate would then
    // admit the whole class or none of it.
    const correlated = makePyramid(20_000, 3, 0.1, 200, objectRank);
    const correlatedDepths = objectDepths(correlated);
    const admission = admissionForBudget(
      correlated,
      correlatedDepths,
      BPV,
      cost(800),
    );
    const admitted = countAdmitted(correlated, correlatedDepths, admission);
    expect(admitted).toBeGreaterThan(700);
    expect(admitted).toBeLessThanOrEqual(800);
  });

  it("still draws the coarsest level for a zero or negative budget", () => {
    // Sparse beats blank: a budget of nothing must not report an empty view as
    // a successful answer.
    for (const budget of [0, -1]) {
      const admission = admissionForBudget(levels, depths, BPV, budget);
      expect(admission.loadLevel).toBe(3);
      expect(countAdmitted(levels, depths, admission)).toBe(
        countLive(levels[3]),
      );
    }
  });

  it("falls back to admitting everything without usable inputs", () => {
    expect(admissionForBudget([], depths, BPV, 1e9)).toEqual(ADMIT_ALL);
    expect(admissionForBudget(levels, depths, 0, 1e9)).toEqual(ADMIT_ALL);
  });

  it("never overruns the byte budget, and gets close to it", () => {
    for (const objects of [12_000, 20_000, 35_000]) {
      const budget = cost(objects);
      const admission = admissionForBudget(levels, depths, BPV, budget);
      const spent = countAdmitted(levels, depths, admission) * 200 * BPV;
      expect(spent).toBeLessThanOrEqual(budget);
      expect(spent).toBeGreaterThan(budget * 0.99);
    }
  });
});

describe("admitsObject", () => {
  const admission = { loadLevel: 1, fraction: 0.5 };

  it("always keeps objects that survive into a coarser level", () => {
    expect(admitsObject(0, 2, admission)).toBe(true);
    expect(admitsObject(1, 3, admission)).toBe(true);
  });

  it("refuses objects the loaded level does not contain", () => {
    expect(admitsObject(0, 0, admission)).toBe(false);
    expect(admitsObject(0, OBJECT_DEPTH_ABSENT, admission)).toBe(false);
  });

  it("rations objects new at the loaded level by rank", () => {
    let admitted = 0;
    for (let id = 0; id < 10_000; ++id) {
      if (admitsObject(id, 1, admission)) admitted++;
    }
    expect(admitted).toBeGreaterThan(4_800);
    expect(admitted).toBeLessThan(5_200);
  });

  it("short-circuits the degenerate fractions", () => {
    expect(admitsObject(7, 1, { loadLevel: 1, fraction: 1 })).toBe(true);
    expect(admitsObject(7, 1, { loadLevel: 1, fraction: 0 })).toBe(false);
  });
});

/** Objects present at a level. */
function countLive(counts: Uint32Array): number {
  let n = 0;
  for (const c of counts) if (c !== 0) n++;
  return n;
}

/** How many objects `admission` draws out of the whole id space. */
function countAdmitted(
  levels: ReadonlyArray<Uint32Array>,
  depths: Uint8Array,
  admission: { loadLevel: number; fraction: number },
): number {
  let n = 0;
  for (let id = 0; id < levels[0].length; ++id) {
    if (admitsObject(id, depths[id], admission)) n++;
  }
  return n;
}
