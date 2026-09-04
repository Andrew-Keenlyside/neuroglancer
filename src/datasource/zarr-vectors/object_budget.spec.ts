/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import type { SkeletonChunk } from "#src/datasource/zarr-vectors/geometry_chunk.js";
import {
  admissionForBudget,
  ADMIT_ALL,
  admitsObject,
  computePerLevelObjectCount,
  computePyramidDensityScales,
  filterChunkByAdmittedObjects,
  levelsAreNested,
  OBJECT_DEPTH_ABSENT,
  objectDepths,
  objectRank,
} from "#src/datasource/zarr-vectors/object_budget.js";

// ---------------------------------------------------------------- pyramid_objects

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

// ---------------------------------------------------------------- object_admission

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

// ---------------------------------------------------------------- object_filter

/**
 * A chunk holding `objects.length` polylines laid end to end, each of
 * `perObject` vertices, with one explicit fragment per object.
 */
function makeChunk(objects: number[], perObject = 3): SkeletonChunk {
  const numVertices = objects.length * perObject;
  const positions = new Float32Array(numVertices * 3);
  const segmentIds = new Uint32Array(numVertices * 2);
  const tangents = new Float32Array(numVertices * 3);
  const attr = new Float32Array(numVertices);
  const edges: number[] = [];
  const offsets: number[] = [0];
  const rows: bigint[] = [];
  for (let o = 0; o < objects.length; ++o) {
    for (let i = 0; i < perObject; ++i) {
      const v = o * perObject + i;
      positions[v * 3] = v;
      positions[v * 3 + 1] = v * 2;
      positions[v * 3 + 2] = v * 3;
      segmentIds[v * 2] = objects[o];
      tangents[v * 3] = o + 1;
      attr[v] = v * 10;
      rows.push(BigInt(v));
      if (i > 0) edges.push(v - 1, v);
    }
    offsets.push(rows.length);
  }
  return {
    rank: 3,
    numVertices,
    positions,
    segmentIdsAreGlobal: true,
    numEdges: edges.length / 2,
    edges: Uint32Array.from(edges),
    tangents,
    vertexAttributes: [attr],
    segmentIds,
    fragmentIndex: new FragmentIndex(
      objects.length,
      new Uint8Array((objects.length + 7) >> 3),
      new BigInt64Array(0),
      Uint32Array.from(offsets),
      BigInt64Array.from(rows),
    ),
  };
}

/** Every object id the fragment index attributes geometry to. */
function objectsInIndex(chunk: SkeletonChunk): number[] {
  const out: number[] = [];
  for (let f = 0; f < chunk.fragmentIndex.numFragments; ++f) {
    const rows = chunk.fragmentIndex.indices(f);
    expect(rows.length).toBeGreaterThan(0);
    out.push(chunk.segmentIds![rows[0] * 2]);
  }
  return out;
}

describe("filterChunkByAdmittedObjects", () => {
  it("returns the very same chunk when everything is admitted", () => {
    // Not merely equal — identical, so the overwhelmingly common case costs one
    // pass and no allocation.
    const chunk = makeChunk([7, 8, 9]);
    expect(filterChunkByAdmittedObjects(chunk, () => true)).toBe(chunk);
  });

  it("is inert for a store with no segment column", () => {
    const chunk = { ...makeChunk([7]), segmentIds: undefined };
    expect(filterChunkByAdmittedObjects(chunk, () => false)).toBe(chunk);
  });

  it("refuses to act on chunk-local stand-in ids", () => {
    // The decoder always populates a segment column for an object-model kind,
    // but substitutes the fragment's index WITHIN THE CHUNK when
    // `fragment_attributes/segment_id` is missing. Rationing on those would give
    // one tract a different id in every cell it crosses, keeping it here and
    // dropping it next door -- shattering tracts instead of thinning them.
    const chunk = { ...makeChunk([7, 8]), segmentIdsAreGlobal: false };
    expect(filterChunkByAdmittedObjects(chunk, () => false)).toBe(chunk);
    const unset = { ...makeChunk([7, 8]), segmentIdsAreGlobal: undefined };
    expect(filterChunkByAdmittedObjects(unset, () => false)).toBe(unset);
  });

  it("keeps range fragments as ranges", () => {
    // Not cosmetic: this index is retained for the ROI filter and charged to
    // the system-memory budget, and an all-explicit rebuild costs 8 bytes per
    // vertex against 16 per fragment.
    const chunk = makeChunk([7, 8, 9], 50);
    const filtered = filterChunkByAdmittedObjects(chunk, (low) => low !== 8);
    expect(filtered.fragmentIndex.numFragments).toBe(2);
    for (let f = 0; f < 2; ++f) {
      expect(filtered.fragmentIndex.isRange(f)).toBe(true);
    }
    expect(filtered.fragmentIndex.byteLength).toBeLessThan(
      filtered.numVertices * 8,
    );
    expect(objectsInIndex(filtered)).toEqual([7, 9]);
  });

  it("keeps only the admitted objects' vertices", () => {
    const chunk = makeChunk([7, 8, 9]);
    const filtered = filterChunkByAdmittedObjects(chunk, (low) => low === 8);
    expect(filtered.numVertices).toBe(3);
    for (let v = 0; v < filtered.numVertices; ++v) {
      expect(filtered.segmentIds![v * 2]).toBe(8);
    }
    // Object 8's vertices were 3,4,5 in the source.
    expect(Array.from(filtered.positions.slice(0, 3))).toEqual([3, 6, 9]);
  });

  it("remaps edges and leaves none dangling", () => {
    const chunk = makeChunk([7, 8, 9]);
    const filtered = filterChunkByAdmittedObjects(chunk, (low) => low !== 8);
    expect(filtered.numEdges).toBe(4); // two objects, two edges each
    for (const v of filtered.edges) {
      expect(v).toBeLessThan(filtered.numVertices);
    }
    expect(filtered.edges.length).toBe(filtered.numEdges * 2);
  });

  it("carries tangents and attributes through the same remap", () => {
    const chunk = makeChunk([7, 8, 9]);
    const filtered = filterChunkByAdmittedObjects(chunk, (low) => low === 9);
    // Object 9 is the third, so tangent marker 3 and attributes 60/70/80.
    expect(Array.from(filtered.tangents!.slice(0, 3))).toEqual([3, 0, 0]);
    expect(Array.from(filtered.vertexAttributes[0] as Float32Array)).toEqual([
      60, 70, 80,
    ]);
  });

  it("rebuilds a fragment index that addresses the filtered vertices", () => {
    // The sharpest hazard: leaving the ROI filter row indices into a numbering
    // that no longer exists would silently attribute one tract's geometry to
    // another.
    const chunk = makeChunk([7, 8, 9, 10]);
    const filtered = filterChunkByAdmittedObjects(
      chunk,
      (low) => low === 8 || low === 10,
    );
    expect(filtered.fragmentIndex.numFragments).toBe(2);
    expect(objectsInIndex(filtered)).toEqual([8, 10]);
    for (let f = 0; f < filtered.fragmentIndex.numFragments; ++f) {
      for (const row of filtered.fragmentIndex.indices(f)) {
        expect(row).toBeLessThan(filtered.numVertices);
      }
    }
  });

  it("keeps a ghost vertex with the tract it bridges", () => {
    // A ghost inherits its host's segment id (`appendGhostVertices`), so a
    // cross-chunk bridge is kept or dropped together with its tract — that is
    // what keeps admitted tracts continuous across cell boundaries.
    const chunk = makeChunk([7, 8]);
    const withGhost: SkeletonChunk = {
      ...chunk,
      numVertices: chunk.numVertices + 1,
      positions: Float32Array.of(...chunk.positions, 99, 99, 99),
      segmentIds: Uint32Array.of(...chunk.segmentIds!, 8, 0),
      tangents: Float32Array.of(...chunk.tangents!, 0, 0, 1),
      vertexAttributes: [
        Float32Array.of(...(chunk.vertexAttributes[0] as Float32Array), 0),
      ],
      edges: Uint32Array.of(...chunk.edges, 5, chunk.numVertices),
      numEdges: chunk.numEdges + 1,
    };
    const filtered = filterChunkByAdmittedObjects(
      withGhost,
      (low) => low === 8,
    );
    expect(filtered.numVertices).toBe(4); // object 8's three, plus the ghost
    expect(Array.from(filtered.positions.slice(9, 12))).toEqual([99, 99, 99]);
    // The bridge edge survived, remapped.
    expect(filtered.numEdges).toBe(3);
  });

  it("drops a ghost whose tract was not admitted", () => {
    const chunk = makeChunk([7, 8]);
    const withGhost: SkeletonChunk = {
      ...chunk,
      numVertices: chunk.numVertices + 1,
      positions: Float32Array.of(...chunk.positions, 99, 99, 99),
      segmentIds: Uint32Array.of(...chunk.segmentIds!, 8, 0),
      tangents: Float32Array.of(...chunk.tangents!, 0, 0, 1),
      vertexAttributes: [
        Float32Array.of(...(chunk.vertexAttributes[0] as Float32Array), 0),
      ],
      edges: Uint32Array.of(...chunk.edges, 5, chunk.numVertices),
      numEdges: chunk.numEdges + 1,
    };
    const filtered = filterChunkByAdmittedObjects(
      withGhost,
      (low) => low === 7,
    );
    expect(filtered.numVertices).toBe(3);
    for (const v of filtered.edges) expect(v).toBeLessThan(3);
  });

  it("distinguishes ids that differ only in the high word", () => {
    const chunk = makeChunk([5, 5]);
    chunk.segmentIds![1] = 1; // first vertex of object A gets high word 1
    for (let v = 0; v < 3; ++v) chunk.segmentIds![v * 2 + 1] = 1;
    const filtered = filterChunkByAdmittedObjects(
      chunk,
      (_low, high) => high === 1,
    );
    expect(filtered.numVertices).toBe(3);
  });

  it("filters faces as a unit for surface geometry", () => {
    const chunk = makeChunk([7, 8], 3);
    const withFaces: SkeletonChunk = {
      ...chunk,
      faces: Uint32Array.of(0, 1, 2, 3, 4, 5),
      numFaces: 2,
    };
    const filtered = filterChunkByAdmittedObjects(
      withFaces,
      (low) => low === 8,
    );
    expect(filtered.numFaces).toBe(1);
    expect(Array.from(filtered.faces!)).toEqual([0, 1, 2]);
  });

  it("keeps a mixed range/explicit index addressable", () => {
    // A fragment whose source rows are not contiguous (a branch point sharing a
    // vertex, say) must stay explicit while its neighbours stay ranges. The two
    // kinds are indexed by different counters inside FragmentIndex -- popcount
    // prefix for ranges, fragment-minus-popcount for explicit -- so getting the
    // push order wrong silently returns another fragment's rows.
    const chunk = makeChunk([7, 8, 9], 4);
    // Rebuild object 8's fragment as an explicit, interleaved row list.
    const interleaved: SkeletonChunk = {
      ...chunk,
      fragmentIndex: new FragmentIndex(
        3,
        Uint8Array.of(0b101), // fragments 0 and 2 are ranges, 1 is explicit
        BigInt64Array.of(0n, 4n, 8n, 4n),
        Uint32Array.of(0, 4),
        BigInt64Array.of(7n, 5n, 6n, 4n),
      ),
    };
    const filtered = filterChunkByAdmittedObjects(
      interleaved,
      (low) => low !== 7,
    );
    expect(filtered.fragmentIndex.numFragments).toBe(2);
    const [first, second] = [0, 1].map((f) =>
      Array.from(filtered.fragmentIndex.indices(f)),
    );
    // Object 8's four vertices remap to 0..3, in their declared (interleaved)
    // order; object 9's stay a contiguous range at 4..7.
    expect(first).toEqual([3, 1, 2, 0]);
    expect(filtered.fragmentIndex.isRange(0)).toBe(false);
    expect(second).toEqual([4, 5, 6, 7]);
    expect(filtered.fragmentIndex.isRange(1)).toBe(true);
  });

  it("survives admitting nothing at all", () => {
    const filtered = filterChunkByAdmittedObjects(
      makeChunk([7, 8]),
      () => false,
    );
    expect(filtered.numVertices).toBe(0);
    expect(filtered.numEdges).toBe(0);
    expect(filtered.fragmentIndex.numFragments).toBe(0);
  });
});
