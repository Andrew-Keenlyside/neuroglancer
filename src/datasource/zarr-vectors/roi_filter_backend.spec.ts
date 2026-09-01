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
import { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import {
  combineRoiVerdicts,
  RoiOperator,
  RoiPredicate,
  type Roi,
  type RoiGroupConfig,
  type RoiShape,
} from "#src/datasource/zarr-vectors/roi.js";
import {
  computeChunkCrossings,
  computeGroupedPassingSet,
  computePassingSet,
  computePerGroupPassingSets,
  computeVertexAttrPassingSet,
  diffPassingSet,
  RoiFilterAccumulator,
  type RoiFilterableChunk,
} from "#src/datasource/zarr-vectors/roi_filter_backend.js";

// --- fixture builders --------------------------------------------------------

/** FragmentIndex of contiguous `[start, count)` range fragments. */
function rangeIndex(ranges: [number, number][]): FragmentIndex {
  const n = ranges.length;
  const bitmap = new Uint8Array(Math.ceil(n / 8));
  for (let i = 0; i < n; ++i) bitmap[i >> 3] |= 1 << (i & 7); // all ranges
  const rangeTable = BigInt64Array.from(
    ranges.flatMap(([s, c]) => [BigInt(s), BigInt(c)]),
  );
  return new FragmentIndex(
    n,
    bitmap,
    rangeTable,
    new Uint32Array([0]),
    new BigInt64Array(0),
  );
}

/** FragmentIndex of explicit (non-contiguous) index-list fragments. */
function explicitIndex(fragments: number[][]): FragmentIndex {
  const n = fragments.length;
  const bitmap = new Uint8Array(Math.ceil(n / 8)); // all zero -> all explicit
  const csrOffsets = new Uint32Array(n + 1);
  let total = 0;
  for (let i = 0; i < n; ++i) {
    csrOffsets[i] = total;
    total += fragments[i].length;
  }
  csrOffsets[n] = total;
  const csrIndices = BigInt64Array.from(fragments.flat().map((x) => BigInt(x)));
  return new FragmentIndex(
    n,
    new Uint8Array(bitmap),
    new BigInt64Array(0),
    csrOffsets,
    csrIndices,
  );
}

/** Build a uvec2 segment column assigning each vertex an object id. */
function segColumn(ids: bigint[]): Uint32Array {
  const out = new Uint32Array(ids.length * 2);
  for (let v = 0; v < ids.length; ++v) {
    out[v * 2] = Number(ids[v] & 0xffffffffn);
    out[v * 2 + 1] = Number((ids[v] >> 32n) & 0xffffffffn);
  }
  return out;
}

const sphere = (cx: number, cy: number, cz: number, r: number): RoiShape => ({
  kind: "ellipsoid",
  center: Float32Array.from([cx, cy, cz]),
  radii: Float32Array.from([r, r, r]),
});
const roi = (shape: RoiShape, operator: RoiOperator): Roi => ({
  shape,
  operator,
  predicate: RoiPredicate.ANY_SEGMENT,
});

/**
 * A point-cloud chunk: one spatial BIN holding unrelated points, each its own
 * object, optionally carrying per-vertex attribute columns.
 */
function pointChunk(
  points: [number, number, number][],
  ids: bigint[],
  vertexAttributes?: Map<string, Float32Array>,
): RoiFilterableChunk {
  return {
    rank: 3,
    numVertices: points.length,
    positions: Float32Array.from(points.flat()),
    segmentIds: segColumn(ids),
    // One fragment for the whole bin, which is exactly the shape that makes the
    // per-fragment fold wrong for this kind.
    fragmentIndex: rangeIndex([[0, points.length]]),
    perVertexObjects: true,
    ...(vertexAttributes === undefined ? {} : { vertexAttributes }),
  };
}

describe("computeChunkCrossings", () => {
  it("marks an object whose fragment crosses the ROI", () => {
    // One object (id 42), one range fragment crossing a sphere at the origin.
    const chunk: RoiFilterableChunk = {
      rank: 3,
      numVertices: 2,
      positions: Float32Array.from([-5, 0, 0, 5, 0, 0]),
      segmentIds: segColumn([42n, 42n]),
      fragmentIndex: rangeIndex([[0, 2]]),
    };
    const crossings = computeChunkCrossings(chunk, [
      roi(sphere(0, 0, 0, 1), RoiOperator.AND),
    ]);
    expect(crossings.get(42n)).toEqual([true]);
  });

  it("OR-s crossings across an object's fragments (spanning AND)", () => {
    // Object 42 has two fragments: one crosses ROI-A (origin), the other
    // crosses ROI-B (x=10). Neither fragment crosses both, but the object does.
    const rois = [
      roi(sphere(0, 0, 0, 1), RoiOperator.AND),
      roi(sphere(10, 0, 0, 1), RoiOperator.AND),
    ];
    const chunk: RoiFilterableChunk = {
      rank: 3,
      numVertices: 5,
      // frag0: verts 0-1 cross origin; frag1: verts 2-3 cross x=10; vert 4 = object 7.
      positions: Float32Array.from([
        -5, 0, 0, 5, 0, 0, /* frag0 */ 5, 0, 0, 15, 0, 0, /* frag1 */ 100, 0,
        0 /* obj7 */,
      ]),
      segmentIds: segColumn([42n, 42n, 42n, 42n, 7n]),
      fragmentIndex: rangeIndex([
        [0, 2],
        [2, 2],
        [4, 1],
      ]),
    };
    const crossings = computeChunkCrossings(chunk, rois);
    expect(crossings.get(42n)).toEqual([true, true]);
    // Fold: object 42 passes the AND; object 7 crosses neither.
    expect(combineRoiVerdicts(crossings.get(42n)!, rois)).toBe(true);
    expect(combineRoiVerdicts(crossings.get(7n)!, rois)).toBe(false);
  });

  it("gathers an explicit (non-contiguous) fragment before testing", () => {
    // Object 9's fragment is vertices [0, 2] (skipping vertex 1); those two
    // form a segment crossing the origin sphere. Vertex 1 sits far away and
    // must NOT be part of the tested polyline.
    const chunk: RoiFilterableChunk = {
      rank: 3,
      numVertices: 3,
      positions: Float32Array.from([-5, 0, 0, /* v1 far */ 0, 500, 0, 5, 0, 0]),
      segmentIds: segColumn([9n, 9n, 9n]),
      fragmentIndex: explicitIndex([[0, 2]]),
    };
    const crossings = computeChunkCrossings(chunk, [
      roi(sphere(0, 0, 0, 1), RoiOperator.AND),
    ]);
    // The gathered polyline (-5,0,0)->(5,0,0) crosses the origin sphere.
    expect(crossings.get(9n)).toEqual([true]);
  });

  it("returns an empty map when there is no segment column", () => {
    const chunk: RoiFilterableChunk = {
      rank: 3,
      numVertices: 2,
      positions: Float32Array.from([-5, 0, 0, 5, 0, 0]),
      fragmentIndex: rangeIndex([[0, 2]]),
    };
    expect(
      computeChunkCrossings(chunk, [roi(sphere(0, 0, 0, 1), RoiOperator.AND)])
        .size,
    ).toBe(0);
  });

  it("returns an empty map when there are no ROIs", () => {
    const chunk: RoiFilterableChunk = {
      rank: 3,
      numVertices: 2,
      positions: Float32Array.from([-5, 0, 0, 5, 0, 0]),
      segmentIds: segColumn([42n, 42n]),
      fragmentIndex: rangeIndex([[0, 2]]),
    };
    expect(computeChunkCrossings(chunk, []).size).toBe(0);
  });

  it("reads a full uint64 segment id (> 2^53) without truncation", () => {
    const big = 0x0123456789abcdefn;
    const chunk: RoiFilterableChunk = {
      rank: 3,
      numVertices: 2,
      positions: Float32Array.from([-5, 0, 0, 5, 0, 0]),
      segmentIds: segColumn([big, big]),
      fragmentIndex: rangeIndex([[0, 2]]),
    };
    const crossings = computeChunkCrossings(chunk, [
      roi(sphere(0, 0, 0, 1), RoiOperator.AND),
    ]);
    expect([...crossings.keys()]).toEqual([big]);
  });
});

describe("RoiFilterAccumulator", () => {
  const A = sphere(0, 0, 0, 1);
  const B = sphere(10, 0, 0, 1);

  it("adds an object that satisfies a single inclusion ROI", () => {
    const acc = new RoiFilterAccumulator([roi(A, RoiOperator.AND)]);
    const diff = acc.addChunk(new Map([[1n, [true]]]));
    expect(diff.added).toEqual([1n]);
    expect(diff.removed).toEqual([]);
    expect([...acc.passingSet]).toEqual([1n]);
  });

  it("holds an AND until BOTH regions are crossed across chunks", () => {
    const acc = new RoiFilterAccumulator([
      roi(A, RoiOperator.AND),
      roi(B, RoiOperator.AND),
    ]);
    // Chunk 1: object 42 crosses A only -> not yet passing.
    expect(acc.addChunk(new Map([[42n, [true, false]]]))).toEqual({
      added: [],
      removed: [],
    });
    expect(acc.passingSet.has(42n)).toBe(false);
    // Chunk 2: object 42 crosses B -> now passes A AND B.
    const diff = acc.addChunk(new Map([[42n, [false, true]]]));
    expect(diff.added).toEqual([42n]);
    expect(acc.passingSet.has(42n)).toBe(true);
  });

  it("REMOVES an object when a later chunk reveals it crosses an exclusion region", () => {
    // include A AND NOT B.
    const acc = new RoiFilterAccumulator([
      roi(A, RoiOperator.AND),
      roi(B, RoiOperator.ANDNOT),
    ]);
    // Chunk 1: object 5 crosses A, has not (yet) crossed B -> passes.
    expect(acc.addChunk(new Map([[5n, [true, false]]])).added).toEqual([5n]);
    expect(acc.passingSet.has(5n)).toBe(true);
    // Chunk 2: object 5 turns out to cross the exclusion region B -> dropped.
    const diff = acc.addChunk(new Map([[5n, [false, true]]]));
    expect(diff.removed).toEqual([5n]);
    expect(diff.added).toEqual([]);
    expect(acc.passingSet.has(5n)).toBe(false);
  });

  it("emits no diff when a chunk adds no new crossings (idempotent)", () => {
    const acc = new RoiFilterAccumulator([roi(A, RoiOperator.AND)]);
    acc.addChunk(new Map([[1n, [true]]]));
    const diff = acc.addChunk(new Map([[1n, [true]]]));
    expect(diff).toEqual({ added: [], removed: [] });
  });

  it("reset clears the accumulated crossings and passing set", () => {
    const acc = new RoiFilterAccumulator([roi(A, RoiOperator.AND)]);
    acc.addChunk(new Map([[1n, [true]]]));
    acc.reset();
    expect([...acc.passingSet]).toEqual([]);
    // After reset the same object can be re-added cleanly.
    expect(acc.addChunk(new Map([[1n, [true]]])).added).toEqual([1n]);
  });
});

describe("computePassingSet", () => {
  // Two chunks of one object (id 42): frag in chunk 1 crosses A, frag in
  // chunk 2 crosses B. Under "A AND B", the object passes only when both
  // chunks are folded in — the whole point of recomputing over the batch.
  const rois = [
    roi(sphere(0, 0, 0, 1), RoiOperator.AND),
    roi(sphere(10, 0, 0, 1), RoiOperator.AND),
  ];
  const chunkCrossesA: RoiFilterableChunk = {
    rank: 3,
    numVertices: 2,
    positions: Float32Array.from([-5, 0, 0, 5, 0, 0]),
    segmentIds: segColumn([42n, 42n]),
    fragmentIndex: rangeIndex([[0, 2]]),
  };
  const chunkCrossesB: RoiFilterableChunk = {
    rank: 3,
    numVertices: 2,
    positions: Float32Array.from([5, 0, 0, 15, 0, 0]),
    segmentIds: segColumn([42n, 42n]),
    fragmentIndex: rangeIndex([[0, 2]]),
  };

  it("passes an object only when the whole batch satisfies the fold", () => {
    expect([...computePassingSet([chunkCrossesA], rois)]).toEqual([]);
    expect([
      ...computePassingSet([chunkCrossesA, chunkCrossesB], rois),
    ]).toEqual([42n]);
  });

  it("is order-independent across chunks", () => {
    const a = computePassingSet([chunkCrossesA, chunkCrossesB], rois);
    const b = computePassingSet([chunkCrossesB, chunkCrossesA], rois);
    expect([...a].sort()).toEqual([...b].sort());
  });

  it("returns an empty set for an empty ROI list (no filter)", () => {
    expect(computePassingSet([chunkCrossesA, chunkCrossesB], []).size).toBe(0);
  });

  it("forgets chunks not in the batch (unlike the accumulator)", () => {
    // Recomputing over only chunk B drops object 42, which a never-forgetting
    // accumulator would still report from a prior addChunk of chunk A.
    const single = [roi(sphere(0, 0, 0, 1), RoiOperator.AND)];
    expect([...computePassingSet([chunkCrossesA], single)]).toEqual([42n]);
    expect([...computePassingSet([chunkCrossesB], single)]).toEqual([]);
  });
});

describe("computeGroupedPassingSet", () => {
  // Object 1 near origin, object 2 near x=10, object 3 near x=100 (crosses none).
  const chunk: RoiFilterableChunk = {
    rank: 3,
    numVertices: 6,
    positions: Float32Array.from([
      -5, 0, 0, 5, 0, 0 /* obj1 */, 5, 0, 0, 15, 0, 0 /* obj2 */, 95, 0, 0, 105,
      0, 0 /* obj3 */,
    ]),
    segmentIds: segColumn([1n, 1n, 2n, 2n, 3n, 3n]),
    fragmentIndex: rangeIndex([
      [0, 2],
      [2, 2],
      [4, 2],
    ]),
  };
  const RED = 0xff0000;
  const BLUE = 0x0000ff;
  const groupA: RoiGroupConfig = {
    rois: [roi(sphere(0, 0, 0, 1), RoiOperator.AND)],
    colorPacked: RED,
    visible: true,
  };
  const groupB: RoiGroupConfig = {
    rois: [roi(sphere(10, 0, 0, 1), RoiOperator.AND)],
    colorPacked: BLUE,
    visible: true,
  };

  it("unions passing objects across groups and colours by group", () => {
    const { passing, colorById } = computeGroupedPassingSet(
      [chunk],
      [groupA, groupB],
    );
    expect([...passing].sort()).toEqual([1n, 2n]);
    expect(colorById.get(1n)).toBe(RED);
    expect(colorById.get(2n)).toBe(BLUE);
    expect(colorById.has(3n)).toBe(false);
  });

  it("skips invisible groups entirely", () => {
    const { passing } = computeGroupedPassingSet(
      [chunk],
      [groupA, { ...groupB, visible: false }],
    );
    expect([...passing]).toEqual([1n]);
  });

  it("keeps groups independent (one group's exclusion does not drop another's)", () => {
    // Group A excludes the x=10 region; group B includes it. B's object 2 must
    // survive — a flat fold would have A's ANDNOT drop it.
    const a: RoiGroupConfig = {
      rois: [
        roi(sphere(0, 0, 0, 1), RoiOperator.AND),
        roi(sphere(10, 0, 0, 1), RoiOperator.ANDNOT),
      ],
      colorPacked: RED,
      visible: true,
    };
    const { passing } = computeGroupedPassingSet([chunk], [a, groupB]);
    expect([...passing].sort()).toEqual([1n, 2n]);
  });

  it("first (topmost) visible group wins the colour for a shared object", () => {
    // Both groups include the origin region; object 1 is in both -> gets A's.
    const bAlsoOrigin: RoiGroupConfig = {
      rois: [roi(sphere(0, 0, 0, 1), RoiOperator.AND)],
      colorPacked: BLUE,
      visible: true,
    };
    const { colorById } = computeGroupedPassingSet(
      [chunk],
      [groupA, bAlsoOrigin],
    );
    expect(colorById.get(1n)).toBe(RED);
  });

  it("returns empty for no groups", () => {
    const { passing, colorById } = computeGroupedPassingSet([chunk], []);
    expect(passing.size).toBe(0);
    expect(colorById.size).toBe(0);
  });

  // Per-object attribute values, keyed by attribute name, as shipped to the
  // worker for the length filter and object-attribute colouring.
  const lengthColumns = (values: Record<number, number>, min = 0, max = 100) =>
    new Map([
      [
        "length",
        {
          ids: BigUint64Array.from(Object.keys(values).map((k) => BigInt(k))),
          values: Float32Array.from(Object.values(values)),
          min,
          max,
        },
      ],
    ]);

  it("narrows a group's passing set by an object attribute range", () => {
    const cols = lengthColumns({ 1: 50, 2: 10, 3: 999 });
    // Object 1 (origin, length 50) is inside [40, 60] -> kept.
    const kept = computeGroupedPassingSet(
      [chunk],
      [{ ...groupA, attrFilters: [{ name: "length", min: 40, max: 60 }] }],
      cols,
    );
    expect([...kept.passing]).toEqual([1n]);
    // Outside [60, 100] -> dropped, even though its ROI matches.
    const dropped = computeGroupedPassingSet(
      [chunk],
      [{ ...groupA, attrFilters: [{ name: "length", min: 60, max: 100 }] }],
      cols,
    );
    expect(dropped.passing.size).toBe(0);
  });

  it("colours a group by an object attribute (colourmap), not the flat colour", () => {
    const { colorById } = computeGroupedPassingSet(
      [chunk],
      [{ ...groupA, colorBy: { kind: "objectAttr", name: "length" } }],
      lengthColumns({ 1: 50 }),
    );
    const c = colorById.get(1n);
    expect(c).toBeTypeOf("number");
    expect(c).not.toBe(RED); // colourmapped from the value, not the group swatch
  });

  it("leaves per-vertex-coloured groups out of the colour override", () => {
    const specs = [
      { kind: "direction" as const },
      { kind: "position" as const },
      { kind: "vertexAttr" as const, name: "fa" },
    ];
    for (const colorBy of specs) {
      const { passing, colorById } = computeGroupedPassingSet(
        [chunk],
        [{ ...groupA, colorBy }],
      );
      expect([...passing]).toEqual([1n]); // still passes the filter
      expect(colorById.has(1n)).toBe(false); // shader colours it per vertex
    }
  });

  it("falls back to the group colour / no narrowing when values are absent", () => {
    const { passing, colorById } = computeGroupedPassingSet(
      [chunk],
      [
        {
          ...groupA,
          colorBy: { kind: "objectAttr", name: "length" },
          attrFilters: [{ name: "length", min: 40, max: 60 }],
        },
      ],
      new Map(), // no columns loaded yet
    );
    expect([...passing]).toEqual([1n]); // not dropped despite the range
    expect(colorById.get(1n)).toBe(RED); // falls back to the group colour
  });
});

describe("computePerGroupPassingSets", () => {
  // Object 1 near origin, object 2 near x=10, object 3 near x=100 (crosses none).
  const chunk: RoiFilterableChunk = {
    rank: 3,
    numVertices: 6,
    positions: Float32Array.from([
      -5, 0, 0, 5, 0, 0, 5, 0, 0, 15, 0, 0, 95, 0, 0, 105, 0, 0,
    ]),
    segmentIds: segColumn([1n, 1n, 2n, 2n, 3n, 3n]),
    fragmentIndex: rangeIndex([
      [0, 2],
      [2, 2],
      [4, 2],
    ]),
  };
  const groupA: RoiGroupConfig = {
    rois: [roi(sphere(0, 0, 0, 1), RoiOperator.AND)],
    colorPacked: 0,
    visible: true,
  };
  const groupB: RoiGroupConfig = {
    rois: [roi(sphere(10, 0, 0, 1), RoiOperator.AND)],
    colorPacked: 0,
    visible: true,
  };

  it("returns each group's passing set separately, positionally", () => {
    const sets = computePerGroupPassingSets([chunk], [groupA, groupB]);
    expect(sets.length).toBe(2);
    expect([...sets[0]]).toEqual([1n]);
    expect([...sets[1]]).toEqual([2n]);
  });

  it("does not merge groups: a shared object stays in every group that selects it", () => {
    // The union path (computeGroupedPassingSet) would keep object 1 once; here it
    // must appear in BOTH groups so per-group export counts are right.
    const bAlsoOrigin: RoiGroupConfig = {
      ...groupB,
      rois: [roi(sphere(0, 0, 0, 1), RoiOperator.AND)],
    };
    const sets = computePerGroupPassingSets([chunk], [groupA, bAlsoOrigin]);
    expect([...sets[0]]).toEqual([1n]);
    expect([...sets[1]]).toEqual([1n]);
  });

  it("evaluates invisible groups too (export honours the tick, not visibility)", () => {
    // computeGroupedPassingSet SKIPS invisible groups; the export helper must not
    // -- the user ticked the group to export it.
    const sets = computePerGroupPassingSets(
      [chunk],
      [{ ...groupA, visible: false }],
    );
    expect([...sets[0]]).toEqual([1n]);
  });

  it("yields an empty set for a group with no rois", () => {
    const sets = computePerGroupPassingSets([chunk], [{ ...groupA, rois: [] }]);
    expect(sets[0].size).toBe(0);
  });

  it("keeps groups independent under exclusion", () => {
    // Group A includes the origin and EXCLUDES x=10; group B includes x=10.
    // B's object 2 must survive -- a flat fold would let A's ANDNOT drop it.
    const a: RoiGroupConfig = {
      ...groupA,
      rois: [
        roi(sphere(0, 0, 0, 1), RoiOperator.AND),
        roi(sphere(10, 0, 0, 1), RoiOperator.ANDNOT),
      ],
    };
    const sets = computePerGroupPassingSets([chunk], [a, groupB]);
    expect([...sets[0]]).toEqual([1n]);
    expect([...sets[1]]).toEqual([2n]);
  });

  it("narrows a group's set by an object attribute range", () => {
    const cols = new Map([
      [
        "length",
        {
          ids: BigUint64Array.from([1n]),
          values: Float32Array.from([50]),
          min: 0,
          max: 100,
        },
      ],
    ]);
    const kept = computePerGroupPassingSets(
      [chunk],
      [{ ...groupA, attrFilters: [{ name: "length", min: 40, max: 60 }] }],
      cols,
    );
    expect([...kept[0]]).toEqual([1n]);
    const dropped = computePerGroupPassingSets(
      [chunk],
      [{ ...groupA, attrFilters: [{ name: "length", min: 60, max: 100 }] }],
      cols,
    );
    expect(dropped[0].size).toBe(0);
  });

  it("returns an empty array for no groups", () => {
    expect(computePerGroupPassingSets([chunk], [])).toEqual([]);
  });
});

describe("computeChunkCrossings for geometry without an object model", () => {
  it("tests each point on its own rather than folding the bin it sits in", () => {
    // Three unrelated cells in ONE fragment. Only the one at the origin is in
    // the sphere; folding per fragment would have put all three in it.
    const chunk = pointChunk(
      [
        [0, 0, 0],
        [50, 0, 0],
        [100, 0, 0],
      ],
      [1n, 2n, 3n],
    );
    const crossings = computeChunkCrossings(chunk, [
      roi(sphere(0, 0, 0, 1), RoiOperator.AND),
    ]);
    expect(crossings.get(1n)).toEqual([true]);
    expect(crossings.get(2n)).toEqual([false]);
    expect(crossings.get(3n)).toEqual([false]);
  });

  it("does not join consecutive points into a segment", () => {
    // The sphere sits BETWEEN two points. A polyline through them would cross
    // it; two independent points do not.
    const chunk = pointChunk(
      [
        [-5, 0, 0],
        [5, 0, 0],
      ],
      [1n, 2n],
    );
    const crossings = computeChunkCrossings(chunk, [
      roi(sphere(0, 0, 0, 1), RoiOperator.AND),
    ]);
    expect(crossings.get(1n)).toEqual([false]);
    expect(crossings.get(2n)).toEqual([false]);
  });

  it("evaluates a surface's vertices, not the walk through them", () => {
    // Same geometry as a tract fragment, but flagged as a face soup: the ROI
    // between the two vertices is not crossed by anything real.
    const chunk: RoiFilterableChunk = {
      rank: 3,
      numVertices: 2,
      positions: Float32Array.from([-5, 0, 0, 5, 0, 0]),
      segmentIds: segColumn([42n, 42n]),
      fragmentIndex: rangeIndex([[0, 2]]),
      surfaceVertices: true,
    };
    const rois = [roi(sphere(0, 0, 0, 1), RoiOperator.AND)];
    expect(computeChunkCrossings(chunk, rois).get(42n)).toEqual([false]);
    // A vertex actually inside it still counts.
    const inside: RoiFilterableChunk = {
      ...chunk,
      positions: Float32Array.from([0.5, 0, 0, 5, 0, 0]),
    };
    expect(computeChunkCrossings(inside, rois).get(42n)).toEqual([true]);
  });
});

describe("computeVertexAttrPassingSet", () => {
  const chunk = () =>
    pointChunk(
      [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
      ],
      [1n, 2n, 3n],
      new Map([
        ["gene_a", Float32Array.from([0.1, 5, 9])],
        ["flag", Float32Array.from([1, 0, 1])],
      ]),
    );

  it("selects the vertices whose value is in range", () => {
    const passing = computeVertexAttrPassingSet(
      [chunk()],
      [{ name: "gene_a", min: 1, max: 10, scope: "vertex" }],
    );
    expect([...passing].sort()).toEqual([2n, 3n]);
  });

  it("requires ONE vertex to satisfy every predicate at once", () => {
    // Vertex 3 has the high gene value; vertex 1 has the flag. Neither vertex
    // has both, so nothing passes -- co-expression, not either-or.
    const passing = computeVertexAttrPassingSet(
      [chunk()],
      [
        { name: "gene_a", min: 8, max: 10, scope: "vertex" },
        { name: "flag", min: 0.5, max: 1, scope: "vertex" },
      ],
    );
    expect([...passing]).toEqual([3n]);
    const none = computeVertexAttrPassingSet(
      [chunk()],
      [
        { name: "gene_a", min: 4, max: 6, scope: "vertex" },
        { name: "flag", min: 0.5, max: 1, scope: "vertex" },
      ],
    );
    expect(none.size).toBe(0);
  });

  it("selects nothing from a chunk missing the column", () => {
    // A filter naming an unloaded attribute must not read as "select all".
    const passing = computeVertexAttrPassingSet(
      [chunk()],
      [{ name: "gene_absent", min: 0, max: 1, scope: "vertex" }],
    );
    expect(passing.size).toBe(0);
  });
});

describe("attribute-only and mixed groups", () => {
  const attrChunk = () =>
    pointChunk(
      [
        [0, 0, 0],
        [50, 0, 0],
      ],
      [1n, 2n],
      new Map([["gene_a", Float32Array.from([9, 9])]]),
    );
  const group = (config: Partial<RoiGroupConfig>): RoiGroupConfig => ({
    rois: [],
    colorPacked: 0xff0000ff,
    visible: true,
    ...config,
  });

  it("selects by attribute alone, with no ROI drawn at all", () => {
    const { passing } = computeGroupedPassingSet(
      [attrChunk()],
      [
        group({
          attrFilters: [{ name: "gene_a", min: 5, max: 10, scope: "vertex" }],
        }),
      ],
    );
    expect([...passing].sort()).toEqual([1n, 2n]);
  });

  it("intersects an attribute predicate with the ROI fold", () => {
    // Both cells express the gene; only the one at the origin is in the sphere.
    const { passing } = computeGroupedPassingSet(
      [attrChunk()],
      [
        group({
          rois: [roi(sphere(0, 0, 0, 1), RoiOperator.AND)],
          attrFilters: [{ name: "gene_a", min: 5, max: 10, scope: "vertex" }],
        }),
      ],
    );
    expect([...passing]).toEqual([1n]);
    // Narrowing the range past the data empties the group even though the ROI
    // still matches.
    const { passing: none } = computeGroupedPassingSet(
      [attrChunk()],
      [
        group({
          rois: [roi(sphere(0, 0, 0, 1), RoiOperator.AND)],
          attrFilters: [{ name: "gene_a", min: 0, max: 1, scope: "vertex" }],
        }),
      ],
    );
    expect(none.size).toBe(0);
  });

  it("selects objects the view has not loaded from an object-scope column", () => {
    // No geometry at all: the column alone decides. This is what lets a group
    // be "every object with FA > 0.5" rather than "every loaded one".
    const columns = new Map([
      [
        "fa",
        {
          ids: BigUint64Array.from([10n, 11n, 12n]),
          values: Float32Array.from([0.2, 0.6, 0.9]),
          min: 0,
          max: 1,
        },
      ],
    ]);
    const { passing } = computeGroupedPassingSet(
      [],
      [group({ attrFilters: [{ name: "fa", min: 0.5, max: 1 }] })],
      columns,
    );
    expect([...passing].sort()).toEqual([11n, 12n]);
  });

  it("still selects nothing for a group with neither ROIs nor predicates", () => {
    const { passing } = computeGroupedPassingSet([attrChunk()], [group({})]);
    expect(passing.size).toBe(0);
  });

  it("reports attribute-only groups per group for the export path too", () => {
    const sets = computePerGroupPassingSets(
      [attrChunk()],
      [
        group({
          attrFilters: [{ name: "gene_a", min: 5, max: 10, scope: "vertex" }],
        }),
        group({
          attrFilters: [{ name: "gene_a", min: 0, max: 1, scope: "vertex" }],
        }),
      ],
    );
    expect([...sets[0]].sort()).toEqual([1n, 2n]);
    expect(sets[1].size).toBe(0);
  });
});

describe("diffPassingSet", () => {
  it("reports adds and removes to turn current into target", () => {
    const diff = diffPassingSet(new Set([1n, 2n, 3n]), new Set([2n, 3n, 4n]));
    expect(diff.added.sort()).toEqual([1n]);
    expect(diff.removed.sort()).toEqual([4n]);
  });

  it("is empty when the sets already match (makes a redundant recompute free)", () => {
    const diff = diffPassingSet(new Set([5n, 6n]), new Set([6n, 5n]));
    expect(diff).toEqual({ added: [], removed: [] });
  });

  it("adds everything against an empty current, removes everything against an empty target", () => {
    expect(diffPassingSet(new Set([1n, 2n]), new Set()).added.sort()).toEqual([
      1n,
      2n,
    ]);
    expect(diffPassingSet(new Set(), new Set([1n, 2n])).removed.sort()).toEqual(
      [1n, 2n],
    );
  });
});
