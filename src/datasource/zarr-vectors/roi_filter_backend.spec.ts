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
  const rois = [roi(sphere(0, 0, 0, 1), RoiOperator.AND), roi(sphere(10, 0, 0, 1), RoiOperator.AND)];
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
    expect([...computePassingSet([chunkCrossesA, chunkCrossesB], rois)]).toEqual([
      42n,
    ]);
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
    highDetail: false,
  };
  const groupB: RoiGroupConfig = {
    rois: [roi(sphere(10, 0, 0, 1), RoiOperator.AND)],
    colorPacked: BLUE,
    visible: true,
    highDetail: false,
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
      highDetail: false,
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
      highDetail: false,
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
    expect(diffPassingSet(new Set(), new Set([1n, 2n])).removed.sort()).toEqual([
      1n,
      2n,
    ]);
  });
});
