/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";

import { buildSpatialSkeletonNodes } from "#src/datasource/zarr-vectors/geometry_nodes.js";

/** `n` vertices at `(i, 0, 0)`, the shape a straight streamline decodes to. */
function positions(n: number, rank = 3): Float32Array {
  const out = new Float32Array(n * rank);
  for (let i = 0; i < n; ++i) out[i * rank] = i;
  return out;
}

function build(
  numVertices: number,
  edges: number[],
  overrides: Partial<Parameters<typeof buildSpatialSkeletonNodes>[0]> = {},
) {
  return buildSpatialSkeletonNodes({
    vertexPositions: positions(numVertices),
    indices: Uint32Array.from(edges),
    rank: 3,
    segmentId: 7,
    ...overrides,
  });
}

/** `nodeId -> parentNodeId ?? null`, the shape assertions read best in. */
function parents(nodes: ReturnType<typeof buildSpatialSkeletonNodes>) {
  return nodes.map((n) => [n.nodeId, n.parentNodeId ?? null]);
}

describe("buildSpatialSkeletonNodes", () => {
  it("roots a path at its first endpoint and chains parents along it", () => {
    // The `implicit_sequential` case: fragment walk order 0->1->2->3.
    const nodes = build(4, [0, 1, 1, 2, 2, 3]);
    expect(parents(nodes)).toEqual([
      [1, null],
      [2, 1],
      [3, 2],
      [4, 3],
    ]);
    expect(nodes.every((n) => n.segmentId === 7)).toBe(true);
  });

  it("is independent of the order edges arrive in", () => {
    // Cross-chunk bridges are appended after the intra-chunk edges, so the
    // same path routinely arrives scrambled. The tree must not depend on it.
    const walkOrder = build(4, [0, 1, 1, 2, 2, 3]);
    const scrambled = build(4, [2, 3, 0, 1, 2, 1]);
    expect(parents(scrambled)).toEqual(parents(walkOrder));
  });

  it("hangs both arms of a branch off the branch point", () => {
    // A `skeleton` store's branch edge: vertex 1 has degree 3.
    const nodes = build(4, [0, 1, 1, 2, 1, 3]);
    expect(parents(nodes)).toEqual([
      [1, null],
      [2, 1],
      [3, 2],
      [4, 2],
    ]);
  });

  it("does not thread a branch point's children into a chain", () => {
    // The depth-first failure mode: 3 must parent to 1, never to 2.
    const nodes = build(4, [0, 1, 1, 2, 1, 3]);
    expect(nodes[3].parentNodeId).toBe(nodes[1].nodeId);
  });

  it("roots each connected component independently", () => {
    // An object whose fragments never got bridged: two paths, one object.
    const nodes = build(4, [0, 1, 2, 3]);
    expect(parents(nodes)).toEqual([
      [1, null],
      [2, 1],
      [3, null],
      [4, 3],
    ]);
  });

  it("falls back to the lowest vertex when a component has no endpoint", () => {
    // A cycle: every vertex has degree 2, so no endpoint exists to prefer.
    const nodes = build(3, [0, 1, 1, 2, 2, 0]);
    expect(parents(nodes)).toEqual([
      [1, null],
      [2, 1],
      [3, 1],
    ]);
  });

  it("drops self-edges and edges naming a vertex that is not there", () => {
    // `filterChunkByFragments` should not emit these, but a dangling endpoint
    // would index past the positions array and yield NaN coordinates.
    const nodes = build(3, [0, 0, 0, 1, 1, 9]);
    expect(parents(nodes)).toEqual([
      [1, null],
      [2, 1],
      [3, null],
    ]);
    expect(Array.from(nodes[2].position)).toEqual([2, 0, 0]);
  });

  it("uses a supplied id column for both the node and its parent", () => {
    const nodes = build(3, [0, 1, 1, 2], { nodeIds: [41, 42, 43] });
    expect(parents(nodes)).toEqual([
      [41, null],
      [42, 41],
      [43, 42],
    ]);
  });

  it("offsets synthesised ids so two objects do not collide", () => {
    const nodes = build(2, [0, 1], { idOffset: 1000 });
    expect(parents(nodes)).toEqual([
      [1001, null],
      [1002, 1001],
    ]);
  });

  it("rejects an id the Int32Array wire format cannot carry", () => {
    // Silently truncating would alias two vertices onto one node id.
    expect(() => build(2, [0, 1], { nodeIds: [1, 0x80000000] })).toThrow(
      /outside the editable range/,
    );
    expect(() => build(2, [0, 1], { nodeIds: [0, 1] })).toThrow(
      /outside the editable range/,
    );
  });

  it("carries only the first three components of a higher-rank position", () => {
    const nodes = buildSpatialSkeletonNodes({
      vertexPositions: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
      indices: Uint32Array.from([0, 1]),
      rank: 4,
      segmentId: 1,
    });
    expect(Array.from(nodes[0].position)).toEqual([1, 2, 3]);
    expect(Array.from(nodes[1].position)).toEqual([5, 6, 7]);
  });

  it("returns nothing for an object with no geometry", () => {
    expect(build(0, [])).toEqual([]);
  });
});
