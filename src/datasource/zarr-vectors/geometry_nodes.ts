/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Turn one aggregated zarr-vectors object into the rooted node tree the
 * spatial-skeleton editing UI consumes.
 *
 * The two models disagree about direction. A ZVF object is an undirected edge
 * list -- `synthesizeEdgesForConvention` emits `i -> i+1` pairs within a
 * fragment and the branch records in `links/0/0.0.0` are bare index pairs, with
 * nothing marking which endpoint is the parent (see
 * `computeTangentsFromEdges`, which flood-fills to make tangent SIGN
 * consistent precisely because direction is not recoverable from the data).
 * `SpatiallyIndexedSkeletonNode` on the other side is a tree: every node but
 * one carries a `parentNodeId`.
 *
 * So a root is chosen by convention and parents fall out of a traversal. The
 * convention is deterministic, because the UI stores node ids in URLs and in
 * edit documents: re-reading the same object must produce the same tree.
 *
 *   - Each connected component is rooted independently, so an object whose
 *     fragments never got bridged still yields a usable tree per piece rather
 *     than dropping all but one.
 *   - The root is the lowest-indexed degree-1 vertex in the component -- the
 *     start of the walk for a path, which is what `implicit_sequential`
 *     geometry is. A component with no endpoint at all (a cycle) falls back to
 *     its lowest-indexed vertex, since something must be the root.
 *   - Neighbours are visited in ascending index order.
 *
 * This is deliberately pure: no store reads, no chunk types, no RPC. It is the
 * one piece of the ZVF->node conversion with real semantics, so it is testable
 * on its own.
 */

import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";

export interface BuildSpatialSkeletonNodesOptions {
  /** `numVertices * rank` floats, as aggregated by `downloadSegmentSkeleton`. */
  readonly vertexPositions: Float32Array;
  /** `numEdges * 2` vertex indices, undirected. */
  readonly indices: Uint32Array;
  /** Components per position; only the first three are carried to a node. */
  readonly rank: number;
  /**
   * The object id these nodes belong to, as the UI's `number`-typed segment id.
   * Every node in one call shares it.
   */
  readonly segmentId: number;
  /**
   * Stable per-vertex ids, parallel to the vertices. When absent, ids are
   * synthesised as `idOffset + vertexIndex + 1` -- usable within a session but
   * NOT stable across one, which is why a store must carry a real id column
   * before it may be edited.
   */
  readonly nodeIds?: ArrayLike<number>;
  /** Base for synthesised ids; ignored when `nodeIds` is given. */
  readonly idOffset?: number;
}

/** Largest node id the wire format can carry: ids ride an `Int32Array`. */
const MAX_NODE_ID = 0x7fffffff;

/**
 * Build the adjacency list, dropping any edge that does not name two distinct
 * in-range vertices. A dangling endpoint would otherwise index past the
 * positions array and produce a node with `NaN` coordinates.
 */
function buildAdjacency(indices: Uint32Array, numVertices: number): number[][] {
  const adjacency: number[][] = new Array(numVertices);
  for (let i = 0; i < numVertices; ++i) adjacency[i] = [];
  for (let e = 0; e + 1 < indices.length; e += 2) {
    const a = indices[e];
    const b = indices[e + 1];
    if (a === b) continue;
    if (a >= numVertices || b >= numVertices) continue;
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  for (let i = 0; i < numVertices; ++i) {
    // Ascending order makes the traversal -- and therefore every parent
    // assignment -- independent of the order edges happened to arrive in.
    adjacency[i].sort((x, y) => x - y);
  }
  return adjacency;
}

/**
 * Pick the root of the component containing `start`: its lowest-indexed
 * degree-1 vertex, else its lowest-indexed vertex.
 */
function chooseRoot(
  adjacency: number[][],
  start: number,
  visited: Uint8Array,
): { root: number; component: number[] } {
  const component: number[] = [];
  const stack = [start];
  const seen = new Set<number>([start]);
  while (stack.length > 0) {
    const v = stack.pop()!;
    component.push(v);
    for (const n of adjacency[v]) {
      if (seen.has(n)) continue;
      seen.add(n);
      stack.push(n);
    }
  }
  component.sort((a, b) => a - b);
  for (const v of component) visited[v] = 1;
  let root = component[0];
  for (const v of component) {
    if (adjacency[v].length === 1) {
      root = v;
      break;
    }
  }
  return { root, component };
}

/**
 * Convert aggregated geometry into rooted nodes. Vertices map one-to-one onto
 * nodes; order is by vertex index, so `result[i]` describes vertex `i`.
 */
export function buildSpatialSkeletonNodes(
  options: BuildSpatialSkeletonNodesOptions,
): SpatiallyIndexedSkeletonNode[] {
  const {
    vertexPositions,
    indices,
    rank,
    segmentId,
    nodeIds,
    idOffset = 0,
  } = options;
  if (rank < 1) return [];
  const numVertices = Math.floor(vertexPositions.length / rank);
  if (numVertices === 0) return [];

  const adjacency = buildAdjacency(indices, numVertices);

  const idOf = (vertexIndex: number): number => {
    const raw =
      nodeIds === undefined
        ? idOffset + vertexIndex + 1
        : Math.round(Number(nodeIds[vertexIndex]));
    return raw;
  };

  const parentOf = new Int32Array(numVertices).fill(-1);
  const visited = new Uint8Array(numVertices);
  for (let start = 0; start < numVertices; ++start) {
    if (visited[start]) continue;
    const { root } = chooseRoot(adjacency, start, visited);
    // Breadth-first from the root so a node's parent is always the neighbour
    // nearest the root -- a depth-first walk would thread a branch point's
    // children into a chain.
    const queue = [root];
    const enqueued = new Uint8Array(numVertices);
    enqueued[root] = 1;
    for (let head = 0; head < queue.length; ++head) {
      const v = queue[head];
      for (const n of adjacency[v]) {
        if (enqueued[n]) continue;
        enqueued[n] = 1;
        parentOf[n] = v;
        queue.push(n);
      }
    }
  }

  const nodes: SpatiallyIndexedSkeletonNode[] = new Array(numVertices);
  for (let i = 0; i < numVertices; ++i) {
    const nodeId = idOf(i);
    if (!Number.isSafeInteger(nodeId) || nodeId <= 0 || nodeId > MAX_NODE_ID) {
      throw new Error(
        `zarr-vectors: vertex ${i} of object ${segmentId} has node id ` +
          `${nodeId}, which is outside the editable range [1, ${MAX_NODE_ID}]. ` +
          "Node ids cross the worker boundary in an Int32Array.",
      );
    }
    const base = i * rank;
    const position = new Float32Array(3);
    for (let c = 0; c < 3 && c < rank; ++c) {
      position[c] = vertexPositions[base + c];
    }
    const parentIndex = parentOf[i];
    nodes[i] = {
      nodeId,
      segmentId,
      position,
      ...(parentIndex < 0 ? {} : { parentNodeId: idOf(parentIndex) }),
    };
  }
  return nodes;
}
