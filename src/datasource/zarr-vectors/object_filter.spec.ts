/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";

import { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import type { SkeletonChunk } from "#src/datasource/zarr-vectors/geometry_chunk.js";
import { filterChunkByAdmittedObjects } from "#src/datasource/zarr-vectors/object_filter.js";

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
