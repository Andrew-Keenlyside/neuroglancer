/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import { computeChunkIndexBounds } from "#src/datasource/zarr-vectors/chunk_bounds.js";

describe("computeChunkIndexBounds", () => {
  it("handles a fractional chunk_shape", () => {
    // The MERFISH case: a 0.5 mm grid over a ~12 x 8 x 6 mm brain. Truncating
    // the shape to an integer would give 0 and a bound of Infinity, and the
    // frustum walk would recurse until it blew the stack.
    const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
      [0.5378, 0.1033, 0.5055],
      [12.6255, 7.855, 5.875],
      [0.5, 0.5, 0.5],
    );
    expect(Array.from(lowerChunkBound)).toEqual([1, 0, 1]);
    expect(Array.from(upperChunkBound)).toEqual([26, 16, 12]);
  });

  it("indexes chunks around the world origin, so negative indices are fine", () => {
    const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
      [-30, -0.5, -100],
      [10, 0.5, -60],
      [10, 1, 25],
    );
    expect(Array.from(lowerChunkBound)).toEqual([-3, -1, -4]);
    expect(Array.from(upperChunkBound)).toEqual([1, 1, -2]);
  });

  it("leaves one chunk on a degenerate axis", () => {
    // A flat (2-D) store still has to draw: a zero-volume range makes the walk
    // terminate before it reaches a single chunk.
    const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
      [0, 0, 4],
      [10, 10, 4],
      [5, 5, 5],
    );
    expect(Array.from(lowerChunkBound)).toEqual([0, 0, 0]);
    expect(Array.from(upperChunkBound)).toEqual([2, 2, 1]);
  });

  it("rejects a non-positive chunk extent instead of producing Infinity", () => {
    expect(() =>
      computeChunkIndexBounds([0, 0, 0], [1, 1, 1], [1, 0, 1]),
    ).toThrow(/not a positive extent/);
  });
});
