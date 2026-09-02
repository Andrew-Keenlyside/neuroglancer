/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";

import {
  forEachSpatialSkeletonVolumeCell,
  MAX_SPATIAL_SKELETON_VOLUME_CELLS,
} from "#src/skeleton/spatial_base.js";

function collect(lower: number[], upper: number[]) {
  const seen: string[] = [];
  const visited = forEachSpatialSkeletonVolumeCell(lower, upper, (p) =>
    seen.push(p.join()),
  );
  return { seen, visited };
}

describe("forEachSpatialSkeletonVolumeCell", () => {
  it("visits every cell of the grid", () => {
    const { seen, visited } = collect([0, 0, 0], [2, 2, 2]);
    expect(visited).toBe(8);
    expect(new Set(seen).size).toBe(8);
    expect(seen).toContain("0,0,0");
    expect(seen).toContain("1,1,1");
  });

  it("covers the real store's 6x7x5 index exactly once", () => {
    const { seen, visited } = collect([0, 0, 0], [6, 7, 5]);
    expect(visited).toBe(210);
    expect(new Set(seen).size).toBe(210);
  });

  it("honours a negative lower bound", () => {
    // Chunk coordinates are signed: zarr-vectors indexes cells around the world
    // origin, and a brain volume straddles it.
    const { seen, visited } = collect([-1, -1, 0], [1, 1, 1]);
    expect(visited).toBe(4);
    expect(seen).toContain("-1,-1,0");
    expect(seen).toContain("0,0,0");
  });

  it("visits nothing for an empty grid", () => {
    expect(collect([0, 0, 0], [0, 5, 5]).visited).toBe(0);
    expect(collect([3, 0, 0], [1, 5, 5]).visited).toBe(0);
  });

  it("refuses an oversized grid without half-enumerating it", () => {
    // -1 rather than a truncated walk, so the caller can fall back to the
    // frustum instead of silently loading part of the volume.
    const big = Math.ceil(Math.cbrt(MAX_SPATIAL_SKELETON_VOLUME_CELLS)) + 10;
    const seen: string[] = [];
    const visited = forEachSpatialSkeletonVolumeCell(
      [0, 0, 0],
      [big, big, big],
      (p) => seen.push(p.join()),
    );
    expect(visited).toBe(-1);
    expect(seen).toHaveLength(0);
  });

  it("refuses a non-finite bound", () => {
    expect(
      forEachSpatialSkeletonVolumeCell(
        [0, 0, 0],
        [Number.POSITIVE_INFINITY, 1, 1],
        () => {},
      ),
    ).toBe(-1);
  });

  it("hands out a reusable position, so callers must copy to retain", () => {
    const kept: Float32Array[] = [];
    forEachSpatialSkeletonVolumeCell([0, 0, 0], [2, 1, 1], (p) => kept.push(p));
    expect(kept[0]).toBe(kept[1]);
  });
});
