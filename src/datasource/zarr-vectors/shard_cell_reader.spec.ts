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
import type { ChunkGridDescriptor } from "#src/datasource/zarr-vectors/shard_cell_reader.js";
import { resolveChunkCell } from "#src/datasource/zarr-vectors/shard_cell_reader.js";

// Grid of the deployed hcp1065_whole_brain.zarrvectors store, verified against
// zarr-vectors-py: origin [-4,-5,-2], shard shape [8,8,8], separator "/".
const SHARDED: ChunkGridDescriptor = {
  origin: [-4, -5, -2],
  sharded: true,
  shardShape: [8, 8, 8],
  separator: "/",
};

describe("resolveChunkCell (sharded, nonzero origin)", () => {
  it("maps abs (-2,-1,-1) → arrayIndex (2,4,1), shard (0,0,0), inner (2,4,1)", () => {
    const r = resolveChunkCell(SHARDED, "-2.-1.-1");
    expect(r.arrayIndex).toEqual([2, 4, 1]);
    expect(r.shard).toEqual([0, 0, 0]);
    expect(r.inner).toEqual([2, 4, 1]);
    // C-order linear inner index (entry = lin*2 in the [8,8,8,2] index table).
    const [i0, i1, i2] = r.inner!;
    expect((i0 * 8 + i1) * 8 + i2).toBe(161);
  });

  it("maps the world origin abs (0,0,0) → arrayIndex (4,5,2)", () => {
    expect(resolveChunkCell(SHARDED, "0.0.0").arrayIndex).toEqual([4, 5, 2]);
  });

  it("routes a coord into a non-(0,0,0) shard by //8", () => {
    // abs (3,3,3) → arrayIndex (7,8,5) → shard (0,1,0), inner (7,0,5).
    const r = resolveChunkCell(SHARDED, "3.3.3");
    expect(r.arrayIndex).toEqual([7, 8, 5]);
    expect(r.shard).toEqual([0, 1, 0]);
    expect(r.inner).toEqual([7, 0, 5]);
  });
});

describe("resolveChunkCell (unsharded / back-compat)", () => {
  it("nonzero origin, unsharded: arrayIndex only, no shard/inner", () => {
    const grid: ChunkGridDescriptor = {
      origin: [-4, -5, -2],
      sharded: false,
      shardShape: [],
      separator: "/",
    };
    const r = resolveChunkCell(grid, "-2.-1.-1");
    expect(r.arrayIndex).toEqual([2, 4, 1]);
    expect(r.shard).toBeUndefined();
    expect(r.inner).toBeUndefined();
  });

  it("zero origin, unsharded reduces to identity (historical behaviour)", () => {
    const grid: ChunkGridDescriptor = {
      origin: [0, 0, 0],
      sharded: false,
      shardShape: [],
      separator: "/",
    };
    expect(resolveChunkCell(grid, "3.4.1").arrayIndex).toEqual([3, 4, 1]);
  });

  it("absent origin components default to 0", () => {
    const grid: ChunkGridDescriptor = {
      origin: [],
      sharded: false,
      shardShape: [],
      separator: "/",
    };
    expect(resolveChunkCell(grid, "5.6.7").arrayIndex).toEqual([5, 6, 7]);
  });
});
