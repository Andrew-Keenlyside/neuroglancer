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
import { writeTrk } from "#src/datasource/zarr-vectors/trk_writer.js";

// Golden generated from the Python nibabel path -- `_write_trk_bytes` in
// `python/neuroglancer/tract_export/browser.py` -- for exactly the streamlines
// and affine below. If this fails, the TS writer has drifted from the file the
// native exporter produces (or one of them changed a header convention).
const GOLDEN_B64 =
  "VFJBQ0sAoQAaAC4AAAAAQAAAAEAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAACBBAAAAAAAAAEAAAAAAAACAwAAAAAAAAAAAAAAAQAAAQEAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUkFTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAACAAAA6AMAAAIAAAAAAIA/AACAPwAAgD8AgKBDAAAkQgAAqEEDAAAAAAAwQQAAMEEAADBBAAAwQQAATEIAADBBAAAwQQAATEIAALZC";

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; ++i) out[i] = bin.charCodeAt(i);
  return out;
}

describe("writeTrk", () => {
  const streamlines = [
    Float32Array.from([0, 0, 0, 160, 20, 10]),
    Float32Array.from([5, 5, 5, 5, 25, 5, 5, 25, 45]),
  ];
  const affine = [
    [2, 0, 0, 10],
    [0, 2, 0, -4],
    [0, 0, 2, 3],
    [0, 0, 0, 1],
  ];

  it("is byte-identical to the Python nibabel exporter", () => {
    const got = writeTrk(streamlines, affine);
    expect(Array.from(got)).toEqual(Array.from(base64ToBytes(GOLDEN_B64)));
  });

  it("writes a 1000-byte header then the streamline body", () => {
    const got = writeTrk(streamlines, affine);
    const view = new DataView(got.buffer);
    expect(String.fromCharCode(...got.slice(0, 5))).toBe("TRACK");
    expect(view.getInt32(996, true)).toBe(1000); // hdr_size
    expect(view.getInt32(992, true)).toBe(2); // version
    expect(view.getInt32(988, true)).toBe(2); // n_count = 2 streamlines
    // voxel_size = affine column norms (2,2,2).
    expect(view.getFloat32(12, true)).toBe(2);
    // First body record: point count of the first streamline.
    expect(view.getInt32(1000, true)).toBe(2);
    // Non-degenerate reference dims (not the (1,1,1) that broke freeview).
    const dims = [
      view.getInt16(6, true),
      view.getInt16(8, true),
      view.getInt16(10, true),
    ];
    expect(dims).not.toEqual([1, 1, 1]);
  });

  it("stores points at the voxel centre, scaled by voxel size", () => {
    // A 1 mm identity affine: point (10,20,30) -> voxmm (10.5,20.5,30.5).
    const got = writeTrk(
      [Float32Array.from([10, 20, 30])],
      [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ],
    );
    const view = new DataView(got.buffer);
    expect(view.getInt32(1000, true)).toBe(1); // one point
    expect(view.getFloat32(1004, true)).toBeCloseTo(10.5);
    expect(view.getFloat32(1008, true)).toBeCloseTo(20.5);
    expect(view.getFloat32(1012, true)).toBeCloseTo(30.5);
  });

  it("derives voxel_order from the affine orientation", () => {
    // Negative x column -> L instead of R.
    const got = writeTrk(
      [Float32Array.from([0, 0, 0, 1, 1, 1])],
      [
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ],
    );
    expect(String.fromCharCode(...got.slice(948, 951))).toBe("LAS");
  });
});
