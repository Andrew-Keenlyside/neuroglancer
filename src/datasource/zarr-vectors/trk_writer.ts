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

/**
 * @file A pure TrackVis `.trk` writer, for the fast in-browser export that
 * assembles streamlines from geometry already resident in the worker -- no store
 * re-read, no Pyodide.
 *
 * The output is **byte-identical** to what the Python exporter's nibabel path
 * (`_write_trk_bytes` / `_trk_header` in `tract_export/browser.py`) produces for
 * the same streamlines and affine -- pinned by `trk_writer.spec.ts` against a
 * golden generated from that path. So a file written here and one written by the
 * native exporter are the same file.
 *
 * Conventions matched to nibabel:
 *  - streamlines are in the store's own (voxel/model) coordinates; `affine` is
 *    the voxel→RAS matrix **in millimetres** (TRK is a millimetre format);
 *  - `voxel_size` = the norms of the affine's direction columns;
 *  - points are stored in TrackVis "voxmm" at the **voxel centre**:
 *    `(coord + 0.5) × voxel_size`;
 *  - `dimensions` is a reference grid bounding the vertices (int16), so tools
 *    like freeview have a non-degenerate volume to place the tract against;
 *  - `voxel_order` is derived from each axis's dominant world direction.
 */

/** A voxel→RAS-mm 4×4, row-major. */
export type Affine4x4 = readonly (readonly number[])[];

/** Per-axis scale = norm of the affine's direction column. */
function columnNorms(a: Affine4x4): [number, number, number] {
  const norm = (c: number) => Math.hypot(a[0][c], a[1][c], a[2][c]);
  return [norm(0), norm(1), norm(2)];
}

/**
 * The 3-letter TrackVis `voxel_order` from the affine: for each voxel axis, the
 * world direction its dominant component points along (R/L, A/P, S/I). A positive
 * diagonal affine yields "RAS".
 */
function voxelOrder(a: Affine4x4): string {
  const letters: [string, string][] = [
    ["L", "R"],
    ["P", "A"],
    ["I", "S"],
  ];
  let order = "";
  for (let c = 0; c < 3; ++c) {
    let axis = 0;
    let best = -1;
    for (let r = 0; r < 3; ++r) {
      const v = Math.abs(a[r][c]);
      if (v > best) {
        best = v;
        axis = r;
      }
    }
    order += letters[axis][a[axis][c] >= 0 ? 1 : 0];
  }
  return order;
}

const TRK_HEADER_BYTES = 1000;

/**
 * Assemble a TrackVis `.trk` file from `streamlines` (each a flat `Float32Array`
 * of `[x,y,z, x,y,z, …]` in voxel/model coordinates) and a voxel→RAS-mm
 * `affine`. See the file header for the conventions; the output matches the
 * Python nibabel path byte-for-byte.
 */
export function writeTrk(
  streamlines: readonly Float32Array[],
  affine: Affine4x4,
): Uint8Array<ArrayBuffer> {
  const voxelSizes = columnNorms(affine);

  // Reference grid: bound the vertices (they are in voxel space). int16 range,
  // at least 1 -- matches `_trk_header`'s `ceil(abs(max)) + 1`.
  const maxAbs = [1, 1, 1];
  for (const s of streamlines) {
    for (let v = 0; v < s.length; v += 3) {
      maxAbs[0] = Math.max(maxAbs[0], Math.abs(s[v]));
      maxAbs[1] = Math.max(maxAbs[1], Math.abs(s[v + 1]));
      maxAbs[2] = Math.max(maxAbs[2], Math.abs(s[v + 2]));
    }
  }
  const dims = maxAbs.map((m) => Math.min(32767, Math.ceil(m) + 1));
  const order = voxelOrder(affine);

  let bodyBytes = 0;
  for (const s of streamlines) {
    // int32 point count + one xyz float triple per vertex.
    bodyBytes += 4 + (s.length / 3) * 3 * 4;
  }

  const buffer = new ArrayBuffer(TRK_HEADER_BYTES + bodyBytes);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // id_string "TRACK\0"
  bytes.set([0x54, 0x52, 0x41, 0x43, 0x4b, 0x00], 0);
  for (let i = 0; i < 3; ++i) view.setInt16(6 + i * 2, dims[i], true);
  for (let i = 0; i < 3; ++i) view.setFloat32(12 + i * 4, voxelSizes[i], true);
  // origin (24), n_scalars (36), scalar names, n_properties (238), property
  // names, and reserved regions are all left zero.
  // vox_to_ras: 4×4 float32, row-major, at offset 440.
  let o = 440;
  for (let r = 0; r < 4; ++r) {
    for (let c = 0; c < 4; ++c) {
      view.setFloat32(o, affine[r][c], true);
      o += 4;
    }
  }
  // voxel_order[4] at 948.
  for (let i = 0; i < order.length && i < 4; ++i) {
    bytes[948 + i] = order.charCodeAt(i);
  }
  view.setInt32(988, streamlines.length, true); // n_count
  view.setInt32(992, 2, true); // version
  view.setInt32(996, TRK_HEADER_BYTES, true); // hdr_size

  // Body: each streamline is an int32 point count then voxel-centre voxmm points.
  let p = TRK_HEADER_BYTES;
  for (const s of streamlines) {
    const numVertices = s.length / 3;
    view.setInt32(p, numVertices, true);
    p += 4;
    for (let v = 0; v < numVertices; ++v) {
      view.setFloat32(p, (s[v * 3] + 0.5) * voxelSizes[0], true);
      view.setFloat32(p + 4, (s[v * 3 + 1] + 0.5) * voxelSizes[1], true);
      view.setFloat32(p + 8, (s[v * 3 + 2] + 0.5) * voxelSizes[2], true);
      p += 12;
    }
  }
  return bytes;
}
