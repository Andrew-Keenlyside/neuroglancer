/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
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
import { computeEllipsoidRadii } from "#src/annotation/index.js";

const f32 = (...xs: number[]) => Float32Array.from(xs);

describe("computeEllipsoidRadii", () => {
  it("takes the absolute per-axis distance from centre to corner", () => {
    // Fully constrained drag: every axis differs, so nothing is filled in.
    const radii = computeEllipsoidRadii(f32(10, 20, 30), f32(13, 16, 35));
    expect(Array.from(radii)).toEqual([3, 4, 5]);
  });

  it("is sign-independent (corner may be on either side of the centre)", () => {
    const pos = computeEllipsoidRadii(f32(0, 0, 0), f32(2, 3, 4));
    const neg = computeEllipsoidRadii(f32(0, 0, 0), f32(-2, -3, -4));
    expect(Array.from(neg)).toEqual(Array.from(pos));
  });

  it("gives an axis the drag cannot constrain the largest constrained radius", () => {
    // A slice-view drag: both endpoints share the z of the plane, so without
    // this the ellipsoid would be a zero-thickness disc that only renders at
    // all because the shader clamps radii to 1e-3.
    const radii = computeEllipsoidRadii(f32(10, 20, 30), f32(15, 24, 30));
    expect(Array.from(radii)).toEqual([5, 4, 5]);
    expect(radii[2]).toBeGreaterThan(0);
  });

  it("turns an in-plane circular drag into a sphere", () => {
    // Equal in-plane extents => all three radii equal => sphere, which is what
    // a TrackVis-style spherical ROI needs from a single slice-view gesture.
    const radii = computeEllipsoidRadii(f32(0, 0, 50), f32(7, 7, 50));
    expect(Array.from(radii)).toEqual([7, 7, 7]);
  });

  it("fills every unconstrained axis, not just one", () => {
    const radii = computeEllipsoidRadii(f32(1, 2, 3), f32(6, 2, 3));
    expect(Array.from(radii)).toEqual([5, 5, 5]);
  });

  it("leaves the pre-drag state alone rather than inventing a radius", () => {
    // Cursor still on the centre: degenerate, but that is "not dragged yet",
    // not a collapsed axis to repair.
    const radii = computeEllipsoidRadii(f32(4, 5, 6), f32(4, 5, 6));
    expect(Array.from(radii)).toEqual([0, 0, 0]);
  });

  it("does not mutate its inputs", () => {
    const center = f32(10, 20, 30);
    const corner = f32(15, 24, 30);
    computeEllipsoidRadii(center, corner);
    expect(Array.from(center)).toEqual([10, 20, 30]);
    expect(Array.from(corner)).toEqual([15, 24, 30]);
  });

  it("handles ranks other than 3", () => {
    const radii = computeEllipsoidRadii(f32(0, 0), f32(3, 0));
    expect(Array.from(radii)).toEqual([3, 3]);
  });
});
