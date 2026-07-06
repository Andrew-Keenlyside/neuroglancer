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
import {
  computePhysicalUnitsPerScreenPixelAtPoint,
  getChunkSpacing,
  getMetersPerUnit,
  quantizeSpacingForArbitration,
} from "#src/skeleton/screen_size.js";

const IDENTITY_MVP = Float32Array.from([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
]);

describe("getChunkSpacing", () => {
  it("returns the minimum of the three axis sizes", () => {
    expect(getChunkSpacing(Float32Array.of(4, 2, 8))).toBe(2);
  });

  it("clamps to a small positive floor instead of returning zero", () => {
    expect(getChunkSpacing(Float32Array.of(0, 5, 5))).toBeGreaterThan(0);
    expect(getChunkSpacing(Float32Array.of(0, 5, 5))).toBeLessThan(1e-5);
  });
});

describe("quantizeSpacingForArbitration", () => {
  it("is idempotent on its own output", () => {
    const once = quantizeSpacingForArbitration(3.7);
    const twice = quantizeSpacingForArbitration(once);
    expect(twice).toBeCloseTo(once);
  });

  it("snaps nearby values to the same quantized bucket", () => {
    expect(quantizeSpacingForArbitration(4.01)).toBeCloseTo(
      quantizeSpacingForArbitration(4.0),
    );
  });

  it("distinguishes values a full octave apart", () => {
    expect(quantizeSpacingForArbitration(8)).not.toBeCloseTo(
      quantizeSpacingForArbitration(4),
    );
  });

  it("clamps non-positive input instead of producing NaN/-Infinity", () => {
    expect(Number.isFinite(quantizeSpacingForArbitration(0))).toBe(true);
    expect(Number.isFinite(quantizeSpacingForArbitration(-5))).toBe(true);
  });
});

describe("getMetersPerUnit", () => {
  it("returns 1 when no display dimension scales are present", () => {
    expect(getMetersPerUnit({})).toBe(1);
    expect(
      getMetersPerUnit({
        displayDimensionRenderInfo: { displayDimensionScales: undefined },
      }),
    ).toBe(1);
  });

  it("returns the smallest positive scale", () => {
    expect(
      getMetersPerUnit({
        displayDimensionRenderInfo: {
          displayDimensionScales: Float64Array.of(1e-6, 4e-9, 2e-6),
        },
      }),
    ).toBe(4e-9);
  });

  it("ignores non-finite or non-positive entries", () => {
    expect(
      getMetersPerUnit({
        displayDimensionRenderInfo: {
          displayDimensionScales: Float64Array.of(
            Number.NaN,
            -1,
            0,
            3e-9,
          ),
        },
      }),
    ).toBe(3e-9);
  });
});

describe("computePhysicalUnitsPerScreenPixelAtPoint", () => {
  it("returns 1/max(width,height) for an identity MVP", () => {
    const pixelSize = computePhysicalUnitsPerScreenPixelAtPoint(
      IDENTITY_MVP,
      100,
      50,
      Float32Array.of(1, 2, 3),
    );
    expect(pixelSize).toBeCloseTo(1 / 100);
  });

  it("returns Infinity when the point is behind the camera (w <= 0)", () => {
    const behindCamera = Float32Array.from(IDENTITY_MVP);
    behindCamera[15] = -1; // m33
    const pixelSize = computePhysicalUnitsPerScreenPixelAtPoint(
      behindCamera,
      100,
      100,
      Float32Array.of(0, 0, 0),
    );
    expect(pixelSize).toBe(Number.POSITIVE_INFINITY);
  });

  it("scales down with a larger viewport (more pixels covering the same world extent)", () => {
    const small = computePhysicalUnitsPerScreenPixelAtPoint(
      IDENTITY_MVP,
      100,
      100,
      Float32Array.of(0, 0, 0),
    );
    const large = computePhysicalUnitsPerScreenPixelAtPoint(
      IDENTITY_MVP,
      1000,
      1000,
      Float32Array.of(0, 0, 0),
    );
    expect(large).toBeLessThan(small);
  });

  it("divides by displayDimensionScales when supplied", () => {
    const withoutScales = computePhysicalUnitsPerScreenPixelAtPoint(
      IDENTITY_MVP,
      100,
      100,
      Float32Array.of(0, 0, 0),
    );
    const withScales = computePhysicalUnitsPerScreenPixelAtPoint(
      IDENTITY_MVP,
      100,
      100,
      Float32Array.of(0, 0, 0),
      Float64Array.of(2, 2, 2),
    );
    // Dividing the matrix's linear part by a larger per-axis scale shrinks
    // its screen-space Jacobian, which *increases* the returned pixel size.
    expect(withScales).toBeGreaterThan(withoutScales);
  });
});
