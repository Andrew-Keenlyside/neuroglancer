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
  COLOR_BY_DIRECTION_SHADER,
  resolveSkeletonDefaultShader,
} from "#src/skeleton/default_shader.js";

const DIRECTIONAL = "void main() { emitRGB(abs(prop_tangent())); }";
const OTHER = "void main() { emitRGB(vec3(1.0)); }";

describe("resolveSkeletonDefaultShader", () => {
  it("returns undefined for a layer with no skeleton subsource", () => {
    expect(resolveSkeletonDefaultShader([])).toBeUndefined();
  });

  it("adopts the shader when the only source nominates one", () => {
    expect(resolveSkeletonDefaultShader([DIRECTIONAL])).toBe(DIRECTIONAL);
  });

  it("adopts it when every source nominates the same shader", () => {
    // The real pairing this protects: zarr-vectors pass-1 (spatially indexed)
    // plus pass-2 (object keyed) are the same geometry and agree, so enabling
    // pass-2 must not cost the directional colouring.
    expect(resolveSkeletonDefaultShader([DIRECTIONAL, DIRECTIONAL])).toBe(
      DIRECTIONAL,
    );
  });

  it("keeps the generic default when a source abstains", () => {
    // The mixed-source guard: a precomputed skeleton nominates nothing and has
    // no tangent attribute, so the streamline shader must not be applied to a
    // layer that also carries one -- it would fail to compile.
    expect(
      resolveSkeletonDefaultShader([DIRECTIONAL, undefined]),
    ).toBeUndefined();
  });

  it("keeps the generic default regardless of abstention order", () => {
    expect(
      resolveSkeletonDefaultShader([undefined, DIRECTIONAL]),
    ).toBeUndefined();
  });

  it("keeps the generic default when two sources disagree", () => {
    expect(resolveSkeletonDefaultShader([DIRECTIONAL, OTHER])).toBeUndefined();
  });

  it("treats a lone abstention as no opinion", () => {
    expect(resolveSkeletonDefaultShader([undefined])).toBeUndefined();
  });
});

describe("COLOR_BY_DIRECTION_SHADER", () => {
  it("reads the tangent through the namespaced prop_ accessor", () => {
    expect(COLOR_BY_DIRECTION_SHADER).toContain("prop_tangent()");
  });

  it("uses no swizzle, so an attribute name cannot rewrite it", () => {
    // A per-vertex attribute is also exposed as a bare `#define`, so a store
    // shipping a `z` column turned this shader's `d.z` into `d.<varying>`: it
    // stopped compiling and the layer silently drew per-object hash colours.
    // `isUnsafeBareAttributeAlias` withholds those aliases; keeping the
    // default swizzle-free means it never depended on that in the first place.
    expect(COLOR_BY_DIRECTION_SHADER).not.toMatch(/\.[xyzwrgbastpq]/);
  });
});
