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

if (!("WebGL2RenderingContext" in globalThis)) {
  Object.defineProperty(globalThis, "WebGL2RenderingContext", {
    value: new Proxy(class WebGL2RenderingContext {} as any, {
      get(target, property, receiver) {
        if (Reflect.has(target, property)) {
          return Reflect.get(target, property, receiver);
        }
        return 0;
      },
    }),
    configurable: true,
  });
}

const { isUnsafeBareAttributeAlias } = await import(
  "#src/skeleton/frontend.js"
);

describe("isUnsafeBareAttributeAlias", () => {
  it("withholds the bare alias for swizzle-shaped attribute names", () => {
    // The real case: a tractogram shipping a per-vertex `z`. `#define z` is
    // preprocessor-level, so it rewrote `d.z` in the colour-by-direction
    // default into a member that does not exist -- the shader failed to
    // compile and the layer fell back to per-object hash colours.
    for (const name of ["x", "y", "z", "w", "r", "g", "b", "a", "s", "t"]) {
      expect(isUnsafeBareAttributeAlias(name)).toBe(true);
    }
  });

  it("withholds it for multi-component swizzles too", () => {
    // `.xy` / `.rgb` are just as much a member access as `.z`.
    expect(isUnsafeBareAttributeAlias("xy")).toBe(true);
    expect(isUnsafeBareAttributeAlias("rgb")).toBe(true);
    expect(isUnsafeBareAttributeAlias("xyzw")).toBe(true);
  });

  it("keeps it for names that cannot be a swizzle", () => {
    // Ordinary attribute names stay usable bare, as they always were --
    // withholding more than necessary would break existing hand-written
    // shaders for no gain.
    for (const name of ["arc_length", "fa", "tangent", "curvature", "xyzwr"]) {
      expect(isUnsafeBareAttributeAlias(name)).toBe(false);
    }
  });

  it("does not mix components from different swizzle sets", () => {
    // `.xr` is not a legal swizzle, so `#define xr` cannot corrupt a member
    // access.
    expect(isUnsafeBareAttributeAlias("xr")).toBe(false);
    expect(isUnsafeBareAttributeAlias("")).toBe(false);
  });
});
