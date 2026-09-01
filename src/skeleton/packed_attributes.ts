/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * GLSL for reading per-vertex attributes that share ONE texture instead of
 * taking a texture unit each.
 *
 * Split out of `skeleton/frontend.ts` so the addressing can be exercised
 * against a real WebGL2 context (`packed_attributes.browser_test.ts`) without
 * standing up a layer, a datasource and a chunk pipeline. The arithmetic is one
 * line and wrong is invisible: a shader that samples the wrong texel compiles,
 * links, draws, and silently colours by the wrong column.
 */

/**
 * A contiguous run of vertex attributes packed into one texture.
 *
 * Per-attribute textures put two ceilings on a layer. Texture units: WebGL2
 * guarantees only 16, and position, segment, the hash tables and the ROI tiers
 * spend several, leaving about ten for the store. Varying components: each
 * attribute also crossed the vertex/fragment boundary in its own varying, and
 * GLES 3.0 guarantees only 60 components. Neither has anything to do with how
 * many attributes the data has -- a MERFISH panel has one column per gene --
 * and exceeding either does not degrade, it draws nothing.
 *
 * Packing removes both: the run is one float32 texture addressed as
 * `attributeIndex * stride + vertexIndex`, sampled in the fragment stage, so
 * the only varyings are the vertex indices of the primitive.
 */
export interface PackedAttributeRange {
  /** Index into `vertexAttributes` of the first packed attribute. */
  start: number;
  /** How many consecutive entries are packed. */
  count: number;
}

/**
 * How a shader addresses the packed run: the vertices of the primitive being
 * drawn, and the weights to blend them by.
 *
 * A per-attribute varying used to do this implicitly -- the vertex stage read
 * the value and the rasteriser interpolated it. With the read moved to the
 * fragment stage the interpolation has to be spelled out, once per primitive
 * kind, in the same handful of varyings however many attributes there are.
 */
export type PackedAttributeInterp =
  | { mode: "point"; vertexIndexExpr: string }
  | { mode: "line"; pairExpr: string; endpointCoefficientExpr: string }
  | { mode: "face"; triExpr: string; cornerExpr: string };

/** Name of the generated two-argument accessor. */
export const PACKED_ATTRIBUTE_ACCESSOR = "readPackedAttribute";

/** Name of the uniform holding the per-chunk attribute stride (vertex count). */
export const PACKED_ATTRIBUTE_STRIDE_UNIFORM = "uPackedAttributeStride";

/**
 * GLSL wrapping a one-dimensional texture accessor as
 * `readPackedAttribute(attributeIndex, vertexIndex)`.
 *
 * `valueAccessorName` reads element `i` of the packed texture; the run is
 * attribute-major, so attribute `a` of vertex `v` is element
 * `a * stride + v` where the stride is that chunk's vertex count.
 */
export function packedAttributeAccessorCode(valueAccessorName: string): string {
  return `
highp float ${PACKED_ATTRIBUTE_ACCESSOR}(highp uint attributeIndex, highp uint vertexIndex) {
  return ${valueAccessorName}(attributeIndex * ${PACKED_ATTRIBUTE_STRIDE_UNIFORM} + vertexIndex);
}
`;
}

/** The varyings a packed run needs, and the vertex-stage code that fills them. */
export function packedAttributeVaryings(interp: PackedAttributeInterp): {
  varyings: Array<{
    type: string;
    name: string;
    interpolationMode: "flat" | "";
  }>;
  vertexMain: string;
} {
  switch (interp.mode) {
    case "point":
      return {
        varyings: [
          {
            type: "highp uint",
            name: "vPackedVertexIndex",
            interpolationMode: "flat",
          },
        ],
        vertexMain: `vPackedVertexIndex = ${interp.vertexIndexExpr};\n`,
      };
    case "line":
      return {
        varyings: [
          {
            type: "highp uvec2",
            name: "vPackedVertexPair",
            interpolationMode: "flat",
          },
          // Smooth, unlike the pair: 0 at endpoint A and 1 at endpoint B, so
          // the rasteriser hands each fragment its position along the line --
          // the blend the old per-attribute varying got for free.
          {
            type: "highp float",
            name: "vPackedLineCoefficient",
            interpolationMode: "",
          },
        ],
        vertexMain:
          `vPackedVertexPair = ${interp.pairExpr};\n` +
          `vPackedLineCoefficient = ${interp.endpointCoefficientExpr};\n`,
      };
    case "face":
      return {
        varyings: [
          {
            type: "highp uvec3",
            name: "vPackedVertexTri",
            interpolationMode: "flat",
          },
          // One-hot per corner interpolates to barycentric weights.
          {
            type: "highp vec3",
            name: "vPackedBarycentric",
            interpolationMode: "",
          },
        ],
        vertexMain:
          `vPackedVertexTri = ${interp.triExpr};\n` +
          `vPackedBarycentric = vec3(${interp.cornerExpr} == 0u ? 1.0 : 0.0, ` +
          `${interp.cornerExpr} == 1u ? 1.0 : 0.0, ` +
          `${interp.cornerExpr} == 2u ? 1.0 : 0.0);\n`,
      };
  }
}

/**
 * The expression `prop_<name>()` expands to for a packed attribute: its value
 * fetched at each vertex of the primitive and blended.
 */
export function packedAttributePropExpr(
  packedIndex: number,
  interp: PackedAttributeInterp,
): string {
  const fetch = (vertexExpr: string) =>
    `${PACKED_ATTRIBUTE_ACCESSOR}(${packedIndex}u, ${vertexExpr})`;
  switch (interp.mode) {
    case "point":
      return fetch("vPackedVertexIndex");
    case "line":
      return (
        `mix(${fetch("vPackedVertexPair.x")}, ` +
        `${fetch("vPackedVertexPair.y")}, vPackedLineCoefficient)`
      );
    case "face":
      return (
        `(${fetch("vPackedVertexTri.x")} * vPackedBarycentric.x + ` +
        `${fetch("vPackedVertexTri.y")} * vPackedBarycentric.y + ` +
        `${fetch("vPackedVertexTri.z")} * vPackedBarycentric.z)`
      );
  }
}
