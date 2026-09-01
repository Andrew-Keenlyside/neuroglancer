/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  packedAttributeAccessorCode,
  packedAttributePropExpr,
  packedAttributeVaryings,
} from "#src/skeleton/packed_attributes.js";

describe("packedAttributeAccessorCode", () => {
  it("addresses the run attribute-major, by the per-chunk stride", () => {
    const code = packedAttributeAccessorCode("readValue");
    expect(code).toContain(
      "readValue(attributeIndex * uPackedAttributeStride + vertexIndex)",
    );
  });
});

describe("packedAttributeVaryings", () => {
  it("costs one flat varying for points, whatever the attribute count", () => {
    const { varyings, vertexMain } = packedAttributeVaryings({
      mode: "point",
      vertexIndexExpr: "vertexIndex",
    });
    expect(varyings).toEqual([
      {
        type: "highp uint",
        name: "vPackedVertexIndex",
        interpolationMode: "flat",
      },
    ]);
    expect(vertexMain).toBe("vPackedVertexIndex = vertexIndex;\n");
  });

  it("carries a smooth coefficient for lines, so values still interpolate", () => {
    // The pair is flat (indices must not be interpolated); the coefficient is
    // not (it IS the interpolation).
    const { varyings, vertexMain } = packedAttributeVaryings({
      mode: "line",
      pairExpr: "aVertexIndex",
      endpointCoefficientExpr: "getLineEndpointCoefficient()",
    });
    expect(varyings.map((v) => [v.name, v.interpolationMode])).toEqual([
      ["vPackedVertexPair", "flat"],
      ["vPackedLineCoefficient", ""],
    ]);
    expect(vertexMain).toContain("vPackedVertexPair = aVertexIndex;");
    expect(vertexMain).toContain(
      "vPackedLineCoefficient = getLineEndpointCoefficient();",
    );
  });

  it("builds one-hot corner weights for faces", () => {
    const { vertexMain } = packedAttributeVaryings({
      mode: "face",
      triExpr: "aFaceIndex",
      cornerExpr: "corner",
    });
    expect(vertexMain).toContain("vPackedVertexTri = aFaceIndex;");
    expect(vertexMain).toContain(
      "vPackedBarycentric = vec3(corner == 0u ? 1.0 : 0.0, corner == 1u ? 1.0 : 0.0, corner == 2u ? 1.0 : 0.0);",
    );
  });
});

describe("packedAttributePropExpr", () => {
  it("reads one texel for a point", () => {
    expect(
      packedAttributePropExpr(7, { mode: "point", vertexIndexExpr: "v" }),
    ).toBe("readPackedAttribute(7u, vPackedVertexIndex)");
  });

  it("blends the endpoints for a line", () => {
    expect(
      packedAttributePropExpr(0, {
        mode: "line",
        pairExpr: "aVertexIndex",
        endpointCoefficientExpr: "c",
      }),
    ).toBe(
      "mix(readPackedAttribute(0u, vPackedVertexPair.x), " +
        "readPackedAttribute(0u, vPackedVertexPair.y), vPackedLineCoefficient)",
    );
  });

  it("weights the three corners for a face", () => {
    const expr = packedAttributePropExpr(2, {
      mode: "face",
      triExpr: "aFaceIndex",
      cornerExpr: "corner",
    });
    expect(expr).toContain("readPackedAttribute(2u, vPackedVertexTri.x)");
    expect(expr).toContain("vPackedBarycentric.z");
  });

  it("keeps attribute indices distinct — the aliasing bug this guards", () => {
    const a = packedAttributePropExpr(0, {
      mode: "point",
      vertexIndexExpr: "v",
    });
    const b = packedAttributePropExpr(39, {
      mode: "point",
      vertexIndexExpr: "v",
    });
    expect(a).not.toBe(b);
  });
});
