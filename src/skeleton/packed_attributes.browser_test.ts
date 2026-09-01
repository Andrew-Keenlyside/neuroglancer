/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Real-WebGL2 checks for the packed per-vertex attribute texture.
 *
 * The addressing is one line of GLSL, and getting it wrong is invisible from
 * everywhere else: the shader still compiles, links and draws -- it just
 * colours by the wrong column, or by nothing. These tests fetch known values
 * through the same generated code the skeleton shader uses, including
 * attribute indices far past the ~10 a texture-unit-per-attribute layer could
 * bind at all.
 */

import { describe, expect, it } from "vitest";
import {
  PACKED_ATTRIBUTE_STRIDE_UNIFORM,
  packedAttributeAccessorCode,
} from "#src/skeleton/packed_attributes.js";
import { DataType } from "#src/util/data_type.js";
import { fragmentShaderTest } from "#src/webgl/shader_testing.js";
import {
  OneDimensionalTextureAccessHelper,
  setOneDimensionalTextureData,
  TextureFormat,
  computeTextureFormat,
} from "#src/webgl/texture_access.js";

const NUM_ATTRIBUTES = 40;
const NUM_VERTICES = 97;

/** Attribute-major packing, as the chunk serializer produces. */
function packedValues(): Float32Array {
  const data = new Float32Array(NUM_ATTRIBUTES * NUM_VERTICES);
  for (let a = 0; a < NUM_ATTRIBUTES; ++a) {
    for (let v = 0; v < NUM_VERTICES; ++v) {
      // Distinct per (attribute, vertex), and exactly representable in float32.
      data[a * NUM_VERTICES + v] = a * 1000 + v;
    }
  }
  return data;
}

describe("skeleton/packed_attributes", () => {
  it("fetches every attribute of every vertex from one texture", () => {
    fragmentShaderTest(
      { uAttributeIndex: "uint", uVertexIndex: "uint" },
      { outputValue: "float" },
      (tester) => {
        const { gl, builder } = tester;
        const helper = new OneDimensionalTextureAccessHelper("packedTest");
        helper.defineShader(builder);
        builder.addTextureSampler(
          "sampler2D",
          "uPackedAttributeSampler",
          Symbol("packedTest"),
        );
        builder.addUniform("highp uint", PACKED_ATTRIBUTE_STRIDE_UNIFORM);
        builder.addFragmentCode(
          helper.getAccessor(
            "readPackedAttributeValue",
            "uPackedAttributeSampler",
            DataType.FLOAT32,
            1,
          ),
        );
        builder.addFragmentCode(
          packedAttributeAccessorCode("readPackedAttributeValue"),
        );
        builder.setFragmentMain(
          "outputValue = readPackedAttribute(uAttributeIndex, uVertexIndex);",
        );
        tester.build();
        const { shader } = tester;
        shader.bind();

        const texture = gl.createTexture();
        gl.activeTexture(
          WebGL2RenderingContext.TEXTURE0 +
            shader.textureUnit(
              (shader as any).textureUnits.keys().next().value,
            ),
        );
        gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
        setOneDimensionalTextureData(
          gl,
          computeTextureFormat(new TextureFormat(), DataType.FLOAT32, 1),
          packedValues(),
        );
        gl.uniform1ui(
          shader.uniform(PACKED_ATTRIBUTE_STRIDE_UNIFORM),
          NUM_VERTICES,
        );

        try {
          const probes: Array<[number, number]> = [
            [0, 0],
            [0, NUM_VERTICES - 1],
            [1, 5],
            // Past the ten a texture-unit-per-attribute layer could bind, and
            // past the 16-unit WebGL2 floor.
            [17, 42],
            [NUM_ATTRIBUTES - 1, NUM_VERTICES - 1],
            [NUM_ATTRIBUTES - 1, 0],
          ];
          for (const [attributeIndex, vertexIndex] of probes) {
            tester.execute({
              uAttributeIndex: attributeIndex,
              uVertexIndex: vertexIndex,
            });
            expect(
              tester.values.outputValue,
              `attribute ${attributeIndex} of vertex ${vertexIndex}`,
            ).toEqual(attributeIndex * 1000 + vertexIndex);
          }
        } finally {
          gl.deleteTexture(texture);
        }
      },
    );
  });

  it("keeps attributes independent — a wrong stride would alias them", () => {
    fragmentShaderTest(
      { uAttributeIndex: "uint", uVertexIndex: "uint" },
      { outputValue: "float" },
      (tester) => {
        const { gl, builder } = tester;
        const helper = new OneDimensionalTextureAccessHelper("packedTest2");
        helper.defineShader(builder);
        builder.addTextureSampler(
          "sampler2D",
          "uPackedAttributeSampler",
          Symbol("packedTest2"),
        );
        builder.addUniform("highp uint", PACKED_ATTRIBUTE_STRIDE_UNIFORM);
        builder.addFragmentCode(
          helper.getAccessor(
            "readPackedAttributeValue",
            "uPackedAttributeSampler",
            DataType.FLOAT32,
            1,
          ),
        );
        builder.addFragmentCode(
          packedAttributeAccessorCode("readPackedAttributeValue"),
        );
        builder.setFragmentMain(
          "outputValue = readPackedAttribute(uAttributeIndex, uVertexIndex);",
        );
        tester.build();
        const { shader } = tester;
        shader.bind();
        const texture = gl.createTexture();
        gl.activeTexture(
          WebGL2RenderingContext.TEXTURE0 +
            shader.textureUnit(
              (shader as any).textureUnits.keys().next().value,
            ),
        );
        gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
        setOneDimensionalTextureData(
          gl,
          computeTextureFormat(new TextureFormat(), DataType.FLOAT32, 1),
          packedValues(),
        );
        gl.uniform1ui(
          shader.uniform(PACKED_ATTRIBUTE_STRIDE_UNIFORM),
          NUM_VERTICES,
        );
        try {
          const seen = new Set<number>();
          for (let a = 0; a < NUM_ATTRIBUTES; ++a) {
            tester.execute({ uAttributeIndex: a, uVertexIndex: 3 });
            seen.add(tester.values.outputValue as number);
          }
          expect(seen.size).toEqual(NUM_ATTRIBUTES);
        } finally {
          gl.deleteTexture(texture);
        }
      },
    );
  });
});
