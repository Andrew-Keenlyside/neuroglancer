/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  buildVertexAttributeMap,
  zvPackedAttributeRange,
} from "#src/datasource/zarr-vectors/geometry_shader_bridge.js";
import { DataType } from "#src/util/data_type.js";

describe("buildVertexAttributeMap — `prop_<name>()` shader bridge", () => {
  it("prepends a synthesised tangent vec3 for streamline geometry", () => {
    const map = buildVertexAttributeMap({
      attributeNames: [],
      attributeDtypes: [],
      geometryKind: "streamline",
    });
    expect(Array.from(map.keys())).toEqual(["tangent"]);
    const tangent = map.get("tangent")!;
    expect(tangent.dataType).toBe(DataType.FLOAT32);
    expect(tangent.numComponents).toBe(3);
  });

  it("keys the map by the shader-facing id, not the on-disk name", () => {
    // Store attribute names are unrestricted; the map keys become
    // `prop_<key>()` GLSL macros, so the datasource supplies legal ids.
    const map = buildVertexAttributeMap({
      attributeNames: ["gene_H2-Q2"],
      attributeDtypes: ["float32"],
      attributePropertyIds: ["gene_H2_Q2"],
      geometryKind: "streamline",
    });
    expect(Array.from(map.keys())).toEqual(["tangent", "gene_H2_Q2"]);
    expect(map.get("gene_H2_Q2")!.dataType).toBe(DataType.FLOAT32);
  });

  it("falls back to the on-disk names when no ids are supplied", () => {
    const map = buildVertexAttributeMap({
      attributeNames: ["fa"],
      attributeDtypes: ["float32"],
      geometryKind: "streamline",
    });
    expect(Array.from(map.keys())).toEqual(["tangent", "fa"]);
  });

  it("prepends tangent for polyline geometry too", () => {
    const map = buildVertexAttributeMap({
      attributeNames: [],
      attributeDtypes: [],
      geometryKind: "polyline",
    });
    expect(Array.from(map.keys())).toEqual(["tangent"]);
  });

  it("prepends tangent for skeleton geometry (edge-adjacency tangent → prop_tangent())", () => {
    const map = buildVertexAttributeMap({
      attributeNames: [],
      attributeDtypes: [],
      geometryKind: "skeleton",
    });
    expect(Array.from(map.keys())).toEqual(["tangent"]);
  });

  it("prepends tangent for graph geometry (edge-adjacency tangent algorithm)", () => {
    const map = buildVertexAttributeMap({
      attributeNames: [],
      attributeDtypes: [],
      geometryKind: "graph",
    });
    expect(Array.from(map.keys())).toEqual(["tangent"]);
  });

  it("appends user-declared attributes after the tangent in declaration order", () => {
    const map = buildVertexAttributeMap({
      attributeNames: ["radius", "label"],
      attributeDtypes: ["float32", "uint16"],
      geometryKind: "streamline",
    });
    expect(Array.from(map.keys())).toEqual(["tangent", "radius", "label"]);
    expect(map.get("radius")).toEqual({
      dataType: DataType.FLOAT32,
      numComponents: 1,
    });
    expect(map.get("label")).toEqual({
      dataType: DataType.FLOAT32,
      numComponents: 1,
    });
  });

  it("presents every on-disk dtype as float32", () => {
    // The decoder converts everything (see `vertex_attribute_float.ts`), so
    // the shader sees one type. That is what lets a shader written against a
    // float gene column compile against an int32 section label -- integer
    // attributes used to arrive as an `int32_t` struct needing `toRaw()`.
    const map = buildVertexAttributeMap({
      attributeNames: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
      attributeDtypes: [
        "float32",
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "float64",
        "int64",
        "uint64",
      ],
      geometryKind: "skeleton",
    });
    for (const name of ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]) {
      expect(map.get(name)!.dataType).toBe(DataType.FLOAT32);
    }
  });

  it("ordering matches the backend's chunk.vertexAttributes packing convention", () => {
    // The backend (skeleton_backend.ts) packs:
    //   [tangent? , user_attr_0, user_attr_1, ...]
    // The frontend map produces:
    //   [tangent? , user_attr_0, user_attr_1, ...]
    // — same order, so the shader's `prop_<name>()` macros bind to the
    // right texture sampler.  Verify ordering invariant across all
    // geometry kinds.
    const cases: Array<{
      kind: "streamline" | "polyline" | "skeleton" | "graph";
      expectedKeys: string[];
    }> = [
      { kind: "streamline", expectedKeys: ["tangent", "u", "v"] },
      { kind: "polyline", expectedKeys: ["tangent", "u", "v"] },
      { kind: "skeleton", expectedKeys: ["tangent", "u", "v"] },
      { kind: "graph", expectedKeys: ["tangent", "u", "v"] },
    ];
    for (const { kind, expectedKeys } of cases) {
      const map = buildVertexAttributeMap({
        attributeNames: ["u", "v"],
        attributeDtypes: ["float32", "uint8"],
        geometryKind: kind,
      });
      expect(Array.from(map.keys())).toEqual(expectedKeys);
    }
  });
});

describe("DEFAULT_STREAMLINE_FRAGMENT_MAIN", () => {
  it("references prop_tangent() (the shader-bridge name buildVertexAttributeMap produces)", () => {
    expect(DEFAULT_STREAMLINE_FRAGMENT_MAIN).toContain("prop_tangent()");
  });

  it("maps direction components to [0, 1] via abs() (standard tractography colour-by-direction)", () => {
    // |d| ∈ [0, 1] for a unit-tangent → safe to feed into emitRGB.
    expect(DEFAULT_STREAMLINE_FRAGMENT_MAIN).toMatch(/abs\(/);
    expect(DEFAULT_STREAMLINE_FRAGMENT_MAIN).toContain("emitRGB");
  });

  it("has a void-main GLSL entry point", () => {
    expect(DEFAULT_STREAMLINE_FRAGMENT_MAIN).toMatch(/^void main\(\) \{/);
  });
});

describe("zvPackedAttributeRange", () => {
  it("covers the store's attributes, after position", () => {
    // A point cloud has no synthesised tangent: position, then the store's
    // columns, then the synthesised segment column.
    expect(
      zvPackedAttributeRange({
        attributeNames: ["gene_a", "gene_b", "brain_section_label"],
        geometryKind: "point_cloud",
      }),
    ).toEqual({ start: 1, count: 3 });
  });

  it("steps over the synthesised tangent when the kind has one", () => {
    expect(
      zvPackedAttributeRange({
        attributeNames: ["radius"],
        geometryKind: "streamline",
      }),
    ).toEqual({ start: 2, count: 1 });
  });

  it("packs nothing when the store declares no attributes", () => {
    expect(
      zvPackedAttributeRange({ attributeNames: [], geometryKind: "skeleton" }),
    ).toBeUndefined();
  });

  it("scales to a store far wider than the old texture-unit budget", () => {
    // 1136 columns is what Zhuang-ABCA-1 has; the point of the packing is that
    // the number stops mattering.
    const attributeNames = Array.from({ length: 1136 }, (_, i) => `gene_${i}`);
    expect(
      zvPackedAttributeRange({ attributeNames, geometryKind: "point_cloud" }),
    ).toEqual({ start: 1, count: 1136 });
  });
});
