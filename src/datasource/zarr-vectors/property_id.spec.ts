/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import { parseAnnotationPropertyId } from "#src/annotation/index.js";
import { toAnnotationPropertyId } from "#src/datasource/zarr-vectors/property_id.js";

describe("toAnnotationPropertyId", () => {
  it("passes through names that are already valid", () => {
    const used = new Set<string>();
    expect(toAnnotationPropertyId("fa", used)).toBe("fa");
    expect(toAnnotationPropertyId("gene_Sst", used)).toBe("gene_Sst");
  });

  it("rewrites characters that are illegal in GLSL identifiers", () => {
    const used = new Set<string>();
    // Real MERFISH gene panels ship names like this.
    expect(toAnnotationPropertyId("gene_H2-Q2", used)).toBe("gene_H2_Q2");
    expect(toAnnotationPropertyId("mean fa (%)", used)).toBe("mean_fa____");
  });

  it("prefixes names that do not start with a lowercase letter", () => {
    const used = new Set<string>();
    expect(toAnnotationPropertyId("Gad1", used)).toBe("p_Gad1");
    expect(toAnnotationPropertyId("3prime", used)).toBe("p_3prime");
    expect(toAnnotationPropertyId("_x", used)).toBe("p__x");
  });

  it("disambiguates names that collide after rewriting", () => {
    const used = new Set<string>();
    expect(toAnnotationPropertyId("gene-1", used)).toBe("gene_1");
    expect(toAnnotationPropertyId("gene.1", used)).toBe("gene_1_2");
    expect(toAnnotationPropertyId("gene_1", used)).toBe("gene_1_3");
  });

  it("always produces an id neuroglancer accepts", () => {
    const used = new Set<string>();
    for (const name of ["gene_H2-Q2", "Gad1", "3prime", "α-syn", "x.y z"]) {
      const id = toAnnotationPropertyId(name, used);
      expect(parseAnnotationPropertyId(id)).toBe(id);
    }
  });
});
