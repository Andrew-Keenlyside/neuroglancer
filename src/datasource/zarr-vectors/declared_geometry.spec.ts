/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import { resolveDeclaredGeometry } from "#src/datasource/zarr-vectors/declared_geometry.js";

const NO_LINKS = { hasLinks: false };
const EDGE_LINKS = { hasLinks: true, linkWidth: 2 };
const FACE_LINKS = { hasLinks: true, linkWidth: 3 };

describe("resolveDeclaredGeometry", () => {
  it("takes the single declared kind without consulting the arrays", () => {
    const r = resolveDeclaredGeometry(["streamline"], NO_LINKS);
    expect(r.kind).toBe("streamline");
    expect(r.skipped).toEqual([]);
    expect(r.ambiguous).toBe(false);
  });

  it("keeps a store readable after a half-finished add_geometry()", () => {
    // The writer stamps the new type into the root metadata before it fails, so
    // a working point cloud ends up declaring a graph it does not contain.
    // Refusing the store here would lose the data that IS there.
    const r = resolveDeclaredGeometry(["point_cloud", "graph"], NO_LINKS);
    expect(r.kind).toBe("point_cloud");
    expect(r.skipped).toEqual(["graph"]);
    expect(r.ambiguous).toBe(false);
  });

  it("uses face-arity links to pick the surface", () => {
    const r = resolveDeclaredGeometry(["streamline", "mesh"], FACE_LINKS);
    expect(r.kind).toBe("mesh");
    expect(r.skipped).toEqual(["streamline"]);
    expect(r.ambiguous).toBe(false);
  });

  it("uses pair-arity links to rule the surface out", () => {
    const r = resolveDeclaredGeometry(["mesh", "graph"], EDGE_LINKS);
    expect(r.kind).toBe("graph");
    expect(r.ambiguous).toBe(false);
  });

  it("falls back to declaration order and says so when the arrays cannot decide", () => {
    // Both are pair-arity link geometries; nothing on disk separates them.
    const r = resolveDeclaredGeometry(["skeleton", "graph"], EDGE_LINKS);
    expect(r.kind).toBe("skeleton");
    expect(r.ambiguous).toBe(true);
    expect(r.skipped).toEqual(["graph"]);
  });

  it("reports unrecognised types separately from skipped ones", () => {
    const r = resolveDeclaredGeometry(["streamline", "hypercube"], EDGE_LINKS);
    expect(r.kind).toBe("streamline");
    expect(r.unsupported).toEqual(["hypercube"]);
    expect(r.skipped).toEqual([]);
  });

  it("throws only when nothing declared is recognisable", () => {
    expect(() => resolveDeclaredGeometry(["hypercube"], NO_LINKS)).toThrow(
      /no recognised geometry type/,
    );
  });
});
