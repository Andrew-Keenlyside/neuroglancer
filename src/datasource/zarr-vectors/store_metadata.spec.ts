/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  computeChunkIndexBounds,
  formatAttributesFragment,
  groupIdForDocument,
  parseAttributesFragment,
  rememberSavedDocument,
  resolveDeclaredGeometry,
  savedDocumentFor,
  toAnnotationPropertyId,
} from "#src/datasource/zarr-vectors/store_metadata.js";
import { parseAnnotationPropertyId } from "#src/annotation/index.js";
import { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";

// ---------------------------------------------------------------- chunk_bounds

describe("computeChunkIndexBounds", () => {
  it("handles a fractional chunk_shape", () => {
    // The MERFISH case: a 0.5 mm grid over a ~12 x 8 x 6 mm brain. Truncating
    // the shape to an integer would give 0 and a bound of Infinity, and the
    // frustum walk would recurse until it blew the stack.
    const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
      [0.5378, 0.1033, 0.5055],
      [12.6255, 7.855, 5.875],
      [0.5, 0.5, 0.5],
    );
    expect(Array.from(lowerChunkBound)).toEqual([1, 0, 1]);
    expect(Array.from(upperChunkBound)).toEqual([26, 16, 12]);
  });

  it("indexes chunks around the world origin, so negative indices are fine", () => {
    const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
      [-30, -0.5, -100],
      [10, 0.5, -60],
      [10, 1, 25],
    );
    expect(Array.from(lowerChunkBound)).toEqual([-3, -1, -4]);
    expect(Array.from(upperChunkBound)).toEqual([1, 1, -2]);
  });

  it("leaves one chunk on a degenerate axis", () => {
    // A flat (2-D) store still has to draw: a zero-volume range makes the walk
    // terminate before it reaches a single chunk.
    const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
      [0, 0, 4],
      [10, 10, 4],
      [5, 5, 5],
    );
    expect(Array.from(lowerChunkBound)).toEqual([0, 0, 0]);
    expect(Array.from(upperChunkBound)).toEqual([2, 2, 1]);
  });

  it("rejects a non-positive chunk extent instead of producing Infinity", () => {
    expect(() =>
      computeChunkIndexBounds([0, 0, 0], [1, 1, 1], [1, 0, 1]),
    ).toThrow(/not a positive extent/);
  });
});

// ---------------------------------------------------------------- property_id

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

// ---------------------------------------------------------------- attributes_fragment

describe("the #attributes= fragment", () => {
  it("is absent when the URL has no fragment", () => {
    expect(parseAttributesFragment(undefined)).toBeUndefined();
    expect(parseAttributesFragment("")).toBeUndefined();
    expect(formatAttributesFragment(undefined)).toBe("");
  });

  it("parses a plain list, trimming whitespace", () => {
    expect(parseAttributesFragment("attributes=fa, md ,radius")).toEqual([
      "fa",
      "md",
      "radius",
    ]);
  });

  it("round-trips names that need percent-encoding", () => {
    // The formatted fragment is saved into the layer's JSON, so parse(format(x))
    // must be x -- otherwise reopening a saved link fails or selects the wrong
    // columns.
    for (const names of [
      ["gene_Sst"],
      ["odd name"],
      ["pct%"],
      ["has,comma", "plain"],
      ["100%", "a+b"],
    ]) {
      const fragment = formatAttributesFragment(names);
      expect(parseAttributesFragment(fragment.slice(1))).toEqual(names);
    }
  });

  it("keeps a percent-encoded comma as one name", () => {
    expect(parseAttributesFragment("attributes=has%2Ccomma,plain")).toEqual([
      "has,comma",
      "plain",
    ]);
  });

  it("takes a stray percent literally rather than failing the load", () => {
    // A hand-typed `pct%` is not valid percent-encoding; refusing to open the
    // layer over it would be worse than reading it as written.
    expect(parseAttributesFragment("attributes=pct%")).toEqual(["pct%"]);
  });

  it("rejects a fragment that is not an attribute list", () => {
    expect(() => parseAttributesFragment("something=else")).toThrow(
      /only supported fragment/,
    );
  });
});

// ---------------------------------------------------------------- declared_geometry

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

// ---------------------------------------------------------------- store_provenance

/**
 * @file Which store document each live group is backed by.
 */

const REF = { id: "doc-abc", createdAt: "2026-07-19T00:00:00.000Z" };

describe("store provenance", () => {
  it("round-trips a mapping in both directions", () => {
    const state = new RoiFilterState();
    const groupId = state.addGroup();
    rememberSavedDocument(state, groupId, REF);

    expect(savedDocumentFor(state, groupId)).toEqual(REF);
    expect(groupIdForDocument(state, "doc-abc")).toEqual(groupId);
  });

  it("reports nothing for an unmapped group or document", () => {
    const state = new RoiFilterState();
    const groupId = state.addGroup();
    expect(savedDocumentFor(state, groupId)).toBeUndefined();
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
  });

  it("stops reporting a document whose group was deleted", () => {
    // Otherwise the store checklist would keep the entry ticked, claiming a
    // dissection is on screen after the user removed it.
    const state = new RoiFilterState();
    const groupId = state.addGroup();
    rememberSavedDocument(state, groupId, REF);
    expect(groupIdForDocument(state, "doc-abc")).toEqual(groupId);

    state.removeGroup(groupId);
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
    // And again, now that the stale entry has been dropped.
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
  });

  it("does not confuse ids between layers", () => {
    // Group ids are per-RoiFilterState, so two layers both have a group 1.
    const a = new RoiFilterState();
    const b = new RoiFilterState();
    const inA = a.addGroup();
    const inB = b.addGroup();
    expect(inA).toEqual(inB);

    rememberSavedDocument(a, inA, REF);
    expect(groupIdForDocument(a, "doc-abc")).toEqual(inA);
    expect(groupIdForDocument(b, "doc-abc")).toBeUndefined();
    expect(savedDocumentFor(b, inB)).toBeUndefined();
  });

  it("re-saving a group replaces its mapping rather than accumulating", () => {
    const state = new RoiFilterState();
    const groupId = state.addGroup();
    rememberSavedDocument(state, groupId, REF);
    const updated = { id: "doc-xyz", createdAt: REF.createdAt };
    rememberSavedDocument(state, groupId, updated);

    expect(savedDocumentFor(state, groupId)).toEqual(updated);
    expect(groupIdForDocument(state, "doc-xyz")).toEqual(groupId);
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
  });

  it("tracks several documents in one layer independently", () => {
    const state = new RoiFilterState();
    const first = state.addGroup();
    const second = state.addGroup();
    rememberSavedDocument(state, first, REF);
    rememberSavedDocument(state, second, { id: "doc-2", createdAt: "x" });

    expect(groupIdForDocument(state, "doc-abc")).toEqual(first);
    expect(groupIdForDocument(state, "doc-2")).toEqual(second);

    state.removeGroup(first);
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
    expect(groupIdForDocument(state, "doc-2")).toEqual(second);
  });
});
