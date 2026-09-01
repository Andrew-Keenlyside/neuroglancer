/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Choosing which declared geometry a store's arrays actually hold.
 *
 * A ZVF store's root `zarr_vectors.geometry_types` is a LIST, and the writer
 * library has an `add_geometry()` for multi-geometry ("composite") stores that
 * would put each extra geometry in namespaced `vertices_<type>/` arrays. That
 * writer does not work: `add_geometry` needs the same path to be both a Zarr
 * group and a Zarr array, raises before writing any vertex data, and the
 * library marks the module unfinished for exactly that reason
 * (`zarr_vectors/_stability.py`: "add_geometry currently raises on its own
 * documented example"). So no store in the wild has a `vertices_<type>` array.
 *
 * What DOES reach users is the wreckage: `add_geometry` appends the new type to
 * the root metadata BEFORE it fails, so an aborted call leaves a perfectly good
 * single-geometry store whose `geometry_types` claims two. Refusing to open
 * that store -- which is what "exactly one geometry type" did -- loses the data
 * that is actually there over metadata that is merely aspirational.
 *
 * Hence: pick the kind the on-disk arrays are consistent with, render it, and
 * report the rest rather than failing.
 *
 * Lives in its own module so the choice can be unit-tested without the
 * WebGL-coupled datasource.
 */

import type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";

/** What the level's arrays reveal about the geometry they hold. */
export interface GeometryObservations {
  /** Whether `links/0` exists at all. */
  readonly hasLinks: boolean;
  /** `links/0`'s declared `link_width`, when it has one. */
  readonly linkWidth?: number;
}

export interface DeclaredGeometryResolution {
  /** The kind the primary arrays are read as. */
  readonly kind: ZarrVectorsGeometryKind;
  /** Declared kinds with no readable arrays of their own. */
  readonly skipped: string[];
  /** Declared strings this reader does not know at all. */
  readonly unsupported: string[];
  /** True when the arrays did not single out `kind` and declaration order did. */
  readonly ambiguous: boolean;
}

/**
 * Resolve which declared geometry to read the primary arrays as.
 *
 * With one declared kind this is that kind. With several, the observations
 * decide where they can: a links family of arity >= 3 can only be a surface,
 * and a store with no links at all can only be a point cloud. Where they cannot
 * decide, declaration order wins and `ambiguous` says so, because guessing
 * silently is worse than a caller that can warn.
 */
export function resolveDeclaredGeometry(
  declared: readonly string[],
  observations: GeometryObservations,
): DeclaredGeometryResolution {
  const known = new Set(Object.keys(KIND_CAPABILITIES));
  const unsupported = declared.filter((g) => !known.has(g));
  const candidates = declared.filter((g) =>
    known.has(g),
  ) as ZarrVectorsGeometryKind[];
  if (candidates.length === 0) {
    throw new Error(
      `zarr-vectors: no recognised geometry type in ${JSON.stringify(declared)}; ` +
        `expected one of ${JSON.stringify([...known])}`,
    );
  }
  if (candidates.length === 1) {
    return {
      kind: candidates[0],
      skipped: [],
      unsupported,
      ambiguous: false,
    };
  }

  const { hasLinks, linkWidth } = observations;
  const consistent = candidates.filter((kind) => {
    const caps = KIND_CAPABILITIES[kind];
    // A surface needs face-arity links; a curve needs pair-arity ones.
    if (linkWidth !== undefined) {
      const wantsFaces = caps.primitive === "triangles";
      if (wantsFaces !== linkWidth >= 3) return false;
    }
    // Something with no connectivity cannot own a links family, and something
    // whose every edge is explicit cannot be missing one.
    if (caps.edgeSource === "none" && hasLinks) return false;
    if (caps.edgeSource === "explicit" && !hasLinks) return false;
    return true;
  });

  const kind = consistent.length > 0 ? consistent[0] : candidates[0];
  return {
    kind,
    skipped: candidates.filter((g) => g !== kind),
    unsupported,
    ambiguous: consistent.length !== 1,
  };
}
