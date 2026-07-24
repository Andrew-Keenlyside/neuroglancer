/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Canonical zarr-vectors geometry-kind union and per-kind capability
 * table.  Centralises the decisions that were previously scattered
 * through `skeleton_chunk.ts`, `skeleton_shader_bridge.ts`, and
 * `skeleton_frontend.ts` as ad-hoc `geometryKind === "streamline" ||
 * "polyline"` checks.  Adding a new kind (e.g. a future mesh
 * geometry that fits the same chunk layout) is a one-line edit to
 * the {@link KIND_CAPABILITIES} table.
 *
 * Naming note: the type is `ZarrVectorsGeometryKind` despite the
 * `skeleton_*` filenames that surround it.  Those filenames mirror the
 * `SkeletonChunk` data structure they back; the geometry kinds they
 * support are not strictly skeletons (graphs and streamlines also
 * route through the same chunk machinery).
 */

import { COLOR_BY_DIRECTION_SHADER } from "#src/skeleton/default_shader.js";

/**
 * Default skeleton-shader fragment-main text for **streamline** stores:
 * map the unit-sphere direction of the tangent at each vertex to an
 * RGB colour — the standard tractography "colour-by-direction"
 * convention.  Hosts of the `prop_tangent()` macro consume the
 * synthesised per-vertex tangent attribute the chunk decoder produces
 * for kinds where {@link hasSynthesisedTangent} is true.
 *
 * Aliases {@link COLOR_BY_DIRECTION_SHADER} (the datasource-neutral canonical
 * string) so the store-nominated default, the Rendering tab's "Direction"
 * preset, and the on-load `backgroundColorBy` sync are all byte-identical —
 * see that constant for why the identity matters.  `skeleton_shader_bridge.ts`
 * re-exports for callers that imported it from there before the refactor.
 */
export const DEFAULT_STREAMLINE_FRAGMENT_MAIN = COLOR_BY_DIRECTION_SHADER;

/**
 * The geometry-type strings zarr-vectors emits in its
 * `zarr_vectors.geometry_types` root attribute that route through the
 * spatially-indexed-skeleton render path.  Other kinds (notably
 * `"point_cloud"` and `"mesh"`) are handled elsewhere.
 */
export type ZarrVectorsGeometryKind =
  | "streamline"
  | "polyline"
  | "skeleton"
  | "graph";

/** Per-kind metadata consumed by the chunk decoder, shader bridge, and
 *  frontend chunk-source classes. */
export interface GeometryKindCapabilities {
  /**
   * Whether the chunk decoder should synthesise a per-vertex
   * `tangent` (vec3) by walking the fragment index in implicit-
   * sequential order.  True only for `streamline` and `polyline`
   * geometries — those have a well-defined walk direction at every
   * vertex, including endpoints, which we need so that cross-chunk
   * ghost-tangent signs match up across bridge edges (see
   * `appendGhostVertices`).
   */
  readonly hasWalkOrderTangent: boolean;
  /**
   * Whether the chunk decoder should synthesise per-vertex tangents
   * from the edge adjacency (degree-2 vertices get the central
   * difference; degree-1 endpoints get the direction to their lone
   * neighbour; branch points pick the central difference of the
   * first two listed neighbours).  True for `graph`: the on-disk
   * `links/0/<chunk>` array carries the full edge structure and
   * walk-order is undefined, but most non-branch vertices still
   * have a meaningful direction.  Skeletons could opt in similarly
   * but currently don't, preserving prior "no tangents for
   * skeletons" behaviour until a use case shows up.
   */
  readonly hasEdgeAdjacencyTangent: boolean;
  /**
   * Renderer's default shader text (paste-in for users); `undefined`
   * falls back to the neuroglancer-built-in segment-coloured default.
   * Every current kind synthesises a per-vertex tangent, so all four
   * auto-apply the RGB-by-tangent default -- matching the segmentation
   * layer's `roiFilter.backgroundColorBy` default of "direction", so a
   * freshly loaded tract layer renders colour-by-direction on the first
   * frame rather than the generic per-object hash colour. A future kind
   * without a tangent would nominate `undefined` and keep the hash default.
   *
   * The nomination still passes through {@link resolveSkeletonDefaultShader}'s
   * agreement check on the layer, so a layer that also draws a no-tangent
   * skeleton subsource (sharing the single skeleton shader) safely falls back
   * to the generic default rather than installing an uncompilable
   * `prop_tangent()`.
   */
  readonly defaultFragmentMain: string | undefined;
}

export const KIND_CAPABILITIES: Record<
  ZarrVectorsGeometryKind,
  GeometryKindCapabilities
> = {
  streamline: {
    hasWalkOrderTangent: true,
    hasEdgeAdjacencyTangent: false,
    defaultFragmentMain: DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  },
  polyline: {
    hasWalkOrderTangent: true,
    hasEdgeAdjacencyTangent: false,
    // Polylines carry a well-defined walk-order tangent at every vertex, just
    // like streamlines (tractography is commonly written as "polyline"), so they
    // auto-apply the same colour-by-direction default rather than the segment
    // hash colour.
    defaultFragmentMain: DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  },
  skeleton: {
    // Synthesise per-vertex tangents from edge adjacency so shaders can
    // `prop_tangent()` (colour-by-direction).  Branch points get the
    // central difference of their first two neighbours.  Auto-applies the
    // direction default like the other tangent-bearing kinds, matching the
    // "direction" background colour-by default (see `defaultFragmentMain`);
    // the agreement check keeps it safe in a mixed-subsource layer.
    hasWalkOrderTangent: false,
    hasEdgeAdjacencyTangent: true,
    defaultFragmentMain: DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  },
  graph: {
    hasWalkOrderTangent: false,
    hasEdgeAdjacencyTangent: true,
    // Edge-adjacency tangents give graph vertices a meaningful direction, so
    // graphs (a common tractography-as-connectivity encoding) auto-apply the
    // colour-by-direction default too, matching the background colour-by default.
    defaultFragmentMain: DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  },
};

/** True iff the geometry has *any* synthesised per-vertex tangent
 *  (regardless of which algorithm produced it).  Drives whether the
 *  shader bridge exposes `prop_tangent()`. */
export function hasSynthesisedTangent(kind: ZarrVectorsGeometryKind): boolean {
  const caps = KIND_CAPABILITIES[kind];
  return caps.hasWalkOrderTangent || caps.hasEdgeAdjacencyTangent;
}
