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

import type {
  AnnotationPropertySpec,
  AnnotationType,
} from "#src/annotation/index.js";
import type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";

export type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";

/**
 * Numpy-style dtype string for a per-vertex attribute as written by
 * zarr-vectors.
 *
 * Every one of these decodes to `float32` before it reaches the GPU (see
 * `decodeAttributeToFloat32`), which is what lets an arbitrary number of
 * attributes share one texture and what makes `prop_<name>()` return a plain
 * `float` for all of them. The 64-bit members are downcast: a MERFISH store's
 * obs columns are float64 scores and int64 category codes, and dropping them
 * for want of a 32-bit spelling left the store with nothing to colour by but
 * whichever genes sorted first alphabetically. The same downcast is already
 * how per-OBJECT attributes are handled (`reinterpretWideToFloat32`).
 *
 * Precision: values above 2^24 are no longer exact. That is immaterial for
 * scores, coordinates and category codes, and id-shaped columns are excluded
 * from the one place exactness matters -- `resolveVertexIdColumn` refuses
 * them rather than truncating ids silently.
 */
export type ZarrVectorsAttributeDtype =
  | "float32"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "float64"
  | "int64"
  | "uint64";

/** The 64-bit members of {@link ZarrVectorsAttributeDtype}, downcast on decode. */
export const WIDE_ATTRIBUTE_DTYPES = new Set<ZarrVectorsAttributeDtype>([
  "float64",
  "int64",
  "uint64",
]);

/**
 * How the annotation renderer should combine spatial-index levels of a
 * multi-resolution pyramid.
 *
 * - "additive": levels are non-overlapping; renderer accumulates them
 *   into the buffer (precomputed annotations' classic behavior).  Each
 *   point lives at exactly one level.
 * - "replace": levels are complete representations at decreasing
 *   fidelity; renderer picks one level per zoom (image-style).  Right
 *   choice for metanode pyramids — drawing a metanode and its children
 *   simultaneously would double-count.
 */
export type ZarrVectorsPyramidMode = "additive" | "replace";

export class ZarrVectorsAnnotationSourceParameters {
  rank: number;
  type: AnnotationType;
  properties: AnnotationPropertySpec[];
  pyramidMode: ZarrVectorsPyramidMode;
  static RPC_ID = "zarr-vectors/AnnotationSource";
}

export class ZarrVectorsAnnotationSpatialIndexSourceParameters {
  // Pipeline URL of the level directory (ends with "/"), e.g.
  // ".../store.zvr/0/".
  baseUrl: string;
  rank: number;
  // Parallel arrays: attributeNames[i] is the directory name under
  // <baseUrl>/vertex_attributes/, and attributeDtypes[i] is the numpy dtype of
  // the chunk byte blob.  Index i in this list corresponds to property
  // index i on the parent AnnotationSource.
  attributeNames: string[];
  attributeDtypes: ZarrVectorsAttributeDtype[];
  static RPC_ID = "zarr-vectors/AnnotationSpatialIndexSource";
}

/**
 * How vertex-to-vertex edges are encoded inside a chunk.  Mirrors the
 * spec's root-level ``links_convention`` field.
 *
 * - "implicit_sequential": polyline / streamline — edges go vertex
 *   ``i`` → ``i+1`` inside each fragment; the chunk has no
 *   ``links/0/<chunk>`` array.
 * - "implicit_sequential_with_branches": skeleton — implicit sequential
 *   edges plus an explicit ``links/0/<chunk>`` array of branch edges.
 * - "explicit": all edges live in ``links/0/<chunk>`` (general graphs).
 */
export type ZarrVectorsLinksConvention =
  | "implicit_sequential"
  | "implicit_sequential_with_branches"
  | "explicit";

/**
 * Integer dtype for ``links/0/<chunk>``.  Writers pick the narrowest
 * width that covers ``n_vertices_in_chunk`` (see spec §7.5); the
 * reader honours whatever was declared in ``.zattrs.dtype``.  Unused
 * for stores with ``links_convention = "implicit_sequential"``.
 */
export type ZarrVectorsLinkDtype =
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "int64";

/**
 * Request/response RPC: one object's geometry as a node list, for the
 * spatial-skeleton editing UI. Named here so both sides of the worker boundary
 * agree without the frontend importing the backend module.
 */
export const ZARR_VECTORS_GET_OBJECT_NODES_RPC_ID =
  "zarr-vectors/getObjectNodes";

/**
 * Parameters for the spatially-indexed skeleton chunk source (pass 1).
 * Mirrors :class:`SpatiallyIndexedSkeletonSourceBackend` semantics: the
 * source enumerates chunks visible to the camera and downloads each one
 * via the zarr-vectors chunk-download orchestrator.
 *
 * One instance per **resolution level** in the multiscale pyramid.
 * ``baseUrl`` ends with ``/`` and points at the level directory (e.g.
 * ``".../store.zvr/0/"``).
 */
export class ZarrVectorsSpatialGeometrySourceParameters {
  baseUrl!: string;
  rank!: number;
  /** Parallel arrays describing per-vertex attribute discovery. */
  attributeNames!: string[];
  attributeDtypes!: ZarrVectorsAttributeDtype[];
  /**
   * Shader-facing name for each attribute, parallel to `attributeNames`.
   * Store attribute names are unrestricted (`gene_H2-Q2`), but the render
   * layer turns these into `prop_<id>()` GLSL macros, so the datasource maps
   * each on-disk name to a legal identifier (see
   * `zarr-vectors/property_id.ts`) and passes it here.  `attributeNames` stays
   * the directory to read.  Absent → fall back to `attributeNames`.
   */
  attributePropertyIds?: string[];
  /** From the store's ``zarr_vectors.links_convention``. */
  linksConvention!: ZarrVectorsLinksConvention;
  /** Drives tangent precomputation for streamline/polyline shaders. */
  geometryKind!: ZarrVectorsGeometryKind;
  /**
   * Declared ``links/0/.zattrs.dtype``.  Unused when
   * ``linksConvention === "implicit_sequential"`` — keep ``"int64"`` as
   * a defensive default in that case.
   */
  linkDtype!: ZarrVectorsLinkDtype;
  /**
   * Zero-based index of this level in the multiscale pyramid (finest = 0).
   * Read by neuroglancer's spatially-indexed skeleton render layer to
   * decide which source backs the user-selected `spatialSkeletonGridLevel`.
   * See [src/skeleton/source_selection.ts:51-57]
   * (#src/skeleton/source_selection.ts) for the consumer side.
   */
  gridIndex!: number;
  /** Whether `fragment_attributes/segment_id` exists for this level. */
  hasFragmentSegmentIds!: boolean;
  /**
   * Name of a per-vertex attribute carrying a meaningful integer id per vertex
   * (from the store's `zarr_vectors.vertex_id_attribute`).  Only consulted for
   * kinds without the discrete-object model, where each vertex is its own
   * segment; see `skeleton_chunk_download.fillPerVertexSegmentIds`.
   */
  vertexIdAttribute?: string;
  /**
   * Base URL of a loopback edit service, from the source URL's `#edit=`
   * fragment, and the store name to hand it. Present only when the layer was
   * opened for editing; the source reports `readonly` unless both are set.
   *
   * Editing cannot go through the kvstore -- it is read-only by construction
   * (`src/kvstore/index.ts`) -- so the write happens out of band and the layer
   * re-reads afterwards.
   */
  editServiceUrl?: string;
  editStore?: string;
  /**
   * The links family's declared `link_width`: 2 for edges, >= 3 for the faces
   * of a surface. Absent means 2, which is every non-mesh geometry.
   */
  linkWidth?: number;
  /**
   * Per-chunk-array grid geometry from `<level>/vertices/zarr.json` (v0.9.0
   * single-array format).  `chunkGridOrigin` is `attributes.chunk_grid_origin`
   * (the on-disk 0-based cell index of absolute chunk coord `C` is
   * `C - chunkGridOrigin`); `sharded` is `codecs[0].name === "sharding_indexed"`;
   * `shardChunkShape` is `chunk_grid.configuration.chunk_shape` (empty when
   * unsharded); `cellSeparator` is the chunk-key separator (default `/`).
   * See {@link shard_cell_reader.ChunkGridDescriptor}.
   */
  chunkGridOrigin!: number[];
  sharded!: boolean;
  shardChunkShape!: number[];
  cellSeparator!: string;
  /**
   * Which objects the NEXT COARSER level holds, as a bitset over the shared
   * object-id space (bit `id` set = present). Empty for the coarsest level.
   *
   * This is the "backbone" half of the object-admission test: an object that
   * also survives into a coarser level is drawn unconditionally, so the picture
   * never loses a tract that a coarser level would have shown. Everything new
   * at this level is rationed instead, by `objectRank` against a fraction
   * supplied at request time. See `object_admission.ts`.
   *
   * A bitset because this crosses to the worker: 503k objects is 63 KB packed
   * against 2 MB as counts. Absent for a store with no per-level membership
   * data, which disables per-object admission and keeps whole-level selection.
   */
  coarserMembership?: Uint8Array;
  /**
   * Whether this store's levels PARTITION the objects between them, so that
   * drawing several levels at once draws each object exactly once.
   *
   * True exactly when {@link coarserMembership} is populated, and carried
   * separately only because the generic skeleton layer -- which decides whether
   * to draw the union of levels -- must be able to ask the question without
   * knowing about zarr-vectors bitsets. See
   * `getSpatiallyIndexedSkeletonPartitionsObjects` in
   * `skeleton/source_selection.ts`.
   *
   * False for a plain RESOLUTION pyramid (`object_sparsity` 1.0 at every
   * level: mesh and point-cloud stores, and any skeleton store without
   * `object_attributes/vertex_count`), where every level holds every object
   * and the union would superimpose decimated copies of the same geometry.
   */
  partitionsObjects?: boolean;
  static RPC_ID = "zarr-vectors/SpatiallyIndexedSkeletonSource";
}

/**
 * Parameters for the per-segment (object-keyed) skeleton chunk source
 * used by the **pass-2** rendering path.  The source is parametrised
 * the same way as pass 1 (the underlying chunk format is identical),
 * but its ``download(chunk)`` is called once per visible object_id and
 * is responsible for resolving the object's manifest in
 * ``object_index/manifests`` and aggregating its fragments across
 * chunks.
 *
 * The manifest resolution is intentionally pinned to a single
 * resolution level (typically level 0).  Pass 1 is the level-aware
 * multiscale path; pass 2 always renders the highlighted objects at
 * full fidelity.
 */
export class ZarrVectorsObjectKeyedGeometrySourceParameters {
  baseUrl!: string;
  rank!: number;
  attributeNames!: string[];
  attributeDtypes!: ZarrVectorsAttributeDtype[];
  /**
   * Shader-facing name for each attribute, parallel to `attributeNames`.
   * Store attribute names are unrestricted (`gene_H2-Q2`), but the render
   * layer turns these into `prop_<id>()` GLSL macros, so the datasource maps
   * each on-disk name to a legal identifier (see
   * `zarr-vectors/property_id.ts`) and passes it here.  `attributeNames` stays
   * the directory to read.  Absent → fall back to `attributeNames`.
   */
  attributePropertyIds?: string[];
  linksConvention!: ZarrVectorsLinksConvention;
  geometryKind!: ZarrVectorsGeometryKind;
  linkDtype!: ZarrVectorsLinkDtype;
  /** Whether `fragment_attributes/segment_id` exists at level 0. */
  hasFragmentSegmentIds!: boolean;
  /**
   * Per-chunk-array grid geometry from level 0's `vertices/zarr.json`; see the
   * matching fields on {@link ZarrVectorsSpatialGeometrySourceParameters}.
   */
  chunkGridOrigin!: number[];
  sharded!: boolean;
  shardChunkShape!: number[];
  cellSeparator!: string;
  static RPC_ID = "zarr-vectors/ObjectKeyedSkeletonSource";
}
