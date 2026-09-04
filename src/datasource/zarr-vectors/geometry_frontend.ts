/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Frontend-side chunk source classes for zarr-vectors skeleton /
 * polyline / streamline rendering.  Each one is paired with a backend
 * class in `./geometry_backend.ts` via a matching ``RPC_ID`` on the
 * parameter type.
 *
 * - `ZarrVectorsSpatialGeometrySource` — the **pass-1** chunk
 *   source.  Subclass of neuroglancer's
 *   `SpatiallyIndexedSkeletonSource`; the backend pairs with
 *   `ZarrVectorsSpatialGeometrySourceBackend` and downloads one
 *   chunk per `(chunkGridPosition, lod)` pair.
 *
 * - `ZarrVectorsObjectKeyedGeometrySource` — the **pass-2** chunk source
 *   (intentionally **stubbed** in this slice).  Will subclass
 *   `SkeletonSource` and resolve object IDs via the
 *   `object_index/manifests` zarr-vlen-bytes array; that decoder lands
 *   in slice 4b.
 *
 * The synthesised `prop_tangent()` vertex-attribute exposure for the
 * default streamline shader is wired in slice 4d.  Here, both sources
 * keep neuroglancer's default `[vertexPositionAttribute, segmentAttribute]`
 * shape so the file compiles in isolation.
 */

import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import {
  ZarrVectorsObjectKeyedGeometrySourceParameters,
  ZarrVectorsSpatialGeometrySourceParameters,
  type ZarrVectorsAttributeDtype,
  type ZarrVectorsGeometryKind,
} from "#src/datasource/zarr-vectors/base.js";
import { ZARR_VECTORS_GET_OBJECT_NODES_RPC_ID } from "#src/datasource/zarr-vectors/base.js";
import { computeChunkIndexBounds } from "#src/datasource/zarr-vectors/store_metadata.js";
import {
  KIND_CAPABILITIES,
  hasSynthesisedTangent,
} from "#src/datasource/zarr-vectors/geometry_kind.js";
import {
  buildVertexAttributeMap,
  zvPackedAttributeRange,
} from "#src/datasource/zarr-vectors/geometry_shader_bridge.js";
import type { ObjectAdmission } from "#src/datasource/zarr-vectors/object_admission.js";
import { admissionForBudget } from "#src/datasource/zarr-vectors/object_admission.js";
import { computePyramidDensityScales } from "#src/datasource/zarr-vectors/pyramid_objects.js";
import type { ZarrVectorsEditTarget } from "#src/datasource/zarr-vectors/spatial_skeleton_edit.js";
import { makeZarrVectorsEditCommands } from "#src/datasource/zarr-vectors/spatial_skeleton_edit.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import type {
  SpatiallyIndexedSkeletonMetadata,
  SpatiallyIndexedSkeletonNode,
  SpatiallyIndexedSkeletonNodeBase,
} from "#src/skeleton/api.js";
import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import {
  SkeletonSource,
  type GeometryPrimitive,
  type PackedAttributeRange,
} from "#src/skeleton/frontend.js";
import {
  MultiscaleSpatiallyIndexedSkeletonSource,
  SPATIAL_SKELETON_SOURCE_OPTIONS,
  SpatiallyIndexedSkeletonSource,
  type SpatiallyIndexedSkeletonChunkSpecification,
} from "#src/skeleton/spatial_frontend.js";
import type { SliceViewSourceOptions } from "#src/sliceview/base.js";
import { makeSliceViewChunkSpecification } from "#src/sliceview/base.js";
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import type { SliceViewSingleResolutionSource } from "#src/sliceview/frontend.js";
import { DataType } from "#src/util/data_type.js";
import type { Borrowed } from "#src/util/disposable.js";
import { mat4, vec3 } from "#src/util/geom.js";
import { getShaderType } from "#src/webgl/shader_lib.js";
import {
  TextureFormat,
  computeTextureFormat,
} from "#src/webgl/texture_access.js";

// Re-export for callers (e.g. UI code) that need these without
// importing the heavy WebGL-coupled symbols from `skeleton/frontend.js`.
export { DEFAULT_STREAMLINE_FRAGMENT_MAIN } from "#src/datasource/zarr-vectors/geometry_kind.js";
export { buildVertexAttributeMap } from "#src/datasource/zarr-vectors/geometry_shader_bridge.js";

/**
 * One entry in the array shape `SpatiallyIndexedSkeletonSource`
 * exposes to the render layer.  The fields after `numComponents` exist
 * because the existing skeleton render layer pulls
 * `webglDataType` / `glslDataType` directly off this struct when
 * building shaders.
 */
interface ZvVertexAttributeRenderInfo {
  name: string;
  dataType: DataType;
  numComponents: number;
  webglDataType: number;
  glslDataType: string;
}

/**
 * On-GPU width of an attribute, for the level cost estimate. Uniform across
 * dtypes: every per-vertex attribute is decoded to float32 before upload (see
 * `vertex_attribute_float.ts`), so an int8 column costs the same 4 bytes a
 * gene does.
 */
const ATTR_GPU_BYTES = 4;

/**
 * Map every zarr-vectors attribute dtype to a WebGL2 scalar type enum.
 * The standard `skeleton/frontend.ts` helper only handles
 * FLOAT32/INT32/UINT32 (it throws for 8/16-bit widths), so we keep our
 * own table here covering all the dtypes zarr-vectors emits.
 */
function zvWebglDataType(dt: DataType): number {
  switch (dt) {
    case DataType.FLOAT32:
      return WebGL2RenderingContext.FLOAT;
    case DataType.UINT8:
      return WebGL2RenderingContext.UNSIGNED_BYTE;
    case DataType.INT8:
      return WebGL2RenderingContext.BYTE;
    case DataType.UINT16:
      return WebGL2RenderingContext.UNSIGNED_SHORT;
    case DataType.INT16:
      return WebGL2RenderingContext.SHORT;
    case DataType.UINT32:
      return WebGL2RenderingContext.UNSIGNED_INT;
    case DataType.INT32:
      return WebGL2RenderingContext.INT;
    default:
      throw new Error(`Unsupported attribute DataType for WebGL: ${dt}`);
  }
}

/**
 * Build the `VertexAttributeRenderInfo[]` shape the existing
 * spatially-indexed skeleton render layer expects.  Mirrors how the
 * backend (`geometry_backend.ts:ZarrVectorsSpatialGeometrySourceBackend
 * .download`) packs `chunk.vertexAttributes`: position (implicit, slot
 * 0), then synthesised `tangent` (streamline / polyline only), then
 * user-declared attributes in declaration order.
 *
 * Extends `SpatiallyIndexedSkeletonSource`'s baked-in `[position,
 * segment]` shape: we keep a `"segment"` column (so the render layer's
 * `segmentAttributeIndex` resolves and per-segment colouring works) but
 * also slot in a synthesised `tangent` (streamline / polyline / graph /
 * skeleton) and the user-declared attributes.  The on-disk format has no
 * *per-vertex* segment column, so the backend synthesises one from the
 * per-fragment `fragment_attributes/segment_id` (truncated to uint32),
 * falling back to the fragment's chunk-local index.
 */
function buildZvSpatialVertexAttributes(parameters: {
  attributeNames: string[];
  attributeDtypes: ZarrVectorsAttributeDtype[];
  attributePropertyIds?: string[];
  geometryKind: ZarrVectorsGeometryKind;
}): ZvVertexAttributeRenderInfo[] {
  const out: ZvVertexAttributeRenderInfo[] = [
    {
      name: "",
      dataType: DataType.FLOAT32,
      numComponents: 3,
      webglDataType: WebGL2RenderingContext.FLOAT,
      glslDataType: "vec3",
    },
  ];
  if (hasSynthesisedTangent(parameters.geometryKind)) {
    out.push({
      name: "tangent",
      dataType: DataType.FLOAT32,
      numComponents: 3,
      webglDataType: WebGL2RenderingContext.FLOAT,
      glslDataType: "vec3",
    });
  }
  // `name` is the shader-facing identifier (`prop_<name>()`), which is not
  // necessarily the on-disk attribute directory name — see
  // `attributePropertyIds`.
  const shaderNames =
    parameters.attributePropertyIds ?? parameters.attributeNames;
  for (let i = 0; i < parameters.attributeNames.length; ++i) {
    // float32 whatever the on-disk dtype: the decoder converts, so the render
    // layer sees one type for every attribute and `prop_<name>()` is always a
    // plain `float`.
    out.push({
      name: shaderNames[i],
      dataType: DataType.FLOAT32,
      numComponents: 1,
      webglDataType: zvWebglDataType(DataType.FLOAT32),
      glslDataType: getShaderType(DataType.FLOAT32, 1),
    });
  }
  // Synthesised per-vertex `"segment"` column (last slot — mirrors the
  // backend's `download()` packing).  Naming it `"segment"` is what makes
  // the render layer wire `segmentAttributeIndex` and colour each fragment
  // by its owning segment via `segmentColorHash`.  Two uint32 components
  // (`uvec2`) carry the FULL uint64 flywire id `[lo, hi]`, so dense
  // fragments colour identically to the flat segmentation's voxels for the
  // same id (the render layer hashes the full uint64 with the shared
  // `segmentColorHash`) and a picked fragment surfaces the global id.  The
  // backend always synthesises this column (per-fragment `segment_id`, or
  // the fragment's chunk-local index as a fallback), so it is unconditional.
  out.push({
    name: "segment",
    dataType: DataType.UINT64,
    numComponents: 1,
    webglDataType: WebGL2RenderingContext.UNSIGNED_INT,
    glslDataType: getShaderType(DataType.UINT64, 1),
  });
  return out;
}

/**
 * Frontend chunk source backing the spatially-indexed (pass-1) render
 * layer.  Paired with `ZarrVectorsSpatialGeometrySourceBackend`
 * via `RPC_ID` on the parameter class.
 *
 * One instance per resolution level.  The render layer enumerates
 * visible chunks via the inherited `SpatiallyIndexedSkeletonSource`
 * frustum-culling machinery and the matching backend's `download()`
 * fetches + decodes zarr-vectors chunks.
 */
export class ZarrVectorsSpatialGeometrySource extends WithParameters(
  WithSharedKvStoreContext(SpatiallyIndexedSkeletonSource),
  ZarrVectorsSpatialGeometrySourceParameters,
) {
  private zvAttributeTextureFormats_?: TextureFormat[];

  constructor(
    ...args: ConstructorParameters<typeof SpatiallyIndexedSkeletonSource>
  ) {
    super(...args);
    // `SpatiallyIndexedSkeletonSource`'s constructor bakes in
    // `[position, segment]` for `vertexAttributes`.  We replace it with a
    // shape that matches what the backend's `download()` actually packs
    // into `chunk.vertexAttributes`: position, then a synthesised tangent
    // (streamline / polyline / graph / skeleton), then user-declared
    // attributes, then a synthesised `"segment"` column last (mirroring
    // `geometry_backend.ts:ZarrVectorsSpatialGeometrySourceBackend`).
    this.vertexAttributes = buildZvSpatialVertexAttributes(this.parameters);
  }

  /**
   * Hand the store's attributes to the render layer as one packed run. Without
   * it a MERFISH store's columns would take a texture unit and a varying each,
   * which is what capped a layer at about ten of them.
   */
  get packedAttributeRange(): PackedAttributeRange | undefined {
    return zvPackedAttributeRange(this.parameters);
  }

  /**
   * Texture-format array indexed in lock-step with `vertexAttributes`
   * and the `vertexAttributeOffsets` produced by
   * `serializeSkeletonChunkData`.  Returning the right number of entries
   * here is what stops the runtime crash described at
   * `skeleton/frontend.ts:1593` ("`Cannot destructure property
   * 'arrayConstructor' of 'format' as it is undefined`").
   *
   * Overrides the parent's cached `[position, segment]` formats — see
   * `skeleton/frontend.ts:1716-1734`.
   */
  get attributeTextureFormats(): TextureFormat[] {
    let cached = this.zvAttributeTextureFormats_;
    if (cached === undefined) {
      cached = this.zvAttributeTextureFormats_ = this.vertexAttributes.map(
        ({ dataType, numComponents }) =>
          computeTextureFormat(new TextureFormat(), dataType, numComponents),
      );
    }
    return cached;
  }

  /**
   * Map driving the `prop_<name>()` shader bridge.  The order here
   * must match how `ZarrVectorsSpatialGeometrySourceBackend.download()`
   * populates `chunk.vertexAttributes`: tangent (streamline / polyline only)
   * first, then user-declared attributes in declaration order.
   */
  get zvVertexAttributeMap(): Map<string, VertexAttributeInfo> {
    return buildVertexAttributeMap(this.parameters);
  }

  /**
   * Preferred default shader text, looked up from the per-kind
   * capability table in `geometry_kind.ts`.  Streamlines auto-apply
   * the RGB-by-tangent shader; polylines, skeletons, and graphs fall
   * through to the segmentation layer's segment-coloured default
   * (`undefined` here).
   *
   * The integration point that consumes this is a follow-up to slice
   * 4d (segmentation-layer mount-time hook); for now the getter is
   * available for documentation tools and tests.
   */
  get defaultFragmentMain(): string | undefined {
    return KIND_CAPABILITIES[this.parameters.geometryKind].defaultFragmentMain;
  }

  /**
   * What primitive this store's geometry is drawn as, read by the render layer.
   * A `point_cloud` chunk decodes to zero edges and a `mesh` chunk to faces
   * rather than edges, so neither can be drawn as line segments.
   */
  get geometryPrimitive(): GeometryPrimitive {
    return KIND_CAPABILITIES[this.parameters.geometryKind].primitive;
  }

  // -------------------------------------------------------------------------
  // Spatial-skeleton source contract (`#src/skeleton/api.ts`)
  //
  // Implementing these makes the fork's Skeleton tab and node-inspection UI
  // available for a zarr-vectors layer: `isSpatiallyIndexedSkeletonSource`
  // duck-types this object for `readonly` plus the four methods below, and the
  // layer hides the tab when it does not match (`layer/segmentation/index.ts`).
  // The same duck-type, extended with the edit command factories, is what
  // `isEditableSpatiallyIndexedSkeletonSource` looks for -- so editing is added
  // by growing this block, not by changing any UI.
  // -------------------------------------------------------------------------

  /**
   * Editable exactly when the layer was opened with an edit service
   * (`#edit=<url>`), since that is the only way a ZVF store can be written --
   * neuroglancer's kvstore is read-only by construction. Node inspection and
   * navigation work either way; only the edit actions are gated on this
   * (`getSpatialSkeletonActionsDisabledReason`).
   */
  get readonly(): boolean {
    return this.editTarget === undefined;
  }

  /** Where edits go, or `undefined` for a plain read-only layer. */
  private get editTarget(): ZarrVectorsEditTarget | undefined {
    const { editServiceUrl, editStore } = this.parameters;
    if (!editServiceUrl || !editStore) return undefined;
    return { serviceUrl: editServiceUrl, store: editStore };
  }

  private editCommands_?: ReturnType<typeof makeZarrVectorsEditCommands>;
  private get editCommands() {
    const target = this.editTarget;
    if (target === undefined) return undefined;
    return (this.editCommands_ ??= makeZarrVectorsEditCommands(target));
  }

  // The duck type inspects these by name and requires all five before the
  // source counts as editable (`SPATIAL_SKELETON_EDIT_COMMAND_METADATA`).
  get splitSkeletonsCommand() {
    return this.editCommands?.splitSkeletonsCommand;
  }
  get mergeSkeletonsCommand() {
    return this.editCommands?.mergeSkeletonsCommand;
  }
  get addNodesCommand() {
    return this.editCommands?.addNodesCommand;
  }
  get moveNodesCommand() {
    return this.editCommands?.moveNodesCommand;
  }
  get deleteNodesCommand() {
    return this.editCommands?.deleteNodesCommand;
  }

  /**
   * Never called: `listSkeletons` exists only for the duck-type check. The
   * segments list is driven by the segment property map, and object ids come
   * from picking, so nothing enumerates the store this way -- which is just as
   * well, since a tractography store holds hundreds of thousands of objects.
   */
  async listSkeletons(): Promise<number[]> {
    return [];
  }

  /**
   * Also never called; part of the duck-type only. The spatial grid this source
   * exposes to the framework is the chunk specification built by
   * `ZarrVectorsMultiscaleGeometrySource.getSources`, not this metadata, which
   * exists for backends whose node fetches are bounded by a query box.
   */
  async getSpatialIndexMetadata(): Promise<SpatiallyIndexedSkeletonMetadata | null> {
    return null;
  }

  /**
   * Part of the duck-type only. The CATMAID backend fetches nodes per spatial
   * cell to BUILD its chunks; zarr-vectors builds chunks from the store's own
   * spatial arrays in the worker, so there is nothing to serve here.
   */
  async fetchNodes(): Promise<SpatiallyIndexedSkeletonNodeBase[]> {
    return [];
  }

  /**
   * One object's whole geometry as a rooted node tree -- the only one of these
   * four methods anything calls (`spatial_skeleton_manager.getFullSegmentNodes`,
   * which caches the result per segment).
   *
   * The work happens in the worker, where the store is; see
   * `ZarrVectorsSpatialGeometrySourceBackend.getObjectSkeletonNodes`.
   */
  async getSkeleton(
    skeletonId: number,
    options?: { signal?: AbortSignal },
  ): Promise<SpatiallyIndexedSkeletonNode[]> {
    const { rpc } = this;
    if (rpc == null) {
      throw new Error(
        "zarr-vectors: this geometry source is not connected to a worker.",
      );
    }
    const { nodes } = await rpc.promiseInvoke<{
      nodes: SpatiallyIndexedSkeletonNode[];
    }>(
      ZARR_VECTORS_GET_OBJECT_NODES_RPC_ID,
      { source: this.rpcId, objectId: String(BigInt(skeletonId)) },
      { signal: options?.signal },
    );
    return nodes;
  }
}

/**
 * Frontend chunk source backing the object-keyed (pass-2) render layer.
 * Paired with `ZarrVectorsObjectKeyedGeometrySourceBackend` (to be
 * implemented in slice 4b once the `object_index/manifests` zarr-vlen-
 * bytes reader exists).
 *
 * Today this class is a thin shell so the RPC parameter type is
 * referenced in at least one frontend module and `tsgo` keeps it in the
 * dependency graph; the backend's `download()` is not yet implemented.
 */
export class ZarrVectorsObjectKeyedGeometrySource extends WithParameters(
  WithSharedKvStoreContext(SkeletonSource),
  ZarrVectorsObjectKeyedGeometrySourceParameters,
) {
  /**
   * Marks this as the zarr-vectors pass-2 source the ROI filter repurposes as
   * its full-detail render layer: the segmentation layer gives its render layer
   * a dedicated visible set (`roiHighDetailSegments`) instead of the user's
   * selection, so a group's "high detail" toggle draws its tracts at full
   * resolution on top of the coarse pass-1 bulk.
   */
  readonly isRoiHighDetailSource = true;

  /** The store's declared geometry type, for the data-sources tab to name. */
  get geometryKind(): ZarrVectorsGeometryKind {
    return this.parameters.geometryKind;
  }

  /**
   * Vertex positions are physical coordinates (NGFF
   * `multiscales[0].axes` units), NOT voxel indices.  The render layer
   * uses this flag to skip the implicit voxel→world transform.
   */
  get skeletonVertexCoordinatesInVoxels() {
    return false;
  }

  /**
   * Map driving the `prop_<name>()` shader bridge.  Same ordering
   * convention as the spatially-indexed source — the backend's
   * `download()` packs `chunk.vertexAttributes` in this order.
   */
  get vertexAttributes(): Map<string, VertexAttributeInfo> {
    return buildVertexAttributeMap(this.parameters);
  }

  /** See {@link zvPackedAttributeRange}; same layout as the pass-1 source. */
  get packedAttributeRange(): PackedAttributeRange | undefined {
    return zvPackedAttributeRange(this.parameters);
  }

  /**
   * Preferred default shader text for streamline stores.  See the
   * matching getter on `ZarrVectorsSpatialGeometrySource` for
   * design notes.
   */
  get defaultFragmentMain(): string | undefined {
    return KIND_CAPABILITIES[this.parameters.geometryKind].defaultFragmentMain;
  }
}

// ---------------------------------------------------------------------------
// Multiscale spatially-indexed source (pass-1 wrapper)
// ---------------------------------------------------------------------------

/**
 * One pyramid-level entry for the spatially-indexed (pass-1) source.
 * Each level owns its own chunk-source parameter blob (`baseUrl`,
 * `attributeNames`, etc.); the parent multiscale source builds the
 * per-level `SpatiallyIndexedSkeletonChunkSpecification` from shared
 * grid info (`chunkShape`, `gridShapeInVoxels`) since zarr-vectors
 * keeps the chunk grid uniform across levels.
 */
export interface ZarrVectorsGeometrySpatialLevel {
  readonly parameters: ZarrVectorsSpatialGeometrySourceParameters;
}

/**
 * Multiscale wrapper that hands out per-level `SpatiallyIndexedSkeletonSource`
 * chunk sources to the segmentation layer's pass-1 spatial render path.
 * Mirrors the catmaid template at
 * `src/datasource/catmaid/frontend.ts:202-307` but without credentials
 * (we use a shared kvstore context) and reads its grid info from the
 * zarr-vectors store metadata.
 *
 * Constraint: positions are 3-D.  Neuroglancer's
 * `spatiallyIndexedSkeletonTextureAttributeSpecs` hardcodes
 * `position: float32×3` (see `skeleton/frontend.ts:1706-1709`), so 2-D
 * or higher-rank zarr-vectors stores fall back to pass-2 only.  The
 * caller (`buildGeometryMetadata`) must enforce this.
 */
export class ZarrVectorsMultiscaleGeometrySource extends MultiscaleSpatiallyIndexedSkeletonSource {
  /**
   * Opt this source into the ROI streamline filter. Its pass-1 chunks carry a
   * per-vertex segment column and retain a `roiFilterableChunk`, so the render-
   * layer backend can attribute geometry to objects and compute a passing set.
   * Other spatially-indexed skeleton sources (e.g. CATMAID) share these base
   * classes but emit no segment column, so the filter would ghost everything —
   * the segmentation layer gates the whole ROI channel on this flag rather than
   * on the shared source class.
   */
  /**
   * What primitive every level of this store draws as. All levels share one
   * geometry kind, so level 0 answers for the source.
   */
  get geometryPrimitive(): GeometryPrimitive {
    return KIND_CAPABILITIES[this.levels[0].parameters.geometryKind].primitive;
  }

  /** The store's declared geometry type, for the data-sources tab to name. */
  get geometryKind(): ZarrVectorsGeometryKind {
    return this.levels[0].parameters.geometryKind;
  }

  get supportsRoiStreamlineFilter(): boolean {
    // Every zarr-vectors geometry kind: the pass-1 chunks all carry a per-vertex
    // segment column and a retained `roiFilterableChunk`, which is what the fold
    // needs. What differs between kinds is only HOW a chunk is folded, and the
    // chunk says which: `perVertexObjects` for a point cloud (one id per point,
    // and a fragment is a spatial bin of unrelated points, so folding per
    // fragment would put a whole bin in one region), `surfaceVertices` for a
    // mesh (a face soup has no walk order, so every predicate is "any vertex").
    return true;
  }

  /**
   * Whether the tract export applies to this store. Both of its formats are
   * streamline-shaped -- TrackVis `.trk` is polylines by definition, and the
   * zarr-vectors exporter reads whole tracts -- so a point cloud or surface
   * gets the Filter tab without the Export tab.
   */
  get supportsTractExport(): boolean {
    return this.geometryPrimitive === "lines";
  }

  /**
   * The per-vertex attribute names this source actually loaded, in the order it
   * loaded them: the on-disk `vertex_attributes/<name>` directory names, which
   * are what an attribute predicate persists and what the worker keys its
   * retained columns by. Not the GLSL-safe `prop_<id>()` spellings.
   *
   * All levels of a store share one selection, so level 0 answers for it.
   */
  get vertexAttributeNames(): readonly string[] {
    return this.levels[0]?.parameters.attributeNames ?? [];
  }

  /** On-disk dtypes parallel to {@link vertexAttributeNames}. */
  get vertexAttributeDtypes(): readonly string[] {
    return this.levels[0]?.parameters.attributeDtypes ?? [];
  }

  /**
   * Opt in to camera-driven LOD picking ONLY when there are ≥2 pyramid
   * levels for the picker to choose between.  With a single level, auto-LOD
   * is meaningless: `maybeUpdateAutoSpatialSkeletonGridResolutionTarget`
   * overwrites the resolution target every frame with a camera-derived
   * spacing (e.g. 100–285 mm when zoomed out) that is far coarser than the
   * lone level's spacing (~6.4 mm).  Empirically that mismatch leaves the
   * chunks loaded ("N/N present" in the render-scale widget) but **undrawn**,
   * so a single-level store renders nothing.  Returning false here for the
   * single-level case reverts to the manual/default picker, which always
   * draws the one available level regardless of zoom.  Multi-level stores
   * keep camera-driven LOD.  See `getSpatialSkeletonGridSizes()`.
   */
  override get prefersAutoSpatialSkeletonGridLevel(): boolean {
    return this.levels.length > 1;
  }

  /**
   * Draw streamlines by orientation (|tangent| as RGB) rather than by the
   * built-in per-segment hash colour -- the standard tractography convention,
   * and the reason `DEFAULT_STREAMLINE_FRAGMENT_MAIN` exists.
   *
   * Only `streamline` supplies one (`KIND_CAPABILITIES`); every other geometry
   * kind returns undefined and keeps `emitDefault()`. That gating is what makes
   * `prop_tangent()` safe to reference: the shader is only offered for kinds
   * with `hasSynthesisedTangent`, which is what puts `tangent` in the attribute
   * map and emits the `#define`.
   *
   * `levels` is finest-first, but geometryKind is a property of the store, not
   * of a level, so level 0 speaks for all of them.
   */
  override get defaultFragmentMain(): string | undefined {
    const kind = this.levels[0]?.parameters.geometryKind;
    return kind === undefined
      ? undefined
      : KIND_CAPABILITIES[kind].defaultFragmentMain;
  }

  /** Per-level chunk-source parameter blobs in finest-first order. */
  readonly levels: ReadonlyArray<ZarrVectorsGeometrySpatialLevel>;
  /**
   * Per-level chunk shape in world units, finest-first.  Length ==
   * `levels.length`.  Each entry comes from the level's own
   * ``zarr_vectors_level.chunk_shape`` if present, otherwise from the
   * root chunk_shape.  Used verbatim for chunk addressing in
   * `getSources()`; `getSpatialSkeletonGridSizes()` reports a
   * density-corrected size instead, so an object-sparsity pyramid (which
   * keeps one chunk_shape for every level) is still pickable.
   */
  readonly perLevelChunkShape: Float32Array[];
  /**
   * Live object count per level, finest-first; `undefined` where it could not
   * be determined.  This is the detail axis of an object-sparsity pyramid --
   * coarser levels hold fewer *complete* objects at the same chunk_shape.
   * See `getSpatialSkeletonGridSizes()`.
   */
  readonly perLevelObjectCount: (number | undefined)[];
  /**
   * ``zarr_vectors_level.vertex_count`` per level, finest-first; `undefined`
   * where the writer did not stamp it.
   *
   * Distinct from `perLevelObjectCount`, and both are needed: object count is
   * the *detail* axis (how many complete streamlines a level holds), while
   * vertex count is the *cost* axis (what actually lands on the GPU). They
   * only track each other while vertices-per-object stays constant.
   */
  readonly perLevelVertexCount: (number | undefined)[];
  /**
   * Meters per coordinate unit, per axis (from the store's NGFF
   * scale + unit).  Used to report grid sizes in physical meters so the
   * resolution widget + auto-LOD picker are unit-consistent regardless of
   * the global coordinate space's voxel size.
   */
  readonly metersPerUnit: Float64Array;
  /** World-space lower bound of the data; can be negative. */
  readonly lowerBounds: Float32Array;
  /** World-space upper bound of the data. */
  readonly upperBounds: Float32Array;

  get rank(): number {
    return 3;
  }

  /**
   * The pyramid's levels, **coarsest first** — the order `lod` is derived
   * from, and the order the picker and render-scale widget present.
   *
   * `levels` is finest-first, mirroring the store's `multiscales` directory
   * order, so this reverses it. That directory order *is* the level structure:
   * each level's `gridIndex` is assigned straight from it
   * (`levelPaths.length - 1 - k` in `frontend.ts`), and the backend matches
   * the selected level back to a source by that `gridIndex`. Declaring the
   * order here keeps the two definitions in agreement by construction, rather
   * than relying on a spacing sort to rediscover it — which it cannot do for
   * an object-sparsity pyramid, where every level shares one `chunk_shape`
   * and the sort degenerates to a no-op that leaves position and `gridIndex`
   * contradicting each other.
   *
   * `size` is no longer what orders the levels, only what scales them: it is
   * reported as the mean spacing between *objects*, so the widget reads in
   * real units. Sparser levels genuinely are coarser — spacing between
   * objects grows as the cube root of the drop in their number — and for this
   * store that is the only axis that varies: vertices-per-object stays ~203
   * at every level, so the levels differ in how many complete streamlines
   * they hold, not in per-streamline fidelity.
   *
   * `getSources()` keeps using the true `chunk_shape`: this affects which
   * level is chosen and how it is described, never how its chunks are read.
   */
  override getSpatialSkeletonGridSizes(liveScale?: Float64Array): {
    x: number;
    y: number;
    z: number;
  }[] {
    // Report in physical meters (chunk_shape × meters-per-unit) so the
    // resolution widget reads in real units and the auto-LOD target
    // (also meters) compares correctly.  Prefer `liveScale` — it
    // composes the store's own declared native-unit scale with any
    // output CoordinateSpaceTransform the user has applied (e.g.
    // correcting a source's declared voxel size from mm to µm) — over
    // the frozen `metersPerUnit` captured at construction time, which
    // only reflects the store's own metadata and silently goes stale
    // relative to the camera-driven target after such an edit.
    const m = liveScale ?? this.metersPerUnit;
    const density = this.computeDensityScales();
    const finestFirst = this.perLevelChunkShape.map((cs, k) => {
      const s = density[k];
      return { x: cs[0] * m[0] * s, y: cs[1] * m[1] * s, z: cs[2] * m[2] * s };
    });
    return finestFirst.reverse();
  }

  /**
   * Estimated bytes each level puts on the GPU if fully resident, **coarsest
   * first** to match `getSpatialSkeletonGridSizes()`. `NaN` for a level whose
   * vertex count the writer did not stamp — an unknown cost must not read as
   * an affordable one.
   *
   * Counts what the backend actually packs per vertex (see `download()` in
   * `geometry_backend.ts`): positions, the synthesised tangent where the
   * geometry kind has one, each declared 1-component attribute, and the
   * synthesised uint64 segment column. Edges are implicit `i -> i+1` for a
   * streamline, so there is ~1 edge per vertex, each a `uvec2` index pair.
   *
   * This is an upper bound on a fully-resident level, not a live measurement:
   * only chunks in view are actually fetched. It is used to refuse a level
   * that could not fit even in principle -- for this store level 0 estimates
   * ~4 GB against a 1 GB budget -- rather than to predict real usage.
   */
  /**
   * Objects (streamlines) per level, **coarsest first** to match
   * `getSpatialSkeletonGridSizes()`. `undefined` for a level whose count the
   * writer did not stamp.
   */
  /**
   * How many grid cells each level's spatial index spans, **coarsest first**.
   *
   * Turns a whole-level byte cost into a per-cell one, which is the unit LOCAL
   * detail focus budgets in: the memory limit divided by the cells in view names
   * the finest level each cell can afford.
   */
  getSpatialSkeletonLevelCellCounts(): number[] {
    const finestFirst = this.perLevelChunkShape.map((chunkShape) => {
      const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
        this.lowerBounds,
        this.upperBounds,
        chunkShape,
      );
      let cells = 1;
      for (let i = 0; i < lowerChunkBound.length; ++i) {
        cells *= Math.max(0, upperChunkBound[i] - lowerChunkBound[i]);
      }
      return cells;
    });
    return finestFirst.reverse();
  }

  getSpatialSkeletonLevelObjectCounts(): (number | undefined)[] {
    return this.perLevelObjectCount.slice().reverse();
  }

  /** What one vertex of level `k` (finest-first) costs on the GPU. */
  private bytesPerVertexAtLevel(k: number): number {
    const p = this.levels[k].parameters;
    let bytesPerVertex = p.rank * 4; // positions, float32
    if (hasSynthesisedTangent(p.geometryKind)) bytesPerVertex += 3 * 4;
    bytesPerVertex += p.attributeDtypes.length * ATTR_GPU_BYTES;
    bytesPerVertex += 8; // synthesised segment column, uvec2
    bytesPerVertex += 8; // ~1 implicit edge per vertex, uvec2 index pair
    return bytesPerVertex;
  }

  getSpatialSkeletonLevelCostsBytes(): number[] {
    const finestFirst = this.levels.map((_level, k) => {
      const count = this.perLevelVertexCount[k];
      if (count === undefined) return Number.NaN;
      return count * this.bytesPerVertexAtLevel(k);
    });
    return finestFirst.reverse();
  }

  /**
   * Per-level multiplier that spreads levels the chunk-shape signal cannot
   * separate. All 1 when vertex counts are unavailable, when every level has
   * the same count, or when chunk growth already reflects the vertex drop.
   */
  private computeDensityScales(): number[] {
    // Objects first: on a tractogram the detail axis IS "how many complete
    // streamlines does this level hold". A store with no object model has no
    // such count, so `computePyramidDensityScales` falls back to vertices.
    return computePyramidDensityScales(
      this.perLevelObjectCount,
      this.perLevelVertexCount,
      this.perLevelChunkShape.map((cs) => Math.min(cs[0], cs[1], cs[2])),
    );
  }

  /**
   * Expose every pyramid level to the render layer.  The default
   * implementation on `MultiscaleSpatiallyIndexedSkeletonSource` returns
   * `scales[0][0]` per group — i.e. only the first source.  Because we
   * put all levels in a single scale group (see `getSources` below),
   * the default would drop two of our three levels on the floor.  Match
   * the catmaid datasource's `getPerspectiveSources` override which
   * returns the full first group.
   */
  override getPerspectiveSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    const sources = this.getSources(SPATIAL_SKELETON_SOURCE_OPTIONS);
    return sources.length > 0 ? sources[0] : [];
  }

  override getSliceViewPanelSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    return this.getPerspectiveSources();
  }

  constructor(
    chunkManager: Borrowed<ChunkManager>,
    private readonly sharedKvStoreContext: SharedKvStoreContext,
    options: {
      levels: ReadonlyArray<ZarrVectorsGeometrySpatialLevel>;
      perLevelChunkShape: Float32Array[];
      perLevelObjectCount?: (number | undefined)[];
      perLevelVertexCount?: (number | undefined)[];
      perLevelObjectVertexCounts?: Uint32Array[];
      objectDepths?: Uint8Array;
      metersPerUnit: Float64Array;
      lowerBounds: Float32Array;
      upperBounds: Float32Array;
    },
  ) {
    super(chunkManager);
    this.levels = options.levels;
    this.perLevelChunkShape = options.perLevelChunkShape;
    this.perLevelObjectCount =
      options.perLevelObjectCount ??
      new Array<number | undefined>(options.perLevelChunkShape.length).fill(
        undefined,
      );
    this.perLevelVertexCount =
      options.perLevelVertexCount ??
      new Array<number | undefined>(options.perLevelChunkShape.length).fill(
        undefined,
      );
    this.perLevelObjectVertexCounts = options.perLevelObjectVertexCounts;
    this.objectDepths = options.objectDepths;
    this.metersPerUnit = options.metersPerUnit;
    this.lowerBounds = options.lowerBounds;
    this.upperBounds = options.upperBounds;
  }

  /**
   * Per-object vertex counts per level, finest-first (`0` = absent), and the
   * coarsest level holding each object. Present together or not at all; absent
   * for a store lacking `object_attributes/vertex_count` or whose levels are not
   * nested. See `object_admission.ts`.
   */
  readonly perLevelObjectVertexCounts: Uint32Array[] | undefined;
  readonly objectDepths: Uint8Array | undefined;

  /**
   * The largest set of WHOLE objects `budgetBytes` affords, or `undefined` when
   * this store cannot be budgeted per object.
   *
   * Costed at each level's own geometry, so the answer accounts for the fact
   * that drawing a finer level re-costs even the coarse backbone.
   */
  /**
   * Whether this store actually carries the per-level object membership that
   * {@link computeObjectAdmission} needs.
   *
   * Distinct from the method merely EXISTING.  A store whose levels omit
   * `object_attributes/vertex_count` still has the method, but it can only ever
   * answer `undefined` — and a consumer that installs the admission closure on
   * the strength of the method being present ends up with a store that can
   * neither budget per object nor fall back to whole-level selection, freezing
   * the pyramid at whatever level was picked first.
   */
  get canBudgetPerObject(): boolean {
    return (
      this.perLevelObjectVertexCounts !== undefined &&
      this.objectDepths !== undefined
    );
  }

  computeObjectAdmission(budgetBytes: number): ObjectAdmission | undefined {
    const counts = this.perLevelObjectVertexCounts;
    const depths = this.objectDepths;
    if (counts === undefined || depths === undefined) return undefined;
    // Bytes per vertex is a property of what the renderer packs, and it is the
    // same at every level for one store, so level 0 answers for all of them.
    return admissionForBudget(
      counts,
      depths,
      this.bytesPerVertexAtLevel(0),
      budgetBytes,
    );
  }

  getSources(
    _options: SliceViewSourceOptions,
  ): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[][] {
    const sources: SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] =
      [];
    const { perLevelChunkShape, lowerBounds, upperBounds } = this;

    for (let k = 0; k < this.levels.length; ++k) {
      const level = this.levels[k];
      const chunkShape = perLevelChunkShape[k];
      // zarr-vectors chunks are indexed around world origin (chunk
      // `(i,j,k)` covers world `[i*chunkShape, (i+1)*chunkShape]`),
      // and chunk indices can be negative.  Both chunkLayout and
      // chunkToMultiscaleTransform are identity — vertex world coords
      // come straight off disk in NGFF physical units.  Per-level
      // `chunkShape` may differ when the writer used
      // `chunk_scale_factors`.
      const chunkLayoutTransform = mat4.create();
      const chunkLayout = new ChunkLayout(
        vec3.fromValues(chunkShape[0], chunkShape[1], chunkShape[2]),
        chunkLayoutTransform,
        3,
      );

      // FLOAT chunk size, deliberately. A zarr-vectors `chunk_shape` is a
      // physical extent, not a voxel count, and is routinely fractional -- a
      // 0.5 mm MERFISH grid, say. Through a Uint32Array it truncates to 0, and
      // the chunk-index bounds below become +/-Infinity; the frustum walk in
      // `forEachVolumetricChunkWithinFrustrum` then binary-splits a box it can
      // never reduce to a single chunk and blows the stack.
      const chunkDataSize = Float32Array.from(chunkShape.subarray(0, 3));
      // lowerVoxelBound / upperVoxelBound encode the data extent in
      // world (= chunk-layout) units.  The chunk-index bounds are computed
      // here rather than taken from `makeSliceViewChunkSpecification`, whose
      // `(upper - 1) / size + 1` upper bound assumes integer voxel units and
      // loses a chunk when the extent is measured in millimetres.  Negative
      // chunk indices are fine either way: zarr-vectors indexes chunks around
      // the world origin, and floor/ceil handle the sign.
      const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
        lowerBounds,
        upperBounds,
        chunkDataSize,
      );
      const spec: SpatiallyIndexedSkeletonChunkSpecification = {
        ...makeSliceViewChunkSpecification({
          rank: 3,
          chunkDataSize,
          lowerVoxelBound: lowerBounds,
          upperVoxelBound: upperBounds,
        }),
        lowerChunkBound,
        upperChunkBound,
        chunkLayout,
      };

      const chunkSource = this.chunkManager.getChunkSource(
        ZarrVectorsSpatialGeometrySource,
        {
          sharedKvStoreContext: this.sharedKvStoreContext,
          spec,
          parameters: level.parameters,
        },
      );

      // Identity chunk-to-multiscale transform — chunks already live in
      // the same coordinate frame as the rest of the layer.  See the
      // chunk-grid comment above.
      const chunkToMultiscaleTransform = mat4.create();

      sources.push({ chunkSource, chunkToMultiscaleTransform });
    }

    // Single scale group — all levels are alternative representations
    // of the same data at decreasing fidelity.  The render layer picks
    // levels per the layer's pyramid-mode setting.
    return [sources];
  }
}
