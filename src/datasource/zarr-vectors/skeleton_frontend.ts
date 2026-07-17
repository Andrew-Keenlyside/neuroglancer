/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Frontend-side chunk source classes for zarr-vectors skeleton /
 * polyline / streamline rendering.  Each one is paired with a backend
 * class in `./skeleton_backend.ts` via a matching ``RPC_ID`` on the
 * parameter type.
 *
 * - `ZarrVectorsSpatiallyIndexedSkeletonSource` — the **pass-1** chunk
 *   source.  Subclass of neuroglancer's
 *   `SpatiallyIndexedSkeletonSource`; the backend pairs with
 *   `ZarrVectorsSpatiallyIndexedSkeletonSourceBackend` and downloads one
 *   chunk per `(chunkGridPosition, lod)` pair.
 *
 * - `ZarrVectorsObjectKeyedSkeletonSource` — the **pass-2** chunk source
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
  ZarrVectorsObjectKeyedSkeletonSourceParameters,
  ZarrVectorsSpatiallyIndexedSkeletonSourceParameters,
  type ZarrVectorsAttributeDtype,
  type ZarrVectorsSkeletonGeometryKind,
} from "#src/datasource/zarr-vectors/base.js";
import {
  KIND_CAPABILITIES,
  hasSynthesisedTangent,
} from "#src/datasource/zarr-vectors/geometry_kind.js";
import { buildVertexAttributeMap } from "#src/datasource/zarr-vectors/skeleton_shader_bridge.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import {
  MultiscaleSpatiallyIndexedSkeletonSource,
  SkeletonSource,
  SPATIAL_SKELETON_SOURCE_OPTIONS,
  SpatiallyIndexedSkeletonSource,
  type SpatiallyIndexedSkeletonChunkSpecification,
} from "#src/skeleton/frontend.js";
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
export {
  DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  buildVertexAttributeMap,
} from "#src/datasource/zarr-vectors/skeleton_shader_bridge.js";

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

/** On-GPU width of each attribute dtype, for the level cost estimate. */
const ATTR_DTYPE_BYTES: Record<ZarrVectorsAttributeDtype, number> = {
  float32: 4,
  uint8: 1,
  uint16: 2,
  uint32: 4,
  int8: 1,
  int16: 2,
  int32: 4,
};

const ATTR_DTYPE_TO_DATA_TYPE: Record<ZarrVectorsAttributeDtype, DataType> = {
  float32: DataType.FLOAT32,
  uint8: DataType.UINT8,
  uint16: DataType.UINT16,
  uint32: DataType.UINT32,
  int8: DataType.INT8,
  int16: DataType.INT16,
  int32: DataType.INT32,
};

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
 * backend (`skeleton_backend.ts:ZarrVectorsSpatiallyIndexedSkeletonSourceBackend
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
  geometryKind: ZarrVectorsSkeletonGeometryKind;
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
  for (let i = 0; i < parameters.attributeNames.length; ++i) {
    const dt = ATTR_DTYPE_TO_DATA_TYPE[parameters.attributeDtypes[i]];
    out.push({
      name: parameters.attributeNames[i],
      dataType: dt,
      numComponents: 1,
      webglDataType: zvWebglDataType(dt),
      glslDataType: getShaderType(dt, 1),
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
 * layer.  Paired with `ZarrVectorsSpatiallyIndexedSkeletonSourceBackend`
 * via `RPC_ID` on the parameter class.
 *
 * One instance per resolution level.  The render layer enumerates
 * visible chunks via the inherited `SpatiallyIndexedSkeletonSource`
 * frustum-culling machinery and the matching backend's `download()`
 * fetches + decodes zarr-vectors chunks.
 */
export class ZarrVectorsSpatiallyIndexedSkeletonSource extends WithParameters(
  WithSharedKvStoreContext(SpatiallyIndexedSkeletonSource),
  ZarrVectorsSpatiallyIndexedSkeletonSourceParameters,
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
    // `skeleton_backend.ts:ZarrVectorsSpatiallyIndexedSkeletonSourceBackend`).
    this.vertexAttributes = buildZvSpatialVertexAttributes(this.parameters);
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
   * must match how `ZarrVectorsSpatiallyIndexedSkeletonSourceBackend.download()`
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
}

/**
 * Frontend chunk source backing the object-keyed (pass-2) render layer.
 * Paired with `ZarrVectorsObjectKeyedSkeletonSourceBackend` (to be
 * implemented in slice 4b once the `object_index/manifests` zarr-vlen-
 * bytes reader exists).
 *
 * Today this class is a thin shell so the RPC parameter type is
 * referenced in at least one frontend module and `tsgo` keeps it in the
 * dependency graph; the backend's `download()` is not yet implemented.
 */
export class ZarrVectorsObjectKeyedSkeletonSource extends WithParameters(
  WithSharedKvStoreContext(SkeletonSource),
  ZarrVectorsObjectKeyedSkeletonSourceParameters,
) {
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

  /**
   * Preferred default shader text for streamline stores.  See the
   * matching getter on `ZarrVectorsSpatiallyIndexedSkeletonSource` for
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
export interface ZarrVectorsSkeletonSpatialLevel {
  readonly parameters: ZarrVectorsSpatiallyIndexedSkeletonSourceParameters;
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
 * caller (`buildSkeletonMetadata`) must enforce this.
 */
export class ZarrVectorsMultiscaleSpatiallyIndexedSkeletonSource extends MultiscaleSpatiallyIndexedSkeletonSource {
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
  readonly levels: ReadonlyArray<ZarrVectorsSkeletonSpatialLevel>;
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
   * `skeleton_backend.ts`): positions, the synthesised tangent where the
   * geometry kind has one, each declared 1-component attribute, and the
   * synthesised uint64 segment column. Edges are implicit `i -> i+1` for a
   * streamline, so there is ~1 edge per vertex, each a `uvec2` index pair.
   *
   * This is an upper bound on a fully-resident level, not a live measurement:
   * only chunks in view are actually fetched. It is used to refuse a level
   * that could not fit even in principle -- for this store level 0 estimates
   * ~4 GB against a 1 GB budget -- rather than to predict real usage.
   */
  getSpatialSkeletonLevelCostsBytes(): number[] {
    const finestFirst = this.levels.map((level, k) => {
      const count = this.perLevelVertexCount[k];
      if (count === undefined) return Number.NaN;
      const p = level.parameters;
      let bytesPerVertex = p.rank * 4; // positions, float32
      if (hasSynthesisedTangent(p.geometryKind)) bytesPerVertex += 3 * 4;
      for (const dtype of p.attributeDtypes) {
        bytesPerVertex += ATTR_DTYPE_BYTES[dtype];
      }
      bytesPerVertex += 8; // synthesised segment column, uvec2
      bytesPerVertex += 8; // ~1 implicit edge per vertex, uvec2 index pair
      return count * bytesPerVertex;
    });
    return finestFirst.reverse();
  }

  /**
   * Per-level multiplier that spreads levels the chunk-shape signal cannot
   * separate. All 1 when vertex counts are unavailable, when every level has
   * the same count, or when chunk growth already reflects the vertex drop.
   */
  private computeDensityScales(): number[] {
    const counts = this.perLevelObjectCount;
    const n = this.perLevelChunkShape.length;
    const ones = new Array<number>(n).fill(1);
    if (counts.length !== n) return ones;
    // Anchor on the densest level; `levels` is finest-first, but derive it
    // rather than assume, so an unordered store still behaves.
    let finestCount: number | undefined;
    for (const c of counts) {
      if (c === undefined) return ones; // partial metadata: don't half-apply
      if (finestCount === undefined || c > finestCount) finestCount = c;
    }
    if (finestCount === undefined || finestCount <= 0) return ones;
    const spacing = this.perLevelChunkShape.map((cs) =>
      Math.min(cs[0], cs[1], cs[2]),
    );
    const finestSpacing = Math.min(...spacing);
    if (!(finestSpacing > 0)) return ones;
    return counts.map((c, k) => {
      const densityFactor = Math.cbrt(finestCount! / c!);
      const chunkGrowth = spacing[k] / finestSpacing;
      // Only make up the shortfall the chunk shape does not already express.
      return Math.max(1, densityFactor / chunkGrowth);
    });
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
      levels: ReadonlyArray<ZarrVectorsSkeletonSpatialLevel>;
      perLevelChunkShape: Float32Array[];
      perLevelObjectCount?: (number | undefined)[];
      perLevelVertexCount?: (number | undefined)[];
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
    this.metersPerUnit = options.metersPerUnit;
    this.lowerBounds = options.lowerBounds;
    this.upperBounds = options.upperBounds;
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

      const chunkDataSize = new Uint32Array([
        chunkShape[0],
        chunkShape[1],
        chunkShape[2],
      ]);
      // lowerVoxelBound / upperVoxelBound encode the data extent in
      // world (= chunk-layout) units; `makeSliceViewChunkSpecification`
      // floors / ceils these to chunk-index bounds, which handle
      // negative chunk indices fine.
      const spec: SpatiallyIndexedSkeletonChunkSpecification = {
        ...makeSliceViewChunkSpecification({
          rank: 3,
          chunkDataSize,
          lowerVoxelBound: lowerBounds,
          upperVoxelBound: upperBounds,
        }),
        chunkLayout,
      };

      const chunkSource = this.chunkManager.getChunkSource(
        ZarrVectorsSpatiallyIndexedSkeletonSource,
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
