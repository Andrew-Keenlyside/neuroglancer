/**
 * @license
 * Copyright 2016 Google Inc.
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

import { ChunkState, LayerChunkProgressInfo } from "#src/chunk_manager/base.js";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import {
  Chunk,
  ChunkRenderLayerFrontend,
  ChunkSource,
} from "#src/chunk_manager/frontend.js";
import type {
  RoiBackgroundUniforms,
  RoiGroupConfig,
  RoiLabelField,
  RoiObjectAttrColumn,
} from "#src/datasource/zarr-vectors/roi.js";
import type {
  PackedAttributeInterp,
  PackedAttributeRange,
} from "#src/skeleton/packed_attributes.js";
import {
  PACKED_ATTRIBUTE_STRIDE_UNIFORM,
  packedAttributeAccessorCode,
  packedAttributePropExpr,
  packedAttributeVaryings,
} from "#src/skeleton/packed_attributes.js";
import { hashCombine } from "#src/gpu_hash/hash_function.js";
import type { HashMapUint64, HashSetUint64 } from "#src/gpu_hash/hash_table.js";
import { GPUHashTable, HashSetShaderManager } from "#src/gpu_hash/shader.js";
import type {
  LayerView,
  MouseSelectionState,
  PickState,
  VisibleLayerInfo,
} from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type {
  PerspectiveViewReadyRenderContext,
  PerspectiveViewRenderContext,
} from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import type {
  ChunkTransformParameters,
  RenderLayerTransform,
} from "#src/render_coordinate_transform.js";
import { getChunkTransformParameters } from "#src/render_coordinate_transform.js";
import type { RenderScaleHistogram } from "#src/render_scale_statistics.js";
import type {
  RenderLayer,
  ThreeDimensionalRenderLayerAttachmentState,
} from "#src/renderlayer.js";
import { update3dRenderLayerAttachment } from "#src/renderlayer.js";
import {
  SegmentColorShaderManager,
  SegmentStatedColorShaderManager,
} from "#src/segment_color.js";
import {
  forEachVisibleSegment,
  getVisibleSegments,
  getObjectKey,
} from "#src/segmentation_display_state/base.js";
import type { SegmentationDisplayState3D } from "#src/segmentation_display_state/frontend.js";
import {
  forEachVisibleSegmentToDraw,
  registerRedrawWhenSegmentationDisplayState3DChanged,
  SegmentationLayerSharedObject,
} from "#src/segmentation_display_state/frontend.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type {
  SpatiallyIndexedSkeletonNode,
  SpatialSkeletonSourceState,
} from "#src/skeleton/api.js";
import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import type { VertexAttrStats } from "#src/skeleton/spatial_base.js";
import { SKELETON_LAYER_RPC_ID } from "#src/skeleton/base.js";
import {
  forEachSpatialSkeletonVolumeCell,
  SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_ROI_EXPORT_IDS_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_VERTEX_ATTR_STATS_RPC_ID,
} from "#src/skeleton/spatial_base.js";
import {
  buildSpatiallyIndexedSkeletonOverlayGeometry,
  type SpatiallyIndexedSkeletonOverlayGeometry,
} from "#src/skeleton/segment_overlay.js";
import {
  DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS,
  mergeSpatiallyIndexedSkeletonOverlaySegmentIds,
  retainSpatiallyIndexedSkeletonOverlaySegment,
} from "#src/skeleton/segment_overlay.js";
import {
  getSpatiallyIndexedSkeletonGridIndex,
  getSpatiallyIndexedSkeletonPartitionsObjects,
  getSpatiallyIndexedSkeletonSourceView,
  selectSpatiallyIndexedSkeletonEntriesForView,
  selectSpatiallyIndexedSkeletonEntriesForViewWithFallback,
  type SpatiallyIndexedSkeletonView,
} from "#src/skeleton/source_selection.js";
import {
  SpatialSkeletonDetailFocus,
  targetSpacingForCellBudget,
} from "#src/skeleton/spatial_chunk_sizing.js";
import {
  forEachVisibleVolumetricChunk,
  type SliceViewChunkSpecification,
  type SliceViewSourceOptions,
  type TransformedSource,
} from "#src/sliceview/base.js";
import type { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import type { SliceViewSingleResolutionSource } from "#src/sliceview/frontend.js";
import {
  getVolumetricTransformedSources,
  serializeAllTransformedSources,
  SliceViewChunk,
  SliceViewChunkSource,
  MultiscaleSliceViewChunkSource,
} from "#src/sliceview/frontend.js";
import type { SliceViewPanel } from "#src/sliceview/panel.js";
import type {
  SliceViewPanelRenderContext,
  SliceViewPanelReadyRenderContext,
} from "#src/sliceview/renderlayer.js";
import { SliceViewPanelRenderLayer } from "#src/sliceview/renderlayer.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import {
  makeCachedLazyDerivedWatchableValue,
  TrackableValue,
  WatchableValue,
  registerNested,
} from "#src/trackable_value.js";
import type { Uint64Map } from "#src/uint64_map.js";
import { Uint64Set } from "#src/uint64_set.js";
import { gatherUpdate } from "#src/util/array.js";
import { hsvToRgb } from "#src/util/colorspace.js";
import { DATA_TYPE_SIGNED, DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";
import type { ValueOrError } from "#src/util/error.js";
import { makeValueOrError, valueOrThrow } from "#src/util/error.js";
import {
  kOneVec4,
  mat3,
  mat3FromMat4,
  mat4,
  scaleMat3Output,
  vec3,
  vec4,
} from "#src/util/geom.js";
import { verifyFinitePositiveFloat } from "#src/util/json.js";
import * as matrix from "#src/util/matrix.js";
import { getObjectId } from "#src/util/object_id.js";
import { NullarySignal } from "#src/util/signal.js";
import type { Trackable } from "#src/util/trackable.js";
import { CompoundTrackable } from "#src/util/trackable.js";
import { TrackableEnum } from "#src/util/trackable_enum.js";
import {
  drawBoxEdges,
  glsl_getBoxEdgeVertexPosition,
} from "#src/webgl/bounding_box.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import {
  defineCircleShader,
  drawCircles,
  initializeCircleShader,
} from "#src/webgl/circles.js";
import { glsl_COLORMAPS } from "#src/webgl/colormaps.js";
import type { GL } from "#src/webgl/context.js";
import type { WatchableShaderError } from "#src/webgl/dynamic_shader.js";
import {
  makeTrackableFragmentMain,
  parameterizedEmitterDependentShaderGetter,
  shaderCodeWithLineDirective,
} from "#src/webgl/dynamic_shader.js";
import {
  defineLineShader,
  drawLines,
  initializeLineShader,
} from "#src/webgl/lines.js";
import type {
  ShaderModule,
  ShaderProgram,
  ShaderSamplerType,
} from "#src/webgl/shader.js";
import { ShaderBuilder } from "#src/webgl/shader.js";
import {
  dataTypeShaderDefinition,
  getShaderType,
  getShaderVectorType,
} from "#src/webgl/shader_lib.js";
import type { ShaderControlsBuilderState } from "#src/webgl/shader_ui_controls.js";
import {
  addControlsToBuilder,
  getFallbackBuilderState,
  parseShaderUiControls,
  setControlsInShader,
  ShaderControlState,
} from "#src/webgl/shader_ui_controls.js";
import {
  computeTextureFormat,
  getSamplerPrefixForDataType,
  OneDimensionalTextureAccessHelper,
  setOneDimensionalTextureData,
  TextureFormat,
  updateOneDimensionalTextureElement,
} from "#src/webgl/texture_access.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";
import type { RPC } from "#src/worker_rpc.js";

const DEBUG_SPATIAL_SKELETON_OVERLAY = false;
const DEBUG_EXCLUDED_SEGMENTS = false;
const DEBUG_SPATIAL_SKELETON_CHUNKS = false;
// Used for debugging chunks via a different color for each chunk
const tempChunkKeyToColorMap = new Map<string, Float32Array>();

const tempMat4 = mat4.create();
// Scratch for the per-object ROI group colour/opacity handed to setColor() in
// the legacy pass-2 draw loop (unpacked from roiSegmentColors).
const tempRoiColor = vec4.create();
const OVERLAY_SELECTED_FLOAT_ZERO = new Float32Array([0]);
const OVERLAY_SELECTED_FLOAT_ONE = new Float32Array([1]);
/**
 * The generic built-in skeleton shader: segment-coloured (`emitDefault()`). The
 * initial `SkeletonRenderingOptions.shader` default, and the shader the layer
 * reverts to when its skeleton subsources nominate no agreed default (see
 * `applySkeletonDefaultShader`).
 */
export const DEFAULT_FRAGMENT_MAIN = `void main() {
  emitDefault();
}
`;

/**
 * The attribute name the renderer synthesises as `prop_position()` (the vertex
 * world position). A declared attribute of the same name is not exposed, so the
 * macro is never defined twice with different bodies.
 */
const SYNTHESISED_POSITION_ATTRIBUTE_NAME = "position";

/** GLSL's three interchangeable swizzle component sets. */
const GLSL_SWIZZLE_SETS = ["xyzw", "rgba", "stpq"];

/**
 * Whether exposing `name` as a BARE `#define` would corrupt shader source that
 * has nothing to do with the attribute.
 *
 * A `#define` is preprocessor-level and applies to every occurrence of the
 * token, member accesses included -- `#define z vCustom3` rewrites `d.z` into
 * `d.vCustom3`. So any name that is also a legal swizzle (`z`, `a`, `xy`,
 * `rgb`, ...) breaks arbitrary unrelated shader code, and the failure is
 * near-invisible: the shader does not compile, the layer falls back to the
 * built-in segment-coloured one, and the tracts simply come out in per-object
 * hash colours.
 *
 * Only the bare alias is withheld. `prop_<name>()` is namespaced, cannot
 * collide, and is the form the shader property list documents.
 */
export function isUnsafeBareAttributeAlias(name: string): boolean {
  if (name.length === 0 || name.length > 4) return false;
  return GLSL_SWIZZLE_SETS.some((set) =>
    [...name].every((component) => set.includes(component)),
  );
}

const SELECTED_NODE_OUTLINE_COLOR_RGB = "1.0, 0.95, 0.35";
const SELECTED_NODE_OUTLINE_MIN_WIDTH_2D = "1.75";
const SELECTED_NODE_OUTLINE_MAX_WIDTH_2D = "3.0";
const SELECTED_NODE_OUTLINE_MIN_WIDTH_3D = "1.5";
const SELECTED_NODE_OUTLINE_MAX_WIDTH_3D = "2.5";

interface VertexAttributeRenderInfo extends VertexAttributeInfo {
  name: string;
  webglDataType: number;
  glslDataType: string;
}

const tempNormalMat3 = mat3.create();

const vertexAttributeSamplerSymbols: symbol[] = [];

/** Texture unit for the one texture a {@link PackedAttributeRange} occupies. */
const packedAttributeSamplerSymbol = Symbol(
  "SkeletonShader.packedAttributeTextureUnit",
);

const vertexPositionTextureFormat = computeTextureFormat(
  new TextureFormat(),
  DataType.FLOAT32,
  3,
);

interface VisibleChunk {
  chunk: SpatiallyIndexedSkeletonChunk;
  chunkLayout: ChunkLayout;
}

interface SkeletonShaderParameters {
  dynamicSegmentAppearance: boolean;
  /**
   * Compile the ROI-filter alpha tier (zarr-vectors tract layers). Gated so
   * every other skeleton layer's shader is byte-identical to before.
   */
  hasRoiFilter: boolean;
  /**
   * Compile the pass-1 hide tier: suppress tracts that the object-keyed pass-2
   * layer draws at full resolution, so each is drawn exactly once.
   *
   * Set only on PASS 1. Pass 2 runs this same shader code over the same display
   * state, so compiling it there would make that layer hide precisely the tracts
   * it exists to draw.
   */
  hasRoiHighDetailHide: boolean;
  /** Compile the ROI colour-by-group tier (zarr-vectors tract layers). */
  hasRoiSegmentColors: boolean;
  /**
   * Compile the background per-object value tier: the whole-tractogram length
   * filter (discard) and flat colour-by-attribute. Pass 1 only (the background
   * tracts it governs are never drawn by the pass-2 high-detail layer).
   */
  hasRoiObjectValues: boolean;
  hasSegmentStatedColors: boolean;
  hasSegmentDefaultColor: boolean;
  hoverHighlight: boolean;
  spatialChunkCulling: boolean;
}

export type {
  PackedAttributeInterp,
  PackedAttributeRange,
} from "#src/skeleton/packed_attributes.js";

interface SkeletonShaderContext {
  vertexAttributes: VertexAttributeRenderInfo[];
  gl: GL;
  fallbackShaderParameters: WatchableValue<ShaderControlsBuilderState>;
  displayState: SkeletonLayerDisplayState;
  skeletonShaderParameters: WatchableValueInterface<SkeletonShaderParameters>;
  segmentColorAttributeIndex?: number;
  /** See {@link PackedAttributeRange}; absent means one texture per attribute. */
  packedAttributeRange?: PackedAttributeRange;
}

interface SkeletonGPUGeometry {
  vertexAttributeTextures: (WebGLTexture | null)[];
  /** One texture for the whole {@link PackedAttributeRange}, when there is one. */
  packedAttributeTexture?: WebGLTexture | null;
  indexBuffer: GLBuffer;
  numIndices: number;
  numVertices: number;
  pickNodeIds?: Int32Array;
  pickNodePositions?: Float32Array;
  pickSegmentIds?: Uint32Array;
  pickEdgeSegmentIds?: Uint32Array;
}

interface PackedSkeletonGeometry {
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  nodeIds?: Int32Array;
  nodeSourceStates?: Array<SpatialSkeletonSourceState | undefined>;
}

type SpatiallyIndexedSkeletonPickData =
  | {
      kind: "node";
      nodeIds: Int32Array;
      nodePositions: Float32Array;
      segmentIds: Uint32Array;
    }
  | {
      kind: "edge";
      segmentIds: Uint32Array;
    }
  | {
      kind: "segment-node";
      chunk: SpatiallyIndexedSkeletonChunk;
    }
  | {
      kind: "segment-edge";
      chunk: SpatiallyIndexedSkeletonChunk;
    }
  | {
      /** Surface geometry: the picked primitive is one triangle. */
      kind: "segment-face";
      chunk: SpatiallyIndexedSkeletonChunk;
    };

class RenderHelper extends RefCounted {
  private textureAccessHelper = new OneDimensionalTextureAccessHelper(
    "vertexData",
  );
  private vertexIdHelper;
  private segmentAttributeIndex: number | undefined;
  private segmentColorAttributeIndex: number | undefined;
  private selectedNodeAttributeIndex: number | undefined;
  private visibleSegmentsShaderManager = new HashSetShaderManager(
    "visibleSegments",
  );
  private excludedSegmentsShaderManager = new HashSetShaderManager(
    "excludedSegments",
  );
  private roiPassingSegmentsShaderManager = new HashSetShaderManager(
    "roiPassingSegments",
  );
  private roiHighDetailSegmentsShaderManager = new HashSetShaderManager(
    "roiHighDetailSegments",
  );
  private segmentColorShaderManager = new SegmentColorShaderManager(
    "segmentColorHash",
  );
  private segmentStatedColorShaderManager = new SegmentStatedColorShaderManager(
    "segmentStatedColor",
  );
  // ROI colour-by-group tier: id -> packed group colour, overriding the
  // streamline's directional RGB when colour-by-group is on (tract layers only).
  private roiSegmentColorShaderManager = new SegmentStatedColorShaderManager(
    "roiSegmentColor",
  );
  // Background per-object value tier: id -> normalised attribute value (16-bit,
  // packed into the low two colour bytes). Drives the whole-tractogram length
  // filter (discard) and flat colour-by-attribute, for tracts NOT claimed by a
  // group (or the whole tractogram when the ROI filter is off).
  private roiObjectValueShaderManager = new SegmentStatedColorShaderManager(
    "roiObjectValue",
  );
  private readonly clearedTextureUnits = new Set<number>();
  private emptySegmentSet = new Uint64Set();
  private gpuVisibleSegmentsHashTable: GPUHashTable<HashSetUint64>;
  private gpuTemporaryVisibleSegmentsHashTable: GPUHashTable<HashSetUint64>;
  private gpuEmptySegmentsHashTable: GPUHashTable<HashSetUint64>;
  private gpuRoiPassingSegmentsHashTable:
    | GPUHashTable<HashSetUint64>
    | undefined;
  private gpuRoiHighDetailSegmentsHashTable:
    | GPUHashTable<HashSetUint64>
    | undefined;
  private gpuRoiSegmentColorHashTable: GPUHashTable<HashMapUint64> | undefined;
  private gpuRoiObjectValueHashTable: GPUHashTable<HashMapUint64> | undefined;
  // Lazily acquired and re-checked each draw; see getSegmentStatedColorHashTable.
  private gpuSegmentStatedColorHashTable:
    | GPUHashTable<HashMapUint64>
    | undefined;
  get vertexAttributes(): VertexAttributeRenderInfo[] {
    return this.base.vertexAttributes;
  }

  private defineCommonShader(
    builder: ShaderBuilder,
    shaderBuilderState: ShaderControlsBuilderState,
    skeletonParams: SkeletonShaderParameters,
  ): void {
    if (shaderBuilderState.parseResult.errors.length !== 0) {
      throw new Error("Invalid UI control specification");
    }
    defineVertexId(builder);
    builder.addUniform("highp vec4", "uColor");
    builder.addUniform("highp mat4", "uProjection");
    builder.addUniform("highp uint", "uPickID");
    builder.addVarying("highp uint", "vPickID", "flat");
    builder.addUniform("highp uint", "uPickInstanceStride");
    this.defineAttributeAccess(builder);
    if (skeletonParams.dynamicSegmentAppearance) {
      this.defineDynamicSegmentAppearance(builder, skeletonParams);
    } else if (skeletonParams.hasRoiSegmentColors) {
      // Legacy path (pass-2 object-keyed high-detail streamlines) ROI
      // colour-by-group: the per-object group colour arrives in `uColor`
      // (set on the CPU from roiSegmentColors), and this flag selects it over
      // the shader-computed colour (e.g. colour-by-direction). Per-group
      // opacity always rides `uColor.a` and needs no flag.
      builder.addUniform("highp float", "uRoiColorByGroup");
    }
    if (skeletonParams.spatialChunkCulling) {
      builder.addUniform("highp vec3", "uChunkOrigin");
      builder.addUniform("highp vec3", "uChunkBound");
      builder.addVarying("highp vec3", "vCullPos");
      builder.addFragmentCode(`
void spatialChunkCull() {
  if (any(lessThan(vCullPos, uChunkOrigin)) ||
      any(greaterThanEqual(vCullPos, uChunkBound))) discard;
}
`);
    }
  }

  // TODO (SKM): segmentAttribute is UINT32 but segments can be UINT64.
  // Change segmentAttribute.dataType to DataType.UINT64, update vSegmentValue
  // from `highp uint` (flat) to `highp uvec2` (flat), update
  // getSegmentAppearanceId to take uvec2 directly, and getSegmentAppearance
  // signature accordingly. Also pull segmentAttribute and selectedNodeAttribute
  // out of vertexAttributes entirely (they are internal, not user-defined).
  private finalizeShaderBuilder(
    builder: ShaderBuilder,
    shaderBuilderState: ShaderControlsBuilderState,
    skeletonParams: SkeletonShaderParameters,
    vertexMain: string,
    options?: {
      /**
       * Draw fragments outside the host chunk's box instead of discarding them.
       *
       * The cull exists because line geometry is duplicated across a chunk
       * boundary -- each side draws its own bridge stub to a ghost vertex, and
       * without the cull the overlap draws twice. A boundary FACE is the
       * opposite case: the links family files it under exactly one chunk, so it
       * is drawn once, and clipping it at the box leaves a visible gap along
       * every chunk seam rather than preventing anything.
       */
      skipChunkCull?: boolean;
      /**
       * How this shader addresses packed attributes: what vertex a fragment
       * belongs to, and how to blend between the vertices of its primitive.
       * Required when the source declares a {@link PackedAttributeRange};
       * ignored otherwise.
       */
      packedAttributeInterp?: PackedAttributeInterp;
    },
  ): void {
    builder.addFragmentCode(glsl_COLORMAPS);
    const { vertexAttributes } = this;
    const numAttributes = vertexAttributes.length;
    if (
      skeletonParams.dynamicSegmentAppearance &&
      this.segmentAttributeIndex !== undefined
    ) {
      const segInfo = vertexAttributes[this.segmentAttributeIndex];
      builder.addFragmentCode(dataTypeShaderDefinition[segInfo.dataType]);
      builder.addFragmentCode(
        `#define ${segInfo.name} ${segInfo.glslDataType}(vSegmentValue)\n`,
      );
      builder.addFragmentCode(
        `#define prop_${segInfo.name}() ${segInfo.glslDataType}(vSegmentValue)\n`,
      );
    }
    const packedRange = this.base.packedAttributeRange;
    if (packedRange !== undefined && packedRange.count > 0) {
      const interp = options?.packedAttributeInterp;
      if (interp === undefined) {
        throw new Error(
          "packedAttributeInterp is required when the source packs attributes",
        );
      }
      vertexMain += this.definePackedAttributeVaryings(builder, interp);
    }
    for (let i = 1; i < numAttributes; ++i) {
      if (
        i === this.segmentAttributeIndex ||
        i === this.selectedNodeAttributeIndex
      ) {
        continue;
      }
      const info = vertexAttributes[i];
      if (packedRange !== undefined && this.isPackedAttribute(i)) {
        // No varying and no vertex-stage read: the fragment fetches this
        // attribute's value from the shared texture only where the user shader
        // actually names it.
        this.definePackedAttributeMacros(
          builder,
          info.name,
          i - packedRange.start,
          options!.packedAttributeInterp!,
        );
        continue;
      }
      // A varying must be a built-in scalar/vector type, and an integer one
      // must be `flat`. `info.glslDataType` is neuroglancer's own wrapper
      // (`int32_t` is a struct), which is legal as a function return but not as
      // a varying: declaring it produced `out int32_t vCustomN;` and a vertex
      // shader that would not compile, taking every shader on the layer with
      // it. So carry the raw value across the stage boundary and rebuild the
      // wrapper in the fragment stage, exactly as `vSegmentValue` does.
      const isFloat = info.dataType === DataType.FLOAT32;
      const rawType = isFloat
        ? info.glslDataType
        : getShaderVectorType(
            DATA_TYPE_SIGNED[info.dataType] ? "int" : "uint",
            info.numComponents,
          );
      const propExpr = isFloat
        ? `vCustom${i}`
        : `${info.glslDataType}(vCustom${i})`;
      builder.addVarying(
        `highp ${rawType}`,
        `vCustom${i}`,
        isFloat ? "" : "flat",
      );
      vertexMain += isFloat
        ? `vCustom${i} = readAttribute${i}(vertexIndex);\n`
        : `vCustom${i} = toRaw(readAttribute${i}(vertexIndex));\n`;
      if (!isFloat) {
        // `int32_t(...)` and friends only exist where the type is defined.
        builder.addFragmentCode(dataTypeShaderDefinition[info.dataType]);
      }
      // `position` is synthesised below as the vertex world position. A store
      // that ALSO declares an attribute of that name would redefine the macro
      // with a different body -- a preprocessing error that kills every shader
      // on the layer. The synthesised one wins, matching how the zarr-vectors
      // reader resolves the same clash on `tangent`.
      if (info.name === SYNTHESISED_POSITION_ATTRIBUTE_NAME) continue;
      // The bare alias is a legacy convenience; `prop_<name>()` is the
      // documented form and the only one the shader property list advertises.
      // It is skipped for swizzle-shaped names because a `#define` applies to
      // the WHOLE user shader, member accesses included: a store carrying a
      // per-vertex `z` (they do -- e.g. a depth/coordinate column) would
      // `#define z vCustom3` and rewrite `d.z` into `d.vCustom3`, so the
      // colour-by-direction default stopped compiling and the layer fell back
      // to the generic per-object hash colour with no visible error.
      if (!isUnsafeBareAttributeAlias(info.name)) {
        builder.addFragmentCode(`#define ${info.name} ${propExpr}\n`);
      }
      builder.addFragmentCode(`#define prop_${info.name}() ${propExpr}\n`);
    }
    // Expose the vertex world position as `prop_position()` so a shader can
    // colour by position (attribute slot 0 is always position; `vertexIndex` is
    // defined in both the edge and node vertex mains). Interpolates along a line
    // segment, giving a smooth per-vertex position colour.
    builder.addVarying("highp vec3", "vPosition");
    vertexMain += `vPosition = readAttribute0(vertexIndex);\n`;
    builder.addFragmentCode(`#define prop_position() vPosition\n`);
    builder.setVertexMain(vertexMain);
    addControlsToBuilder(shaderBuilderState, builder);
    builder.addFragmentCode(`void userMain();\n`);
    builder.addFragmentCode(
      "#define main userMain\n" +
        shaderCodeWithLineDirective(shaderBuilderState.parseResult.code) +
        "\n#undef main\n",
    );
    builder.setFragmentMain(
      skeletonParams.spatialChunkCulling && !options?.skipChunkCull
        ? "spatialChunkCull();\nuserMain();"
        : "userMain();",
    );
  }

  private getSegmentColorExpression() {
    const index = this.segmentColorAttributeIndex;
    if (index === undefined) {
      return "uColor";
    }
    return `vCustom${index}`;
  }

  /**
   * Emit the vertex-stage assignment of the `uvec2 vSegmentValue` (the full
   * uint64 segment id) from the segment attribute.  A 1-component uint32
   * attribute (CATMAID) is zero-extended into the high half; a 2-component
   * (zarr-vectors `[lo, hi]`) attribute fills both halves.  Caller must have
   * verified `this.segmentAttributeIndex !== undefined`.
   */
  private segmentValueAssignment(vertexIndexExpr: string): string {
    const i = this.segmentAttributeIndex!;
    const read = `readAttribute${i}(${vertexIndexExpr})`;
    // A UINT64 attribute (zarr-vectors full uint64) reads as a `uint64_t`
    // whose `.value` is the uvec2; a uint32 attribute (CATMAID) reads as a
    // `uint32_t` → `toRaw(...)` gives a `uint`, zero-extended into the high
    // half.
    const rhs =
      this.vertexAttributes[i].dataType === DataType.UINT64
        ? `${read}.value`
        : `uvec2(toRaw(${read}), 0u)`;
    return `vSegmentValue = ${rhs};\n`;
  }

  edgeShaderGetter;
  nodeShaderGetter;
  /**
   * Built on first use, unlike the edge and node getters: only a surface
   * source ever asks for it, and constructing it eagerly would compile a third
   * program for every skeleton layer in the viewer that will never draw a face.
   */
  private faceShaderGetter_?: typeof this.edgeShaderGetter;

  get gl(): GL {
    return this.base.gl;
  }

  private defineDynamicSegmentAppearance(
    builder: ShaderBuilder,
    params: SkeletonShaderParameters,
  ) {
    let colorExpression = `return ${this.segmentColorShaderManager.prefix}(segmentId);`;
    let alphaExpression = `return isVisible ? uVisibleAlpha : uHiddenAlpha;`;
    let excludedSegmentAlpha = "0.0";

    if (DEBUG_EXCLUDED_SEGMENTS) {
      colorExpression = `
        if (${this.excludedSegmentsShaderManager.hasFunctionName}(segmentId)) {
          return vec3(0.0, 0.0, 1.0);
        }
        ${colorExpression}
      `;
      if (!DEBUG_SPATIAL_SKELETON_OVERLAY) alphaExpression = `return 0.0;`;
      excludedSegmentAlpha = "1.0";
    }

    // Pass-1 hide tier: a tract the pass-2 layer draws at full resolution must
    // not also be drawn here, or it renders twice -- blended over itself, so it
    // reads brighter and thicker than its neighbours. Because the pyramid is
    // strictly nested, the doubled tracts would be exactly the coarse backbone,
    // inverting the emphasis. Placed before the ghost tier: a tract pass 2 owns
    // is not ghosted, it is simply not drawn here at all.
    const roiHighDetailHideFragment = params.hasRoiHighDetailHide
      ? `
  if (uRoiHighDetailActive > 0.5 &&
      ${this.roiHighDetailSegmentsShaderManager.hasFunctionName}(segmentId)) {
    return 0.0;
  }`
      : "";

    // ROI filter tier: ghost segments that are not in the passing set while the
    // filter is active. Defined only when compiled in, so other skeleton layers
    // are unaffected.
    const roiFilterFragment = params.hasRoiFilter
      ? `
  if (uRoiFilterActive > 0.5 &&
      !${this.roiPassingSegmentsShaderManager.hasFunctionName}(segmentId)) {
    return uRoiGhostAlpha;
  }`
      : "";

    // Per-group "on" opacity: a passing tract takes the alpha byte of its
    // group's colour (packed into roiSegmentColors). Independent of
    // colour-by-group (that only gates the RGB override, below); applied only
    // when the filter is active so an inactive filter keeps the global alpha.
    const roiOpacityFragment = params.hasRoiSegmentColors
      ? `
  if (uRoiFilterActive > 0.5) {
    vec4 roiColor;
    if (${this.roiSegmentColorShaderManager.getFunctionName}(segmentId, roiColor)) {
      return roiColor.a;
    }
  }`
      : "";

    // Background (whole-tractogram) length filter: discard tracts whose value is
    // out of range, but only the BACKGROUND ones -- a tract a group claims (in
    // the passing set while the filter is active) is governed by its group's own
    // length filter, evaluated in the worker. With the filter off, every tract
    // is "background", giving the "overall" filter.
    const bgPassingExpr = params.hasRoiFilter
      ? `(uRoiFilterActive > 0.5 && ${this.roiPassingSegmentsShaderManager.hasFunctionName}(segmentId))`
      : `false`;
    const bgLengthFragment = params.hasRoiObjectValues
      ? `
  if (uBgLengthActive > 0.5 && !(${bgPassingExpr})) {
    float bgLen; float bgCol;
    if (getRoiObjectValues(segmentId, bgLen, bgCol) &&
        (bgLen < uBgLo || bgLen > uBgHi)) {
      return 0.0;
    }
  }`
      : "";

    this.visibleSegmentsShaderManager.defineShader(builder);
    this.excludedSegmentsShaderManager.defineShader(builder);
    this.segmentColorShaderManager.defineShader(builder);
    if (params.hasSegmentStatedColors) {
      this.segmentStatedColorShaderManager.defineShader(builder);
    }
    if (params.hasRoiFilter) {
      this.roiPassingSegmentsShaderManager.defineShader(builder);
      builder.addUniform("highp float", "uRoiFilterActive");
      builder.addUniform("highp float", "uRoiGhostAlpha");
    }
    if (params.hasRoiHighDetailHide) {
      this.roiHighDetailSegmentsShaderManager.defineShader(builder);
      // Its own uniform rather than a second reader of `uRoiFilterActive`: the
      // high-detail set is now also driven by the object-focused fill, which
      // runs with the ROI filter off, and pass 1 must hide whatever pass 2 draws
      // in either case.
      builder.addUniform("highp float", "uRoiHighDetailActive");
    }
    if (params.hasRoiSegmentColors) {
      this.roiSegmentColorShaderManager.defineShader(builder);
      builder.addUniform("highp float", "uRoiColorByGroup");
    }
    if (params.hasRoiObjectValues) {
      // colormapJet (glsl_COLORMAPS) is called by the edge/node emitRGB below,
      // which are added to the builder BEFORE finalizeShaderBuilder's own
      // glsl_COLORMAPS. GLSL has no function hoisting, so inject it here first;
      // ShaderCode dedups the later identical add to a no-op.
      builder.addFragmentCode(glsl_COLORMAPS);
      this.roiObjectValueShaderManager.defineShader(builder);
      builder.addUniform("highp float", "uBgLengthActive");
      builder.addUniform("highp float", "uBgLo");
      builder.addUniform("highp float", "uBgHi");
      builder.addUniform("highp float", "uBgColorMode");
      // One packed per-object value holds TWO independent normalised [0,1]
      // attributes: the length-filter attribute in bytes 0-1 and the colour
      // attribute in bytes 2-3 (16-bit each). So a length filter on attribute A
      // and colour-by attribute B coexist without a second map.
      builder.addFragmentCode(`
bool getRoiObjectValues(uint64_t segmentId, out float lengthValue, out float colorValue) {
  vec4 v = vec4(0.0);
  bool found = ${this.roiObjectValueShaderManager.getFunctionName}(segmentId, v);
  lengthValue = (floor(v.r * 255.0 + 0.5) + floor(v.g * 255.0 + 0.5) * 256.0) / 65535.0;
  colorValue = (floor(v.b * 255.0 + 0.5) + floor(v.a * 255.0 + 0.5) * 256.0) / 65535.0;
  return found;
}
`);
    }
    builder.addUniform("highp float", "uVisibleAlpha");
    builder.addUniform("highp float", "uHiddenAlpha");
    builder.addUniform("highp float", "uSaturation");
    if (params.hasSegmentDefaultColor) {
      builder.addUniform("highp vec3", "uSegmentDefaultColor");
    }
    if (params.hoverHighlight) {
      builder.addUniform("highp uvec2", "uHoveredSegmentId");
    }
    // Full uint64 segment id as a uvec2 [lo, hi].  A 1-component uint32
    // segment attribute (CATMAID) is zero-extended into this at the vertex
    // stage; a 2-component (zarr-vectors) attribute fills both halves.
    builder.addVarying("highp uvec2", "vSegmentValue", "flat");

    const statedColorFragment = params.hasSegmentStatedColors
      ? `
  vec4 statedColor;
  if (${this.segmentStatedColorShaderManager.getFunctionName}(segmentId, statedColor)) {
    return statedColor.rgb;
  }`
      : "";

    const defaultColorFragment = params.hasSegmentDefaultColor
      ? "  return uSegmentDefaultColor;"
      : `  ${colorExpression}`;

    const hoverAdjustFragment = params.hoverHighlight
      ? `
  if (segmentId.value.x == uHoveredSegmentId.x &&
      segmentId.value.y == uHoveredSegmentId.y) {
    if (saturation > 0.5) { saturation -= 0.5; }
    else { saturation += 0.5; }
  }`
      : "";

    builder.addFragmentCode(`
uint64_t getSegmentAppearanceId(highp uvec2 segmentValue) {
  return uint64_t(segmentValue);
}
vec3 getSegmentBaseColor(uint64_t segmentId) {
${statedColorFragment}
${defaultColorFragment}
}
vec3 getSegmentLookupColor(uint64_t segmentId) {
  vec3 baseColor = getSegmentBaseColor(segmentId);
  float saturation = uSaturation;
${hoverAdjustFragment}
  return mix(vec3(1.0, 1.0, 1.0), baseColor, saturation);
}
float getSegmentLookupAlpha(uint64_t segmentId) {
  if (${this.excludedSegmentsShaderManager.hasFunctionName}(segmentId)) {
    return ${excludedSegmentAlpha};
  }${bgLengthFragment}${roiHighDetailHideFragment}${roiFilterFragment}${roiOpacityFragment}
  bool isVisible = ${this.visibleSegmentsShaderManager.hasFunctionName}(segmentId);
  ${alphaExpression}
}
vec4 getSegmentAppearance(highp uvec2 segmentValue) {
  uint64_t segmentId = getSegmentAppearanceId(segmentValue);
  return vec4(getSegmentLookupColor(segmentId), getSegmentLookupAlpha(segmentId));
}
`);
  }

  maybeEnableDynamicSegmentAppearance(
    gl: GL,
    shader: ShaderProgram,
    skeletonParams: SkeletonShaderParameters,
    excludedGPUTable?: GPUHashTable<HashSetUint64>,
  ) {
    if (!skeletonParams.dynamicSegmentAppearance) return;
    const segmentationGroupState =
      this.base.displayState.segmentationGroupState.value;
    this.visibleSegmentsShaderManager.enable(
      gl,
      shader,
      segmentationGroupState.useTemporaryVisibleSegments.value
        ? this.gpuTemporaryVisibleSegmentsHashTable
        : this.gpuVisibleSegmentsHashTable,
    );
    this.excludedSegmentsShaderManager.enable(
      gl,
      shader,
      excludedGPUTable ?? this.gpuEmptySegmentsHashTable,
    );
    if (skeletonParams.hasRoiFilter) {
      const dss = this.base.displayState;
      this.roiPassingSegmentsShaderManager.enable(
        gl,
        shader,
        this.gpuRoiPassingSegmentsHashTable ?? this.gpuEmptySegmentsHashTable,
      );
      gl.uniform1f(
        shader.uniform("uRoiFilterActive"),
        dss.roiFilterActive?.value ? 1 : 0,
      );
      gl.uniform1f(
        shader.uniform("uRoiGhostAlpha"),
        dss.roiGhostAlpha?.value ?? 0,
      );
    }
    if (skeletonParams.hasRoiHighDetailHide) {
      const highDetail = this.base.displayState.roiHighDetailSegments;
      this.roiHighDetailSegmentsShaderManager.enable(
        gl,
        shader,
        this.gpuRoiHighDetailSegmentsHashTable ??
          this.gpuEmptySegmentsHashTable,
      );
      gl.uniform1f(
        shader.uniform("uRoiHighDetailActive"),
        highDetail !== undefined && highDetail.size !== 0 ? 1 : 0,
      );
    }
    // 3D (perspective) uses objectAlpha/hiddenObjectAlpha; 2D (slice) uses
    // the cross-section sliders selectedAlpha ("Opacity (on)") /
    // notSelectedAlpha ("Opacity (off)") when available, so the dense
    // skeleton's 2D and 3D opacities are controlled independently.
    const ds = this.base.displayState;
    const visibleAlpha =
      this.targetIsSliceView && ds.selectedAlpha !== undefined
        ? ds.selectedAlpha.value
        : ds.objectAlpha.value;
    const hiddenAlpha =
      this.targetIsSliceView && ds.notSelectedAlpha !== undefined
        ? ds.notSelectedAlpha.value
        : ds.hiddenObjectAlpha.value;
    gl.uniform1f(shader.uniform("uVisibleAlpha"), visibleAlpha);
    gl.uniform1f(shader.uniform("uHiddenAlpha"), hiddenAlpha);

    const colorGroupState =
      this.base.displayState.segmentationColorGroupState.value;
    this.segmentColorShaderManager.enable(
      gl,
      shader,
      colorGroupState.segmentColorHash.value,
    );

    if (skeletonParams?.hasSegmentDefaultColor) {
      const segmentDefaultColor = colorGroupState.segmentDefaultColor.value;
      if (segmentDefaultColor !== undefined) {
        gl.uniform3fv(
          shader.uniform("uSegmentDefaultColor"),
          segmentDefaultColor,
        );
      }
      if (DEBUG_SPATIAL_SKELETON_OVERLAY && excludedGPUTable === undefined) {
        gl.uniform3f(shader.uniform("uSegmentDefaultColor"), 1.0, 0.0, 0.0);
      }
    }

    if (skeletonParams?.hasSegmentStatedColors) {
      this.segmentStatedColorShaderManager.enable(
        gl,
        shader,
        this.getSegmentStatedColorHashTable(),
      );
    }

    if (
      skeletonParams?.hasRoiSegmentColors &&
      this.gpuRoiSegmentColorHashTable !== undefined
    ) {
      this.roiSegmentColorShaderManager.enable(
        gl,
        shader,
        this.gpuRoiSegmentColorHashTable,
      );
      gl.uniform1f(
        shader.uniform("uRoiColorByGroup"),
        this.base.displayState.roiColorByGroup?.value ? 1 : 0,
      );
    }

    if (
      skeletonParams?.hasRoiObjectValues &&
      this.gpuRoiObjectValueHashTable !== undefined
    ) {
      this.roiObjectValueShaderManager.enable(
        gl,
        shader,
        this.gpuRoiObjectValueHashTable,
      );
      const bg = this.base.displayState.roiBackground?.value;
      gl.uniform1f(shader.uniform("uBgLengthActive"), bg?.lengthActive ? 1 : 0);
      gl.uniform1f(shader.uniform("uBgLo"), bg?.lo ?? 0);
      gl.uniform1f(shader.uniform("uBgHi"), bg?.hi ?? 1);
      gl.uniform1f(shader.uniform("uBgColorMode"), bg?.colorMode ? 1 : 0);
    }

    const { saturation, segmentSelectionState } = this.base.displayState;
    gl.uniform1f(shader.uniform("uSaturation"), saturation.value);
    if (skeletonParams.hoverHighlight) {
      const seg = segmentSelectionState.hasSelectedSegment
        ? segmentSelectionState.selectedSegment
        : 0n;
      gl.uniform2ui(
        shader.uniform("uHoveredSegmentId"),
        Number(seg & 0xffff_ffffn),
        Number((seg >> 32n) & 0xffff_ffffn),
      );
    }
  }

  maybeDisableDynamicSegmentAppearance(
    gl: GL,
    shader: ShaderProgram,
    skeletonParams: SkeletonShaderParameters | undefined,
  ) {
    if (!skeletonParams?.dynamicSegmentAppearance) return;
    this.visibleSegmentsShaderManager.disable(gl, shader);
    this.excludedSegmentsShaderManager.disable(gl, shader);
    if (skeletonParams?.hasSegmentStatedColors) {
      this.segmentStatedColorShaderManager.disable(gl, shader);
    }
    if (skeletonParams?.hasRoiSegmentColors) {
      this.roiSegmentColorShaderManager.disable(gl, shader);
    }
    if (
      skeletonParams?.hasRoiObjectValues &&
      this.gpuRoiObjectValueHashTable !== undefined
    ) {
      this.roiObjectValueShaderManager.disable(gl, shader);
    }
  }

  constructor(
    public base: SkeletonShaderContext,
    public targetIsSliceView: boolean,
  ) {
    super();
    this.vertexIdHelper = this.registerDisposer(VertexIdHelper.get(this.gl));
    const { maxTextureImageUnits } = this.gl;
    // Packed attributes share one unit, so the count that matters is units, not
    // attributes: a store with a thousand columns is fine, and a store with
    // twenty unpacked ones is not.
    const packedCount = this.base.packedAttributeRange?.count ?? 0;
    const textureUnitsNeeded =
      this.vertexAttributes.length - packedCount + (packedCount > 0 ? 1 : 0);
    if (textureUnitsNeeded > maxTextureImageUnits) {
      console.warn(
        `Skeleton needs ${textureUnitsNeeded} texture units for ` +
          `${this.vertexAttributes.length} vertex attributes but device only ` +
          `supports ${maxTextureImageUnits}`,
      );
    }
    const segmentAttrIndex = this.vertexAttributes.findIndex(
      (x) => x.name === segmentAttribute.name,
    );
    this.segmentAttributeIndex =
      segmentAttrIndex >= 0 ? segmentAttrIndex : undefined;
    this.segmentColorAttributeIndex = base.segmentColorAttributeIndex;
    const selectedNodeAttrIndex = this.vertexAttributes.findIndex(
      (x) => x.name === selectedNodeAttribute.name,
    );
    this.selectedNodeAttributeIndex =
      selectedNodeAttrIndex >= 0 ? selectedNodeAttrIndex : undefined;

    // `segmentationGroupState` may be captured once: changing it goes through
    // `linkedSegmentationGroup.changed` -> `updateDataSubsourceActivations()`,
    // which rebuilds this layer. The colour group has no such handler and is
    // swapped in place, so its table is resolved per draw instead -- see
    // `getSegmentStatedColorHashTable`.
    const segmentationGroupState =
      base.displayState.segmentationGroupState.value;

    this.gpuVisibleSegmentsHashTable = this.registerDisposer(
      GPUHashTable.get(
        this.gl,
        segmentationGroupState.visibleSegments.hashTable,
      ),
    );
    this.gpuTemporaryVisibleSegmentsHashTable = this.registerDisposer(
      GPUHashTable.get(
        this.gl,
        segmentationGroupState.temporaryVisibleSegments.hashTable,
      ),
    );
    this.gpuEmptySegmentsHashTable = this.registerDisposer(
      GPUHashTable.get(this.gl, this.emptySegmentSet.hashTable),
    );
    const roiPassingSegments = base.displayState.roiPassingSegments;
    if (roiPassingSegments !== undefined) {
      this.gpuRoiPassingSegmentsHashTable = this.registerDisposer(
        GPUHashTable.get(this.gl, roiPassingSegments.hashTable),
      );
    }
    const roiHighDetailSegments = base.displayState.roiHighDetailSegments;
    if (roiHighDetailSegments !== undefined) {
      this.gpuRoiHighDetailSegmentsHashTable = this.registerDisposer(
        GPUHashTable.get(this.gl, roiHighDetailSegments.hashTable),
      );
    }
    const roiSegmentColors = base.displayState.roiSegmentColors;
    if (roiSegmentColors !== undefined) {
      this.gpuRoiSegmentColorHashTable = this.registerDisposer(
        GPUHashTable.get(this.gl, roiSegmentColors.hashTable),
      );
    }
    const roiObjectValues = base.displayState.roiObjectValues;
    if (roiObjectValues !== undefined) {
      this.gpuRoiObjectValueHashTable = this.registerDisposer(
        GPUHashTable.get(this.gl, roiObjectValues.hashTable),
      );
    }

    this.edgeShaderGetter = parameterizedEmitterDependentShaderGetter(
      this,
      this.gl,
      {
        memoizeKey: {
          type: "skeleton/SkeletonShaderManager/edge",
          vertexAttributes: this.vertexAttributes,
        },
        fallbackParameters: this.base.fallbackShaderParameters,
        parameters:
          this.base.displayState.skeletonRenderingOptions.shaderControlState
            .builderState,
        extraParameters: this.base.skeletonShaderParameters,
        shaderError: this.base.displayState.shaderError,
        defineShader: (
          builder: ShaderBuilder,
          shaderBuilderState: ShaderControlsBuilderState,
          skeletonParams: SkeletonShaderParameters,
        ) => {
          this.defineCommonShader(builder, shaderBuilderState, skeletonParams);
          defineLineShader(builder);
          builder.addAttribute("highp uvec2", "aVertexIndex");
          builder.addUniform("highp float", "uLineWidth");
          let vertexMain = `
highp uint pickOffset = uint(gl_InstanceID) * uPickInstanceStride;
vPickID = uPickID + pickOffset;
highp vec3 vertexA = readAttribute0(aVertexIndex.x);
highp vec3 vertexB = readAttribute0(aVertexIndex.y);
emitLine(uProjection, vertexA, vertexB, uLineWidth);
highp uint lineEndpointIndex = getLineEndpointIndex();
highp uint vertexIndex = aVertexIndex.x * (1u - lineEndpointIndex) + aVertexIndex.y * lineEndpointIndex;
`;
          if (skeletonParams.spatialChunkCulling) {
            vertexMain += `vCullPos = mix(vertexA, vertexB, float(lineEndpointIndex));\n`;
          }
          if (
            skeletonParams.dynamicSegmentAppearance &&
            this.segmentAttributeIndex !== undefined
          ) {
            vertexMain += this.segmentValueAssignment("aVertexIndex.x");
          }

          const segmentColorExpression = this.getSegmentColorExpression();
          const segmentAlphaExpression =
            this.segmentColorAttributeIndex === undefined
              ? "uColor.a"
              : `${segmentColorExpression}.a`;
          if (skeletonParams.dynamicSegmentAppearance) {
            // ROI colour-by-group: override the (directional) RGB with the
            // streamline's group colour when the filter is colouring by group
            // and this object is in the group colour map. Reads the dedicated
            // roiSegmentColor tier — never the user-facing segmentStatedColors.
            const roiColorFragment = skeletonParams.hasRoiSegmentColors
              ? `
  if (uRoiColorByGroup > 0.5) {
    vec4 roiColor;
    if (${this.roiSegmentColorShaderManager.getFunctionName}(getSegmentAppearanceId(vSegmentValue), roiColor)) {
      rgb = roiColor.rgb;
    }
  }`
              : "";
            // Background flat colour-by-attribute: recolour the BACKGROUND tracts
            // (not claimed by a group) by their per-object value through a
            // colourmap. Group-claimed tracts keep their group/attribute colour
            // (roiColorFragment, above), which is why the passing check mirrors
            // the length tier's.
            const bgColorPassingExpr = skeletonParams.hasRoiFilter
              ? `(uRoiFilterActive > 0.5 && ${this.roiPassingSegmentsShaderManager.hasFunctionName}(getSegmentAppearanceId(vSegmentValue)))`
              : `false`;
            const bgColorFragment = skeletonParams.hasRoiObjectValues
              ? `
  if (uBgColorMode > 0.5 && !(${bgColorPassingExpr})) {
    float bgLen; float bgCol;
    if (getRoiObjectValues(getSegmentAppearanceId(vSegmentValue), bgLen, bgCol)) {
      rgb = colormapJet(bgCol);
    }
  }`
              : "";
            // Dynamic path (spatial skeletons): per-segment color, visibility,
            // and hover highlight resolved in the shader via
            // getSegmentAppearance(). uColor is unused in this path. Saturation
            // is applied to the emitDefault (segment-colour) path inside
            // getSegmentLookupColor, but a custom shader's emitRGB colour (e.g.
            // the tract colour-by-direction default, or a group colour) bypasses
            // it, so mix it in here too — otherwise the Saturation slider has no
            // effect on directional/group-coloured streamlines.
            builder.addFragmentCode(
              this.dynamicAppearanceEmitters({
                coverage: "getLineAlpha()",
                roiColorFragment,
                bgColorFragment,
              }),
            );
          } else if (this.segmentColorAttributeIndex === undefined) {
            // Legacy path (non-spatial skeletons): one skeleton drawn per call;
            // uColor is set per-skeleton by the CPU via getObjectColor(), which
            // already incorporates saturation and hover highlighting.
            const roiLegacyColorAssign = skeletonParams.hasRoiSegmentColors
              ? "\n  if (uRoiColorByGroup > 0.5) { rgb = uColor.rgb; }"
              : "";
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGB(vec3 color) {
  vec3 rgb = color;${roiLegacyColorAssign}
  emit(vec4(rgb * uColor.a, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), vPickID);
}
void emitDefault() {
  emit(vec4(uColor.rgb, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), vPickID);
}
`);
          } else {
            // Per-vertex color attribute path: color comes from a per-vertex
            // attribute; alpha is taken from uColor.
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGB(vec3 color) {
  highp float alpha = ${segmentAlphaExpression} * getLineAlpha() * ${this.getCrossSectionFadeFactor()};
  ${this.emitColorStatement("color", "alpha")}
}
void emitDefault() {
  vec4 baseColor = segmentColor();
  highp float alpha = baseColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()};
  ${this.emitColorStatement("baseColor.rgb", "alpha")}
}
`);
          }
          this.finalizeShaderBuilder(
            builder,
            shaderBuilderState,
            skeletonParams,
            vertexMain,
            {
              // A line fragment lies between two vertices; the smooth
              // coefficient reproduces the blend a per-attribute varying
              // used to get from the rasteriser.
              packedAttributeInterp: {
                mode: "line",
                pairExpr: "aVertexIndex",
                endpointCoefficientExpr: "getLineEndpointCoefficient()",
              },
            },
          );
        },
      },
    );

    this.nodeShaderGetter = parameterizedEmitterDependentShaderGetter(
      this,
      this.gl,
      {
        memoizeKey: {
          type: "skeleton/SkeletonShaderManager/node",
          vertexAttributes: this.vertexAttributes,
        },
        fallbackParameters: this.base.fallbackShaderParameters,
        parameters:
          this.base.displayState.skeletonRenderingOptions.shaderControlState
            .builderState,
        extraParameters: this.base.skeletonShaderParameters,
        shaderError: this.base.displayState.shaderError,
        defineShader: (
          builder: ShaderBuilder,
          shaderBuilderState: ShaderControlsBuilderState,
          skeletonParams: SkeletonShaderParameters,
        ) => {
          this.defineCommonShader(builder, shaderBuilderState, skeletonParams);
          defineCircleShader(
            builder,
            /*crossSectionFade=*/ this.targetIsSliceView,
          );
          builder.addUniform("highp float", "uNodeDiameter");
          let selectedOutlineWidthExpression = "0.0";
          if (this.selectedNodeAttributeIndex !== undefined) {
            builder.addVarying("highp float", "vSelectedNode", "flat");
            const selectedOutlineMinWidth = this.targetIsSliceView
              ? SELECTED_NODE_OUTLINE_MIN_WIDTH_2D
              : SELECTED_NODE_OUTLINE_MIN_WIDTH_3D;
            const selectedOutlineMaxWidth = this.targetIsSliceView
              ? SELECTED_NODE_OUTLINE_MAX_WIDTH_2D
              : SELECTED_NODE_OUTLINE_MAX_WIDTH_3D;
            selectedOutlineWidthExpression = `((vSelectedNode > 0.5) ? clamp(0.25 * uNodeDiameter, ${selectedOutlineMinWidth}, ${selectedOutlineMaxWidth}) : 0.0)`;
          }
          let vertexMain = `
highp uint vertexIndex = uint(gl_InstanceID);
highp uint pickOffset = vertexIndex * uPickInstanceStride;
vPickID = uPickID + pickOffset;
highp vec3 vertexPosition = readAttribute0(vertexIndex);
`;
          if (skeletonParams.spatialChunkCulling) {
            vertexMain += `vCullPos = vertexPosition;\n`;
          }
          if (this.selectedNodeAttributeIndex !== undefined) {
            vertexMain += `vSelectedNode = readAttribute${this.selectedNodeAttributeIndex}(vertexIndex);\n`;
          }
          if (
            skeletonParams.dynamicSegmentAppearance &&
            this.segmentAttributeIndex !== undefined
          ) {
            vertexMain += this.segmentValueAssignment("vertexIndex");
          }
          vertexMain += `
emitCircle(
  uProjection * vec4(vertexPosition, 1.0),
  uNodeDiameter,
  ${selectedOutlineWidthExpression}
);
`;
          const segmentColorExpression = this.getSegmentColorExpression();
          if (
            skeletonParams.dynamicSegmentAppearance &&
            this.segmentAttributeIndex !== undefined
          ) {
            // Dynamic path (spatial skeletons): per-segment color, visibility,
            // saturation and hover highlight all resolved in the shader via
            // getSegmentAppearance(). uColor is unused in this path.
            const segmentExpression = `vSegmentValue`;
            const selectedNodeExpression =
              this.selectedNodeAttributeIndex === undefined
                ? undefined
                : "vSelectedNode";
            const borderColorExpression =
              selectedNodeExpression === undefined
                ? "renderColor"
                : `((${selectedNodeExpression} > 0.5) ? vec4(${SELECTED_NODE_OUTLINE_COLOR_RGB}, renderColor.a) : renderColor)`;
            // Background attribute colour for node dots, mirroring the edge path
            // so a tract's line and its vertex dots agree.
            const nodeBgColorPassingExpr = skeletonParams.hasRoiFilter
              ? `(uRoiFilterActive > 0.5 && ${this.roiPassingSegmentsShaderManager.hasFunctionName}(getSegmentAppearanceId(vSegmentValue)))`
              : `false`;
            const nodeBgColorFragment = skeletonParams.hasRoiObjectValues
              ? `
  if (uBgColorMode > 0.5 && !(${nodeBgColorPassingExpr})) {
    float bgLen; float bgCol;
    if (getRoiObjectValues(getSegmentAppearanceId(vSegmentValue), bgLen, bgCol)) {
      rgb = colormapJet(bgCol);
    }
  }`
              : "";
            builder.addFragmentCode(`
vec4 segmentColor() {
  return getSegmentAppearance(${segmentExpression});
}
void emitRGBA(vec4 color) {
  vec4 baseColor = segmentColor();
  highp float alpha = color.a * baseColor.a;
  if (alpha <= 0.0) discard;
  vec4 renderColor = vec4(color.rgb, alpha);
  vec4 borderColor = ${borderColorExpression};
  vec4 circleColor = getCircleColor(renderColor, borderColor);
  ${this.emitColorStatement("circleColor.rgb", "circleColor.a")}
}
void emitRGB(vec3 color) {
  // Saturation for a custom shader's node colour (see the edge path); the
  // emitDefault segment-colour path is already saturated via getSegmentLookupColor.
  vec3 rgb = color;${nodeBgColorFragment}
  emitRGBA(vec4(mix(vec3(1.0), rgb, uSaturation), 1.0));
}
void emitDefault() {
  emitRGBA(vec4(segmentColor().rgb, 1.0));
}
`);
          } else if (this.segmentColorAttributeIndex === undefined) {
            // Legacy path (non-spatial skeletons): one skeleton drawn per call;
            // uColor is set per-skeleton by the CPU via getObjectColor(), which
            // already incorporates saturation and hover highlighting.
            const roiLegacyColorAssign = skeletonParams.hasRoiSegmentColors
              ? "\n  if (uRoiColorByGroup > 0.5) { rgb = uColor.rgb; }"
              : "";
            // For ROI high-detail streamline nodes the per-group opacity rides
            // uColor.a and the rgb is premultiplied by it (matching the edge
            // path and the perspective OIT premultiplied-alpha convention);
            // other legacy skeletons keep the historical fully-opaque,
            // straight-rgb node behaviour (rgb * 1.0 == rgb).
            const roiLegacyNodeAlpha = skeletonParams.hasRoiSegmentColors
              ? "uColor.a"
              : "1.0";
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGBA(vec4 color) {
  vec4 borderColor = color;
  emit(getCircleColor(color, borderColor), vPickID);
}
void emitRGB(vec3 color) {
  vec3 rgb = color;${roiLegacyColorAssign}
  emitRGBA(vec4(rgb * ${roiLegacyNodeAlpha}, ${roiLegacyNodeAlpha}));
}
void emitDefault() {
  emitRGBA(uColor);
}
`);
          } else {
            // Per-vertex color attribute path: color comes from a per-vertex
            // attribute; alpha is taken from the attribute's alpha component.
            const selectedNodeExpression =
              this.selectedNodeAttributeIndex === undefined
                ? undefined
                : "vSelectedNode";
            const borderColorExpression =
              selectedNodeExpression === undefined
                ? "renderColor"
                : `((${selectedNodeExpression} > 0.5) ? vec4(${SELECTED_NODE_OUTLINE_COLOR_RGB}, renderColor.a) : renderColor)`;
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGBA(vec4 color) {
  vec4 renderColor = color;
  vec4 borderColor = ${borderColorExpression};
  vec4 circleColor = getCircleColor(renderColor, borderColor);
  ${this.emitColorStatement("circleColor.rgb", "circleColor.a")}
}
void emitRGB(vec3 color) {
  emitRGBA(vec4(color, 1.0));
}
void emitDefault() {
  emitRGBA(segmentColor());
}
`);
          }
          this.finalizeShaderBuilder(
            builder,
            shaderBuilderState,
            skeletonParams,
            vertexMain,
            {
              packedAttributeInterp: {
                mode: "point",
                vertexIndexExpr: "vertexIndex",
              },
            },
          );
        },
      },
    );
  }

  /** Whether attribute `i` rides the packed texture rather than its own. */
  private isPackedAttribute(i: number) {
    const range = this.base.packedAttributeRange;
    if (range === undefined) return false;
    // The internal columns keep their own texture whatever a source declares:
    // each is read in the VERTEX stage (segment ids feed `vSegmentValue`, the
    // selected-node flag sizes the outline, the colour column is read as
    // `vCustom<i>`), which a fragment-stage packed fetch cannot serve. A source
    // that packed one of them would produce a shader referencing a varying that
    // was never declared.
    if (
      i === this.segmentAttributeIndex ||
      i === this.selectedNodeAttributeIndex ||
      i === this.segmentColorAttributeIndex
    ) {
      return false;
    }
    return i >= range.start && i < range.start + range.count;
  }

  defineAttributeAccess(builder: ShaderBuilder) {
    const { textureAccessHelper } = this;
    textureAccessHelper.defineShader(builder);
    const numAttributes = this.vertexAttributes.length;
    for (let j = vertexAttributeSamplerSymbols.length; j < numAttributes; ++j) {
      vertexAttributeSamplerSymbols[j] = Symbol(
        `SkeletonShader.vertexAttributeTextureUnit${j}`,
      );
    }
    this.vertexAttributes.forEach((info, i) => {
      // Packed attributes share one sampler, declared below; giving each its
      // own here is exactly the texture-unit exhaustion the packing avoids.
      if (this.isPackedAttribute(i)) return;
      builder.addTextureSampler(
        `${getSamplerPrefixForDataType(
          info.dataType,
        )}sampler2D` as ShaderSamplerType,
        `uVertexAttributeSampler${i}`,
        vertexAttributeSamplerSymbols[i],
      );
      builder.addVertexCode(
        textureAccessHelper.getAccessor(
          `readAttribute${i}`,
          `uVertexAttributeSampler${i}`,
          info.dataType,
          info.numComponents,
        ),
      );
    });
    if (this.base.packedAttributeRange !== undefined) {
      builder.addTextureSampler(
        "sampler2D",
        "uPackedAttributeSampler",
        packedAttributeSamplerSymbol,
      );
      // Per chunk, not per layer: the run is laid out attribute-major, so the
      // stride between one attribute and the next is that chunk's vertex count.
      builder.addUniform("highp uint", PACKED_ATTRIBUTE_STRIDE_UNIFORM);
      // Fragment stage, deliberately. Reading in the vertex stage would need a
      // varying per attribute to carry the value across, which is the second
      // ceiling (60 varying components) the packing exists to remove. Fetching
      // here also means an attribute no shader mentions costs nothing at all.
      builder.addFragmentCode(
        textureAccessHelper.getAccessor(
          "readPackedAttributeValue",
          "uPackedAttributeSampler",
          DataType.FLOAT32,
          1,
        ),
      );
      builder.addFragmentCode(
        packedAttributeAccessorCode("readPackedAttributeValue"),
      );
    }
  }

  /**
   * Declare the varyings a packed run needs and return the vertex-stage
   * statements that fill them. A fixed cost -- one or two varyings -- however
   * many attributes the store has.
   */
  private definePackedAttributeVaryings(
    builder: ShaderBuilder,
    interp: PackedAttributeInterp,
  ): string {
    const { varyings, vertexMain } = packedAttributeVaryings(interp);
    for (const { type, name, interpolationMode } of varyings) {
      builder.addVarying(type, name, interpolationMode);
    }
    return vertexMain;
  }

  /**
   * Define `prop_<name>()` (and the bare alias) for one packed attribute, as a
   * fetch blended over the primitive's vertices.
   */
  private definePackedAttributeMacros(
    builder: ShaderBuilder,
    name: string,
    packedIndex: number,
    interp: PackedAttributeInterp,
  ) {
    if (name === SYNTHESISED_POSITION_ATTRIBUTE_NAME) return;
    const propExpr = packedAttributePropExpr(packedIndex, interp);
    if (!isUnsafeBareAttributeAlias(name)) {
      builder.addFragmentCode(`#define ${name} ${propExpr}\n`);
    }
    builder.addFragmentCode(`#define prop_${name}() ${propExpr}\n`);
  }

  /**
   * Bind one chunk's attribute textures for `shader`: a unit per unpacked
   * attribute, plus one for the whole packed run.
   */
  private bindAttributeTextures(
    gl: GL,
    shader: ShaderProgram,
    geometry: SkeletonGPUGeometry,
  ) {
    const { vertexAttributes } = this;
    const { vertexAttributeTextures } = geometry;
    for (let i = 0; i < vertexAttributes.length; ++i) {
      // A packed attribute has no sampler of its own to bind to; asking the
      // program for its texture unit would return undefined.
      if (this.isPackedAttribute(i)) continue;
      gl.activeTexture(
        WebGL2RenderingContext.TEXTURE0 +
          shader.textureUnit(vertexAttributeSamplerSymbols[i]),
      );
      gl.bindTexture(
        WebGL2RenderingContext.TEXTURE_2D,
        vertexAttributeTextures[i],
      );
    }
    if (this.base.packedAttributeRange !== undefined) {
      gl.activeTexture(
        WebGL2RenderingContext.TEXTURE0 +
          shader.textureUnit(packedAttributeSamplerSymbol),
      );
      gl.bindTexture(
        WebGL2RenderingContext.TEXTURE_2D,
        geometry.packedAttributeTexture ?? null,
      );
    }
  }

  /**
   * Tell `shader` how far apart consecutive attributes sit in the packed
   * texture. Per chunk and per program: uniforms do not survive a program
   * switch, so this runs after every `bind()` rather than once per chunk.
   */
  private setPackedAttributeStride(
    gl: GL,
    shader: ShaderProgram,
    geometry: SkeletonGPUGeometry,
  ) {
    if (this.base.packedAttributeRange === undefined) return;
    gl.uniform1ui(
      shader.uniform("uPackedAttributeStride"),
      geometry.numVertices,
    );
  }

  /**
   * The fragment-stage emitters for the dynamic (spatially-indexed) path, where
   * colour, visibility and every ROI tier are resolved per segment in the
   * shader by `getSegmentAppearance`.
   *
   * Shared by the edge and face shaders, which differ only in two places:
   * `coverage` is the primitive's own anti-aliasing factor (a line's analytic
   * quad coverage; a triangle has none, so `1.0`), and `rgbModulation` scales
   * the emitted colour without touching alpha -- how a lit surface shades.
   * Keeping the ROI tiers in one place is the point: they are subtle, ordered,
   * and a second copy would drift.
   */
  /**
   * Shader for surface geometry: one instanced triangle per face.
   *
   * The face index triple arrives as an instanced integer vertex attribute and
   * `gl_VertexID % 3` picks the corner, exactly as the edge shader extrudes a
   * quad from an index pair. Instancing (rather than an indexed draw) is what
   * makes `gl_InstanceID` the face number, which is what the pick IDs are keyed
   * on, and it also puts all three corner positions in every vertex invocation
   * -- so the flat normal is a cross product with no normal data uploaded.
   */
  get faceShaderGetter(): typeof this.edgeShaderGetter {
    let getter = this.faceShaderGetter_;
    if (getter !== undefined) return getter;
    getter = this.faceShaderGetter_ = parameterizedEmitterDependentShaderGetter(
      this,
      this.gl,
      {
        // Distinct from the edge shader's key: same vertexAttributes, different
        // program, and gl.memoize would otherwise hand back the wrong one.
        memoizeKey: {
          type: "skeleton/SkeletonShaderManager/face",
          vertexAttributes: this.vertexAttributes,
        },
        fallbackParameters: this.base.fallbackShaderParameters,
        parameters:
          this.base.displayState.skeletonRenderingOptions.shaderControlState
            .builderState,
        extraParameters: this.base.skeletonShaderParameters,
        shaderError: this.base.displayState.shaderError,
        defineShader: (
          builder: ShaderBuilder,
          shaderBuilderState: ShaderControlsBuilderState,
          skeletonParams: SkeletonShaderParameters,
        ) => {
          this.defineCommonShader(builder, shaderBuilderState, skeletonParams);
          builder.addAttribute("highp uvec3", "aFaceIndex");
          // Model-space light direction packed with the ambient term, as the
          // mesh layer does. Zero in a slice view, which has no lighting inputs
          // at all -- a cross-section of a surface is then flat-shaded.
          builder.addUniform("highp vec4", "uLightDirection");
          // Model -> display normal matrix. The corner positions are model
          // space, so the cross product is a model-space normal, while the
          // light direction is a display-space vector; under an anisotropic
          // transform the two frames disagree and the surface is lit from the
          // wrong direction. Same correction `MeshLayer` applies.
          builder.addUniform("highp mat3", "uNormalMatrix");
          builder.addVarying("highp vec3", "vNormal", "flat");
          let vertexMain = `
highp uint pickOffset = uint(gl_InstanceID) * uPickInstanceStride;
vPickID = uPickID + pickOffset;
highp vec3 cornerA = readAttribute0(aFaceIndex.x);
highp vec3 cornerB = readAttribute0(aFaceIndex.y);
highp vec3 cornerC = readAttribute0(aFaceIndex.z);
vNormal = normalize(uNormalMatrix * cross(cornerB - cornerA, cornerC - cornerA));
highp uint corner = uint(gl_VertexID % 3);
// Ternary chain rather than aFaceIndex[corner]: dynamic indexing of a vector
// by a non-constant is where quad.ts documents an Apple GPU miscompilation.
highp uint vertexIndex = corner == 0u ? aFaceIndex.x : (corner == 1u ? aFaceIndex.y : aFaceIndex.z);
highp vec3 vertexPosition = corner == 0u ? cornerA : (corner == 1u ? cornerB : cornerC);
gl_Position = uProjection * vec4(vertexPosition, 1.0);
`;
          // `vCullPos` is still declared by defineCommonShader when culling is
          // compiled in, and an unwritten varying is undefined -- but the face
          // shader never calls spatialChunkCull(), so nothing reads it.
          if (skeletonParams.spatialChunkCulling) {
            vertexMain += "vCullPos = vertexPosition;\n";
          }
          if (
            skeletonParams.dynamicSegmentAppearance &&
            this.segmentAttributeIndex !== undefined
          ) {
            // One face, one segment: the first corner decides, mirroring the
            // edge shader taking endpoint A.
            vertexMain += this.segmentValueAssignment("aFaceIndex.x");
          }
          // `abs` makes the shading independent of winding, which ZVF does not
          // declare and nothing here culls on.
          const lighting =
            "(abs(dot(vNormal, uLightDirection.xyz)) + uLightDirection.w)";
          builder.addFragmentCode(
            this.dynamicAppearanceEmitters({
              // A triangle has no analytic edge coverage to fade by.
              coverage: "1.0",
              roiColorFragment: "",
              bgColorFragment: "",
              rgbModulation: lighting,
            }),
          );
          this.finalizeShaderBuilder(
            builder,
            shaderBuilderState,
            skeletonParams,
            vertexMain,
            {
              skipChunkCull: true,
              // One flat uvec3 plus barycentric weights carries every
              // packed attribute across the stage boundary.
              packedAttributeInterp: {
                mode: "face",
                triExpr: "aFaceIndex",
                cornerExpr: "corner",
              },
            },
          );
        },
      },
    );
    return getter;
  }

  /**
   * Draw one instanced triangle per face. Mirrors the edge pass in
   * {@link drawSkeletons}: the index buffer is bound as an instanced integer
   * attribute rather than an element array, so `gl_InstanceID` is the face
   * number the pick IDs are keyed on.
   */
  private tempLightVec = new Float32Array(4);

  /**
   * Upload the model-space light direction packed with the ambient term, the
   * same `vec4(direction * directional, ambient)` convention `MeshLayer` uses
   * (src/mesh/frontend.ts:250-262).
   *
   * A slice view carries no lighting inputs at all -- `SliceViewPanelRenderContext`
   * has no `lightDirection` -- so a cross-section of a surface is drawn fully
   * ambient rather than half-lit by a direction that does not exist there.
   */
  setLightDirection(
    gl: GL,
    shader: ShaderProgram,
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    modelMatrix: mat4,
  ) {
    // Normal matrix: the inverse transpose of the model matrix, scaled by the
    // canonical voxel factors, exactly as `MeshLayer.beginLayer` builds it.
    mat3FromMat4(tempNormalMat3, modelMatrix);
    scaleMat3Output(
      tempNormalMat3,
      tempNormalMat3,
      renderContext.projectionParameters.displayDimensionRenderInfo
        .canonicalVoxelFactors,
    );
    mat3.invert(tempNormalMat3, tempNormalMat3);
    mat3.transpose(tempNormalMat3, tempNormalMat3);
    gl.uniformMatrix3fv(shader.uniform("uNormalMatrix"), false, tempNormalMat3);
    const lightVec = this.tempLightVec;
    const perspective = renderContext as Partial<PerspectiveViewRenderContext>;
    if (perspective.lightDirection === undefined) {
      lightVec[0] = lightVec[1] = lightVec[2] = 0;
      lightVec[3] = 1;
    } else {
      const directional = perspective.directionalLighting ?? 0;
      lightVec[0] = perspective.lightDirection[0] * directional;
      lightVec[1] = perspective.lightDirection[1] * directional;
      lightVec[2] = perspective.lightDirection[2] * directional;
      lightVec[3] = perspective.ambientLighting ?? 1;
    }
    gl.uniform4fv(shader.uniform("uLightDirection"), lightVec);
  }

  drawTriangles(
    gl: GL,
    faceShader: ShaderProgram,
    skeletonGpuGeometry: SkeletonGPUGeometry,
  ) {
    this.bindAttributeTextures(gl, faceShader, skeletonGpuGeometry);
    faceShader.bind();
    this.setPackedAttributeStride(gl, faceShader, skeletonGpuGeometry);
    const aFaceIndex = faceShader.attribute("aFaceIndex");
    skeletonGpuGeometry.indexBuffer.bindToVertexAttribI(
      aFaceIndex,
      3,
      WebGL2RenderingContext.UNSIGNED_INT,
    );
    gl.vertexAttribDivisor(aFaceIndex, 1);
    gl.drawArraysInstanced(
      WebGL2RenderingContext.TRIANGLES,
      0,
      3,
      skeletonGpuGeometry.numIndices / 3,
    );
    gl.vertexAttribDivisor(aFaceIndex, 0);
    gl.disableVertexAttribArray(aFaceIndex);
  }

  private dynamicAppearanceEmitters(options: {
    coverage: string;
    roiColorFragment: string;
    bgColorFragment: string;
    rgbModulation?: string;
  }): string {
    const { coverage, roiColorFragment, bgColorFragment } = options;
    const modulate =
      options.rgbModulation === undefined
        ? ""
        : `\n  rgb *= ${options.rgbModulation};`;
    const fade = this.getCrossSectionFadeFactor();
    return `
vec4 segmentColor() {
  return getSegmentAppearance(vSegmentValue);
}
void emitRGB(vec3 color) {
  vec4 baseColor = segmentColor();
  highp float alpha = baseColor.a * ${coverage} * ${fade};
  if (alpha <= 0.0) discard;
  // Group colour first (claimed tracts), then the background attribute colour,
  // which skips claimed tracts -- so with the ROI filter off the whole
  // tractogram is "background" and owns the colour, matching the length tier.
  vec3 rgb = color;${roiColorFragment}${bgColorFragment}
  rgb = mix(vec3(1.0), rgb, uSaturation);${modulate}
  ${this.emitColorStatement("rgb", "alpha")}
}
void emitDefault() {
  vec4 baseColor = segmentColor();
  highp float alpha = baseColor.a * ${coverage} * ${fade};
  if (alpha <= 0.0) discard;
  vec3 rgb = baseColor.rgb;${modulate}
  ${this.emitColorStatement("rgb", "alpha")}
}
`;
  }

  getCrossSectionFadeFactor() {
    if (this.targetIsSliceView) {
      return "(clamp(1.0 - 2.0 * abs(0.5 - gl_FragCoord.z), 0.0, 1.0))";
    }
    return "(1.0)";
  }

  // GLSL statement emitting an (rgb, alpha) color with the alpha convention the
  // target expects: the 2D slice view blends with straight alpha
  // (`SRC_ALPHA, ONE_MINUS_SRC_ALPHA`), while the perspective OIT path requires
  // premultiplied color. Emitting premultiplied into the slice view's straight
  // blend would multiply rgb by alpha twice, darkening (rather than fading)
  // colors as the cross-section fade lowers alpha.
  private emitColorStatement(rgb: string, alpha: string): string {
    return this.targetIsSliceView
      ? `emit(vec4(${rgb}, ${alpha}), vPickID);`
      : `emit(vec4((${rgb}) * (${alpha}), ${alpha}), vPickID);`;
  }

  beginLayer(
    gl: GL,
    shader: ShaderProgram,
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    modelMatrix: mat4,
  ) {
    const { viewProjectionMat } = renderContext.projectionParameters;
    const mat = mat4.multiply(tempMat4, viewProjectionMat, modelMatrix);
    gl.uniformMatrix4fv(shader.uniform("uProjection"), false, mat);
    this.vertexIdHelper.enable();
  }

  setColor(gl: GL, shader: ShaderProgram, color: vec4) {
    gl.uniform4fv(shader.uniform("uColor"), color);
  }

  setPickID(gl: GL, shader: ShaderProgram, pickID: number) {
    gl.uniform1ui(shader.uniform("uPickID"), pickID);
  }

  setPickInstanceStride(gl: GL, shader: ShaderProgram, stride: number) {
    gl.uniform1ui(shader.uniform("uPickInstanceStride"), stride);
  }

  setChunkBounds(
    gl: GL,
    shader: ShaderProgram,
    origin: Float32Array,
    upperBound: Float32Array,
  ) {
    gl.uniform3fv(shader.uniform("uChunkOrigin"), origin);
    gl.uniform3fv(shader.uniform("uChunkBound"), upperBound);
  }

  drawSkeletons(
    gl: GL,
    edgeShader: ShaderProgram,
    nodeShader: ShaderProgram | null,
    skeletonGpuGeometry: SkeletonGPUGeometry,
    projectionParameters: { width: number; height: number },
    renderMode: SkeletonRenderMode = SkeletonRenderMode.LINES_AND_POINTS,
    primitive: GeometryPrimitive = "lines",
  ) {
    // Bind vertex attribute textures to be used across edge and node shaders
    // The edge shader and node shader share the same texture unit for each attribute
    // so we only bind once. However, if this ever changes, we
    // instead must bind for the edge shader, draw, then bind for node shader
    this.bindAttributeTextures(gl, edgeShader, skeletonGpuGeometry);

    // Draw edges.  Skipped entirely for point geometry: there are no edges,
    // and binding the edge shader for a zero-instance draw is pure overhead.
    if (primitive !== "points") {
      edgeShader.bind();
      this.setPackedAttributeStride(gl, edgeShader, skeletonGpuGeometry);
      const aVertexIndex = edgeShader.attribute("aVertexIndex");
      skeletonGpuGeometry.indexBuffer.bindToVertexAttribI(
        aVertexIndex,
        2,
        WebGL2RenderingContext.UNSIGNED_INT,
      );
      gl.vertexAttribDivisor(aVertexIndex, 1);
      initializeLineShader(
        edgeShader,
        projectionParameters,
        this.targetIsSliceView ? 1.0 : 0.0,
      );
      drawLines(gl, 1, skeletonGpuGeometry.numIndices / 2);
      gl.vertexAttribDivisor(aVertexIndex, 0);
      gl.disableVertexAttribArray(aVertexIndex);
    }

    // Draw node dots in "lines and points" mode — in "lines" mode the user
    // wants line segments only.  Point geometry always draws them: its vertices
    // are the whole drawing, so "lines" would leave the layer empty.
    if (
      nodeShader !== null &&
      (primitive === "points" || renderMode !== SkeletonRenderMode.LINES)
    ) {
      nodeShader.bind();
      this.setPackedAttributeStride(gl, nodeShader, skeletonGpuGeometry);
      initializeCircleShader(nodeShader, projectionParameters, {
        featherWidthInPixels: this.targetIsSliceView ? 1.0 : 0.0,
      });
      // ONE circle per instance: the node shader calls `emitCircle` exactly
      // once and keys it on `gl_InstanceID`, so a second quad's worth of
      // vertices just re-runs `gl_VertexID % 6` over the same corners and
      // rasterises the identical disc on top of itself. Harmless on a skeleton
      // with a few thousand nodes; on a point cloud the vertices ARE the
      // drawing, so it doubled both vertex invocations and blended coverage
      // (a translucent point came out twice as opaque as asked).
      drawCircles(nodeShader.gl, 1, skeletonGpuGeometry.numVertices);
    }
  }

  endLayer(gl: GL, ...shaders: Array<ShaderProgram | null>) {
    const { vertexAttributes, clearedTextureUnits } = this;
    const numAttributes = vertexAttributes.length;
    clearedTextureUnits.clear();
    const clearUnit = (unit: number) => {
      const curTextureUnit = unit + WebGL2RenderingContext.TEXTURE0;
      if (clearedTextureUnits.has(curTextureUnit)) return;
      clearedTextureUnits.add(curTextureUnit);
      gl.activeTexture(curTextureUnit);
      gl.bindTexture(gl.TEXTURE_2D, null);
    };
    for (const shader of shaders) {
      if (shader === null) continue;
      for (let i = 0; i < numAttributes; ++i) {
        if (this.isPackedAttribute(i)) continue;
        clearUnit(shader.textureUnit(vertexAttributeSamplerSymbols[i]));
      }
      if (this.base.packedAttributeRange !== undefined) {
        clearUnit(shader.textureUnit(packedAttributeSamplerSymbol));
      }
    }
    this.vertexIdHelper.disable();
  }

  /**
   * `linkedSegmentationColorGroup` swaps the colour group in place without
   * rebuilding this layer, so a table captured at construction goes stale and
   * the layer keeps rendering the previous group's colours. Re-resolve by
   * identity each draw, as `SegmentationRenderLayer` does.
   */
  private getSegmentStatedColorHashTable(): GPUHashTable<HashMapUint64> {
    const { hashTable } =
      this.base.displayState.segmentationColorGroupState.value
        .segmentStatedColors;
    let table = this.gpuSegmentStatedColorHashTable;
    if (table === undefined || table.hashTable !== hashTable) {
      table?.dispose();
      table = this.gpuSegmentStatedColorHashTable = GPUHashTable.get(
        this.gl,
        hashTable,
      );
    }
    return table;
  }

  disposed() {
    this.gpuSegmentStatedColorHashTable?.dispose();
    this.gpuSegmentStatedColorHashTable = undefined;
    super.disposed();
  }
}

// Draws the spatial bounds of each chunk as a box overlay, for debugging.
// One shader is compiled per emitter so the emitter can inject the correct
// output-buffer declarations and `emit(color, pickID)` function.
class ChunkWireframeHelper extends RefCounted {
  private shaderCache = new Map<ShaderModule, ShaderProgram>();

  constructor(private gl: GL) {
    super();
  }

  disposed() {
    for (const shader of this.shaderCache.values()) {
      shader.dispose();
    }
    this.shaderCache.clear();
    super.disposed();
  }

  getShader(emitter: ShaderModule): ShaderProgram {
    let shader = this.shaderCache.get(emitter);
    if (shader === undefined) {
      const builder = new ShaderBuilder(this.gl);
      builder.require(emitter);
      builder.addUniform("highp mat4", "uChunkToClip");
      builder.addUniform("highp vec3", "uTranslation");
      builder.addUniform("highp vec3", "uChunkDataSize");
      builder.addVertexCode(glsl_getBoxEdgeVertexPosition);
      builder.setVertexMain(`
vec3 boxVertex = getBoxEdgeVertexPosition(gl_VertexID);
gl_Position = uChunkToClip * vec4(uTranslation + boxVertex * uChunkDataSize, 1.0);
`);
      builder.setFragmentMain(`emit(vec4(1.0, 1.0, 1.0, 1.0), 0u);`);
      shader = builder.build();
      this.shaderCache.set(emitter, shader);
    }
    return shader;
  }

  setChunkUniforms(
    gl: GL,
    shader: ShaderProgram,
    chunkLayout: ChunkLayout,
    chunkGridPosition: Float32Array,
  ) {
    const { size } = chunkLayout;
    gl.uniform3f(
      shader.uniform("uTranslation"),
      chunkGridPosition[0] * size[0],
      chunkGridPosition[1] * size[1],
      chunkGridPosition[2] * size[2],
    );
    gl.uniform3fv(shader.uniform("uChunkDataSize"), size);
  }

  static get(gl: GL) {
    return gl.memoize.get(
      "skeleton/ChunkWireframeHelper",
      () => new ChunkWireframeHelper(gl),
    );
  }
}

/**
 * The user's lines-versus-points PREFERENCE for skeleton geometry.
 *
 * Deliberately only these two: every key of this enum becomes an option in the
 * "Skeleton mode" dropdown that every segmentation layer shows
 * (`EnumSelectWidget` enumerates the enum object), so a value that exists to
 * describe one datasource's geometry would appear as a choice on all of them.
 * What primitive a source's geometry IS -- points, lines, triangles -- is a
 * property of the source instead; see {@link GeometryPrimitive}.
 */
export enum SkeletonRenderMode {
  LINES = 0,
  LINES_AND_POINTS = 1,
}

/**
 * What GPU primitive a source's geometry is made of.
 *
 * Read structurally off the chunk source (see `SpatiallyIndexedSkeletonLayer`'s
 * constructor), so a datasource opts in by declaring a `geometryPrimitive`
 * getter and every other source keeps the `"lines"` default. This is not a user
 * setting: a point cloud has no edges to draw and a mesh has no line segments,
 * so neither is a mode anyone could sensibly switch away from.
 */
export type GeometryPrimitive = "points" | "lines" | "triangles";

export function setSpatialSkeletonModesToLinesAndPoints(layer: {
  displayState: { skeletonRenderingOptions: SkeletonRenderingOptions };
}) {
  layer.displayState.skeletonRenderingOptions.params2d.mode.value =
    SkeletonRenderMode.LINES_AND_POINTS;
  layer.displayState.skeletonRenderingOptions.params3d.mode.value =
    SkeletonRenderMode.LINES_AND_POINTS;
}

export class TrackableSkeletonRenderMode extends TrackableEnum<SkeletonRenderMode> {
  constructor(
    value: SkeletonRenderMode,
    defaultValue: SkeletonRenderMode = value,
  ) {
    super(SkeletonRenderMode, value, defaultValue);
  }
}

export class TrackableSkeletonLineWidth extends TrackableValue<number> {
  constructor(value: number, defaultValue: number = value) {
    super(value, verifyFinitePositiveFloat, defaultValue);
  }
}

function getSkeletonNodeDiameter(
  renderMode: SkeletonRenderMode,
  lineWidth: number,
) {
  if (renderMode === SkeletonRenderMode.LINES_AND_POINTS) {
    return Math.max(5, lineWidth * 2);
  }
  return lineWidth;
}

function setMouseStatePositionFromSpatialSkeletonNode(
  mouseState: MouseSelectionState,
  nodePosition: Float32Array,
  transform: RenderLayerTransform,
) {
  const rank = transform.rank;
  const modelPosition = new Float32Array(rank);
  for (let i = 0; i < Math.min(nodePosition.length, rank); ++i) {
    const v = nodePosition[i];
    if (!Number.isFinite(v)) return;
    modelPosition[i] = v;
  }
  const layerPosition = new Float32Array(rank);
  matrix.transformPoint(
    layerPosition,
    transform.modelToRenderLayerTransform,
    rank + 1,
    modelPosition,
    rank,
  );
  gatherUpdate(
    mouseState.position,
    layerPosition,
    transform.globalToRenderLayerDimensions,
  );
}

export interface ViewSpecificSkeletonRenderingOptions {
  mode: TrackableSkeletonRenderMode;
  lineWidth: TrackableSkeletonLineWidth;
}

// TODO (SKM): think this could likely extend compound trackable instead
export class SkeletonRenderingOptions implements Trackable {
  private compound = new CompoundTrackable();
  get changed() {
    return this.compound.changed;
  }

  shader = makeTrackableFragmentMain(DEFAULT_FRAGMENT_MAIN);
  shaderControlState = new ShaderControlState(this.shader);
  params2d: ViewSpecificSkeletonRenderingOptions = {
    mode: new TrackableSkeletonRenderMode(SkeletonRenderMode.LINES_AND_POINTS),
    lineWidth: new TrackableSkeletonLineWidth(2),
  };
  params3d: ViewSpecificSkeletonRenderingOptions = {
    mode: new TrackableSkeletonRenderMode(SkeletonRenderMode.LINES),
    lineWidth: new TrackableSkeletonLineWidth(1),
  };

  constructor() {
    const { compound } = this;
    compound.add("shader", this.shader);
    compound.add("shaderControls", this.shaderControlState);
    compound.add("mode2d", this.params2d.mode);
    compound.add("lineWidth2d", this.params2d.lineWidth);
    compound.add("mode3d", this.params3d.mode);
    compound.add("lineWidth3d", this.params3d.lineWidth);
  }

  reset() {
    this.compound.reset();
  }

  restoreState(obj: any) {
    if (obj === undefined) return;
    this.compound.restoreState(obj);
  }

  toJSON(): any {
    const obj = this.compound.toJSON();
    for (const v of Object.values(obj)) {
      if (v !== undefined) return obj;
    }
    return undefined;
  }
}

export interface SkeletonLayerDisplayState extends SegmentationDisplayState3D {
  shaderError: WatchableShaderError;
  skeletonRenderingOptions: SkeletonRenderingOptions;
  /**
   * ROI streamline-filter display state — present only for zarr-vectors tract
   * layers. When set, the dynamic-appearance shader gains a tier that ghosts
   * segments absent from `roiPassingSegments` while `roiFilterActive` is on.
   * Absent for every other skeleton layer, so their shaders are unchanged.
   */
  roiPassingSegments?: Uint64Set;
  roiFilterActive?: WatchableValueInterface<boolean>;
  roiGhostAlpha?: WatchableValueInterface<number>;
  /**
   * The ordered ROI groups, threaded to the backend (which recomputes
   * `roiPassingSegments` from them). Plain serialisable data — carried here only
   * to hand to the render layer's shared-object counterpart.
   */
  roiGroups?: WatchableValueInterface<readonly RoiGroupConfig[]>;
  /**
   * Per-object numeric attribute columns (length, …), threaded to the backend so
   * a group's length filter and object-attribute colouring can be evaluated.
   */
  roiObjectAttrColumns?: WatchableValueInterface<
    ReadonlyMap<string, RoiObjectAttrColumn>
  >;
  /**
   * Dense anatomical label grid from a linked parcellation layer, threaded to the
   * backend so `labelMask` ROIs can be sampled per streamline vertex.
   */
  roiLabelField?: WatchableValueInterface<RoiLabelField | undefined>;
  /**
   * Frontend-only per-object value map (id -> normalised attribute value) +
   * resolved uniforms for the background length filter / flat colour-by-attribute
   * shader tier. Read directly by the shader (not threaded to the backend).
   */
  roiObjectValues?: Uint64Map;
  roiBackground?: WatchableValueInterface<RoiBackgroundUniforms | undefined>;
  /**
   * Shared id -> packed group colour the backend fills for passing tracts. Read
   * directly by the ROI colour-by-group shader tier (never the user-facing
   * `segmentStatedColors`); its rpcId is also threaded to the backend.
   */
  roiSegmentColors?: Uint64Map;
  /** Whether to apply `roiSegmentColors` (the colour-by-group shader uniform). */
  roiColorByGroup?: WatchableValueInterface<boolean>;
  /**
   * Shared set the pass-1 backend fills = passing tracts of visible high-detail
   * groups. This layer (pass-1) only mutates it; the object-keyed pass-2 layer
   * reads it as its visible set. Threaded to the backend counterpart.
   */
  roiHighDetailSegments?: Uint64Set;
}

export class SkeletonLayer extends RefCounted implements SkeletonShaderContext {
  layerChunkProgressInfo = new LayerChunkProgressInfo();
  redrawNeeded = new NullarySignal();
  private sharedObject: SegmentationLayerSharedObject;
  vertexAttributes: VertexAttributeRenderInfo[];
  /** See {@link PackedAttributeRange}; set by sources that pack. */
  packedAttributeRange: PackedAttributeRange | undefined;
  segmentColorAttributeIndex: number | undefined = undefined;
  // Non-spatial skeletons iterate segments individually and pass color/alpha via
  // uniforms (getObjectColor), so the dynamic per-vertex segment appearance path
  // is not needed. Stated colors and default color are likewise handled upstream
  // before the draw call, not looked up in the shader.
  readonly skeletonShaderParameters =
    new WatchableValue<SkeletonShaderParameters>({
      dynamicSegmentAppearance: false,
      hasRoiFilter: false,
      hasRoiHighDetailHide: false,
      hasRoiSegmentColors: false,
      hasRoiObjectValues: false,
      hasSegmentStatedColors: false,
      hasSegmentDefaultColor: false,
      hoverHighlight: false,
      spatialChunkCulling: false,
    });
  fallbackShaderParameters = new WatchableValue(
    getFallbackBuilderState(parseShaderUiControls(DEFAULT_FRAGMENT_MAIN)),
  );

  get visibility() {
    return this.sharedObject.visibility;
  }

  constructor(
    public chunkManager: ChunkManager,
    public source: SkeletonSource,
    public displayState: SkeletonLayerDisplayState,
  ) {
    super();

    registerRedrawWhenSegmentationDisplayState3DChanged(displayState, this);
    this.displayState.shaderError.value = undefined;
    const { skeletonRenderingOptions: renderingOptions } = displayState;
    this.registerDisposer(
      renderingOptions.shader.changed.add(() => {
        this.displayState.shaderError.value = undefined;
        this.redrawNeeded.dispatch();
      }),
    );
    // ROI high-detail pass-2 layer: its per-object colour/opacity comes from the
    // roiSegmentColors map (CPU uColor) and the colour-by-group / active
    // uniforms, none of which the standard segmentation-state redraw watches.
    // Without this, dragging a group's opacity (or toggling colour-by-group /
    // the filter) does not re-render pass-2 until its visible set next changes
    // (e.g. an ROI move) — so the change appears "stuck". No-op for ordinary
    // (non-ROI) skeleton layers, which lack these fields.
    const requestRoiRedraw = () => this.redrawNeeded.dispatch();
    for (const w of [
      displayState.roiSegmentColors,
      displayState.roiColorByGroup,
      displayState.roiFilterActive,
    ]) {
      if (w !== undefined) {
        this.registerDisposer(w.changed.add(requestRoiRedraw));
      }
    }
    const sharedObject = (this.sharedObject = this.registerDisposer(
      new SegmentationLayerSharedObject(
        chunkManager,
        displayState,
        this.layerChunkProgressInfo,
      ),
    ));
    sharedObject.RPC_TYPE_ID = SKELETON_LAYER_RPC_ID;
    sharedObject.initializeCounterpartWithChunkManager({
      source: source.addCounterpartRef(),
    });

    // The ROI high-detail pass-2 layer receives roiSegmentColors in its
    // displayState (threaded from the tract layer). Compile the legacy shader's
    // colour-by-group branch for it so a group's colour/opacity apply to its
    // full-detail streamlines just as they do on the coarse pass-1 bulk. Plain
    // (non-ROI) skeletons never get this map, so their shader is unchanged.
    if (displayState.roiSegmentColors !== undefined) {
      this.skeletonShaderParameters.value = {
        ...this.skeletonShaderParameters.value,
        hasRoiSegmentColors: true,
      };
    }

    const vertexAttributes = (this.vertexAttributes = [
      vertexPositionAttribute,
    ]);

    for (const [name, info] of source.vertexAttributes) {
      vertexAttributes.push({
        name,
        dataType: info.dataType,
        numComponents: info.numComponents,
        webglDataType: getWebglDataType(info.dataType),
        glslDataType: getShaderType(info.dataType, info.numComponents),
      });
    }
    // Position is prepended above and nothing else is inserted, so a range the
    // source states in its own attribute order still lines up here.
    this.packedAttributeRange = (
      source as { packedAttributeRange?: PackedAttributeRange }
    ).packedAttributeRange;
  }

  get gl() {
    return this.chunkManager.chunkQueueManager.gl;
  }

  draw(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    renderHelper: RenderHelper,
    renderOptions: ViewSpecificSkeletonRenderingOptions,
    attachment: VisibleLayerInfo<
      LayerView,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const lineWidth = renderOptions.lineWidth.value;
    const { gl, displayState, source } = this;
    if (displayState.objectAlpha.value <= 0.0) {
      // Skip drawing.
      return;
    }
    if (
      displayState.roiHighDetailSegments !== undefined &&
      displayState.roiHighDetailSegments.size === 0
    ) {
      // Object-keyed pass-2 layer: nothing is claimed, so draw nothing and let
      // the coarse pass-1 bulk show those tracts instead. Keyed on the SET, not
      // on whether the ROI filter is on: the set has two independent drivers now
      // -- a dissection asking for full detail, and the object-focused fill
      // spending the pyramid's leftover memory on whole tracts -- and gating on
      // one of them would blank the other. It is also exact where the old check
      // was approximate: whoever empties the set turns this layer off in the
      // same frame, with no window where pass 1 hides tracts nothing draws.
      return;
    }
    const modelMatrix = update3dRenderLayerAttachment(
      displayState.transform.value,
      renderContext.projectionParameters.displayDimensionRenderInfo,
      attachment,
    );
    if (modelMatrix === undefined) return;
    const pointDiameter = getSkeletonNodeDiameter(
      renderOptions.mode.value,
      lineWidth,
    );

    const edgeShaderResult = renderHelper.edgeShaderGetter(
      renderContext.emitter,
    );
    const nodeShaderResult = renderHelper.nodeShaderGetter(
      renderContext.emitter,
    );
    const { shader: edgeShader, parameters: edgeShaderParameters } =
      edgeShaderResult;
    const { shader: nodeShader, parameters: nodeShaderParameters } =
      nodeShaderResult;
    if (edgeShader === null || nodeShader === null) {
      // Shader error, skip drawing.
      return;
    }

    const { shaderControlState } = this.displayState.skeletonRenderingOptions;

    edgeShader.bind();
    renderHelper.beginLayer(gl, edgeShader, renderContext, modelMatrix);
    renderHelper.setPickInstanceStride(gl, edgeShader, 0);
    setControlsInShader(
      gl,
      edgeShader,
      shaderControlState,
      edgeShaderParameters.parseResult.controls,
    );
    gl.uniform1f(edgeShader.uniform("uLineWidth"), lineWidth!);

    nodeShader.bind();
    renderHelper.beginLayer(gl, nodeShader, renderContext, modelMatrix);
    gl.uniform1f(nodeShader.uniform("uNodeDiameter"), pointDiameter);
    renderHelper.setPickInstanceStride(gl, nodeShader, 0);
    setControlsInShader(
      gl,
      nodeShader,
      shaderControlState,
      nodeShaderParameters.parseResult.controls,
    );

    // ROI high-detail pass-2: this streamline's group colour/opacity is packed
    // into roiSegmentColors (id -> RGBA). Opacity always rides uColor.a below;
    // uRoiColorByGroup selects uColor.rgb (the group colour) over the shader's
    // colour-by-direction, matching the coarse pass-1 bulk. Only meaningful
    // while the filter is active (pass-2 draws nothing otherwise).
    const roiColors = displayState.roiSegmentColors;
    const roiColorActive =
      roiColors !== undefined && displayState.roiFilterActive?.value === true;
    if (roiColors !== undefined) {
      const colorByGroup =
        roiColorActive && displayState.roiColorByGroup?.value === true ? 1 : 0;
      edgeShader.bind();
      gl.uniform1f(edgeShader.uniform("uRoiColorByGroup"), colorByGroup);
      nodeShader.bind();
      gl.uniform1f(nodeShader.uniform("uRoiColorByGroup"), colorByGroup);
    }

    const skeletons = source.chunks;

    forEachVisibleSegmentToDraw(
      displayState,
      layer,
      renderContext.emitColor,
      renderContext.emitPickID ? renderContext.pickIDs : undefined,
      (objectId, color, pickIndex) => {
        const key = getObjectKey(objectId);
        const skeleton = skeletons.get(key);
        if (
          skeleton === undefined ||
          skeleton.state !== ChunkState.GPU_MEMORY
        ) {
          return;
        }
        if (color !== undefined) {
          // Give the shader a STRAIGHT (non-premultiplied) rgba in uColor: the
          // group's colour/opacity when this tract is in the map, else the
          // object's own colour with its premultiply undone (a transient during
          // a filter recompute). getObjectColor returns premultiplied rgb, so if
          // that value reached uColor.rgb the colour-by-group branch would
          // premultiply a second time (base*alpha^2).
          let drawColor = color;
          if (roiColorActive) {
            const packed = roiColors!.get(objectId);
            if (packed !== undefined) {
              const p = Number(packed) >>> 0;
              tempRoiColor[0] = (p & 0xff) / 255;
              tempRoiColor[1] = ((p >>> 8) & 0xff) / 255;
              tempRoiColor[2] = ((p >>> 16) & 0xff) / 255;
              tempRoiColor[3] = ((p >>> 24) & 0xff) / 255;
            } else {
              const a = color[3] > 0 ? color[3] : 1;
              tempRoiColor[0] = color[0] / a;
              tempRoiColor[1] = color[1] / a;
              tempRoiColor[2] = color[2] / a;
              tempRoiColor[3] = color[3];
            }
            drawColor = tempRoiColor;
          }
          edgeShader.bind();
          renderHelper.setColor(gl, edgeShader, drawColor);
          nodeShader.bind();
          renderHelper.setColor(gl, nodeShader, drawColor);
        }
        if (pickIndex !== undefined) {
          edgeShader.bind();
          renderHelper.setPickID(gl, edgeShader, pickIndex);
          nodeShader.bind();
          renderHelper.setPickID(gl, nodeShader, pickIndex);
        }
        renderHelper.drawSkeletons(
          gl,
          edgeShader,
          nodeShader,
          skeleton,
          renderContext.projectionParameters,
          renderOptions.mode.value,
        );
      },
    );
    renderHelper.endLayer(gl, edgeShader, nodeShader);
  }

  isReady() {
    const { source, displayState } = this;
    if (displayState.objectAlpha.value <= 0.0) {
      // Skip drawing.
      return true;
    }

    const skeletons = source.chunks;

    let ready = true;

    forEachVisibleSegment(
      displayState.segmentationGroupState.value,
      (objectId) => {
        const key = getObjectKey(objectId);
        const skeleton = skeletons.get(key);
        if (
          skeleton === undefined ||
          skeleton.state !== ChunkState.GPU_MEMORY
        ) {
          ready = false;
          return;
        }
      },
    );
    return ready;
  }
}

export class PerspectiveViewSkeletonLayer extends PerspectiveViewRenderLayer {
  private renderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  constructor(public base: SkeletonLayer) {
    super();
    this.renderHelper = this.registerDisposer(new RenderHelper(base, false));
    this.renderOptions = base.displayState.skeletonRenderingOptions.params3d;

    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(base.visibility.add(this.visibility));
  }
  get gl() {
    return this.base.gl;
  }

  get isTransparent() {
    return this.base.displayState.objectAlpha.value < 1.0;
  }

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    if (!renderContext.emitColor && renderContext.alreadyEmittedPickID) {
      // No need for a separate pick ID pass.
      return;
    }
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.renderOptions,
      attachment,
    );
  }

  isReady() {
    return this.base.isReady();
  }
}

export class SliceViewPanelSkeletonLayer extends SliceViewPanelRenderLayer {
  private renderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  constructor(public base: SkeletonLayer) {
    super();
    this.renderHelper = this.registerDisposer(new RenderHelper(base, true));
    this.renderOptions = base.displayState.skeletonRenderingOptions.params2d;
    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    this.registerDisposer(base.visibility.add(this.visibility));
  }
  get gl() {
    return this.base.gl;
  }

  draw(
    renderContext: SliceViewPanelRenderContext,
    attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.renderOptions,
      attachment,
    );
  }

  isReady() {
    return this.base.isReady();
  }
}

// Per-vertex attributes reach the shader as TEXTURES (`uVertexAttributeSampler<i>`,
// typed from `info.dataType`), so this enum is not used to set up any attribute
// pointer -- but it is computed eagerly for every attribute in the `SkeletonLayer`
// constructor, so a dtype missing here fails the whole layer at load time rather
// than degrading one attribute. The 8/16-bit widths are all legal WebGL2 vertex
// types; they are listed so a store declaring e.g. a `uint8` compartment label
// loads. Mirrors `zvWebglDataType` in `datasource/zarr-vectors/geometry_frontend.ts`.
function getWebglDataType(dataType: DataType) {
  switch (dataType) {
    case DataType.FLOAT32:
      return WebGL2RenderingContext.FLOAT;
    case DataType.INT8:
      return WebGL2RenderingContext.BYTE;
    case DataType.UINT8:
      return WebGL2RenderingContext.UNSIGNED_BYTE;
    case DataType.INT16:
      return WebGL2RenderingContext.SHORT;
    case DataType.UINT16:
      return WebGL2RenderingContext.UNSIGNED_SHORT;
    case DataType.INT32:
      return WebGL2RenderingContext.INT;
    case DataType.UINT32:
      return WebGL2RenderingContext.UNSIGNED_INT;
    default:
      throw new Error(
        `Data type not supported by WebGL: ${DataType[dataType]}`,
      );
  }
}

const vertexPositionAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 3,
  name: "",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "vec3",
};

const segmentAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.UINT32,
  numComponents: 1,
  name: "segment",
  webglDataType: WebGL2RenderingContext.UNSIGNED_INT,
  glslDataType: getShaderType(DataType.UINT32, 1),
};

const selectedNodeAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 1,
  name: "selectedNodeAttr",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "float",
};

interface SkeletonChunkBase extends SkeletonGPUGeometry {
  vertexAttributes: Uint8Array;
  vertexAttributeOffsets: Uint32Array;
  indices: Uint32Array;
  source: {
    attributeTextureFormats: TextureFormat[];
    packedAttributeRange?: PackedAttributeRange;
  };
}

/** Byte range attribute `i` occupies in the chunk's packed attribute buffer. */
function attributeByteRange(
  chunk: SkeletonChunkBase,
  i: number,
): [number, number] {
  const { vertexAttributes, vertexAttributeOffsets } = chunk;
  return [
    vertexAttributeOffsets[i],
    i + 1 !== vertexAttributeOffsets.length
      ? vertexAttributeOffsets[i + 1]
      : vertexAttributes.length,
  ];
}

// Used by both SkeletonChunk and SpatiallyIndexedSkeletonChunk.
function uploadSkeletonChunkToGPU(gl: GL, chunk: SkeletonChunkBase) {
  const { attributeTextureFormats, packedAttributeRange } = chunk.source;
  const { vertexAttributes, vertexAttributeOffsets } = chunk;
  const vertexAttributeTextures: (WebGLTexture | null)[] =
    (chunk.vertexAttributeTextures = []);
  const packedEnd =
    packedAttributeRange === undefined
      ? -1
      : packedAttributeRange.start + packedAttributeRange.count;
  for (
    let i = 0, numAttributes = vertexAttributeOffsets.length;
    i < numAttributes;
    ++i
  ) {
    if (
      packedAttributeRange !== undefined &&
      i >= packedAttributeRange.start &&
      i < packedEnd
    ) {
      // Uploaded once for the whole run, below.
      vertexAttributeTextures[i] = null;
      continue;
    }
    const texture = gl.createTexture();
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
    const [begin, end] = attributeByteRange(chunk, i);
    setOneDimensionalTextureData(
      gl,
      attributeTextureFormats[i],
      vertexAttributes.subarray(begin, end),
    );
    vertexAttributeTextures[i] = texture;
  }
  if (packedAttributeRange !== undefined && packedAttributeRange.count > 0) {
    // The run is already contiguous in the serialized chunk -- the backend
    // packs attributes back to back in declaration order -- so one texture over
    // [first attribute, last attribute] needs no repacking on this side. Every
    // entry is 1-component float32, so element `a * numVertices + v` addresses
    // attribute `a` of vertex `v`.
    const texture = gl.createTexture();
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
    const begin = attributeByteRange(chunk, packedAttributeRange.start)[0];
    const end = attributeByteRange(chunk, packedEnd - 1)[1];
    setOneDimensionalTextureData(
      gl,
      attributeTextureFormats[packedAttributeRange.start],
      vertexAttributes.subarray(begin, end),
    );
    chunk.packedAttributeTexture = texture;
  }
  gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
  chunk.indexBuffer = GLBuffer.fromData(
    gl,
    chunk.indices,
    WebGL2RenderingContext.ARRAY_BUFFER,
    WebGL2RenderingContext.STATIC_DRAW,
  );
}

function freeSkeletonChunkGPUMemory(gl: GL, chunk: SkeletonChunkBase) {
  chunk.indexBuffer.dispose();
  const { vertexAttributeTextures } = chunk;
  for (let i = 0, length = vertexAttributeTextures.length; i < length; ++i) {
    gl.deleteTexture(vertexAttributeTextures[i]);
  }
  vertexAttributeTextures.length = 0;
  if (chunk.packedAttributeTexture != null) {
    gl.deleteTexture(chunk.packedAttributeTexture);
    chunk.packedAttributeTexture = null;
  }
}

export class SkeletonChunk extends Chunk implements SkeletonChunkBase {
  declare source: SkeletonSource;
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  indexBuffer!: GLBuffer;
  numIndices: number;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  vertexAttributeTextures: (WebGLTexture | null)[] = [];

  constructor(source: SkeletonSource, x: PackedSkeletonGeometry) {
    super(source);
    this.vertexAttributes = x.vertexAttributes;
    const indices = (this.indices = x.indices);
    this.numVertices = x.numVertices;
    this.vertexAttributeOffsets = x.vertexAttributeOffsets;
    this.numIndices = indices.length;
  }

  copyToGPU(gl: GL) {
    super.copyToGPU(gl);
    uploadSkeletonChunkToGPU(gl, this);
  }

  freeGPUMemory(gl: GL) {
    super.freeGPUMemory(gl);
    freeSkeletonChunkGPUMemory(gl, this);
  }
}

export class SpatiallyIndexedSkeletonChunk
  extends SliceViewChunk
  implements SkeletonChunkBase
{
  declare source: SpatiallyIndexedSkeletonSource;
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  indexBuffer!: GLBuffer;
  numIndices: number;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  vertexAttributeTextures: (WebGLTexture | null)[] = [];
  nodeIds: Int32Array;
  nodeSourceStates: Array<SpatialSkeletonSourceState | undefined> = [];

  constructor(
    source: SpatiallyIndexedSkeletonSource,
    chunkData: PackedSkeletonGeometry,
  ) {
    super(source, chunkData);
    this.vertexAttributes = chunkData.vertexAttributes;
    const indices = (this.indices = chunkData.indices);
    this.numVertices = chunkData.numVertices;
    this.numIndices = indices.length;
    this.vertexAttributeOffsets = chunkData.vertexAttributeOffsets;
    this.nodeIds = chunkData.nodeIds ?? new Int32Array(0);
    const nodeSourceStates = chunkData.nodeSourceStates;
    this.nodeSourceStates = Array.isArray(nodeSourceStates)
      ? nodeSourceStates
      : [];
  }

  copyToGPU(gl: GL) {
    super.copyToGPU(gl);
    uploadSkeletonChunkToGPU(gl, this);
  }

  freeGPUMemory(gl: GL) {
    super.freeGPUMemory(gl);
    freeSkeletonChunkGPUMemory(gl, this);
  }
}

export interface SpatiallyIndexedSkeletonChunkSpecification
  extends SliceViewChunkSpecification {
  chunkLayout: ChunkLayout;
}

type SpatiallyIndexedSkeletonChunkListener = (
  key: string,
  chunk: SpatiallyIndexedSkeletonChunk,
) => void;

const spatiallyIndexedSkeletonTextureAttributeSpecs = Object.freeze([
  { name: "position", dataType: DataType.FLOAT32, numComponents: 3 },
  { name: "segment", dataType: DataType.UINT32, numComponents: 1 },
]);

export class SpatiallyIndexedSkeletonSource extends SliceViewChunkSource<
  SpatiallyIndexedSkeletonChunkSpecification,
  SpatiallyIndexedSkeletonChunk
> {
  vertexAttributes: VertexAttributeRenderInfo[];
  private attributeTextureFormats_?: TextureFormat[];
  private chunkListeners = new Set<SpatiallyIndexedSkeletonChunkListener>();

  constructor(chunkManager: ChunkManager, options: any) {
    super(chunkManager, options);
    this.vertexAttributes = [vertexPositionAttribute, segmentAttribute];
  }

  get attributeTextureFormats() {
    let attributeTextureFormats = this.attributeTextureFormats_;
    if (attributeTextureFormats === undefined) {
      attributeTextureFormats = this.attributeTextureFormats_ =
        spatiallyIndexedSkeletonTextureAttributeSpecs.map(
          ({ dataType, numComponents }) =>
            computeTextureFormat(new TextureFormat(), dataType, numComponents),
        );
    }
    return attributeTextureFormats;
  }

  static encodeSpec(spec: SpatiallyIndexedSkeletonChunkSpecification) {
    const base = SliceViewChunkSource.encodeSpec(spec);
    return { ...base, chunkLayout: spec.chunkLayout.toObject() };
  }

  addChunkListener(listener: SpatiallyIndexedSkeletonChunkListener) {
    this.chunkListeners.add(listener);
    return () => this.chunkListeners.delete(listener);
  }

  addChunk(key: string, chunk: SpatiallyIndexedSkeletonChunk) {
    super.addChunk(key, chunk);
    for (const listener of this.chunkListeners) {
      listener(key, chunk);
    }
  }

  getChunk(chunkData: PackedSkeletonGeometry) {
    return new SpatiallyIndexedSkeletonChunk(this, chunkData);
  }
}

// Options are provided by the SliceView framework for scale selection,
// but spatial skeleton sources expose all grid levels unconditionally.
// TODO (SKM): validate if this is an ok deviation from the SliceView
export const SPATIAL_SKELETON_SOURCE_OPTIONS: SliceViewSourceOptions = {
  displayRank: 0,
  multiscaleToViewTransform: new Float32Array(0),
  modelChannelDimensionIndices: [],
};

export function getSpatialSkeletonCellKeyPrefix(
  position: ArrayLike<number>,
  chunkDataSize: ArrayLike<number>,
) {
  const cell = new Array<number>(3);
  for (let i = 0; i < 3; ++i) {
    const coordinate = Number(position[i]);
    const chunkSize = Number(chunkDataSize[i]);
    if (
      !Number.isFinite(coordinate) ||
      !Number.isFinite(chunkSize) ||
      chunkSize <= 0
    ) {
      return undefined;
    }
    cell[i] = Math.floor(coordinate / chunkSize);
  }
  // A chunk key is the grid position plus the terminator, and nothing else
  // (`getChunk` in `skeleton/backend.ts`), so the whole key IS the prefix. The
  // terminator is what keeps the match exact -- see its docstring.
  return `${cell[0]},${cell[1]},${cell[2]}${SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR}`;
}

export abstract class MultiscaleSpatiallyIndexedSkeletonSource extends MultiscaleSliceViewChunkSource<SpatiallyIndexedSkeletonSource> {
  /**
   * When `true`, the segmentation layer enables
   * `autoSpatialSkeletonGridLevel{3d,2d}` on attach: the render layer
   * will overwrite `spatialSkeletonGridResolutionTarget*` every frame
   * from the camera projection.  Default `false` preserves the
   * existing manual-slider UX for sources that haven't opted in
   * (CATMAID).  Subclasses (e.g. zarr-vectors) override to `true`.
   */
  get prefersAutoSpatialSkeletonGridLevel(): boolean {
    return false;
  }

  /**
   * Fragment shader this source's geometry is best drawn with, or `undefined`
   * to keep the built-in `emitDefault()`.
   *
   * The layer applies it as the *default* rather than as a value, so a user's
   * explicit `skeletonRendering.shader` still wins and the state still
   * round-trips — see the consumer in `layer/segmentation/index.ts`. Existing
   * sources (CATMAID) do not override this and are unaffected.
   */
  get defaultFragmentMain(): string | undefined {
    return undefined;
  }

  getPerspectiveSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    const sources = this.getSources(SPATIAL_SKELETON_SOURCE_OPTIONS);
    const flattened: SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] =
      [];
    for (const scale of sources) {
      if (scale.length > 0) {
        flattened.push(scale[0]);
      }
    }
    return flattened;
  }

  getSliceViewPanelSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    return this.getPerspectiveSources();
  }

  /**
   * @param liveScale Optional current effective meters-per-model-axis
   *     scale (see `computeDiagonalModelToGlobalMetersScale`), composing
   *     the datasource's own declared native-unit scale with any output
   *     `CoordinateSpaceTransform` the user has applied.  `undefined`
   *     when the caller couldn't reduce the live transform to a
   *     diagonal per-axis scale (rotation/shear) or it isn't available
   *     yet — implementations should fall back to a static scale in
   *     that case.  Subclasses that don't need transform-awareness
   *     (e.g. CATMAID, which never opts into
   *     `prefersAutoSpatialSkeletonGridLevel`) may ignore this.
   */
  getSpatialSkeletonGridSizes(
    liveScale?: Float64Array,
  ): { x: number; y: number; z: number }[] | undefined {
    liveScale;
    return undefined;
  }
}

type SpatiallyIndexedSkeletonSourceEntry =
  SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>;

// TODO (SKM): is all of this really optional?
interface SpatiallyIndexedSkeletonLayerOptions {
  gridLevel?: WatchableValueInterface<number>;
  lod?: WatchableValueInterface<number>;
  gridLevel2d?: WatchableValueInterface<number>;
  lod2d?: WatchableValueInterface<number>;
  /**
   * Which detail-focus mode the layer draws in; see
   * {@link SpatialSkeletonDetailFocus}. Decides whether each visible cell picks
   * its own level or every cell is pinned to the selected one.
   */
  detailFocus?: WatchableValueInterface<number>;
  /**
   * Share of the drawn level's new objects to decode, in [0, 1]; `1` disables
   * per-object admission. See `object_admission.ts`.
   */
  admissionFraction?: WatchableValueInterface<number>;
  sources2d?: SpatiallyIndexedSkeletonSourceEntry[];
  selectedNodeId?: WatchableValueInterface<number | undefined>;
  pendingNodePositionVersion?: WatchableValueInterface<number>;
  getPendingNodePosition?: (nodeId: number) => ArrayLike<number> | undefined;
  getCachedNode?: (nodeId: number) => SpatiallyIndexedSkeletonNode | undefined;
  inspectionState?: SpatiallyIndexedSkeletonInspectionState;
  maxRetainedOverlaySegments?: number;
}

interface SpatiallyIndexedSkeletonInspectionState {
  readonly nodeDataVersion: WatchableValueInterface<number>;
  readonly pendingNodePositionVersion: WatchableValueInterface<number>;
  getCachedSegmentNodes(
    segmentId: number,
  ): readonly SpatiallyIndexedSkeletonNode[] | undefined;
  getFullSegmentNodes(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    segmentId: number,
  ): Promise<readonly SpatiallyIndexedSkeletonNode[]>;
  evictInactiveSegmentNodes(activeSegmentIds: Iterable<number>): void;
}

class SkeletonOverlayChunk implements SkeletonGPUGeometry {
  readonly vertexAttributeTextures: (WebGLTexture | null)[];
  readonly indexBuffer: GLBuffer;
  readonly numIndices: number;
  readonly numVertices: number;
  readonly pickNodeIds: Int32Array;
  readonly pickNodePositions: Float32Array;
  readonly pickSegmentIds: Uint32Array;
  readonly pickEdgeSegmentIds: Uint32Array;
  private readonly nodeIdToVertexIndex: Map<number, number>;
  private readonly selectedFormat: TextureFormat;

  constructor(
    gl: GL,
    geometry: SpatiallyIndexedSkeletonOverlayGeometry,
    formats: TextureFormat[],
  ) {
    const attributeBuffers = [
      new Uint8Array(
        geometry.positions.buffer,
        geometry.positions.byteOffset,
        geometry.positions.byteLength,
      ),
      new Uint8Array(
        geometry.segmentIds.buffer,
        geometry.segmentIds.byteOffset,
        geometry.segmentIds.byteLength,
      ),
      new Uint8Array(
        geometry.selected.buffer,
        geometry.selected.byteOffset,
        geometry.selected.byteLength,
      ),
    ];
    const overlayTextures: (WebGLTexture | null)[] =
      (this.vertexAttributeTextures = []);
    for (let i = 0; i < attributeBuffers.length; i++) {
      const texture = gl.createTexture();
      gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
      setOneDimensionalTextureData(gl, formats[i], attributeBuffers[i]);
      overlayTextures[i] = texture;
    }
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
    this.indexBuffer = GLBuffer.fromData(
      gl,
      geometry.indices,
      WebGL2RenderingContext.ARRAY_BUFFER,
      WebGL2RenderingContext.STATIC_DRAW,
    );
    this.numIndices = geometry.indices.length;
    this.numVertices = geometry.numVertices;
    this.pickNodeIds = geometry.nodeIds;
    // positions and nodePositions were identical — reuse positions for picking.
    this.pickNodePositions = geometry.positions;
    this.pickSegmentIds = geometry.pickSegmentIds;
    this.pickEdgeSegmentIds = geometry.pickEdgeSegmentIds;
    const nodeIdToVertexIndex = new Map<number, number>();
    const { nodeIds } = geometry;
    for (let i = 0; i < nodeIds.length; i++) {
      const nodeId = nodeIds[i];
      if (nodeId > 0) nodeIdToVertexIndex.set(nodeId, i);
    }
    this.nodeIdToVertexIndex = nodeIdToVertexIndex;
    this.selectedFormat = formats[2];
  }

  // Updates the selected-node highlight in-place without a full GPU rebuild.
  // Clears oldNodeId's texel and sets newNodeId's texel.
  updateSelectedNode(
    gl: GL,
    oldNodeId: number | undefined,
    newNodeId: number | undefined,
  ) {
    if (oldNodeId === newNodeId) return;
    const texture = this.vertexAttributeTextures[2];
    if (texture === null) return;
    if (oldNodeId !== undefined) {
      const idx = this.nodeIdToVertexIndex.get(oldNodeId);
      if (idx !== undefined) {
        updateOneDimensionalTextureElement(
          gl,
          texture,
          this.selectedFormat,
          this.numVertices,
          idx,
          OVERLAY_SELECTED_FLOAT_ZERO,
        );
      }
    }
    if (newNodeId !== undefined) {
      const idx = this.nodeIdToVertexIndex.get(newNodeId);
      if (idx !== undefined) {
        updateOneDimensionalTextureElement(
          gl,
          texture,
          this.selectedFormat,
          this.numVertices,
          idx,
          OVERLAY_SELECTED_FLOAT_ONE,
        );
      }
    }
  }

  dispose(gl: GL) {
    for (const texture of this.vertexAttributeTextures) {
      if (texture) gl.deleteTexture(texture);
    }
    this.indexBuffer.dispose();
  }
}

/**
 * Attempts to compute, for each of a multiscale skeleton source's first 3
 * "model" (native/chunk-grid) dimensions, the CURRENT effective scale to
 * real-world meters — composing the datasource's own `modelToRenderLayer
 * Transform` (which reflects any output `CoordinateSpaceTransform` the user
 * has applied, e.g. correcting a source's declared voxel size from mm to
 * µm) with the corresponding global dimension's declared meters-per-unit
 * (`globalScales`, i.e. `coordinateSpace.scales`).
 *
 * Diagonal-only: returns `undefined` (caller should fall back to a static,
 * transform-oblivious scale) when a model dimension is unmapped, when a
 * render-layer dimension mixes contributions from more than one model
 * dimension by more than a small tolerance (rotation/shear — this function
 * does not attempt to reduce a non-axis-aligned transform to a per-axis
 * spacing), or when the corresponding global scale isn't a finite positive
 * number.
 *
 * This exists because `getSpatialSkeletonGridSizes()` previously reported
 * level sizes using only the datasource's own frozen, load-time native-unit
 * scale, which silently went stale relative to the camera-driven resolution
 * target (itself always computed from the live transform) whenever the
 * user edited the layer's output coordinate transform — producing a
 * systematic mismatch proportional to the rescale factor.
 */
export function computeDiagonalModelToGlobalMetersScale(
  transform: RenderLayerTransform,
  globalScales: Float64Array,
): Float64Array | undefined {
  const {
    rank: layerRank,
    localToRenderLayerDimensions,
    globalToRenderLayerDimensions,
    modelToRenderLayerTransform,
  } = transform;
  const result = new Float64Array(3);
  for (let modelDim = 0; modelDim < 3; ++modelDim) {
    const layerDim = localToRenderLayerDimensions[modelDim];
    if (layerDim === -1) return undefined;
    const diagValue =
      modelToRenderLayerTransform[layerDim + (layerRank + 1) * modelDim];
    // Reject if any OTHER model dimension also contributes non-negligibly
    // to this render-layer dimension (rotation/shear) — diagonal-only.
    for (let otherModelDim = 0; otherModelDim < layerRank; ++otherModelDim) {
      if (otherModelDim === modelDim) continue;
      const v =
        modelToRenderLayerTransform[layerDim + (layerRank + 1) * otherModelDim];
      if (Math.abs(v) > Math.abs(diagValue) * 1e-3 + 1e-12) {
        return undefined;
      }
    }
    let globalDim = -1;
    for (let g = 0; g < globalToRenderLayerDimensions.length; ++g) {
      if (globalToRenderLayerDimensions[g] === layerDim) {
        globalDim = g;
        break;
      }
    }
    if (globalDim === -1 || globalDim >= globalScales.length) {
      return undefined;
    }
    const metersPerGlobalUnit = globalScales[globalDim];
    if (!Number.isFinite(metersPerGlobalUnit) || metersPerGlobalUnit <= 0) {
      return undefined;
    }
    result[modelDim] = Math.abs(diagValue) * metersPerGlobalUnit;
  }
  return result;
}

/**
 * Physical units (meters) per screen pixel at `worldPoint`, derived from a
 * model-view-projection matrix and viewport dimensions with axis-aware
 * conversion from world units to meters.
 *
 * This is the same MVP-based formulation used by multiscale mesh LOD, but
 * adjusted for anisotropic coordinate systems (e.g. 4,4,40): each axis scale
 * is converted to pixels-per-meter before taking the max scale factor.
 *
 * Returns `+Infinity` for a behind-camera or invalid `w` so callers fall back
 * to the largest available level.
 */
function computePhysicalUnitsPerScreenPixel(
  modelViewProjection: mat4,
  viewportWidth: number,
  viewportHeight: number,
  worldPoint: Float32Array,
  displayDimensionScales?: Float64Array,
): number {
  const m = modelViewProjection;
  // Column-major mat4 indices.
  const m00 = m[0],
    m10 = m[1];
  const m01 = m[4],
    m11 = m[5];
  const m02 = m[8],
    m12 = m[9];
  const m30 = m[3],
    m31 = m[7],
    m32 = m[11],
    m33 = m[15];
  const w =
    m30 * worldPoint[0] + m31 * worldPoint[1] + m32 * worldPoint[2] + m33;
  if (!Number.isFinite(w) || w <= 0) return Number.POSITIVE_INFINITY;

  const sx =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 0 &&
    Number.isFinite(displayDimensionScales[0]) &&
    displayDimensionScales[0] > 0
      ? displayDimensionScales[0]
      : 1;
  const sy =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 1 &&
    Number.isFinite(displayDimensionScales[1]) &&
    displayDimensionScales[1] > 0
      ? displayDimensionScales[1]
      : sx;
  const sz =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 2 &&
    Number.isFinite(displayDimensionScales[2]) &&
    displayDimensionScales[2] > 0
      ? displayDimensionScales[2]
      : sy;

  const xScale = Math.sqrt(
    ((m00 / sx) * viewportWidth) ** 2 + ((m10 / sx) * viewportHeight) ** 2,
  );
  const yScale = Math.sqrt(
    ((m01 / sy) * viewportWidth) ** 2 + ((m11 / sy) * viewportHeight) ** 2,
  );
  const zScale = Math.sqrt(
    ((m02 / sz) * viewportWidth) ** 2 + ((m12 / sz) * viewportHeight) ** 2,
  );
  const scaleFactor = Math.max(xScale, yScale, zScale);
  if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  return w / scaleFactor;
}

/**
 * Multiplier from "world units per screen pixel" to the resolution
 * target the level picker matches against `levels[k].size`.  The picker
 * finds the level whose spacing is closest to the target, so target
 * needs to live in the same magnitude as a level spacing (typical
 * neuroscience streamline chunk = 50–500 µm/mm, while raw world-units-
 * per-pixel is a couple of orders of magnitude smaller for a normal
 * zoom).  200 means "a chunk should be ~200 screen pixels at the
 * matching level" — finest level when chunks are bigger than that on
 * screen, coarser otherwise.  Mirrors the mesh layer's `detailCutoff`
 * tunable in [src/mesh/multiscale.ts:202](src/mesh/multiscale.ts#L202).
 */
const AUTO_SPATIAL_SKELETON_GRID_DETAIL_CUTOFF = 200;
// Limit auto-target movement per update to avoid jumping across multiple
// pyramid levels in a single zoom step (especially in anisotropic spaces).
const AUTO_SPATIAL_SKELETON_GRID_MAX_OCTAVE_STEP = 0.5;

/**
 * If `displayState.autoSpatialSkeletonGridLevel{view}` is enabled,
 * compute the camera-derived "world units per screen pixel" at a
 * representative world position, multiply by the detail-cutoff
 * constant, and write the result to
 * `spatialSkeletonGridResolutionTarget{view}`.  Because
 * `findClosestSpatialSkeletonGridLevelBySpacing` re-runs on every
 * change of that target, the picker (and the histogram widget) snap
 * to the appropriate discrete level for the current zoom.
 *
 * No-op when:
 * - The flag is absent or false (default; CATMAID/legacy behaviour).
 * - The target watchable isn't wired up.
 * - The computed target isn't finite.
 *
 * Reference point: prefer `projectionParameters.globalPosition` (the
 * camera's look-at point in global space).  Fall back to
 * `localPosition` (per-layer position, usually empty for segmentation
 * layers without layer-local dimensions) and finally to the origin.
 * In orthographic projection `w` is independent of position, so the
 * choice only matters for perspective views; either way the helper
 * returns a meaningful value for any of these fallbacks.
 */
/**
 * The grid spacing this layer's memory budget sustains across `visibleCellCount`
 * cells, or `undefined` when the layer publishes no per-cell costs.
 */
function memoryAwareSpacingTarget(
  displayState: SpatiallyIndexedSkeletonLayerDisplayState,
  visibleCellCount: number,
): number | undefined {
  const perCellCost = displayState.spatialSkeletonPerCellCostBytes?.value;
  const budgetBytes = displayState.spatialSkeletonGpuBudgetBytes?.value;
  const levels = displayState.spatialSkeletonGridLevels?.value;
  if (
    perCellCost === undefined ||
    perCellCost.length === 0 ||
    budgetBytes === undefined ||
    levels === undefined
  ) {
    return undefined;
  }
  return targetSpacingForCellBudget(
    levels.map(({ size }) => Math.min(size.x, size.y, size.z)),
    perCellCost,
    visibleCellCount,
    budgetBytes,
  );
}

export function maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
  displayState: SpatiallyIndexedSkeletonLayerDisplayState,
  projectionParameters: {
    viewProjectionMat: mat4;
    width: number;
    height: number;
    globalPosition?: Float32Array;
    displayDimensionRenderInfo?: { displayDimensionScales: Float64Array };
  },
  localPosition: Float32Array,
  view: "2d" | "3d",
  /**
   * Cells the view covers, for the memory-aware target. Omit to keep the
   * pixel-derived target (every non-tract spatially-indexed skeleton layer).
   */
  visibleCellCount?: number,
): void {
  const autoFlag =
    view === "3d"
      ? displayState.autoSpatialSkeletonGridLevel3d
      : displayState.autoSpatialSkeletonGridLevel2d;
  if (autoFlag === undefined || autoFlag.value !== true) return;
  const target =
    view === "3d"
      ? displayState.spatialSkeletonGridResolutionTarget3d
      : displayState.spatialSkeletonGridResolutionTarget2d;
  if (target === undefined) return;
  // Pick the first non-empty reference position.  GlobalPosition is
  // populated for all 3D-view renders even when the segmentation
  // layer has no layer-local dimensions (the common case, where
  // `localPosition` is a zero-length array).
  let reference: Float32Array | undefined;
  if (
    projectionParameters.globalPosition !== undefined &&
    projectionParameters.globalPosition.length >= 3
  ) {
    reference = projectionParameters.globalPosition;
  } else if (localPosition.length >= 3) {
    reference = localPosition;
  } else {
    reference = new Float32Array(3); // origin fallback
  }
  const pixelSize = computePhysicalUnitsPerScreenPixel(
    projectionParameters.viewProjectionMat,
    projectionParameters.width,
    projectionParameters.height,
    reference,
    projectionParameters.displayDimensionRenderInfo?.displayDimensionScales,
  );
  if (!Number.isFinite(pixelSize) || pixelSize <= 0) return;
  // Camera-only target (no user bias yet).
  //
  // Prefer a MEMORY-derived target where the layer can supply one. The
  // pixel-derived figure below is the right answer for a pyramid whose levels
  // differ in resolution, but for one whose levels differ in how many whole
  // objects they hold it saturates: `pixelSize * CUTOFF` is already at or below
  // the finest level's spacing at a whole-volume view and only falls further as
  // you zoom in, so the level it selects never changes over the entire useful
  // range. What does vary usefully is how many cells the view covers — see
  // `targetSpacingForCellBudget`.
  const memoryTarget =
    visibleCellCount === undefined
      ? undefined
      : memoryAwareSpacingTarget(displayState, visibleCellCount);
  const autoUnbiased =
    memoryTarget ?? pixelSize * AUTO_SPATIAL_SKELETON_GRID_DETAIL_CUTOFF;

  // The multiplicative detail `bias` (set when the user clicks/drags the
  // widget) lives in the *persistent* displayState so the calibration
  // survives a page refresh; the `lastAuto` value we wrote last frame is
  // transient and stays in a WeakMap keyed by the target.  A manual
  // interaction changes `target.value` away from `lastAuto`; we interpret
  // that as "make this zoom map to the level I picked" and fold it into
  // `bias`, so zoom keeps driving the level afterwards (the click rebiases
  // the offset, it does NOT freeze the target).
  const biasWatchable =
    view === "3d"
      ? displayState.spatialSkeletonGridResolutionBias3d
      : displayState.spatialSkeletonGridResolutionBias2d;
  let st = autoSpatialSkeletonBias.get(target);
  if (st === undefined) {
    st = { lastAuto: undefined };
    autoSpatialSkeletonBias.set(target, st);
  }
  let bias =
    biasWatchable !== undefined &&
    Number.isFinite(biasWatchable.value) &&
    biasWatchable.value > 0
      ? biasWatchable.value
      : 1;
  const cur = target.value as number;
  if (
    st.lastAuto !== undefined &&
    Math.abs(cur - st.lastAuto) > Math.max(st.lastAuto, 1e-30) * 1e-6 &&
    autoUnbiased > 0 &&
    Number.isFinite(cur) &&
    cur > 0
  ) {
    bias = cur / autoUnbiased;
    if (biasWatchable !== undefined && biasWatchable.value !== bias) {
      biasWatchable.value = bias;
    }
  }
  const next = autoUnbiased * bias;
  let stabilizedNext = next;
  if (
    st.lastAuto !== undefined &&
    Number.isFinite(st.lastAuto) &&
    st.lastAuto > 0 &&
    Number.isFinite(stabilizedNext) &&
    stabilizedNext > 0
  ) {
    const maxFactor = 2 ** AUTO_SPATIAL_SKELETON_GRID_MAX_OCTAVE_STEP;
    const upper = st.lastAuto * maxFactor;
    const lower = st.lastAuto / maxFactor;
    if (stabilizedNext > upper) stabilizedNext = upper;
    if (stabilizedNext < lower) stabilizedNext = lower;
  }
  // Only write when the value changes by more than 0.1% — the setter
  // dispatches `changed` unconditionally (level pick → re-attach chain).
  // Leave `lastAuto` untouched on this branch: it must only track the
  // last value actually WRITTEN to `target.value`.  Advancing it here
  // (as this used to do) lets it drift away from `cur` — which stays at
  // its last-written value while we skip the write — by up to just
  // under 0.1% per skipped frame.  That's comfortably past the 1e-6
  // "did the user manually move the widget" threshold above, so nearly
  // every skipped frame got misread as manual interaction on the very
  // next frame, corrupting the persisted `bias` continuously.
  if (
    st.lastAuto !== undefined &&
    Math.abs(cur - stabilizedNext) < Math.max(cur, stabilizedNext) * 1e-3
  ) {
    return;
  }
  target.value = stabilizedNext;
  st.lastAuto = stabilizedNext;
}

// Transient per-resolution-target auto-LOD state: the last auto-written
// target value, used to detect manual widget interaction frame-to-frame.
// The persistent detail `bias` lives in displayState (see
// `spatialSkeletonGridResolutionBias{2d,3d}`), not here.  Keyed weakly by
// the target.
const autoSpatialSkeletonBias = new WeakMap<
  WatchableValueInterface<number>,
  { lastAuto: number | undefined }
>();

export interface SpatiallyIndexedSkeletonLayerDisplayState
  extends SkeletonLayerDisplayState {
  spatialSkeletonGridLevel2d?: WatchableValueInterface<number>;
  spatialSkeletonGridLevel3d?: WatchableValueInterface<number>;
  /**
   * Bytes ONE cell of each pyramid level costs on the GPU, coarsest-first.
   * Drives the memory-aware resolution target; absent for sources that publish
   * no per-level costs, which keeps the pixel-derived target.
   */
  spatialSkeletonPerCellCostBytes?: WatchableValueInterface<number[]>;
  /**
   * The GPU byte budget the layer is sizing against. A watchable rather than a
   * number because render layers hold a spread copy of the display state, in
   * which a plain field would freeze at its value on activation.
   */
  spatialSkeletonGpuBudgetBytes?: WatchableValueInterface<number>;
  skeletonLod?: WatchableValueInterface<number>;
  spatialSkeletonLod2d?: WatchableValueInterface<number>;
  spatialSkeletonGridLevels?: WatchableValueInterface<
    Array<{ size: { x: number; y: number; z: number }; lod: number }>
  >;
  spatialSkeletonGridRenderScaleHistogram2d?: RenderScaleHistogram;
  spatialSkeletonGridRenderScaleHistogram3d?: RenderScaleHistogram;
  /**
   * Optional writable target that the picker is matched against;
   * paired with the `auto*` flags below.  When auto is enabled the
   * render layer overwrites this every frame from the camera
   * projection.
   */
  spatialSkeletonGridResolutionTarget2d?: WatchableValueInterface<number> & {
    value: number;
  };
  spatialSkeletonGridResolutionTarget3d?: WatchableValueInterface<number> & {
    value: number;
  };
  /**
   * Persistent multiplicative detail bias applied on top of the
   * camera-derived auto target (1 = pure camera).  Clicking/dragging the
   * render-scale widget folds the manual offset into this value so the
   * calibration survives a page refresh (the camera-derived target itself
   * is recomputed every frame and is not meaningful to persist).  Paired
   * with the `auto*` flags below.
   */
  spatialSkeletonGridResolutionBias2d?: WatchableValueInterface<number> & {
    value: number;
  };
  spatialSkeletonGridResolutionBias3d?: WatchableValueInterface<number> & {
    value: number;
  };
  /**
   * When `true` the render layer auto-derives `spatialSkeletonGridResolutionTarget*`
   * from the current camera projection at the layer's `localPosition`
   * (world units per screen pixel).  Default off; opt in to get
   * camera-driven LOD switching for spatially-indexed skeletons.
   */
  autoSpatialSkeletonGridLevel2d?: WatchableValueInterface<boolean>;
  autoSpatialSkeletonGridLevel3d?: WatchableValueInterface<boolean>;
}

/**
 * Resolve a picked node/edge offset to its owning segment as a full uint64
 * `bigint`.  `segmentIds` is the interleaved per-vertex segment column with
 * `segmentComponents` uint32 per vertex (2 = `[lo, hi]` full uint64; 1 = a
 * uint32 id with implicit high half 0).  Returns `undefined` for an absent /
 * zero id.
 */
export function resolveSpatiallyIndexedSkeletonSegmentPick(
  chunk: { indices: Uint32Array; numVertices: number },
  segmentIds: Uint32Array,
  pickedOffset: number,
  kind: "node" | "edge" | "face",
  segmentComponents = 1,
): bigint | undefined {
  const readId = (vertex: number): bigint | undefined => {
    const base = vertex * segmentComponents;
    if (vertex < 0 || base + segmentComponents > segmentIds.length) {
      return undefined;
    }
    const lo = BigInt(segmentIds[base] >>> 0);
    const hi = segmentComponents >= 2 ? BigInt(segmentIds[base + 1] >>> 0) : 0n;
    const id = lo | (hi << 32n);
    return id > 0n ? id : undefined;
  };
  if (pickedOffset < 0) return undefined;
  if (kind === "node") {
    if (pickedOffset >= chunk.numVertices) return undefined;
    return readId(pickedOffset);
  }
  // The picked primitive spans this many indices: an edge two, a triangle
  // three. Any of its corners can answer for the whole primitive -- they belong
  // to one object -- so take the first that has an id, which is also what makes
  // a bridge edge to a ghost vertex resolvable.
  const verticesPerPrimitive = kind === "face" ? 3 : 2;
  const indexOffset = pickedOffset * verticesPerPrimitive;
  if (indexOffset + verticesPerPrimitive > chunk.indices.length) {
    return undefined;
  }
  for (let i = 0; i < verticesPerPrimitive; ++i) {
    const id = readId(chunk.indices[indexOffset + i]);
    if (id !== undefined) return id;
  }
  return undefined;
}

export class SpatiallyIndexedSkeletonLayer
  extends RefCounted
  implements SkeletonShaderContext
{
  layerChunkProgressInfo = new LayerChunkProgressInfo();
  redrawNeeded = new NullarySignal();
  vertexAttributes: VertexAttributeRenderInfo[];
  segmentColorAttributeIndex: number | undefined;
  selectedNodeAttributeIndex: number | undefined;
  /** See {@link PackedAttributeRange}; set by sources that pack. */
  packedAttributeRange: PackedAttributeRange | undefined;
  /** What primitive this source's geometry is drawn as; see the constructor. */
  readonly geometryPrimitive: GeometryPrimitive;
  readonly browsePassLayerView: SkeletonShaderContext;
  readonly skeletonShaderParameters: WatchableValue<SkeletonShaderParameters>;
  readonly browsePassSkeletonShaderParameters: WatchableValueInterface<SkeletonShaderParameters>;
  fallbackShaderParameters = new WatchableValue(
    getFallbackBuilderState(parseShaderUiControls(DEFAULT_FRAGMENT_MAIN)),
  );
  backend: ChunkRenderLayerFrontend;
  localPosition: WatchableValueInterface<Float32Array>;
  readonly chunkTransform: WatchableValueInterface<
    ValueOrError<ChunkTransformParameters>
  >;
  rpc: RPC | undefined;

  private overlayAttributeTextureFormats_?: TextureFormat[];
  private get overlayAttributeTextureFormats(): TextureFormat[] {
    return (this.overlayAttributeTextureFormats_ ??= this.vertexAttributes.map(
      ({ dataType, numComponents }) =>
        computeTextureFormat(new TextureFormat(), dataType, numComponents),
    ));
  }
  gridLevel: WatchableValueInterface<number>;
  lod: WatchableValueInterface<number>;
  gridLevel2d: WatchableValueInterface<number>;
  lod2d: WatchableValueInterface<number>;
  detailFocus: WatchableValueInterface<number>;
  admissionFraction: WatchableValueInterface<number>;
  private selectedNodeId:
    | WatchableValueInterface<number | undefined>
    | undefined;
  private pendingNodePositionVersion:
    | WatchableValueInterface<number>
    | undefined;
  private getPendingNodePositionOverride:
    | ((nodeId: number) => ArrayLike<number> | undefined)
    | undefined;
  private getCachedNodeInfo:
    | ((nodeId: number) => SpatiallyIndexedSkeletonNode | undefined)
    | undefined;
  private inspectionState: SpatiallyIndexedSkeletonInspectionState | undefined;
  private overlayChunk: SkeletonOverlayChunk | undefined;
  private overlayChunkKey: string | undefined;
  private overlayGeometryKey: string | undefined;
  private cachedSelectedNodeId: number | undefined;
  private overlayRebuildFrame = -1;
  private pendingOverlaySegmentLoads = new Set<number>();
  // Segments whose `getFullSegmentNodes` call rejected (e.g. the active
  // data source doesn't implement full-skeleton inspection at all, as is
  // the case for every zarr-vectors layer today). Without this, a
  // permanently-failing fetch is retried on every redraw forever: each
  // retry's `.finally()` disposes the overlay chunk and dispatches
  // another redraw, which triggers another retry — an infinite loop that
  // starves the render loop the moment any segment becomes visible.
  // Evicted alongside the success cache in `resolveSourceBackedOverlayChunk`
  // so a segment that becomes active again after going inactive gets one
  // more attempt, in case the earlier failure was transient.
  private failedOverlaySegmentLoads = new Set<number>();
  private browseExcludedSegments = new Uint64Set();
  private gpuBrowseExcludedSegmentsHashTable: GPUHashTable<HashSetUint64>;
  private browseExcludedSegmentsKey: string | undefined;
  private suppressedBrowseSegmentIds = new Set<number>();
  private retainedOverlaySegmentIds: number[] = [];
  private maxRetainedOverlaySegments: number;

  private disposeOverlayChunk() {
    this.overlayChunk?.dispose(this.gl);
    this.overlayChunk = undefined;
    this.overlayChunkKey = undefined;
    this.overlayGeometryKey = undefined;
    this.cachedSelectedNodeId = undefined;
  }

  private requestOverlaySegmentLoad(segmentId: number) {
    if (
      this.inspectionState === undefined ||
      this.pendingOverlaySegmentLoads.has(segmentId) ||
      this.failedOverlaySegmentLoads.has(segmentId)
    ) {
      return;
    }
    this.pendingOverlaySegmentLoads.add(segmentId);
    let failed = false;
    void this.inspectionState
      .getFullSegmentNodes(this, segmentId)
      .catch(() => {
        failed = true;
      })
      .finally(() => {
        this.pendingOverlaySegmentLoads.delete(segmentId);
        if (failed) {
          this.failedOverlaySegmentLoads.add(segmentId);
        }
        this.disposeOverlayChunk();
        this.redrawNeeded.dispatch();
      });
  }

  private getOverlayGeometryKey(segmentIds: readonly number[]) {
    return [
      segmentIds.join(","),
      `pending:${this.pendingNodePositionVersion?.value ?? ""}`,
      `data:${this.inspectionState?.nodeDataVersion.value ?? ""}`,
    ].join("|");
  }

  private getActiveEditableSegmentIds() {
    const segments = getVisibleSegments(
      this.displayState.segmentationGroupState.value,
    );
    const segmentIds: number[] = [];
    for (const segmentId of segments.keys()) {
      const normalizedSegmentId = Number(segmentId);
      if (
        !Number.isSafeInteger(normalizedSegmentId) ||
        normalizedSegmentId <= 0
      ) {
        continue;
      }
      segmentIds.push(normalizedSegmentId);
    }
    segmentIds.sort((a, b) => a - b);
    return segmentIds;
  }

  getRetainedOverlaySegmentIds() {
    return this.retainedOverlaySegmentIds;
  }

  retainOverlaySegment(segmentId: number) {
    const nextRetainedOverlaySegmentIds =
      retainSpatiallyIndexedSkeletonOverlaySegment(
        this.retainedOverlaySegmentIds,
        segmentId,
        { maxRetained: this.maxRetainedOverlaySegments },
      );
    if (
      nextRetainedOverlaySegmentIds.length ===
        this.retainedOverlaySegmentIds.length &&
      nextRetainedOverlaySegmentIds.every(
        (candidateSegmentId, index) =>
          candidateSegmentId === this.retainedOverlaySegmentIds[index],
      )
    ) {
      return false;
    }
    this.retainedOverlaySegmentIds = nextRetainedOverlaySegmentIds;
    this.redrawNeeded.dispatch();
    return true;
  }

  suppressBrowseSegment(segmentId: number) {
    const normalizedSegmentId = Math.round(Number(segmentId));
    if (
      !Number.isSafeInteger(normalizedSegmentId) ||
      normalizedSegmentId <= 0 ||
      this.suppressedBrowseSegmentIds.has(normalizedSegmentId)
    ) {
      return false;
    }
    this.suppressedBrowseSegmentIds.add(normalizedSegmentId);
    this.redrawNeeded.dispatch();
    return true;
  }

  private getOverlayRenderSegmentIds() {
    return mergeSpatiallyIndexedSkeletonOverlaySegmentIds(
      this.getActiveEditableSegmentIds(),
      this.retainedOverlaySegmentIds,
    );
  }

  private getLoadedOverlaySegmentIds(
    segmentIds: readonly number[] = this.getOverlayRenderSegmentIds(),
  ) {
    if (this.inspectionState === undefined) {
      return [];
    }
    return segmentIds.filter(
      (segmentId) =>
        this.inspectionState?.getCachedSegmentNodes(segmentId) !== undefined,
    );
  }

  private getNormalizedBrowsePassExcludedSegmentIds() {
    const segmentIds = new Set<number>();
    for (const segmentId of this.getLoadedOverlaySegmentIds()) {
      const normalizedSegmentId = Math.round(Number(segmentId));
      if (
        !Number.isSafeInteger(normalizedSegmentId) ||
        normalizedSegmentId <= 0
      ) {
        continue;
      }
      segmentIds.add(normalizedSegmentId);
    }
    for (const segmentId of this.suppressedBrowseSegmentIds) {
      const normalizedSegmentId = Math.round(Number(segmentId));
      if (
        !Number.isSafeInteger(normalizedSegmentId) ||
        normalizedSegmentId <= 0
      ) {
        continue;
      }
      segmentIds.add(normalizedSegmentId);
    }
    return [...segmentIds].sort((a, b) => a - b);
  }

  private getBrowsePassExcludedSegments() {
    const segmentIds = this.getNormalizedBrowsePassExcludedSegmentIds();
    if (segmentIds.length === 0) {
      if (this.browseExcludedSegments.size !== 0) {
        this.browseExcludedSegments.clear();
      }
      this.browseExcludedSegmentsKey = undefined;
      return undefined;
    }
    const excludedSegmentsKey = segmentIds.join(",");
    if (this.browseExcludedSegmentsKey !== excludedSegmentsKey) {
      this.browseExcludedSegments.clear();
      this.browseExcludedSegments.add(
        segmentIds
          .filter(
            (segmentId) => Number.isSafeInteger(segmentId) && segmentId > 0,
          )
          .map((segmentId) => BigInt(segmentId)),
      );
      this.browseExcludedSegmentsKey = excludedSegmentsKey;
    }
    return this.browseExcludedSegments;
  }

  private resolveSourceBackedOverlayChunk(): SkeletonOverlayChunk | undefined {
    const frameNumber =
      this.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber;
    // Cache result for the entire frame — both slice and perspective draw calls
    // share the same chunk, and "no overlay" is also cached to avoid per-frame
    // allocation when the inspection overlay is inactive.
    if (this.overlayRebuildFrame === frameNumber) {
      return this.overlayChunk;
    }
    this.overlayRebuildFrame = frameNumber;
    if (this.inspectionState === undefined) {
      this.disposeOverlayChunk();
      return undefined;
    }
    const overlaySegmentIds = this.getOverlayRenderSegmentIds();
    if (overlaySegmentIds.length === 0) {
      this.disposeOverlayChunk();
      return undefined;
    }
    this.inspectionState.evictInactiveSegmentNodes(overlaySegmentIds);
    if (this.failedOverlaySegmentLoads.size > 0) {
      const activeSegmentIds = new Set(overlaySegmentIds);
      for (const segmentId of this.failedOverlaySegmentLoads) {
        if (!activeSegmentIds.has(segmentId)) {
          this.failedOverlaySegmentLoads.delete(segmentId);
        }
      }
    }

    // Pass 1: cheap scan to determine which segments are loaded and check cache.
    const loadedSegmentIds: number[] = [];
    for (const segmentId of overlaySegmentIds) {
      if (this.inspectionState.getCachedSegmentNodes(segmentId) !== undefined) {
        loadedSegmentIds.push(segmentId);
      } else {
        this.requestOverlaySegmentLoad(segmentId);
      }
    }
    if (loadedSegmentIds.length === 0) {
      this.disposeOverlayChunk();
      return undefined;
    }

    const overlayGeometryKey = this.getOverlayGeometryKey(loadedSegmentIds);
    const selectedNodeId = this.selectedNodeId?.value;
    const overlayChunkKey = `${overlayGeometryKey}|selected:${selectedNodeId ?? ""}`;

    if (this.overlayChunk !== undefined) {
      if (this.overlayGeometryKey === overlayGeometryKey) {
        // Geometry unchanged — update only the selected-node highlight in-place
        // rather than reallocating all GPU textures.
        if (this.overlayChunkKey !== overlayChunkKey) {
          this.overlayChunk.updateSelectedNode(
            this.gl,
            this.cachedSelectedNodeId,
            selectedNodeId,
          );
          this.overlayChunkKey = overlayChunkKey;
          this.cachedSelectedNodeId = selectedNodeId;
        }
        return this.overlayChunk;
      }
    }

    // Pass 2: geometry cache miss — collect node sets and rebuild.
    const segmentNodeSets: (readonly SpatiallyIndexedSkeletonNode[])[] = [];
    for (const segmentId of loadedSegmentIds) {
      const segmentNodes =
        this.inspectionState.getCachedSegmentNodes(segmentId);
      if (segmentNodes !== undefined) {
        segmentNodeSets.push(segmentNodes);
      }
    }
    this.disposeOverlayChunk();
    const geometry = buildSpatiallyIndexedSkeletonOverlayGeometry(
      segmentNodeSets,
      {
        selectedNodeId,
        getPendingNodePosition: this.getPendingNodePositionOverride,
      },
    );
    this.overlayChunk = new SkeletonOverlayChunk(
      this.gl,
      geometry,
      this.overlayAttributeTextureFormats,
    );
    this.overlayChunkKey = overlayChunkKey;
    this.overlayGeometryKey = overlayGeometryKey;
    this.cachedSelectedNodeId = selectedNodeId;
    return this.overlayChunk;
  }

  sources: SpatiallyIndexedSkeletonSourceEntry[];
  sources2d: SpatiallyIndexedSkeletonSourceEntry[];
  source: SpatiallyIndexedSkeletonSource;

  constructor(
    public chunkManager: ChunkManager,
    sources:
      | SpatiallyIndexedSkeletonSourceEntry[]
      | SpatiallyIndexedSkeletonSource,
    public displayState: SpatiallyIndexedSkeletonLayerDisplayState & {
      localPosition: WatchableValueInterface<Float32Array>;
    },
    options: SpatiallyIndexedSkeletonLayerOptions = {},
  ) {
    super();
    this.registerDisposer(() => {
      this.disposeOverlayChunk();
    });
    let sources3d: SpatiallyIndexedSkeletonSourceEntry[];
    let sources2d = options.sources2d ?? [];
    if (Array.isArray(sources)) {
      sources3d = sources;
    } else {
      sources3d = [
        {
          chunkSource: sources,
          chunkToMultiscaleTransform: mat4.create(),
        },
      ];
    }
    if (sources3d.length === 0 && sources2d.length > 0) {
      sources3d = sources2d;
    }
    if (sources2d.length === 0) {
      sources2d = sources3d;
    }
    if (sources3d.length === 0) {
      throw new Error(
        "SpatiallyIndexedSkeletonLayer requires at least one source.",
      );
    }
    this.sources = sources3d;
    this.sources2d = sources2d;
    this.source = sources3d[0].chunkSource;
    this.localPosition = displayState.localPosition;
    this.chunkTransform = this.registerDisposer(
      makeCachedLazyDerivedWatchableValue(
        (modelTransform) =>
          makeValueOrError(() =>
            getChunkTransformParameters(valueOrThrow(modelTransform)),
          ),
        this.displayState.transform,
      ),
    );
    this.gridLevel =
      options.gridLevel ??
      displayState.spatialSkeletonGridLevel3d ??
      new WatchableValue(0);
    this.lod = options.lod ?? displayState.skeletonLod ?? new WatchableValue(0);
    this.gridLevel2d =
      options.gridLevel2d ??
      displayState.spatialSkeletonGridLevel2d ??
      this.gridLevel;
    this.lod2d = options.lod2d ?? displayState.spatialSkeletonLod2d ?? this.lod;
    // A source that names no mode gets LOCAL, which is the behaviour every
    // spatially-indexed skeleton layer had before the mode existed.
    this.detailFocus =
      options.detailFocus ??
      new WatchableValue<number>(SpatialSkeletonDetailFocus.LOCAL);
    this.admissionFraction =
      options.admissionFraction ?? new WatchableValue<number>(1);
    this.selectedNodeId = options.selectedNodeId;
    this.pendingNodePositionVersion = options.pendingNodePositionVersion;
    this.getPendingNodePositionOverride = options.getPendingNodePosition;
    this.getCachedNodeInfo = options.getCachedNode;
    this.inspectionState = options.inspectionState;
    this.maxRetainedOverlaySegments = Math.max(
      1,
      Math.round(
        options.maxRetainedOverlaySegments ??
          DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS,
      ),
    );
    registerRedrawWhenSegmentationDisplayState3DChanged(displayState, this);
    this.displayState.shaderError.value = undefined;
    const { skeletonRenderingOptions: renderingOptions } = displayState;
    this.registerDisposer(
      renderingOptions.shader.changed.add(() => {
        this.displayState.shaderError.value = undefined;
        this.redrawNeeded.dispatch();
      }),
    );

    this.vertexAttributes = [
      ...this.source.vertexAttributes,
      selectedNodeAttribute,
    ];
    // Indices are into the source's list, and this only appends, so they carry
    // over unchanged.
    this.packedAttributeRange = (
      this.source as { packedAttributeRange?: PackedAttributeRange }
    ).packedAttributeRange;
    // What the geometry IS, as opposed to how the user prefers to see it. A
    // zarr-vectors point cloud has no edges and a zarr-vectors mesh has no line
    // segments, so neither can honour the lines/points preference. Structural
    // read, like `isRoiHighDetailSource` below -- other datasources sharing this
    // class declare nothing and keep the `"lines"` default.
    this.geometryPrimitive =
      (this.source as { geometryPrimitive?: GeometryPrimitive })
        .geometryPrimitive ?? "lines";
    // Constant for the layer's lifetime: the passing set is attached at layer
    // creation for zarr-vectors tract layers and never for others.
    const hasRoiFilter = this.displayState.roiPassingSegments !== undefined;
    const hasRoiSegmentColors =
      this.displayState.roiSegmentColors !== undefined;
    // The pass-2 layer runs this same shader over the same display state, so the
    // hide tier must be compiled ONLY into pass 1 -- otherwise pass 2 would hide
    // exactly the tracts it exists to draw, and nothing would render at all.
    // The source is what distinguishes them.
    const isRoiHighDetailLayer =
      (this.source as { isRoiHighDetailSource?: boolean })
        .isRoiHighDetailSource === true;
    const hasRoiHighDetailHide =
      hasRoiFilter &&
      !isRoiHighDetailLayer &&
      this.displayState.roiHighDetailSegments !== undefined;
    // Background per-object value tier: pass 1 only (see hasRoiHighDetailHide;
    // the background tracts it governs are the coarse bulk pass 2 never draws).
    const hasRoiObjectValues =
      this.displayState.roiObjectValues !== undefined && !isRoiHighDetailLayer;
    this.skeletonShaderParameters =
      new WatchableValue<SkeletonShaderParameters>({
        dynamicSegmentAppearance: true,
        hasRoiFilter,
        hasRoiHighDetailHide,
        hasRoiSegmentColors,
        hasRoiObjectValues,
        hasSegmentStatedColors: false,
        hasSegmentDefaultColor: false,
        hoverHighlight: false,
        spatialChunkCulling: false,
      });
    const updateSkeletonShaderParameters = () => {
      const colorGroupState =
        this.displayState.segmentationColorGroupState.value;
      this.skeletonShaderParameters.value = {
        dynamicSegmentAppearance: true,
        hasRoiFilter,
        hasRoiHighDetailHide,
        hasRoiSegmentColors,
        hasRoiObjectValues,
        hasSegmentStatedColors: colorGroupState.segmentStatedColors.size !== 0,
        hasSegmentDefaultColor:
          colorGroupState.segmentDefaultColor.value !== undefined ||
          DEBUG_SPATIAL_SKELETON_CHUNKS,
        hoverHighlight: this.displayState.hoverHighlight.value,
        spatialChunkCulling: false,
      };
    };
    this.registerDisposer(
      registerNested((context, colorGroupState) => {
        context.registerDisposer(
          colorGroupState.segmentStatedColors.changed.add(
            updateSkeletonShaderParameters,
          ),
        );
        context.registerDisposer(
          colorGroupState.segmentDefaultColor.changed.add(
            updateSkeletonShaderParameters,
          ),
        );
        updateSkeletonShaderParameters();
      }, this.displayState.segmentationColorGroupState),
    );
    this.registerDisposer(
      this.displayState.hoverHighlight.changed.add(
        updateSkeletonShaderParameters,
      ),
    );
    this.browsePassSkeletonShaderParameters = this.registerDisposer(
      makeCachedLazyDerivedWatchableValue(
        (params) => ({ ...params, spatialChunkCulling: true }),
        this.skeletonShaderParameters,
      ),
    );

    // Browse pass uses uniform-based dynamic segment color (not per-vertex attribute),
    // so segmentColorAttributeIndex is intentionally undefined here.
    this.browsePassLayerView = {
      vertexAttributes: this.source.vertexAttributes,
      segmentColorAttributeIndex: undefined,
      packedAttributeRange: this.packedAttributeRange,
      gl: this.gl,
      fallbackShaderParameters: this.fallbackShaderParameters,
      displayState: this.displayState,
      skeletonShaderParameters: this.browsePassSkeletonShaderParameters,
    };
    const selectedNodeIndex = this.vertexAttributes.findIndex(
      (x) => x.name === selectedNodeAttribute.name,
    );
    this.selectedNodeAttributeIndex =
      selectedNodeIndex >= 0 ? selectedNodeIndex : undefined;
    const requestRedraw = () => this.redrawNeeded.dispatch();
    // The mode changes which source each visible cell draws from, so a change
    // to it must repaint even though nothing about the camera moved.
    if (this.detailFocus.changed) {
      this.registerDisposer(this.detailFocus.changed.add(requestRedraw));
    }
    // A change to the admission fraction changes what each chunk CONTAINS, not
    // merely which chunk is drawn, so already-decoded chunks are wrong and must
    // go. Dropping them beats keying them by fraction: an invalidated chunk is
    // freed, whereas a re-keyed one lingers in the budget under a dead key until
    // something outbids it — the trap the `lod` suffix used to set.
    if (this.admissionFraction.changed) {
      this.registerDisposer(
        this.admissionFraction.changed.add(() => {
          for (const entries of [this.sources, this.sources2d]) {
            for (const entry of entries) entry.chunkSource.invalidateCache();
          }
          requestRedraw();
        }),
      );
    }
    const selectedNodeWatchable = this.selectedNodeId;
    if (selectedNodeWatchable?.changed) {
      this.registerDisposer(selectedNodeWatchable.changed.add(requestRedraw));
    }
    const pendingNodePositionVersion = options.pendingNodePositionVersion;
    if (pendingNodePositionVersion?.changed) {
      this.registerDisposer(
        pendingNodePositionVersion.changed.add(requestRedraw),
      );
    }
    const inspectionState = this.inspectionState;
    if (inspectionState !== undefined) {
      this.registerDisposer(
        inspectionState.nodeDataVersion.changed.add(() => {
          this.redrawNeeded.dispatch();
        }),
      );
    }
    // Create backend for perspective view chunk management
    const sharedObject = this.registerDisposer(
      new ChunkRenderLayerFrontend(this.layerChunkProgressInfo),
    );
    const rpc = chunkManager.rpc!;
    this.rpc = rpc;
    sharedObject.RPC_TYPE_ID = SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID;

    const renderScaleTargetWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(
        rpc,
        displayState.renderScaleTarget,
      ),
    );

    const skeletonLodWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.lod),
    );

    const skeletonGridLevelWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.gridLevel),
    );

    const skeletonLod2dWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.lod2d),
    );

    const skeletonGridLevel2dWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.gridLevel2d),
    );

    const skeletonDetailFocusWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.detailFocus),
    );

    const skeletonAdmissionFractionWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.admissionFraction),
    );

    // Per-level spacing in metres, indexed by grid level. The backend's
    // arbitration needs the same density-corrected figure this side uses; see
    // `getLevelSpacingsMeters`.
    const skeletonLevelSpacingsMetersWatchable = this.registerDisposer(
      SharedWatchableValue.make<number[]>(rpc, this.getLevelSpacingsMeters()),
    );
    const gridLevels = displayState.spatialSkeletonGridLevels;
    if (gridLevels?.changed !== undefined) {
      this.registerDisposer(
        gridLevels.changed.add(() => {
          skeletonLevelSpacingsMetersWatchable.value =
            this.getLevelSpacingsMeters();
        }),
      );
    }

    const skeletonGridResolutionTarget3dWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(
        rpc,
        displayState.spatialSkeletonGridResolutionTarget3d!,
      ),
    );

    const counterpartOptions: { [key: string]: any } = {
      chunkManager: chunkManager.rpcId,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.localPosition),
      ).rpcId,
      renderScaleTarget: renderScaleTargetWatchable.rpcId,
      skeletonLod: skeletonLodWatchable.rpcId,
      skeletonGridLevel: skeletonGridLevelWatchable.rpcId,
      skeletonLod2d: skeletonLod2dWatchable.rpcId,
      skeletonGridLevel2d: skeletonGridLevel2dWatchable.rpcId,
      skeletonDetailFocus: skeletonDetailFocusWatchable.rpcId,
      skeletonAdmissionFraction: skeletonAdmissionFractionWatchable.rpcId,
      skeletonLevelSpacingsMeters: skeletonLevelSpacingsMetersWatchable.rpcId,
      skeletonGridResolutionTarget3d:
        skeletonGridResolutionTarget3dWatchable.rpcId,
    };

    // ROI streamline filter channel (zarr-vectors tract layers only): hand the
    // backend the ROI groups (so it recomputes the passing set) and the shared
    // passing set it mutates. `roiPassingSegments` is already a shared object
    // (it drives this layer's shader too); `roiGroups` is wrapped from the plain
    // display-state watchable the way `skeletonLod` is. The active flag and
    // ghost opacity are NOT sent to the backend — they only feed shader uniforms
    // here (the backend keeps the passing set current whenever ROIs exist, so
    // enabling the filter is instant). Redraw when the passing set, the active
    // flag, or the ghost opacity changes.
    if (
      displayState.roiPassingSegments !== undefined &&
      displayState.roiGroups !== undefined &&
      displayState.roiFilterActive !== undefined
    ) {
      counterpartOptions.roiPassingSegments =
        displayState.roiPassingSegments.rpcId;
      if (displayState.roiSegmentColors !== undefined) {
        counterpartOptions.roiSegmentColors =
          displayState.roiSegmentColors.rpcId;
      }
      counterpartOptions.roiGroups = this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, displayState.roiGroups),
      ).rpcId;
      if (displayState.roiObjectAttrColumns !== undefined) {
        counterpartOptions.roiObjectAttrColumns = this.registerDisposer(
          SharedWatchableValue.makeFromExisting(
            rpc,
            displayState.roiObjectAttrColumns,
          ),
        ).rpcId;
      }
      if (displayState.roiLabelField !== undefined) {
        // Shared, not snapshotted: the dense parcellation grid arrives
        // asynchronously (and can change if the linked layer changes), so the
        // backend must see the value settle rather than the initial undefined.
        counterpartOptions.roiLabelField = this.registerDisposer(
          SharedWatchableValue.makeFromExisting(
            rpc,
            displayState.roiLabelField,
          ),
        ).rpcId;
      }
      const roiRedraw = () => this.redrawNeeded.dispatch();
      this.registerDisposer(
        displayState.roiPassingSegments.changed.add(roiRedraw),
      );
      this.registerDisposer(
        displayState.roiFilterActive.changed.add(roiRedraw),
      );
      if (displayState.roiGhostAlpha !== undefined) {
        this.registerDisposer(
          displayState.roiGhostAlpha.changed.add(roiRedraw),
        );
      }
      if (displayState.roiSegmentColors !== undefined) {
        this.registerDisposer(
          displayState.roiSegmentColors.changed.add(roiRedraw),
        );
      }
      if (displayState.roiColorByGroup !== undefined) {
        this.registerDisposer(
          displayState.roiColorByGroup.changed.add(roiRedraw),
        );
      }
      if (displayState.roiObjectValues !== undefined) {
        this.registerDisposer(
          displayState.roiObjectValues.changed.add(roiRedraw),
        );
      }
      if (displayState.roiBackground !== undefined) {
        this.registerDisposer(
          displayState.roiBackground.changed.add(roiRedraw),
        );
      }
      if (displayState.roiHighDetailSegments !== undefined) {
        // Pass 1's hide tier reads this set, but it is not pass 1's visible set,
        // so no segmentation-state redraw covers it. Without this, tracts pass 2
        // has taken over keep drawing here until something else forces a frame --
        // which shows as every newly claimed tract briefly drawn twice.
        this.registerDisposer(
          displayState.roiHighDetailSegments.changed.add(roiRedraw),
        );
      }
    }

    sharedObject.initializeCounterpart(rpc, counterpartOptions);
    this.backend = sharedObject;
    this.gpuBrowseExcludedSegmentsHashTable = this.registerDisposer(
      GPUHashTable.get(this.gl, this.browseExcludedSegments.hashTable),
    );
  }

  /**
   * Ask the backend for each group's on-screen passing object ids -- the tract
   * export selection. Positional: `result[i]` corresponds to `groups[i]`. A
   * request/response RPC (the tab needs the ids before it can build the export
   * job); see the backend's `computeRoiExportIds`.
   */
  async computeRoiExportIds(
    groups: readonly RoiGroupConfig[],
  ): Promise<bigint[][]> {
    const rpc = this.rpc;
    if (rpc === undefined) {
      throw new Error("This tract layer is not connected to a worker.");
    }
    const { perGroup } = await rpc.promiseInvoke<{ perGroup: string[][] }>(
      SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_ROI_EXPORT_IDS_RPC_ID,
      { layer: this.backend.rpcId, groups },
    );
    return perGroup.map((ids) => ids.map((s) => BigInt(s)));
  }

  /**
   * Ask the backend for the observed range of each named per-vertex attribute
   * over the resident chunks -- what the Filter tab's attribute picker needs to
   * offer a meaningful control. Positional: `result[i]` ↔ `names[i]`. The
   * values exist only in the worker, hence the round trip.
   */
  async computeRoiVertexAttrStats(
    names: readonly string[],
  ): Promise<VertexAttrStats[]> {
    const rpc = this.rpc;
    if (rpc === undefined) {
      throw new Error("This geometry layer is not connected to a worker.");
    }
    const { stats } = await rpc.promiseInvoke<{ stats: VertexAttrStats[] }>(
      SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_VERTEX_ATTR_STATS_RPC_ID,
      { layer: this.backend.rpcId, names: [...names] },
    );
    return stats;
  }

  get gl() {
    return this.chunkManager.chunkQueueManager.gl;
  }

  getSources(view: SpatiallyIndexedSkeletonView) {
    return view === "2d" ? this.sources2d : this.sources;
  }

  /**
   * Whether OBJECT focus may draw the UNION of levels for this view.
   *
   * Object focus has two halves, and only one of them needs the store's help.
   * Drawing one level everywhere instead of arbitrating per cell -- so an
   * object is never cut off where the finer chunks ran out -- works on any
   * pyramid. Drawing several levels at once only works where they partition the
   * objects between them; on a resolution pyramid (mesh and point-cloud stores,
   * whose levels are decimated copies of every object) the union superimposes
   * those copies. So the union is gated here and the rest of object focus is
   * not.
   */
  objectPartitionAvailable(view: SpatiallyIndexedSkeletonView) {
    const sources = this.getSources(view);
    return (
      sources.length > 0 &&
      sources.every((entry) =>
        getSpatiallyIndexedSkeletonPartitionsObjects(entry.chunkSource),
      )
    );
  }

  private selectSourcesForViewAndGrid(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
  ) {
    return selectSpatiallyIndexedSkeletonEntriesForView(
      this.getSources(view),
      view,
      gridLevel,
      getSpatiallyIndexedSkeletonSourceView,
      getSpatiallyIndexedSkeletonGridIndex,
    );
  }

  private selectSourcesForViewAndGridWithFallback(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
  ) {
    return selectSpatiallyIndexedSkeletonEntriesForViewWithFallback(
      this.getSources(view),
      view,
      gridLevel,
      getSpatiallyIndexedSkeletonSourceView,
      getSpatiallyIndexedSkeletonGridIndex,
    );
  }

  /**
   * Per-level chunk spacing in METRES, indexed by grid level (coarsest first,
   * which is what `gridIndex` counts).
   *
   * Read from the same `spatialSkeletonGridLevels` list the resolution widget
   * and the camera-driven target use, so all three agree on how coarse a level
   * is. For an object-sparsity pyramid that figure is density-corrected -- mean
   * spacing between OBJECTS -- and is the only thing that distinguishes the
   * levels at all, since they share a chunk shape.
   *
   * `[]` where the display state publishes no levels, which sends every caller
   * back to the chunk-shape spacing they used before.
   */
  private getLevelSpacingsMeters(): number[] {
    const levels = this.displayState.spatialSkeletonGridLevels?.value;
    if (levels === undefined) return [];
    return levels.map(({ size }) =>
      Math.max(Math.min(size.x, size.y, size.z), 0),
    );
  }

  /**
   * How many grid cells the view covers, counted on the COARSEST level's grid.
   *
   * Deliberately not the level currently being drawn: the count feeds the
   * memory-aware resolution target, which then decides which level to draw, and
   * counting on the drawn level would close that loop into a feedback cycle. The
   * coarsest grid is a stable reference, and for an object-sparsity pyramid
   * (where every level shares one chunk_shape) it is the same count anyway.
   */
  countVisibleCells(
    view: SpatiallyIndexedSkeletonView,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
  ): number {
    let coarsest: TransformedSource | undefined;
    let coarsestGridIndex = Number.POSITIVE_INFINITY;
    const wanted = new Set(
      this.getSources(view).map((e) => getObjectId(e.chunkSource)),
    );
    for (const scales of transformedSources) {
      for (const tsource of scales) {
        if (!wanted.has(getObjectId(tsource.source))) continue;
        // Grid index 0 is the coarsest level, so the smallest wins.
        const gridIndex =
          getSpatiallyIndexedSkeletonGridIndex(tsource.source) ?? 0;
        if (gridIndex < coarsestGridIndex) {
          coarsestGridIndex = gridIndex;
          coarsest = tsource;
        }
      }
    }
    if (coarsest === undefined) return 0;
    let count = 0;
    forEachVisibleVolumetricChunk(
      projectionParameters,
      this.localPosition.value,
      coarsest,
      () => {
        ++count;
      },
    );
    return count;
  }

  private getChunkSpacing(chunkLayout: ChunkLayout): number {
    const { size } = chunkLayout;
    return Math.max(Math.min(size[0], size[1], size[2]), 1e-6);
  }

  private getChunkCenterWorld(
    chunkLayout: ChunkLayout,
    positionInChunks: Float32Array,
    out: Float32Array,
  ) {
    out[0] = (positionInChunks[0] + 0.5) * chunkLayout.size[0];
    out[1] = (positionInChunks[1] + 0.5) * chunkLayout.size[1];
    out[2] = (positionInChunks[2] + 0.5) * chunkLayout.size[2];
    vec3.transformMat4(out as vec3, out as vec3, chunkLayout.transform);
  }

  private getChunkGridPositionForWorldPoint(
    tsource: TransformedSource,
    worldPoint: Float32Array,
    out: Float32Array,
  ): boolean {
    const localPoint = vec3.create();
    tsource.chunkLayout.globalToLocalSpatial(localPoint, worldPoint as vec3);
    const { size } = tsource.chunkLayout;
    const { lowerChunkBound, upperChunkBound } = (
      tsource.source as SpatiallyIndexedSkeletonSource
    ).spec;
    for (let i = 0; i < 3; ++i) {
      const dimSize = size[i];
      if (!Number.isFinite(dimSize) || dimSize <= 0) {
        return false;
      }
      const chunkCoord = Math.floor(localPoint[i] / dimSize);
      if (
        Number.isFinite(lowerChunkBound[i]) &&
        Number.isFinite(upperChunkBound[i]) &&
        (chunkCoord < lowerChunkBound[i] || chunkCoord >= upperChunkBound[i])
      ) {
        return false;
      }
      out[i] = chunkCoord;
    }
    return true;
  }

  private getMetersPerUnit(projectionParameters: ProjectionParameters): number {
    const ddScales =
      projectionParameters.displayDimensionRenderInfo?.displayDimensionScales;
    if (ddScales === undefined || ddScales.length === 0) {
      return 1;
    }
    let metersPerUnit = Infinity;
    for (let i = 0; i < ddScales.length; ++i) {
      const s = ddScales[i];
      if (Number.isFinite(s) && s > 0) {
        metersPerUnit = Math.min(metersPerUnit, s);
      }
    }
    return Number.isFinite(metersPerUnit) ? metersPerUnit : 1;
  }

  private getReferencePixelSize(
    projectionParameters: ProjectionParameters,
  ): number {
    let reference: Float32Array | undefined;
    if (projectionParameters.globalPosition !== undefined) {
      reference = projectionParameters.globalPosition;
    } else if (this.localPosition.value.length >= 3) {
      reference = this.localPosition.value;
    } else {
      reference = new Float32Array(3);
    }
    const pixelSize = computePhysicalUnitsPerScreenPixel(
      projectionParameters.viewProjectionMat,
      projectionParameters.width,
      projectionParameters.height,
      reference,
      projectionParameters.displayDimensionRenderInfo?.displayDimensionScales,
    );
    return Number.isFinite(pixelSize) && pixelSize > 0 ? pixelSize : 1;
  }

  private getArbitrationTargetSpacingMeters3d(
    projectionParameters: ProjectionParameters,
  ): number {
    const target =
      this.displayState.spatialSkeletonGridResolutionTarget3d?.value;
    if (Number.isFinite(target) && target !== undefined && target > 0) {
      return target;
    }
    return this.getMetersPerUnit(projectionParameters);
  }

  // Quantize spacing to quarter-octave bins so tiny camera rotations do not
  // globally thrash chunk-level arbitration decisions.
  private quantizeSpacingForArbitration(spacing: number): number {
    const clamped = Math.max(spacing, 1e-12);
    const log2Spacing = Math.log2(clamped);
    const quantizedLog = Math.round(log2Spacing * 4) / 4;
    return 2 ** quantizedLog;
  }

  private getCachedNodeSnapshot(nodeId: number) {
    const cachedNode = this.getCachedNodeInfo?.(nodeId);
    if (cachedNode === undefined) {
      return undefined;
    }
    const pendingPosition =
      this.getPendingNodePositionOverride?.(cachedNode.nodeId) ??
      cachedNode.position;
    return {
      ...cachedNode,
      position: new Float32Array([
        Number(pendingPosition[0]),
        Number(pendingPosition[1]),
        Number(pendingPosition[2]),
      ]),
    };
  }

  invalidateSourceCellsForPositions(
    positions: Iterable<ArrayLike<number> | undefined>,
  ) {
    const positionList = [...positions].filter(
      (position): position is ArrayLike<number> => position !== undefined,
    );
    if (positionList.length === 0) {
      return false;
    }
    let invalidated = false;
    const seenSourceIds = new Set<string>();
    for (const sourceEntry of [...this.sources, ...this.sources2d]) {
      const chunkSource = sourceEntry.chunkSource;
      const sourceId = getObjectId(chunkSource);
      if (seenSourceIds.has(sourceId)) continue;
      seenSourceIds.add(sourceId);
      const keyPrefixes = new Set<string>();
      const { chunkDataSize } = chunkSource.spec;
      for (const position of positionList) {
        // Spatial skeleton node positions are already source/model coordinates;
        // render-layer transforms do not apply to CATMAID grid-cell keys.
        const keyPrefix = getSpatialSkeletonCellKeyPrefix(
          position,
          chunkDataSize,
        );
        if (keyPrefix !== undefined) {
          keyPrefixes.add(keyPrefix);
        }
      }
      if (keyPrefixes.size === 0) {
        continue;
      }
      chunkSource.invalidateCacheKeyPrefixes(keyPrefixes);
      invalidated = true;
    }
    if (!invalidated) {
      return false;
    }
    this.redrawNeeded.dispatch();
    return true;
  }

  private getChunkPositionAndSegmentArrays(
    chunk: SpatiallyIndexedSkeletonChunk,
  ) {
    const offsets = chunk.vertexAttributeOffsets;
    if (!offsets || offsets.length < 1) return undefined;
    const positions = new Float32Array(
      chunk.vertexAttributes.buffer,
      chunk.vertexAttributes.byteOffset + offsets[0],
      chunk.numVertices * 3,
    );
    // Locate the "segment" column by its actual attribute index — the
    // zarr-vectors layout is [position, tangent, …, segment], so segment is
    // NOT necessarily offsets[1].  Count the uint32 it occupies per vertex:
    // a UINT64 attribute (zarr-vectors full uint64) is 2 uint32 [lo, hi]
    // despite numComponents===1; a UINT32 attribute (CATMAID) is 1 (high
    // half implicitly 0).
    const segIdx = this.vertexAttributes.findIndex((a) => a.name === "segment");
    if (segIdx < 0 || segIdx >= offsets.length) return undefined;
    const segInfo = this.vertexAttributes[segIdx];
    const segmentComponents =
      segInfo.dataType === DataType.UINT64
        ? 2 * segInfo.numComponents
        : segInfo.numComponents;
    const segmentIds = new Uint32Array(
      chunk.vertexAttributes.buffer,
      chunk.vertexAttributes.byteOffset + offsets[segIdx],
      chunk.numVertices * segmentComponents,
    );
    return { positions, segmentIds, segmentComponents };
  }

  resolveSegmentPickFromChunk(
    chunk: SpatiallyIndexedSkeletonChunk,
    pickedOffset: number,
    kind: "node" | "edge" | "face",
  ) {
    const data = this.getChunkPositionAndSegmentArrays(chunk);
    if (data === undefined) {
      return undefined;
    }
    return resolveSpatiallyIndexedSkeletonSegmentPick(
      chunk,
      data.segmentIds,
      pickedOffset,
      kind,
      data.segmentComponents,
    );
  }

  resolveNodePickFromChunk(
    chunk: SpatiallyIndexedSkeletonChunk,
    pickedOffset: number,
  ) {
    const data = this.getChunkPositionAndSegmentArrays(chunk);
    if (
      data === undefined ||
      pickedOffset < 0 ||
      pickedOffset >= chunk.numVertices ||
      pickedOffset >= chunk.nodeIds.length
    ) {
      return undefined;
    }
    const nodeId = chunk.nodeIds[pickedOffset];
    if (!Number.isSafeInteger(nodeId) || nodeId <= 0) {
      return undefined;
    }
    const segmentId = resolveSpatiallyIndexedSkeletonSegmentPick(
      chunk,
      data.segmentIds,
      pickedOffset,
      "node",
      data.segmentComponents,
    );
    if (segmentId === undefined) {
      return undefined;
    }
    const baseOffset = pickedOffset * 3;
    return {
      nodeId,
      segmentId,
      position: data.positions.subarray(baseOffset, baseOffset + 3),
      sourceState: chunk.nodeSourceStates[pickedOffset],
    };
  }

  // Iterates every chunk slot in view for the given view/gridLevel.
  // Callback receives (chunkKey, chunkSource, chunkLayout); return false to stop early.
  //
  // Takes no `lod`: chunk keys are grid position alone (see `getChunk` in
  // `skeleton/backend.ts`). Callers still resolve their own `lod` first, but only
  // to decide whether there is a level to draw at all.
  private forEachVisibleChunkSlot(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
    callback: (
      chunkKey: string,
      chunkSource: SpatiallyIndexedSkeletonSource,
      chunkLayout: ChunkLayout,
    ) => boolean | void,
  ) {
    const selectedSources =
      view === "3d"
        ? this.selectSourcesForViewAndGridWithFallback(view, gridLevel)
        : this.selectSourcesForViewAndGrid(view, gridLevel);
    // Which sources the non-arbitrating path below draws from.
    //
    // Normally exactly one: the preferred level. `...WithFallback` returns every
    // level in preference order for a caller to try in sequence, not a list to
    // draw simultaneously, and drawing the whole list would render every
    // resident level on top of itself.
    //
    // Under OBJECT focus it is every level from the drawn one up to the
    // coarsest, because there they do not overlap: each draws only the objects
    // that are new at it (see `admitObjects` in the zarr-vectors backend), so
    // together they cover the admitted set exactly once. Grid index counts from
    // the coarsest, so that is `gridIndex <= gridLevel`.
    //
    // ...and only where the levels really do partition the objects. Without
    // that the union draws every resident level's copy of the same object on
    // top of itself; object focus still means "one level everywhere", it just
    // means the SELECTED level rather than a coarse-to-fine stack.
    const objectFocus =
      this.detailFocus.value === SpatialSkeletonDetailFocus.OBJECT &&
      this.objectPartitionAvailable(view);
    const drawSourceIds = new Set<string>();
    if (objectFocus && gridLevel !== undefined) {
      for (const entry of this.getSources(view)) {
        const entryGrid = getSpatiallyIndexedSkeletonGridIndex(
          entry.chunkSource,
        );
        if (entryGrid !== undefined && entryGrid <= gridLevel) {
          drawSourceIds.add(getObjectId(entry.chunkSource));
        }
      }
    }
    if (drawSourceIds.size === 0 && selectedSources.length > 0) {
      drawSourceIds.add(getObjectId(selectedSources[0].chunkSource));
    }

    // Per-cell level arbitration -- the standard neuroglancer behaviour, where
    // a cell near the camera resolves finer than one far from it. OBJECT focus
    // opts out: it draws ONE level everywhere and spends the memory that leaves
    // on whole objects, which scattering levels across the view would undo.
    if (
      view === "3d" &&
      this.detailFocus.value === SpatialSkeletonDetailFocus.LOCAL &&
      selectedSources.length > 1
    ) {
      const transformedBySourceId = new Map<string, TransformedSource>();
      for (const scales of transformedSources) {
        for (const tsource of scales) {
          transformedBySourceId.set(getObjectId(tsource.source), tsource);
        }
      }
      const metersPerUnit = this.getMetersPerUnit(projectionParameters);
      const levelSpacings = this.getLevelSpacingsMeters();
      const arbitrationCandidates = selectedSources
        .map((source, fallbackRank) => {
          const sourceId = getObjectId(source.chunkSource);
          const transformed = transformedBySourceId.get(sourceId);
          if (transformed === undefined) return undefined;
          const spacing = this.getChunkSpacing(transformed.chunkLayout);
          // Published level spacing in preference to chunk shape, and for the
          // same reason the backend's copy of this does it: on an
          // object-sparsity pyramid every level shares one chunk_shape, so
          // chunk-derived spacings all tie and every cell falls through to the
          // selected level. The two sides must agree, or the view draws a level
          // the priorities never fetched.
          const gridIndex = getSpatiallyIndexedSkeletonGridIndex(
            source.chunkSource,
          );
          const published =
            gridIndex === undefined ? undefined : levelSpacings[gridIndex];
          return {
            fallbackRank,
            sourceId,
            transformed,
            spacing,
            spacingMeters:
              published !== undefined && published > 0
                ? published
                : spacing * metersPerUnit,
          };
        })
        .filter((x): x is NonNullable<typeof x> => x !== undefined);
      if (arbitrationCandidates.length > 1) {
        const worldCenter = new Float32Array(3);
        const candidateChunkPosition = new Float32Array(3);
        const targetSpacingMeters =
          this.getArbitrationTargetSpacingMeters3d(projectionParameters);
        const referencePixelSize =
          this.getReferencePixelSize(projectionParameters);
        // Anchor the position-enumeration grid on whichever candidate's
        // spacing is CLOSEST to the desired target, not unconditionally the
        // finest level -- same bug (and same fix) as
        // recomputeChunkPriorities's grid-anchor arbitration in
        // skeleton/backend.ts: anchoring on the finest level unconditionally
        // makes this walk the finest level's cell density across the whole
        // visible frustum even for a coarse/zoomed-out view.
        const anchor = arbitrationCandidates.reduce((best, candidate) =>
          Math.abs(candidate.spacingMeters - targetSpacingMeters) <
          Math.abs(best.spacingMeters - targetSpacingMeters)
            ? candidate
            : best,
        );
        const emittedChunkSlots = new Set<string>();
        let shouldContinue = true;
        forEachVisibleVolumetricChunk(
          projectionParameters,
          this.localPosition.value,
          anchor.transformed,
          (anchorPositionInChunks) => {
            if (!shouldContinue) return;
            this.getChunkCenterWorld(
              anchor.transformed.chunkLayout,
              anchorPositionInChunks,
              worldCenter,
            );
            const chunkPixelSize = computePhysicalUnitsPerScreenPixel(
              projectionParameters.viewProjectionMat,
              projectionParameters.width,
              projectionParameters.height,
              worldCenter,
              projectionParameters.displayDimensionRenderInfo
                ?.displayDimensionScales,
            );
            const desiredSpacingRaw =
              Number.isFinite(chunkPixelSize) && chunkPixelSize > 0
                ? targetSpacingMeters * (chunkPixelSize / referencePixelSize)
                : targetSpacingMeters;
            const desiredSpacing =
              this.quantizeSpacingForArbitration(desiredSpacingRaw);
            const orderedCandidates = [...arbitrationCandidates].sort(
              (a, b) => {
                const da = Math.abs(a.spacingMeters - desiredSpacing);
                const db = Math.abs(b.spacingMeters - desiredSpacing);
                if (da !== db) return da - db;
                return a.fallbackRank - b.fallbackRank;
              },
            );

            let selected:
              | {
                  candidate: (typeof orderedCandidates)[number];
                  chunkKey: string;
                  state: ChunkState | undefined;
                }
              | undefined;
            for (const candidate of orderedCandidates) {
              if (
                !this.getChunkGridPositionForWorldPoint(
                  candidate.transformed,
                  worldCenter,
                  candidateChunkPosition,
                )
              ) {
                continue;
              }
              const chunkKey = `${candidateChunkPosition.join()}${SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR}`;
              const chunk = (
                candidate.transformed.source as SpatiallyIndexedSkeletonSource
              ).chunks.get(chunkKey);
              const state = chunk?.state;
              if (state === ChunkState.GPU_MEMORY) {
                selected = { candidate, chunkKey, state };
                break;
              }
              if (selected === undefined) {
                selected = { candidate, chunkKey, state };
              }
            }
            if (selected === undefined) {
              return;
            }
            const emitKey = `${selected.candidate.sourceId}|${selected.chunkKey}`;
            if (emittedChunkSlots.has(emitKey)) {
              return;
            }
            emittedChunkSlots.add(emitKey);
            if (
              callback(
                selected.chunkKey,
                selected.candidate.transformed
                  .source as SpatiallyIndexedSkeletonSource,
                selected.candidate.transformed.chunkLayout,
              ) === false
            ) {
              shouldContinue = false;
            }
          },
        );
        return;
      }
    }

    let shouldContinue = true;
    for (const scales of transformedSources) {
      for (const tsource of scales) {
        if (!shouldContinue) return;
        if (!drawSourceIds.has(getObjectId(tsource.source))) continue;
        const emit = (positionInChunks: Float32Array) => {
          if (!shouldContinue) return;
          const chunkKey = `${positionInChunks.join()}${SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR}`;
          if (
            callback(
              chunkKey,
              tsource.source as SpatiallyIndexedSkeletonSource,
              tsource.chunkLayout,
            ) === false
          ) {
            shouldContinue = false;
          }
        };
        // OBJECT focus draws the WHOLE VOLUME, not the frustum -- the same
        // enumeration the worker requests with, so the two agree cell for cell.
        // Drawing only what the frustum reaches would make a complete tract
        // appear to end at the edge of the view and reappear on panning, which
        // is the opposite of what loading whole objects is for.
        const source = tsource.source as SpatiallyIndexedSkeletonSource;
        const wholeVolume =
          objectFocus &&
          forEachSpatialSkeletonVolumeCell(
            source.spec.lowerChunkBound,
            source.spec.upperChunkBound,
            emit,
          ) >= 0;
        if (!wholeVolume) {
          forEachVisibleVolumetricChunk(
            projectionParameters,
            this.localPosition.value,
            tsource,
            emit,
          );
        }
      }
    }
  }

  getVisibleChunksInCurrentViewAndLod(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: any,
    lod: number | undefined,
  ): VisibleChunk[] {
    if (lod === undefined) {
      return [];
    }
    const result: VisibleChunk[] = [];
    this.forEachVisibleChunkSlot(
      view,
      gridLevel,
      transformedSources,
      projectionParameters,
      (chunkKey, chunkSource, chunkLayout) => {
        const chunk = chunkSource.chunks.get(chunkKey);
        if (chunk?.state === ChunkState.GPU_MEMORY) {
          result.push({ chunk, chunkLayout });
        }
      },
    );
    return result;
  }

  /**
   * Populate the spatial-skeleton grid render-scale histogram using the
   * SAME visible-chunk enumeration the render path uses
   * (`forEachVisibleChunkSlot`), so the widget's bars reflect exactly the
   * chunks being drawn — present chunks fill the "present" (bright) colour
   * and grow the bar height as they load; not-yet-loaded slots contribute
   * to the dim "not-present" portion.
   *
   * This replaces an earlier implementation that ran a *parallel*
   * per-source `forEachVisibleVolumetricChunk` pass, which enumerated no
   * chunks in the perspective view (source arbitration is required) and so
   * left every bar an empty placeholder.  Placeholder bars are still added
   * for pyramid levels with no chunks in view, so all scales stay visible
   * in the widget.
   */
  updateGridRenderScaleHistogram(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
    lod: number | undefined,
    levels:
      | ReadonlyArray<{
          size: { x: number; y: number; z: number };
          lod: number;
          objectCount?: number;
        }>
      | undefined,
    histogram: RenderScaleHistogram,
    frameNumber: number,
  ) {
    histogram.begin(frameNumber);
    if (lod === undefined || transformedSources.length === 0) return;
    const spacingOf = (size: { x: number; y: number; z: number }) =>
      Math.max(Math.min(size.x, size.y, size.z), 1e-6);
    const perSpacing = new Map<number, { present: number; missing: number }>();
    this.forEachVisibleChunkSlot(
      view,
      gridLevel,
      transformedSources,
      projectionParameters,
      (chunkKey, chunkSource, chunkLayout) => {
        const gridIndex = getSpatiallyIndexedSkeletonGridIndex(chunkSource);
        // Prefer the level's physical (meters) spacing — the same value the
        // histogram axis is calibrated against — falling back to the raw
        // chunk-layout spacing only when the level lookup is unavailable.
        const levelSize =
          gridIndex !== undefined ? levels?.[gridIndex]?.size : undefined;
        const spacing =
          levelSize !== undefined
            ? spacingOf(levelSize)
            : this.getChunkSpacing(chunkLayout);
        let entry = perSpacing.get(spacing);
        if (entry === undefined) {
          entry = { present: 0, missing: 0 };
          perSpacing.set(spacing, entry);
        }
        const chunk = chunkSource.chunks.get(chunkKey);
        if (chunk?.state === ChunkState.GPU_MEMORY) {
          entry.present++;
        } else {
          entry.missing++;
        }
      },
    );
    // When the source can say how many objects each level holds, size the bars
    // by that instead of by chunk-slot demand. For a tractogram, "how big is
    // this level" means its streamline count -- a pyramid running ~503k / 50k /
    // 5k / 503 / 50 is describing sparsity, and that is what the user is
    // choosing between. Chunk counts answer a different question (how much grid
    // is in view) and only ever existed for the one level being drawn, which
    // left every other bar an identical stub.
    const objectCounts = levels?.some((l) => l.objectCount !== undefined)
      ? levels
      : undefined;
    if (objectCounts !== undefined) {
      for (const level of objectCounts) {
        const spacing = spacingOf(level.size);
        const count = level.objectCount ?? 0;
        // The drawn level counts as present, the rest as not-present, so the
        // bar for what is on screen reads differently from the alternatives.
        const drawn = perSpacing.has(spacing);
        histogram.add(spacing, spacing, drawn ? count : 0, drawn ? 0 : count);
      }
      return;
    }

    for (const [spacing, { present, missing }] of perSpacing) {
      histogram.add(spacing, spacing, present, missing);
    }
    // No per-level object counts: fall back to chunk slots. Only the selected
    // level's are enumerated, so the others are derived -- levels tile the same
    // viewed volume, so slots scale with the inverse cube of chunk spacing.
    // Scaling one measured count is O(levels) per frame, where enumerating each
    // level's slots would repeat the source selection and arbitration above for
    // every one of them. Flagged render-only so an estimate is never counted as
    // real chunk demand.
    let reference: { spacing: number; count: number } | undefined;
    for (const [spacing, { present, missing }] of perSpacing) {
      const count = present + missing;
      if (
        count > 0 &&
        (reference === undefined || spacing < reference.spacing)
      ) {
        reference = { spacing, count };
      }
    }
    for (const level of levels ?? []) {
      const spacing = spacingOf(level.size);
      if (histogram.spatialScales.has(spacing)) continue;
      const estimate =
        reference === undefined
          ? 1
          : Math.max(
              1,
              Math.round(reference.count * (reference.spacing / spacing) ** 3),
            );
      histogram.add(spacing, spacing, 0, estimate, true);
    }
  }

  private areVisibleChunksReady(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
    lod: number | undefined,
  ) {
    if (
      this.displayState.objectAlpha.value <= 0.0 &&
      this.displayState.hiddenObjectAlpha.value <= 0.0
    ) {
      return true;
    }
    if (lod === undefined) {
      // No LOD configured — draw() renders nothing in this case, so nothing to wait for.
      return true;
    }
    if (transformedSources.length === 0) {
      return false;
    }
    let ready = true;
    this.forEachVisibleChunkSlot(
      view,
      gridLevel,
      transformedSources,
      projectionParameters,
      (chunkKey, chunkSource, _) => {
        const chunk = chunkSource.chunks.get(chunkKey);
        if (chunk?.state !== ChunkState.GPU_MEMORY) {
          ready = false;
          return false;
        }
        return true;
      },
    );
    return ready;
  }

  getNode(
    nodeId: number,
    options: {
      lod?: number;
    } = {},
  ): SpatiallyIndexedSkeletonNode | undefined {
    void options.lod;
    if (!Number.isSafeInteger(nodeId) || nodeId <= 0) return undefined;
    return this.getCachedNodeSnapshot(nodeId);
  }

  getNodes(
    options: {
      segmentId?: bigint;
      lod?: number;
    } = {},
  ): SpatiallyIndexedSkeletonNode[] {
    void options.lod;
    const normalizedSegmentFilter =
      options.segmentId === undefined
        ? undefined
        : Math.round(Number(options.segmentId));
    const useSegmentFilter =
      normalizedSegmentFilter !== undefined &&
      Number.isFinite(normalizedSegmentFilter);
    const segmentIds =
      normalizedSegmentFilter === undefined
        ? this.getActiveEditableSegmentIds()
        : [normalizedSegmentFilter];
    const nodes = new Map<number, SpatiallyIndexedSkeletonNode>();
    for (const segmentId of segmentIds) {
      const segmentNodes =
        this.inspectionState?.getCachedSegmentNodes(segmentId) ?? [];
      for (const node of segmentNodes) {
        if (nodes.has(node.nodeId)) continue;
        const cachedNode = this.getCachedNodeSnapshot(node.nodeId);
        if (cachedNode === undefined) continue;
        if (
          useSegmentFilter &&
          normalizedSegmentFilter !== undefined &&
          cachedNode.segmentId !== normalizedSegmentFilter
        ) {
          continue;
        }
        nodes.set(cachedNode.nodeId, cachedNode);
      }
    }
    return [...nodes.values()].sort((a, b) => a.nodeId - b.nodeId);
  }

  private beginSkeletonRenderPass(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    renderHelper: RenderHelper,
    modelMatrix: mat4,
    lineWidth: number,
    pointDiameter: number,
    excludedGPUTable?: GPUHashTable<HashSetUint64>,
  ):
    | {
        gl: GL;
        edgeShader: ShaderProgram;
        nodeShader: ShaderProgram;
        faceShader: ShaderProgram | undefined;
        skeletonParams: SkeletonShaderParameters;
      }
    | undefined {
    const { gl } = this;
    const edgeShaderResult = renderHelper.edgeShaderGetter(
      renderContext.emitter,
    );
    const nodeShaderResult = renderHelper.nodeShaderGetter(
      renderContext.emitter,
    );
    const {
      shader: edgeShader,
      parameters: edgeShaderParameters,
      extraParameters: skeletonParams,
    } = edgeShaderResult;
    const { shader: nodeShader, parameters: nodeShaderParameters } =
      nodeShaderResult;
    if (edgeShader === null || nodeShader === null) return undefined;

    const { shaderControlState } = this.displayState.skeletonRenderingOptions;

    edgeShader.bind();
    renderHelper.beginLayer(gl, edgeShader, renderContext, modelMatrix);
    gl.uniform1f(edgeShader.uniform("uLineWidth"), lineWidth);
    renderHelper.setPickInstanceStride(gl, edgeShader, 0);
    setControlsInShader(
      gl,
      edgeShader,
      shaderControlState,
      edgeShaderParameters.parseResult.controls,
    );
    renderHelper.setColor(gl, edgeShader, kOneVec4);
    renderHelper.maybeEnableDynamicSegmentAppearance(
      gl,
      edgeShader,
      skeletonParams,
      excludedGPUTable,
    );

    nodeShader.bind();
    renderHelper.beginLayer(gl, nodeShader, renderContext, modelMatrix);
    gl.uniform1f(nodeShader.uniform("uNodeDiameter"), pointDiameter);
    renderHelper.setPickInstanceStride(gl, nodeShader, 0);
    setControlsInShader(
      gl,
      nodeShader,
      shaderControlState,
      nodeShaderParameters.parseResult.controls,
    );
    renderHelper.setColor(gl, nodeShader, kOneVec4);
    renderHelper.maybeEnableDynamicSegmentAppearance(
      gl,
      nodeShader,
      skeletonParams,
      excludedGPUTable,
    );

    // Surface geometry only: a third program, compiled on first use.
    let faceShader: ShaderProgram | undefined;
    if (this.geometryPrimitive === "triangles") {
      const faceShaderResult = renderHelper.faceShaderGetter(
        renderContext.emitter,
      );
      const { shader, parameters: faceShaderParameters } = faceShaderResult;
      if (shader === null) return undefined;
      faceShader = shader;
      faceShader.bind();
      renderHelper.beginLayer(gl, faceShader, renderContext, modelMatrix);
      renderHelper.setPickInstanceStride(gl, faceShader, 0);
      setControlsInShader(
        gl,
        faceShader,
        shaderControlState,
        faceShaderParameters.parseResult.controls,
      );
      renderHelper.setColor(gl, faceShader, kOneVec4);
      renderHelper.setLightDirection(
        gl,
        faceShader,
        renderContext,
        modelMatrix,
      );
      renderHelper.maybeEnableDynamicSegmentAppearance(
        gl,
        faceShader,
        skeletonParams,
        excludedGPUTable,
      );
    }

    return { gl, edgeShader, nodeShader, faceShader, skeletonParams };
  }

  private endSkeletonRenderPass(
    renderHelper: RenderHelper,
    gl: GL,
    edgeShader: ShaderProgram,
    nodeShader: ShaderProgram,
    skeletonParams: SkeletonShaderParameters,
    faceShader?: ShaderProgram,
  ) {
    renderHelper.maybeDisableDynamicSegmentAppearance(
      gl,
      edgeShader,
      skeletonParams,
    );
    renderHelper.maybeDisableDynamicSegmentAppearance(
      gl,
      nodeShader,
      skeletonParams,
    );
    if (faceShader !== undefined) {
      renderHelper.maybeDisableDynamicSegmentAppearance(
        gl,
        faceShader,
        skeletonParams,
      );
    }
    renderHelper.endLayer(gl, edgeShader, nodeShader, faceShader ?? null);
  }

  private drawBrowsePass(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    renderHelper: RenderHelper,
    modelMatrix: mat4,
    lineWidth: number,
    pointDiameter: number,
    visibleChunks: VisibleChunk[],
    renderMode: SkeletonRenderMode = SkeletonRenderMode.LINES_AND_POINTS,
  ) {
    if (visibleChunks.length === 0) return;
    const hasExcludedSegments =
      this.getBrowsePassExcludedSegments() !== undefined;
    const passState = this.beginSkeletonRenderPass(
      renderContext,
      renderHelper,
      modelMatrix,
      lineWidth,
      pointDiameter,
      hasExcludedSegments ? this.gpuBrowseExcludedSegmentsHashTable : undefined,
    );
    if (passState === undefined) return;
    const { gl, edgeShader, nodeShader, faceShader, skeletonParams } =
      passState;

    const chunkOrigin = vec3.create();
    const chunkBound = vec3.create();
    for (const { chunk, chunkLayout } of visibleChunks) {
      if (skeletonParams.spatialChunkCulling) {
        vec3.mul(chunkOrigin, chunk.chunkGridPosition, chunkLayout.size);
        vec3.add(chunkBound, chunkOrigin, chunkLayout.size);
        if (faceShader !== undefined) {
          faceShader.bind();
          renderHelper.setChunkBounds(gl, faceShader, chunkOrigin, chunkBound);
        }
        edgeShader.bind();
        renderHelper.setChunkBounds(gl, edgeShader, chunkOrigin, chunkBound);
        nodeShader.bind();
        renderHelper.setChunkBounds(gl, nodeShader, chunkOrigin, chunkBound);
      }
      if (faceShader !== undefined) {
        // Surface geometry: `chunk.indices` holds triangles, so one instanced
        // triangle per face and one pick id per face -- `gl_InstanceID` is the
        // face number.
        if (renderContext.emitPickID) {
          let facePickId = 0;
          let facePickStride = 0;
          if (chunk.numIndices > 0) {
            facePickId = renderContext.pickIDs.register(
              layer,
              chunk.numIndices / 3,
              0n,
              {
                kind: "segment-face",
                chunk,
              } satisfies SpatiallyIndexedSkeletonPickData,
            );
            facePickStride = 1;
          }
          faceShader.bind();
          renderHelper.setPickID(gl, faceShader, facePickId);
          renderHelper.setPickInstanceStride(gl, faceShader, facePickStride);
        }
        renderHelper.drawTriangles(gl, faceShader, chunk);
        continue;
      }
      if (renderContext.emitPickID) {
        let edgePickId = 0;
        let edgePickStride = 0;
        let nodePickId = 0;
        let nodePickStride = 0;
        if (chunk.numIndices > 0) {
          edgePickId = renderContext.pickIDs.register(
            layer,
            chunk.numIndices / 2,
            0n,
            {
              kind: "segment-edge",
              chunk,
            } satisfies SpatiallyIndexedSkeletonPickData,
          );
          edgePickStride = 1;
        }
        if (chunk.numVertices > 0) {
          nodePickId = renderContext.pickIDs.register(
            layer,
            chunk.numVertices,
            0n,
            {
              kind: "segment-node",
              chunk,
            } satisfies SpatiallyIndexedSkeletonPickData,
          );
          nodePickStride = 1;
        }
        edgeShader.bind();
        renderHelper.setPickID(gl, edgeShader, edgePickId);
        renderHelper.setPickInstanceStride(gl, edgeShader, edgePickStride);
        nodeShader.bind();
        renderHelper.setPickID(gl, nodeShader, nodePickId);
        renderHelper.setPickInstanceStride(gl, nodeShader, nodePickStride);
      }
      // Render each chunk with different node/edge colors for debugging
      if (DEBUG_SPATIAL_SKELETON_CHUNKS) {
        const chunkKey = `${chunk.chunkGridPosition[0]},${chunk.chunkGridPosition[1]},${chunk.chunkGridPosition[2]}`;
        let randomColor = tempChunkKeyToColorMap.get(chunkKey);
        if (randomColor === undefined) {
          // Use same strategy as segment color hashing to be consistent
          // in colors across neuroglancer sessions
          randomColor = new Float32Array([0, 0, 0]);
          let h = hashCombine(0, chunk.chunkGridPosition[0]);
          h = hashCombine(h, chunk.chunkGridPosition[1]);
          h = hashCombine(h, chunk.chunkGridPosition[2]);
          const c0 = (h & 0xff) / 255;
          const c1 = ((h >> 8) & 0xff) / 255;
          hsvToRgb(randomColor, c0, 0.5 + 0.5 * c1, 1.0);
          tempChunkKeyToColorMap.set(chunkKey, randomColor);
        }
        if (skeletonParams.hasSegmentDefaultColor) {
          nodeShader.bind();
          gl.uniform3fv(
            nodeShader.uniform("uSegmentDefaultColor"),
            randomColor,
          );
          edgeShader.bind();
          gl.uniform3fv(
            edgeShader.uniform("uSegmentDefaultColor"),
            randomColor,
          );
        }
      }
      renderHelper.drawSkeletons(
        gl,
        edgeShader,
        nodeShader,
        chunk,
        renderContext.projectionParameters,
        renderMode,
        this.geometryPrimitive,
      );
    }
    this.endSkeletonRenderPass(
      renderHelper,
      gl,
      edgeShader,
      nodeShader,
      skeletonParams,
      faceShader,
    );
  }

  private drawInspectionOverlayPass(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    renderHelper: RenderHelper,
    modelMatrix: mat4,
    lineWidth: number,
    pointDiameter: number,
    renderMode: SkeletonRenderMode = SkeletonRenderMode.LINES_AND_POINTS,
  ) {
    const overlayChunk = this.resolveSourceBackedOverlayChunk();
    if (overlayChunk === undefined) return;
    const passState = this.beginSkeletonRenderPass(
      renderContext,
      renderHelper,
      modelMatrix,
      lineWidth,
      pointDiameter,
    );
    if (passState === undefined) return;
    const { gl, edgeShader, nodeShader, skeletonParams } = passState;

    if (renderContext.emitPickID) {
      const edgePickId =
        overlayChunk.numIndices > 0 &&
        overlayChunk.pickEdgeSegmentIds !== undefined &&
        overlayChunk.pickEdgeSegmentIds.length > 0
          ? renderContext.pickIDs.register(
              layer,
              overlayChunk.pickEdgeSegmentIds.length,
              0n,
              {
                kind: "edge",
                segmentIds: overlayChunk.pickEdgeSegmentIds,
              } satisfies SpatiallyIndexedSkeletonPickData,
            )
          : 0;
      edgeShader.bind();
      renderHelper.setPickID(gl, edgeShader, edgePickId);
      renderHelper.setPickInstanceStride(
        gl,
        edgeShader,
        edgePickId === 0 ? 0 : 1,
      );

      const nodePickId =
        overlayChunk.numVertices > 0 &&
        overlayChunk.pickNodeIds !== undefined &&
        overlayChunk.pickNodePositions !== undefined &&
        overlayChunk.pickSegmentIds !== undefined
          ? renderContext.pickIDs.register(
              layer,
              overlayChunk.numVertices,
              0n,
              {
                kind: "node",
                nodeIds: overlayChunk.pickNodeIds,
                nodePositions: overlayChunk.pickNodePositions,
                segmentIds: overlayChunk.pickSegmentIds,
              } satisfies SpatiallyIndexedSkeletonPickData,
            )
          : 0;
      nodeShader.bind();
      renderHelper.setPickID(gl, nodeShader, nodePickId);
      renderHelper.setPickInstanceStride(
        gl,
        nodeShader,
        nodePickId === 0 ? 0 : 1,
      );
    }

    renderHelper.drawSkeletons(
      gl,
      edgeShader,
      nodeShader,
      overlayChunk,
      renderContext.projectionParameters,
      renderMode,
      this.geometryPrimitive,
    );
    this.endSkeletonRenderPass(
      renderHelper,
      gl,
      edgeShader,
      nodeShader,
      skeletonParams,
    );
  }

  draw(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    overlayRenderHelper: RenderHelper,
    browseRenderHelper: RenderHelper,
    renderOptions: ViewSpecificSkeletonRenderingOptions,
    modelMatrix: mat4,
    visibleChunks: VisibleChunk[],
  ) {
    const { displayState } = this;
    if (
      displayState.objectAlpha.value <= 0.0 &&
      displayState.hiddenObjectAlpha.value <= 0.0
    ) {
      return;
    }

    const lineWidth = renderOptions.lineWidth.value;
    const renderMode = renderOptions.mode.value;
    // Point geometry always draws its vertices at the larger dot size: there is
    // no line for the "lines" preference to fall back to.
    const pointDiameter =
      this.geometryPrimitive === "points"
        ? Math.max(5, lineWidth * 2)
        : getSkeletonNodeDiameter(renderMode, lineWidth);

    this.drawBrowsePass(
      renderContext,
      layer,
      browseRenderHelper,
      modelMatrix,
      lineWidth,
      pointDiameter,
      visibleChunks,
      renderMode,
    );
    this.drawInspectionOverlayPass(
      renderContext,
      layer,
      overlayRenderHelper,
      modelMatrix,
      lineWidth,
      pointDiameter,
      renderMode,
    );
  }

  isReady(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
    lod?: number,
  ) {
    return this.areVisibleChunksReady(
      view,
      gridLevel,
      transformedSources,
      projectionParameters,
      lod,
    );
  }
}

function transformSpatiallyIndexedSkeletonPickedValue(
  pickState: PickState,
): bigint | undefined {
  const u64 = pickState.pickedSpatialSkeleton?.segmentIdU64;
  return typeof u64 === "bigint" && u64 > 0n ? u64 : undefined;
}

const MAX_SAFE_SEGMENT_ID = BigInt(Number.MAX_SAFE_INTEGER);

/**
 * Split a full uint64 segment id into the dual representation stored on
 * `PickedSpatialSkeletonState`: the `bigint` for the selection widget, and a
 * safe-integer `number` for the legacy edit/overlay tooling (undefined when
 * the id exceeds 2⁵³, e.g. flywire ids).
 */
function pickedSegmentIdFields(u64: bigint): {
  segmentIdU64: bigint;
  segmentId?: number;
} {
  return {
    segmentIdU64: u64,
    segmentId: u64 <= MAX_SAFE_SEGMENT_ID ? Number(u64) : undefined,
  };
}

function updateSpatiallyIndexedSkeletonMouseState(
  base: SpatiallyIndexedSkeletonLayer,
  mouseState: MouseSelectionState,
  pickedOffset: number,
  data: SpatiallyIndexedSkeletonPickData | undefined,
): void {
  if (data === undefined) return;
  if (data.kind === "node") {
    if (
      pickedOffset < 0 ||
      pickedOffset >= data.nodeIds.length ||
      pickedOffset >= data.segmentIds.length
    ) {
      return;
    }
    const rawSegmentId = data.segmentIds[pickedOffset];
    if (!Number.isSafeInteger(rawSegmentId) || rawSegmentId <= 0) {
      return;
    }
    const segmentId = BigInt(rawSegmentId);
    mouseState.pickedSpatialSkeleton = pickedSegmentIdFields(segmentId);
    if (
      !getVisibleSegments(base.displayState.segmentationGroupState.value).has(
        segmentId,
      )
    ) {
      return;
    }
    const nodeId = data.nodeIds[pickedOffset];
    if (!Number.isSafeInteger(nodeId) || nodeId <= 0) return;
    const nodePosition = data.nodePositions.subarray(
      pickedOffset * 3,
      pickedOffset * 3 + 3,
    );
    mouseState.pickedSpatialSkeleton = {
      nodeId,
      ...pickedSegmentIdFields(segmentId),
      position: new Float32Array(nodePosition),
    };
    const transform = base.displayState.transform.value;
    if (transform.error === undefined) {
      setMouseStatePositionFromSpatialSkeletonNode(
        mouseState,
        nodePosition,
        transform,
      );
    }
    return;
  }
  if (data.kind === "edge") {
    if (pickedOffset < 0 || pickedOffset >= data.segmentIds.length) {
      return;
    }
    const rawSegmentId = data.segmentIds[pickedOffset];
    if (Number.isSafeInteger(rawSegmentId) && rawSegmentId > 0) {
      mouseState.pickedSpatialSkeleton = pickedSegmentIdFields(
        BigInt(rawSegmentId),
      );
    }
    return;
  }
  if (
    data.kind === "segment-node" ||
    data.kind === "segment-edge" ||
    data.kind === "segment-face"
  ) {
    if (data.kind === "segment-node") {
      const pickedNode = base.resolveNodePickFromChunk(
        data.chunk,
        pickedOffset,
      );
      if (pickedNode !== undefined) {
        mouseState.pickedSpatialSkeleton = {
          nodeId: pickedNode.nodeId,
          ...pickedSegmentIdFields(pickedNode.segmentId),
          position: new Float32Array(pickedNode.position),
          sourceState: pickedNode.sourceState,
        };
        return;
      }
      // No node identity: `resolveNodePickFromChunk` needs `chunk.nodeIds`,
      // which only the CATMAID backend populates. Everything else -- notably a
      // zarr-vectors point cloud, whose vertices are ALL that is drawn and whose
      // decoder gives each one its own segment id -- would otherwise pick
      // nothing at all. The vertex still knows its segment, so fall through and
      // report that.
      const nodeSegmentId = base.resolveSegmentPickFromChunk(
        data.chunk,
        pickedOffset,
        "node",
      );
      if (nodeSegmentId !== undefined) {
        mouseState.pickedSpatialSkeleton = pickedSegmentIdFields(nodeSegmentId);
      }
      return;
    }
    const segmentId = base.resolveSegmentPickFromChunk(
      data.chunk,
      pickedOffset,
      data.kind === "segment-face" ? "face" : "edge",
    );
    if (segmentId !== undefined) {
      mouseState.pickedSpatialSkeleton = pickedSegmentIdFields(segmentId);
    }
  }
}

function attachSpatiallyIndexedSkeletonLayer(
  base: SpatiallyIndexedSkeletonLayer,
  renderLayer: {
    transformedSources: TransformedSource[][];
    redrawNeeded: NullarySignal;
  },
  attachment: VisibleLayerInfo<
    LayerView,
    ThreeDimensionalRenderLayerAttachmentState
  >,
  view: "2d" | "3d",
): void {
  const { redrawNeeded } = renderLayer;
  attachment.registerDisposer(
    registerNested(
      (context, transform, displayDimensionRenderInfo) => {
        const transformedSources = getVolumetricTransformedSources(
          displayDimensionRenderInfo,
          transform,
          () => [
            base.getSources(view).map((sourceEntry) => ({
              chunkSource: sourceEntry.chunkSource,
              chunkToMultiscaleTransform:
                sourceEntry.chunkToMultiscaleTransform,
            })),
          ],
          attachment.messages,
          renderLayer,
        );
        for (const scales of transformedSources) {
          for (const tsource of scales) {
            context.registerDisposer(tsource.source);
          }
        }
        attachment.view.flushBackendProjectionParameters();
        renderLayer.transformedSources = transformedSources;
        base.rpc!.invoke(
          SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
          {
            layer: base.backend.rpcId,
            view: attachment.view.rpcId,
            displayDimensionRenderInfo,
            sources: serializeAllTransformedSources(transformedSources),
          },
        );
        redrawNeeded.dispatch();
        return transformedSources;
      },
      base.displayState.transform,
      attachment.view.displayDimensionRenderInfo,
    ),
  );
}

export class PerspectiveViewSpatiallyIndexedSkeletonLayer extends PerspectiveViewRenderLayer {
  private renderHelper: RenderHelper;
  private browseRenderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  transformedSources: TransformedSource[][] = [];
  backend: ChunkRenderLayerFrontend;

  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super();
    this.backend = base.backend;
    this.renderHelper = this.registerDisposer(new RenderHelper(base, false));
    this.browseRenderHelper = this.registerDisposer(
      new RenderHelper(base.browsePassLayerView, false),
    );
    this.renderOptions = base.displayState.skeletonRenderingOptions.params3d;

    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    const histogram3d =
      base.displayState.spatialSkeletonGridRenderScaleHistogram3d;
    if (histogram3d !== undefined) {
      this.registerDisposer(histogram3d.visibility.add(this.visibility));
    }
  }

  attach(
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    super.attach(attachment);
    attachSpatiallyIndexedSkeletonLayer(this.base, this, attachment, "3d");
  }

  get gl() {
    return this.base.gl;
  }

  get isTransparent() {
    const { objectAlpha, hiddenObjectAlpha } = this.base.displayState;
    const opaque =
      (objectAlpha.value == 1.0 &&
        (hiddenObjectAlpha.value == 1.0 || hiddenObjectAlpha.value == 0.0)) ||
      (objectAlpha.value == 0.0 && hiddenObjectAlpha.value == 1.0);
    return !opaque;
  }

  getValueAt(_position: Float32Array) {
    return undefined;
  }

  transformPickedValue(pickState: PickState) {
    return transformSpatiallyIndexedSkeletonPickedValue(pickState);
  }

  updateMouseState(
    mouseState: MouseSelectionState,
    _pickedValue: bigint,
    pickedOffset: number,
    data: unknown,
  ) {
    updateSpatiallyIndexedSkeletonMouseState(
      this.base,
      mouseState,
      pickedOffset,
      data as SpatiallyIndexedSkeletonPickData | undefined,
    );
  }

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    if (!renderContext.emitColor && renderContext.alreadyEmittedPickID) {
      return;
    }
    const { displayState } = this.base;
    // Auto-LOD: refresh the resolution target from the current camera
    // projection before chunk selection, so the picker tracks zoom
    // (same path the manual slider takes).  Opt-in via
    // `autoSpatialSkeletonGridLevel3d` to preserve existing
    // user-driven behaviour for layers that don't want it.
    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      renderContext.projectionParameters,
      this.base.localPosition.value,
      "3d",
      // Only where this target decides the level. Under LOCAL focus it always
      // does; under OBJECT focus the memory budget decides instead -- but only
      // on a store that can be budgeted per object, which is the same store
      // property that lets the levels be drawn as a union. Everywhere else
      // OBJECT focus leaves level selection to the camera, and skipping this
      // would freeze the level at whatever was picked first.
      this.base.detailFocus.value === SpatialSkeletonDetailFocus.LOCAL ||
        !this.base.objectPartitionAvailable("3d")
        ? this.base.countVisibleCells(
            "3d",
            this.transformedSources,
            renderContext.projectionParameters,
          )
        : undefined,
    );
    const lodValue = displayState.skeletonLod?.value;
    const visibleChunks = this.base.getVisibleChunksInCurrentViewAndLod(
      "3d",
      displayState.spatialSkeletonGridLevel3d?.value,
      this.transformedSources,
      renderContext.projectionParameters,
      lodValue,
    );
    const levels = displayState.spatialSkeletonGridLevels?.value;
    const histogram = displayState.spatialSkeletonGridRenderScaleHistogram3d;
    if (histogram !== undefined) {
      const frameNumber =
        this.base.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber;
      this.base.updateGridRenderScaleHistogram(
        "3d",
        displayState.spatialSkeletonGridLevel3d?.value,
        this.transformedSources,
        renderContext.projectionParameters,
        lodValue,
        levels,
        histogram,
        frameNumber,
      );
    }
    const modelMatrix = update3dRenderLayerAttachment(
      displayState.transform.value,
      renderContext.projectionParameters.displayDimensionRenderInfo,
      attachment,
    );
    if (modelMatrix === undefined) return;
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.browseRenderHelper,
      this.renderOptions,
      modelMatrix,
      visibleChunks,
    );
    if (renderContext.wireFrame) {
      this.drawChunkBoundsWireframe(renderContext, visibleChunks, modelMatrix);
    }
  }

  private drawChunkBoundsWireframe(
    renderContext: PerspectiveViewRenderContext,
    visibleChunks: VisibleChunk[],
    modelMatrix?: mat4,
  ) {
    if (
      visibleChunks.length === 0 ||
      !renderContext.emitColor ||
      modelMatrix === undefined
    )
      return;

    const { gl } = this.base;
    const wireframeHelper = ChunkWireframeHelper.get(gl);
    const shader = wireframeHelper.getShader(renderContext.emitter);
    shader.bind();
    const { viewProjectionMat } = renderContext.projectionParameters;

    mat4.multiply(tempMat4, viewProjectionMat, modelMatrix);
    gl.uniformMatrix4fv(shader.uniform("uChunkToClip"), false, tempMat4);

    for (const { chunk, chunkLayout } of visibleChunks) {
      wireframeHelper.setChunkUniforms(
        gl,
        shader,
        chunkLayout,
        chunk.chunkGridPosition,
      );
      drawBoxEdges(gl, 1, 1);
    }
  }

  isReady(
    renderContext: PerspectiveViewReadyRenderContext,
    _attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const { displayState } = this.base;
    return this.base.isReady(
      "3d",
      displayState.spatialSkeletonGridLevel3d?.value,
      this.transformedSources,
      renderContext.projectionParameters,
      displayState.skeletonLod?.value,
    );
  }
}

export class SliceViewPanelSpatiallyIndexedSkeletonLayer extends SliceViewPanelRenderLayer {
  private renderHelper: RenderHelper;
  private browseRenderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  transformedSources: TransformedSource[][] = [];
  backend: ChunkRenderLayerFrontend;
  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super();
    this.backend = base.backend;
    this.renderHelper = this.registerDisposer(new RenderHelper(base, true));
    this.browseRenderHelper = this.registerDisposer(
      new RenderHelper(base.browsePassLayerView, true),
    );
    this.renderOptions = base.displayState.skeletonRenderingOptions.params2d;
    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    const { displayState: displayState2d } = base;
    const gridLevel2d = displayState2d.spatialSkeletonGridLevel2d;
    if (gridLevel2d?.changed) {
      this.registerDisposer(
        gridLevel2d.changed.add(this.redrawNeeded.dispatch),
      );
    }
    const lod2d = displayState2d.spatialSkeletonLod2d;
    if (lod2d?.changed) {
      this.registerDisposer(lod2d.changed.add(this.redrawNeeded.dispatch));
    }
    const histogram2d =
      displayState2d.spatialSkeletonGridRenderScaleHistogram2d;
    if (histogram2d !== undefined) {
      this.registerDisposer(histogram2d.visibility.add(this.visibility));
    }
  }

  get gl() {
    return this.base.gl;
  }

  getValueAt(_position: Float32Array) {
    return undefined;
  }

  transformPickedValue(pickState: PickState) {
    return transformSpatiallyIndexedSkeletonPickedValue(pickState);
  }

  updateMouseState(
    mouseState: MouseSelectionState,
    _pickedValue: bigint,
    pickedOffset: number,
    data: unknown,
  ) {
    updateSpatiallyIndexedSkeletonMouseState(
      this.base,
      mouseState,
      pickedOffset,
      data as SpatiallyIndexedSkeletonPickData | undefined,
    );
  }

  attach(
    attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    super.attach(attachment);
    attachSpatiallyIndexedSkeletonLayer(this.base, this, attachment, "2d");
  }

  draw(
    renderContext: SliceViewPanelRenderContext,
    attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const { displayState } = this.base;
    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      renderContext.sliceView.projectionParameters.value,
      this.base.localPosition.value,
      "2d",
      // Only where this target decides the level. Under LOCAL focus it always
      // does; under OBJECT focus the memory budget decides instead -- but only
      // on a store that can be budgeted per object, which is the same store
      // property that lets the levels be drawn as a union. Everywhere else
      // OBJECT focus leaves level selection to the camera, and skipping this
      // would freeze the level at whatever was picked first.
      this.base.detailFocus.value === SpatialSkeletonDetailFocus.LOCAL ||
        !this.base.objectPartitionAvailable("2d")
        ? this.base.countVisibleCells(
            "2d",
            this.transformedSources,
            renderContext.sliceView.projectionParameters.value,
          )
        : undefined,
    );
    const lodValue = displayState.spatialSkeletonLod2d?.value;
    const visibleChunks = this.base.getVisibleChunksInCurrentViewAndLod(
      "2d",
      displayState.spatialSkeletonGridLevel2d?.value,
      this.transformedSources,
      renderContext.sliceView.projectionParameters.value,
      lodValue,
    );
    const levels = displayState.spatialSkeletonGridLevels?.value;
    const histogram = displayState.spatialSkeletonGridRenderScaleHistogram2d;
    if (histogram !== undefined) {
      const frameNumber =
        this.base.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber;
      this.base.updateGridRenderScaleHistogram(
        "2d",
        displayState.spatialSkeletonGridLevel2d?.value,
        this.transformedSources,
        renderContext.sliceView.projectionParameters.value,
        lodValue,
        levels,
        histogram,
        frameNumber,
      );
    }
    const modelMatrix = update3dRenderLayerAttachment(
      displayState.transform.value,
      renderContext.projectionParameters.displayDimensionRenderInfo,
      attachment,
    );
    if (modelMatrix === undefined) return;
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.browseRenderHelper,
      this.renderOptions,
      modelMatrix,
      visibleChunks,
    );
  }

  isReady(
    renderContext: SliceViewPanelReadyRenderContext,
    _attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const { displayState } = this.base;
    return this.base.isReady(
      "2d",
      displayState.spatialSkeletonGridLevel2d?.value,
      this.transformedSources,
      renderContext.projectionParameters,
      displayState.spatialSkeletonLod2d?.value,
    );
  }
}

const emptyVertexAttributes = new Map<string, VertexAttributeInfo>();

function getAttributeTextureFormats(
  vertexAttributes: Map<string, VertexAttributeInfo>,
): TextureFormat[] {
  const attributeTextureFormats: TextureFormat[] = [
    vertexPositionTextureFormat,
  ];
  for (const info of vertexAttributes.values()) {
    attributeTextureFormats.push(
      computeTextureFormat(
        new TextureFormat(),
        info.dataType,
        info.numComponents,
      ),
    );
  }
  return attributeTextureFormats;
}

export type SkeletonSourceOptions = object;

export class SkeletonSource extends ChunkSource {
  private attributeTextureFormats_?: TextureFormat[];

  get attributeTextureFormats() {
    let attributeTextureFormats = this.attributeTextureFormats_;
    if (attributeTextureFormats === undefined) {
      attributeTextureFormats = this.attributeTextureFormats_ =
        getAttributeTextureFormats(this.vertexAttributes);
    }
    return attributeTextureFormats;
  }

  declare chunks: Map<string, SkeletonChunk>;
  getChunk(x: PackedSkeletonGeometry) {
    return new SkeletonChunk(this, x);
  }

  get vertexAttributes(): Map<string, VertexAttributeInfo> {
    return emptyVertexAttributes;
  }
}
