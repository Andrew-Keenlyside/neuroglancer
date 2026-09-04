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
import { Chunk, ChunkSource } from "#src/chunk_manager/frontend.js";
import type {
  RoiBackgroundUniforms,
  RoiGroupConfig,
  RoiLabelField,
  RoiObjectAttrColumn,
} from "#src/datasource/zarr-vectors/roi.js";

import type { HashMapUint64, HashSetUint64 } from "#src/gpu_hash/hash_table.js";
import { GPUHashTable, HashSetShaderManager } from "#src/gpu_hash/shader.js";
import type {
  LayerView,
  MouseSelectionState,
  VisibleLayerInfo,
} from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type { PerspectiveViewRenderContext } from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";

import type { RenderLayerTransform } from "#src/render_coordinate_transform.js";

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
  getObjectKey,
} from "#src/segmentation_display_state/base.js";
import type { SegmentationDisplayState3D } from "#src/segmentation_display_state/frontend.js";
import {
  forEachVisibleSegmentToDraw,
  registerRedrawWhenSegmentationDisplayState3DChanged,
  SegmentationLayerSharedObject,
} from "#src/segmentation_display_state/frontend.js";

import type { SpatialSkeletonSourceState } from "#src/skeleton/api.js";
import type { VertexAttributeInfo } from "#src/skeleton/base.js";

import { SKELETON_LAYER_RPC_ID } from "#src/skeleton/base.js";
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
import type { SpatiallyIndexedSkeletonChunk } from "#src/skeleton/spatial_frontend.js";
import type { ChunkLayout } from "#src/sliceview/chunk_layout.js";

import type { SliceViewPanel } from "#src/sliceview/panel.js";
import type { SliceViewPanelRenderContext } from "#src/sliceview/renderlayer.js";
import { SliceViewPanelRenderLayer } from "#src/sliceview/renderlayer.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { TrackableValue, WatchableValue } from "#src/trackable_value.js";
import type { Uint64Map } from "#src/uint64_map.js";
import { Uint64Set } from "#src/uint64_set.js";
import { gatherUpdate } from "#src/util/array.js";

import { DATA_TYPE_SIGNED, DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";

import {
  mat3,
  mat3FromMat4,
  mat4,
  scaleMat3Output,
  vec4,
} from "#src/util/geom.js";
import { verifyFinitePositiveFloat } from "#src/util/json.js";
import * as matrix from "#src/util/matrix.js";

import { NullarySignal } from "#src/util/signal.js";
import type { Trackable } from "#src/util/trackable.js";
import { CompoundTrackable } from "#src/util/trackable.js";
import { TrackableEnum } from "#src/util/trackable_enum.js";
import { glsl_getBoxEdgeVertexPosition } from "#src/webgl/bounding_box.js";
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
} from "#src/webgl/texture_access.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

const DEBUG_SPATIAL_SKELETON_OVERLAY = false;
const DEBUG_EXCLUDED_SEGMENTS = false;
export const DEBUG_SPATIAL_SKELETON_CHUNKS = false;
// Used for debugging chunks via a different color for each chunk
export const tempChunkKeyToColorMap = new Map<string, Float32Array>();

export const tempMat4 = mat4.create();
// Scratch for the per-object ROI group colour/opacity handed to setColor() in
// the legacy pass-2 draw loop (unpacked from roiSegmentColors).
const tempRoiColor = vec4.create();
export const OVERLAY_SELECTED_FLOAT_ZERO = new Float32Array([0]);
export const OVERLAY_SELECTED_FLOAT_ONE = new Float32Array([1]);
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

export interface VertexAttributeRenderInfo extends VertexAttributeInfo {
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

export interface VisibleChunk {
  chunk: SpatiallyIndexedSkeletonChunk;
  chunkLayout: ChunkLayout;
}

export interface SkeletonShaderParameters {
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

export interface SkeletonShaderContext {
  vertexAttributes: VertexAttributeRenderInfo[];
  gl: GL;
  fallbackShaderParameters: WatchableValue<ShaderControlsBuilderState>;
  displayState: SkeletonLayerDisplayState;
  skeletonShaderParameters: WatchableValueInterface<SkeletonShaderParameters>;
  segmentColorAttributeIndex?: number;
  /** See {@link PackedAttributeRange}; absent means one texture per attribute. */
  packedAttributeRange?: PackedAttributeRange;
}

export interface SkeletonGPUGeometry {
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

export interface PackedSkeletonGeometry {
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  nodeIds?: Int32Array;
  nodeSourceStates?: Array<SpatialSkeletonSourceState | undefined>;
}

export type SpatiallyIndexedSkeletonPickData =
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

export class RenderHelper extends RefCounted {
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
export class ChunkWireframeHelper extends RefCounted {
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

export function getSkeletonNodeDiameter(
  renderMode: SkeletonRenderMode,
  lineWidth: number,
) {
  if (renderMode === SkeletonRenderMode.LINES_AND_POINTS) {
    return Math.max(5, lineWidth * 2);
  }
  return lineWidth;
}

export function setMouseStatePositionFromSpatialSkeletonNode(
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

export const vertexPositionAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 3,
  name: "",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "vec3",
};

export const segmentAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.UINT32,
  numComponents: 1,
  name: "segment",
  webglDataType: WebGL2RenderingContext.UNSIGNED_INT,
  glslDataType: getShaderType(DataType.UINT32, 1),
};

export const selectedNodeAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 1,
  name: "selectedNodeAttr",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "float",
};

export interface SkeletonChunkBase extends SkeletonGPUGeometry {
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
export function uploadSkeletonChunkToGPU(gl: GL, chunk: SkeletonChunkBase) {
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

export function freeSkeletonChunkGPUMemory(gl: GL, chunk: SkeletonChunkBase) {
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
