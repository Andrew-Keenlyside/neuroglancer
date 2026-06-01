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
import type { LayerView, VisibleLayerInfo } from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type { PerspectiveViewRenderContext } from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type {
  RenderLayer,
  ThreeDimensionalRenderLayerAttachmentState,
} from "#src/renderlayer.js";
import { update3dRenderLayerAttachment } from "#src/renderlayer.js";
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
import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import { SKELETON_LAYER_RPC_ID } from "#src/skeleton/base.js";
import type { SliceViewPanel } from "#src/sliceview/panel.js";
import type { SliceViewPanelRenderContext } from "#src/sliceview/renderlayer.js";
import { SliceViewPanelRenderLayer } from "#src/sliceview/renderlayer.js";
import { TrackableValue, WatchableValue } from "#src/trackable_value.js";
import { DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";
import type { vec3 } from "#src/util/geom.js";
import { mat4 } from "#src/util/geom.js";
import { verifyFinitePositiveFloat } from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";
import type { Trackable } from "#src/util/trackable.js";
import { CompoundTrackable } from "#src/util/trackable.js";
import { TrackableEnum } from "#src/util/trackable_enum.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import {
  defineCircleShader,
  drawCircles,
  initializeCircleShader,
} from "#src/webgl/circles.js";
import type { GL } from "#src/webgl/context.js";
import type { WatchableShaderError } from "#src/webgl/dynamic_shader.js";
import {
  makeTrackableFragmentMain,
  parameterizedEmitterDependentShaderGetter,
  shaderCodeWithLineDirective,
} from "#src/webgl/dynamic_shader.js";
import type { HistogramSpecifications } from "#src/webgl/empirical_cdf.js";
import {
  defineInvlerpShaderFunction,
  enableLerpShaderFunction,
} from "#src/webgl/lerp.js";
import {
  defineLineShader,
  drawLines,
  initializeLineShader,
} from "#src/webgl/lines.js";
import { ShaderBuilder } from "#src/webgl/shader.js";
import type { ShaderProgram, ShaderSamplerType } from "#src/webgl/shader.js";
import type {
  ShaderControlsBuilderState,
  ShaderDataContext,
} from "#src/webgl/shader_ui_controls.js";
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

const tempMat2 = mat4.create();

const DEFAULT_FRAGMENT_MAIN = `void main() {
  emitDefault();
}
`;

interface VertexAttributeRenderInfo extends VertexAttributeInfo {
  name: string;
  webglDataType: number;
  glslDataType: string;
}

const vertexAttributeSamplerSymbols: symbol[] = [];

const vertexPositionTextureFormat = computeTextureFormat(
  new TextureFormat(),
  DataType.FLOAT32,
  3,
);

class RenderHelper extends RefCounted {
  private textureAccessHelper = new OneDimensionalTextureAccessHelper(
    "vertexData",
  );
  private vertexIdHelper;
  get vertexAttributes(): VertexAttributeRenderInfo[] {
    return this.base.vertexAttributes;
  }

  defineCommonShader(builder: ShaderBuilder) {
    defineVertexId(builder);
    builder.addUniform("highp vec4", "uColor");
    builder.addUniform("highp mat4", "uProjection");
    builder.addUniform("highp uint", "uPickID");
  }

  edgeShaderGetter;
  nodeShaderGetter;

  get gl(): GL {
    return this.base.gl;
  }

  constructor(
    public base: SkeletonLayer,
    public targetIsSliceView: boolean,
  ) {
    super();
    this.vertexIdHelper = this.registerDisposer(VertexIdHelper.get(this.gl));
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
        shaderError: this.base.displayState.shaderError,
        defineShader: (
          builder: ShaderBuilder,
          shaderBuilderState: ShaderControlsBuilderState,
        ) => {
          if (shaderBuilderState.parseResult.errors.length !== 0) {
            throw new Error("Invalid UI control specification");
          }
          this.defineCommonShader(builder);
          this.defineAttributeAccess(builder);
          defineLineShader(builder);
          builder.addAttribute("highp uvec2", "aVertexIndex");
          builder.addUniform("highp float", "uLineWidth");
          const code = shaderBuilderState.parseResult.code;
          // Line width / node diameter can be set per-vertex from the shader via
          // setLineWidth(). Because that value is consumed in the vertex stage
          // (emitLine), the user main() is run in the vertex stage too, but only
          // when the shader actually calls setLineWidth() — otherwise the shader
          // stays fragment-only and width is the global uLineWidth (unchanged).
          const usesSetLineWidth = code.includes("setLineWidth");
          const { vertexAttributes } = this;
          const numAttributes = vertexAttributes.length;

          builder.addFragmentCode(`
vec4 segmentColor() {
  return uColor;
}
void emitRGB(vec3 color) {
  emit(vec4(color * uColor.a, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), uPickID);
}
void emitDefault() {
  emit(vec4(uColor.rgb, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), uPickID);
}
void setLineWidth(float widthInPixels) {}
`);
          for (let i = 1; i < numAttributes; ++i) {
            const info = vertexAttributes[i];
            builder.addVarying(`highp ${info.glslDataType}`, `vCustom${i}`);
            const defines = `#define ${info.name} vCustom${i}\n#define prop_${info.name}() vCustom${i}\n`;
            builder.addFragmentCode(defines);
            builder.addVertexCode(defines);
          }

          let vertexMain = `
highp vec3 vertexA = readAttribute0(aVertexIndex.x);
highp vec3 vertexB = readAttribute0(aVertexIndex.y);
`;
          if (usesSetLineWidth) {
            // Width pass: sample attributes at the segment start vertex so the
            // width is constant across the expanded line quad, then run the user
            // shader, which may call setLineWidth() to set ng_lineWidth.
            for (let i = 1; i < numAttributes; ++i) {
              vertexMain += `vCustom${i} = readAttribute${i}(aVertexIndex.x);\n`;
            }
            vertexMain += `ng_lineWidth = uLineWidth;
userMain();
emitLine(uProjection, vertexA, vertexB, ng_lineWidth);
`;
          } else {
            vertexMain += `emitLine(uProjection, vertexA, vertexB, uLineWidth);\n`;
          }
          vertexMain += `highp uint lineEndpointIndex = getLineEndpointIndex();
highp uint vertexIndex = aVertexIndex.x * (1u - lineEndpointIndex) + aVertexIndex.y * lineEndpointIndex;
`;
          // Per-endpoint values for the color varyings consumed by the fragment.
          for (let i = 1; i < numAttributes; ++i) {
            vertexMain += `vCustom${i} = readAttribute${i}(vertexIndex);\n`;
          }
          if (usesSetLineWidth) {
            builder.addVertexCode(`
float ng_lineWidth;
void setLineWidth(float widthInPixels) { ng_lineWidth = widthInPixels; }
void emitRGB(vec3 color) {}
void emitDefault() {}
vec4 segmentColor() { return uColor; }
`);
          }
          builder.setVertexMain(vertexMain);
          addControlsToBuilder(shaderBuilderState, builder, true);
          if (usesSetLineWidth) {
            builder.addVertexCode(
              "\n#define main userMain\n" +
                shaderCodeWithLineDirective(code) +
                "\n#undef main\n",
            );
          }
          builder.setFragmentMainFunction(shaderCodeWithLineDirective(code));
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
        shaderError: this.base.displayState.shaderError,
        defineShader: (
          builder: ShaderBuilder,
          shaderBuilderState: ShaderControlsBuilderState,
        ) => {
          if (shaderBuilderState.parseResult.errors.length !== 0) {
            throw new Error("Invalid UI control specification");
          }
          this.defineCommonShader(builder);
          this.defineAttributeAccess(builder);
          defineCircleShader(
            builder,
            /*crossSectionFade=*/ this.targetIsSliceView,
          );
          builder.addUniform("highp float", "uNodeDiameter");
          const code = shaderBuilderState.parseResult.code;
          // See the edge shader: when the shader calls setLineWidth() the user
          // main() also runs in the vertex stage, and the node diameter tracks
          // the set width; otherwise the global uNodeDiameter is used.
          const usesSetLineWidth = code.includes("setLineWidth");
          const { vertexAttributes } = this;
          const numAttributes = vertexAttributes.length;

          builder.addFragmentCode(`
vec4 segmentColor() {
  return uColor;
}
void emitRGBA(vec4 color) {
  vec4 borderColor = color;
  emit(getCircleColor(color, borderColor), uPickID);
}
void emitRGB(vec3 color) {
  emitRGBA(vec4(color, 1.0));
}
void emitDefault() {
  emitRGBA(uColor);
}
void setLineWidth(float widthInPixels) {}
`);
          for (let i = 1; i < numAttributes; ++i) {
            const info = vertexAttributes[i];
            builder.addVarying(`highp ${info.glslDataType}`, `vCustom${i}`);
            const defines = `#define ${info.name} vCustom${i}\n#define prop_${info.name}() vCustom${i}\n`;
            builder.addFragmentCode(defines);
            builder.addVertexCode(defines);
          }

          let vertexMain = `
highp uint vertexIndex = uint(gl_InstanceID);
highp vec3 vertexPosition = readAttribute0(vertexIndex);
`;
          for (let i = 1; i < numAttributes; ++i) {
            vertexMain += `vCustom${i} = readAttribute${i}(vertexIndex);\n`;
          }
          if (usesSetLineWidth) {
            vertexMain += `ng_lineWidth = uNodeDiameter;
userMain();
emitCircle(uProjection * vec4(vertexPosition, 1.0), ng_lineWidth, 0.0);
`;
            builder.addVertexCode(`
float ng_lineWidth;
void setLineWidth(float widthInPixels) { ng_lineWidth = widthInPixels; }
void emitRGBA(vec4 color) {}
void emitRGB(vec3 color) {}
void emitDefault() {}
vec4 segmentColor() { return uColor; }
`);
          } else {
            vertexMain += `emitCircle(uProjection * vec4(vertexPosition, 1.0), uNodeDiameter, 0.0);\n`;
          }
          builder.setVertexMain(vertexMain);
          addControlsToBuilder(shaderBuilderState, builder, true);
          if (usesSetLineWidth) {
            builder.addVertexCode(
              "\n#define main userMain\n" +
                shaderCodeWithLineDirective(code) +
                "\n#undef main\n",
            );
          }
          builder.setFragmentMainFunction(shaderCodeWithLineDirective(code));
        },
      },
    );
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
  }

  private histogramShaders = new Map<number, ShaderProgram>();

  /**
   * Returns a shader that scatters one point per skeleton vertex into a 256-bin
   * histogram framebuffer, reading vertex attribute `attributeIndex` and mapping
   * it through `invlerpForHistogram` (bounds supplied at draw time). Mirrors the
   * annotation property-histogram shader.
   */
  private getHistogramShader(attributeIndex: number): ShaderProgram {
    const { histogramShaders } = this;
    let shader = histogramShaders.get(attributeIndex);
    if (shader === undefined) {
      const { gl } = this;
      shader = gl.memoize.get(
        JSON.stringify({
          t: "skeleton/SkeletonShaderManager/histogram",
          attributeIndex,
          vertexAttributes: this.vertexAttributes.map((a) => ({
            name: a.name,
            dataType: a.dataType,
            numComponents: a.numComponents,
          })),
        }),
        () => {
          const builder = new ShaderBuilder(gl);
          this.defineAttributeAccess(builder);
          builder.addOutputBuffer("vec4", "out_histogram", 0);
          builder.addVertexCode(
            defineInvlerpShaderFunction(
              builder,
              "invlerpForHistogram",
              DataType.FLOAT32,
              /*clamp=*/ false,
            ),
          );
          builder.setVertexMain(`
highp uint vertexIndex = uint(gl_VertexID);
float x = invlerpForHistogram(readAttribute${attributeIndex}(vertexIndex));
if (x < 0.0) x = 0.0;
else if (x > 1.0) x = 1.0;
else x = (1.0 + x * 253.0) / 255.0;
gl_Position = vec4(2.0 * (x * 255.0 + 0.5) / 256.0 - 1.0, 0.0, 0.0, 1.0);
gl_PointSize = 1.0;
`);
          builder.setFragmentMain("out_histogram = vec4(1.0, 1.0, 1.0, 1.0);");
          return builder.build();
        },
      );
      histogramShaders.set(attributeIndex, shader);
    }
    return shader;
  }

  /**
   * Accumulates per-vertex-property histograms for the given chunk into the
   * framebuffers owned by `histogramSpecifications`, one per referenced vertex
   * property. Uses additive blending and clears once per frame so contributions
   * from all visible skeleton chunks accumulate.
   */
  computeHistograms(
    skeletonChunk: SkeletonChunk,
    histogramSpecifications: HistogramSpecifications,
    frameNumber: number,
  ) {
    const { gl, vertexAttributes } = this;
    const properties = histogramSpecifications.properties.value;
    const numHistograms = properties.length;
    if (numHistograms === 0 || skeletonChunk.numVertices === 0) return;
    const bounds = histogramSpecifications.bounds.value;
    const framebuffers = histogramSpecifications.getFramebuffers(gl);
    const oldFrameNumber = histogramSpecifications.frameNumber;
    histogramSpecifications.frameNumber = frameNumber;
    const { vertexAttributeTextures } = skeletonChunk;
    gl.enable(WebGL2RenderingContext.BLEND);
    gl.disable(WebGL2RenderingContext.SCISSOR_TEST);
    gl.disable(WebGL2RenderingContext.DEPTH_TEST);
    gl.blendFunc(WebGL2RenderingContext.ONE, WebGL2RenderingContext.ONE);
    for (
      let histogramIndex = 0;
      histogramIndex < numHistograms;
      ++histogramIndex
    ) {
      const attributeIndex = vertexAttributes.findIndex(
        (a) => a.name === properties[histogramIndex],
      );
      if (attributeIndex <= 0) continue;
      const shader = this.getHistogramShader(attributeIndex);
      shader.bind();
      // Bind the vertex attribute textures the accessor functions sample from.
      for (let i = 0; i < vertexAttributes.length; ++i) {
        gl.activeTexture(
          WebGL2RenderingContext.TEXTURE0 +
            shader.textureUnit(vertexAttributeSamplerSymbols[i]),
        );
        gl.bindTexture(
          WebGL2RenderingContext.TEXTURE_2D,
          vertexAttributeTextures[i],
        );
      }
      enableLerpShaderFunction(
        shader,
        "invlerpForHistogram",
        DataType.FLOAT32,
        bounds[histogramIndex],
      );
      framebuffers[histogramIndex].bind(256, 1);
      if (frameNumber !== oldFrameNumber) {
        gl.clearColor(0, 0, 0, 0);
        gl.clear(WebGL2RenderingContext.COLOR_BUFFER_BIT);
      }
      gl.drawArrays(
        WebGL2RenderingContext.POINTS,
        0,
        skeletonChunk.numVertices,
      );
    }
    gl.disable(WebGL2RenderingContext.BLEND);
  }

  getCrossSectionFadeFactor() {
    if (this.targetIsSliceView) {
      return "(clamp(1.0 - 2.0 * abs(0.5 - gl_FragCoord.z), 0.0, 1.0))";
    }
    return "(1.0)";
  }

  beginLayer(
    gl: GL,
    shader: ShaderProgram,
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    modelMatrix: mat4,
  ) {
    const { viewProjectionMat } = renderContext.projectionParameters;
    const mat = mat4.multiply(tempMat2, viewProjectionMat, modelMatrix);
    gl.uniformMatrix4fv(shader.uniform("uProjection"), false, mat);
    this.vertexIdHelper.enable();
  }

  setColor(gl: GL, shader: ShaderProgram, color: vec3) {
    gl.uniform4fv(shader.uniform("uColor"), color);
  }

  setPickID(gl: GL, shader: ShaderProgram, pickID: number) {
    gl.uniform1ui(shader.uniform("uPickID"), pickID);
  }

  drawSkeleton(
    gl: GL,
    edgeShader: ShaderProgram,
    nodeShader: ShaderProgram | null,
    skeletonChunk: SkeletonChunk,
    projectionParameters: { width: number; height: number },
  ) {
    const { vertexAttributes } = this;
    const numAttributes = vertexAttributes.length;
    const { vertexAttributeTextures } = skeletonChunk;
    for (let i = 0; i < numAttributes; ++i) {
      const textureUnit =
        WebGL2RenderingContext.TEXTURE0 +
        edgeShader.textureUnit(vertexAttributeSamplerSymbols[i]);
      gl.activeTexture(textureUnit);
      gl.bindTexture(
        WebGL2RenderingContext.TEXTURE_2D,
        vertexAttributeTextures[i],
      );
    }

    // Draw edges
    {
      edgeShader.bind();
      const aVertexIndex = edgeShader.attribute("aVertexIndex");
      skeletonChunk.indexBuffer.bindToVertexAttribI(
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
      drawLines(gl, 1, skeletonChunk.numIndices / 2);
      gl.vertexAttribDivisor(aVertexIndex, 0);
      gl.disableVertexAttribArray(aVertexIndex);
    }

    if (nodeShader !== null) {
      nodeShader.bind();
      initializeCircleShader(nodeShader, projectionParameters, {
        featherWidthInPixels: this.targetIsSliceView ? 1.0 : 0.0,
      });
      drawCircles(nodeShader.gl, 2, skeletonChunk.numVertices);
    }
  }

  endLayer(gl: GL, shader: ShaderProgram) {
    const { vertexAttributes } = this;
    const numAttributes = vertexAttributes.length;
    for (let i = 0; i < numAttributes; ++i) {
      const curTextureUnit =
        shader.textureUnit(vertexAttributeSamplerSymbols[i]) +
        WebGL2RenderingContext.TEXTURE0;
      gl.activeTexture(curTextureUnit);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
    this.vertexIdHelper.disable();
  }
}

export enum SkeletonRenderMode {
  LINES = 0,
  LINES_AND_POINTS = 1,
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

export interface ViewSpecificSkeletonRenderingOptions {
  mode: TrackableSkeletonRenderMode;
  lineWidth: TrackableSkeletonLineWidth;
}

export class SkeletonRenderingOptions implements Trackable {
  private compound = new CompoundTrackable();
  get changed() {
    return this.compound.changed;
  }

  shader = makeTrackableFragmentMain(DEFAULT_FRAGMENT_MAIN);
  // Exposes the skeleton's per-vertex attributes to the shader controls as
  // `properties`, so `#uicontrol invlerp name(property="…")` can bind to them.
  // Populated once the skeleton source's vertex attributes are known.
  dataContext = new WatchableValue<ShaderDataContext>({});
  shaderControlState = new ShaderControlState(this.shader, this.dataContext);
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
}

export class SkeletonLayer extends RefCounted {
  layerChunkProgressInfo = new LayerChunkProgressInfo();
  redrawNeeded = new NullarySignal();
  private sharedObject: SegmentationLayerSharedObject;
  vertexAttributes: VertexAttributeRenderInfo[];
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

    const vertexAttributes = (this.vertexAttributes = [
      vertexPositionAttribute,
    ]);

    for (const [name, info] of source.vertexAttributes) {
      vertexAttributes.push({
        name,
        dataType: info.dataType,
        numComponents: info.numComponents,
        webglDataType: getWebglDataType(info.dataType),
        glslDataType:
          info.numComponents > 1 ? `vec${info.numComponents}` : "float",
      });
    }

    // Expose scalar vertex attributes as shader-control properties so that
    // `#uicontrol invlerp name(property="…")` (used by the skeleton shader
    // wizard) can bind to them.
    const properties = new Map<string, DataType>();
    for (const attr of vertexAttributes) {
      if (attr.name !== "" && attr.numComponents === 1) {
        properties.set(attr.name, attr.dataType);
      }
    }
    displayState.skeletonRenderingOptions.dataContext.value = { properties };
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
    const { gl, source, displayState } = this;
    if (displayState.objectAlpha.value <= 0.0) {
      // Skip drawing.
      return;
    }
    const modelMatrix = update3dRenderLayerAttachment(
      displayState.transform.value,
      renderContext.projectionParameters.displayDimensionRenderInfo,
      attachment,
    );
    if (modelMatrix === undefined) return;
    let pointDiameter: number;
    if (renderOptions.mode.value === SkeletonRenderMode.LINES_AND_POINTS) {
      pointDiameter = Math.max(5, lineWidth * 2);
    } else {
      pointDiameter = lineWidth;
    }

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
    setControlsInShader(
      gl,
      edgeShader,
      shaderControlState,
      edgeShaderParameters.parseResult.controls,
      edgeShaderParameters.parseResult.compatColormaps,
    );
    gl.uniform1f(edgeShader.uniform("uLineWidth"), lineWidth!);

    nodeShader.bind();
    renderHelper.beginLayer(gl, nodeShader, renderContext, modelMatrix);
    gl.uniform1f(nodeShader.uniform("uNodeDiameter"), pointDiameter);
    setControlsInShader(
      gl,
      nodeShader,
      shaderControlState,
      nodeShaderParameters.parseResult.controls,
      nodeShaderParameters.parseResult.compatColormaps,
    );

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
          edgeShader.bind();
          renderHelper.setColor(gl, edgeShader, <vec3>(<Float32Array>color));
          nodeShader.bind();
          renderHelper.setColor(gl, nodeShader, <vec3>(<Float32Array>color));
        }
        if (pickIndex !== undefined) {
          edgeShader.bind();
          renderHelper.setPickID(gl, edgeShader, pickIndex);
          nodeShader.bind();
          renderHelper.setPickID(gl, nodeShader, pickIndex);
        }
        renderHelper.drawSkeleton(
          gl,
          edgeShader,
          nodeShader,
          skeleton,
          renderContext.projectionParameters,
        );
      },
    );
    renderHelper.endLayer(gl, edgeShader);

    // After drawing, accumulate per-vertex-property histograms for any invlerp
    // controls that reference vertex attributes, so their widgets show a CDF and
    // auto-range works. Done in a separate pass (it rebinds the framebuffer) and
    // gated so it costs nothing when no histograms are needed.
    const { histogramSpecifications } = shaderControlState;
    if (histogramSpecifications.visibleHistograms > 0) {
      forEachVisibleSegment(
        displayState.segmentationGroupState.value,
        (objectId) => {
          const skeleton = skeletons.get(getObjectKey(objectId));
          if (
            skeleton === undefined ||
            skeleton.state !== ChunkState.GPU_MEMORY
          ) {
            return;
          }
          renderHelper.computeHistograms(
            skeleton,
            histogramSpecifications,
            renderContext.frameNumber,
          );
        },
      );
      renderContext.bindFramebuffer();
    }
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
    // Mark this layer as a producer of the property histograms so the invlerp
    // widgets draw the CDF curve (the per-vertex histogram is computed in draw).
    this.registerDisposer(
      base.displayState.skeletonRenderingOptions.shaderControlState.histogramSpecifications.producerVisibility.add(
        this.visibility,
      ),
    );
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
    // Mark this layer as a producer of the property histograms so the invlerp
    // widgets draw the CDF curve (the per-vertex histogram is computed in draw).
    this.registerDisposer(
      base.displayState.skeletonRenderingOptions.shaderControlState.histogramSpecifications.producerVisibility.add(
        this.visibility,
      ),
    );
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

function getWebglDataType(dataType: DataType) {
  switch (dataType) {
    case DataType.FLOAT32:
      return WebGL2RenderingContext.FLOAT;
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

export class SkeletonChunk extends Chunk {
  declare source: SkeletonSource;
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  indexBuffer: GLBuffer;
  numIndices: number;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  vertexAttributeTextures: (WebGLTexture | null)[];

  constructor(source: SkeletonSource, x: any) {
    super(source);
    this.vertexAttributes = x.vertexAttributes;
    const indices = (this.indices = x.indices);
    this.numVertices = x.numVertices;
    this.vertexAttributeOffsets = x.vertexAttributeOffsets;
    this.numIndices = indices.length;
  }

  copyToGPU(gl: GL) {
    super.copyToGPU(gl);
    const { attributeTextureFormats } = this.source;
    const { vertexAttributes, vertexAttributeOffsets } = this;
    const vertexAttributeTextures: (WebGLTexture | null)[] =
      (this.vertexAttributeTextures = []);
    for (
      let i = 0, numAttributes = vertexAttributeOffsets.length;
      i < numAttributes;
      ++i
    ) {
      const texture = gl.createTexture();
      gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
      setOneDimensionalTextureData(
        gl,
        attributeTextureFormats[i],
        vertexAttributes.subarray(
          vertexAttributeOffsets[i],
          i + 1 !== numAttributes
            ? vertexAttributeOffsets[i + 1]
            : vertexAttributes.length,
        ),
      );
      vertexAttributeTextures[i] = texture;
    }
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
    this.indexBuffer = GLBuffer.fromData(
      gl,
      this.indices,
      WebGL2RenderingContext.ARRAY_BUFFER,
      WebGL2RenderingContext.STATIC_DRAW,
    );
  }

  freeGPUMemory(gl: GL) {
    super.freeGPUMemory(gl);
    const { vertexAttributeTextures } = this;
    for (const texture of vertexAttributeTextures) {
      gl.deleteTexture(texture);
    }
    vertexAttributeTextures.length = 0;
    this.indexBuffer.dispose();
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
  getChunk(x: any) {
    return new SkeletonChunk(this, x);
  }

  get vertexAttributes(): Map<string, VertexAttributeInfo> {
    return emptyVertexAttributes;
  }
}
