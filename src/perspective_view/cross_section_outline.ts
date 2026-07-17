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

/**
 * @file Draws the border of a cross-section plane in the 3-d perspective view.
 *
 * Pairs with `hideCrossSectionBackground3D`: hiding the background removes the
 * grey quad that obscures 3-d geometry, but with it goes the only cue for where
 * the section actually is. This restores that cue as an outline.
 */

import { RefCounted } from "#src/util/disposable.js";
import type { mat4, vec4 } from "#src/util/geom.js";
import { getObjectId } from "#src/util/object_id.js";
import type { GL } from "#src/webgl/context.js";
import {
  defineLineShader,
  drawLines,
  initializeLineShader,
  VERTICES_PER_LINE,
} from "#src/webgl/lines.js";
import type { ShaderModule, ShaderProgram } from "#src/webgl/shader.js";
import { ShaderBuilder } from "#src/webgl/shader.js";

/**
 * The 4 edges of the canonical square as (start, end) pairs, matching the
 * matrix `PerspectivePanel.drawSliceViews` builds, which maps (+/-1, +/-1, 0)
 * onto the slice-view quad. Same winding as `getSquareCornersBuffer`, which
 * supplies the fill, so the outline traces exactly the fill's boundary.
 */
const glsl_outlineCornerOffsets = `
const vec2[8] outlineCornerOffsets = vec2[](
  vec2(-1.0, -1.0), vec2(-1.0,  1.0),
  vec2(-1.0,  1.0), vec2( 1.0,  1.0),
  vec2( 1.0,  1.0), vec2( 1.0, -1.0),
  vec2( 1.0, -1.0), vec2(-1.0, -1.0)
);
`;

const EDGES_PER_OUTLINE = 4;

/**
 * Renders a cross-section plane's border as anti-aliased lines.
 *
 * Lines are quad-expanded in screen space by `webgl/lines.ts` rather than drawn
 * with `gl.LINES`, because `gl.lineWidth` above 1 is unsupported on essentially
 * every WebGL2 implementation -- the repo never sets it above 1 anywhere.
 *
 * The shader emits through the caller-supplied `emitter` (rather than the
 * simpler `trivialColorShader` that `AxesLineHelper` uses) specifically so it
 * writes ALL of the perspective panel's colour attachments. `drawSliceViews`
 * runs with the full 3-attachment draw-buffer list bound, so a single-output
 * shader would leave the Z and PICK attachments undefined. `AxesLineHelper`
 * avoids that by narrowing `gl.drawBuffers` to COLOR_ATTACHMENT0, which is only
 * safe because it draws after the annotation pass; doing the same here would
 * require restoring the list before the annotation and transparent passes that
 * follow.
 */
export class CrossSectionOutlineRenderHelper extends RefCounted {
  private shader: ShaderProgram;

  constructor(
    public gl: GL,
    emitter: ShaderModule,
  ) {
    super();
    const builder = new ShaderBuilder(gl);
    builder.addUniform("highp mat4", "uProjectionMatrix");
    builder.addUniform("highp vec4", "uColor");
    builder.require(emitter);
    defineLineShader(builder);
    builder.addVertexCode(glsl_outlineCornerOffsets);
    builder.setVertexMain(`
int edgeIndex = gl_VertexID / ${VERTICES_PER_LINE};
emitLine(uProjectionMatrix * vec4(outlineCornerOffsets[edgeIndex * 2], 0.0, 1.0),
         uProjectionMatrix * vec4(outlineCornerOffsets[edgeIndex * 2 + 1], 0.0, 1.0),
         2.0);
`);
    builder.setFragmentMain(`
emit(vec4(uColor.rgb, uColor.a * getLineAlpha()), 0u);
`);
    this.shader = this.registerDisposer(builder.build());
  }

  /**
   * Draws the outline of the unit square as transformed by `projectionMatrix`.
   *
   * Assumes DEPTH_TEST is enabled and the panel's full draw-buffer list is
   * bound. Enables blending and relaxes depthFunc for the duration of the draw,
   * restoring both -- callers after this point (the annotation pass, then the
   * transparent pass) assume blend off and depthFunc LESS.
   */
  draw(
    projectionMatrix: mat4,
    color: vec4,
    projectionParameters: { width: number; height: number },
  ) {
    const { gl, shader } = this;
    shader.bind();
    gl.uniformMatrix4fv(
      shader.uniform("uProjectionMatrix"),
      /*transpose=*/ false,
      projectionMatrix,
    );
    gl.uniform4fv(shader.uniform("uColor"), color);
    initializeLineShader(
      shader,
      projectionParameters,
      /*featherWidthInPixels=*/ 1,
    );

    // The outline is coplanar with the fill it traces, so under the default
    // LESS the outline's own fragments lose to the fill's at equal depth and
    // half the border width disappears. LEQUAL lets it win the tie, which also
    // makes the outline usable when the background is left visible.
    gl.depthFunc(WebGL2RenderingContext.LEQUAL);
    // Required for getLineAlpha()'s feathered edges. SliceViewRenderHelper.draw
    // disables blending on every call, so this cannot be hoisted out of the
    // per-slice-view loop.
    gl.enable(WebGL2RenderingContext.BLEND);
    gl.blendFunc(
      WebGL2RenderingContext.SRC_ALPHA,
      WebGL2RenderingContext.ONE_MINUS_SRC_ALPHA,
    );

    drawLines(gl, EDGES_PER_OUTLINE, 1);

    gl.disable(WebGL2RenderingContext.BLEND);
    gl.depthFunc(WebGL2RenderingContext.LESS);
  }

  static get(gl: GL, emitter: ShaderModule) {
    return gl.memoize.get(
      `perspective_view/CrossSectionOutlineRenderHelper:${getObjectId(emitter)}`,
      () => new CrossSectionOutlineRenderHelper(gl, emitter),
    );
  }
}
