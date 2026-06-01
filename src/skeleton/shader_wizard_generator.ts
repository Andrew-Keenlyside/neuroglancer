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
 * @file Pure GLSL generator for the skeleton shader wizard.
 *
 * Given a WizardConfig describing how the user wants skeleton vertices colored
 * and how line width should scale, emits a complete fragment_main shader with
 * the appropriate #uicontrol directives. The generator is pure (no DOM, no
 * neuroglancer state) so it can be unit-tested directly.
 *
 * Coloring mirrors the annotation wizard (constant / by-property colormap /
 * additive blend) but emits via `emitRGB(vec3)` (available for both edge and
 * node skeleton shaders). Line width uses the reserved control names
 * (`lineWidth`, or `lineWidth_invlerp` + `lineWidth_min`/`lineWidth_max`) that
 * the skeleton renderer reads in the vertex stage to size edges and node
 * points.
 */

import type { ColormapName } from "#src/webgl/colormaps.js";

export type SkeletonColorMode =
  | "default"
  | "constant"
  | "byProperty"
  | "additive";
export type SkeletonWidthMode = "default" | "constant" | "byProperty";

/** A single channel of an additive color blend. */
export interface SkeletonColorChannel {
  property: string;
  /** Hex color string like "#ff0000" scaling this channel. */
  color: string;
  clamp?: boolean;
}

/**
 * Default per-channel colors for the additive blend: a pure red, green, blue
 * series, falling back to white for any channel beyond the third.
 */
export const ADDITIVE_DEFAULT_COLORS = ["#ff0000", "#00ff00", "#0000ff"];

export function additiveDefaultColor(index: number): string {
  return ADDITIVE_DEFAULT_COLORS[index] ?? "#ffffff";
}

export interface SkeletonColorRule {
  mode: SkeletonColorMode;
  /** Hex color string, used when mode === "constant". */
  constant?: string;
  /** Vertex attribute name, used when mode === "byProperty". */
  property?: string;
  colormap?: ColormapName;
  clamp?: boolean;
  /** Channels of the additive blend, used when mode === "additive". */
  channels?: SkeletonColorChannel[];
}

export interface SkeletonWidthRule {
  mode: SkeletonWidthMode;
  /** Constant width (slider default), used when mode === "constant". */
  constant?: number;
  /** Vertex attribute name, used when mode === "byProperty". */
  property?: string;
  /** Output width range mapped from the invlerp's [0, 1]. */
  outputMin?: number;
  outputMax?: number;
  clamp?: boolean;
}

export interface SkeletonWizardConfig {
  color: SkeletonColorRule;
  width: SkeletonWidthRule;
}

export function defaultSkeletonWizardConfig(): SkeletonWizardConfig {
  return {
    color: { mode: "default" },
    width: { mode: "default" },
  };
}

const WIDTH_SLIDER_MIN = 0;
const WIDTH_SLIDER_MAX = 50;
const WIDTH_DEFAULT_CONSTANT = 3;
const WIDTH_DEFAULT_OUT_MIN = 1;
const WIDTH_DEFAULT_OUT_MAX = 6;

interface Builder {
  directives: string[];
  body: string[];
}

function emitInvlerp(
  builder: Builder,
  name: string,
  property: string,
  clamp: boolean | undefined,
): void {
  builder.directives.push(
    `#uicontrol invlerp ${name}(property="${property}", clamp=${clamp !== false})`,
  );
}

function emitColor(
  rule: SkeletonColorRule,
  properties: readonly string[],
  builder: Builder,
): void {
  if (rule.mode === "default") {
    builder.body.push("emitDefault();");
    return;
  }
  if (rule.mode === "constant") {
    const hex = rule.constant ?? "#ffffff";
    builder.directives.push(
      `#uicontrol vec3 color_constant color(default="${hex}")`,
    );
    builder.body.push("emitRGB(color_constant);");
    return;
  }
  if (rule.mode === "additive") {
    emitAdditiveColor(rule.channels ?? [], properties, builder);
    return;
  }
  // byProperty
  if (rule.property === undefined || !properties.includes(rule.property)) {
    builder.body.push("emitDefault();");
    return;
  }
  emitInvlerp(builder, "color_invlerp", rule.property, rule.clamp);
  builder.directives.push(
    `#uicontrol colormap color_cmap colormap(default="${rule.colormap ?? "viridis"}")`,
  );
  builder.body.push("emitRGB(color_cmap(color_invlerp()));");
}

function emitAdditiveColor(
  channels: readonly SkeletonColorChannel[],
  properties: readonly string[],
  builder: Builder,
): void {
  const rgbTerms: string[] = [];
  channels.forEach((channel) => {
    if (!properties.includes(channel.property)) return;
    const i = rgbTerms.length;
    const invlerpName = `blend${i}_invlerp`;
    const colorName = `blend${i}_color`;
    emitInvlerp(builder, invlerpName, channel.property, channel.clamp);
    builder.directives.push(
      `#uicontrol vec3 ${colorName} color(default="${channel.color}")`,
    );
    rgbTerms.push(`${colorName} * ${invlerpName}()`);
  });
  if (rgbTerms.length === 0) {
    builder.body.push("emitDefault();");
    return;
  }
  builder.body.push(`emitRGB(${rgbTerms.join(" + ")});`);
}

function emitWidth(
  rule: SkeletonWidthRule,
  properties: readonly string[],
  builder: Builder,
): void {
  if (rule.mode === "default") return;
  if (rule.mode === "constant") {
    const value = rule.constant ?? WIDTH_DEFAULT_CONSTANT;
    builder.directives.push(
      `#uicontrol float lineWidth slider(min=${WIDTH_SLIDER_MIN}, max=${WIDTH_SLIDER_MAX}, default=${value})`,
    );
    builder.body.push("setLineWidth(lineWidth);");
    return;
  }
  // byProperty
  if (rule.property === undefined || !properties.includes(rule.property)) {
    return;
  }
  const outMin = rule.outputMin ?? WIDTH_DEFAULT_OUT_MIN;
  const outMax = rule.outputMax ?? WIDTH_DEFAULT_OUT_MAX;
  emitInvlerp(builder, "lineWidth_invlerp", rule.property, rule.clamp);
  builder.directives.push(
    `#uicontrol float lineWidth_min slider(min=${WIDTH_SLIDER_MIN}, max=${WIDTH_SLIDER_MAX}, default=${outMin})`,
  );
  builder.directives.push(
    `#uicontrol float lineWidth_max slider(min=${WIDTH_SLIDER_MIN}, max=${WIDTH_SLIDER_MAX}, default=${outMax})`,
  );
  builder.body.push(
    "setLineWidth(mix(lineWidth_min, lineWidth_max, lineWidth_invlerp()));",
  );
}

export function generateSkeletonWizardShader(
  config: SkeletonWizardConfig,
  properties: readonly string[],
): string {
  const builder: Builder = { directives: [], body: [] };
  emitColor(config.color, properties, builder);
  emitWidth(config.width, properties, builder);
  const lines: string[] = [];
  if (builder.directives.length > 0) {
    lines.push(...builder.directives, "");
  }
  lines.push("void main() {");
  for (const line of builder.body) {
    lines.push(`  ${line}`);
  }
  lines.push("}");
  lines.push("");
  return lines.join("\n");
}
