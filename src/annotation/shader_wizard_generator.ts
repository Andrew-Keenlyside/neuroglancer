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
 * @file Pure GLSL generator for the annotation shader wizard.
 *
 * Given a WizardConfig describing how the user wants annotations colored and
 * sized, emits a complete fragment_main shader with the appropriate #uicontrol
 * directives. The generator is pure (no DOM, no neuroglancer state) so it can
 * be unit-tested directly.
 */

import {
  isAnnotationNumericPropertySpec,
  type AnnotationPropertySpec,
} from "#src/annotation/index.js";
import type { ColormapName } from "#src/webgl/colormaps.js";

export type ColorMode = "default" | "constant" | "byProperty" | "additive";
export type SizeMode = "default" | "constant" | "byProperty";

/** A single channel of an additive color blend. */
export interface WizardColorChannel {
  /** Annotation property identifier driving this channel's invlerp. */
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
export type BorderMode =
  | "default"
  | "remove"
  | "conditional"
  | "colorByProperty";

export interface WizardColorRule {
  mode: ColorMode;
  /** Hex color string like "#ffaa00", used when mode === "constant". */
  constant?: string;
  /** Annotation property identifier, used when mode === "byProperty". */
  property?: string;
  colormap?: ColormapName;
  clamp?: boolean;
  /** Channels of the additive blend, used when mode === "additive". */
  channels?: WizardColorChannel[];
}

export interface WizardSizeRule {
  mode: SizeMode;
  /** Single value, used when mode === "constant". Becomes the slider default. */
  constant?: number;
  /** Annotation property identifier, used when mode === "byProperty". */
  property?: string;
  /** Output range mapped from invlerp's [0, 1], used when mode === "byProperty". */
  outputMin?: number;
  outputMax?: number;
  clamp?: boolean;
}

export interface WizardBorderRule {
  mode: BorderMode;
  /**
   * Property whose invlerp drives the show/hide decision (border shown when
   * normalized value > 0.5). Used when mode === "conditional".
   */
  conditionProperty?: string;
  conditionClamp?: boolean;
  /** Width of the border when shown (slider default). */
  showWidth?: number;
  /** Property + colormap for mode === "colorByProperty". */
  colorProperty?: string;
  colorColormap?: ColormapName;
  colorClamp?: boolean;
}

export interface WizardConfig {
  color: WizardColorRule;
  pointSize: WizardSizeRule;
  lineWidth: WizardSizeRule;
  pointBorder: WizardBorderRule;
  boxBorder: WizardBorderRule;
}

export function defaultWizardConfig(): WizardConfig {
  return {
    color: { mode: "default" },
    pointSize: { mode: "default" },
    lineWidth: { mode: "default" },
    pointBorder: { mode: "default" },
    boxBorder: { mode: "default" },
  };
}

interface Builder {
  directives: string[];
  body: string[];
}

function findProperty(
  properties: readonly AnnotationPropertySpec[],
  id: string | undefined,
): AnnotationPropertySpec | undefined {
  if (id === undefined) return undefined;
  return properties.find((p) => p.identifier === id);
}

function emitInvlerp(
  builder: Builder,
  name: string,
  propertyId: string,
  clamp: boolean | undefined,
): void {
  builder.directives.push(
    `#uicontrol invlerp ${name}(property="${propertyId}", clamp=${clamp !== false})`,
  );
}

function emitColormap(
  builder: Builder,
  name: string,
  colormap: ColormapName,
): void {
  builder.directives.push(
    `#uicontrol colormap ${name} colormap(default="${colormap}")`,
  );
}

function emitColor(
  rule: WizardColorRule,
  properties: readonly AnnotationPropertySpec[],
  builder: Builder,
): void {
  if (rule.mode === "default") {
    builder.body.push("setColor(defaultColor());");
    return;
  }
  if (rule.mode === "constant") {
    const hex = rule.constant ?? "#ffffff";
    builder.directives.push(
      `#uicontrol vec3 color_constant color(default="${hex}")`,
    );
    builder.body.push("setColor(vec4(color_constant, 1.0));");
    return;
  }
  if (rule.mode === "additive") {
    emitAdditiveColor(rule.channels ?? [], properties, builder);
    return;
  }
  // byProperty
  const prop = findProperty(properties, rule.property);
  if (prop === undefined || !isAnnotationNumericPropertySpec(prop)) {
    builder.body.push("setColor(defaultColor());");
    return;
  }
  emitInvlerp(builder, "color_invlerp", prop.identifier, rule.clamp);
  emitColormap(builder, "color_cmap", rule.colormap ?? "viridis");
  builder.body.push("setColor(vec4(color_cmap(color_invlerp()), 1.0));");
}

/**
 * Emits the additive-blend color: each valid channel contributes
 * `tint_i * invlerp_i()` to the RGB sum, and the alpha is the max of the
 * channels' invlerps so low-signal annotations fade out. Falls back to the
 * layer default color when no channel resolves to a numeric property.
 */
function emitAdditiveColor(
  channels: readonly WizardColorChannel[],
  properties: readonly AnnotationPropertySpec[],
  builder: Builder,
): void {
  const rgbTerms: string[] = [];
  const alphaTerms: string[] = [];
  channels.forEach((channel) => {
    const prop = findProperty(properties, channel.property);
    if (prop === undefined || !isAnnotationNumericPropertySpec(prop)) return;
    const i = rgbTerms.length;
    const invlerpName = `blend${i}_invlerp`;
    const colorName = `blend${i}_color`;
    emitInvlerp(builder, invlerpName, prop.identifier, channel.clamp);
    builder.directives.push(
      `#uicontrol vec3 ${colorName} color(default="${channel.color}")`,
    );
    rgbTerms.push(`${colorName} * ${invlerpName}()`);
    alphaTerms.push(`${invlerpName}()`);
  });
  if (rgbTerms.length === 0) {
    builder.body.push("setColor(defaultColor());");
    return;
  }
  const rgb = rgbTerms.join(" + ");
  // Right-fold into max(t0, max(t1, t2)); a single term yields just that term.
  const alpha = alphaTerms.reduceRight((acc, term) => `max(${term}, ${acc})`);
  builder.body.push(`setColor(vec4(${rgb}, ${alpha}));`);
}

interface SizeTargetInfo {
  setter: string;
  prefix: string;
  sliderMin: number;
  sliderMax: number;
  defaultConstant: number;
  defaultOutputMin: number;
  defaultOutputMax: number;
}

const POINT_SIZE_INFO: SizeTargetInfo = {
  setter: "setPointMarkerSize",
  prefix: "point_size",
  sliderMin: 0,
  sliderMax: 50,
  defaultConstant: 5,
  defaultOutputMin: 3,
  defaultOutputMax: 15,
};

const LINE_WIDTH_INFO: SizeTargetInfo = {
  setter: "setLineWidth",
  prefix: "line_width",
  sliderMin: 0,
  sliderMax: 20,
  defaultConstant: 1,
  defaultOutputMin: 1,
  defaultOutputMax: 6,
};

function emitSize(
  rule: WizardSizeRule,
  info: SizeTargetInfo,
  properties: readonly AnnotationPropertySpec[],
  builder: Builder,
): void {
  if (rule.mode === "default") return;
  if (rule.mode === "constant") {
    const value = rule.constant ?? info.defaultConstant;
    builder.directives.push(
      `#uicontrol float ${info.prefix} slider(min=${info.sliderMin}, max=${info.sliderMax}, default=${value})`,
    );
    builder.body.push(`${info.setter}(${info.prefix});`);
    return;
  }
  // byProperty
  const prop = findProperty(properties, rule.property);
  if (prop === undefined || !isAnnotationNumericPropertySpec(prop)) return;
  const outMin = rule.outputMin ?? info.defaultOutputMin;
  const outMax = rule.outputMax ?? info.defaultOutputMax;
  emitInvlerp(builder, `${info.prefix}_invlerp`, prop.identifier, rule.clamp);
  builder.directives.push(
    `#uicontrol float ${info.prefix}_min slider(min=${info.sliderMin}, max=${info.sliderMax}, default=${outMin})`,
  );
  builder.directives.push(
    `#uicontrol float ${info.prefix}_max slider(min=${info.sliderMin}, max=${info.sliderMax}, default=${outMax})`,
  );
  builder.body.push(
    `${info.setter}(mix(${info.prefix}_min, ${info.prefix}_max, ${info.prefix}_invlerp()));`,
  );
}

interface BorderTargetInfo {
  widthSetter: string;
  colorSetter: string;
  prefix: string;
  defaultShowWidth: number;
}

const POINT_BORDER_INFO: BorderTargetInfo = {
  widthSetter: "setPointMarkerBorderWidth",
  colorSetter: "setPointMarkerBorderColor",
  prefix: "point_border",
  defaultShowWidth: 1,
};

const BOX_BORDER_INFO: BorderTargetInfo = {
  widthSetter: "setBoundingBoxBorderWidth",
  colorSetter: "setBoundingBoxBorderColor",
  prefix: "box_border",
  defaultShowWidth: 1,
};

function emitBorder(
  rule: WizardBorderRule,
  info: BorderTargetInfo,
  properties: readonly AnnotationPropertySpec[],
  builder: Builder,
): void {
  if (rule.mode === "default") return;
  if (rule.mode === "remove") {
    builder.body.push(`${info.widthSetter}(0.0);`);
    return;
  }
  if (rule.mode === "conditional") {
    const prop = findProperty(properties, rule.conditionProperty);
    if (prop === undefined) return;
    const width = rule.showWidth ?? info.defaultShowWidth;
    emitInvlerp(
      builder,
      `${info.prefix}_condition`,
      prop.identifier,
      rule.conditionClamp,
    );
    builder.directives.push(
      `#uicontrol float ${info.prefix}_width slider(min=0, max=10, default=${width})`,
    );
    builder.body.push(
      `${info.widthSetter}(${info.prefix}_condition() > 0.5 ? ${info.prefix}_width : 0.0);`,
    );
    return;
  }
  // colorByProperty
  const prop = findProperty(properties, rule.colorProperty);
  if (prop === undefined || !isAnnotationNumericPropertySpec(prop)) return;
  emitInvlerp(
    builder,
    `${info.prefix}_invlerp`,
    prop.identifier,
    rule.colorClamp,
  );
  emitColormap(builder, `${info.prefix}_cmap`, rule.colorColormap ?? "viridis");
  builder.body.push(
    `${info.colorSetter}(vec4(${info.prefix}_cmap(${info.prefix}_invlerp()), 1.0));`,
  );
}

export function generateWizardShader(
  config: WizardConfig,
  properties: readonly AnnotationPropertySpec[],
): string {
  const builder: Builder = { directives: [], body: [] };
  emitColor(config.color, properties, builder);
  emitSize(config.pointSize, POINT_SIZE_INFO, properties, builder);
  emitSize(config.lineWidth, LINE_WIDTH_INFO, properties, builder);
  emitBorder(config.pointBorder, POINT_BORDER_INFO, properties, builder);
  emitBorder(config.boxBorder, BOX_BORDER_INFO, properties, builder);
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
