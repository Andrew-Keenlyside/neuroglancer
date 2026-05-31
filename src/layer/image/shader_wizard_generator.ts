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
 * @file Pure GLSL generator for the image-layer shader wizard.
 *
 * Given a WizardConfig describing how the user wants their image data colored,
 * emits a complete fragment_main shader with the appropriate `#uicontrol`
 * directives. Pure (no DOM, no neuroglancer state) so it can be unit-tested.
 */

import type { ColormapName } from "#src/webgl/colormaps.js";

export type ColorTreatment =
  | { type: "single-color"; color: string }
  | { type: "named-colormap"; name: ColormapName }
  | {
      type: "transfer-function";
      defaultColor: string;
      /** When set, the transfer function RGB is driven by this colormap. */
      colormap?: ColormapName;
    };

export type AlphaSource = "from-intensity" | "separate-invlerp";

export interface SingleChannelConfig {
  mode: "single";
  /** Channel coordinate tuple; omit (or empty) for rank-0 layers. */
  channel?: readonly number[];
  treatment: ColorTreatment;
  alpha2D: boolean;
  alpha3D: boolean;
  /** Ignored when treatment.type === "transfer-function". */
  alphaSource: AlphaSource;
}

export interface ChannelRule {
  channel: readonly number[];
  treatment: ColorTreatment;
}

export interface MultiChannelConfig {
  mode: "multi";
  rules: readonly ChannelRule[];
  alpha2D: boolean;
  alpha3D: boolean;
}

export type WizardConfig = SingleChannelConfig | MultiChannelConfig;

export interface WizardContext {
  /** Rank of the layer's channelCoordinateSpace. */
  channelRank: number;
}

export function defaultSingleChannelConfig(
  context: WizardContext,
): SingleChannelConfig {
  return {
    mode: "single",
    channel:
      context.channelRank > 0
        ? new Array(context.channelRank).fill(0)
        : undefined,
    treatment: { type: "single-color", color: "#ffaa00" },
    alpha2D: false,
    alpha3D: true,
    alphaSource: "from-intensity",
  };
}

export function defaultMultiChannelConfig(
  context: WizardContext,
): MultiChannelConfig {
  return {
    mode: "multi",
    rules: [
      {
        channel:
          context.channelRank > 0
            ? new Array(context.channelRank).fill(0)
            : [0],
        treatment: { type: "named-colormap", name: "viridis" },
      },
    ],
    alpha2D: false,
    alpha3D: true,
  };
}

function channelClause(channel: readonly number[] | undefined): string {
  if (channel === undefined || channel.length === 0) return "";
  return `channel=[${channel.join(",")}]`;
}

function joinParams(...parts: string[]): string {
  const nonEmpty = parts.filter((p) => p.length > 0);
  return nonEmpty.length === 0 ? "" : `(${nonEmpty.join(", ")})`;
}

interface Builder {
  directives: string[];
  body: string[];
}

/**
 * Emits the `#uicontrol` directives plus the GLSL expressions that yield
 * (rgb: vec3, a: float) for a single color treatment bound to a specific
 * channel. Returns the GLSL expression names for the caller to compose.
 */
function emitTreatment(
  builder: Builder,
  prefix: string,
  channel: readonly number[] | undefined,
  treatment: ColorTreatment,
): { rgbExpr: string; alphaExpr: string; isTransferFunction: boolean } {
  const chClause = channelClause(channel);
  switch (treatment.type) {
    case "single-color": {
      const normName = `${prefix}_norm`;
      const tintName = `${prefix}_tint`;
      builder.directives.push(
        `#uicontrol invlerp ${normName}${joinParams(chClause, "clamp=true")}`,
      );
      builder.directives.push(
        `#uicontrol vec3 ${tintName} color(default="${treatment.color}")`,
      );
      return {
        rgbExpr: `${tintName} * ${normName}()`,
        alphaExpr: `${normName}()`,
        isTransferFunction: false,
      };
    }
    case "named-colormap": {
      const normName = `${prefix}_norm`;
      const cmapName = `${prefix}_cmap`;
      builder.directives.push(
        `#uicontrol invlerp ${normName}${joinParams(chClause, "clamp=true")}`,
      );
      builder.directives.push(
        `#uicontrol colormap ${cmapName} colormap(default="${treatment.name}")`,
      );
      return {
        rgbExpr: `${cmapName}(${normName}())`,
        alphaExpr: `${normName}()`,
        isTransferFunction: false,
      };
    }
    case "transfer-function": {
      const tfName = `${prefix}_tf`;
      // Intentionally omit `controlPoints` and `window`: leaving both unset lets
      // the transfer-function widget auto-range from the currently loaded data
      // and seed default control points (transparent -> defaultColor) at 30%/70%
      // of that computed window. Emitting an explicit window=[0,1] would defeat
      // that auto-ranging and bunch the points at the bottom of the data range.
      const colormapClause =
        treatment.colormap !== undefined
          ? `colormap="${treatment.colormap}"`
          : "";
      builder.directives.push(
        `#uicontrol transferFunction ${tfName}${joinParams(
          chClause,
          `defaultColor="${treatment.defaultColor}"`,
          colormapClause,
        )}`,
      );
      return {
        rgbExpr: `${tfName}().rgb`,
        alphaExpr: `${tfName}().a`,
        isTransferFunction: true,
      };
    }
  }
}

function emitOutputBlock(
  builder: Builder,
  rgbVar: string,
  alphaVar: string,
  alpha2D: boolean,
  alpha3D: boolean,
): void {
  if (alpha2D === alpha3D) {
    // Same behaviour in both — no branch needed.
    if (alpha3D) {
      builder.body.push(`emitRGBA(vec4(${rgbVar}, ${alphaVar}));`);
    } else {
      builder.body.push(`emitRGB(${rgbVar});`);
    }
    return;
  }
  builder.body.push("if (VOLUME_RENDERING) {");
  if (alpha3D) {
    builder.body.push(`  emitRGBA(vec4(${rgbVar}, ${alphaVar}));`);
  } else {
    builder.body.push(`  emitRGB(${rgbVar});`);
  }
  builder.body.push("} else {");
  if (alpha2D) {
    builder.body.push(`  emitRGBA(vec4(${rgbVar}, ${alphaVar}));`);
  } else {
    builder.body.push(`  emitRGB(${rgbVar});`);
  }
  builder.body.push("}");
}

function emitSingleChannel(
  config: SingleChannelConfig,
  builder: Builder,
): void {
  const { rgbExpr, alphaExpr, isTransferFunction } = emitTreatment(
    builder,
    "main",
    config.channel,
    config.treatment,
  );
  // Resolve alpha. Transfer functions always carry their own alpha;
  // otherwise pick from-intensity (reuse rgbExpr's invlerp) or separate-invlerp.
  let finalAlphaExpr = alphaExpr;
  if (!isTransferFunction && config.alphaSource === "separate-invlerp") {
    const alphaName = "alpha_norm";
    const chClause = channelClause(config.channel);
    builder.directives.push(
      `#uicontrol invlerp ${alphaName}${joinParams(chClause, "clamp=true")}`,
    );
    finalAlphaExpr = `${alphaName}()`;
  }
  // Store into locals so the output block is simple.
  builder.body.push(`vec3 color = ${rgbExpr};`);
  builder.body.push(`float a = ${finalAlphaExpr};`);
  emitOutputBlock(builder, "color", "a", config.alpha2D, config.alpha3D);
}

function emitMultiChannel(config: MultiChannelConfig, builder: Builder): void {
  if (config.rules.length === 0) {
    builder.body.push("emitTransparent();");
    return;
  }
  const rgbTerms: string[] = [];
  const alphaTerms: string[] = [];
  config.rules.forEach((rule, i) => {
    const { rgbExpr, alphaExpr } = emitTreatment(
      builder,
      `ch${i}`,
      rule.channel,
      rule.treatment,
    );
    rgbTerms.push(rgbExpr);
    alphaTerms.push(alphaExpr);
  });
  builder.body.push(`vec3 color = ${rgbTerms.join(" + ")};`);
  const alphaExpr =
    alphaTerms.length === 1
      ? alphaTerms[0]
      : alphaTerms.reduce(
          (acc, term, idx) => (idx === 0 ? term : `max(${acc}, ${term})`),
          "",
        );
  builder.body.push(`float a = ${alphaExpr};`);
  emitOutputBlock(builder, "color", "a", config.alpha2D, config.alpha3D);
}

export function generateImageWizardShader(
  config: WizardConfig,
  _context: WizardContext,
): string {
  const builder: Builder = { directives: [], body: [] };
  if (config.mode === "single") {
    emitSingleChannel(config, builder);
  } else {
    emitMultiChannel(config, builder);
  }
  const lines: string[] = [];
  if (builder.directives.length > 0) {
    lines.push(...builder.directives, "");
  }
  lines.push("void main() {");
  for (const bodyLine of builder.body) {
    lines.push(`  ${bodyLine}`);
  }
  lines.push("}");
  lines.push("");
  return lines.join("\n");
}
