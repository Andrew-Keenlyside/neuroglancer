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

import { describe, it, expect } from "vitest";
import {
  defaultSingleChannelConfig,
  generateImageWizardShader,
  type WizardConfig,
} from "#src/layer/image/shader_wizard_generator.js";

const SINGLE_CHANNEL_CTX = { channelRank: 0 };
const MULTI_CHANNEL_CTX = { channelRank: 1 };

describe("generateImageWizardShader", () => {
  it("single-channel default (single-color, alpha 3D-only, from intensity)", () => {
    const config: WizardConfig = defaultSingleChannelConfig(SINGLE_CHANNEL_CTX);
    const shader = generateImageWizardShader(config, SINGLE_CHANNEL_CTX);
    expect(shader).toContain("#uicontrol invlerp main_norm(clamp=true)");
    expect(shader).toContain(
      `#uicontrol vec3 main_tint color(default="#ffaa00")`,
    );
    expect(shader).toContain("vec3 color = main_tint * main_norm();");
    expect(shader).toContain("float a = main_norm();");
    expect(shader).toContain("if (VOLUME_RENDERING) {");
    expect(shader).toContain("emitRGBA(vec4(color, a));");
    expect(shader).toContain("emitRGB(color);");
  });

  it("single-channel named colormap, separate alpha invlerp, alpha in both 2D and 3D", () => {
    const config: WizardConfig = {
      mode: "single",
      treatment: { type: "named-colormap", name: "viridis" },
      alpha2D: true,
      alpha3D: true,
      alphaSource: "separate-invlerp",
    };
    const shader = generateImageWizardShader(config, SINGLE_CHANNEL_CTX);
    expect(shader).toContain("#uicontrol invlerp main_norm(clamp=true)");
    expect(shader).toContain(
      `#uicontrol colormap main_cmap colormap(default="viridis")`,
    );
    expect(shader).toContain("#uicontrol invlerp alpha_norm(clamp=true)");
    expect(shader).toContain("vec3 color = main_cmap(main_norm());");
    expect(shader).toContain("float a = alpha_norm();");
    // Same behavior in both branches → no `if (VOLUME_RENDERING)` branch.
    expect(shader).not.toContain("if (VOLUME_RENDERING)");
    expect(shader).toContain("emitRGBA(vec4(color, a));");
  });

  it("single-channel transfer function omits controlPoints/window so the widget auto-ranges", () => {
    const config: WizardConfig = {
      mode: "single",
      treatment: { type: "transfer-function", defaultColor: "#ffaa00" },
      alpha2D: false,
      alpha3D: true,
      alphaSource: "from-intensity",
    };
    const shader = generateImageWizardShader(config, SINGLE_CHANNEL_CTX);
    expect(shader).toContain(
      `#uicontrol transferFunction main_tf(defaultColor="#ffaa00")`,
    );
    // No explicit control points or window: the TF widget auto-ranges from the
    // loaded data and seeds default control points using defaultColor.
    expect(shader).not.toContain("controlPoints=");
    expect(shader).not.toContain("window=");
    // No colormap requested → manual node colors, no colormap= parameter.
    expect(shader).not.toContain("colormap=");
    expect(shader).toContain("vec3 color = main_tf().rgb;");
    expect(shader).toContain("float a = main_tf().a;");
  });

  it('transfer function with a colormap emits colormap="..." alongside defaultColor', () => {
    const config: WizardConfig = {
      mode: "single",
      treatment: {
        type: "transfer-function",
        defaultColor: "#ffaa00",
        colormap: "viridis",
      },
      alpha2D: false,
      alpha3D: true,
      alphaSource: "from-intensity",
    };
    const shader = generateImageWizardShader(config, SINGLE_CHANNEL_CTX);
    expect(shader).toContain(
      `#uicontrol transferFunction main_tf(defaultColor="#ffaa00", colormap="viridis")`,
    );
    expect(shader).not.toContain("controlPoints=");
    expect(shader).not.toContain("window=");
  });

  it("transfer function ignores alphaSource (TF carries its own alpha)", () => {
    const config: WizardConfig = {
      mode: "single",
      treatment: { type: "transfer-function", defaultColor: "#88ff00" },
      alpha2D: true,
      alpha3D: true,
      alphaSource: "separate-invlerp", // should be ignored
    };
    const shader = generateImageWizardShader(config, SINGLE_CHANNEL_CTX);
    expect(shader).not.toContain("alpha_norm");
    expect(shader).toContain("float a = main_tf().a;");
  });

  it("channel pick on multi-channel layer threads channel=[...] into directives", () => {
    const config: WizardConfig = {
      mode: "single",
      channel: [2],
      treatment: { type: "named-colormap", name: "magma" },
      alpha2D: false,
      alpha3D: true,
      alphaSource: "from-intensity",
    };
    const shader = generateImageWizardShader(config, MULTI_CHANNEL_CTX);
    expect(shader).toContain(
      "#uicontrol invlerp main_norm(channel=[2], clamp=true)",
    );
    expect(shader).toContain(
      `#uicontrol colormap main_cmap colormap(default="magma")`,
    );
  });

  it("alpha both off → unconditional emitRGB and no branch", () => {
    const config: WizardConfig = {
      mode: "single",
      treatment: { type: "single-color", color: "#ffffff" },
      alpha2D: false,
      alpha3D: false,
      alphaSource: "from-intensity",
    };
    const shader = generateImageWizardShader(config, SINGLE_CHANNEL_CTX);
    expect(shader).not.toContain("if (VOLUME_RENDERING)");
    expect(shader).toContain("emitRGB(color);");
    expect(shader).not.toContain("emitRGBA(");
  });

  it("multi-channel additive: per-rule prefixes, sum RGB, max alpha", () => {
    const config: WizardConfig = {
      mode: "multi",
      rules: [
        {
          channel: [0],
          treatment: { type: "named-colormap", name: "viridis" },
        },
        { channel: [1], treatment: { type: "single-color", color: "#ff0000" } },
      ],
      alpha2D: false,
      alpha3D: true,
    };
    const shader = generateImageWizardShader(config, MULTI_CHANNEL_CTX);
    expect(shader).toContain(
      "#uicontrol invlerp ch0_norm(channel=[0], clamp=true)",
    );
    expect(shader).toContain(
      `#uicontrol colormap ch0_cmap colormap(default="viridis")`,
    );
    expect(shader).toContain(
      "#uicontrol invlerp ch1_norm(channel=[1], clamp=true)",
    );
    expect(shader).toContain(
      `#uicontrol vec3 ch1_tint color(default="#ff0000")`,
    );
    expect(shader).toContain(
      "vec3 color = ch0_cmap(ch0_norm()) + ch1_tint * ch1_norm();",
    );
    expect(shader).toContain("float a = max(ch0_norm(), ch1_norm());");
  });

  it("multi-channel mixed TF + colormap: prefixes still deterministic", () => {
    const config: WizardConfig = {
      mode: "multi",
      rules: [
        {
          channel: [0],
          treatment: { type: "transfer-function", defaultColor: "#ff00ff" },
        },
        { channel: [1], treatment: { type: "named-colormap", name: "plasma" } },
      ],
      alpha2D: false,
      alpha3D: true,
    };
    const shader = generateImageWizardShader(config, MULTI_CHANNEL_CTX);
    expect(shader).toContain("#uicontrol transferFunction ch0_tf(channel=[0]");
    expect(shader).toContain("#uicontrol invlerp ch1_norm(channel=[1]");
    expect(shader).toContain("ch0_tf().rgb + ch1_cmap(ch1_norm())");
    expect(shader).toContain("max(ch0_tf().a, ch1_norm())");
  });

  it("multi-channel with no rules emits emitTransparent()", () => {
    const config: WizardConfig = {
      mode: "multi",
      rules: [],
      alpha2D: false,
      alpha3D: true,
    };
    const shader = generateImageWizardShader(config, MULTI_CHANNEL_CTX);
    expect(shader).toContain("emitTransparent();");
    expect(shader).not.toContain("#uicontrol");
  });

  it("multi-channel single rule omits max() and just uses the lone alpha term", () => {
    const config: WizardConfig = {
      mode: "multi",
      rules: [
        { channel: [3], treatment: { type: "named-colormap", name: "turbo" } },
      ],
      alpha2D: false,
      alpha3D: true,
    };
    const shader = generateImageWizardShader(config, MULTI_CHANNEL_CTX);
    expect(shader).toContain("float a = ch0_norm();");
    expect(shader).not.toContain("max(");
  });
});
