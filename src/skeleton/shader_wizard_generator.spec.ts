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
  defaultSkeletonWizardConfig,
  generateSkeletonWizardShader,
} from "#src/skeleton/shader_wizard_generator.js";

const PROPS = ["radius", "confidence"];

describe("generateSkeletonWizardShader", () => {
  it("default config emits emitDefault() and no controls", () => {
    const shader = generateSkeletonWizardShader(
      defaultSkeletonWizardConfig(),
      [],
    );
    expect(shader).toContain("emitDefault();");
    expect(shader).not.toContain("#uicontrol");
  });

  it("constant color emits a color picker + emitRGB", () => {
    const shader = generateSkeletonWizardShader(
      {
        ...defaultSkeletonWizardConfig(),
        color: { mode: "constant", constant: "#ff8800" },
      },
      [],
    );
    expect(shader).toContain(
      `#uicontrol vec3 color_constant color(default="#ff8800")`,
    );
    expect(shader).toContain("emitRGB(color_constant);");
  });

  it("color byProperty emits invlerp + colormap + emitRGB", () => {
    const shader = generateSkeletonWizardShader(
      {
        ...defaultSkeletonWizardConfig(),
        color: {
          mode: "byProperty",
          property: "radius",
          colormap: "viridis",
          clamp: true,
        },
      },
      PROPS,
    );
    expect(shader).toContain(
      `#uicontrol invlerp color_invlerp(property="radius", clamp=true)`,
    );
    expect(shader).toContain(
      `#uicontrol colormap color_cmap colormap(default="viridis")`,
    );
    expect(shader).toContain("emitRGB(color_cmap(color_invlerp()));");
  });

  it("color additive blends per-channel invlerp × color", () => {
    const shader = generateSkeletonWizardShader(
      {
        ...defaultSkeletonWizardConfig(),
        color: {
          mode: "additive",
          channels: [
            { property: "radius", color: "#ff0000", clamp: true },
            { property: "confidence", color: "#00ff00", clamp: true },
            { property: "radius", color: "#0000ff", clamp: true },
          ],
        },
      },
      PROPS,
    );
    expect(shader).toContain(
      `#uicontrol invlerp blend0_invlerp(property="radius", clamp=true)`,
    );
    expect(shader).toContain(
      `#uicontrol vec3 blend0_color color(default="#ff0000")`,
    );
    expect(shader).toContain(
      `#uicontrol vec3 blend1_color color(default="#00ff00")`,
    );
    expect(shader).toContain(
      `#uicontrol vec3 blend2_color color(default="#0000ff")`,
    );
    expect(shader).toContain(
      "emitRGB(blend0_color * blend0_invlerp() + blend1_color * blend1_invlerp() + blend2_color * blend2_invlerp());",
    );
  });

  it("color byProperty falls back to default when property missing", () => {
    const shader = generateSkeletonWizardShader(
      {
        ...defaultSkeletonWizardConfig(),
        color: { mode: "byProperty", property: "does_not_exist" },
      },
      PROPS,
    );
    expect(shader).toContain("emitDefault();");
    expect(shader).not.toContain("color_invlerp");
  });

  it("constant line width emits a slider + setLineWidth call", () => {
    const shader = generateSkeletonWizardShader(
      {
        ...defaultSkeletonWizardConfig(),
        width: { mode: "constant", constant: 4 },
      },
      [],
    );
    expect(shader).toContain(
      "#uicontrol float lineWidth slider(min=0, max=50, default=4)",
    );
    expect(shader).toContain("setLineWidth(lineWidth);");
  });

  it("line width byProperty emits invlerp + min/max controls + setLineWidth(mix(...))", () => {
    const shader = generateSkeletonWizardShader(
      {
        ...defaultSkeletonWizardConfig(),
        width: {
          mode: "byProperty",
          property: "radius",
          outputMin: 1,
          outputMax: 8,
          clamp: true,
        },
      },
      PROPS,
    );
    expect(shader).toContain(
      `#uicontrol invlerp lineWidth_invlerp(property="radius", clamp=true)`,
    );
    expect(shader).toContain(
      "#uicontrol float lineWidth_min slider(min=0, max=50, default=1)",
    );
    expect(shader).toContain(
      "#uicontrol float lineWidth_max slider(min=0, max=50, default=8)",
    );
    expect(shader).toContain(
      "setLineWidth(mix(lineWidth_min, lineWidth_max, lineWidth_invlerp()));",
    );
  });

  it("line width byProperty falls back to default when property missing", () => {
    const shader = generateSkeletonWizardShader(
      {
        ...defaultSkeletonWizardConfig(),
        width: { mode: "byProperty", property: "does_not_exist" },
      },
      PROPS,
    );
    expect(shader).not.toContain("lineWidth_invlerp");
    expect(shader).not.toContain("setLineWidth");
  });

  it("default width emits no setLineWidth call", () => {
    const shader = generateSkeletonWizardShader(
      defaultSkeletonWizardConfig(),
      PROPS,
    );
    expect(shader).not.toContain("setLineWidth");
  });

  it("combines color + width controls in one shader", () => {
    const shader = generateSkeletonWizardShader(
      {
        color: {
          mode: "byProperty",
          property: "radius",
          colormap: "magma",
          clamp: true,
        },
        width: {
          mode: "byProperty",
          property: "confidence",
          outputMin: 2,
          outputMax: 10,
        },
      },
      PROPS,
    );
    expect(shader).toContain("color_invlerp");
    expect(shader).toContain("color_cmap");
    expect(shader).toContain("lineWidth_invlerp");
    expect(shader).toContain("lineWidth_min");
    expect(shader).toContain("emitRGB(color_cmap(color_invlerp()));");
  });
});
