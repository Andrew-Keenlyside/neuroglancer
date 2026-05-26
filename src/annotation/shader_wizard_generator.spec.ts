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
import type { AnnotationPropertySpec } from "#src/annotation/index.js";
import {
  defaultWizardConfig,
  generateWizardShader,
} from "#src/annotation/shader_wizard_generator.js";

const PROPS: AnnotationPropertySpec[] = [
  {
    identifier: "score",
    description: undefined,
    type: "float32",
    default: 0,
  },
  {
    identifier: "confidence",
    description: undefined,
    type: "float32",
    default: 0,
  },
  {
    identifier: "is_good",
    description: undefined,
    type: "bool",
    default: 0,
  },
];

describe("generateWizardShader", () => {
  it("default config emits the same shape as the built-in default", () => {
    const shader = generateWizardShader(defaultWizardConfig(), []);
    expect(shader).toContain("setColor(defaultColor());");
    expect(shader).not.toContain("#uicontrol");
  });

  it("color byProperty emits invlerp + colormap + setColor", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        color: {
          mode: "byProperty",
          property: "score",
          colormap: "viridis",
          clamp: true,
        },
      },
      PROPS,
    );
    expect(shader).toContain(
      `#uicontrol invlerp color_invlerp(property="score", clamp=true)`,
    );
    expect(shader).toContain(
      `#uicontrol colormap color_cmap colormap(default="viridis")`,
    );
    expect(shader).toContain(
      "setColor(vec4(color_cmap(color_invlerp()), 1.0));",
    );
  });

  it("color constant emits a color picker", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        color: { mode: "constant", constant: "#ff8800" },
      },
      [],
    );
    expect(shader).toContain(
      `#uicontrol vec3 color_constant color(default="#ff8800")`,
    );
    expect(shader).toContain("setColor(vec4(color_constant, 1.0));");
  });

  it("point size byProperty emits invlerp + min/max sliders + mix()", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        pointSize: {
          mode: "byProperty",
          property: "confidence",
          outputMin: 3,
          outputMax: 12,
          clamp: true,
        },
      },
      PROPS,
    );
    expect(shader).toContain(
      `#uicontrol invlerp point_size_invlerp(property="confidence", clamp=true)`,
    );
    expect(shader).toContain(
      "#uicontrol float point_size_min slider(min=0, max=50, default=3)",
    );
    expect(shader).toContain(
      "#uicontrol float point_size_max slider(min=0, max=50, default=12)",
    );
    expect(shader).toContain(
      "setPointMarkerSize(mix(point_size_min, point_size_max, point_size_invlerp()));",
    );
  });

  it("point size constant emits a single slider", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        pointSize: { mode: "constant", constant: 8 },
      },
      [],
    );
    expect(shader).toContain(
      "#uicontrol float point_size slider(min=0, max=50, default=8)",
    );
    expect(shader).toContain("setPointMarkerSize(point_size);");
    expect(shader).not.toContain("point_size_invlerp");
  });

  it("line width byProperty uses the line_width prefix and setLineWidth", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        lineWidth: {
          mode: "byProperty",
          property: "score",
          outputMin: 1,
          outputMax: 4,
        },
      },
      PROPS,
    );
    expect(shader).toContain(
      "setLineWidth(mix(line_width_min, line_width_max, line_width_invlerp()));",
    );
    expect(shader).toContain(
      `#uicontrol invlerp line_width_invlerp(property="score", clamp=true)`,
    );
  });

  it("point border remove emits setPointMarkerBorderWidth(0.0) with no controls", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        pointBorder: { mode: "remove" },
      },
      [],
    );
    expect(shader).toContain("setPointMarkerBorderWidth(0.0);");
    expect(shader).not.toContain("point_border_");
    expect(shader).not.toContain("#uicontrol");
  });

  it("point border conditional gates width by an invlerp threshold", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        pointBorder: {
          mode: "conditional",
          conditionProperty: "is_good",
          showWidth: 2,
        },
      },
      PROPS,
    );
    expect(shader).toContain(
      `#uicontrol invlerp point_border_condition(property="is_good", clamp=true)`,
    );
    expect(shader).toContain(
      "#uicontrol float point_border_width slider(min=0, max=10, default=2)",
    );
    expect(shader).toContain(
      "setPointMarkerBorderWidth(point_border_condition() > 0.5 ? point_border_width : 0.0);",
    );
  });

  it("box border colorByProperty emits setBoundingBoxBorderColor with colormap", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        boxBorder: {
          mode: "colorByProperty",
          colorProperty: "score",
          colorColormap: "magma",
        },
      },
      PROPS,
    );
    expect(shader).toContain(
      `#uicontrol invlerp box_border_invlerp(property="score", clamp=true)`,
    );
    expect(shader).toContain(
      `#uicontrol colormap box_border_cmap colormap(default="magma")`,
    );
    expect(shader).toContain(
      "setBoundingBoxBorderColor(vec4(box_border_cmap(box_border_invlerp()), 1.0));",
    );
  });

  it("combines multiple rules into one shader with stable name prefixes", () => {
    const shader = generateWizardShader(
      {
        color: {
          mode: "byProperty",
          property: "score",
          colormap: "viridis",
          clamp: true,
        },
        pointSize: {
          mode: "byProperty",
          property: "confidence",
          outputMin: 3,
          outputMax: 12,
        },
        lineWidth: { mode: "constant", constant: 2 },
        pointBorder: { mode: "remove" },
        boxBorder: { mode: "default" },
      },
      PROPS,
    );
    // All expected directives present, no leakage from disabled rules.
    expect(shader).toContain("color_invlerp");
    expect(shader).toContain("color_cmap");
    expect(shader).toContain("point_size_invlerp");
    expect(shader).toContain("point_size_min");
    expect(shader).toContain("point_size_max");
    expect(shader).toContain("#uicontrol float line_width slider");
    expect(shader).toContain("setPointMarkerBorderWidth(0.0)");
    expect(shader).not.toContain("box_border");
  });

  it("byProperty silently falls back when the named property is missing", () => {
    const shader = generateWizardShader(
      {
        ...defaultWizardConfig(),
        color: { mode: "byProperty", property: "does_not_exist" },
        pointSize: { mode: "byProperty", property: "does_not_exist" },
      },
      PROPS,
    );
    // Color falls back to defaultColor(); pointSize emits nothing.
    expect(shader).toContain("setColor(defaultColor());");
    expect(shader).not.toContain("point_size");
  });
});
