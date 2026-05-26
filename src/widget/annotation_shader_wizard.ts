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

import "#src/widget/annotation_shader_wizard.css";

import {
  isAnnotationNumericPropertySpec,
  type AnnotationPropertySpec,
} from "#src/annotation/index.js";
import {
  defaultWizardConfig,
  generateWizardShader,
  type WizardBorderRule,
  type WizardColorRule,
  type WizardConfig,
  type WizardSizeRule,
} from "#src/annotation/shader_wizard_generator.js";
import {
  ElementVisibilityFromTrackableBoolean,
  TrackableBoolean,
} from "#src/trackable_boolean.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import { COLORMAP_NAMES, colormapDisplayName } from "#src/webgl/colormaps.js";

type PropertyFilter = "numeric" | "numericOrBool";

interface RuleHandle<T> {
  element: HTMLElement;
  getConfig(): T;
  refreshProperties(properties: readonly AnnotationPropertySpec[]): void;
}

function filterProperties(
  properties: readonly AnnotationPropertySpec[],
  filter: PropertyFilter,
): AnnotationPropertySpec[] {
  return properties.filter((p) => {
    if (isAnnotationNumericPropertySpec(p)) return true;
    return filter === "numericOrBool" && p.type === "bool";
  });
}

function makePropertySelect(
  initial: readonly AnnotationPropertySpec[],
  filter: PropertyFilter,
): HTMLSelectElement {
  const select = document.createElement("select");
  populatePropertySelect(select, initial, filter);
  return select;
}

function populatePropertySelect(
  select: HTMLSelectElement,
  properties: readonly AnnotationPropertySpec[],
  filter: PropertyFilter,
): void {
  const previous = select.value;
  const allowed = filterProperties(properties, filter);
  select.replaceChildren();
  for (const p of allowed) {
    const option = document.createElement("option");
    option.value = p.identifier;
    option.textContent = p.identifier;
    select.appendChild(option);
  }
  if (allowed.some((p) => p.identifier === previous)) {
    select.value = previous;
  }
  select.disabled = allowed.length === 0;
}

function makeColormapSelect(initial = "viridis"): HTMLSelectElement {
  const select = document.createElement("select");
  for (const name of COLORMAP_NAMES) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = colormapDisplayName(name);
    select.appendChild(option);
  }
  select.value = initial;
  return select;
}

function makeLabeled(
  labelText: string,
  control: HTMLElement,
): HTMLLabelElement {
  const label = document.createElement("label");
  label.append(labelText, control);
  return label;
}

function makeNumberInput(
  defaultValue: number,
  options: { min?: number; max?: number; step?: number } = {},
): HTMLInputElement {
  const input = document.createElement("input");
  input.type = "number";
  if (options.min !== undefined) input.min = String(options.min);
  if (options.max !== undefined) input.max = String(options.max);
  input.step = String(options.step ?? "any");
  input.value = String(defaultValue);
  return input;
}

function makeColorInput(initial: string): HTMLInputElement {
  const input = document.createElement("input");
  input.type = "color";
  input.value = initial;
  return input;
}

function makeRule(label: string): {
  row: HTMLDivElement;
  mode: HTMLSelectElement;
  detail: HTMLDivElement;
} {
  const row = document.createElement("div");
  row.classList.add("neuroglancer-annotation-wizard-row");
  const labelEl = document.createElement("div");
  labelEl.classList.add("neuroglancer-annotation-wizard-row-label");
  labelEl.textContent = label;
  const mode = document.createElement("select");
  const detail = document.createElement("div");
  detail.classList.add("neuroglancer-annotation-wizard-row-detail");
  row.append(labelEl, mode, detail);
  return { row, mode, detail };
}

function makeColorRule(
  initialProps: readonly AnnotationPropertySpec[],
): RuleHandle<WizardColorRule> {
  const { row, mode, detail } = makeRule("Color");
  for (const [value, label] of [
    ["default", "Default (layer color)"],
    ["constant", "Constant color"],
    ["byProperty", "By property + colormap"],
  ] as const) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    mode.appendChild(option);
  }
  const colorInput = makeColorInput("#ffaa00");
  const propertySelect = makePropertySelect(initialProps, "numeric");
  const colormapSelect = makeColormapSelect("viridis");
  const clampCheckbox = document.createElement("input");
  clampCheckbox.type = "checkbox";
  clampCheckbox.checked = true;

  function render() {
    detail.replaceChildren();
    if (mode.value === "constant") {
      detail.append(makeLabeled("color ", colorInput));
    } else if (mode.value === "byProperty") {
      if (propertySelect.options.length === 0) {
        const note = document.createElement("span");
        note.classList.add("neuroglancer-annotation-wizard-note");
        note.textContent = "no numeric properties available";
        detail.append(note);
      } else {
        detail.append(
          makeLabeled("property ", propertySelect),
          makeLabeled("colormap ", colormapSelect),
          makeLabeled("clamp ", clampCheckbox),
        );
      }
    }
  }
  mode.addEventListener("change", render);
  render();

  return {
    element: row,
    getConfig(): WizardColorRule {
      if (mode.value === "constant") {
        return { mode: "constant", constant: colorInput.value };
      }
      if (mode.value === "byProperty") {
        if (propertySelect.value === "") return { mode: "default" };
        return {
          mode: "byProperty",
          property: propertySelect.value,
          colormap: colormapSelect.value as WizardColorRule["colormap"],
          clamp: clampCheckbox.checked,
        };
      }
      return { mode: "default" };
    },
    refreshProperties(properties) {
      populatePropertySelect(propertySelect, properties, "numeric");
      render();
    },
  };
}

function makeSizeRule(
  label: string,
  initialProps: readonly AnnotationPropertySpec[],
  defaults: { constant: number; outMin: number; outMax: number },
): RuleHandle<WizardSizeRule> {
  const { row, mode, detail } = makeRule(label);
  for (const [value, label] of [
    ["default", "Default"],
    ["constant", "Constant"],
    ["byProperty", "By property"],
  ] as const) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    mode.appendChild(option);
  }
  const constantInput = makeNumberInput(defaults.constant, { min: 0 });
  const propertySelect = makePropertySelect(initialProps, "numeric");
  const outMinInput = makeNumberInput(defaults.outMin, { min: 0 });
  const outMaxInput = makeNumberInput(defaults.outMax, { min: 0 });
  const clampCheckbox = document.createElement("input");
  clampCheckbox.type = "checkbox";
  clampCheckbox.checked = true;

  function render() {
    detail.replaceChildren();
    if (mode.value === "constant") {
      detail.append(makeLabeled("value ", constantInput));
    } else if (mode.value === "byProperty") {
      if (propertySelect.options.length === 0) {
        const note = document.createElement("span");
        note.classList.add("neuroglancer-annotation-wizard-note");
        note.textContent = "no numeric properties available";
        detail.append(note);
      } else {
        detail.append(
          makeLabeled("property ", propertySelect),
          makeLabeled("min ", outMinInput),
          makeLabeled("max ", outMaxInput),
          makeLabeled("clamp ", clampCheckbox),
        );
      }
    }
  }
  mode.addEventListener("change", render);
  render();

  return {
    element: row,
    getConfig(): WizardSizeRule {
      if (mode.value === "constant") {
        return {
          mode: "constant",
          constant: Number(constantInput.value),
        };
      }
      if (mode.value === "byProperty") {
        if (propertySelect.value === "") return { mode: "default" };
        return {
          mode: "byProperty",
          property: propertySelect.value,
          outputMin: Number(outMinInput.value),
          outputMax: Number(outMaxInput.value),
          clamp: clampCheckbox.checked,
        };
      }
      return { mode: "default" };
    },
    refreshProperties(properties) {
      populatePropertySelect(propertySelect, properties, "numeric");
      render();
    },
  };
}

function makeBorderRule(
  label: string,
  initialProps: readonly AnnotationPropertySpec[],
): RuleHandle<WizardBorderRule> {
  const { row, mode, detail } = makeRule(label);
  for (const [value, label] of [
    ["default", "Default"],
    ["remove", "Remove"],
    ["conditional", "Show conditionally"],
    ["colorByProperty", "Color by property"],
  ] as const) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    mode.appendChild(option);
  }
  const condSelect = makePropertySelect(initialProps, "numericOrBool");
  const widthInput = makeNumberInput(1, { min: 0 });
  const condClamp = document.createElement("input");
  condClamp.type = "checkbox";
  condClamp.checked = true;
  const colorSelect = makePropertySelect(initialProps, "numeric");
  const colormapSelect = makeColormapSelect("viridis");
  const colorClamp = document.createElement("input");
  colorClamp.type = "checkbox";
  colorClamp.checked = true;

  function render() {
    detail.replaceChildren();
    if (mode.value === "conditional") {
      if (condSelect.options.length === 0) {
        const note = document.createElement("span");
        note.classList.add("neuroglancer-annotation-wizard-note");
        note.textContent = "no numeric/bool properties available";
        detail.append(note);
      } else {
        detail.append(
          makeLabeled("show where ", condSelect),
          makeLabeled("width ", widthInput),
          makeLabeled("clamp ", condClamp),
        );
      }
    } else if (mode.value === "colorByProperty") {
      if (colorSelect.options.length === 0) {
        const note = document.createElement("span");
        note.classList.add("neuroglancer-annotation-wizard-note");
        note.textContent = "no numeric properties available";
        detail.append(note);
      } else {
        detail.append(
          makeLabeled("property ", colorSelect),
          makeLabeled("colormap ", colormapSelect),
          makeLabeled("clamp ", colorClamp),
        );
      }
    }
  }
  mode.addEventListener("change", render);
  render();

  return {
    element: row,
    getConfig(): WizardBorderRule {
      if (mode.value === "remove") return { mode: "remove" };
      if (mode.value === "conditional") {
        if (condSelect.value === "") return { mode: "default" };
        return {
          mode: "conditional",
          conditionProperty: condSelect.value,
          conditionClamp: condClamp.checked,
          showWidth: Number(widthInput.value),
        };
      }
      if (mode.value === "colorByProperty") {
        if (colorSelect.value === "") return { mode: "default" };
        return {
          mode: "colorByProperty",
          colorProperty: colorSelect.value,
          colorColormap:
            colormapSelect.value as WizardBorderRule["colorColormap"],
          colorClamp: colorClamp.checked,
        };
      }
      return { mode: "default" };
    },
    refreshProperties(properties) {
      populatePropertySelect(condSelect, properties, "numericOrBool");
      populatePropertySelect(colorSelect, properties, "numeric");
      render();
    },
  };
}

/**
 * Inline panel that helps users compose a GLSL annotation shader by clicking
 * through a small set of rules. Visibility is controlled by the `visible`
 * trackable so an external toggle (e.g. the magic-wand icon in the shader top
 * row) can show or hide the panel.
 *
 * On Generate, builds a new shader string and writes it to the layer's shader
 * trackable (overwriting whatever was there), then hides itself and minimizes
 * the code editor so the auto-generated #uicontrol widgets take focus.
 *
 * Wizard state is intentionally ephemeral: reopening starts from defaults; the
 * generated shader text is the persisted artifact.
 */
export class ShaderWizardWidget extends RefCounted {
  element: HTMLDivElement;
  visible = new TrackableBoolean(false);

  constructor(
    private shader: WatchableValueInterface<string>,
    private codeVisible: TrackableBoolean,
    private annotationProperties: WatchableValueInterface<
      readonly AnnotationPropertySpec[] | undefined
    >,
  ) {
    super();

    const container = (this.element = document.createElement("div"));
    container.classList.add("neuroglancer-annotation-wizard");

    const initialProps = annotationProperties.value ?? [];
    const colorRule = makeColorRule(initialProps);
    const pointSizeRule = makeSizeRule("Point size", initialProps, {
      constant: 5,
      outMin: 3,
      outMax: 15,
    });
    const lineWidthRule = makeSizeRule("Line width", initialProps, {
      constant: 1,
      outMin: 1,
      outMax: 6,
    });
    const pointBorderRule = makeBorderRule("Point border", initialProps);
    const boxBorderRule = makeBorderRule("Box border", initialProps);
    const rules = [
      colorRule,
      pointSizeRule,
      lineWidthRule,
      pointBorderRule,
      boxBorderRule,
    ];

    for (const rule of rules) container.appendChild(rule.element);

    const actions = document.createElement("div");
    actions.classList.add("neuroglancer-annotation-wizard-actions");
    const generateButton = document.createElement("button");
    generateButton.type = "button";
    generateButton.classList.add("neuroglancer-annotation-wizard-generate");
    generateButton.textContent = "Generate shader";
    actions.appendChild(generateButton);
    container.appendChild(actions);

    this.registerDisposer(
      new ElementVisibilityFromTrackableBoolean(this.visible, container),
    );

    this.registerDisposer(
      annotationProperties.changed.add(() => {
        const props = annotationProperties.value ?? [];
        for (const rule of rules) rule.refreshProperties(props);
      }),
    );

    generateButton.addEventListener("click", () => {
      const config: WizardConfig = {
        ...defaultWizardConfig(),
        color: colorRule.getConfig(),
        pointSize: pointSizeRule.getConfig(),
        lineWidth: lineWidthRule.getConfig(),
        pointBorder: pointBorderRule.getConfig(),
        boxBorder: boxBorderRule.getConfig(),
      };
      const props = this.annotationProperties.value ?? [];
      this.shader.value = generateWizardShader(config, props);
      this.codeVisible.value = false;
      this.visible.value = false;
    });
  }
}
