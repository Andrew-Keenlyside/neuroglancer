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

import "#src/widget/skeleton_shader_wizard.css";

import {
  additiveDefaultColor,
  defaultSkeletonWizardConfig,
  generateSkeletonWizardShader,
  type SkeletonColorChannel,
  type SkeletonColorRule,
  type SkeletonWidthRule,
  type SkeletonWizardConfig,
} from "#src/skeleton/shader_wizard_generator.js";
import {
  ElementVisibilityFromTrackableBoolean,
  TrackableBoolean,
} from "#src/trackable_boolean.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import {
  COLORMAP_NAMES,
  type ColormapName,
  colormapDisplayName,
} from "#src/webgl/colormaps.js";
import type { ShaderControlState } from "#src/webgl/shader_ui_controls.js";
import { autoRangeInvlerpControls } from "#src/widget/shader_wizard_auto_range.js";

interface RuleHandle<T> {
  element: HTMLElement;
  getConfig(): T;
  refreshProperties(properties: readonly string[]): void;
}

function populatePropertySelect(
  select: HTMLSelectElement,
  properties: readonly string[],
): void {
  const previous = select.value;
  select.replaceChildren();
  for (const name of properties) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }
  if (properties.includes(previous)) {
    select.value = previous;
  }
  select.disabled = properties.length === 0;
}

function makePropertySelect(properties: readonly string[]): HTMLSelectElement {
  const select = document.createElement("select");
  populatePropertySelect(select, properties);
  return select;
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

function makeNumberInput(defaultValue: number): HTMLInputElement {
  const input = document.createElement("input");
  input.type = "number";
  input.min = "0";
  input.step = "any";
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
  row.classList.add("neuroglancer-skeleton-wizard-row");
  const labelEl = document.createElement("div");
  labelEl.classList.add("neuroglancer-skeleton-wizard-row-label");
  labelEl.textContent = label;
  const mode = document.createElement("select");
  const detail = document.createElement("div");
  detail.classList.add("neuroglancer-skeleton-wizard-row-detail");
  row.append(labelEl, mode, detail);
  return { row, mode, detail };
}

function makeNote(text: string): HTMLSpanElement {
  const note = document.createElement("span");
  note.classList.add("neuroglancer-skeleton-wizard-note");
  note.textContent = text;
  return note;
}

interface AdditiveChannelHandle {
  element: HTMLElement;
  propertySelect: HTMLSelectElement;
  colorInput: HTMLInputElement;
  clampCheckbox: HTMLInputElement;
}

function makeColorRule(
  initialProps: readonly string[],
): RuleHandle<SkeletonColorRule> {
  const { row, mode, detail } = makeRule("Color");
  for (const [value, label] of [
    ["default", "Default (segment color)"],
    ["constant", "Constant color"],
    ["byProperty", "By property + colormap"],
    ["additive", "Additive (properties × colors)"],
  ] as const) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    mode.appendChild(option);
  }
  const colorInput = makeColorInput("#ffaa00");
  const propertySelect = makePropertySelect(initialProps);
  const colormapSelect = makeColormapSelect("viridis");
  const clampCheckbox = document.createElement("input");
  clampCheckbox.type = "checkbox";
  clampCheckbox.checked = true;

  let currentProps = initialProps;
  const channels: AdditiveChannelHandle[] = [];
  const additiveList = document.createElement("div");
  additiveList.classList.add("neuroglancer-skeleton-wizard-channels");
  const addChannelButton = document.createElement("button");
  addChannelButton.type = "button";
  addChannelButton.classList.add("neuroglancer-skeleton-wizard-add-channel");
  addChannelButton.textContent = "+ Add color";

  function defaultPropertyFor(index: number): string {
    if (currentProps.length === 0) return "";
    return currentProps[Math.min(index, currentProps.length - 1)];
  }

  function makeChannel(color: string, property: string): AdditiveChannelHandle {
    const element = document.createElement("div");
    element.classList.add("neuroglancer-skeleton-wizard-channel");
    const channelPropertySelect = makePropertySelect(currentProps);
    if (property !== "") channelPropertySelect.value = property;
    const channelColorInput = makeColorInput(color);
    const channelClamp = document.createElement("input");
    channelClamp.type = "checkbox";
    channelClamp.checked = true;
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.classList.add("neuroglancer-skeleton-wizard-channel-remove");
    removeButton.textContent = "Remove";
    const handle: AdditiveChannelHandle = {
      element,
      propertySelect: channelPropertySelect,
      colorInput: channelColorInput,
      clampCheckbox: channelClamp,
    };
    removeButton.addEventListener("click", () => {
      const idx = channels.indexOf(handle);
      if (idx === -1) return;
      channels.splice(idx, 1);
      element.remove();
    });
    element.append(
      makeLabeled("property ", channelPropertySelect),
      makeLabeled("color ", channelColorInput),
      makeLabeled("clamp ", channelClamp),
      removeButton,
    );
    return handle;
  }

  function addChannel() {
    const index = channels.length;
    const handle = makeChannel(
      additiveDefaultColor(index),
      defaultPropertyFor(index),
    );
    channels.push(handle);
    additiveList.appendChild(handle.element);
  }
  addChannelButton.addEventListener("click", () => addChannel());
  for (let i = 0; i < 3; ++i) addChannel();

  function render() {
    detail.replaceChildren();
    if (mode.value === "constant") {
      detail.append(makeLabeled("color ", colorInput));
    } else if (mode.value === "byProperty") {
      if (currentProps.length === 0) {
        detail.append(makeNote("no scalar vertex properties available"));
      } else {
        detail.append(
          makeLabeled("property ", propertySelect),
          makeLabeled("colormap ", colormapSelect),
          makeLabeled("clamp ", clampCheckbox),
        );
      }
    } else if (mode.value === "additive") {
      if (currentProps.length === 0) {
        detail.append(makeNote("no scalar vertex properties available"));
      } else {
        detail.append(additiveList, addChannelButton);
      }
    }
  }
  mode.addEventListener("change", render);
  render();

  return {
    element: row,
    getConfig(): SkeletonColorRule {
      if (mode.value === "constant") {
        return { mode: "constant", constant: colorInput.value };
      }
      if (mode.value === "byProperty") {
        if (propertySelect.value === "") return { mode: "default" };
        return {
          mode: "byProperty",
          property: propertySelect.value,
          colormap: colormapSelect.value as ColormapName,
          clamp: clampCheckbox.checked,
        };
      }
      if (mode.value === "additive") {
        const ruleChannels: SkeletonColorChannel[] = [];
        for (const channel of channels) {
          if (channel.propertySelect.value === "") continue;
          ruleChannels.push({
            property: channel.propertySelect.value,
            color: channel.colorInput.value,
            clamp: channel.clampCheckbox.checked,
          });
        }
        if (ruleChannels.length === 0) return { mode: "default" };
        return { mode: "additive", channels: ruleChannels };
      }
      return { mode: "default" };
    },
    refreshProperties(properties) {
      currentProps = properties;
      populatePropertySelect(propertySelect, properties);
      for (const channel of channels) {
        populatePropertySelect(channel.propertySelect, properties);
      }
      render();
    },
  };
}

function makeWidthRule(
  initialProps: readonly string[],
): RuleHandle<SkeletonWidthRule> {
  const { row, mode, detail } = makeRule("Line width");
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
  let currentProps = initialProps;
  const constantInput = makeNumberInput(3);
  const propertySelect = makePropertySelect(initialProps);
  const outMinInput = makeNumberInput(1);
  const outMaxInput = makeNumberInput(6);
  const clampCheckbox = document.createElement("input");
  clampCheckbox.type = "checkbox";
  clampCheckbox.checked = true;

  function render() {
    detail.replaceChildren();
    if (mode.value === "constant") {
      detail.append(makeLabeled("width ", constantInput));
    } else if (mode.value === "byProperty") {
      if (currentProps.length === 0) {
        detail.append(makeNote("no scalar vertex properties available"));
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
    getConfig(): SkeletonWidthRule {
      if (mode.value === "constant") {
        return { mode: "constant", constant: Number(constantInput.value) };
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
      currentProps = properties;
      populatePropertySelect(propertySelect, properties);
      render();
    },
  };
}

/**
 * Inline panel that helps users compose a skeleton GLSL shader by clicking
 * through color + line-width rules. Visibility is controlled by the `visible`
 * trackable so an external toggle (the magic-wand icon) can show/hide it.
 *
 * On Generate, builds a new shader string, writes it to the skeleton shader
 * trackable, minimizes the code editor, and hides the panel. Wizard state is
 * ephemeral; the generated shader text is the persisted artifact.
 */
export class SkeletonShaderWizardWidget extends RefCounted {
  element: HTMLDivElement;
  visible = new TrackableBoolean(false);

  constructor(
    private shader: WatchableValueInterface<string>,
    private codeVisible: TrackableBoolean,
    private properties: WatchableValueInterface<readonly string[]>,
    private shaderControlState: ShaderControlState,
  ) {
    super();

    const container = (this.element = document.createElement("div"));
    container.classList.add("neuroglancer-skeleton-wizard");

    const initialProps = properties.value ?? [];
    const colorRule = makeColorRule(initialProps);
    const widthRule = makeWidthRule(initialProps);
    const rules = [colorRule, widthRule];

    for (const rule of rules) container.appendChild(rule.element);

    const actions = document.createElement("div");
    actions.classList.add("neuroglancer-skeleton-wizard-actions");
    const generateButton = document.createElement("button");
    generateButton.type = "button";
    generateButton.classList.add("neuroglancer-skeleton-wizard-generate");
    generateButton.textContent = "Generate shader";
    actions.appendChild(generateButton);
    container.appendChild(actions);

    this.registerDisposer(
      new ElementVisibilityFromTrackableBoolean(this.visible, container),
    );

    this.registerDisposer(
      properties.changed.add(() => {
        const props = properties.value ?? [];
        for (const rule of rules) rule.refreshProperties(props);
      }),
    );

    generateButton.addEventListener("click", () => {
      const config: SkeletonWizardConfig = {
        ...defaultSkeletonWizardConfig(),
        color: colorRule.getConfig(),
        width: widthRule.getConfig(),
      };
      const props = this.properties.value ?? [];
      this.shader.value = generateSkeletonWizardShader(config, props);
      // The new invlerp controls default to the data-type range; ask them to
      // auto-range from the loaded data (now that skeleton property histograms
      // are computed, this resolves to the real data range).
      autoRangeInvlerpControls(this.shaderControlState);
      this.codeVisible.value = false;
      this.visible.value = false;
    });
  }
}
