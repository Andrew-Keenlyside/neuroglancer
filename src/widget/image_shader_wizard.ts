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

import "#src/widget/image_shader_wizard.css";

import {
  defaultMultiChannelConfig,
  defaultSingleChannelConfig,
  generateImageWizardShader,
  type ChannelRule,
  type ColorTreatment,
  type MultiChannelConfig,
  type SingleChannelConfig,
  type WizardConfig,
  type WizardContext,
} from "#src/layer/image/shader_wizard_generator.js";
import type { ChannelSpace } from "#src/render_coordinate_transform.js";
import {
  ElementVisibilityFromTrackableBoolean,
  TrackableBoolean,
} from "#src/trackable_boolean.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import { valueOrThrow, type ValueOrError } from "#src/util/error.js";
import {
  COLORMAP_NAMES,
  type ColormapName,
  colormapDisplayName,
} from "#src/webgl/colormaps.js";
import type { ShaderControlState } from "#src/webgl/shader_ui_controls.js";
import { autoRangeInvlerpControls } from "#src/widget/shader_wizard_auto_range.js";

const DEFAULT_TINT = "#ffaa00";

function channelLabel(coords: readonly number[]): string {
  return coords.length === 1 ? String(coords[0]) : coords.join(",");
}

/**
 * Enumerates the channels of a (possibly multi-dim) channel coordinate space.
 * Returns an empty list when the channel space is invalid or rank=0.
 */
function enumerateChannels(channelSpace: ChannelSpace | undefined): number[][] {
  if (channelSpace === undefined) return [];
  const { coordinates, numChannels, channelCoordinateSpace } = channelSpace;
  const rank = channelCoordinateSpace.rank;
  if (rank === 0) return [];
  const result: number[][] = [];
  for (let i = 0; i < numChannels; i++) {
    const tuple: number[] = [];
    for (let d = 0; d < rank; d++) {
      tuple.push(coordinates[i * rank + d]);
    }
    result.push(tuple);
  }
  return result;
}

function makeLabeled(
  labelText: string,
  control: HTMLElement,
): HTMLLabelElement {
  const label = document.createElement("label");
  label.append(labelText, control);
  return label;
}

function makeColorInput(initial: string): HTMLInputElement {
  const input = document.createElement("input");
  input.type = "color";
  input.value = initial;
  return input;
}

function makeColormapSelect(initial: string): HTMLSelectElement {
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

/**
 * Colormap select for the transfer-function treatment. Includes a leading
 * "Manual (seed color)" entry (value ""); any other value is a colormap name.
 */
function makeTfColormapSelect(
  initial: ColormapName | undefined,
): HTMLSelectElement {
  const select = document.createElement("select");
  const manual = document.createElement("option");
  manual.value = "";
  manual.textContent = "Manual (seed color)";
  select.appendChild(manual);
  for (const name of COLORMAP_NAMES) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = colormapDisplayName(name);
    select.appendChild(option);
  }
  select.value = initial ?? "";
  return select;
}

function makeChannelSelect(
  channels: number[][],
  initial?: readonly number[],
): HTMLSelectElement {
  const select = document.createElement("select");
  for (const ch of channels) {
    const option = document.createElement("option");
    option.value = JSON.stringify(ch);
    option.textContent = channelLabel(ch);
    select.appendChild(option);
  }
  if (initial !== undefined) {
    const key = JSON.stringify(initial);
    if (channels.some((c) => JSON.stringify(c) === key)) {
      select.value = key;
    }
  }
  return select;
}

function readChannelSelect(select: HTMLSelectElement): number[] {
  if (select.value === "") return [];
  return JSON.parse(select.value) as number[];
}

interface TreatmentHandle {
  element: HTMLElement;
  getTreatment(): ColorTreatment;
}

/**
 * Builds the inline UI for a single color-treatment choice: a mode dropdown
 * plus the conditional sub-widget (color picker / colormap select / TF default
 * color picker).
 */
function makeTreatmentControl(
  initial: ColorTreatment = { type: "single-color", color: DEFAULT_TINT },
): TreatmentHandle {
  const container = document.createElement("span");
  container.style.display = "inline-flex";
  container.style.alignItems = "center";
  container.style.gap = "4px";
  const modeSelect = document.createElement("select");
  for (const [value, label] of [
    ["single-color", "Single color"],
    ["named-colormap", "Colormap"],
    ["transfer-function", "Transfer function"],
  ] as const) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label;
    modeSelect.appendChild(opt);
  }
  modeSelect.value = initial.type;

  const detail = document.createElement("span");
  detail.style.display = "inline-flex";
  detail.style.alignItems = "center";
  detail.style.gap = "4px";

  const colorInput = makeColorInput(
    initial.type === "single-color" ? initial.color : DEFAULT_TINT,
  );
  const colormapSelect = makeColormapSelect(
    initial.type === "named-colormap" ? initial.name : "viridis",
  );
  const tfColorInput = makeColorInput(
    initial.type === "transfer-function" ? initial.defaultColor : DEFAULT_TINT,
  );
  // Transfer-function coloring: a colormap dropdown (default viridis) with a
  // leading "Manual (seed color)" entry to opt out and color nodes by hand.
  const tfColormapSelect = makeTfColormapSelect(
    initial.type === "transfer-function" ? initial.colormap : "viridis",
  );

  function render() {
    detail.replaceChildren();
    switch (modeSelect.value) {
      case "single-color":
        detail.append(colorInput);
        break;
      case "named-colormap":
        detail.append(colormapSelect);
        break;
      case "transfer-function":
        detail.append(
          makeLabeled("colors ", tfColormapSelect),
          makeLabeled("seed color ", tfColorInput),
        );
        break;
    }
  }
  modeSelect.addEventListener("change", render);
  render();

  container.append(modeSelect, detail);

  return {
    element: container,
    getTreatment(): ColorTreatment {
      switch (modeSelect.value) {
        case "single-color":
          return { type: "single-color", color: colorInput.value };
        case "named-colormap":
          return {
            type: "named-colormap",
            name: colormapSelect.value as ColorTreatment extends {
              type: "named-colormap";
              name: infer N;
            }
              ? N
              : never,
          };
        case "transfer-function":
          return {
            type: "transfer-function",
            defaultColor: tfColorInput.value,
            colormap:
              tfColormapSelect.value === ""
                ? undefined
                : (tfColormapSelect.value as ColormapName),
          };
      }
      return { type: "single-color", color: DEFAULT_TINT };
    },
  };
}

interface SingleSectionHandle {
  element: HTMLElement;
  getConfig(): SingleChannelConfig;
  refreshChannels(channels: number[][]): void;
}

function makeSingleSection(
  initialChannels: number[][],
  context: WizardContext,
): SingleSectionHandle {
  const wrapper = document.createElement("div");
  wrapper.style.display = "flex";
  wrapper.style.flexDirection = "column";
  wrapper.style.gap = "6px";

  const defaults = defaultSingleChannelConfig(context);

  // Channel row.
  const channelRow = document.createElement("div");
  channelRow.classList.add("neuroglancer-image-wizard-row");
  const channelLabelEl = document.createElement("div");
  channelLabelEl.classList.add("neuroglancer-image-wizard-row-label");
  channelLabelEl.textContent = "Channel";
  const channelDetail = document.createElement("div");
  channelDetail.classList.add("neuroglancer-image-wizard-row-detail");
  const channelSelect = makeChannelSelect(initialChannels, defaults.channel);
  channelDetail.append(channelSelect);
  channelRow.append(channelLabelEl, channelSelect);

  // Color treatment row.
  const treatmentRow = document.createElement("div");
  treatmentRow.classList.add("neuroglancer-image-wizard-row");
  const treatmentLabelEl = document.createElement("div");
  treatmentLabelEl.classList.add("neuroglancer-image-wizard-row-label");
  treatmentLabelEl.textContent = "Color";
  const treatmentDetail = document.createElement("div");
  treatmentDetail.classList.add("neuroglancer-image-wizard-row-detail");
  const treatment = makeTreatmentControl(defaults.treatment);
  treatmentDetail.append(treatment.element);
  treatmentRow.append(treatmentLabelEl, treatmentDetail);

  // Alpha row.
  const alphaRow = document.createElement("div");
  alphaRow.classList.add("neuroglancer-image-wizard-row");
  const alphaLabelEl = document.createElement("div");
  alphaLabelEl.classList.add("neuroglancer-image-wizard-row-label");
  alphaLabelEl.textContent = "Alpha";
  const alphaDetail = document.createElement("div");
  alphaDetail.classList.add("neuroglancer-image-wizard-row-detail");
  const alpha2D = document.createElement("input");
  alpha2D.type = "checkbox";
  alpha2D.checked = defaults.alpha2D;
  const alpha3D = document.createElement("input");
  alpha3D.type = "checkbox";
  alpha3D.checked = defaults.alpha3D;
  const alphaSourceSelect = document.createElement("select");
  for (const [value, label] of [
    ["from-intensity", "Same as color (intensity)"],
    ["separate-invlerp", "Separate invlerp"],
  ] as const) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label;
    alphaSourceSelect.appendChild(opt);
  }
  alphaSourceSelect.value = defaults.alphaSource;
  const alphaSourceLabel = makeLabeled("source ", alphaSourceSelect);

  function renderAlpha() {
    alphaDetail.replaceChildren(
      makeLabeled("2D ", alpha2D),
      makeLabeled("3D ", alpha3D),
    );
    // Alpha source only applies when the treatment is not a TF.
    if (treatment.getTreatment().type !== "transfer-function") {
      alphaDetail.append(alphaSourceLabel);
    }
  }
  renderAlpha();
  treatment.element.addEventListener("change", renderAlpha);
  alphaRow.append(alphaLabelEl, alphaDetail);

  wrapper.append(channelRow, treatmentRow, alphaRow);

  function refreshChannels(channels: number[][]) {
    const previous = channelSelect.value;
    channelSelect.replaceChildren();
    for (const ch of channels) {
      const option = document.createElement("option");
      option.value = JSON.stringify(ch);
      option.textContent = channelLabel(ch);
      channelSelect.appendChild(option);
    }
    if (channels.some((c) => JSON.stringify(c) === previous)) {
      channelSelect.value = previous;
    }
    channelRow.style.display = channels.length === 0 ? "none" : "";
  }
  refreshChannels(initialChannels);

  return {
    element: wrapper,
    getConfig(): SingleChannelConfig {
      const t = treatment.getTreatment();
      const channels =
        channelSelect.options.length > 0
          ? readChannelSelect(channelSelect)
          : undefined;
      return {
        mode: "single",
        channel: channels && channels.length > 0 ? channels : undefined,
        treatment: t,
        alpha2D: alpha2D.checked,
        alpha3D: alpha3D.checked,
        alphaSource:
          alphaSourceSelect.value as SingleChannelConfig["alphaSource"],
      };
    },
    refreshChannels,
  };
}

interface RuleHandle {
  element: HTMLElement;
  getRule(): ChannelRule;
  refreshChannels(channels: number[][]): void;
}

function makeRule(
  initialChannels: number[][],
  initialRule: ChannelRule,
  onRemove: () => void,
): RuleHandle {
  const wrapper = document.createElement("div");
  wrapper.classList.add("neuroglancer-image-wizard-rule");
  const channelSelect = makeChannelSelect(initialChannels, initialRule.channel);
  const treatment = makeTreatmentControl(initialRule.treatment);
  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.classList.add("neuroglancer-image-wizard-rule-remove");
  removeButton.textContent = "Remove";
  removeButton.addEventListener("click", onRemove);
  wrapper.append(
    makeLabeled("channel ", channelSelect),
    treatment.element,
    removeButton,
  );

  function refreshChannels(channels: number[][]) {
    const previous = channelSelect.value;
    channelSelect.replaceChildren();
    for (const ch of channels) {
      const option = document.createElement("option");
      option.value = JSON.stringify(ch);
      option.textContent = channelLabel(ch);
      channelSelect.appendChild(option);
    }
    if (channels.some((c) => JSON.stringify(c) === previous)) {
      channelSelect.value = previous;
    }
  }

  return {
    element: wrapper,
    getRule(): ChannelRule {
      return {
        channel: readChannelSelect(channelSelect),
        treatment: treatment.getTreatment(),
      };
    },
    refreshChannels,
  };
}

interface MultiSectionHandle {
  element: HTMLElement;
  getConfig(): MultiChannelConfig;
  refreshChannels(channels: number[][]): void;
}

function makeMultiSection(
  initialChannels: number[][],
  context: WizardContext,
): MultiSectionHandle {
  const wrapper = document.createElement("div");
  wrapper.style.display = "flex";
  wrapper.style.flexDirection = "column";
  wrapper.style.gap = "6px";

  const defaults = defaultMultiChannelConfig(context);
  let currentChannels = initialChannels;

  // Rule list.
  const rulesRow = document.createElement("div");
  rulesRow.classList.add("neuroglancer-image-wizard-row");
  const rulesLabel = document.createElement("div");
  rulesLabel.classList.add("neuroglancer-image-wizard-row-label");
  rulesLabel.textContent = "Channels";
  const rulesDetail = document.createElement("div");
  rulesDetail.classList.add("neuroglancer-image-wizard-row-detail");
  const rulesList = document.createElement("div");
  rulesList.classList.add("neuroglancer-image-wizard-rules");
  const addButton = document.createElement("button");
  addButton.type = "button";
  addButton.classList.add("neuroglancer-image-wizard-add-rule");
  addButton.textContent = "+ Add channel";
  rulesDetail.append(rulesList, addButton);
  rulesRow.append(rulesLabel, rulesDetail);

  const rules: RuleHandle[] = [];

  function addRule(rule?: ChannelRule) {
    const seedRule: ChannelRule = rule ?? {
      channel:
        currentChannels.length > 0
          ? currentChannels[Math.min(rules.length, currentChannels.length - 1)]
          : [0],
      treatment: { type: "named-colormap", name: "viridis" },
    };
    const handle = makeRule(currentChannels, seedRule, () => {
      const idx = rules.indexOf(handle);
      if (idx === -1) return;
      rules.splice(idx, 1);
      handle.element.remove();
    });
    rules.push(handle);
    rulesList.appendChild(handle.element);
  }
  addButton.addEventListener("click", () => addRule());
  for (const r of defaults.rules) addRule(r);

  // Alpha row.
  const alphaRow = document.createElement("div");
  alphaRow.classList.add("neuroglancer-image-wizard-row");
  const alphaLabelEl = document.createElement("div");
  alphaLabelEl.classList.add("neuroglancer-image-wizard-row-label");
  alphaLabelEl.textContent = "Alpha";
  const alphaDetail = document.createElement("div");
  alphaDetail.classList.add("neuroglancer-image-wizard-row-detail");
  const alpha2D = document.createElement("input");
  alpha2D.type = "checkbox";
  alpha2D.checked = defaults.alpha2D;
  const alpha3D = document.createElement("input");
  alpha3D.type = "checkbox";
  alpha3D.checked = defaults.alpha3D;
  alphaDetail.append(makeLabeled("2D ", alpha2D), makeLabeled("3D ", alpha3D));
  alphaRow.append(alphaLabelEl, alphaDetail);

  wrapper.append(rulesRow, alphaRow);

  return {
    element: wrapper,
    getConfig(): MultiChannelConfig {
      return {
        mode: "multi",
        rules: rules.map((r) => r.getRule()),
        alpha2D: alpha2D.checked,
        alpha3D: alpha3D.checked,
      };
    },
    refreshChannels(channels) {
      currentChannels = channels;
      for (const rule of rules) rule.refreshChannels(channels);
    },
  };
}

/**
 * Inline panel that builds an image-layer shader from a small set of choices.
 * Visibility is controlled by the `visible` trackable so an external toggle
 * (the magic-wand icon in the shader top row) can show/hide it.
 *
 * On Generate, writes a complete shader to `fragmentMain.value`, sets
 * `codeVisible.value = false`, and hides the panel. State is ephemeral.
 */
export class ImageShaderWizardWidget extends RefCounted {
  element: HTMLDivElement;
  visible = new TrackableBoolean(false);

  constructor(
    private fragmentMain: WatchableValueInterface<string>,
    private codeVisible: TrackableBoolean,
    channelSpace: WatchableValueInterface<ValueOrError<ChannelSpace>>,
    private shaderControlState: ShaderControlState,
  ) {
    super();

    const container = (this.element = document.createElement("div"));
    container.classList.add("neuroglancer-image-wizard");

    const currentChannelSpace = (): ChannelSpace | undefined => {
      try {
        return valueOrThrow(channelSpace.value);
      } catch {
        return undefined;
      }
    };

    const cs = currentChannelSpace();
    const rank = cs?.channelCoordinateSpace.rank ?? 0;
    let channels = enumerateChannels(cs);

    const context: WizardContext = { channelRank: rank };

    // Mode selector (only relevant when there are channels).
    const modeRow = document.createElement("div");
    modeRow.classList.add("neuroglancer-image-wizard-row");
    const modeLabel = document.createElement("div");
    modeLabel.classList.add("neuroglancer-image-wizard-row-label");
    modeLabel.textContent = "Mode";
    const modeDetail = document.createElement("div");
    modeDetail.classList.add("neuroglancer-image-wizard-row-detail");
    const modeSelect = document.createElement("select");
    for (const [value, label] of [
      ["single", "Single channel"],
      ["multi", "Multi channel (additive)"],
    ] as const) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      modeSelect.appendChild(opt);
    }
    modeSelect.value = "single";
    modeDetail.append(modeSelect);
    modeRow.append(modeLabel, modeDetail);
    if (rank === 0) modeRow.style.display = "none";

    const singleSection = makeSingleSection(channels, context);
    const multiSection = makeMultiSection(channels, context);
    multiSection.element.style.display = "none";

    function renderMode() {
      const isMulti = modeSelect.value === "multi";
      singleSection.element.style.display = isMulti ? "none" : "";
      multiSection.element.style.display = isMulti ? "" : "none";
    }
    modeSelect.addEventListener("change", renderMode);
    renderMode();

    container.append(modeRow, singleSection.element, multiSection.element);

    const actions = document.createElement("div");
    actions.classList.add("neuroglancer-image-wizard-actions");
    const generateButton = document.createElement("button");
    generateButton.type = "button";
    generateButton.classList.add("neuroglancer-image-wizard-generate");
    generateButton.textContent = "Generate shader";
    actions.appendChild(generateButton);
    container.appendChild(actions);

    this.registerDisposer(
      new ElementVisibilityFromTrackableBoolean(this.visible, container),
    );

    this.registerDisposer(
      channelSpace.changed.add(() => {
        const newCs = currentChannelSpace();
        channels = enumerateChannels(newCs);
        singleSection.refreshChannels(channels);
        multiSection.refreshChannels(channels);
        const newRank = newCs?.channelCoordinateSpace.rank ?? 0;
        modeRow.style.display = newRank === 0 ? "none" : "";
        if (newRank === 0 && modeSelect.value === "multi") {
          modeSelect.value = "single";
          renderMode();
        }
      }),
    );

    generateButton.addEventListener("click", () => {
      const config: WizardConfig =
        modeSelect.value === "multi"
          ? multiSection.getConfig()
          : singleSection.getConfig();
      const cs2 = currentChannelSpace();
      const ctx: WizardContext = {
        channelRank: cs2?.channelCoordinateSpace.rank ?? 0,
      };
      this.fragmentMain.value = generateImageWizardShader(config, ctx);
      // The new invlerp controls default to the data-type range; ask them to
      // auto-range from the data that is currently loaded.
      autoRangeInvlerpControls(this.shaderControlState);
      this.codeVisible.value = false;
      this.visible.value = false;
    });
  }
}
