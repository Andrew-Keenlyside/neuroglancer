/**
 * @license
 * Copyright 2024 Google Inc.
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
 * @file UI widget for per-layer ROI bounding box controls (Annotations tab).
 */

import type { TrackableRoiBoxState } from "#src/roi_box.js";
import { TrackableBooleanCheckbox } from "#src/trackable_boolean.js";
import { animationFrameDebounce } from "#src/util/animation_frame_debounce.js";
import { RefCounted } from "#src/util/disposable.js";
import { formatScaleWithUnitAsString, parseScale } from "#src/util/si_units.js";
import { ColorWidget } from "#src/widget/color.js";

const AXIS_NAMES = ["x", "y", "z"] as const;

/**
 * Parse a user-typed physical size string (e.g. "500nm", "1μm", "-200nm") to
 * SI meters. Returns `undefined` if the string cannot be parsed.
 */
function parseSiMeters(s: string): number | undefined {
  const trimmed = s.trim();
  const neg = trimmed.startsWith("-");
  const abs = neg ? trimmed.slice(1).trimStart() : trimmed;
  const result = parseScale(abs);
  if (result === undefined) return undefined;
  if (result.scale === 0) return 0;
  if (result.unit !== "m") return undefined;
  return neg ? -result.scale : result.scale;
}

/** Format SI meters for display. Returns "0" for exactly zero. */
function formatSiMeters(meters: number): string {
  if (meters === 0) return "0";
  const sign = meters < 0 ? "-" : "";
  return (
    sign + formatScaleWithUnitAsString(Math.abs(meters), "m", { precision: 3 })
  );
}

/**
 * Per-layer ROI box control section rendered inside the Annotations tab.
 * Exposes: enable toggle, edit-mode toggle, per-axis size inputs with uniform
 * lock, follow-view toggle, zoom-relative toggle.
 */
export class RoiBoxWidget extends RefCounted {
  element: HTMLDivElement;

  constructor(private readonly state: TrackableRoiBoxState) {
    super();
    this.element = document.createElement("div");
    this.element.classList.add("neuroglancer-roi-box-widget");
    this.build();
  }

  private build() {
    const { element, state } = this;

    // ── Header: [✓] ROI Box  [□] Edit ────────────────────────────────────
    const headerRow = this.makeRow("neuroglancer-roi-box-header");

    const enableCb = this.registerDisposer(
      new TrackableBooleanCheckbox(state.enabled, {
        enabledTitle: "Disable ROI box",
        disabledTitle: "Enable ROI box",
      }),
    );
    const enableLabel = document.createElement("label");
    enableLabel.classList.add("neuroglancer-roi-box-enable-label");
    enableLabel.appendChild(enableCb.element);
    enableLabel.appendChild(document.createTextNode(" ROI Box"));
    headerRow.appendChild(enableLabel);

    const showCb = this.registerDisposer(
      new TrackableBooleanCheckbox(state.showBox, {
        enabledTitle: "Hide box outline (loading restriction stays active)",
        disabledTitle: "Show box outline",
      }),
    );
    const showLabel = document.createElement("label");
    showLabel.classList.add("neuroglancer-roi-box-show-label");
    showLabel.appendChild(showCb.element);
    showLabel.appendChild(document.createTextNode(" Show"));
    headerRow.appendChild(showLabel);

    const colorWidget = this.registerDisposer(new ColorWidget(state.color));
    colorWidget.element.title = "Box outline color";
    headerRow.appendChild(colorWidget.element);

    const editCb = this.registerDisposer(
      new TrackableBooleanCheckbox(state.editActive, {
        enabledTitle: "Exit ROI-edit mode",
        disabledTitle: "Enter ROI-edit mode (Alt-drag faces to resize)",
      }),
    );
    const editLabel = document.createElement("label");
    editLabel.classList.add("neuroglancer-roi-box-edit-label");
    editLabel.appendChild(editCb.element);
    editLabel.appendChild(document.createTextNode(" Edit"));
    headerRow.appendChild(editLabel);

    element.appendChild(headerRow);

    // ── Size: [x____] [y____] [z____] [□ uniform] ────────────────────────
    const sizeRow = this.makeRow();
    const sizeHeading = document.createElement("span");
    sizeHeading.classList.add("neuroglancer-roi-box-row-label");
    sizeHeading.textContent = "Size:";
    sizeRow.appendChild(sizeHeading);

    const sizeInputs: HTMLInputElement[] = [];
    for (let i = 0; i < 3; i++) {
      const inp = this.makeSizeInput(i);
      sizeInputs.push(inp);
      sizeRow.appendChild(inp);
    }

    const uniformCb = this.registerDisposer(
      new TrackableBooleanCheckbox(state.uniformSize, {
        enabledTitle: "All axes locked to the same size — click to unlock",
        disabledTitle: "Per-axis sizes — click to lock uniform",
      }),
    );
    const uniformLabel = document.createElement("label");
    uniformLabel.classList.add("neuroglancer-roi-box-uniform-label");
    uniformLabel.title = "Lock x/y/z size to the same value (uniform scaling)";
    uniformLabel.appendChild(uniformCb.element);
    uniformLabel.appendChild(document.createTextNode(" uniform"));
    sizeRow.appendChild(uniformLabel);

    element.appendChild(sizeRow);

    // Gray out y/z inputs when uniform lock is on.
    const updateUniformUI = () => {
      const locked = state.uniformSize.value;
      for (let i = 1; i < 3; i++) {
        sizeInputs[i].disabled = locked;
        sizeInputs[i].title = locked
          ? "Uniform size — edit x to change all axes"
          : AXIS_NAMES[i] + " size";
      }
    };
    this.registerDisposer(state.uniformSize.changed.add(updateUniformUI));
    updateUniformUI();

    // ── Options: [□] Follow view   [□] Zoom-relative ───────────────────
    const optionsRow = this.makeRow();

    const followCb = this.registerDisposer(
      new TrackableBooleanCheckbox(state.followNavCenter, {
        enabledTitle:
          "Box center follows navigation position — click to freeze in place",
        disabledTitle:
          "Box center frozen in place — click to follow navigation",
      }),
    );
    const followLabel = document.createElement("label");
    followLabel.appendChild(followCb.element);
    followLabel.appendChild(document.createTextNode(" Follow view"));
    optionsRow.appendChild(followLabel);

    const zoomCb = this.registerDisposer(
      new TrackableBooleanCheckbox(state.zoomRelative, {
        enabledTitle:
          "Size is zoom-relative (constant screen size) — click to fix physical size",
        disabledTitle:
          "Size is physical — click for zoom-relative (constant screen size)",
      }),
    );
    const zoomLabel = document.createElement("label");
    zoomLabel.appendChild(zoomCb.element);
    zoomLabel.appendChild(document.createTextNode(" Zoom-relative"));
    optionsRow.appendChild(zoomLabel);

    element.appendChild(optionsRow);
  }

  private makeRow(extraClass?: string): HTMLDivElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-roi-box-row");
    if (extraClass) row.classList.add(extraClass);
    return row;
  }

  private makeSizeInput(axis: number): HTMLInputElement {
    const { state } = this;
    const input = document.createElement("input");
    input.type = "text";
    input.classList.add("neuroglancer-roi-box-size-input");
    input.spellcheck = false;
    input.autocomplete = "off";
    input.title = AXIS_NAMES[axis] + " size";

    const updateView = this.registerCancellable(
      animationFrameDebounce(() => {
        if (document.activeElement === input) return;
        const v = state.physicalSize.value[axis] ?? 1e-6;
        input.value = formatSiMeters(v);
      }),
    );

    this.registerDisposer(state.physicalSize.changed.add(updateView));
    updateView();

    const commit = () => {
      const meters = parseSiMeters(input.value);
      if (meters === null || meters === undefined || meters <= 0) {
        updateView();
        return;
      }
      state.setSizeComponent(axis, meters);
    };
    input.addEventListener("change", commit);
    input.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter") {
        commit();
        input.blur();
      } else if (e.key === "Escape") {
        updateView();
        input.blur();
      }
    });
    input.addEventListener("blur", commit);

    return input;
  }
}
