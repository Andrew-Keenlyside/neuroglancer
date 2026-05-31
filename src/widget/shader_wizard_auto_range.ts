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
 * @file Shared helper used by the image and annotation shader wizards.
 *
 * After a wizard replaces the layer's shader, the freshly parsed invlerp
 * controls default to the data-type range (e.g. [0, 1] for float32, [0, 255]
 * for uint8) rather than the range of the data that is actually loaded. Setting
 * `autoCompute: true` on each invlerp control asks the AutoRangeFinder (the same
 * machinery behind the "1-99%" button and `multi_channel_setup`) to recompute
 * the range from the current data histogram once the control's widget renders.
 *
 * Transfer-function controls are intentionally not touched here: the wizard
 * emits them without explicit control points or window, which lets the
 * transfer-function widget auto-range and seed default control points on its
 * own.
 */

import type { ShaderControlState } from "#src/webgl/shader_ui_controls.js";

function applyAutoCompute(shaderControlState: ShaderControlState): boolean {
  const controls = shaderControlState.controls.value;
  if (controls === undefined) return false;
  for (const { control, trackable } of shaderControlState.state.values()) {
    if (control.type === "imageInvlerp" || control.type === "propertyInvlerp") {
      trackable.value = { ...trackable.value, autoCompute: true };
    }
  }
  return true;
}

/**
 * Marks every invlerp control in `shaderControlState` for auto-ranging from the
 * currently loaded data. Safe to call immediately after assigning a new shader:
 * if the controls have not finished parsing yet, the work is deferred until they
 * have (a single one-shot retry).
 */
export function autoRangeInvlerpControls(
  shaderControlState: ShaderControlState,
): void {
  if (applyAutoCompute(shaderControlState)) return;
  const unsubscribe = shaderControlState.controls.changed.add(() => {
    if (applyAutoCompute(shaderControlState)) {
      unsubscribe();
    }
  });
}
