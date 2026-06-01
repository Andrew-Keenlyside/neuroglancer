/**
 * @license
 * Copyright 2016 Google Inc.
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

import { buildShaderPropertyList } from "#src/layer/annotation/shader_ui_property_list.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID } from "#src/layer/segmentation/json_keys.js";
import { LAYER_CONTROLS } from "#src/layer/segmentation/layer_controls.js";
import { Overlay } from "#src/overlay.js";
import { WatchableValue } from "#src/trackable_value.js";
import { CheckboxIcon } from "#src/widget/checkbox_icon.js";
import { DependentViewWidget } from "#src/widget/dependent_view_widget.js";
import { addLayerControlToOptionsTab } from "#src/widget/layer_control.js";
import { LinkedLayerGroupWidget } from "#src/widget/linked_layer.js";
import {
  makeShaderCodeWidgetTopRow,
  ShaderCodeWidget,
} from "#src/widget/shader_code_widget.js";
import { ShaderControls } from "#src/widget/shader_controls.js";
import { SkeletonShaderWizardWidget } from "#src/widget/skeleton_shader_wizard.js";
import { Tab } from "#src/widget/tab_view.js";

function makeSkeletonShaderCodeWidget(layer: SegmentationUserLayer) {
  return new ShaderCodeWidget({
    fragmentMain: layer.displayState.skeletonRenderingOptions.shader,
    shaderError: layer.displayState.shaderError,
    shaderControlState:
      layer.displayState.skeletonRenderingOptions.shaderControlState,
  });
}

export class DisplayOptionsTab extends Tab {
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { element } = this;
    element.classList.add("neuroglancer-segmentation-rendering-tab");

    // Linked segmentation control
    {
      const widget = this.registerDisposer(
        new LinkedLayerGroupWidget(layer.displayState.linkedSegmentationGroup),
      );
      widget.label.textContent = "Linked to: ";
      element.appendChild(widget.element);
    }

    // Linked segmentation control
    {
      const widget = this.registerDisposer(
        new LinkedLayerGroupWidget(
          layer.displayState.linkedSegmentationColorGroup,
        ),
      );
      widget.label.textContent = "Colors linked to: ";
      element.appendChild(widget.element);
    }

    for (const control of LAYER_CONTROLS) {
      element.appendChild(
        addLayerControlToOptionsTab(this, layer, this.visibility, control),
      );
    }

    const skeletonControls = this.registerDisposer(
      new DependentViewWidget(
        layer.hasSkeletonsLayer,
        (hasSkeletonsLayer, parent, refCounted) => {
          if (!hasSkeletonsLayer) return;
          const skeletonLayer = layer.getSkeletonLayer()!;
          if (skeletonLayer.vertexAttributes.length > 1) {
            buildShaderPropertyList(
              skeletonLayer.vertexAttributes.slice(1).map((x) => {
                return {
                  type: x.glslDataType,
                  identifier: x.name,
                };
              }),
              parent,
            );
          }
          const codeWidget = refCounted.registerDisposer(
            makeSkeletonShaderCodeWidget(this.layer),
          );
          // Scalar (single-component) vertex attributes are the properties the
          // wizard's invlerp controls can bind to.
          const scalarProperties = skeletonLayer.vertexAttributes
            .slice(1)
            .filter((x) => x.numComponents === 1)
            .map((x) => x.name);
          const wizard = refCounted.registerDisposer(
            new SkeletonShaderWizardWidget(
              layer.displayState.skeletonRenderingOptions.shader,
              layer.codeVisible,
              new WatchableValue<readonly string[]>(scalarProperties),
              layer.displayState.skeletonRenderingOptions.shaderControlState,
            ),
          );
          const topRow = makeShaderCodeWidgetTopRow(
            this.layer,
            codeWidget,
            ShaderCodeOverlay,
            {
              title: "Documentation on image layer rendering",
              href: "https://github.com/google/neuroglancer/blob/master/src/sliceview/image_layer_rendering.md",
            },
            "neuroglancer-segmentation-dropdown-skeleton-shader-header",
          );
          const wandButton = refCounted.registerDisposer(
            new CheckboxIcon(wizard.visible, {
              svg: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M5 19L12 12"/><path d="M16 2V14"/><path d="M10 8H22"/></svg>`,
              text: "Wizard",
              enableTitle: "Open shader wizard",
              disableTitle: "Close shader wizard",
              backgroundScheme: "dark",
            }),
          ).element;
          wandButton.style.gap = "4px";
          wandButton.style.padding = "1px 8px";
          wandButton.style.marginRight = "4px";
          wandButton.style.border = "1px solid #555";
          refCounted.registerDisposer(
            wizard.visible.changed.add(() => {
              if (wizard.visible.value) {
                layer.codeVisible.value = false;
              }
            }),
          );
          topRow.insertBefore(wandButton, topRow.children[1] ?? null);
          parent.appendChild(topRow);
          parent.appendChild(wizard.element);
          parent.appendChild(codeWidget.element);
          parent.appendChild(
            refCounted.registerDisposer(
              new ShaderControls(
                layer.displayState.skeletonRenderingOptions.shaderControlState,
                this.layer.manager.root.display,
                this.layer,
                {
                  visibility: this.visibility,
                  toolId: SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID,
                },
              ),
            ).element,
          );
          codeWidget.textEditor.refresh();
        },
        this.visibility,
      ),
    );
    element.appendChild(skeletonControls.element);
  }
}

class ShaderCodeOverlay extends Overlay {
  codeWidget: ShaderCodeWidget;
  constructor(public layer: SegmentationUserLayer) {
    super();
    this.codeWidget = this.registerDisposer(
      makeSkeletonShaderCodeWidget(layer),
    );
    this.content.classList.add(
      "neuroglancer-segmentation-layer-skeleton-shader-overlay",
    );
    this.content.appendChild(this.codeWidget.element);
    this.codeWidget.textEditor.refresh();
  }
}
