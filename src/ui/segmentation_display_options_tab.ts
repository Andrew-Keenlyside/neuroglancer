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
import { COLOR_BY_DIRECTION_SHADER } from "#src/skeleton/default_shader.js";
import { RefCounted } from "#src/util/disposable.js";
import { DependentViewWidget } from "#src/widget/dependent_view_widget.js";
import { addLayerControlToOptionsTab } from "#src/widget/layer_control.js";
import { LinkedLayerGroupWidget } from "#src/widget/linked_layer.js";
import { RangeWidget } from "#src/widget/range.js";
import {
  makeShaderCodeWidgetTopRow,
  ShaderCodeWidget,
} from "#src/widget/shader_code_widget.js";
import { ShaderControls } from "#src/widget/shader_controls.js";
import { Tab } from "#src/widget/tab_view.js";

function makeSkeletonShaderCodeWidget(layer: SegmentationUserLayer) {
  return new ShaderCodeWidget({
    fragmentMain: layer.displayState.skeletonRenderingOptions.shader,
    shaderError: layer.displayState.shaderError,
    shaderControlState:
      layer.displayState.skeletonRenderingOptions.shaderControlState,
  });
}

// --- background "Colour by" presets (tract layers) -------------------------
// A friendly way to set the per-vertex streamline shading without hand-writing
// GLSL. Each writes the persisted intent (roiFilter.backgroundColorBy) plus the
// shader text the code widget below shows (still fully hand-editable).

/** Seed a scalar attribute's colourmap range. For x/y/z, use the layer's axis
 *  bounds (a good default for spatial attributes like a per-vertex z); otherwise
 *  fall back to [0, 1] and let the user tune the Shader Controls sliders. */
function seedAttrRange(
  name: string,
  layer: SegmentationUserLayer,
): { lo: number; hi: number } {
  const n = name.toLowerCase();
  const axis = n === "x" ? 0 : n === "y" ? 1 : n === "z" ? 2 : undefined;
  if (axis !== undefined) {
    const cs = layer.manager.root.coordinateSpace.value;
    const lo = cs?.bounds?.lowerBounds?.[axis];
    const hi = cs?.bounds?.upperBounds?.[axis];
    if (
      lo !== undefined &&
      hi !== undefined &&
      Number.isFinite(lo) &&
      Number.isFinite(hi) &&
      hi > lo
    ) {
      return { lo, hi };
    }
  }
  return { lo: 0, hi: 1 };
}

/**
 * Colour-by-vertex-attribute shader preset. A vec3 attribute maps |components|
 * to RGB (like direction); a scalar goes through the jet colourmap over an
 * adjustable [lo, hi] range. There is no per-vertex-attribute histogram to
 * auto-range from, so the range is seeded (axis bounds for x/y/z, else [0,1])
 * and exposed as Shader Controls sliders the user can tune.
 */
function vertexAttrPreset(
  name: string,
  glslDataType: string,
  range: { lo: number; hi: number },
): string {
  if (glslDataType === "vec3") {
    return `void main() {\n  emitRGB(abs(prop_${name}()));\n}\n`;
  }
  const { lo, hi } = range;
  const extent = Math.max(Math.abs(lo), Math.abs(hi), 1) * 10;
  return `#uicontrol float lo slider(min=${-extent}, max=${extent}, default=${lo})
#uicontrol float hi slider(min=${-extent}, max=${extent}, default=${hi})
void main() {
  emitRGB(colormapJet((prop_${name}() - lo) / max(hi - lo, 1e-6)));
}
`;
}

/** A labelled control row (plain DOM; avoids importing the zarr-vectors CSS). */
function labelledRow(text: string, control: HTMLElement): HTMLElement {
  const label = document.createElement("label");
  label.style.display = "flex";
  label.style.gap = "4px";
  label.style.alignItems = "center";
  const span = document.createElement("span");
  span.textContent = text;
  label.appendChild(span);
  label.appendChild(control);
  return label;
}

function selectEl(
  options: { value: string; label: string }[],
  current: string,
  onChange: (v: string) => void,
): HTMLSelectElement {
  const select = document.createElement("select");
  for (const o of options) {
    const el = document.createElement("option");
    el.value = o.value;
    el.textContent = o.label;
    if (o.value === current) el.selected = true;
    select.appendChild(el);
  }
  select.addEventListener("change", () => onChange(select.value));
  return select;
}

/**
 * The parts of the background state that affect the control STRUCTURE (which
 * colour option is selected, which filter attribute is chosen, hence whether the
 * sliders are shown) — but NOT the min/max values, so dragging a slider does not
 * rebuild the section.
 */
function backgroundControlsSignature(layer: SegmentationUserLayer): string {
  const bg = layer.displayState.roiFilter.backgroundColorBy;
  const lf = layer.displayState.roiFilter.backgroundLengthFilter;
  const name = "name" in bg ? bg.name : "";
  // The detected numeric-attribute names are part of the signature: they load
  // ASYNCHRONOUSLY (after this section is first built), and both the colour-by
  // "Object:" options and the whole "Filter by" control depend on them, so the
  // section must rebuild once they arrive.
  const attrSig = (
    layer.displayState.segmentPropertyMap.value?.numericalProperties ?? []
  )
    .map((p) => p.id)
    .join(",");
  return `${attrSig}#${bg.kind}:${name}|${lf?.name ?? ""}`;
}

/**
 * Build the background (whole-tractogram) "Colour by" + "Filter by" controls,
 * mirroring the per-group controls on the Filter tab. Colour options:
 *  - per-vertex (Direction / a vertex attribute) install a shader preset
 *    (hand-editable in the code widget below);
 *  - per-object: a scalar numeric attribute → flat colourmap; a 3-component
 *    (vector) attribute → per-object RGB. Both handled by the shader's
 *    background value/colour tier.
 * "Filter by" picks any per-object numeric attribute and hides tracts outside
 * the min/max range (`roiFilter.backgroundLengthFilter`, enforced by the same
 * tier). Rebuilt by {@link appendBackgroundControls} when the selected colour
 * option / filter attribute changes.
 */
function buildBackgroundControls(
  layer: SegmentationUserLayer,
  skeletonLayer: {
    vertexAttributes: readonly { name: string; glslDataType: string }[];
  },
  ctx: RefCounted,
  container: HTMLElement,
): void {
  const { roiFilter } = layer.displayState;
  const attrs = skeletonLayer.vertexAttributes;
  const hasTangent = attrs.some((a) => a.name === "tangent");
  // Internal attributes the renderer synthesises — not user-selectable colour
  // sources. "tangent" is offered as "Direction (tangent)" instead.
  const INTERNAL_VERTEX_ATTRS = new Set([
    "",
    "position",
    "tangent",
    "segment",
    "selectedNodeAttr",
  ]);
  const vertexAttrs = attrs.filter((a) => !INTERNAL_VERTEX_ATTRS.has(a.name));
  const numericProps =
    layer.displayState.segmentPropertyMap.value?.numericalProperties ?? [];

  // --- Colour by (background) ---
  const options: { value: string; label: string }[] = [];
  if (hasTangent) {
    options.push({ value: "direction", label: "Direction (tangent)" });
  }
  for (const a of vertexAttrs) {
    options.push({ value: `vertex:${a.name}`, label: `Vertex: ${a.name}` });
  }
  for (const p of numericProps) {
    options.push({ value: `object:${p.id}`, label: `Object: ${p.id}` });
  }
  if (options.length > 0) {
    const spec = roiFilter.backgroundColorBy;
    const current =
      spec.kind === "vertexAttr"
        ? `vertex:${spec.name}`
        : spec.kind === "objectAttr"
          ? `object:${spec.name}`
          : "direction";
    const { shader } = layer.displayState.skeletonRenderingOptions;
    const select = selectEl(options, current, (v) => {
      if (v.startsWith("vertex:")) {
        const name = v.slice("vertex:".length);
        const glslDataType =
          vertexAttrs.find((a) => a.name === name)?.glslDataType ?? "float";
        roiFilter.backgroundColorBy = { kind: "vertexAttr", name };
        shader.value = vertexAttrPreset(
          name,
          glslDataType,
          seedAttrRange(name, layer),
        );
      } else if (v.startsWith("object:")) {
        // Flat per-object colour handled by the shader's background value tier;
        // the fragment-main preset is irrelevant for these tracts, so leave it.
        roiFilter.backgroundColorBy = {
          kind: "objectAttr",
          name: v.slice("object:".length),
        };
      } else {
        roiFilter.backgroundColorBy = { kind: "direction" };
        shader.value = COLOR_BY_DIRECTION_SHADER;
      }
    });
    container.appendChild(labelledRow("Colour by (background):", select));
  }

  // --- Filter by (background): any per-object numeric attribute, min/max ---
  if (numericProps.length === 0) return;
  const filter = roiFilter.backgroundLengthFilter;
  const filterSelect = selectEl(
    [
      { value: "", label: "None" },
      ...numericProps.map((p) => ({ value: p.id, label: p.id })),
    ],
    filter?.name ?? "",
    (name) => {
      if (name === "") {
        roiFilter.backgroundLengthFilter = undefined;
        return;
      }
      const p = numericProps.find((x) => x.id === name);
      if (p === undefined) return;
      const keep = filter?.name === name ? filter : undefined;
      roiFilter.backgroundLengthFilter = {
        name,
        min: keep?.min ?? Number(p.bounds[0]),
        max: keep?.max ?? Number(p.bounds[1]),
      };
    },
  );
  container.appendChild(labelledRow("Filter by (background):", filterSelect));

  const p =
    filter !== undefined
      ? numericProps.find((x) => x.id === filter.name)
      : undefined;
  if (p === undefined) return;
  const min = Number(p.bounds[0]);
  const max = Number(p.bounds[1]);
  const span = max - min;
  const step = span > 0 ? Math.max(span / 200, 1e-6) : 1;
  const current = () => {
    const f = roiFilter.backgroundLengthFilter;
    return f !== undefined && f.name === p.id ? f : undefined;
  };
  const set = (lo: number, hi: number) => {
    roiFilter.backgroundLengthFilter = { name: p.id, min: lo, max: hi };
  };
  const adapter = (get: () => number, put: (v: number) => void) => ({
    get value() {
      return get();
    },
    set value(v: number) {
      put(v);
    },
    changed: roiFilter.changed,
  });
  const lo = ctx.registerDisposer(
    new RangeWidget(
      adapter(
        () => current()?.min ?? min,
        (v) => set(v, current()?.max ?? max),
      ),
      { min, max, step },
    ),
  );
  const hi = ctx.registerDisposer(
    new RangeWidget(
      adapter(
        () => current()?.max ?? max,
        (v) => set(current()?.min ?? min, v),
      ),
      { min, max, step },
    ),
  );
  container.appendChild(labelledRow(`${p.id} ≥ (background):`, lo.element));
  container.appendChild(labelledRow(`${p.id} ≤ (background):`, hi.element));
}

/**
 * Mount the background controls in a container that rebuilds when the selected
 * colour option or filter attribute changes (so the sliders track the chosen
 * attribute) — but not on a min/max drag.
 */
function appendBackgroundControls(
  layer: SegmentationUserLayer,
  skeletonLayer: {
    vertexAttributes: readonly { name: string; glslDataType: string }[];
  },
  refCounted: RefCounted,
  parent: HTMLElement,
): void {
  const { roiFilter } = layer.displayState;
  const container = document.createElement("div");
  parent.appendChild(container);
  let ctx: RefCounted | undefined;
  let sig = "";
  const rebuild = () => {
    ctx?.dispose();
    ctx = new RefCounted();
    container.textContent = "";
    buildBackgroundControls(layer, skeletonLayer, ctx, container);
  };
  const maybeRebuild = () => {
    const s = backgroundControlsSignature(layer);
    if (s === sig) return;
    sig = s;
    rebuild();
  };
  sig = backgroundControlsSignature(layer);
  rebuild();
  refCounted.registerDisposer(roiFilter.changed.add(maybeRebuild));
  // Rebuild when the per-object attributes finish loading (they arrive after
  // this section is first built), so the colour "Object:" options and the
  // "Filter by" control appear without needing to reopen the tab.
  refCounted.registerDisposer(
    layer.displayState.segmentPropertyMap.changed.add(maybeRebuild),
  );
  refCounted.registerDisposer(() => ctx?.dispose());
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
          // Tract layers: background "Colour by" + length filter above the raw
          // shader editor.
          if (layer.hasSpatiallyIndexedSkeletonsLayer.value) {
            appendBackgroundControls(layer, skeletonLayer, refCounted, parent);
          }
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
          parent.appendChild(
            makeShaderCodeWidgetTopRow(
              this.layer,
              codeWidget,
              ShaderCodeOverlay,
              {
                title: "Documentation on image layer rendering",
                href: "https://github.com/google/neuroglancer/blob/master/src/sliceview/image_layer_rendering.md",
              },
              "neuroglancer-segmentation-dropdown-skeleton-shader-header",
            ),
          );
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
