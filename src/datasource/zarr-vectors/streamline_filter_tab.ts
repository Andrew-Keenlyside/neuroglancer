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
 * @file The "Filter" tab: authoring UI for the TrackVis-style ROI streamline
 * filter. Organises ROIs into coloured GROUPS (each a dissection); each group's
 * passing streamlines are shown in its colour, everything else ghosted. New
 * ROIs are placed and edited entirely from this tab (sliders for centre +
 * radius / size), not by dragging. All edits write into the layer's
 * `RoiFilterState`, which drives the filter + overlays and round-trips to the
 * URL.
 */

import "#src/datasource/zarr-vectors/streamline_filter_tab.css";

import {
  RoiOperator,
  RoiPredicate,
  type Roi,
  type RoiShape,
} from "#src/datasource/zarr-vectors/roi.js";
import type {
  RoiFilterState,
  RoiGroup,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { TrackableBooleanCheckbox } from "#src/trackable_boolean.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import type { Uint64Set } from "#src/uint64_set.js";
import { serializeColor } from "#src/util/color.js";
import { RefCounted } from "#src/util/disposable.js";
import { vec3 } from "#src/util/geom.js";
import { makeDeleteButton } from "#src/widget/delete_button.js";
import { makeIcon } from "#src/widget/icon.js";
import { RangeWidget } from "#src/widget/range.js";
import { Tab } from "#src/widget/tab_view.js";

/** Default sphere radius / box half-extent (model units) for a new ROI. */
const DEFAULT_ROI_RADIUS = 10;
/** Default plane thickness (model units). */
const DEFAULT_PLANE_THICKNESS = 1;

/** Adapt a scalar getter/setter to the `WatchableValueInterface` the range/
 * checkbox widgets expect. `changed` over-fires (any state change), which is
 * harmless — the widget just re-reads `value`. */
function fieldWatchable<T>(
  changed: RoiFilterState["changed"],
  get: () => T,
  set: (v: T) => void,
): WatchableValueInterface<T> {
  return {
    get value() {
      return get();
    },
    set value(v: T) {
      set(v);
    },
    changed,
  };
}

const ROLE_OPTIONS: { value: RoiOperator; label: string }[] = [
  { value: RoiOperator.AND, label: "Include" },
  { value: RoiOperator.OR, label: "Or" },
  { value: RoiOperator.ANDNOT, label: "Exclude" },
];
const PREDICATE_OPTIONS: { value: RoiPredicate; label: string }[] = [
  { value: RoiPredicate.ANY_SEGMENT, label: "Crosses" },
  { value: RoiPredicate.ANY_VERTEX, label: "Point inside" },
];

function makeSelect<T extends number>(
  options: { value: T; label: string }[],
  current: T,
  onChange: (v: T) => void,
): HTMLSelectElement {
  const select = document.createElement("select");
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = String(opt.value);
    el.textContent = opt.label;
    if (opt.value === current) el.selected = true;
    select.appendChild(el);
  }
  select.addEventListener("change", () => onChange(Number(select.value) as T));
  return select;
}

/** Parse `#rrggbb` (an `<input type=color>` value) into an rgb vec3 in [0,1]. */
function parseHexColor(hex: string): vec3 {
  const n = Number.parseInt(hex.replace(/^#/, ""), 16);
  return vec3.fromValues(
    ((n >> 16) & 0xff) / 255,
    ((n >> 8) & 0xff) / 255,
    (n & 0xff) / 255,
  );
}

// --- shape <-> (centre, radius/size) helpers --------------------------------

function boxCentre(shape: { lower: Float32Array; upper: Float32Array }): number[] {
  return Array.from(shape.lower, (lo, i) => (lo + shape.upper[i]) / 2);
}
function boxSize(shape: { lower: Float32Array; upper: Float32Array }): number[] {
  return Array.from(shape.lower, (lo, i) => shape.upper[i] - lo);
}
function boxFrom(centre: ArrayLike<number>, size: ArrayLike<number>): RoiShape {
  const rank = centre.length;
  const lower = new Float32Array(rank);
  const upper = new Float32Array(rank);
  for (let i = 0; i < rank; ++i) {
    lower[i] = centre[i] - size[i] / 2;
    upper[i] = centre[i] + size[i] / 2;
  }
  return { kind: "box", lower, upper };
}

/** A short human label for an ROI shape. */
function shapeLabel(shape: RoiShape): string {
  const fmt = (a: number[]) => a.map((v) => v.toFixed(0)).join(", ");
  switch (shape.kind) {
    case "ellipsoid":
      return `Sphere @ ${fmt(Array.from(shape.center))}`;
    case "box": {
      const size = boxSize(shape);
      const thin = size.findIndex((s) => s <= DEFAULT_PLANE_THICKNESS + 1e-6);
      const axisName = ["x", "y", "z"];
      if (thin >= 0 && size.filter((s) => s > DEFAULT_PLANE_THICKNESS + 1e-6).length === 2) {
        return `Plane ⊥${axisName[thin] ?? thin} @ ${fmt(boxCentre(shape))}`;
      }
      return `Box @ ${fmt(boxCentre(shape))}`;
    }
    case "halfspace":
      return `Plane @ ${fmt(Array.from(shape.origin))}`;
  }
}

export class StreamlineFilterTab extends Tab {
  private roiFilter: RoiFilterState;
  /** The worker-maintained set of passing object ids (undefined if unwired). */
  private passingSegments: Uint64Set | undefined;
  private countEl: HTMLElement;
  /** The group new ROIs are added to; the tab also shows this group's ROIs. */
  private activeGroupId: number | undefined;
  /** The structural (groups/ROIs/active) signature the body currently reflects. */
  private structuralSig = "";
  private bodyEl: HTMLElement;
  /** Disposers for the widgets in the current body build. */
  private bodyContext = new RefCounted();

  constructor(public layer: SegmentationUserLayer) {
    super();
    this.roiFilter = layer.displayState.roiFilter;
    this.passingSegments = layer.displayState.roiPassingSegments;
    const { element } = this;
    element.classList.add("neuroglancer-streamline-filter-tab");

    element.appendChild(this.makeHeader());
    this.bodyEl = document.createElement("div");
    this.bodyEl.classList.add("neuroglancer-streamline-filter-body");
    element.appendChild(this.bodyEl);
    this.countEl = document.createElement("span");
    this.countEl.classList.add("neuroglancer-streamline-filter-count");
    element.appendChild(this.countEl);

    this.registerDisposer(this.bodyContext);
    this.registerDisposer(this.roiFilter.changed.add(() => this.onChanged()));
    if (this.passingSegments !== undefined) {
      this.registerDisposer(
        this.passingSegments.changed.add(() => this.updateCount()),
      );
    }
    this.onChanged();
  }

  // --- global header (active, ghost opacity, colour-by-group) ---------------

  private makeHeader(): HTMLElement {
    const header = document.createElement("div");
    header.classList.add("neuroglancer-streamline-filter-header");

    const active = this.registerDisposer(
      new TrackableBooleanCheckbox(
        fieldWatchable(
          this.roiFilter.changed,
          () => this.roiFilter.active,
          (v) => (this.roiFilter.active = v),
        ),
      ),
    );
    header.appendChild(
      labelled("Active", active.element, "neuroglancer-streamline-filter-active"),
    );

    const colorByGroup = this.registerDisposer(
      new TrackableBooleanCheckbox(
        fieldWatchable(
          this.roiFilter.changed,
          () => this.roiFilter.colorByGroup,
          (v) => (this.roiFilter.colorByGroup = v),
        ),
      ),
    );
    header.appendChild(labelled("Colour by group", colorByGroup.element));

    const hide2d = this.registerDisposer(
      new TrackableBooleanCheckbox(
        fieldWatchable(
          this.roiFilter.changed,
          () => this.roiFilter.hideOverlays2d,
          (v) => (this.roiFilter.hideOverlays2d = v),
        ),
      ),
    );
    header.appendChild(labelled("Hide regions in 2-d", hide2d.element));

    const ghost = this.registerDisposer(
      new RangeWidget(
        fieldWatchable(
          this.roiFilter.changed,
          () => this.roiFilter.ghostAlpha,
          (v) => (this.roiFilter.ghostAlpha = v),
        ),
        { min: 0, max: 1, step: 0.01 },
      ),
    );
    header.appendChild(labelled("Non-passing opacity", ghost.element));
    return header;
  }

  // --- structural rebuild ---------------------------------------------------

  private structuralSignature(): string {
    return (
      `${this.activeGroupId}|` +
      this.roiFilter.groups
        .map((g) => `${g.id}:${g.rois.length}`)
        .join(",")
    );
  }

  private onChanged(): void {
    const groups = this.roiFilter.groups;
    // Default / repair the active group.
    if (
      this.activeGroupId === undefined ||
      !groups.some((g) => g.id === this.activeGroupId)
    ) {
      this.activeGroupId = groups.length > 0 ? groups[groups.length - 1].id : undefined;
    }
    const sig = this.structuralSignature();
    if (sig !== this.structuralSig) {
      this.structuralSig = sig;
      this.rebuildBody();
    }
    this.updateCount();
  }

  private rebuildBody(): void {
    this.bodyContext.dispose();
    this.bodyContext = new RefCounted();
    const el = this.bodyEl;
    el.textContent = "";

    // Group list.
    const groupList = document.createElement("div");
    groupList.classList.add("neuroglancer-streamline-filter-group-list");
    for (const group of this.roiFilter.groups) {
      groupList.appendChild(this.makeGroupRow(group));
    }
    el.appendChild(groupList);

    const addGroup = makeIcon({
      text: "+ New group",
      title: "Create a new tract group",
      onClick: () => {
        this.activeGroupId = this.roiFilter.addGroup();
      },
    });
    addGroup.classList.add("neuroglancer-streamline-filter-add-group");
    el.appendChild(addGroup);

    // Active group's ROIs.
    const group = this.roiFilter.groups.find((g) => g.id === this.activeGroupId);
    if (group !== undefined) {
      el.appendChild(this.makeRoiSection(group));
    }
  }

  private makeGroupRow(group: RoiGroup): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-streamline-filter-group-row");
    if (group.id === this.activeGroupId) row.classList.add("neuroglancer-selected");

    // Selecting the row makes it the active (edited) group.
    row.addEventListener("click", (e) => {
      if (e.target instanceof HTMLInputElement) return; // let inputs handle themselves
      if (this.activeGroupId !== group.id) {
        this.activeGroupId = group.id;
        this.onChanged();
      }
    });

    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.value = serializeColor(group.color);
    swatch.title = "Group colour";
    swatch.addEventListener("input", () =>
      this.roiFilter.updateGroup(group.id, { color: parseHexColor(swatch.value) }),
    );
    row.appendChild(swatch);

    const name = document.createElement("input");
    name.type = "text";
    name.value = group.name;
    name.classList.add("neuroglancer-streamline-filter-group-name");
    name.addEventListener("change", () =>
      this.roiFilter.updateGroup(group.id, { name: name.value }),
    );
    row.appendChild(name);

    const visible = document.createElement("input");
    visible.type = "checkbox";
    visible.checked = group.visible;
    visible.title = "Show this group";
    visible.addEventListener("change", () =>
      this.roiFilter.updateGroup(group.id, { visible: visible.checked }),
    );
    row.appendChild(visible);

    row.appendChild(
      makeDeleteButton({
        title: "Delete group",
        onClick: (e) => {
          e.stopPropagation();
          this.roiFilter.removeGroup(group.id);
        },
      }),
    );
    return row;
  }

  private makeRoiSection(group: RoiGroup): HTMLElement {
    const section = document.createElement("div");
    section.classList.add("neuroglancer-streamline-filter-roi-section");

    const add = document.createElement("div");
    add.classList.add("neuroglancer-streamline-filter-add");
    add.appendChild(
      makeIcon({
        text: "+ Sphere",
        title: "Add a sphere ROI at the crosshair",
        onClick: () => this.addSphere(group.id),
      }),
    );
    add.appendChild(
      makeIcon({
        text: "+ Box",
        title: "Add a box ROI at the crosshair",
        onClick: () => this.addBox(group.id),
      }),
    );
    add.appendChild(
      makeSelect(
        [
          { value: -1, label: "+ Plane…" },
          { value: 2, label: "xy plane" },
          { value: 0, label: "yz plane" },
          { value: 1, label: "zx plane" },
        ],
        -1,
        (axis) => {
          if (axis >= 0) this.addPlane(group.id, axis);
        },
      ),
    );
    section.appendChild(add);

    if (group.rois.length === 0) {
      const empty = document.createElement("div");
      empty.classList.add("neuroglancer-streamline-filter-empty");
      empty.textContent = "No ROIs yet — add a sphere, box, or plane above.";
      section.appendChild(empty);
    }
    group.rois.forEach((roi, index) => {
      section.appendChild(this.makeRoiCard(group.id, roi, index));
    });
    return section;
  }

  private makeRoiCard(groupId: number, roi: Roi, index: number): HTMLElement {
    const card = document.createElement("div");
    card.classList.add("neuroglancer-streamline-filter-roi-card");

    const head = document.createElement("div");
    head.classList.add("neuroglancer-streamline-filter-roi-head");
    const label = document.createElement("span");
    label.classList.add("neuroglancer-streamline-filter-roi-label");
    label.textContent = shapeLabel(roi.shape);
    head.appendChild(label);
    head.appendChild(
      makeDeleteButton({
        title: "Delete ROI",
        onClick: () => this.roiFilter.removeRoi(groupId, index),
      }),
    );
    card.appendChild(head);

    const controls = document.createElement("div");
    controls.classList.add("neuroglancer-streamline-filter-roi-controls");
    // Role: first ROI seeds the fold, so its operator is fixed to Include.
    const role = makeSelect(ROLE_OPTIONS, roi.operator, (v) =>
      this.roiFilter.updateRoi(groupId, index, { operator: v }),
    );
    if (index === 0) role.disabled = true;
    controls.appendChild(labelled("Role", role));
    controls.appendChild(
      labelled(
        "Test",
        makeSelect(PREDICATE_OPTIONS, roi.predicate, (v) =>
          this.roiFilter.updateRoi(groupId, index, { predicate: v }),
        ),
      ),
    );
    card.appendChild(controls);

    card.appendChild(this.makeShapeSliders(groupId, index, roi.shape));
    return card;
  }

  /** Centre x/y/z + radius (sphere) or size x/y/z (box/plane) sliders. */
  private makeShapeSliders(
    groupId: number,
    index: number,
    shape: RoiShape,
  ): HTMLElement {
    const box = document.createElement("div");
    box.classList.add("neuroglancer-streamline-filter-sliders");
    const axisName = ["x", "y", "z"];
    const rank =
      shape.kind === "ellipsoid" ? shape.center.length : shape.kind === "box" ? shape.lower.length : 3;

    const slider = (
      title: string,
      range: { min: number; max: number },
      get: () => number,
      set: (v: number) => void,
    ) => {
      const step = Math.max((range.max - range.min) / 200, 0.1);
      const w = this.bodyContext.registerDisposer(
        new RangeWidget(fieldWatchable(this.roiFilter.changed, get, set), {
          min: range.min,
          max: range.max,
          step,
        }),
      );
      box.appendChild(labelled(title, w.element));
    };

    if (shape.kind === "ellipsoid") {
      for (let i = 0; i < rank; ++i) {
        slider(
          axisName[i] ?? `x${i}`,
          this.axisRange(i),
          () => this.currentShape(groupId, index, "ellipsoid").center[i],
          (v) => {
            const s = this.currentShape(groupId, index, "ellipsoid");
            const center = Float32Array.from(s.center);
            center[i] = v;
            this.roiFilter.updateRoi(groupId, index, {
              shape: { ...s, center },
            });
          },
        );
      }
      slider(
        "radius",
        { min: 0, max: this.maxExtent() / 2 },
        () => this.currentShape(groupId, index, "ellipsoid").radii[0],
        (v) => {
          const s = this.currentShape(groupId, index, "ellipsoid");
          const radii = new Float32Array(s.radii.length).fill(v);
          this.roiFilter.updateRoi(groupId, index, { shape: { ...s, radii } });
        },
      );
    } else if (shape.kind === "box") {
      for (let i = 0; i < rank; ++i) {
        slider(
          `${axisName[i] ?? i}`,
          this.axisRange(i),
          () => boxCentre(this.currentShape(groupId, index, "box"))[i],
          (v) => {
            const s = this.currentShape(groupId, index, "box");
            const centre = boxCentre(s);
            centre[i] = v;
            this.roiFilter.updateRoi(groupId, index, {
              shape: boxFrom(centre, boxSize(s)),
            });
          },
        );
      }
      for (let i = 0; i < rank; ++i) {
        slider(
          `size ${axisName[i] ?? i}`,
          { min: 0, max: this.maxExtent() },
          () => boxSize(this.currentShape(groupId, index, "box"))[i],
          (v) => {
            const s = this.currentShape(groupId, index, "box");
            const size = boxSize(s);
            size[i] = v;
            this.roiFilter.updateRoi(groupId, index, {
              shape: boxFrom(boxCentre(s), size),
            });
          },
        );
      }
    }
    return box;
  }

  /** Re-read the current shape of an ROI (it may have changed under the slider). */
  private currentShape<K extends RoiShape["kind"]>(
    groupId: number,
    index: number,
    kind: K,
  ): Extract<RoiShape, { kind: K }> {
    const group = this.roiFilter.groups.find((g) => g.id === groupId);
    const shape = group?.rois[index]?.shape;
    if (shape === undefined || shape.kind !== kind) {
      throw new Error("ROI shape changed out from under the slider");
    }
    return shape as Extract<RoiShape, { kind: K }>;
  }

  // --- ROI creation ---------------------------------------------------------

  private crosshair(): Float32Array {
    // The tract model transform is identity, so the global crosshair position is
    // already in streamline space.
    return Float32Array.from(this.layer.manager.root.globalPosition.value);
  }

  private addSphere(groupId: number): void {
    const center = this.crosshair();
    const radii = new Float32Array(center.length).fill(DEFAULT_ROI_RADIUS);
    this.addRoiToGroup(groupId, { kind: "ellipsoid", center, radii });
  }

  private addBox(groupId: number): void {
    const center = this.crosshair();
    const size = new Array(center.length).fill(DEFAULT_ROI_RADIUS * 2);
    this.addRoiToGroup(groupId, boxFrom(center, size));
  }

  private addPlane(groupId: number, normalAxis: number): void {
    // A plane is a thin box: thickness 1 on the normal axis, spanning the data
    // bounds on the other two, centred at the crosshair on the normal axis.
    const cross = this.crosshair();
    const rank = cross.length;
    const centre = new Float32Array(rank);
    const size = new Float32Array(rank);
    for (let i = 0; i < rank; ++i) {
      if (i === normalAxis) {
        centre[i] = cross[i];
        size[i] = DEFAULT_PLANE_THICKNESS;
      } else {
        const { min, max } = this.axisRange(i);
        centre[i] = (min + max) / 2;
        size[i] = max - min;
      }
    }
    this.addRoiToGroup(groupId, boxFrom(centre, size));
  }

  private addRoiToGroup(groupId: number, shape: RoiShape): void {
    this.roiFilter.addRoi(groupId, {
      shape,
      predicate: RoiPredicate.ANY_SEGMENT,
      operator: RoiOperator.AND,
    });
    if (!this.roiFilter.active) this.roiFilter.active = true;
  }

  // --- coordinate ranges ----------------------------------------------------

  private axisRange(axis: number): { min: number; max: number } {
    const cs = this.layer.manager.root.globalPosition.coordinateSpace.value;
    const lo = cs?.bounds?.lowerBounds?.[axis];
    const hi = cs?.bounds?.upperBounds?.[axis];
    if (
      lo !== undefined &&
      hi !== undefined &&
      Number.isFinite(lo) &&
      Number.isFinite(hi) &&
      hi > lo
    ) {
      return { min: lo, max: hi };
    }
    return { min: -1000, max: 1000 };
  }

  /** The largest axis span, for radius/size slider maxima. */
  private maxExtent(): number {
    const cs = this.layer.manager.root.globalPosition.coordinateSpace.value;
    let max = 2000;
    const lower = cs?.bounds?.lowerBounds;
    const upper = cs?.bounds?.upperBounds;
    if (lower !== undefined && upper !== undefined) {
      for (let i = 0; i < lower.length; ++i) {
        const span = upper[i] - lower[i];
        if (Number.isFinite(span)) max = Math.max(max, span);
      }
    }
    return max;
  }

  // --- live pass count ------------------------------------------------------

  private updateCount(): void {
    if (!this.roiFilter.hasVisibleRois()) {
      this.countEl.textContent = "";
      return;
    }
    const n = this.passingSegments?.size ?? 0;
    this.countEl.textContent = `${n.toLocaleString()} streamline${n === 1 ? "" : "s"} pass (this level)`;
  }
}

/** Wrap a control in a `<label>` with leading text. */
function labelled(
  text: string,
  control: HTMLElement,
  className?: string,
): HTMLElement {
  const label = document.createElement("label");
  label.classList.add("neuroglancer-streamline-filter-field");
  if (className !== undefined) label.classList.add(className);
  const span = document.createElement("span");
  span.textContent = text;
  label.appendChild(span);
  label.appendChild(control);
  return label;
}
