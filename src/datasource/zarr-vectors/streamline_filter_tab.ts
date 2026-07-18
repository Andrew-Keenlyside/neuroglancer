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
 * filter. Lists the ordered ROIs (each Include / Or / Exclude with a
 * predicate), offers add/place buttons, and exposes the active toggle, ghost
 * opacity, and colour-by-group. All edits write into the layer's
 * `RoiFilterState`, which drives the filter and round-trips to the URL.
 */

import "#src/datasource/zarr-vectors/streamline_filter_tab.css";

import {
  RoiOperator,
  RoiPredicate,
  type Roi,
  type RoiShape,
} from "#src/datasource/zarr-vectors/roi.js";
import type { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { TrackableBooleanCheckbox } from "#src/trackable_boolean.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import type { Uint64Set } from "#src/uint64_set.js";
import { makeDeleteButton } from "#src/widget/delete_button.js";
import { makeIcon } from "#src/widget/icon.js";
import { RangeWidget } from "#src/widget/range.js";
import { Tab } from "#src/widget/tab_view.js";

/** A default sphere radius (model units) for a click-added ROI before editing. */
const DEFAULT_ROI_RADIUS = 10;

/** Adapt one `RoiFilterState` scalar field to the `WatchableValueInterface` the
 * existing checkbox/range widgets expect. `changed` over-fires (any state
 * change), which is harmless: the widget just re-reads `value`. */
function fieldWatchable<T>(
  state: RoiFilterState,
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
    changed: state.changed,
  };
}

function shapeLabel(shape: RoiShape): string {
  const fmt = (a: Float32Array) =>
    Array.from(a, (v) => v.toFixed(1)).join(", ");
  switch (shape.kind) {
    case "ellipsoid":
      return `Sphere @ ${fmt(shape.center)}`;
    case "box":
      return `Box ${fmt(shape.lower)}–${fmt(shape.upper)}`;
    case "halfspace":
      return `Plane @ ${fmt(shape.origin)}`;
  }
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

export class StreamlineFilterTab extends Tab {
  private roiFilter: RoiFilterState;
  /** The worker-maintained set of passing object ids (undefined if unwired). */
  private passingSegments: Uint64Set | undefined;
  private countEl: HTMLElement | undefined;

  constructor(public layer: SegmentationUserLayer) {
    super();
    this.roiFilter = layer.displayState.roiFilter;
    this.passingSegments = layer.displayState.roiPassingSegments;
    const { element } = this;
    element.classList.add("neuroglancer-streamline-filter-tab");

    element.appendChild(this.makeHeader());
    element.appendChild(this.makeAddButtons());

    const listEl = document.createElement("div");
    listEl.classList.add("neuroglancer-streamline-filter-roi-list");
    element.appendChild(listEl);

    element.appendChild(this.makeDisplayControls());

    const rebuild = () => {
      this.renderRoiList(listEl);
      this.updateCount();
    };
    this.registerDisposer(this.roiFilter.changed.add(rebuild));
    rebuild();
  }

  /**
   * Show how many loaded streamlines pass the current ROIs. The worker keeps the
   * passing set current whenever ROIs exist (even before the filter is switched
   * on), so this is live feedback while dissecting. The count is over the tracts
   * resident at the current LOD, not the whole store — hence "(this level)".
   */
  private updateCount(): void {
    const el = this.countEl;
    if (el === undefined) return;
    if (this.roiFilter.rois.length === 0) {
      el.textContent = "";
      return;
    }
    const n = this.passingSegments?.size ?? 0;
    el.textContent = `${n.toLocaleString()} streamline${n === 1 ? "" : "s"} pass (this level)`;
  }

  private makeHeader(): HTMLElement {
    const header = document.createElement("div");
    header.classList.add("neuroglancer-streamline-filter-header");

    const activeLabel = document.createElement("label");
    activeLabel.classList.add("neuroglancer-streamline-filter-active");
    const activeCheckbox = this.registerDisposer(
      new TrackableBooleanCheckbox(
        fieldWatchable(
          this.roiFilter,
          () => this.roiFilter.active,
          (v) => (this.roiFilter.active = v),
        ),
      ),
    );
    activeLabel.appendChild(activeCheckbox.element);
    activeLabel.appendChild(document.createTextNode(" Active"));
    header.appendChild(activeLabel);

    // Live "N streamlines pass" readout, driven by the worker-maintained
    // passing set. Absent only if the ROI channel was never created (it always
    // is for the tract layers this tab is shown for).
    const count = document.createElement("span");
    count.classList.add("neuroglancer-streamline-filter-count");
    this.countEl = count;
    if (this.passingSegments !== undefined) {
      this.registerDisposer(
        this.passingSegments.changed.add(() => this.updateCount()),
      );
    }
    header.appendChild(count);

    return header;
  }

  private makeAddButtons(): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-streamline-filter-add");
    row.appendChild(
      makeIcon({
        text: "+ Sphere",
        title: "Add an inclusion sphere ROI at the crosshair",
        onClick: () => this.addRoiAtCrosshair("ellipsoid"),
      }),
    );
    row.appendChild(
      makeIcon({
        text: "+ Box",
        title: "Add an inclusion box ROI at the crosshair",
        onClick: () => this.addRoiAtCrosshair("box"),
      }),
    );
    return row;
  }

  private makeDisplayControls(): HTMLElement {
    const container = document.createElement("div");
    container.classList.add("neuroglancer-streamline-filter-display");

    const ghostRow = document.createElement("label");
    ghostRow.classList.add("neuroglancer-streamline-filter-ghost");
    ghostRow.appendChild(document.createTextNode("Non-passing opacity"));
    const ghost = this.registerDisposer(
      new RangeWidget(
        fieldWatchable(
          this.roiFilter,
          () => this.roiFilter.ghostAlpha,
          (v) => (this.roiFilter.ghostAlpha = v),
        ),
        { min: 0, max: 1, step: 0.01 },
      ),
    );
    ghostRow.appendChild(ghost.element);
    container.appendChild(ghostRow);

    const colorLabel = document.createElement("label");
    colorLabel.classList.add("neuroglancer-streamline-filter-color");
    const colorCheckbox = this.registerDisposer(
      new TrackableBooleanCheckbox(
        fieldWatchable(
          this.roiFilter,
          () => this.roiFilter.colorByGroup,
          (v) => (this.roiFilter.colorByGroup = v),
        ),
      ),
    );
    colorLabel.appendChild(colorCheckbox.element);
    colorLabel.appendChild(document.createTextNode(" Colour by ROI group"));
    container.appendChild(colorLabel);

    return container;
  }

  private renderRoiList(listEl: HTMLElement): void {
    listEl.textContent = "";
    const rois = this.roiFilter.rois;
    if (rois.length === 0) {
      const empty = document.createElement("div");
      empty.classList.add("neuroglancer-streamline-filter-empty");
      empty.textContent = "No ROIs. Add a sphere or box to start filtering.";
      listEl.appendChild(empty);
      return;
    }
    rois.forEach((roi, index) => {
      listEl.appendChild(this.makeRoiRow(roi, index, rois.length));
    });
  }

  private makeRoiRow(roi: Roi, index: number, count: number): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-streamline-filter-roi-row");

    const label = document.createElement("span");
    label.classList.add("neuroglancer-streamline-filter-roi-label");
    label.textContent = shapeLabel(roi.shape);
    row.appendChild(label);

    // The first ROI seeds the fold, so its operator is ignored; lock it to
    // "Include" to make that visible rather than a silently-ignored control.
    const roleSelect = makeSelect(ROLE_OPTIONS, roi.operator, (v) =>
      this.roiFilter.updateRoi(index, { operator: v }),
    );
    if (index === 0) roleSelect.disabled = true;
    row.appendChild(roleSelect);

    row.appendChild(
      makeSelect(PREDICATE_OPTIONS, roi.predicate, (v) =>
        this.roiFilter.updateRoi(index, { predicate: v }),
      ),
    );

    const up = makeIcon({
      text: "▲",
      title: "Move up (order is evaluated top to bottom)",
      onClick: () => this.roiFilter.moveRoi(index, index - 1),
    });
    if (index === 0) up.classList.add("neuroglancer-disabled");
    row.appendChild(up);

    const down = makeIcon({
      text: "▼",
      title: "Move down",
      onClick: () => this.roiFilter.moveRoi(index, index + 1),
    });
    if (index === count - 1) down.classList.add("neuroglancer-disabled");
    row.appendChild(down);

    row.appendChild(
      makeDeleteButton({
        title: "Delete this ROI",
        onClick: () => this.roiFilter.removeRoi(index),
      }),
    );

    return row;
  }

  /**
   * Add an ROI centred at the current crosshair, with a default size. This is
   * the quick path; drag-to-place with live overlays lands with the annotation
   * mirror. The tract model transform is identity, so the global crosshair
   * position is already in streamline space.
   */
  private addRoiAtCrosshair(kind: "ellipsoid" | "box"): void {
    const position = this.layer.manager.root.globalPosition.value;
    const rank = position.length;
    const center = Float32Array.from(position);
    let shape: RoiShape;
    if (kind === "ellipsoid") {
      shape = {
        kind: "ellipsoid",
        center,
        radii: new Float32Array(rank).fill(DEFAULT_ROI_RADIUS),
      };
    } else {
      const lower = new Float32Array(rank);
      const upper = new Float32Array(rank);
      for (let i = 0; i < rank; ++i) {
        lower[i] = center[i] - DEFAULT_ROI_RADIUS;
        upper[i] = center[i] + DEFAULT_ROI_RADIUS;
      }
      shape = { kind: "box", lower, upper };
    }
    this.roiFilter.addRoi({
      shape,
      predicate: RoiPredicate.ANY_SEGMENT,
      operator: RoiOperator.AND,
    });
    // Adding the first ROI turns the filter on so the user sees an effect.
    if (this.roiFilter.rois.length === 1) this.roiFilter.active = true;
  }
}
