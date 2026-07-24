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
 * @file The "By segmentation label" panel of the streamline Filter tab.
 *
 * Lets the user dissect the tracts by ANATOMY rather than by hand-drawn spheres
 * and boxes: pick a linked parcellation layer, then stage a set of its labels as
 * Include / Exclude. The staged combination is evaluated LIVE as a preview
 * dissection ({@link RoiFilterState.setPreviewGroup}) — non-passing tracts ghost
 * and the selection re-filters as labels are toggled — but it is deliberately
 * NOT a group yet: the user commits it to a real, persisted group in one step
 * ("Create group from selection"), which is what makes it exportable/saveable.
 *
 * Labels can be added two ways ("both click + searchable list"): by selecting
 * regions in the linked parcellation layer's own viewer (double-click) and
 * pulling them in with "Add selected regions", or by name from a searchable
 * list. Each staged label carries an Include / Exclude / Off role.
 *
 * The membership test is spatial and evaluated in the worker: a streamline is in
 * an included region iff a vertex falls in a voxel with that label (see
 * `roi.ts`'s `labelMask` shape + `label_field.ts`). Combination semantics: pass
 * iff the tract crosses ANY included label (union) AND no excluded label; with
 * only excludes, pass iff it crosses none of them.
 */

import "#src/datasource/zarr-vectors/streamline_filter_tab.css";

import {
  labelled,
  makeStringSelect,
} from "#src/datasource/zarr-vectors/filter_widgets.js";
import {
  readParcellationLabels,
  type ParcellationLabel,
} from "#src/datasource/zarr-vectors/label_field.js";
import {
  RoiOperator,
  RoiPredicate,
  type Roi,
} from "#src/datasource/zarr-vectors/roi.js";
import type { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";
// Value import (not `import type`): `instanceof SegmentationUserLayer` needs the
// runtime class. Only used inside methods, so the index.ts ↔ tab import cycle is
// resolved by the time the panel is instantiated.
import { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { RefCounted } from "#src/util/disposable.js";
import { removeChildren } from "#src/util/dom.js";
import { vec3 } from "#src/util/geom.js";
import { makeDeleteButton } from "#src/widget/delete_button.js";
import { LayerReferenceWidget } from "#src/widget/layer_reference.js";

type Role = "include" | "exclude";

/** Neutral colour for the uncommitted preview (committing assigns a palette one). */
const PREVIEW_COLOR = vec3.fromValues(1, 1, 1);

const ROLE_OPTIONS = [
  { value: "include", label: "Include" },
  { value: "exclude", label: "Exclude" },
  { value: "off", label: "Off" },
];

export class LabelFilterPanel extends RefCounted {
  readonly element = document.createElement("div");
  private roiFilter: RoiFilterState;
  /** Staged label -> role, the not-yet-committed selection driving the preview. */
  private staged = new Map<number, Role>();
  private labelById = new Map<number, ParcellationLabel>();

  private statusEl = document.createElement("div");
  private listEl = document.createElement("div");
  private countEl = document.createElement("span");
  private searchInput = document.createElement("input");
  private datalist = document.createElement("datalist");
  private nameInput = document.createElement("input");
  private createButton = document.createElement("button");

  constructor(public layer: SegmentationUserLayer) {
    super();
    this.roiFilter = layer.displayState.roiFilter;
    const { element } = this;
    element.classList.add("neuroglancer-streamline-filter-label-panel");

    const heading = document.createElement("div");
    heading.classList.add("neuroglancer-streamline-filter-subheading");
    heading.textContent = "By segmentation label";
    element.appendChild(heading);

    // Parcellation layer picker.
    const refWidget = this.registerDisposer(
      new LayerReferenceWidget(this.layer.displayState.roiLabelLayer.addRef()),
    );
    element.appendChild(labelled("Parcellation", refWidget.element));

    this.statusEl.classList.add("neuroglancer-streamline-filter-note");
    element.appendChild(this.statusEl);

    // Add-by-search + add-from-viewer row.
    const addRow = document.createElement("div");
    addRow.classList.add("neuroglancer-streamline-filter-field");
    this.searchInput.type = "text";
    this.searchInput.placeholder = "Add label by name…";
    this.datalist.id = "neuroglancer-roi-label-options";
    this.searchInput.setAttribute("list", this.datalist.id);
    this.registerEventListener(this.searchInput, "change", () =>
      this.addFromSearch(),
    );
    const addSelected = document.createElement("button");
    addSelected.textContent = "Add selected regions";
    addSelected.title =
      "Add the labels currently selected in the parcellation layer";
    this.registerEventListener(addSelected, "click", () =>
      this.addSelectedRegions(),
    );
    addRow.appendChild(this.searchInput);
    addRow.appendChild(this.datalist);
    addRow.appendChild(addSelected);
    element.appendChild(addRow);

    this.listEl.classList.add("neuroglancer-streamline-filter-label-list");
    element.appendChild(this.listEl);

    // Commit row: name + "Create group from selection" + Clear.
    const commitRow = document.createElement("div");
    commitRow.classList.add("neuroglancer-streamline-filter-field");
    this.countEl.classList.add("neuroglancer-streamline-filter-note");
    this.nameInput.type = "text";
    this.nameInput.placeholder = "Group name";
    this.createButton.textContent = "Create group from selection";
    this.registerEventListener(this.createButton, "click", () => this.commit());
    const clearButton = document.createElement("button");
    clearButton.textContent = "Clear";
    this.registerEventListener(clearButton, "click", () => this.clear());
    commitRow.appendChild(this.nameInput);
    commitRow.appendChild(this.createButton);
    commitRow.appendChild(clearButton);
    element.appendChild(this.countEl);
    element.appendChild(commitRow);

    // Refresh labels + status when the reference changes, when a layer loads
    // (the parcellation may not be ready yet), or when its grid finishes.
    this.registerDisposer(
      this.layer.displayState.roiLabelLayer.changed.add(() => this.refresh()),
    );
    this.registerDisposer(
      this.layer.manager.rootLayers.layersChanged.add(() => this.refresh()),
    );
    const field = this.layer.displayState.roiLabelField;
    if (field !== undefined) {
      this.registerDisposer(field.changed.add(() => this.updateStatus()));
    }
    this.refresh();
  }

  /** The linked parcellation layer, or undefined if none/invalid. */
  private parcellation(): SegmentationUserLayer | undefined {
    const userLayer = this.layer.displayState.roiLabelLayer.layer?.layer;
    return userLayer instanceof SegmentationUserLayer &&
      userLayer !== this.layer
      ? userLayer
      : undefined;
  }

  /** Re-read the label LUT + rebuild the search list, status and rows. */
  private refresh(): void {
    const parcellation = this.parcellation();
    const labels = parcellation ? readParcellationLabels(parcellation) : [];
    this.labelById = new Map(labels.map((l) => [l.id, l]));
    removeChildren(this.datalist);
    for (const l of labels) {
      if (l.name.length === 0) continue;
      const option = document.createElement("option");
      option.value = l.name;
      this.datalist.appendChild(option);
    }
    // Drop staged labels that no longer exist in the (possibly changed) LUT.
    // Collect first, then delete, so we do not mutate the map mid-iteration.
    if (labels.length > 0) {
      const stale: number[] = [];
      for (const id of this.staged.keys()) {
        if (!this.labelById.has(id)) stale.push(id);
      }
      for (const id of stale) this.staged.delete(id);
    }
    this.updateStatus();
    this.rebuildList();
    this.updatePreview();
  }

  private updateStatus(): void {
    const parcellation = this.parcellation();
    if (parcellation === undefined) {
      this.statusEl.textContent =
        "Choose a segmentation layer whose labels dissect the tracts.";
      return;
    }
    const field = this.layer.displayState.roiLabelField?.value;
    this.statusEl.textContent =
      field === undefined
        ? "Building label grid… label filters apply once it is ready."
        : `Label grid ready (${field.dims.join(" × ")} voxels).`;
  }

  private displayName(id: number): string {
    const name = this.labelById.get(id)?.name;
    return name !== undefined && name.length > 0 ? name : `Label ${id}`;
  }

  private rebuildList(): void {
    removeChildren(this.listEl);
    for (const [id, role] of this.staged) {
      const row = document.createElement("div");
      row.classList.add("neuroglancer-streamline-filter-label-row");

      const swatch = document.createElement("span");
      swatch.classList.add("neuroglancer-streamline-filter-label-swatch");
      const packed = this.labelById.get(id)?.colorPacked;
      if (packed !== undefined) {
        const r = packed & 0xff;
        const g = (packed >> 8) & 0xff;
        const b = (packed >> 16) & 0xff;
        swatch.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
      }
      row.appendChild(swatch);

      const nameSpan = document.createElement("span");
      nameSpan.classList.add("neuroglancer-streamline-filter-label-name");
      nameSpan.textContent = this.displayName(id);
      row.appendChild(nameSpan);

      const roleSelect = makeStringSelect(ROLE_OPTIONS, role, (value) => {
        if (value === "off") {
          this.staged.delete(id);
          this.rebuildList();
        } else {
          this.staged.set(id, value as Role);
        }
        this.updatePreview();
      });
      row.appendChild(roleSelect);

      row.appendChild(
        makeDeleteButton({
          title: "Remove label",
          onClick: () => {
            this.staged.delete(id);
            this.rebuildList();
            this.updatePreview();
          },
        }),
      );
      this.listEl.appendChild(row);
    }
    const n = this.staged.size;
    this.countEl.textContent =
      n === 0
        ? "No labels selected."
        : `${n} label${n === 1 ? "" : "s"} selected.`;
    this.createButton.disabled = n === 0;
  }

  private addFromSearch(): void {
    const query = this.searchInput.value.trim();
    if (query.length === 0) return;
    // Match by exact name (case-insensitive), else by numeric id typed directly.
    let matched: number | undefined;
    for (const [id, label] of this.labelById) {
      if (label.name.toLowerCase() === query.toLowerCase()) {
        matched = id;
        break;
      }
    }
    if (matched === undefined) {
      const asNum = Number(query);
      if (Number.isInteger(asNum) && this.labelById.has(asNum)) matched = asNum;
    }
    if (matched === undefined) return;
    this.searchInput.value = "";
    if (!this.staged.has(matched)) this.staged.set(matched, "include");
    this.rebuildList();
    this.updatePreview();
  }

  private addSelectedRegions(): void {
    const parcellation = this.parcellation();
    if (parcellation === undefined) return;
    const visible =
      parcellation.displayState.segmentationGroupState.value.visibleSegments;
    let added = false;
    for (const id of visible.keys()) {
      const numId = Number(id);
      if (numId === 0) continue; // background
      if (!this.staged.has(numId)) {
        this.staged.set(numId, "include");
        added = true;
      }
    }
    if (added) {
      this.rebuildList();
      this.updatePreview();
    }
  }

  /** Rebuild the live preview dissection from the staged labels. */
  private updatePreview(): void {
    const includes: number[] = [];
    const excludes: number[] = [];
    for (const [id, role] of this.staged) {
      (role === "include" ? includes : excludes).push(id);
    }
    if (includes.length === 0 && excludes.length === 0) {
      this.roiFilter.setPreviewGroup(undefined);
      return;
    }
    const rois: Roi[] = [];
    const labelRoi = (id: number, operator: RoiOperator): Roi => ({
      shape: { kind: "labelMask", labels: [id] },
      predicate: RoiPredicate.ANY_SEGMENT,
      operator,
      name: this.displayName(id),
    });
    // Includes first (union: first seeds AND, rest OR), then excludes (ANDNOT).
    // With no includes the first exclude seeds ANDNOT = "everything except".
    includes.forEach((id, i) =>
      rois.push(labelRoi(id, i === 0 ? RoiOperator.AND : RoiOperator.OR)),
    );
    excludes.forEach((id) => rois.push(labelRoi(id, RoiOperator.ANDNOT)));
    this.roiFilter.setPreviewGroup({
      name: "Label selection",
      color: vec3.clone(PREVIEW_COLOR),
      visible: true,
      opacity: 1,
      highDetail: false,
      colorBy: { kind: "group" },
      rois,
    });
    // A staged selection is meant to be seen, so turn the filter on.
    this.roiFilter.active = true;
  }

  private commit(): void {
    if (this.staged.size === 0) return;
    const name = this.nameInput.value.trim();
    this.roiFilter.commitPreviewGroup(name.length > 0 ? name : undefined);
    // The preview was consumed into a real group; reset the staging area.
    this.staged.clear();
    this.nameInput.value = "";
    this.rebuildList();
  }

  private clear(): void {
    if (this.staged.size === 0) return;
    this.staged.clear();
    this.roiFilter.setPreviewGroup(undefined);
    this.rebuildList();
  }
}
