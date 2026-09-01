/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * @file The "By attribute" panel of the Filter tab.
 *
 * Selects objects by their DATA rather than by their position: flags a store
 * carries per object or per vertex (`high_quality_transfer`), and ranges on its
 * measurements (a gene's expression, a tract's FA, a cell's confidence score).
 * For a `point_cloud` — a MERFISH panel, where one vertex is one cell and there
 * is no object model at all — this is the whole of what a dissection can mean,
 * which is why the Filter tab is offered for those stores at all.
 *
 * Same staging model as the "By segmentation label" panel next to it, and
 * deliberately so: stage predicates, watch them evaluated LIVE as a preview
 * dissection ({@link RoiFilterState.setPreviewGroup} — non-matching geometry
 * ghosts as the ranges move), then commit the combination to a real, persisted,
 * exportable group in one step. Nothing is a group until the user says it is.
 *
 * Predicates AND together, and for a per-vertex tier they must be satisfied by
 * ONE vertex at once: `gene_A` high AND `gene_B` high means co-expression in a
 * single cell, not two cells of the same object. See `RoiAttrFilter`.
 *
 * There is ONE preview slot per layer, shared with the "By segmentation label"
 * panel, so staging here replaces a label selection staged there (and vice
 * versa). Committing either to a group is what makes it permanent; to combine
 * the two, commit one and then add the other's predicates to that group in its
 * row in the group list.
 */

import "#src/datasource/zarr-vectors/streamline_filter_tab.css";

import type {
  AttrChoice,
  AttrStatsCache,
} from "#src/datasource/zarr-vectors/attribute_catalog.js";
import {
  attrKey,
  filterScope,
  flagFilter,
  fullRangeFilter,
  hasApproximateValues,
  isFlagAttr,
  listAttrChoices,
} from "#src/datasource/zarr-vectors/attribute_catalog.js";
import { makeAttrFilterControl } from "#src/datasource/zarr-vectors/attribute_filter_widgets.js";
import {
  labelled,
  makeStringSelect,
} from "#src/datasource/zarr-vectors/filter_widgets.js";
import type {
  RoiAttrFilter,
  RoiAttrScope,
} from "#src/datasource/zarr-vectors/roi.js";
import type { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { RefCounted } from "#src/util/disposable.js";
import { removeChildren } from "#src/util/dom.js";
import { vec3 } from "#src/util/geom.js";

/** Neutral colour for the uncommitted preview (committing assigns a palette one). */
const PREVIEW_COLOR = vec3.fromValues(1, 1, 1);

export class AttributeFilterPanel extends RefCounted {
  readonly element = document.createElement("div");
  private roiFilter: RoiFilterState;
  /** The staged, not-yet-committed predicates driving the preview. */
  private staged: RoiAttrFilter[] = [];
  private rowsEl = document.createElement("div");
  private pickerEl = document.createElement("div");
  private statusEl = document.createElement("div");
  private nameInput = document.createElement("input");
  private createButton = document.createElement("button");
  /** Disposers for the widgets of one render pass. */
  private rowContext = new RefCounted();

  constructor(
    public layer: SegmentationUserLayer,
    private attrStats: AttrStatsCache,
  ) {
    super();
    this.roiFilter = layer.displayState.roiFilter;
    const { element } = this;
    element.classList.add("neuroglancer-streamline-filter-label-panel");

    const heading = document.createElement("div");
    heading.classList.add("neuroglancer-streamline-filter-subheading");
    heading.textContent = "By attribute";
    element.appendChild(heading);

    this.statusEl.classList.add("neuroglancer-streamline-filter-note");
    element.appendChild(this.statusEl);
    element.appendChild(this.rowsEl);
    element.appendChild(this.pickerEl);

    // Name + commit, mirroring the label panel's footer so the two staging
    // areas commit the same way.
    const commitRow = document.createElement("div");
    commitRow.classList.add("neuroglancer-streamline-filter-field");
    this.nameInput.type = "text";
    this.nameInput.placeholder = "Group name (optional)";
    commitRow.appendChild(this.nameInput);
    this.createButton.textContent = "Create group from selection";
    this.registerEventListener(this.createButton, "click", () => this.commit());
    commitRow.appendChild(this.createButton);
    const clearButton = document.createElement("button");
    clearButton.textContent = "Clear";
    this.registerEventListener(clearButton, "click", () => this.clear());
    commitRow.appendChild(clearButton);
    element.appendChild(commitRow);

    this.registerDisposer(this.rowContext);
    // A measurement landing turns a "measuring…" row into a real control, and
    // the layer's own attributes may arrive after the panel is built.
    this.registerDisposer(this.attrStats.changed.add(() => this.rebuild()));
    this.registerDisposer(
      this.layer.displayState.segmentPropertyMap.changed.add(() =>
        this.rebuild(),
      ),
    );
    this.rebuild();
  }

  /** Attributes not already staged, in catalogue order. */
  private available(): AttrChoice[] {
    const taken = new Set(
      this.staged.map((f) => attrKey(f.name, filterScope(f))),
    );
    return listAttrChoices(this.layer).filter(
      (c) => !taken.has(attrKey(c.name, c.scope)),
    );
  }

  private rebuild(): void {
    // Widgets from the previous pass are detached below; dispose them with the
    // context they were registered on rather than leaking one per rebuild.
    this.rowContext.dispose();
    this.rowContext = new RefCounted();
    removeChildren(this.rowsEl);
    removeChildren(this.pickerEl);

    for (let i = 0; i < this.staged.length; ++i) {
      const filter = this.staged[i];
      const scope = filterScope(filter);
      this.rowsEl.appendChild(
        makeAttrFilterControl({
          filter,
          stats: this.attrStats.get(filter.name, scope),
          context: this.rowContext,
          changed: this.roiFilter.changed,
          read: () => this.staged[i] ?? filter,
          write: (next) => {
            this.staged[i] = next;
            this.updatePreview();
          },
          remove: () => {
            this.staged.splice(i, 1);
            this.rebuild();
            this.updatePreview();
          },
          requestStats: () => void this.attrStats.request(filter.name, scope),
        }),
      );
    }

    const available = this.available();
    if (available.length > 0) {
      const options = [
        {
          value: "",
          label:
            this.staged.length === 0 ? "Choose an attribute…" : "Add another…",
        },
        ...available.map((c) => ({
          value: attrKey(c.name, c.scope),
          label: c.scope === "object" ? `${c.name} (per object)` : c.name,
        })),
      ];
      this.pickerEl.appendChild(
        labelled(
          this.staged.length === 0 ? "Attribute" : "And",
          makeStringSelect(options, "", (value) => {
            const choice = available.find(
              (c) => attrKey(c.name, c.scope) === value,
            );
            if (choice !== undefined)
              void this.stage(choice.name, choice.scope);
          }),
        ),
      );
    }

    this.createButton.disabled = this.staged.length === 0;
    if (this.statusEl.textContent === "") this.describeSelection();
  }

  /** The default status line: what staging currently means. */
  private describeSelection(): void {
    const choices = listAttrChoices(this.layer);
    if (choices.length === 0) {
      this.statusEl.textContent =
        "This store exposes no filterable attributes. Name the columns you " +
        "want with `#attributes=` on the source URL.";
      return;
    }
    this.statusEl.textContent =
      this.staged.length === 0
        ? "Pick an attribute to select by; the selection previews live and " +
          "becomes a group when you create one."
        : `${this.staged.length} attribute${
            this.staged.length === 1 ? "" : "s"
          } staged — all must match.`;
  }

  /**
   * Stage one attribute, seeded to select everything it has (a flag at "true",
   * a measurement at its full observed range) so staging never blanks the view
   * before the user has narrowed anything.
   */
  private async stage(name: string, scope: RoiAttrScope): Promise<void> {
    const stats = await this.attrStats.request(name, scope);
    if (stats === undefined || stats.count === 0) {
      this.statusEl.textContent =
        scope === "vertex"
          ? `${name} has no values in the loaded chunks — pan or zoom to load ` +
            "geometry, then try again."
          : `${name} has no values yet — wait for the object attributes to load.`;
      return;
    }
    if (this.staged.some((f) => f.name === name && filterScope(f) === scope)) {
      return;
    }
    this.staged.push(
      isFlagAttr(stats) ? flagFilter(stats, true) : fullRangeFilter(stats),
    );
    // A 64-bit column arrives downcast to float32, so an id-shaped one cannot be
    // compared exactly. Say it once, here, rather than silently narrowing on
    // rounded values.
    this.statusEl.textContent = hasApproximateValues(stats)
      ? `${name} is ${stats.dtype} and is read as float32, so values above ` +
        "16,777,216 are approximate — exact for scores and codes, not for ids."
      : "";
    this.rebuild();
    this.updatePreview();
  }

  /**
   * Rebuild the live preview from the staged predicates.
   *
   * The preview group carries NO ROIs: an attribute-only group is a first-class
   * dissection (see `RoiGroupConfig.attrFilters`), so the geometry it selects is
   * everything matching, wherever it is.
   */
  private updatePreview(): void {
    if (this.staged.length === 0) {
      this.roiFilter.setPreviewGroup(undefined);
      return;
    }
    this.roiFilter.setPreviewGroup({
      name: "Attribute selection",
      color: vec3.clone(PREVIEW_COLOR),
      visible: true,
      opacity: 1,
      colorBy: { kind: "group" },
      attrFilters: this.staged.map((f) => ({ ...f })),
      rois: [],
    });
    // A staged selection is meant to be seen, so turn the filter on.
    this.roiFilter.active = true;
  }

  private commit(): void {
    if (this.staged.length === 0) return;
    const name = this.nameInput.value.trim();
    this.roiFilter.commitPreviewGroup(name.length > 0 ? name : undefined);
    // The preview was consumed into a real group; reset the staging area.
    this.staged = [];
    this.nameInput.value = "";
    this.statusEl.textContent = "";
    this.rebuild();
  }

  private clear(): void {
    if (this.staged.length === 0) return;
    this.staged = [];
    this.roiFilter.setPreviewGroup(undefined);
    this.statusEl.textContent = "";
    this.rebuild();
  }
}
