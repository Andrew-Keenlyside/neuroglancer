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
  labelled,
  makeSelect,
} from "#src/datasource/zarr-vectors/filter_widgets.js";
import { ImportGroupDialog } from "#src/datasource/zarr-vectors/import_group_dialog.js";
import {
  RoiOperator,
  RoiPredicate,
  type Roi,
  type RoiShape,
} from "#src/datasource/zarr-vectors/roi.js";
import {
  groupFromJson,
  groupToJson,
  RoiFilterState,
  type RoiGroup,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";
import { StoreGroupPicker } from "#src/datasource/zarr-vectors/store_group_picker.js";
import {
  rememberSavedDocument,
  savedDocumentFor,
} from "#src/datasource/zarr-vectors/store_provenance.js";
import { deleteLayer, makeLayer } from "#src/layer/index.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  LINKED_SEGMENTATION_GROUP_JSON_KEY,
  ROI_FILTER_JSON_KEY,
} from "#src/layer/segmentation/json_keys.js";
import { RoiGroupBrowseDialog } from "#src/roi_store/browse_dialog.js";
import { roiStoreEnabled } from "#src/roi_store/config.js";
import { getRoiStoreAuth } from "#src/roi_store/credentials.js";
import { getRoiGroupStore } from "#src/roi_store/gcs_client.js";
import {
  makeRoiGroupDocument,
  type RoiGroupDocument,
} from "#src/roi_store/schema.js";
import { StatusMessage } from "#src/status.js";
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
/**
 * Roles offered on the first ROI, which seeds the fold instead of folding into
 * it: `Or` is degenerate there (`false || x === x`), so only the two seeding
 * modes are meaningful.
 */
const SEED_ROLE_OPTIONS: { value: RoiOperator; label: string }[] = [
  { value: RoiOperator.AND, label: "Include" },
  { value: RoiOperator.ANDNOT, label: "Exclude" },
];
const PREDICATE_OPTIONS: { value: RoiPredicate; label: string }[] = [
  { value: RoiPredicate.ANY_SEGMENT, label: "Crosses" },
  { value: RoiPredicate.ANY_VERTEX, label: "Point inside" },
];

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

function boxCentre(shape: {
  lower: Float32Array;
  upper: Float32Array;
}): number[] {
  return Array.from(shape.lower, (lo, i) => (lo + shape.upper[i]) / 2);
}
function boxSize(shape: {
  lower: Float32Array;
  upper: Float32Array;
}): number[] {
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
      if (
        thin >= 0 &&
        size.filter((s) => s > DEFAULT_PLANE_THICKNESS + 1e-6).length === 2
      ) {
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
  /**
   * Which groups are expanded (showing their ROI section). Independent per
   * group, so any group can be collapsed/re-expanded via its caret — collapsing
   * one no longer hides the others. New groups start expanded.
   */
  private expandedGroupIds = new Set<number>();
  /**
   * The structural (groups/ROIs/expanded) signature the body currently reflects.
   * `undefined` until the first build so the initial render always happens — a
   * zero-group filter has an empty-string signature, which must still differ
   * from the pre-build state (else the "+ New group" button never renders).
   */
  private structuralSig: string | undefined = undefined;
  private bodyEl: HTMLElement;
  /** Disposers for the widgets in the current body build. */
  private bodyContext = new RefCounted();

  /** Groups with a save in flight, so a second click cannot double-write. */
  private savesInFlight = new Set<number>();
  /** Abandons in-flight saves when the tab goes away. */
  private saveAbortController = new AbortController();

  constructor(public layer: SegmentationUserLayer) {
    super();
    this.roiFilter = layer.displayState.roiFilter;
    this.passingSegments = layer.displayState.roiPassingSegments;
    const { element } = this;
    element.classList.add("neuroglancer-streamline-filter-tab");

    element.appendChild(this.makeHeader());

    // Built once and kept outside `bodyEl`: it does network I/O, and rebuilding
    // it on every structural change would re-list the bucket on each edit.
    if (roiStoreEnabled) {
      const picker = this.registerDisposer(
        new StoreGroupPicker({
          roiFilter: this.roiFilter,
          getSourceUrl: () => this.currentSourceUrl(),
          onGroupInserted: (groupId) => {
            this.expandedGroupIds.add(groupId);
            this.onChanged();
          },
        }),
      );
      element.appendChild(picker.element);
    }

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
    // The two sets change independently -- the budget can be re-spent without
    // the dissection changing -- so the label needs both signals.
    const highDetail = this.layer.displayState.roiHighDetailSegments;
    if (highDetail !== undefined) {
      this.registerDisposer(highDetail.changed.add(() => this.updateCount()));
    }
    // Open with the last group expanded so its ROIs are immediately visible
    // (others start collapsed; each has a caret to expand/collapse on demand).
    const groups = this.roiFilter.groups;
    if (groups.length > 0) {
      this.expandedGroupIds.add(groups[groups.length - 1].id);
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
      labelled(
        "Active",
        active.element,
        "neuroglancer-streamline-filter-active",
      ),
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
    // The non-passing ("off") opacity now lives on the Render tab as a global
    // setting; per-group "on" opacity is a slider on each group row below.
    return header;
  }

  // --- structural rebuild ---------------------------------------------------

  private structuralSignature(): string {
    // Include each ROI's shape kind so a restore that keeps ids + counts but
    // changes a kind still rebuilds (the sliders are wired per kind), and each
    // group's expanded state (its ROI section is only in the DOM when open);
    // exclude group colour/name/visibility/opacity and ROI name (all bound live
    // to inputs — rebuilding on those would recreate an input mid-edit, so a
    // rename would lose focus on the first keystroke that commits).
    return this.roiFilter.groups
      .map(
        (g) =>
          `${g.id}${this.expandedGroupIds.has(g.id) ? "+" : "-"}:` +
          g.rois.map((r) => r.shape.kind[0]).join(""),
      )
      .join(",");
  }

  private onChanged(): void {
    // Drop expansion state for groups that no longer exist so the set does not
    // leak ids across deletes/restores.
    const liveIds = new Set(this.roiFilter.groups.map((g) => g.id));
    for (const id of this.expandedGroupIds) {
      if (!liveIds.has(id)) this.expandedGroupIds.delete(id);
    }
    const sig = this.structuralSignature();
    if (sig !== this.structuralSig) {
      this.structuralSig = sig;
      this.rebuildBody();
    }
    this.updateCount();
  }

  /** Re-read the current (immutable) group by id, or undefined if it is gone. */
  private currentGroup(id: number): RoiGroup | undefined {
    return this.roiFilter.groups.find((g) => g.id === id);
  }

  private toggleExpanded(id: number): void {
    if (this.expandedGroupIds.has(id)) this.expandedGroupIds.delete(id);
    else this.expandedGroupIds.add(id);
    this.onChanged();
  }

  private rebuildBody(): void {
    this.bodyContext.dispose();
    this.bodyContext = new RefCounted();
    const el = this.bodyEl;
    el.textContent = "";

    // Group list. The selected group's ROIs + add-buttons nest directly under
    // its row, so it is clear the ROIs belong to that group.
    const groupList = document.createElement("div");
    groupList.classList.add("neuroglancer-streamline-filter-group-list");
    for (const group of this.roiFilter.groups) {
      const expanded = this.expandedGroupIds.has(group.id);
      const container = document.createElement("div");
      container.classList.add("neuroglancer-streamline-filter-group");
      if (expanded) container.classList.add("neuroglancer-selected");
      container.appendChild(this.makeGroupRow(group, expanded));
      if (expanded) container.appendChild(this.makeRoiSection(group));
      groupList.appendChild(container);
    }
    el.appendChild(groupList);

    const addGroup = makeIcon({
      text: "+ New group",
      title: "Create a new tract group",
      onClick: () => {
        // addGroup() dispatches `changed` (rebuilding the list) before returning
        // the new id; expand the new group and re-run onChanged so its ROI-add
        // buttons show.
        this.expandedGroupIds.add(this.roiFilter.addGroup());
        this.onChanged();
      },
    });
    const addGroupAsLayer = makeIcon({
      text: "+ New group as layer",
      title:
        "Create a new tract group in a layer of its own, on the same data. " +
        "The new layer has its own visibility, ordering and render controls.",
      onClick: () => this.addGroupAsLayer(),
    });
    addGroupAsLayer.classList.add("neuroglancer-streamline-filter-add-group");

    const addRow = document.createElement("div");
    addRow.classList.add("neuroglancer-streamline-filter-add-row");
    addRow.appendChild(addGroup);
    addRow.appendChild(addGroupAsLayer);
    // Not gated on `roiStoreEnabled`: pasting JSON needs no bucket, and is the
    // only import path in a build with the store compiled out.
    addRow.appendChild(
      makeIcon({
        text: "+ From JSON…",
        title:
          "Add a group by pasting or uploading JSON — one group, an array of " +
          "groups, or a saved group document.",
        onClick: () => this.openImportDialog(),
      }),
    );
    if (roiStoreEnabled) {
      addRow.appendChild(
        makeIcon({
          text: "Browse saved…",
          title:
            "Open the shared library of saved groups. Browsing and loading " +
            "need no sign-in.",
          onClick: () => this.openBrowseDialog(),
        }),
      );
    }
    el.appendChild(addRow);
  }

  // --- shared ROI group store ----------------------------------------------

  /** The data source these ROIs were drawn against, if it is resolved. */
  private currentSourceUrl(): string | undefined {
    return this.layer.dataSources[0]?.spec.url;
  }

  /**
   * Write one group to the shared store.
   *
   * The group JSON is `groupToJson` output, identical to what the URL carries,
   * so there is a single serialisation to keep correct.
   */
  private async saveGroup(id: number): Promise<void> {
    if (this.savesInFlight.has(id)) return;
    const sourceUrl = this.currentSourceUrl();
    if (sourceUrl === undefined) {
      StatusMessage.showTemporaryMessage(
        "Cannot save: this layer has no resolved data source.",
        5000,
      );
      return;
    }
    const store = getRoiGroupStore();
    const auth = getRoiStoreAuth();
    const { signal } = this.saveAbortController;

    this.savesInFlight.add(id);
    const status = new StatusMessage(/*delay=*/ true);
    status.setText("Saving…");
    try {
      // Obtain the token BEFORE building the document. Saving is what triggers
      // sign-in, and `auth.email` is undefined until it completes — reading it
      // first left the first save of every session with no author recorded.
      await auth.getAccessToken(signal);
      if (this.wasDisposed) return;

      // Re-read the group only now: the user may have renamed, edited or
      // deleted it while the sign-in popup was open.
      const group = this.currentGroup(id);
      if (group === undefined) {
        StatusMessage.showTemporaryMessage(
          "Cannot save: that group no longer exists.",
          5000,
        );
        return;
      }
      status.setText(`Saving “${group.name}”…`);

      const previous = savedDocumentFor(this.roiFilter, id);
      const doc = makeRoiGroupDocument({
        group: groupToJson(group),
        source: { url: sourceUrl },
        scene: {
          // The URL hash is neuroglancer's own serialisation of the scene.
          url: window.location.href,
          layerName: this.layer.managedLayer.name,
        },
        createdBy: auth.email,
        id: previous?.id,
        createdAt: previous?.createdAt,
      });
      await store.save(doc, signal);
      rememberSavedDocument(this.roiFilter, id, {
        id: doc.id,
        createdAt: doc.createdAt,
      });
      if (this.wasDisposed) return;
      StatusMessage.showTemporaryMessage(`Saved “${group.name}”.`, 3000);
    } catch (e) {
      if (this.wasDisposed || signal.aborted) return;
      StatusMessage.showTemporaryMessage(
        `Could not save: ${(e as Error).message}`,
        6000,
      );
    } finally {
      // Always retire the status and the in-flight mark, including when the
      // sign-in is abandoned — otherwise the modal lingers and the group can
      // never be saved again.
      status.dispose();
      this.savesInFlight.delete(id);
    }
  }

  private openBrowseDialog(): void {
    new RoiGroupBrowseDialog({
      store: getRoiGroupStore(),
      auth: getRoiStoreAuth(),
      currentSourceUrl: this.currentSourceUrl(),
      onLoad: (doc) => this.loadDocument(doc),
    });
  }

  /**
   * Append a saved group to this layer.
   *
   * Uses `insertGroup` rather than `restoreState`: restoring would rebuild the
   * whole group list and renumber ids from 1, invalidating the ids this tab
   * holds in `expandedGroupIds` and re-reads at click time.  Appending assigns
   * a fresh id and leaves every existing group untouched.
   */
  private loadDocument(doc: RoiGroupDocument): string {
    // `id` is assigned by insertGroup; groupFromJson needs a placeholder.
    const { id: _placeholder, ...group } = groupFromJson(doc.group, 0);
    const [groupId] = this.appendGroups([group]);
    // Record where it came from, so re-saving updates this document instead of
    // adding a copy, and the store checklist shows it as loaded.
    rememberSavedDocument(this.roiFilter, groupId, {
      id: doc.id,
      createdAt: doc.createdAt,
    });
    return `Loaded “${group.name}”.`;
  }

  /**
   * Append already-parsed groups and reveal them; returns their new ids.
   *
   * The single insert path: both the store's browse dialog and the local JSON
   * import land here, so the "turn the filter on, expand what you just added"
   * behaviour cannot drift between them.
   */
  private appendGroups(groups: readonly Omit<RoiGroup, "id">[]): number[] {
    const ids: number[] = [];
    for (const group of groups) {
      // insertGroup dispatches `changed` once per group rather than once per
      // ROI, so a multi-group import costs one recompute per group, not per
      // region.
      const id = this.roiFilter.insertGroup(group);
      ids.push(id);
      this.expandedGroupIds.add(id);
    }
    // A loaded dissection is only visible with the filter on, and the user
    // just asked for it — turning it on is what they meant.
    this.roiFilter.active = true;
    this.onChanged();
    return ids;
  }

  private openImportDialog(): void {
    new ImportGroupDialog({
      onImport: (groups) => {
        this.appendGroups(groups);
        return groups.length === 1
          ? `Added “${groups[0].name}”.`
          : `Added ${groups.length} groups.`;
      },
    });
  }

  /**
   * Create a sibling layer on the same data, carrying one empty group.
   *
   * The group is created ONLY in the new layer. Each layer runs its own
   * dissection -- the recompute lives on the render-layer backend, not on the
   * shared chunk source -- so duplicating a group into both would evaluate the
   * same regions twice against the same geometry for no benefit. Moving the
   * work rather than copying it keeps the total roughly constant.
   *
   * The underlying data is not re-fetched: chunk sources are memoised by their
   * parameters on the viewer-wide chunk manager, so both layers share the
   * resolved data source, its metadata and its decoded chunks.
   */
  private addGroupAsLayer(existing?: RoiGroup) {
    const { layer } = this;
    const { managedLayer, manager } = layer;

    // A throwaway state seeds the group, so the new layer's group gets exactly
    // the same defaults, palette colour and JSON shape as `addGroup()` would --
    // or, when separating an existing group, carries it across verbatim.
    const seed = new RoiFilterState();
    if (existing === undefined) {
      seed.addGroup();
    } else {
      seed.insertGroup(existing);
    }
    seed.active = true;
    // The tractogram's ghost context is the ORIGINAL layer's job. Both layers
    // draw every streamline, so leaving the default ghost alpha here would
    // stack two ghost passes and read as a uniformly darker, denser brain.
    seed.ghostAlpha = 0;

    const spec = layer.toJSON();
    spec[ROI_FILTER_JSON_KEY] = seed.toJSON();
    // Cloning tool state would let the copy steal the original's bindings.
    delete spec.tool;
    delete spec.toolBindings;

    // Link the new layer's segmentation group to the one it came from, so the
    // two are views of a single segmentation rather than two that merely happen
    // to share a URL: selecting or hiding a tract in either is reflected in
    // both. `??=` preserves an existing link -- if the parent is already linked
    // to some other layer, the copy joins that same group rather than starting
    // a competing one -- and otherwise points at the parent itself.
    //
    // The colour group follows automatically: `restoreState` defaults it to the
    // segmentation group's name whenever the colour key is absent, and the key
    // is only written when the two roots differ. Per-group ROI colours are
    // unaffected either way, since those ride a dedicated per-layer map rather
    // than the shared stated-colour map.
    spec[LINKED_SEGMENTATION_GROUP_JSON_KEY] ??= managedLayer.name;

    const groupName = seed.groups[0]?.name ?? "Group 1";
    const newLayer = makeLayer(
      manager,
      `${managedLayer.name} / ${groupName}`,
      spec,
    );
    // Adjacent to its parent; `add` uniquifies the name.
    const parentIndex =
      manager.layerManager.managedLayers.indexOf(managedLayer);
    manager.add(newLayer, parentIndex !== -1 ? parentIndex + 1 : undefined);

    const { selectedLayer } = manager.root;
    selectedLayer.layer = newLayer;
    selectedLayer.visible = true;

    // Separating MOVES the group: it now lives in the new layer. Leaving a copy
    // behind would evaluate the same dissection twice against the same geometry
    // (each layer runs its own), and both copies would draw.
    if (existing !== undefined) {
      this.roiFilter.removeGroup(existing.id);
    }
  }

  /**
   * The layer this one's segmentation group is linked to, if it is not itself
   * the root -- i.e. the "main" layer a separated group can be merged back into.
   *
   * The link group's own predicate guarantees members are segmentation layers,
   * so this needs no `instanceof`: importing the class for one would create a
   * cycle, since the segmentation layer imports this tab.
   */
  private mergeTarget(): SegmentationUserLayer | undefined {
    const root = this.layer.displayState.linkedSegmentationGroup.root.value as
      | SegmentationUserLayer
      | undefined;
    if (root === undefined || root === this.layer) return undefined;
    return root.displayState?.roiFilter === undefined ? undefined : root;
  }

  /**
   * Move a group back into the layer this one is linked to.
   *
   * The inverse of separating. If that empties this layer, it is removed: an
   * ROI-filter layer with no groups has an inactive filter, which means it stops
   * ghosting and draws the whole tractogram at full strength — so a leftover
   * empty layer would silently double the tractogram rather than being merely
   * redundant.
   */
  private mergeGroupInto(group: RoiGroup, target: SegmentationUserLayer) {
    target.displayState.roiFilter.insertGroup(group);
    target.displayState.roiFilter.active = true;
    this.roiFilter.removeGroup(group.id);

    if (this.roiFilter.groups.length === 0) {
      const { managedLayer, manager } = this.layer;
      const { selectedLayer } = manager.root;
      selectedLayer.layer = target.managedLayer;
      selectedLayer.visible = true;
      deleteLayer(managedLayer);
    }
  }

  private makeGroupRow(group: RoiGroup, expanded: boolean): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-streamline-filter-group-row");
    if (expanded) row.classList.add("neuroglancer-selected");

    // Expand/collapse caret: toggles this group's ROI section. Clicking the row
    // background (not a control) toggles too, but the caret is the obvious grip.
    const caret = document.createElement("span");
    caret.classList.add("neuroglancer-streamline-filter-group-caret");
    caret.textContent = expanded ? "▾" : "▸"; // ▾ / ▸
    caret.title = expanded ? "Collapse group" : "Expand group";
    caret.addEventListener("click", (e) => {
      e.stopPropagation();
      this.toggleExpanded(group.id);
    });
    row.appendChild(caret);

    row.addEventListener("click", (e) => {
      // Let the row's own inputs/buttons handle their own clicks.
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLButtonElement ||
        (e.target instanceof HTMLElement &&
          e.target.closest(".neuroglancer-delete-button") !== null)
      ) {
        return;
      }
      this.toggleExpanded(group.id);
    });

    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.value = serializeColor(group.color);
    swatch.title = "Group colour";
    swatch.addEventListener("input", () =>
      this.roiFilter.updateGroup(group.id, {
        color: parseHexColor(swatch.value),
      }),
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

    // Separate / merge. The row's background click toggles expansion unless the
    // target is an input or button, and `makeIcon` returns a div — so both of
    // these must stop propagation or clicking them would also collapse the
    // group out from under the user.
    const target = this.mergeTarget();
    if (target === undefined) {
      const separate = makeIcon({
        text: "⇗",
        title:
          "Move this group into a layer of its own, linked to this one. " +
          "Its tracts get their own visibility, ordering and render controls.",
        onClick: (e) => {
          e.stopPropagation();
          // Re-read by id: `group` was captured when this row was built, and
          // moving an ROI does not change the structural signature that would
          // have rebuilt it, so the captured object can be stale.
          const current = this.currentGroup(group.id);
          if (current !== undefined) this.addGroupAsLayer(current);
        },
      });
      separate.classList.add("neuroglancer-streamline-filter-group-action");
      row.appendChild(separate);
    } else {
      const merge = makeIcon({
        text: "⇙",
        title: `Move this group back into “${target.managedLayer.name}”`,
        onClick: (e) => {
          e.stopPropagation();
          // Re-read by id; see the note on the separate action above.
          const current = this.currentGroup(group.id);
          if (current !== undefined) this.mergeGroupInto(current, target);
        },
      });
      merge.classList.add("neuroglancer-streamline-filter-group-action");
      row.appendChild(merge);
    }

    if (roiStoreEnabled) {
      const save = makeIcon({
        text: "Save to store",
        title:
          "Save this group to the shared store, so it can be reloaded later " +
          "or opened by someone else. Prompts for Google sign-in the first " +
          "time; re-saving updates the same entry rather than adding a copy.",
        onClick: (e) => {
          e.stopPropagation();
          // Re-read by id; see the note on the separate action above.
          void this.saveGroup(group.id);
        },
      });
      save.classList.add("neuroglancer-streamline-filter-group-action");
      row.appendChild(save);
    }

    row.appendChild(
      makeDeleteButton({
        title: "Delete group",
        onClick: (e) => {
          e.stopPropagation();
          this.roiFilter.removeGroup(group.id);
        },
      }),
    );

    // Per-group render controls (a sub-row): the group's "on" opacity and a
    // high-detail toggle (loads this group's passing tracts at full detail).
    const controls = document.createElement("div");
    controls.classList.add("neuroglancer-streamline-filter-group-controls");
    const opacity = this.bodyContext.registerDisposer(
      new RangeWidget(
        fieldWatchable(
          this.roiFilter.changed,
          // Live lookup: updateGroup replaces the group with a new immutable
          // object, so reading the captured `group` would be stale and the
          // widget would snap back to the build-time value on every drag.
          () => this.currentGroup(group.id)?.opacity ?? 1,
          (v) => this.roiFilter.updateGroup(group.id, { opacity: v }),
        ),
        { min: 0, max: 1, step: 0.01 },
      ),
    );
    controls.appendChild(labelled("Opacity", opacity.element));

    const highDetail = document.createElement("input");
    highDetail.type = "checkbox";
    highDetail.checked = group.highDetail;
    highDetail.title = "Load this group's tracts at full detail (slower)";
    highDetail.addEventListener("change", () =>
      this.roiFilter.updateGroup(group.id, { highDetail: highDetail.checked }),
    );
    controls.appendChild(labelled("High detail", highDetail));

    const wrapper = document.createElement("div");
    wrapper.classList.add("neuroglancer-streamline-filter-group-header");
    wrapper.appendChild(row);
    wrapper.appendChild(controls);
    return wrapper;
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

    const geometry = shapeLabel(roi.shape);
    const fallbackName = `ROI ${index + 1}`;

    const head = document.createElement("div");
    head.classList.add("neuroglancer-streamline-filter-roi-head");
    const name = document.createElement("input");
    name.type = "text";
    name.classList.add("neuroglancer-streamline-filter-roi-name");
    // Unnamed ROIs show the positional fallback as a *placeholder*, not a
    // value: it tracks the current index, and clearing the field reverts to it
    // instead of storing an empty name.
    name.value = roi.name ?? "";
    name.placeholder = fallbackName;
    name.title = geometry;
    name.addEventListener("change", () => {
      const trimmed = name.value.trim();
      name.value = trimmed;
      this.roiFilter.updateRoi(groupId, index, {
        name: trimmed === "" ? undefined : trimmed,
      });
    });
    head.appendChild(name);
    head.appendChild(
      makeDeleteButton({
        title: "Delete ROI",
        onClick: () => this.roiFilter.removeRoi(groupId, index),
      }),
    );
    card.appendChild(head);

    // The derived geometry string is still the only way to tell two identically
    // named regions apart, so it stays visible as a subtitle rather than moving
    // entirely into the input's tooltip.
    const label = document.createElement("div");
    label.classList.add("neuroglancer-streamline-filter-roi-label");
    label.textContent = geometry;
    card.appendChild(label);

    const controls = document.createElement("div");
    controls.classList.add("neuroglancer-streamline-filter-roi-controls");
    // Role: the first ROI seeds the fold rather than folding into it, so it
    // offers only Include/Exclude. A stale `Or` left at index 0 by a reorder is
    // shown (and evaluated) as Include.
    const isSeed = index === 0;
    const role = makeSelect(
      isSeed ? SEED_ROLE_OPTIONS : ROLE_OPTIONS,
      isSeed && roi.operator !== RoiOperator.ANDNOT
        ? RoiOperator.AND
        : roi.operator,
      (v) => this.roiFilter.updateRoi(groupId, index, { operator: v }),
    );
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
      shape.kind === "ellipsoid"
        ? shape.center.length
        : shape.kind === "box"
          ? shape.lower.length
          : 3;

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
          () => this.currentShape(groupId, index, "ellipsoid")?.center[i] ?? 0,
          (v) => {
            const s = this.currentShape(groupId, index, "ellipsoid");
            if (s === undefined) return;
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
        () => this.currentShape(groupId, index, "ellipsoid")?.radii[0] ?? 0,
        (v) => {
          const s = this.currentShape(groupId, index, "ellipsoid");
          if (s === undefined) return;
          const radii = new Float32Array(s.radii.length).fill(v);
          this.roiFilter.updateRoi(groupId, index, { shape: { ...s, radii } });
        },
      );
    } else if (shape.kind === "box") {
      for (let i = 0; i < rank; ++i) {
        slider(
          `${axisName[i] ?? i}`,
          this.axisRange(i),
          () => {
            const s = this.currentShape(groupId, index, "box");
            return s === undefined ? 0 : boxCentre(s)[i];
          },
          (v) => {
            const s = this.currentShape(groupId, index, "box");
            if (s === undefined) return;
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
          () => {
            const s = this.currentShape(groupId, index, "box");
            return s === undefined ? 0 : boxSize(s)[i];
          },
          (v) => {
            const s = this.currentShape(groupId, index, "box");
            if (s === undefined) return;
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

  /**
   * Re-read the current shape of an ROI, or `undefined` if it is gone or its
   * kind changed. Returning undefined (not throwing) keeps a stale slider
   * handler harmless if it fires on the same `changed` dispatch that deleted the
   * ROI, before the structural rebuild disposes the widget.
   */
  private currentShape<K extends RoiShape["kind"]>(
    groupId: number,
    index: number,
    kind: K,
  ): Extract<RoiShape, { kind: K }> | undefined {
    const shape = this.roiFilter.groups.find((g) => g.id === groupId)?.rois[
      index
    ]?.shape;
    return shape !== undefined && shape.kind === kind
      ? (shape as Extract<RoiShape, { kind: K }>)
      : undefined;
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
    // The dissection is evaluated at the finest level whose regional geometry is
    // resident, so the passing count is no longer "this level" -- it can name
    // tracts that exist only at levels far finer than the one being drawn. Only
    // those the full-detail budget admits are actually shown, so reporting the
    // pass count alone would look like most of them had silently vanished.
    const passing = this.passingSegments?.size ?? 0;
    const shown = this.layer.displayState.roiHighDetailSegments?.size;
    const plural = passing === 1 ? "" : "s";
    this.countEl.textContent =
      shown === undefined
        ? `${passing.toLocaleString()} streamline${plural} pass`
        : `${passing.toLocaleString()} streamline${plural} pass · ` +
          `${shown.toLocaleString()} at full detail`;
  }

  disposed() {
    // Abandon any save still waiting on sign-in or the network, so it cannot
    // report against a panel that has gone away.
    this.saveAbortController.abort();
    super.disposed();
  }
}
