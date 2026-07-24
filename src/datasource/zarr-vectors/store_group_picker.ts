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
 * @file The "From store" checklist in the Filter tab.
 *
 * Lists the dissections already saved for THIS dataset and lets each be shown
 * or hidden with a tick. Ticking one that is not loaded fetches it and inserts
 * it as an ordinary group, so from that point on it has the same colour,
 * opacity, high-detail and expand-to-layer controls as anything drawn by hand.
 *
 * Unticking HIDES rather than removes. Removing would silently discard any
 * edits made since it was loaded, and "view/hide" should not be a destructive
 * action; the group's own delete button is the explicit way to get rid of it.
 *
 * Listing is anonymous — the bucket is public-read — so this populates without
 * any sign-in. Only saving needs a token.
 */

import "#src/datasource/zarr-vectors/store_group_picker.css";

import {
  groupFromJson,
  type RoiFilterState,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";
import {
  groupIdForDocument,
  rememberSavedDocument,
} from "#src/datasource/zarr-vectors/store_provenance.js";
import { getRoiStoreAuth } from "#src/roi_store/credentials.js";
import {
  getRoiGroupStore,
  RoiStoreListForbiddenError,
  roiGroupStoreChanged,
} from "#src/roi_store/gcs_client.js";
import type { RoiGroupSummary } from "#src/roi_store/schema.js";
import { StatusMessage } from "#src/status.js";
import { RefCounted } from "#src/util/disposable.js";
import { makeIcon } from "#src/widget/icon.js";

export interface StoreGroupPickerOptions {
  roiFilter: RoiFilterState;
  /**
   * Data source of the layer, read lazily: the tab can be built before the
   * source has resolved, and the list must filter on the resolved value.
   */
  getSourceUrl: () => string | undefined;
  /** Called after a group is inserted, so the tab can expand it. */
  onGroupInserted: (groupId: number) => void;
}

export class StoreGroupPicker extends RefCounted {
  element = document.createElement("div");
  private summaryEl = document.createElement("summary");
  private statusEl = document.createElement("div");
  private listEl = document.createElement("div");
  private details = document.createElement("details");
  private abortController = new AbortController();
  private summaries: RoiGroupSummary[] = [];
  /** Guards against a slower earlier listing rendering over a newer one. */
  private refreshGeneration = 0;
  private loading = false;

  constructor(private options: StoreGroupPickerOptions) {
    super();
    this.element.classList.add("neuroglancer-store-picker");

    this.details.classList.add("neuroglancer-store-picker-details");
    this.summaryEl.classList.add("neuroglancer-store-picker-summary");
    this.details.appendChild(this.summaryEl);

    const body = document.createElement("div");
    body.classList.add("neuroglancer-store-picker-body");
    this.statusEl.classList.add("neuroglancer-store-picker-status");
    this.listEl.classList.add("neuroglancer-store-picker-list");
    body.appendChild(this.statusEl);
    body.appendChild(this.listEl);

    const refresh = makeIcon({
      text: "Refresh",
      title: "Re-read the shared store",
      onClick: (e) => {
        e.preventDefault();
        void this.refresh();
      },
    });
    refresh.classList.add("neuroglancer-store-picker-refresh");
    body.appendChild(refresh);
    this.details.appendChild(body);
    this.element.appendChild(this.details);

    // A save elsewhere changes what this should list.
    this.registerDisposer(roiGroupStoreChanged.add(() => void this.refresh()));
    // Ticks reflect group visibility, which anything can change.
    this.registerDisposer(
      this.options.roiFilter.changed.add(() => this.updateTicks()),
    );

    this.setHeading();
    void this.refresh();
  }

  private setHeading(): void {
    const n = this.summaries.length;
    const label = this.loading
      ? "From store: loading…"
      : n === 0
        ? "From store: none found"
        : `From store: ${n} group${n === 1 ? "" : "s"}`;
    this.summaryEl.textContent = label;
  }

  async refresh(): Promise<void> {
    const generation = ++this.refreshGeneration;
    this.loading = true;
    this.setHeading();
    try {
      const all = await getRoiGroupStore().list(this.abortController.signal);
      if (this.wasDisposed || generation !== this.refreshGeneration) return;
      const sourceUrl = this.options.getSourceUrl();
      // Only this dataset: ROI coordinates are in tract model space, so a group
      // saved against other data would place its regions meaninglessly. Entries
      // with no recorded source are kept — there is nothing to contradict.
      this.summaries = all.filter(
        (s) =>
          sourceUrl === undefined ||
          s.sourceUrl === undefined ||
          s.sourceUrl === sourceUrl,
      );
      this.summaries.sort((a, b) => a.name.localeCompare(b.name));
      this.loading = false;
      this.setHeading();
      this.render();
    } catch (e) {
      if (this.wasDisposed || generation !== this.refreshGeneration) return;
      this.loading = false;
      this.summaries = [];
      this.setHeading();
      this.listEl.textContent = "";
      this.showError(e as Error);
    }
  }

  /**
   * A bucket that serves objects publicly can still withhold listing, which is
   * a configuration problem with two specific remedies rather than a transient
   * failure — so say which, and offer the one the user can act on here.
   */
  private showError(error: Error): void {
    this.statusEl.textContent = "";
    const message = document.createElement("div");
    message.textContent =
      error instanceof RoiStoreListForbiddenError
        ? error.message
        : `Could not read the store: ${error.message}`;
    this.statusEl.appendChild(message);

    if (error instanceof RoiStoreListForbiddenError && !error.signedIn) {
      const signIn = makeIcon({
        text: "Sign in and retry",
        title: "Sign in to Google, then re-read the store",
        onClick: (e) => {
          e.preventDefault();
          // getCredentialsWithStatus drives its own status UI and rejects only
          // on cancellation, which needs no extra message here.
          void getRoiStoreAuth()
            .signIn()
            .then(() => this.refresh())
            .catch((err) => {
              // Cancellation needs no message, but a throw before the status UI
              // (e.g. no credentials provider) would otherwise look like a no-op.
              console.error("[roi-store] sign-in failed:", err);
            });
        },
      });
      this.statusEl.appendChild(signIn);
    }
  }

  private render(): void {
    this.listEl.textContent = "";
    if (this.summaries.length === 0) {
      this.statusEl.textContent =
        "No saved groups for this dataset yet. Save one with the “Save to store” button on a group.";
      return;
    }
    this.statusEl.textContent = "";
    for (const summary of this.summaries) {
      this.listEl.appendChild(this.makeRow(summary));
    }
    this.updateTicks();
  }

  private makeRow(summary: RoiGroupSummary): HTMLElement {
    const row = document.createElement("label");
    row.classList.add("neuroglancer-store-picker-row");

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.dataset.documentId = summary.id;
    checkbox.addEventListener("change", () => {
      void this.toggle(summary, checkbox);
    });
    row.appendChild(checkbox);

    const name = document.createElement("span");
    name.classList.add("neuroglancer-store-picker-name");
    name.textContent = summary.name;
    row.appendChild(name);

    if (summary.createdBy !== undefined) {
      const author = document.createElement("span");
      author.classList.add("neuroglancer-store-picker-author");
      author.textContent = summary.createdBy;
      row.appendChild(author);
    }
    row.title = "Show this saved group on the current layer";
    return row;
  }

  /** Reflect which stored groups are loaded AND visible. */
  private updateTicks(): void {
    const { roiFilter } = this.options;
    for (const el of this.listEl.querySelectorAll("input[type=checkbox]")) {
      const checkbox = el as HTMLInputElement;
      const documentId = checkbox.dataset.documentId;
      if (documentId === undefined) continue;
      const groupId = groupIdForDocument(roiFilter, documentId);
      const group =
        groupId === undefined
          ? undefined
          : roiFilter.groups.find((g) => g.id === groupId);
      checkbox.checked = group?.visible === true;
    }
  }

  private async toggle(
    summary: RoiGroupSummary,
    checkbox: HTMLInputElement,
  ): Promise<void> {
    const { roiFilter } = this.options;
    const existingId = groupIdForDocument(roiFilter, summary.id);
    if (existingId !== undefined) {
      // Already on screen: this is purely a visibility toggle. Hiding never
      // discards the group, so edits made since loading survive.
      roiFilter.updateGroup(existingId, { visible: checkbox.checked });
      return;
    }
    if (!checkbox.checked) return;

    checkbox.disabled = true;
    try {
      const doc = await getRoiGroupStore().read(
        summary.id,
        this.abortController.signal,
      );
      if (this.wasDisposed) return;
      const { id: _placeholder, ...group } = groupFromJson(doc.group, 0);
      const groupId = roiFilter.insertGroup({ ...group, visible: true });
      // Record where it came from, so re-saving updates this document rather
      // than making a copy, and so the tick survives a re-render.
      rememberSavedDocument(roiFilter, groupId, {
        id: doc.id,
        createdAt: doc.createdAt,
      });
      // A loaded dissection is invisible with the filter off, and ticking it is
      // an explicit request to see it.
      roiFilter.active = true;
      this.options.onGroupInserted(groupId);
    } catch (e) {
      if (this.wasDisposed) return;
      checkbox.checked = false;
      StatusMessage.showTemporaryMessage(
        `Could not load “${summary.name}”: ${(e as Error).message}`,
        6000,
      );
    } finally {
      checkbox.disabled = false;
    }
  }

  disposed() {
    this.abortController.abort();
    this.element.remove();
    super.disposed();
  }
}
