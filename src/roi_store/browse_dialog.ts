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
 * @file Browser for the shared ROI group library.
 *
 * Listing and loading are anonymous — the bucket is public-read — so this opens
 * and works without signing in.  Only Delete needs a token, so it is offered
 * only when there is one.
 */

import "#src/roi_store/browse_dialog.css";

import { Overlay } from "#src/overlay.js";
import type { RoiStoreAuth } from "#src/roi_store/credentials.js";
import type { RoiGroupStore } from "#src/roi_store/gcs_client.js";
import type {
  RoiGroupDocument,
  RoiGroupSummary,
} from "#src/roi_store/schema.js";
import { StatusMessage } from "#src/status.js";
import { makeDeleteButton } from "#src/widget/delete_button.js";
import { makeIcon } from "#src/widget/icon.js";

export interface RoiGroupBrowseOptions {
  store: RoiGroupStore;
  auth: RoiStoreAuth;
  /**
   * Data source of the layer the dialog was opened from.  Entries saved
   * against a different source are flagged: ROI coordinates are in tract model
   * space, so loading one onto unrelated data places the regions meaninglessly.
   */
  currentSourceUrl: string | undefined;
  /** Invoked with the chosen document; returns a short confirmation to show. */
  onLoad: (doc: RoiGroupDocument) => string;
}

export class RoiGroupBrowseDialog extends Overlay {
  private listEl = document.createElement("div");
  private statusEl = document.createElement("div");
  private abortController = new AbortController();
  /**
   * Incremented per refresh so a slower earlier listing cannot render over a
   * newer one.  Three entry points can overlap — the constructor's unawaited
   * refresh, the Refresh button, and the one after a delete — and `render`
   * only appends, so without this every row appears once per in-flight call.
   */
  private refreshGeneration = 0;

  constructor(private options: RoiGroupBrowseOptions) {
    super();
    const { content } = this;
    content.classList.add("neuroglancer-roi-store-browse");

    const header = document.createElement("h3");
    header.textContent = "Saved ROI groups";
    content.appendChild(header);

    this.statusEl.classList.add("neuroglancer-roi-store-browse-status");
    content.appendChild(this.statusEl);

    this.listEl.classList.add("neuroglancer-roi-store-browse-list");
    content.appendChild(this.listEl);

    const footer = document.createElement("div");
    footer.classList.add("neuroglancer-roi-store-browse-footer");
    footer.appendChild(
      makeIcon({ text: "Refresh", onClick: () => this.refresh() }),
    );
    footer.appendChild(
      makeIcon({ text: "Close", onClick: () => this.close() }),
    );
    content.appendChild(footer);

    this.refresh();
  }

  private setStatus(text: string) {
    this.statusEl.textContent = text;
  }

  private async refresh() {
    const generation = ++this.refreshGeneration;
    this.setStatus("Loading…");
    try {
      const summaries = await this.options.store.list(
        this.abortController.signal,
      );
      if (this.wasDisposed || generation !== this.refreshGeneration) return;
      summaries.sort((a, b) =>
        (a.updated ?? "") < (b.updated ?? "") ? 1 : -1,
      );
      // Cleared here rather than before the await, so an overtaken refresh
      // never blanks the list that a newer one is about to fill.
      this.listEl.textContent = "";
      this.render(summaries);
    } catch (e) {
      if (this.wasDisposed || generation !== this.refreshGeneration) return;
      this.listEl.textContent = "";
      this.setStatus(`Could not list saved groups: ${(e as Error).message}`);
    }
  }

  private render(summaries: RoiGroupSummary[]) {
    if (summaries.length === 0) {
      this.setStatus("No saved groups yet.");
      return;
    }
    this.setStatus(
      `${summaries.length} saved group${summaries.length === 1 ? "" : "s"}.`,
    );
    for (const summary of summaries) {
      this.listEl.appendChild(this.makeRow(summary));
    }
  }

  private makeRow(summary: RoiGroupSummary): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-roi-store-browse-row");

    const name = document.createElement("span");
    name.classList.add("neuroglancer-roi-store-browse-name");
    name.textContent = summary.name;
    row.appendChild(name);

    const meta = document.createElement("span");
    meta.classList.add("neuroglancer-roi-store-browse-meta");
    const parts: string[] = [];
    if (summary.createdBy !== undefined) parts.push(summary.createdBy);
    if (summary.updated !== undefined) {
      parts.push(new Date(summary.updated).toLocaleString());
    }
    meta.textContent = parts.join(" · ");
    row.appendChild(meta);

    const { currentSourceUrl } = this.options;
    const mismatched =
      currentSourceUrl !== undefined &&
      summary.sourceUrl !== undefined &&
      summary.sourceUrl !== currentSourceUrl;
    if (mismatched) {
      // A warning, not a block: the same tractography is often served from
      // more than one URL, and only the user can tell whether it is the same
      // data.
      const warn = document.createElement("span");
      warn.classList.add("neuroglancer-roi-store-browse-warn");
      warn.textContent = "different source";
      warn.title =
        `Saved against ${summary.sourceUrl}, but this layer loads ` +
        `${currentSourceUrl}. ROI coordinates are in tract model space, so ` +
        "they will only line up if this is the same data.";
      row.appendChild(warn);
    }

    row.appendChild(
      makeIcon({
        text: "Load",
        title: "Add this group to the current layer",
        onClick: () => this.load(summary),
      }),
    );

    // Writes need a token; listing and loading do not. Rather than prompting
    // for sign-in from a browse dialog, only offer Delete once signed in.
    if (this.options.auth.signedIn) {
      row.appendChild(
        makeDeleteButton({
          title: "Delete this saved group from the shared store",
          onClick: () => this.remove(summary),
        }),
      );
    }
    return row;
  }

  private async load(summary: RoiGroupSummary) {
    this.setStatus(`Loading “${summary.name}”…`);
    try {
      const doc = await this.options.store.read(
        summary.id,
        this.abortController.signal,
      );
      const message = this.options.onLoad(doc);
      StatusMessage.showTemporaryMessage(message, 4000);
      this.close();
    } catch (e) {
      if (this.wasDisposed) return;
      this.setStatus(`Could not load: ${(e as Error).message}`);
    }
  }

  private async remove(summary: RoiGroupSummary) {
    if (
      !window.confirm(
        `Delete “${summary.name}” from the shared store? This affects everyone.`,
      )
    ) {
      return;
    }
    this.setStatus(`Deleting “${summary.name}”…`);
    try {
      await this.options.store.delete(summary.id, this.abortController.signal);
      if (this.wasDisposed) return;
      await this.refresh();
    } catch (e) {
      if (this.wasDisposed) return;
      this.setStatus(`Could not delete: ${(e as Error).message}`);
    }
  }

  disposed() {
    // Abandon any in-flight list/read so a closed dialog stops touching the DOM.
    this.abortController.abort();
    super.disposed();
  }
}
