/**
 * @license
 * Copyright 2024 Google Inc.
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
 * "Add group from JSON": paste or upload a dissection rather than reaching for
 * the shared store.  The store needs a configured bucket and a network round
 * trip; this needs neither, which is what makes it the path for a group that
 * arrived by email or was lifted out of a URL hash.
 */

import "#src/datasource/zarr-vectors/import_group_dialog.css";

import type { ImportedGroup } from "#src/datasource/zarr-vectors/group_import.js";
import { parseGroupsJson } from "#src/datasource/zarr-vectors/group_import.js";
import { Overlay } from "#src/overlay.js";
import { makeIcon } from "#src/widget/icon.js";

export interface ImportGroupOptions {
  /** Returns a message to show on success; throws to report a failure. */
  onImport: (groups: ImportedGroup[]) => string;
}

export class ImportGroupDialog extends Overlay {
  private textarea = document.createElement("textarea");
  private errorEl = document.createElement("div");

  constructor(private options: ImportGroupOptions) {
    super();
    const { content } = this;
    content.classList.add("neuroglancer-roi-import-dialog");

    const title = document.createElement("h3");
    title.textContent = "Add group from JSON";
    content.appendChild(title);

    const help = document.createElement("div");
    help.classList.add("neuroglancer-roi-import-help");
    help.textContent =
      "Paste one group, an array of groups, or a saved group document.";
    content.appendChild(help);

    this.textarea.classList.add("neuroglancer-roi-import-textarea");
    this.textarea.spellcheck = false;
    this.textarea.placeholder =
      '{"name": "Motor CST", "color": "#ff0000", "rois": [...]}';
    // Ctrl/Cmd+Enter imports, matching the "submit without reaching for the
    // mouse" convention of the other text-entry overlays.
    this.textarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        this.submit();
      }
    });
    content.appendChild(this.textarea);

    const file = document.createElement("input");
    file.type = "file";
    file.accept = "application/json,.json";
    file.classList.add("neuroglancer-roi-import-file");
    file.addEventListener("change", async () => {
      const chosen = file.files?.[0];
      if (chosen === undefined) return;
      // Load into the textarea rather than importing straight away, so the
      // content is reviewable and a parse error points at something visible.
      this.textarea.value = await chosen.text();
      this.showError(undefined);
    });
    content.appendChild(file);

    this.errorEl.classList.add("neuroglancer-roi-import-error");
    content.appendChild(this.errorEl);

    const buttons = document.createElement("div");
    buttons.classList.add("neuroglancer-roi-import-buttons");
    buttons.appendChild(
      makeIcon({
        text: "Import",
        title: "Add the pasted group(s) to this layer (Ctrl+Enter)",
        onClick: () => this.submit(),
      }),
    );
    buttons.appendChild(
      makeIcon({
        text: "Cancel",
        title: "Close without importing",
        onClick: () => this.dispose(),
      }),
    );
    content.appendChild(buttons);

    this.textarea.focus();
  }

  private showError(message: string | undefined): void {
    this.errorEl.textContent = message ?? "";
    this.errorEl.style.display = message === undefined ? "none" : "";
  }

  private submit(): void {
    let groups: ImportedGroup[];
    try {
      groups = parseGroupsJson(this.textarea.value);
    } catch (e) {
      // Errors stay in the dialog rather than becoming a transient status
      // message: the text that caused them is still on screen and editable.
      this.showError((e as Error).message);
      return;
    }
    try {
      this.options.onImport(groups);
    } catch (e) {
      this.showError((e as Error).message);
      return;
    }
    this.dispose();
  }
}
