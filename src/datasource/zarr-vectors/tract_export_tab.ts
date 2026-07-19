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
 * @file The "Export" tab: write selected ROI groups out as a TrackVis `.trk`
 * file or a new zarr-vectors store.
 *
 * The viewer cannot do the writing itself -- the writers live in
 * `zarr-vectors-tools`, which needs zarr/numcodecs/nibabel and does not import
 * under Pyodide. So this tab only ever *describes* the work as a job spec and
 * hands it to the native exporter (`neuroglancer.tract_export`). When no
 * exporter is reachable the same spec is offered as a download, to be run with
 * `python -m neuroglancer.tract_export`; the two paths share one payload so a
 * downloaded file and a posted body cannot mean different things.
 *
 * Note the spec carries ROI *geometry*, never the viewer's passing set: that
 * set covers only the chunks resident at the level of detail on screen, and the
 * exporter re-evaluates the dissection at level 0 for exactly that reason.
 */

import "#src/datasource/zarr-vectors/tract_export_tab.css";

import {
  buildExportSpec,
  type ExportFormat,
} from "#src/datasource/zarr-vectors/export_job.js";
import {
  labelled,
  makeStringSelect,
} from "#src/datasource/zarr-vectors/filter_widgets.js";
import type {
  RoiFilterState,
  RoiGroup,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { RefCounted } from "#src/util/disposable.js";
import { saveBlobToFile } from "#src/util/file_download.js";
import { makeIcon } from "#src/widget/icon.js";
import { Tab } from "#src/widget/tab_view.js";

const FORMAT_OPTIONS = [
  { value: "trk", label: "TrackVis (.trk)" },
  { value: "zvf", label: "New zarr-vectors store" },
];

const DEFAULT_BASENAME = "dissection";

/**
 * The sidecar exporter URL, remembered for the session.
 *
 * Module-scoped rather than per-tab: the tab is rebuilt whenever the layer
 * panel is closed and reopened, and retyping a token-bearing URL each time
 * would be miserable. Not persisted to the layer state -- it carries a
 * capability token and has no business in the URL hash or a saved scene.
 */
let sessionExporterUrl = "";

/**
 * Where to POST an export job, or `undefined` if nothing can run one.
 *
 * Two routes, in order. When the viewer is served by the Python integration the
 * page lives under `/v/<token>/`, and that token scopes the export route on the
 * same server -- zero configuration. A statically served build (the browser and
 * Pyodide bundles) has no such server, so it falls back to a sidecar started
 * with `python -m neuroglancer.tract_export --serve`, whose URL the user pastes
 * in. Both are token-bearing because both write a file at a path the request
 * chooses.
 */
function exportEndpoint(): string | undefined {
  const match = window.location.pathname.match(/^(.*)\/v\/([^/]+)/);
  if (match !== null) {
    const prefix = `${window.location.origin}${match[1]}`.replace(/\/+$/, "");
    return `${prefix}/neuroglancer/export/tract/${match[2]}`;
  }
  // Statically served build (browser or Pyodide): no viewer server exists, so
  // fall back to a sidecar the user started and pasted the URL for.
  return sessionExporterUrl === "" ? undefined : sessionExporterUrl;
}

interface ExportSummary {
  written?: boolean;
  streamline_count?: number;
  vertex_count?: number;
  object_count?: number;
  candidate_object_count?: number;
  output_path?: string;
  message?: string;
  groups?: { name: string; count: number }[];
}

export class TractExportTab extends Tab {
  private roiFilter: RoiFilterState;
  /**
   * Which groups to export, by session group id.
   *
   * Tab-local and not persisted, mirroring how the Filter tab keeps
   * `expandedGroupIds`: an export selection is a momentary intent, not part of
   * the scene, and putting it in `RoiFilterState` would push it into the URL.
   */
  private selectedGroupIds = new Set<number>();
  private format: ExportFormat = "trk";
  private outputPath = "";
  /** Cleared and rebuilt whenever the group list changes structurally. */
  private bodyContext = new RefCounted();
  private bodyEl = document.createElement("div");
  private statusEl = document.createElement("div");
  private countEl = document.createElement("div");
  private groupSignature: string | undefined;
  /**
   * Set once a request proves no exporter is listening, so the tab stops
   * re-attempting on every click and shows the download route instead.
   */
  private endpointUnavailable = false;
  private busy = false;

  constructor(public layer: SegmentationUserLayer) {
    super();
    this.roiFilter = layer.displayState.roiFilter!;
    const { element } = this;
    element.classList.add("neuroglancer-tract-export-tab");

    this.bodyEl.classList.add("neuroglancer-tract-export-body");
    element.appendChild(this.bodyEl);
    this.statusEl.classList.add("neuroglancer-tract-export-status");
    element.appendChild(this.statusEl);
    this.countEl.classList.add("neuroglancer-tract-export-note");
    element.appendChild(this.countEl);

    this.registerDisposer(this.roiFilter.changed.add(() => this.onChanged()));
    this.registerDisposer(
      this.visibility.changed.add(() => {
        if (this.visible) this.onChanged();
      }),
    );
    // Default to every group with regions: the common case is "export what I
    // just built", and an empty selection would make the tab look broken.
    for (const g of this.roiFilter.groups) {
      if (g.rois.length > 0) this.selectedGroupIds.add(g.id);
    }
    this.onChanged();
  }

  disposed() {
    this.bodyContext.dispose();
    super.disposed();
  }

  /** Ids + names + roi counts: what changes the rows, excluding live inputs. */
  private structuralSignature(): string {
    return this.roiFilter.groups
      .map((g) => `${g.id}:${g.rois.length}:${g.name}`)
      .join(",");
  }

  private onChanged(): void {
    const live = new Set(this.roiFilter.groups.map((g) => g.id));
    for (const id of this.selectedGroupIds) {
      if (!live.has(id)) this.selectedGroupIds.delete(id);
    }
    const sig = this.structuralSignature();
    if (sig !== this.groupSignature) {
      this.groupSignature = sig;
      this.rebuild();
    }
    this.updateCount();
  }

  private exportableGroups(): RoiGroup[] {
    return this.roiFilter.groups.filter((g) => g.rois.length > 0);
  }

  private selectedGroups(): RoiGroup[] {
    return this.exportableGroups().filter((g) =>
      this.selectedGroupIds.has(g.id),
    );
  }

  private sourceUrl(): string | undefined {
    return this.layer.dataSources[0]?.spec.url;
  }

  private updateCount(): void {
    const onScreen = this.layer.displayState.roiPassingSegments?.size ?? 0;
    // The exporter re-evaluates at level 0 over the whole store, so its count
    // will normally exceed this one. Saying so here stops the difference being
    // read as a bug.
    this.countEl.textContent =
      `${onScreen.toLocaleString()} streamlines pass at the level now on ` +
      `screen. The export re-evaluates at full resolution, so it will usually ` +
      `contain more.`;
  }

  private setStatus(text: string, kind: "info" | "error" | "ok" = "info") {
    this.statusEl.textContent = text;
    this.statusEl.dataset.kind = kind;
  }

  private rebuild(): void {
    this.bodyContext.dispose();
    this.bodyContext = new RefCounted();
    const el = this.bodyEl;
    el.textContent = "";

    const groups = this.exportableGroups();
    if (groups.length === 0) {
      const empty = document.createElement("div");
      empty.classList.add("neuroglancer-tract-export-empty");
      empty.textContent =
        "No groups with regions yet — build a dissection in the Filter tab.";
      el.appendChild(empty);
      return;
    }

    const list = document.createElement("div");
    list.classList.add("neuroglancer-tract-export-group-list");
    for (const group of groups) {
      const row = document.createElement("label");
      row.classList.add("neuroglancer-tract-export-group");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.checked = this.selectedGroupIds.has(group.id);
      check.addEventListener("change", () => {
        if (check.checked) this.selectedGroupIds.add(group.id);
        else this.selectedGroupIds.delete(group.id);
        this.updateButtons();
      });
      const swatch = document.createElement("span");
      swatch.classList.add("neuroglancer-tract-export-swatch");
      const [r, g, b] = group.color;
      swatch.style.backgroundColor = `rgb(${Math.round(r * 255)},${Math.round(
        g * 255,
      )},${Math.round(b * 255)})`;
      const name = document.createElement("span");
      name.classList.add("neuroglancer-tract-export-group-name");
      name.textContent = group.name;
      const count = document.createElement("span");
      count.classList.add("neuroglancer-tract-export-roi-count");
      count.textContent = `${group.rois.length} region${
        group.rois.length === 1 ? "" : "s"
      }`;
      row.append(check, swatch, name, count);
      list.appendChild(row);
    }
    el.appendChild(list);

    el.appendChild(
      labelled(
        "Format",
        makeStringSelect(FORMAT_OPTIONS, this.format, (v) => {
          this.format = v as ExportFormat;
          this.syncOutputPathExtension();
        }),
      ),
    );

    const path = document.createElement("input");
    path.type = "text";
    path.classList.add("neuroglancer-tract-export-path");
    path.placeholder = this.defaultOutputPath();
    path.value = this.outputPath;
    path.addEventListener("change", () => {
      this.outputPath = path.value.trim();
    });
    this.pathInput = path;
    el.appendChild(labelled("Write to", path));

    const hint = document.createElement("div");
    hint.classList.add("neuroglancer-tract-export-note");
    hint.textContent =
      "A path on the machine running the exporter, not in the browser.";
    el.appendChild(hint);

    // A statically served build has no viewer server, so it needs a sidecar.
    // Offer the field there rather than only reporting the absence: the writers
    // genuinely cannot run in the page (zarr drives a dedicated IO thread that
    // Pyodide cannot start), so pointing at a local exporter is the only way to
    // get a zarr-vectors store out of this build at all.
    const nativeServed = /\/v\/[^/]+/.test(window.location.pathname);
    if (!nativeServed) {
      const exporter = document.createElement("input");
      exporter.type = "text";
      exporter.classList.add("neuroglancer-tract-export-path");
      exporter.placeholder = "http://127.0.0.1:9944/export/tract/<token>";
      exporter.value = sessionExporterUrl;
      exporter.addEventListener("change", () => {
        sessionExporterUrl = exporter.value.trim();
        // A freshly supplied URL deserves a fresh attempt.
        this.endpointUnavailable = false;
        this.rebuild();
      });
      el.appendChild(labelled("Exporter URL", exporter));

      const how = document.createElement("div");
      how.classList.add("neuroglancer-tract-export-note");
      how.textContent =
        "This build cannot write the files itself. Start one with: " +
        "python -m neuroglancer.tract_export --serve — then paste the URL it " +
        "prints above.";
      el.appendChild(how);
    }

    const buttons = document.createElement("div");
    buttons.classList.add("neuroglancer-tract-export-buttons");
    const hasExporter = exportEndpoint() !== undefined;
    if (!hasExporter) this.endpointUnavailable = true;

    const download = makeIcon({
      text: "Download job spec",
      title:
        "Save the export description as JSON, to run with " +
        "`python -m neuroglancer.tract_export`",
      onClick: () => this.downloadSpec(),
    });
    if (hasExporter) {
      this.exportButton = makeIcon({
        text: "Export",
        title: "Run the export now via the local exporter",
        onClick: () => void this.runExport(),
      });
      buttons.appendChild(this.exportButton);
      buttons.appendChild(download);
    } else {
      this.exportButton = undefined;
      buttons.appendChild(download);
    }
    el.appendChild(buttons);

    if (!hasExporter && nativeServed) {
      const why = document.createElement("div");
      why.classList.add("neuroglancer-tract-export-note");
      why.textContent =
        "No exporter is reachable. Download the job spec and run it with " +
        "python -m neuroglancer.tract_export.";
      el.appendChild(why);
    }

    this.updateButtons();
  }

  private pathInput: HTMLInputElement | undefined;
  private exportButton: HTMLElement | undefined;

  private defaultOutputPath(): string {
    return `${DEFAULT_BASENAME}.${this.format === "trk" ? "trk" : "zvf"}`;
  }

  private syncOutputPathExtension(): void {
    if (this.pathInput !== undefined) {
      this.pathInput.placeholder = this.defaultOutputPath();
    }
    this.updateButtons();
  }

  private updateButtons(): void {
    const n = this.selectedGroups().length;
    const button = this.exportButton;
    if (button !== undefined) {
      const disabled = n === 0 || this.busy || this.endpointUnavailable;
      button.classList.toggle("neuroglancer-tract-export-disabled", disabled);
      if (this.endpointUnavailable) {
        button.title =
          "No local exporter is reachable — download the job spec and run " +
          "`python -m neuroglancer.tract_export` instead.";
      }
    }
    // Outside the button guard: with no exporter there is no Export button at
    // all, and the status line is then the only thing telling the user what
    // their selection is.
    if (this.busy) return;
    if (n === 0) {
      this.setStatus("Select at least one group to export.");
    } else {
      this.setStatus(`${n} group${n === 1 ? "" : "s"} selected.`);
    }
  }

  /**
   * The job spec: what the exporter is asked to produce.
   *
   * Groups are `groupToJson` output verbatim, the same bytes the URL hash and a
   * saved ROI-group document carry, so there is one serialisation to keep
   * correct rather than three.
   */
  private buildSpec(): any {
    return buildExportSpec({
      sourceUrl: this.sourceUrl() ?? "",
      groups: this.selectedGroups(),
      format: this.format,
      outputPath: this.outputPath || this.defaultOutputPath(),
    });
  }

  private downloadSpec(): void {
    let spec: any;
    try {
      spec = this.buildSpec();
    } catch (e) {
      this.setStatus((e as Error).message, "error");
      return;
    }
    saveBlobToFile(
      new Blob([JSON.stringify(spec, null, 2)], { type: "application/json" }),
      `${DEFAULT_BASENAME}.json`,
    );
    this.setStatus(
      "Job spec saved. Run it with: python -m neuroglancer.tract_export " +
        `${DEFAULT_BASENAME}.json`,
      "ok",
    );
  }

  private async runExport(): Promise<void> {
    if (this.busy || this.endpointUnavailable) return;
    let spec: any;
    try {
      spec = this.buildSpec();
    } catch (e) {
      this.setStatus((e as Error).message, "error");
      return;
    }
    const endpoint = exportEndpoint();
    if (endpoint === undefined) {
      this.endpointUnavailable = true;
      this.updateButtons();
      this.setStatus(
        "No local exporter is reachable from this build. Use “Download job " +
          "spec” and run it with python -m neuroglancer.tract_export.",
        "error",
      );
      return;
    }

    this.busy = true;
    this.updateButtons();
    this.setStatus(
      "Exporting… re-evaluating the dissection at full resolution.",
    );
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
      });
      if (response.status === 404) {
        // Latched: this build of the server has no export route, so every
        // later click would fail the same way.
        this.endpointUnavailable = true;
        this.setStatus(
          "This server has no export route. Download the job spec and run it " +
            "with python -m neuroglancer.tract_export.",
          "error",
        );
        return;
      }
      let body: ExportSummary & { error?: string };
      try {
        body = await response.json();
      } catch {
        this.setStatus(
          `Export failed: the server returned a non-JSON ${response.status} response.`,
          "error",
        );
        return;
      }
      if (!response.ok) {
        this.setStatus(
          body.error ?? `Export failed (${response.status}).`,
          "error",
        );
        return;
      }
      this.reportSuccess(body);
    } catch (e) {
      // A network-level failure is not proof the route is missing (the server
      // may just be restarting), so this does not latch.
      this.setStatus(
        `Could not reach the exporter: ${(e as Error).message}`,
        "error",
      );
    } finally {
      this.busy = false;
      this.updateButtons();
    }
  }

  private reportSuccess(summary: ExportSummary): void {
    if (summary.written === false) {
      this.setStatus(
        summary.message ??
          "No streamlines passed this dissection; nothing was written.",
        "error",
      );
      return;
    }
    const n = summary.streamline_count ?? 0;
    const where = summary.output_path ?? "the requested path";
    const perGroup = (summary.groups ?? [])
      .map((g) => `${g.name}: ${g.count.toLocaleString()}`)
      .join(", ");
    this.setStatus(
      `Wrote ${n.toLocaleString()} streamline${n === 1 ? "" : "s"} to ${where}` +
        (perGroup ? ` (${perGroup})` : ""),
      "ok",
    );
  }
}
