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
 * @file The "Export" tab: write the dissection (or the whole store) out as a
 * TrackVis `.trk` file or a new zarr-vectors store, downloaded to the browser
 * or uploaded to GCS.
 *
 * Three server contexts share one job spec:
 *  - the **Pyodide build** (`/v/pyodide/`) runs the exporter in-browser (see
 *    `python/neuroglancer/tract_export/browser.py`) and returns the file
 *    *bytes*, which this tab downloads or uploads;
 *  - a **native `python -m neuroglancer` server** writes the file on its own
 *    filesystem and returns a JSON summary;
 *  - a **static build with no exporter** offers the spec as a download, to run
 *    with `python -m neuroglancer.tract_export`.
 * The response's Content-Type tells them apart: binary is a file to save, JSON
 * is a server-side summary.
 *
 * The spec carries each selected group's on-screen passing object ids -- the
 * viewer's dissection has already chosen the streamlines, so the exporter reads
 * exactly those objects (a WYSIWYG export) instead of re-reading and re-folding
 * the whole level. The ids are computed on demand by the tract render layer's
 * worker over the currently-resident chunks (`computeRoiExportIds`).
 */

import "#src/datasource/zarr-vectors/tract_export_tab.css";

import { ChunkState } from "#src/chunk_manager/base.js";
import {
  buildExportSpec,
  formatAffineText,
  parseAffineText,
  type ExportFormat,
  type ExportScope,
} from "#src/datasource/zarr-vectors/export_job.js";
import {
  labelled,
  makeStringSelect,
} from "#src/datasource/zarr-vectors/filter_widgets.js";
import type { RoiGroupConfig } from "#src/datasource/zarr-vectors/roi.js";
import type {
  RoiFilterState,
  RoiGroup,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";
import { writeTrk } from "#src/datasource/zarr-vectors/trk_writer.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { roiStoreEnabled } from "#src/roi_store/config.js";
import { getRoiStoreAuth } from "#src/roi_store/credentials.js";
import { getRoiGroupStore } from "#src/roi_store/gcs_client.js";
import { getObjectKey } from "#src/segmentation_display_state/base.js";
import { RefCounted } from "#src/util/disposable.js";
import { saveBlobToFile } from "#src/util/file_download.js";
import { makeIcon } from "#src/widget/icon.js";
import { Tab } from "#src/widget/tab_view.js";

const FORMAT_OPTIONS = [
  { value: "trk", label: "TrackVis (.trk)" },
  { value: "zvf", label: "New zarr-vectors store" },
];

const SCOPE_OPTIONS = [
  { value: "selected", label: "Selected groups" },
  { value: "whole", label: "Whole store" },
];

const DEFAULT_BASENAME = "dissection";

const IDENTITY_AFFINE: readonly (readonly number[])[] = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

/**
 * Rough in-browser export memory ceiling. A level whose estimated fully-resident
 * size exceeds this is treated as infeasible in the browser -- the exporter reads
 * the whole level into memory, so `exportLevel` is auto-set to the finest level
 * under this budget, and full resolution beyond it needs the native exporter
 * (which has no such limit). Heuristic, deliberately conservative.
 */
const BROWSER_EXPORT_BUDGET_BYTES = 600_000_000;

/** Where a "Save" sends the exported bytes. */
type SaveDestination = "download" | "gcs";

/** Human-readable byte size for the "Saved …" status line. */
function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    ++i;
  }
  return `${value.toFixed(1)} ${units[i]}`;
}

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
  /** `"whole"` exports every streamline at level 0, ignoring the group list. */
  private scope: ExportScope = "selected";
  /**
   * Pyramid level the in-browser exporter reads (0 = full resolution). Not a
   * user choice: the browser reads the whole level into memory, so this is set
   * automatically to the finest level that fits (see `snapLevelToFeasibleDefault`).
   * Full resolution of a store too big to hold needs the programmatic job-spec
   * route (`python -m neuroglancer.tract_export`), which has no such limit.
   */
  private exportLevel = 0;
  /**
   * Voxel→RAS affine typed by the user, for the TRK header. Blank means
   * identity (the exporter's default), so nothing is emitted and the streamline
   * export lands in voxel space -- which is why the field is pre-filled from the
   * layer's scales rather than left empty.
   */
  private affineText = "";
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
    // Pyramid levels (and their cost estimates) load asynchronously; rebuild so
    // the level picker appears/refreshes once they are known.
    this.registerDisposer(
      this.layer.displayState.spatialSkeletonGridLevels.changed.add(() => {
        if (this.visible) this.rebuild();
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
    // The export is WYSIWYG: it writes exactly the streamlines the on-screen
    // dissection selects, not a fresh level-0 re-evaluation. So this on-screen
    // count is what the export reflects -- load more of the region (or a finer
    // level) to select more.
    this.countEl.textContent =
      `${onScreen.toLocaleString()} streamlines currently pass the visible ` +
      `dissection. The export writes exactly the streamlines selected at the ` +
      `current view — load more of the region for a fuller export.`;
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

    // Scope first: a whole-store export needs no groups, so the group list is
    // only shown (and only required) for a selected export.
    el.appendChild(
      labelled(
        "Scope",
        makeStringSelect(SCOPE_OPTIONS, this.scope, (v) => {
          this.scope = v as ExportScope;
          this.rebuild();
        }),
      ),
    );

    if (this.scope === "whole") {
      const note = document.createElement("div");
      note.classList.add("neuroglancer-tract-export-note");
      note.textContent =
        "Exports every streamline in the store, evaluated at level 0.";
      el.appendChild(note);
    } else {
      const groups = this.exportableGroups();
      if (groups.length === 0) {
        const empty = document.createElement("div");
        empty.classList.add("neuroglancer-tract-export-empty");
        empty.textContent =
          "No groups with regions yet — build a dissection in the Filter tab, " +
          "or switch Scope to “Whole store”.";
        el.appendChild(empty);
      } else {
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
      }
    }

    el.appendChild(
      labelled(
        "Format",
        makeStringSelect(FORMAT_OPTIONS, this.format, (v) => {
          this.format = v as ExportFormat;
          // Rebuild so the affine field (TRK only) and the default download
          // name follow the format.
          this.rebuild();
        }),
      ),
    );

    // The resolution-level picker lives next to the Download button (below):
    // the level governs what a *browser download* can afford, and is separate
    // from Save to GCS.

    // Affine only applies to TRK (ZVF stores coordinates directly).
    if (this.format === "trk") {
      if (this.affineText === "") this.affineText = this.defaultAffineText();
      const affine = document.createElement("textarea");
      affine.classList.add("neuroglancer-tract-export-affine");
      affine.rows = 4;
      affine.spellcheck = false;
      affine.value = this.affineText;
      affine.addEventListener("change", () => {
        this.affineText = affine.value;
      });
      el.appendChild(labelled("Voxel→RAS", affine));

      const affineNote = document.createElement("div");
      affineNote.classList.add("neuroglancer-tract-export-note");
      affineNote.textContent =
        "Voxel→RAS 4×4 (in millimetres) written into the TRK header. Pre-filled " +
        "from the layer's scales; edit for orientation, or clear for a 1 mm " +
        "identity.";
      el.appendChild(affineNote);
    }

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
    // A selected-scope TRK export is written entirely in-browser from the loaded
    // (high-detail) geometry -- no exporter endpoint needed. Everything else
    // (ZVF, or a whole-store export) still goes through the exporter.
    const fastTrk = this.fastTrkEligible();
    const canRun = hasExporter || fastTrk;

    const downloadSpecButton = makeIcon({
      text: "Download job spec",
      title:
        "Save the export description as JSON, to run with " +
        "`python -m neuroglancer.tract_export`",
      onClick: () => void this.downloadSpec(),
    });
    if (canRun) {
      this.downloadButton = makeIcon({
        text: "Download",
        title: fastTrk
          ? "Write the loaded tracts to a .trk in-browser and download it"
          : "Run the export and download the file to this browser",
        onClick: () => void this.runExport("download"),
      });
      // No level picker: the export is WYSIWYG. The fast TRK path uses the loaded
      // geometry directly; the endpoint path silently reads the finest level the
      // browser can hold. The note below explains the full-resolution route.
      this.snapLevelToFeasibleDefault(this.levelChoices());
      buttons.appendChild(this.downloadButton);
      // GCS upload only where a store is configured; it prompts for sign-in
      // only on click, never on render.
      if (roiStoreEnabled) {
        this.gcsButton = makeIcon({
          text: "Save to GCS",
          title:
            "Run the export and upload the file to the shared store " +
            "(sign-in required)",
          onClick: () => void this.runExport("gcs"),
        });
        buttons.appendChild(this.gcsButton);
      } else {
        this.gcsButton = undefined;
      }
      buttons.appendChild(downloadSpecButton);
    } else {
      this.downloadButton = undefined;
      this.gcsButton = undefined;
      buttons.appendChild(downloadSpecButton);
    }
    el.appendChild(buttons);

    // Explain the Download behaviour and how to get the full dissection at full
    // resolution. The fast TRK path exports the loaded whole tracts; the endpoint
    // path re-reads the store at a browser-affordable level.
    if (canRun) {
      const note = document.createElement("div");
      note.classList.add("neuroglancer-tract-export-note");
      note.textContent = fastTrk
        ? this.fastTrkNoteText()
        : this.levelNoteText();
      el.appendChild(note);
    }

    // Make the absence of "Save to GCS" legible rather than silent: if the
    // button is missing, this says whether it is because no store is configured
    // in this build (rebuild with --define ROI_STORE=...) -- a quick way to tell
    // a stale bundle from a genuinely unconfigured one.
    if (canRun && !roiStoreEnabled) {
      const note = document.createElement("div");
      note.classList.add("neuroglancer-tract-export-note");
      note.textContent =
        "Save to GCS is unavailable: no ROI store is configured in this build.";
      el.appendChild(note);
    }

    if (!canRun && nativeServed) {
      const why = document.createElement("div");
      why.classList.add("neuroglancer-tract-export-note");
      why.textContent =
        "No exporter is reachable. Download the job spec and run it with " +
        "python -m neuroglancer.tract_export.";
      el.appendChild(why);
    }

    this.updateButtons();
  }

  private downloadButton: HTMLElement | undefined;
  private gcsButton: HTMLElement | undefined;

  /** Auto-generated download name; the user no longer types a path. */
  private defaultOutputPath(): string {
    return `${DEFAULT_BASENAME}.${this.format === "trk" ? "trk" : "zvf"}`;
  }

  /**
   * Per-level export choices, finest-first (level 0 first), each flagged for
   * whether an in-browser export can afford it. Empty when the store carries no
   * per-level cost metadata -- then no picker is shown and level 0 is used.
   */
  private levelChoices(): {
    level: number;
    costBytes: number;
    feasible: boolean;
  }[] {
    return this.layer.displayState
      .roiExportLevelCostsBytes()
      .map((costBytes, level) => ({
        level,
        costBytes,
        feasible:
          Number.isFinite(costBytes) &&
          costBytes <= BROWSER_EXPORT_BUDGET_BYTES,
      }));
  }

  /**
   * Point `exportLevel` at the finest level that fits in browser memory (best
   * quality that will actually export here). If none fit, leave it as-is so the
   * note directs the user to the programmatic route.
   */
  private snapLevelToFeasibleDefault(
    levels: { level: number; feasible: boolean }[],
  ): void {
    const current = levels.find((c) => c.level === this.exportLevel);
    if (current?.feasible) return;
    const finestFeasible = levels.find((c) => c.feasible);
    if (finestFeasible !== undefined) this.exportLevel = finestFeasible.level;
  }

  /**
   * The note under the Download button: what resolution the in-browser export
   * uses, and how to get full resolution when the browser cannot hold the level.
   * The native exporter (`python -m neuroglancer.tract_export`, run against the
   * downloaded job spec) reads the store directly and has no browser memory cap.
   */
  private levelNoteText(): string {
    const jsonRoute =
      "download the job spec and run it with " +
      "`python -m neuroglancer.tract_export`";
    const choices = this.levelChoices();
    if (choices.length === 0) {
      // No per-level metadata: the exporter reads level 0.
      return (
        `Exports at full resolution. If it is too large for the browser, ` +
        `${jsonRoute} (no memory limit).`
      );
    }
    if (!choices.some((c) => c.feasible)) {
      return (
        `This store is too large to export in the browser at any resolution. ` +
        `For full resolution, ${jsonRoute} — it reads the store directly.`
      );
    }
    const level0Feasible =
      choices.find((c) => c.level === 0)?.feasible ?? false;
    if (level0Feasible) {
      return (
        `Exports at full resolution. For a very large dissection, ${jsonRoute} ` +
        `(no browser memory limit).`
      );
    }
    return (
      `Full resolution is too large to hold in the browser, so this exports a ` +
      `coarser level. For full resolution, ${jsonRoute} — it reads the store ` +
      `directly with no memory limit.`
    );
  }

  /**
   * Toggle the save buttons' enabled state. Does NOT touch the status line, so
   * an export result (or error) stays readable when this runs after a click.
   */
  private syncButtonDisabled(): void {
    const n = this.selectedGroups().length;
    // A whole-store export ignores the group list, so it is never blocked on a
    // selection; a selected export needs at least one group.
    const nothingToExport = this.scope === "selected" && n === 0;
    // The fast TRK path needs no exporter, so a missing endpoint does not block
    // it -- only the endpoint (ZVF/whole-store) path.
    const blockedOnEndpoint =
      this.endpointUnavailable && !this.fastTrkEligible();
    const disabled = nothingToExport || this.busy || blockedOnEndpoint;
    for (const button of [this.downloadButton, this.gcsButton]) {
      if (button === undefined) continue;
      button.classList.toggle("neuroglancer-tract-export-disabled", disabled);
      if (blockedOnEndpoint) {
        button.title =
          "No exporter is reachable — download the job spec and run " +
          "`python -m neuroglancer.tract_export` instead.";
      }
    }
  }

  private updateButtons(): void {
    this.syncButtonDisabled();
    // Refresh the *selection* status. Skipped while busy so it never clobbers
    // the "Exporting…" / "Saved …" / error message an export is showing --
    // that is also why runExport's `finally` calls syncButtonDisabled(), not
    // this, so the result persists.
    if (this.busy) return;
    const n = this.selectedGroups().length;
    if (this.scope === "whole") {
      this.setStatus("Whole store: every streamline at level 0.");
    } else if (n === 0) {
      this.setStatus("Select at least one group to export.");
    } else {
      this.setStatus(`${n} group${n === 1 ? "" : "s"} selected.`);
    }
  }

  /**
   * The job spec: what the exporter is asked to produce.
   *
   * For a selected export this first asks the tract render layer for each ticked
   * group's on-screen passing object ids (the WYSIWYG selection) and ships those
   * ids in the spec, so the exporter reads exactly those objects rather than
   * re-reading and re-folding the whole level. Async for that reason. The group
   * JSON is still `groupToJson` output (rois carried for provenance); the ids are
   * appended per group.
   */
  private async buildSpecAsync(): Promise<any> {
    const selected = this.selectedGroups();
    let objectIdsByGroup: string[][] | undefined;
    if (this.scope === "selected") {
      const compute = this.layer.displayState.computeRoiExportIds;
      if (compute === undefined) {
        throw new Error(
          "This layer's tract geometry is not ready yet — wait for it to " +
            "load, then export.",
        );
      }
      // Minimal configs: the per-group fold only reads `rois` and (for the
      // length filter) `lengthRange`; the other fields are required by the type
      // but unused here. visible: true so the length-attribute lookup is built.
      const configs: RoiGroupConfig[] = selected.map((g) => ({
        rois: g.rois,
        colorPacked: 0,
        visible: true,
        highDetail: false,
        colorBy: g.colorBy,
        ...(g.lengthFilter !== undefined
          ? { lengthRange: g.lengthFilter }
          : {}),
      }));
      const perGroupIds = await compute(configs);
      // uint64 ids as decimal strings so JSON keeps them exact past 2**53.
      objectIdsByGroup = perGroupIds.map((ids) =>
        ids.map((id) => id.toString()),
      );
    }
    return buildExportSpec({
      sourceUrl: this.sourceUrl() ?? "",
      groups: selected,
      objectIdsByGroup,
      format: this.format,
      scope: this.scope,
      level: this.exportLevel,
      outputPath: this.defaultOutputPath(),
      // Only TRK carries an affine header; parseAffineText throws a user-facing
      // message on malformed input, surfaced by the callers of buildSpecAsync.
      affine:
        this.format === "trk" ? parseAffineText(this.affineText) : undefined,
    });
  }

  /**
   * A diagonal voxel→RAS from the layer's coordinate-space scales, as a starting
   * point the user edits.  Orientation and any axis permutation are the user's to
   * set -- this only seeds the diagonal, and returns identity if the space cannot
   * be read.
   *
   * Neuroglancer scales are in **SI metres** (a 1 mm voxel is 0.001), but a TRK
   * header is in **millimetres** -- so the diagonal is scaled by 1000. Without
   * this the exported streamlines come out ~1000× too small (a whole brain reads
   * as a sub-millimetre speck), and the degenerate header keeps freeview from
   * placing them at all. See `_write_trk_bytes` in `tract_export/browser.py`,
   * which turns this affine into the header's voxel size / dimensions / vox→RAS.
   */
  private defaultAffineText(): string {
    const identity = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];
    try {
      const scales = this.layer.manager.root.coordinateSpace.value?.scales;
      if (scales === undefined || scales.length < 3) {
        return formatAffineText(identity);
      }
      const m = identity.map((row) => [...row]);
      // metres → millimetres for the TRK header.
      for (let i = 0; i < 3; ++i) m[i][i] = scales[i] * 1000;
      return formatAffineText(m);
    } catch {
      return formatAffineText(identity);
    }
  }

  private async downloadSpec(): Promise<void> {
    let spec: any;
    try {
      spec = await this.buildSpecAsync();
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

  /**
   * Whether the fast in-browser TRK path applies: a selected-scope TRK export of
   * a tract layer whose pass-2 (high-detail) geometry is resident. It writes the
   * loaded whole tracts straight from worker memory, needing no exporter.
   */
  private fastTrkEligible(): boolean {
    return (
      this.format === "trk" &&
      this.scope === "selected" &&
      this.layer.displayState.roiHighDetailSkeletonSource !== undefined &&
      this.layer.displayState.computeRoiExportIds !== undefined
    );
  }

  private fastTrkNoteText(): string {
    return (
      "Downloads the whole tracts currently loaded at high detail, straight " +
      "from the view — instant, no store re-read. Raise the high-detail budget " +
      "(Filter tab) to load more tracts. For the full dissection at full " +
      "resolution, use “Download job spec” and run it with " +
      "python -m neuroglancer.tract_export."
    );
  }

  /**
   * Gather whole-tract geometry for the selected groups from the pass-2 source's
   * resident chunks. Only tracts loaded at high detail (drawn) contribute; the
   * rest of the passing set is counted as `notLoaded` so the tab can say so.
   *
   * Positions come straight from each frontend `SkeletonChunk`'s retained
   * `vertexAttributes` (positions are its first `numVertices*3` floats — see
   * `serializeSkeletonChunkData`), copied into an aligned buffer. These are raw
   * store/model coordinates (the layer's model transform is identity), the same
   * space `read_polylines` returns, so the TRK affine applies unchanged.
   */
  private async collectResidentStreamlines(): Promise<{
    streamlines: Float32Array[];
    loaded: number;
    notLoaded: number;
  }> {
    const compute = this.layer.displayState.computeRoiExportIds!;
    const source = this.layer.displayState.roiHighDetailSkeletonSource!;
    const configs: RoiGroupConfig[] = this.selectedGroups().map((g) => ({
      rois: g.rois,
      colorPacked: 0,
      visible: true,
      highDetail: false,
      colorBy: g.colorBy,
      ...(g.lengthFilter !== undefined ? { lengthRange: g.lengthFilter } : {}),
    }));
    const perGroupIds = await compute(configs);
    const seen = new Set<bigint>();
    const streamlines: Float32Array[] = [];
    let loaded = 0;
    let notLoaded = 0;
    for (const ids of perGroupIds) {
      for (const id of ids) {
        if (seen.has(id)) continue; // an id may pass more than one group
        seen.add(id);
        const chunk = source.chunks.get(getObjectKey(id));
        const n = chunk?.numVertices ?? 0;
        if (
          chunk !== undefined &&
          chunk.state === ChunkState.GPU_MEMORY &&
          chunk.vertexAttributes != null &&
          n > 0
        ) {
          const bytes = new Uint8Array(n * 3 * 4);
          bytes.set(chunk.vertexAttributes.subarray(0, n * 3 * 4));
          streamlines.push(new Float32Array(bytes.buffer));
          ++loaded;
        } else {
          ++notLoaded;
        }
      }
    }
    return { streamlines, loaded, notLoaded };
  }

  /**
   * The fast in-browser TRK export: assemble the loaded whole tracts and write a
   * `.trk` with no store re-read (see {@link fastTrkEligible}). The bytes are
   * identical to the native exporter's for the same streamlines and affine.
   */
  private async runFastTrkExport(destination: SaveDestination): Promise<void> {
    if (this.selectedGroups().length === 0) {
      this.setStatus("Select at least one group to export.", "error");
      return;
    }
    let affine: readonly (readonly number[])[];
    try {
      affine = parseAffineText(this.affineText) ?? IDENTITY_AFFINE;
    } catch (e) {
      this.setStatus((e as Error).message, "error");
      return;
    }
    this.busy = true;
    this.updateButtons();
    this.setStatus("Assembling loaded tracts from the current view…");
    try {
      const { streamlines, loaded, notLoaded } =
        await this.collectResidentStreamlines();
      if (streamlines.length === 0) {
        this.setStatus(
          "No whole tracts are loaded at high detail. Raise the high-detail " +
            "budget (Filter tab) or zoom in to load them, then export — or use " +
            "“Download job spec” for the full dissection at full resolution.",
          "error",
        );
        return;
      }
      const bytes = writeTrk(streamlines, affine);
      const blob = new Blob([bytes], { type: "application/octet-stream" });
      const filename = this.defaultOutputPath();
      const note =
        notLoaded > 0
          ? ` (${notLoaded.toLocaleString()} more passed but aren’t loaded ` +
            `at high detail)`
          : "";
      if (destination === "download") {
        saveBlobToFile(blob, filename);
        this.setStatus(
          `Saved ${filename}: ${loaded.toLocaleString()} whole tract` +
            `${loaded === 1 ? "" : "s"} (${formatBytes(blob.size)})${note}.`,
          "ok",
        );
      } else {
        await this.uploadToGcs(blob, filename, "application/octet-stream");
      }
    } catch (e) {
      console.error("[tract-export] in-browser TRK export failed:", e);
      this.setStatus(
        `In-browser export failed: ${(e as Error).message}`,
        "error",
      );
    } finally {
      this.busy = false;
      this.syncButtonDisabled();
    }
  }

  private async runExport(destination: SaveDestination): Promise<void> {
    if (this.busy) return;
    // TRK from a selected dissection is written in-browser from the loaded
    // geometry -- no store re-read, no exporter endpoint.
    if (this.fastTrkEligible()) {
      await this.runFastTrkExport(destination);
      return;
    }
    if (this.endpointUnavailable) return;
    const endpoint = exportEndpoint();
    if (endpoint === undefined) {
      this.endpointUnavailable = true;
      this.updateButtons();
      this.setStatus(
        "No exporter is reachable from this build. Use “Download job spec” " +
          "and run it with python -m neuroglancer.tract_export.",
        "error",
      );
      return;
    }

    this.busy = true;
    this.updateButtons();
    // Build the spec first: for a selected export this asks the worker for the
    // on-screen passing ids, which is where the selection is made.
    this.setStatus("Selecting streamlines from the current view…");
    let spec: any;
    try {
      spec = await this.buildSpecAsync();
    } catch (e) {
      this.busy = false;
      this.syncButtonDisabled();
      this.setStatus((e as Error).message, "error");
      return;
    }

    this.setStatus("Exporting the selected streamlines…");
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
      // Content-Type distinguishes the two success shapes: a JSON summary means
      // a native server wrote the file on its own filesystem (or the selection
      // was empty); a binary body means the in-browser exporter returned the
      // file itself, to download or upload here.
      const contentType = response.headers.get("Content-Type") ?? "";
      if (contentType.includes("application/json")) {
        await this.handleJsonResponse(response, destination);
        return;
      }
      if (!response.ok) {
        const text = await response.text().catch(() => "");
        console.error(
          `[tract-export] ${response.status} from ${endpoint}:`,
          text,
        );
        this.setStatus(text || `Export failed (${response.status}).`, "error");
        return;
      }
      const blob = await response.blob();
      if (blob.size === 0) {
        console.error("[tract-export] empty response body from", endpoint);
        this.setStatus(
          "Export returned no data — see the browser console for details.",
          "error",
        );
        return;
      }
      // A .zvf is delivered as a zip; make the download name say so.
      const base = this.defaultOutputPath();
      const filename =
        contentType.includes("zip") && !base.endsWith(".zip")
          ? `${base}.zip`
          : base;
      if (destination === "download") {
        saveBlobToFile(blob, filename);
        this.setStatus(`Saved ${filename} (${formatBytes(blob.size)}).`, "ok");
      } else {
        await this.uploadToGcs(blob, filename, contentType);
      }
    } catch (e) {
      // A network-level failure is not proof the route is missing (the server
      // may just be restarting), so this does not latch.
      console.error("[tract-export] request failed:", e);
      this.setStatus(
        `Could not reach the exporter: ${(e as Error).message}`,
        "error",
      );
    } finally {
      this.busy = false;
      // Only refresh button state -- NOT updateButtons(), which would overwrite
      // the export result/error message we just set.
      this.syncButtonDisabled();
    }
  }

  /**
   * A JSON export response: a native-server summary, or an empty selection
   * (`written: false`). There are no file bytes, so GCS upload cannot proceed.
   */
  private async handleJsonResponse(
    response: Response,
    destination: SaveDestination,
  ): Promise<void> {
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
    if (body.written === false) {
      this.setStatus(
        body.message ??
          "No streamlines passed this dissection; nothing was written.",
        "error",
      );
      return;
    }
    if (destination === "gcs") {
      this.setStatus(
        "This exporter writes to its own filesystem, so there is nothing to " +
          "upload here. “Save to GCS” needs the in-browser (Pyodide) build.",
        "error",
      );
      return;
    }
    this.reportSuccess(body);
  }

  private async uploadToGcs(
    blob: Blob,
    filename: string,
    contentType: string,
  ): Promise<void> {
    this.setStatus("Uploading to the shared store…");
    try {
      // Sign-in (middleauth) is prompted here, on the GCS path only. The token
      // is obtained up front so a refusal is reported before the upload.
      await getRoiStoreAuth().getAccessToken();
      const name = `exports/${filename}`;
      await getRoiGroupStore().putObject(name, blob, contentType);
      this.setStatus(`Uploaded to the store as ${name}.`, "ok");
    } catch (e) {
      this.setStatus(
        `Upload to the store failed: ${(e as Error).message}`,
        "error",
      );
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
