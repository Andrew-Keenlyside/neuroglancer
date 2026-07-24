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
 * The export job spec: the payload the Export tab hands to the native exporter.
 *
 * Kept apart from the tab and free of DOM so the shape can be tested directly.
 * It is one half of a cross-language contract -- `parse_job` in
 * `python/neuroglancer/tract_export/job.py` is the other -- and the two are
 * pinned by a shared literal: `export_job.spec.ts` here and
 * `tract_export_job_test.py` there assert against the same JSON.
 *
 * A selected-scope group is `groupToJson` output plus an `objectIds` array: the
 * viewer's on-screen dissection has already chosen the passing streamlines, so
 * the spec hands their ids to the exporter and it reads exactly those (a WYSIWYG
 * export), instead of re-reading and re-folding the whole level. `groupToJson`
 * stays untouched so the URL hash and a saved ROI-group document still carry
 * byte-identical group JSON; `objectIds` is appended only here, never persisted.
 * The `rois` are carried for provenance but are not folded when `objectIds` is
 * present.
 *
 * The `groups[].rois` are the *persistence* encoding (`{type: "box"}`, string
 * predicate/operator), not the worker wire encoding (`{kind: "box"}`, numeric)
 * -- see the two unrelated `shapeToJson` pairs in `roi_filter_state.ts` and
 * `roi_filter_service.ts`.
 *
 * Object ids are decimal *strings*, not numbers: an id is a uint64 and can
 * exceed JavaScript's safe-integer range, where `JSON.stringify` would round it.
 */

import {
  groupToJson,
  type RoiGroup,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";

/** Must match `JOB_SCHEMA_VERSION` in `tract_export/job.py`. */
export const JOB_SCHEMA_VERSION = 3;

export type ExportFormat = "trk" | "zvf";

/**
 * `"selected"` exports the dissection (the chosen groups' passing set);
 * `"whole"` exports every object at the level, ignoring the ROI fold -- a
 * straight duplication of the store's geometry.
 */
export type ExportScope = "selected" | "whole";

export interface ExportSpecOptions {
  /** The layer's resolved data source URL, e.g. `zarr-vectors://gs://b/x.zvf`. */
  sourceUrl: string;
  groups: readonly RoiGroup[];
  /**
   * For a `"selected"` export: each group's passing object ids as decimal
   * strings, in the same order and length as `groups` -- the viewer's on-screen
   * dissection. Required for a selected export (the ids ARE the selection);
   * ignored for a `"whole"` export. Strings, not numbers: an id is a uint64 and
   * can exceed JS's safe-integer range.
   */
  objectIdsByGroup?: readonly (readonly string[])[];
  format: ExportFormat;
  /**
   * Pyramid level to export. Defaults to 0 (full resolution). The native
   * exporter re-evaluates at whatever level is given; the in-browser exporter
   * uses a coarser level to fit browser memory.
   */
  level?: number;
  /** Defaults to `"selected"`; `"whole"` ignores `groups`. */
  scope?: ExportScope;
  /** Where the *exporter* writes, on its own filesystem. */
  outputPath: string;
  /**
   * Voxel→RAS 4×4 to stamp into the TRK header, row-major.  Omitted means the
   * exporter defaults to identity — which for TRK means the streamlines carry
   * no world transform and downstream tools place them in voxel space.  The
   * tab pre-fills this from the layer's coordinate space, so a spec built
   * through the UI carries a real transform rather than silently defaulting.
   */
  affine?: readonly (readonly number[])[];
}

/**
 * Parse an affine typed into the Export tab into a 4×4, or undefined if blank.
 *
 * Accepts a JSON matrix (`[[1,0,0,0],...]`), a flat JSON array of 16, or 16
 * whitespace/comma-separated numbers.  Throws a user-facing message on anything
 * that is not exactly 16 finite numbers.  Pure, so it is unit-tested directly.
 */
export function parseAffineText(
  text: string,
): readonly (readonly number[])[] | undefined {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  let nums: number[];
  try {
    const parsed = JSON.parse(trimmed);
    nums = (Array.isArray(parsed) ? parsed.flat(Infinity) : [parsed]).map(
      Number,
    );
  } catch {
    nums = trimmed.split(/[\s,]+/).map(Number);
  }
  if (nums.length !== 16 || nums.some((n) => !Number.isFinite(n))) {
    throw new Error("Affine must be 16 finite numbers (a 4×4 matrix).");
  }
  return [
    nums.slice(0, 4),
    nums.slice(4, 8),
    nums.slice(8, 12),
    nums.slice(12, 16),
  ];
}

/** Format a 4×4 as four lines, for pre-filling the Export tab's affine field. */
export function formatAffineText(m: readonly (readonly number[])[]): string {
  return m.map((row) => row.join(" ")).join("\n");
}

/** Whether a 4×4 is exactly the identity, so an identity affine is not emitted. */
function isIdentity4x4(m: readonly (readonly number[])[]): boolean {
  for (let r = 0; r < 4; ++r) {
    for (let c = 0; c < 4; ++c) {
      if (m[r][c] !== (r === c ? 1 : 0)) return false;
    }
  }
  return true;
}

export function buildExportSpec(options: ExportSpecOptions): any {
  const {
    sourceUrl,
    groups,
    objectIdsByGroup,
    format,
    outputPath,
    affine,
    scope,
    level,
  } = options;
  const exportScope: ExportScope = scope ?? "selected";
  const exportLevel = level ?? 0;
  if (sourceUrl === "") {
    throw new Error("This layer has no resolved data source.");
  }
  // A whole-store export folds no regions, so it needs no groups; a selected
  // export needs the viewer's chosen object ids -- one id list per group -- to
  // have anything to write.
  if (exportScope === "selected") {
    if (groups.length === 0) {
      throw new Error("Select at least one group to export.");
    }
    if (
      objectIdsByGroup === undefined ||
      objectIdsByGroup.length !== groups.length
    ) {
      // A programmer error, not user input: the tab must compute the on-screen
      // passing ids for every selected group before building the spec.
      throw new Error(
        "Export failed: object ids were not computed for every group.",
      );
    }
    // Empty across the board means nothing currently passes the dissection.
    // Refuse here rather than after a round trip that would write no file.
    const total = objectIdsByGroup.reduce((n, ids) => n + ids.length, 0);
    if (total === 0) {
      throw new Error(
        "No streamlines currently pass this dissection — adjust the filter or " +
          "load more of the region, then export.",
      );
    }
  }
  const spec: any = {
    schemaVersion: JOB_SCHEMA_VERSION,
    // The export is of the *data*, not of what is on screen; `level` defaults to
    // 0 (full resolution) and is only raised to fit an in-browser export.
    source: { url: sourceUrl, level: exportLevel },
    // Selected groups carry their on-screen passing ids (the selection) plus the
    // persistence group JSON (rois/colour, for provenance -- not re-folded).
    groups:
      exportScope === "whole"
        ? []
        : groups.map((g, i) => ({
            ...groupToJson(g),
            objectIds: [...(objectIdsByGroup![i] ?? [])],
          })),
    format,
    destination: { kind: "local", path: outputPath },
  };
  // Emit `scope` only when it is "whole": the default matches a v1 spec's
  // shape, so a selected export stays byte-identical to the golden fixture and
  // `parse_job` reads absent scope as "selected".
  if (exportScope === "whole") {
    spec.scope = "whole";
  }
  // Only emit a non-identity affine: identity is the exporter's default, and
  // omitting it keeps the golden-fixture spec (which has no affine) unchanged.
  if (affine !== undefined && !isIdentity4x4(affine)) {
    spec.affine = affine.map((row) => [...row]);
  }
  return spec;
}
