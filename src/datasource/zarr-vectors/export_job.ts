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
 * Groups are `groupToJson` output verbatim. That is deliberate: the spec, the
 * URL hash and a saved ROI-group document then carry byte-identical group JSON,
 * so there is a single serialisation to keep correct rather than three. Note it
 * is the *persistence* encoding (`{type: "box"}`, string predicate/operator),
 * not the worker wire encoding (`{kind: "box"}`, numeric) -- see the two
 * unrelated `shapeToJson` pairs in `roi_filter_state.ts` and
 * `roi_filter_service.ts`.
 */

import {
  groupToJson,
  type RoiGroup,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";

/** Must match `JOB_SCHEMA_VERSION` in `tract_export/job.py`. */
export const JOB_SCHEMA_VERSION = 1;

export type ExportFormat = "trk" | "zvf";

export interface ExportSpecOptions {
  /** The layer's resolved data source URL, e.g. `zarr-vectors://gs://b/x.zvf`. */
  sourceUrl: string;
  groups: readonly RoiGroup[];
  format: ExportFormat;
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
  const { sourceUrl, groups, format, outputPath, affine } = options;
  if (sourceUrl === "") {
    throw new Error("This layer has no resolved data source.");
  }
  if (groups.length === 0) {
    throw new Error("Select at least one group to export.");
  }
  // A group with no regions folds to "everything", which as an export would
  // silently write the whole dataset under that group's name. `parse_job`
  // rejects it too; catching it here means the message arrives without a
  // round trip.
  const empty = groups.find((g) => g.rois.length === 0);
  if (empty !== undefined) {
    throw new Error(
      `Group “${empty.name}” has no regions, which would select every ` +
        `streamline in the dataset.`,
    );
  }
  const spec: any = {
    schemaVersion: JOB_SCHEMA_VERSION,
    // Level 0 always: the export is of the data, not of what is on screen.
    source: { url: sourceUrl, level: 0 },
    groups: groups.map(groupToJson),
    format,
    destination: { kind: "local", path: outputPath },
  };
  // Only emit a non-identity affine: identity is the exporter's default, and
  // omitting it keeps the golden-fixture spec (which has no affine) unchanged.
  if (affine !== undefined && !isIdentity4x4(affine)) {
    spec.affine = affine.map((row) => [...row]);
  }
  return spec;
}
