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
}

export function buildExportSpec(options: ExportSpecOptions): any {
  const { sourceUrl, groups, format, outputPath } = options;
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
  return {
    schemaVersion: JOB_SCHEMA_VERSION,
    // Level 0 always: the export is of the data, not of what is on screen.
    source: { url: sourceUrl, level: 0 },
    groups: groups.map(groupToJson),
    format,
    destination: { kind: "local", path: outputPath },
  };
}
