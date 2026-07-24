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

import { describe, expect, it } from "vitest";
import {
  buildExportSpec,
  formatAffineText,
  parseAffineText,
} from "#src/datasource/zarr-vectors/export_job.js";
import { RoiOperator, RoiPredicate } from "#src/datasource/zarr-vectors/roi.js";
import { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";
import { vec3 } from "#src/util/geom.js";

function stateWithGroup(name: string) {
  const s = new RoiFilterState();
  const g = s.addGroup();
  s.updateGroup(g, { name, color: vec3.fromValues(1, 0, 0) });
  s.addRoi(g, {
    shape: {
      kind: "ellipsoid",
      center: Float32Array.from([10, 20, 30]),
      radii: Float32Array.from([5, 5, 5]),
    },
    predicate: RoiPredicate.ANY_SEGMENT,
    operator: RoiOperator.AND,
    name: "seed",
  });
  s.addRoi(g, {
    shape: {
      kind: "box",
      lower: Float32Array.from([0, 0, 0]),
      upper: Float32Array.from([100, 100, 100]),
    },
    predicate: RoiPredicate.EITHER_ENDPOINT,
    operator: RoiOperator.ANDNOT,
  });
  return s;
}

describe("buildExportSpec", () => {
  /**
   * The cross-language contract. This exact object is also the `VALID` fixture
   * in `python/tests/tract_export_job_test.py`; if either side drifts, one of
   * the two suites fails rather than the feature silently breaking at runtime.
   */
  it("produces the spec the Python exporter parses", () => {
    const s = stateWithGroup("Motor CST");
    const spec = buildExportSpec({
      sourceUrl: "zarr-vectors://gs://bucket/tracts.zvf",
      groups: s.groups,
      // The viewer's on-screen passing ids, as decimal strings. The large id is
      // > 2**53, pinning the string round-trip.
      objectIdsByGroup: [["7", "42", "9007199254740993"]],
      format: "trk",
      outputPath: "out.trk",
    });
    expect(spec).toEqual({
      schemaVersion: 3,
      source: { url: "zarr-vectors://gs://bucket/tracts.zvf", level: 0 },
      groups: [
        {
          name: "Motor CST",
          color: "#ff0000",
          rois: [
            {
              shape: {
                type: "ellipsoid",
                center: [10, 20, 30],
                radii: [5, 5, 5],
              },
              predicate: "any_segment",
              operator: "and",
              name: "seed",
            },
            {
              shape: {
                type: "box",
                lower: [0, 0, 0],
                upper: [100, 100, 100],
              },
              predicate: "either_endpoint",
              operator: "andnot",
            },
          ],
          objectIds: ["7", "42", "9007199254740993"],
        },
      ],
      format: "trk",
      destination: { kind: "local", path: "out.trk" },
    });
  });

  it("omits scope for a selected export (default), matching a v1-shaped spec", () => {
    const s = stateWithGroup("g");
    const spec = buildExportSpec({
      sourceUrl: "zarr-vectors://x",
      groups: s.groups,
      objectIdsByGroup: [["1"]],
      format: "trk",
      outputPath: "o.trk",
    });
    expect("scope" in spec).toBe(false);
  });

  it("appends each group's object ids as decimal strings", () => {
    const s = stateWithGroup("g");
    const spec = buildExportSpec({
      sourceUrl: "zarr-vectors://x",
      groups: s.groups,
      objectIdsByGroup: [["3", "5", "9007199254740993"]],
      format: "trk",
      outputPath: "o.trk",
    });
    expect(spec.groups[0].objectIds).toEqual(["3", "5", "9007199254740993"]);
  });

  it("rejects a selected export whose ids were never computed", () => {
    // A programmer error: the tab must compute the on-screen ids first.
    const s = stateWithGroup("g");
    expect(() =>
      buildExportSpec({
        sourceUrl: "zarr-vectors://x",
        groups: s.groups,
        format: "trk",
        outputPath: "o.trk",
      }),
    ).toThrow(/object ids were not computed/);
  });

  it("rejects a selected export when nothing currently passes", () => {
    // Empty ids across every group means the on-screen dissection selects
    // nothing; refuse before a round trip that would write no file.
    const s = stateWithGroup("g");
    expect(() =>
      buildExportSpec({
        sourceUrl: "zarr-vectors://x",
        groups: s.groups,
        objectIdsByGroup: [[]],
        format: "trk",
        outputPath: "o.trk",
      }),
    ).toThrow(/No streamlines currently pass/);
  });

  it("exports the whole store: scope='whole', empty groups, no group checks", () => {
    // No groups at all -- a whole-store export folds nothing, so the
    // "select at least one group" guard must not fire.
    const spec = buildExportSpec({
      sourceUrl: "zarr-vectors://x",
      groups: [],
      format: "zvf",
      outputPath: "whole.zvf",
      scope: "whole",
    });
    expect(spec.scope).toBe("whole");
    expect(spec.groups).toEqual([]);
    expect(spec.format).toBe("zvf");
  });

  it("always exports at level 0, never the level on screen", () => {
    const s = stateWithGroup("g");
    const spec = buildExportSpec({
      sourceUrl: "zarr-vectors://x",
      groups: s.groups,
      objectIdsByGroup: [["1"]],
      format: "zvf",
      outputPath: "out.zvf",
    });
    expect(spec.source.level).toBe(0);
  });

  it("carries the persistence encoding, not the worker wire encoding", () => {
    // `roi_filter_service.ts` emits {kind, numeric predicate/operator} for the
    // worker; sending that shape would fail `parse_job`.
    const s = stateWithGroup("g");
    const roi = buildExportSpec({
      sourceUrl: "zarr-vectors://x",
      groups: s.groups,
      objectIdsByGroup: [["1"]],
      format: "trk",
      outputPath: "o.trk",
    }).groups[0].rois[0];
    expect(roi.shape).toHaveProperty("type");
    expect(roi.shape).not.toHaveProperty("kind");
    expect(typeof roi.predicate).toBe("string");
  });

  it("rejects an unresolved data source", () => {
    const s = stateWithGroup("g");
    expect(() =>
      buildExportSpec({
        sourceUrl: "",
        groups: s.groups,
        format: "trk",
        outputPath: "o.trk",
      }),
    ).toThrow(/no resolved data source/);
  });

  it("rejects an empty selection", () => {
    expect(() =>
      buildExportSpec({
        sourceUrl: "zarr-vectors://x",
        groups: [],
        format: "trk",
        outputPath: "o.trk",
      }),
    ).toThrow(/at least one group/);
  });

  it("emits a non-identity affine but omits identity and absent", () => {
    const s = stateWithGroup("g");
    const base = {
      sourceUrl: "zarr-vectors://x",
      groups: s.groups,
      objectIdsByGroup: [["1"]],
      format: "trk" as const,
      outputPath: "o.trk",
    };
    // Absent: no affine key (keeps the golden fixture unchanged).
    expect("affine" in buildExportSpec(base)).toBe(false);
    // Identity: also omitted, since it is the exporter's default.
    const identity = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];
    expect("affine" in buildExportSpec({ ...base, affine: identity })).toBe(
      false,
    );
    // A real transform is carried through.
    const affine = [
      [0.5, 0, 0, 1],
      [0, 0.5, 0, 2],
      [0, 0, 0.5, 3],
      [0, 0, 0, 1],
    ];
    expect(buildExportSpec({ ...base, affine }).affine).toEqual(affine);
  });
});

describe("parseAffineText", () => {
  it("returns undefined for blank input", () => {
    expect(parseAffineText("")).toBeUndefined();
    expect(parseAffineText("   \n ")).toBeUndefined();
  });

  it("parses a JSON matrix", () => {
    expect(
      parseAffineText("[[1,0,0,4],[0,1,0,5],[0,0,1,6],[0,0,0,1]]"),
    ).toEqual([
      [1, 0, 0, 4],
      [0, 1, 0, 5],
      [0, 0, 1, 6],
      [0, 0, 0, 1],
    ]);
  });

  it("parses 16 whitespace-separated numbers", () => {
    const m = parseAffineText("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1");
    expect(m).toEqual([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]);
  });

  it("round-trips through formatAffineText", () => {
    const m = [
      [2, 0, 0, 0],
      [0, 3, 0, 0],
      [0, 0, 4, 0],
      [0, 0, 0, 1],
    ];
    expect(parseAffineText(formatAffineText(m))).toEqual(m);
  });

  it("rejects the wrong number of values", () => {
    expect(() => parseAffineText("1 2 3")).toThrow(/16 finite numbers/);
    expect(() => parseAffineText("[[1,0],[0,1]]")).toThrow(/16 finite numbers/);
  });

  it("rejects non-finite values", () => {
    expect(() => parseAffineText("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 x")).toThrow(
      /16 finite numbers/,
    );
  });
});
