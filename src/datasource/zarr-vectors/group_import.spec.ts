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

import { describe, expect, it } from "vitest";
import { parseGroupsJson } from "#src/datasource/zarr-vectors/group_import.js";
import {
  groupToJson,
  RoiFilterState,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";
import { makeRoiGroupDocument } from "#src/roi_store/schema.js";

/** A bare group in `groupToJson` form, built the way the app builds one. */
function bareGroup(name = "Motor CST"): any {
  const s = new RoiFilterState();
  const g = s.addGroup();
  s.updateGroup(g, { name });
  s.addRoi(g, {
    shape: {
      kind: "ellipsoid",
      center: Float32Array.from([1, 2, 3]),
      radii: Float32Array.from([4, 4, 4]),
    },
    predicate: 0 /* ANY_SEGMENT */,
    operator: 0 /* AND */,
    name: "seed",
  });
  return groupToJson(s.groups[0]);
}

describe("parseGroupsJson — accepted shapes", () => {
  it("accepts a bare group", () => {
    const groups = parseGroupsJson(JSON.stringify(bareGroup()));
    expect(groups).toHaveLength(1);
    expect(groups[0].name).toBe("Motor CST");
    expect(groups[0].rois).toHaveLength(1);
    // Stage 2's ROI name survives the import path too.
    expect(groups[0].rois[0].name).toBe("seed");
  });

  it("accepts an array of groups", () => {
    const text = JSON.stringify([bareGroup("A"), bareGroup("B")]);
    expect(parseGroupsJson(text).map((g) => g.name)).toEqual(["A", "B"]);
  });

  it("accepts a saved group document", () => {
    const doc = makeRoiGroupDocument({
      group: bareGroup("From store"),
      source: { url: "zarr-vectors://gs://bucket/tracts.zvf" },
    });
    const groups = parseGroupsJson(JSON.stringify(doc));
    expect(groups).toHaveLength(1);
    expect(groups[0].name).toBe("From store");
  });

  it("round-trips through the state that will receive it", () => {
    const s = new RoiFilterState();
    for (const g of parseGroupsJson(JSON.stringify(bareGroup("RT")))) {
      s.insertGroup(g);
    }
    expect(s.groups).toHaveLength(1);
    expect(groupToJson(s.groups[0])).toEqual(bareGroup("RT"));
  });

  it("does not carry a placeholder id onto the imported group", () => {
    const [group] = parseGroupsJson(JSON.stringify(bareGroup()));
    expect("id" in group).toBe(false);
  });
});

describe("parseGroupsJson — rejected input", () => {
  it("rejects empty input", () => {
    expect(() => parseGroupsJson("   ")).toThrow(/empty/i);
  });

  it("rejects an empty array", () => {
    expect(() => parseGroupsJson("[]")).toThrow(/no groups/i);
  });

  it("rejects malformed JSON with the parser's own message", () => {
    expect(() => parseGroupsJson("{not json")).toThrow(/Not valid JSON/);
  });

  it("names the offending element when an array entry is bad", () => {
    const text = JSON.stringify([bareGroup("ok"), { name: "no rois" }]);
    expect(() => parseGroupsJson(text)).toThrow(/^Group 2: /);
  });

  it("reports a document error as a document, not as a bare group", () => {
    // Has `group`, so it is a document -- and so the complaint should be about
    // schemaVersion, not about a missing `rois`.
    const text = JSON.stringify({ group: bareGroup() });
    expect(() => parseGroupsJson(text)).toThrow(/schemaVersion/);
  });

  it("rejects a document from a newer schema version", () => {
    const doc = makeRoiGroupDocument({
      group: bareGroup(),
      source: { url: "zarr-vectors://gs://bucket/tracts.zvf" },
    });
    const text = JSON.stringify({
      ...doc,
      schemaVersion: doc.schemaVersion + 1,
    });
    expect(() => parseGroupsJson(text)).toThrow(/newer than this viewer/);
  });
});
