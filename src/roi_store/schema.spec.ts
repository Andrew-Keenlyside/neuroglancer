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
 * @file The saved ROI group document format.
 */

import { describe, it, expect } from "vitest";
import {
  makeRoiGroupDocument,
  parseRoiGroupDocument,
  roiGroupCustomMetadata,
  roiGroupIdFromObjectName,
  roiGroupObjectName,
  ROI_GROUP_SCHEMA_VERSION,
} from "#src/roi_store/schema.js";

const GROUP = { name: "Arcuate L", color: "#ff3b30", rois: [] };
const SOURCE = { url: "zarr-vectors://https://example.com/tracts.zarr" };

describe("makeRoiGroupDocument", () => {
  it("mints an id and timestamps for a new document", () => {
    const doc = makeRoiGroupDocument({ group: GROUP, source: SOURCE });
    expect(doc.schemaVersion).toEqual(ROI_GROUP_SCHEMA_VERSION);
    expect(doc.id.length).toBeGreaterThan(0);
    expect(doc.createdAt).toEqual(doc.updatedAt);
  });

  it("mints a distinct id each time", () => {
    const a = makeRoiGroupDocument({ group: GROUP, source: SOURCE });
    const b = makeRoiGroupDocument({ group: GROUP, source: SOURCE });
    expect(a.id).not.toEqual(b.id);
  });

  it("preserves id and createdAt when updating", () => {
    // Re-saving a group must update its document in place, not reset when it
    // was first created.
    const original = makeRoiGroupDocument({ group: GROUP, source: SOURCE });
    const updated = makeRoiGroupDocument({
      group: { ...GROUP, name: "Arcuate L v2" },
      source: SOURCE,
      id: original.id,
      createdAt: original.createdAt,
    });
    expect(updated.id).toEqual(original.id);
    expect(updated.createdAt).toEqual(original.createdAt);
    expect(updated.group.name).toEqual("Arcuate L v2");
  });

  it("omits absent optional fields rather than writing undefined", () => {
    const doc = makeRoiGroupDocument({ group: GROUP, source: SOURCE });
    expect("createdBy" in doc).toBe(false);
    expect("scene" in doc).toBe(false);
  });
});

describe("object naming", () => {
  it("round-trips an id through the object name", () => {
    const name = roiGroupObjectName("abc123");
    expect(name).toEqual("groups/abc123.json");
    expect(roiGroupIdFromObjectName(name)).toEqual("abc123");
  });

  it("rejects names that are not group documents", () => {
    for (const name of [
      "groups/README.txt",
      "groups/nested/x.json", // would yield an id containing a slash
      "other/abc.json",
      "groups/.json", // empty id
      "groups/",
      "abc.json",
    ]) {
      expect(roiGroupIdFromObjectName(name)).toBeUndefined();
    }
  });
});

describe("parseRoiGroupDocument", () => {
  it("round-trips a document including the scene", () => {
    const doc = makeRoiGroupDocument({
      group: GROUP,
      source: SOURCE,
      scene: { url: "https://viewer.example/#!{}", layerName: "tracts" },
      createdBy: "test@example.com",
    });
    const parsed = parseRoiGroupDocument(JSON.parse(JSON.stringify(doc)));
    expect(parsed.id).toEqual(doc.id);
    expect(parsed.group).toEqual(GROUP);
    expect(parsed.source.url).toEqual(SOURCE.url);
    expect(parsed.scene?.url).toEqual("https://viewer.example/#!{}");
    expect(parsed.scene?.layerName).toEqual("tracts");
    expect(parsed.createdBy).toEqual("test@example.com");
  });

  it("refuses a newer schema version", () => {
    const doc = makeRoiGroupDocument({ group: GROUP, source: SOURCE });
    expect(() =>
      parseRoiGroupDocument({
        ...doc,
        schemaVersion: ROI_GROUP_SCHEMA_VERSION + 1,
      }),
    ).toThrow(/newer/);
  });

  it("rejects documents missing required fields", () => {
    const doc = makeRoiGroupDocument({ group: GROUP, source: SOURCE });
    for (const key of ["id", "group", "source", "createdAt", "updatedAt"]) {
      const broken: any = { ...doc };
      delete broken[key];
      expect(() => parseRoiGroupDocument(broken)).toThrow();
    }
  });
});

describe("roiGroupCustomMetadata", () => {
  it("mirrors the fields the browse list renders", () => {
    const doc = makeRoiGroupDocument({
      group: GROUP,
      source: SOURCE,
      createdBy: "test@example.com",
    });
    expect(roiGroupCustomMetadata(doc)).toEqual({
      roiGroupName: "Arcuate L",
      createdBy: "test@example.com",
      sourceUrl: SOURCE.url,
    });
  });

  it("omits fields that are absent, never writing the string undefined", () => {
    // GCS custom metadata values must be strings; an undefined leaking through
    // would surface in the browse list as the literal text.
    const doc = makeRoiGroupDocument({ group: GROUP, source: SOURCE });
    const metadata = roiGroupCustomMetadata(doc);
    expect("createdBy" in metadata).toBe(false);
    expect(Object.values(metadata)).not.toContain(undefined);
    expect(Object.values(metadata)).not.toContain("undefined");
  });
});
