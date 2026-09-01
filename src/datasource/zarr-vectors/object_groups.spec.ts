/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  buildObjectGroupMembership,
  groupSegmentProperties,
  parseGroupCount,
  sanitizeGroupNames,
} from "#src/datasource/zarr-vectors/object_groups.js";
import { decodeVlenBytesChunk } from "#src/datasource/zarr-vectors/vlen_bytes.js";
import {
  executeSegmentQuery,
  normalizeInlineSegmentPropertyMap,
  parseSegmentQuery,
  PreprocessedSegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";

/** One group's member ids, in the little-endian int64 form stored on disk. */
function blob(...ids: number[]): Uint8Array {
  const out = new Uint8Array(ids.length * 8);
  const view = new DataView(out.buffer);
  ids.forEach((id, i) => view.setBigInt64(i * 8, BigInt(id), true));
  return out;
}

/** The `vlen-bytes` chunk encoding, so a test can start from real bytes. */
function vlenChunk(blobs: Uint8Array[]): Uint8Array {
  const total = 4 + blobs.reduce((sum, b) => sum + 4 + b.byteLength, 0);
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, blobs.length, true);
  let offset = 4;
  for (const b of blobs) {
    view.setUint32(offset, b.byteLength, true);
    offset += 4;
    out.set(b, offset);
    offset += b.byteLength;
  }
  return out;
}

describe("parseGroupCount", () => {
  it("prefers num_groups, falls back to the array shape", () => {
    expect(parseGroupCount({ num_groups: 3 }, [7])).toBe(3);
    expect(parseGroupCount({}, [7])).toBe(7);
  });

  it("reports 0 when the count is absent or nonsensical", () => {
    expect(parseGroupCount({}, undefined)).toBe(0);
    expect(parseGroupCount({ num_groups: 0 }, undefined)).toBe(0);
    expect(parseGroupCount({ num_groups: -2 }, undefined)).toBe(0);
    expect(parseGroupCount({ num_groups: "many" }, undefined)).toBe(0);
  });
});

describe("sanitizeGroupNames", () => {
  it("keeps well-formed names untouched", () => {
    expect(sanitizeGroupNames(["ac", "fx", "stnvlpfc"], 3)).toEqual([
      "ac",
      "fx",
      "stnvlpfc",
    ]);
  });

  it("replaces whitespace, which the tag query language splits on", () => {
    expect(sanitizeGroupNames(["anterior commissure"], 1)).toEqual([
      "anterior_commissure",
    ]);
  });

  it("disambiguates duplicates so each group stays selectable", () => {
    expect(sanitizeGroupNames(["cst", "cst"], 2)).toEqual(["cst", "cst_1"]);
  });

  it("names unnamed rows positionally", () => {
    expect(sanitizeGroupNames(undefined, 2)).toEqual(["group_0", "group_1"]);
    expect(sanitizeGroupNames(["", "  "], 2)).toEqual(["group_0", "group_1"]);
  });
});

describe("buildObjectGroupMembership", () => {
  it("maps each object to the group that lists it", () => {
    const result = buildObjectGroupMembership(
      { num_groups: 2, group_names: ["ac", "fx"] },
      [blob(0, 1, 3), blob(2)],
    );
    expect(result).toBeDefined();
    expect(result!.membership.names).toEqual(["ac", "fx"]);
    expect(Array.from(result!.membership.groupByObject)).toEqual([0, 0, 1, 0]);
    expect(result!.overlaps).toBe(0);
  });

  it("leaves objects in no group at -1", () => {
    const result = buildObjectGroupMembership({ num_groups: 1 }, [blob(0, 3)]);
    expect(Array.from(result!.membership.groupByObject)).toEqual([
      0, -1, -1, 0,
    ]);
  });

  it("expands a group stored as a contiguous id range", () => {
    // A range group carries an empty blob; its members live in `group_ranges`.
    const result = buildObjectGroupMembership(
      {
        num_groups: 2,
        group_names: ["bulk", "tail"],
        group_ranges: { "0": [0, 3] },
      },
      [new Uint8Array(0), blob(3)],
    );
    expect(Array.from(result!.membership.groupByObject)).toEqual([0, 0, 0, 1]);
  });

  it("counts objects claimed by more than one group", () => {
    const result = buildObjectGroupMembership({ num_groups: 2 }, [
      blob(0, 1),
      blob(1),
    ]);
    expect(result!.overlaps).toBe(1);
    // The later group wins, so the object is tagged exactly once.
    expect(Array.from(result!.membership.groupByObject)).toEqual([0, 1]);
  });

  it("returns undefined when there is nothing to tag with", () => {
    expect(buildObjectGroupMembership({ num_groups: 0 }, [])).toBeUndefined();
    expect(
      buildObjectGroupMembership({ num_groups: 2 }, [
        new Uint8Array(0),
        new Uint8Array(0),
      ]),
    ).toBeUndefined();
  });

  it("tolerates a short blob list, as an unwritten chunk produces", () => {
    const result = buildObjectGroupMembership({ num_groups: 3 }, [blob(0)]);
    expect(result!.membership.names).toHaveLength(3);
    expect(Array.from(result!.membership.groupByObject)).toEqual([0]);
  });

  it("decodes the on-disk chunk form end to end", () => {
    // The shape a real store hands the reader: vlen-bytes chunk -> blobs ->
    // membership.
    const chunk = vlenChunk([blob(0, 1), blob(2, 3, 4)]);
    const result = buildObjectGroupMembership(
      { num_groups: 2, group_names: ["ac", "fx"] },
      decodeVlenBytesChunk(chunk),
    );
    expect(Array.from(result!.membership.groupByObject)).toEqual([
      0, 0, 1, 1, 1,
    ]);
  });
});

describe("groupSegmentProperties", () => {
  /** A three-bundle store with one ungrouped object, as the UI would see it. */
  function makeMap(keepIndices = [0, 1, 2, 3, 4]) {
    const groups = buildObjectGroupMembership(
      { num_groups: 3, group_names: ["ac", "fx", "cst"] },
      [blob(0, 1), blob(2), blob(3)],
    )!.membership;
    return new PreprocessedSegmentPropertyMap({
      inlineProperties: normalizeInlineSegmentPropertyMap({
        ids: BigUint64Array.from(keepIndices.map((i) => BigInt(i))),
        properties: groupSegmentProperties(groups, keepIndices),
      }),
    });
  }

  /** The ids a filter query matches — what the visibility control acts on. */
  function queryIds(map: PreprocessedSegmentPropertyMap, query: string) {
    const result = executeSegmentQuery(map, parseSegmentQuery(map, query));
    const ids = map.segmentPropertyMap.inlineProperties!.ids;
    return Array.from(result.indices ?? [], (index) => Number(ids[index]));
  }

  it("selects a bundle by name", () => {
    // This is the toggle: the query narrows the segment list to one bundle,
    // and the list's visibility control then acts on exactly these ids.
    const map = makeMap();
    expect(queryIds(map, "#ac")).toEqual([0, 1]);
    expect(queryIds(map, "#fx")).toEqual([2]);
    expect(queryIds(map, "#cst")).toEqual([3]);
  });

  it("excludes a bundle by name", () => {
    // Order follows the list's sort, which is not what the toggle depends on.
    expect(queryIds(makeMap(), "-#ac").sort()).toEqual([2, 3, 4]);
  });

  it("labels each object with its group name", () => {
    // Group 0's tag encodes as character code 0, which a naive
    // empty-string test for "no tags" would drop; check it survives.
    const map = makeMap();
    expect(map.getSegmentLabel(0n)).toBe("ac #ac");
    expect(map.getSegmentLabel(2n)).toBe("fx #fx");
  });

  it("leaves an ungrouped object unlabelled and untagged", () => {
    expect(makeMap().getSegmentLabel(4n)).toBeUndefined();
  });

  it("sidesteps property ids an attribute column already claims", () => {
    const groups = buildObjectGroupMembership({ num_groups: 1 }, [
      blob(0),
    ])!.membership;
    const properties = groupSegmentProperties(
      groups,
      [0],
      new Set(["group", "label"]),
    );
    expect(properties.map((p) => p.id)).toEqual(["group_1", "label_1"]);
  });
});
