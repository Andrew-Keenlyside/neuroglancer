/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Decoding for a level's `groups/` array — the store's own named partition of
 * its objects (tract bundles, cell classes, connected components).
 *
 * A tractogram carries this as the anatomy: 43 bundles with names like `ac`,
 * `fx`, `stnvlpfc`.  Without it a reader sees only object ids, and a store
 * cannot be understood without the writing application's source next to it.
 *
 * On disk the array is one `vlen-bytes` element per group holding that group's
 * member object ids as little-endian int64, with the names on the array's
 * `group_names` attribute.  A group written as a contiguous id run stores an
 * empty blob and a `[start, stop)` entry in `group_ranges` instead, so a
 * billion-member group costs O(1) rather than 8 bytes per member.
 *
 * The bytes-to-blobs step is `decodeVlenBytesChunk`; everything from the blobs
 * to a per-object group id lives here, free of I/O so it can be tested
 * directly.
 */

import type { InlineSegmentProperty } from "#src/segmentation_display_state/property_map.js";

/** Per-object group membership, decoded from a level's `groups/` array. */
export interface ObjectGroupMembership {
  /** Display name per group id, in group-id order.  Unique, whitespace-free. */
  readonly names: string[];
  /** Group id per object id; `-1` for an object that belongs to no group. */
  readonly groupByObject: Int32Array;
}

export interface ObjectGroupDecodeResult {
  readonly membership: ObjectGroupMembership;
  /**
   * How many objects were claimed by more than one group.  The segment-property
   * encoding this feeds models one group per object, so a non-zero count is
   * worth surfacing rather than silently resolving.
   */
  readonly overlaps: number;
}

/**
 * Group count declared by the array's metadata, or `0` when it declares none.
 *
 * `num_groups` is the writer's own count; the array shape is the fallback for
 * a store that omits it.  The caller needs this before reading any chunk, to
 * know how many chunks the group axis spans.
 */
export function parseGroupCount(attrs: any, shape: unknown): number {
  const declared = Number(
    attrs?.num_groups ?? (Array.isArray(shape) ? shape[0] : undefined) ?? 0,
  );
  if (!Number.isFinite(declared) || declared <= 0) return 0;
  return Math.floor(declared);
}

/**
 * Make group names usable as segment-property tags.
 *
 * The tag query language splits on whitespace and looks tags up by name, so a
 * name carrying a space would be unselectable and a duplicate name would make
 * two groups indistinguishable.  Both are repaired rather than rejected: a
 * store whose names merely need tidying should still be filterable.
 */
export function sanitizeGroupNames(raw: unknown, numGroups: number): string[] {
  const names: string[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < numGroups; ++i) {
    const candidate = Array.isArray(raw) ? raw[i] : undefined;
    let name =
      typeof candidate === "string"
        ? candidate.trim().replace(/\s+/g, "_")
        : "";
    // Matches zarr-vectors' own convention for an unnamed row.
    if (name.length === 0) name = `group_${i}`;
    if (seen.has(name)) name = `${name}_${i}`;
    seen.add(name);
    names.push(name);
  }
  return names;
}

/** `{gid: [start, stop)}` for the groups stored as a contiguous id run. */
function parseGroupRanges(raw: unknown): Map<number, [number, number]> {
  const ranges = new Map<number, [number, number]>();
  if (raw === null || typeof raw !== "object") return ranges;
  for (const [key, value] of Object.entries(raw as object)) {
    const gid = Number(key);
    if (!Number.isInteger(gid) || !Array.isArray(value)) continue;
    const start = Number(value[0]);
    const stop = Number(value[1]);
    if (Number.isFinite(start) && Number.isFinite(stop) && stop > start) {
      ranges.set(gid, [start, stop]);
    }
  }
  return ranges;
}

/**
 * Turn decoded `groups/` blobs into a per-object group id.
 *
 * `blobs[gid]` holds group `gid`'s member ids as little-endian int64; a group
 * covered by `group_ranges` has an empty blob and is expanded from the range.
 * Returns `undefined` when the array declares no groups or no members — there
 * is nothing to tag with.
 *
 * The object count is not recorded on this array, so it is taken as the
 * largest member id seen plus one.  A caller that knows the real count (an
 * object-attribute column does) should reconcile against its own.
 */
export function buildObjectGroupMembership(
  attrs: any,
  blobs: readonly Uint8Array[],
  shape?: unknown,
): ObjectGroupDecodeResult | undefined {
  const numGroups = parseGroupCount(attrs, shape);
  if (numGroups === 0) return undefined;
  const names = sanitizeGroupNames(attrs?.group_names, numGroups);
  const ranges = parseGroupRanges(attrs?.group_ranges);

  // Sizing pass: decode every member id, tracking the high-water mark so the
  // dense output array can be allocated once.
  const memberIds: number[][] = [];
  let numObjects = 0;
  for (let gid = 0; gid < numGroups; ++gid) {
    const range = ranges.get(gid);
    if (range !== undefined) {
      memberIds.push([]);
      numObjects = Math.max(numObjects, range[1]);
      continue;
    }
    const blob = blobs[gid];
    if (blob === undefined || blob.byteLength === 0) {
      memberIds.push([]);
      continue;
    }
    const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
    const count = Math.floor(blob.byteLength / 8);
    const ids: number[] = new Array(count);
    for (let i = 0; i < count; ++i) {
      // Read as BigInt because the ids are int64 on disk; object counts stay
      // far inside the safe-integer range, so the Number is exact.
      const id = Number(view.getBigInt64(i * 8, /*littleEndian=*/ true));
      ids[i] = id;
      if (id + 1 > numObjects) numObjects = id + 1;
    }
    memberIds.push(ids);
  }
  if (numObjects === 0) return undefined;

  const groupByObject = new Int32Array(numObjects).fill(-1);
  let overlaps = 0;
  const assign = (id: number, gid: number) => {
    if (id < 0 || id >= numObjects) return;
    if (groupByObject[id] !== -1) ++overlaps;
    groupByObject[id] = gid;
  };
  for (let gid = 0; gid < numGroups; ++gid) {
    const range = ranges.get(gid);
    if (range !== undefined) {
      for (let id = range[0]; id < range[1]; ++id) assign(id, gid);
      continue;
    }
    for (const id of memberIds[gid]) assign(id, gid);
  }
  return { membership: { names, groupByObject }, overlaps };
}

/**
 * Segment properties that make the store's groups selectable BY NAME.
 *
 * Two properties, because they do different jobs in the segment list:
 *
 *  - `tags` is the filter.  Typing `#fx` — or clicking the tag in the query
 *    summary — narrows the list to that bundle, and the list's
 *    visibility-toggle-all control then shows or hides exactly its objects.
 *    A numerical group-id column could only ever be filtered by number.
 *  - `label` is the readout, so a segment row names its bundle instead of
 *    repeating a bare id.
 *
 * `keepIndices[i]` is the object id of row `i` of the property map (the map
 * drops rows an attribute's `present_mask` marks absent, so the rows are not
 * always `0..O-1`).
 *
 * `takenIds` are property ids the object-attribute columns already claim; a
 * store carrying an attribute literally called `group` or `label` would
 * otherwise put two properties under one id, which the query language reads as
 * one field.
 */
export function groupSegmentProperties(
  groups: ObjectGroupMembership,
  keepIndices: readonly number[],
  takenIds: ReadonlySet<string> = new Set(),
): InlineSegmentProperty[] {
  const { names, groupByObject } = groups;
  const freeId = (preferred: string) => {
    let id = preferred;
    for (let n = 1; takenIds.has(id); ++n) id = `${preferred}_${n}`;
    return id;
  };
  const tagValues = new Array<string>(keepIndices.length);
  const labelValues = new Array<string>(keepIndices.length);
  for (let i = 0; i < keepIndices.length; ++i) {
    const objectId = keepIndices[i];
    const gid = objectId < groupByObject.length ? groupByObject[objectId] : -1;
    // One tag per object, encoded as the group id's character code — the
    // encoding `InlineSegmentTagsProperty` documents.  A single tag is
    // trivially "distinct and sorted"; an object in no group gets none.
    tagValues[i] = gid < 0 ? "" : String.fromCharCode(gid);
    labelValues[i] = gid < 0 ? "" : names[gid];
  }
  return [
    {
      id: freeId("group"),
      type: "tags",
      tags: [...names],
      tagDescriptions: names.map(() => ""),
      values: tagValues,
    },
    {
      id: freeId("label"),
      type: "label",
      description: undefined,
      values: labelValues,
    },
  ];
}
