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
 * Pure decode/transform helpers for zarr-vectors group-derived segment
 * properties (TRX `groups`/`group_attributes` -> tags + numerical columns).
 * Kept in their own module, separate from
 * `#src/datasource/zarr-vectors/frontend.js`, so they can be unit-tested
 * without pulling in that file's transitive import of the WebGL-dependent
 * skeleton rendering stack.
 */

/**
 * Decode a zarr-vectors `fixed_length_utf32` array chunk into `count`
 * fixed-width UTF-32LE strings, each `lengthBytes` wide (NUL-padded at the
 * end of each slot).  Not a standard zarr v3 data type — a zarr-vectors
 * convention for names/labels (e.g. group/tract names) — so there's no
 * existing decoder for it elsewhere in neuroglancer.
 */
export function decodeFixedLengthUtf32Strings(
  bytes: Uint8Array,
  lengthBytes: number,
  count: number,
): string[] {
  const codepointsPerString = lengthBytes / 4;
  const expectedBytes = count * lengthBytes;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors: expected ${expectedBytes} bytes (${count} x ` +
        `${lengthBytes}-byte fixed_length_utf32 strings), got ${bytes.byteLength}`,
    );
  }
  const u32 =
    bytes.byteOffset % 4 === 0
      ? new Uint32Array(
          bytes.buffer,
          bytes.byteOffset,
          count * codepointsPerString,
        )
      : (() => {
          const copy = new Uint8Array(bytes.byteLength);
          copy.set(bytes);
          return new Uint32Array(copy.buffer, 0, count * codepointsPerString);
        })();
  const out: string[] = new Array(count);
  for (let i = 0; i < count; ++i) {
    const start = i * codepointsPerString;
    let end = start;
    while (end < start + codepointsPerString && u32[end] !== 0) ++end;
    out[i] = String.fromCodePoint(...u32.subarray(start, end));
  }
  return out;
}

/**
 * Sanitize a bundle/tract name for use as a segment-properties tag or
 * numerical-property-id token.  neuroglancer's segment-list query parser
 * (`parseSegmentQuery` in `#src/segmentation_display_state/property_map.js`)
 * tokenizes `#<tag>` search queries by splitting on the literal space
 * character, with no quoting or escaping — a tag containing a space (e.g.
 * "Optic Radiation") can never be typed as a matching `#Optic Radiation`
 * query. Replacing whitespace with underscores keeps the name searchable;
 * the original, human-readable name is preserved separately as the tag's
 * description.
 */
export function sanitizeTagName(name: string): string {
  return name.trim().replace(/\s+/g, "_");
}

/**
 * Invert group->members into oid->[group indices] and encode each
 * streamline's group memberships as an `InlineSegmentTagsProperty` value
 * string per its contract: character codes are the tag (group) indices,
 * and "must be distinct and sorted" ascending.  Out-of-range member oids
 * (a malformed/truncated store) are silently skipped rather than throwing
 * — the tags subsource is a pure UI augmentation, not load-critical.
 */
export function invertGroupMembershipsToTags(
  numObjects: number,
  numGroups: number,
  memberOids: readonly (readonly number[])[],
): { groupsByOid: (number[] | undefined)[]; tagValues: string[] } {
  const groupsByOid: (number[] | undefined)[] = new Array(numObjects);
  for (let g = 0; g < numGroups; ++g) {
    for (const oid of memberOids[g]) {
      if (oid < 0 || oid >= numObjects) continue;
      const existing = groupsByOid[oid];
      if (existing === undefined) {
        groupsByOid[oid] = [g];
      } else {
        existing.push(g);
      }
    }
  }

  const tagValues: string[] = new Array(numObjects);
  for (let oid = 0; oid < numObjects; ++oid) {
    const gs = groupsByOid[oid];
    if (gs === undefined || gs.length === 0) {
      tagValues[oid] = "";
      continue;
    }
    gs.sort((a, b) => a - b);
    tagValues[oid] = String.fromCharCode(...gs);
  }
  return { groupsByOid, tagValues };
}
