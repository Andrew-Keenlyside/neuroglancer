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

/**
 * Parsing for "add group from JSON".
 *
 * Three shapes reach this, and a user pasting from the URL hash, from a saved
 * document, or from a colleague's message has no reason to know which they
 * hold:
 *
 * 1. a bare group -- `groupToJson` output, `{name, color, rois, ...}`;
 * 2. an array of bare groups -- what the `groups` key of the layer state holds;
 * 3. a full {@link RoiGroupDocument} -- what the shared store round-trips.
 *
 * All three collapse to a list of groups ready for `RoiFilterState.insertGroup`.
 * Kept free of DOM so the shape-detection and the error messages can be tested
 * directly; the dialog is a thin wrapper over {@link parseGroupsJson}.
 */

import type { RoiGroup } from "#src/datasource/zarr-vectors/roi_filter_state.js";
import { groupFromJson } from "#src/datasource/zarr-vectors/roi_filter_state.js";
import { parseRoiGroupDocument } from "#src/roi_store/schema.js";

/** A group ready to be handed to `insertGroup`, which assigns the real id. */
export type ImportedGroup = Omit<RoiGroup, "id">;

/** `groupFromJson` requires an id that `insertGroup` then discards. */
function toImported(json: unknown): ImportedGroup {
  const { id: _placeholder, ...group } = groupFromJson(json, 0);
  return group;
}

/**
 * A document carries its group under `group`; a bare group carries `rois`
 * directly. Discriminating on the key rather than on `schemaVersion` means a
 * hand-written document missing that field still gets the document error
 * message ("schemaVersion: ...") instead of the misleading bare-group one
 * ("rois: expected array").
 */
function looksLikeDocument(json: any): boolean {
  return typeof json === "object" && json !== null && "group" in json;
}

/**
 * Parse pasted or uploaded JSON into groups.
 *
 * Throws with a message meant to be shown verbatim to the user: the underlying
 * `verifyObject`/`parseArray` helpers already say which property is wrong, so
 * this only adds the position of a bad element within an array.
 */
export function parseGroupsJson(text: string): ImportedGroup[] {
  const trimmed = text.trim();
  if (trimmed === "") throw new Error("Nothing to import: the input is empty.");

  let json: unknown;
  try {
    json = JSON.parse(trimmed);
  } catch (e) {
    throw new Error(`Not valid JSON: ${(e as Error).message}`);
  }

  if (Array.isArray(json)) {
    if (json.length === 0) {
      throw new Error("Nothing to import: the array holds no groups.");
    }
    return json.map((entry, i) => {
      try {
        // An array of documents is not a shape the viewer ever emits, but it is
        // an easy thing to assemble by hand, so accept it too.
        return looksLikeDocument(entry)
          ? toImported(parseRoiGroupDocument(entry).group)
          : toImported(entry);
      } catch (e) {
        throw new Error(`Group ${i + 1}: ${(e as Error).message}`);
      }
    });
  }

  if (looksLikeDocument(json)) {
    return [toImported(parseRoiGroupDocument(json).group)];
  }
  return [toImported(json)];
}
