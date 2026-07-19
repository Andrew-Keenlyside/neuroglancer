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
 * @file Which shared-store document each live group came from, or was last
 * written to.
 *
 * One mapping serves both directions, because they are the same relationship:
 * a group loaded from the store and a group saved to it are both *backed by*
 * that document. Re-saving a loaded group therefore updates it in place rather
 * than making a copy, and the store picker can tell which stored groups are
 * already on screen.
 *
 * Keyed on the `RoiFilterState` rather than held on the Filter tab, because the
 * tab is rebuilt every time the layer side panel is closed and reopened — on
 * the tab, re-saving after reopening the panel silently created a second
 * document instead of updating the first. A WeakMap so a disposed layer's
 * entries go with it.
 *
 * Deliberately not persisted: group ids are session-scoped, so after a reload
 * nothing ties a restored group to the document it came from.
 */

import type { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";

export interface SavedDocumentRef {
  /** Document id in the shared store. */
  id: string;
  /** Preserved across re-saves, so updating does not reset the creation time. */
  createdAt: string;
}

const savedDocuments = new WeakMap<
  RoiFilterState,
  Map<number, SavedDocumentRef>
>();

export function rememberSavedDocument(
  state: RoiFilterState,
  groupId: number,
  ref: SavedDocumentRef,
): void {
  let map = savedDocuments.get(state);
  if (map === undefined) {
    map = new Map();
    savedDocuments.set(state, map);
  }
  map.set(groupId, ref);
}

export function savedDocumentFor(
  state: RoiFilterState,
  groupId: number,
): SavedDocumentRef | undefined {
  return savedDocuments.get(state)?.get(groupId);
}

/**
 * The live group backed by `documentId`, or undefined if it is not loaded.
 *
 * Checks the group still exists: a mapping outlives the group it referred to
 * when that group is deleted, and a stale id would make the picker claim a
 * document is on screen when it is not.
 */
export function groupIdForDocument(
  state: RoiFilterState,
  documentId: string,
): number | undefined {
  const map = savedDocuments.get(state);
  if (map === undefined) return undefined;
  for (const [groupId, ref] of map) {
    if (ref.id !== documentId) continue;
    if (state.groups.some((g) => g.id === groupId)) return groupId;
    // The group is gone; drop the mapping so it cannot mislead again.
    map.delete(groupId);
    return undefined;
  }
  return undefined;
}
