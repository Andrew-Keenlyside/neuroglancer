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
 * @file Which store document each live group is backed by.
 */

import { describe, it, expect } from "vitest";
import { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";
import {
  groupIdForDocument,
  rememberSavedDocument,
  savedDocumentFor,
} from "#src/datasource/zarr-vectors/store_provenance.js";

const REF = { id: "doc-abc", createdAt: "2026-07-19T00:00:00.000Z" };

describe("store provenance", () => {
  it("round-trips a mapping in both directions", () => {
    const state = new RoiFilterState();
    const groupId = state.addGroup();
    rememberSavedDocument(state, groupId, REF);

    expect(savedDocumentFor(state, groupId)).toEqual(REF);
    expect(groupIdForDocument(state, "doc-abc")).toEqual(groupId);
  });

  it("reports nothing for an unmapped group or document", () => {
    const state = new RoiFilterState();
    const groupId = state.addGroup();
    expect(savedDocumentFor(state, groupId)).toBeUndefined();
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
  });

  it("stops reporting a document whose group was deleted", () => {
    // Otherwise the store checklist would keep the entry ticked, claiming a
    // dissection is on screen after the user removed it.
    const state = new RoiFilterState();
    const groupId = state.addGroup();
    rememberSavedDocument(state, groupId, REF);
    expect(groupIdForDocument(state, "doc-abc")).toEqual(groupId);

    state.removeGroup(groupId);
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
    // And again, now that the stale entry has been dropped.
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
  });

  it("does not confuse ids between layers", () => {
    // Group ids are per-RoiFilterState, so two layers both have a group 1.
    const a = new RoiFilterState();
    const b = new RoiFilterState();
    const inA = a.addGroup();
    const inB = b.addGroup();
    expect(inA).toEqual(inB);

    rememberSavedDocument(a, inA, REF);
    expect(groupIdForDocument(a, "doc-abc")).toEqual(inA);
    expect(groupIdForDocument(b, "doc-abc")).toBeUndefined();
    expect(savedDocumentFor(b, inB)).toBeUndefined();
  });

  it("re-saving a group replaces its mapping rather than accumulating", () => {
    const state = new RoiFilterState();
    const groupId = state.addGroup();
    rememberSavedDocument(state, groupId, REF);
    const updated = { id: "doc-xyz", createdAt: REF.createdAt };
    rememberSavedDocument(state, groupId, updated);

    expect(savedDocumentFor(state, groupId)).toEqual(updated);
    expect(groupIdForDocument(state, "doc-xyz")).toEqual(groupId);
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
  });

  it("tracks several documents in one layer independently", () => {
    const state = new RoiFilterState();
    const first = state.addGroup();
    const second = state.addGroup();
    rememberSavedDocument(state, first, REF);
    rememberSavedDocument(state, second, { id: "doc-2", createdAt: "x" });

    expect(groupIdForDocument(state, "doc-abc")).toEqual(first);
    expect(groupIdForDocument(state, "doc-2")).toEqual(second);

    state.removeGroup(first);
    expect(groupIdForDocument(state, "doc-abc")).toBeUndefined();
    expect(groupIdForDocument(state, "doc-2")).toEqual(second);
  });
});
