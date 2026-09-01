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
 * @file Round-trips saved ROI groups through a stand-in storage server.
 *
 * Exercises the wire format rather than mocking fetch: object naming, the
 * multipart upload that carries custom metadata, list pagination, the 401
 * retry, and the fact that a group survives save/load with its geometry,
 * predicates and operators intact.
 *
 * See `tests/fixtures/fake_gcs_subset.ts` for what the stand-in does and does
 * not prove.
 */

import { afterAll, beforeAll, describe, it, expect } from "vitest";
import { RoiOperator, RoiPredicate } from "#src/datasource/zarr-vectors/roi.js";
import {
  groupFromJson,
  groupToJson,
  RoiFilterState,
} from "#src/datasource/zarr-vectors/roi_filter_state.js";
import { RoiGroupStore } from "#src/roi_store/gcs_client.js";
import { makeRoiGroupDocument } from "#src/roi_store/schema.js";
import {
  startFakeGcsSubset,
  type FakeGcsSubset,
} from "#tests/fixtures/fake_gcs_subset.js";

let server: FakeGcsSubset;

beforeAll(async () => {
  server = await startFakeGcsSubset();
});

afterAll(async () => {
  await server.close();
});

const stubAuth = {
  getAccessToken: async () => "test-token",
  invalidate: () => {},
};

function makeStore(auth: typeof stubAuth | undefined = stubAuth) {
  return new RoiGroupStore({
    bucket: server.bucket,
    endpoint: server.url,
    auth,
  });
}

/** A group with two ROIs of different shape kinds, predicates and operators. */
function buildGroupJson(name: string) {
  const state = new RoiFilterState();
  const id = state.addGroup();
  state.updateGroup(id, { name, opacity: 0.42 });
  state.addRoi(id, {
    shape: {
      kind: "ellipsoid",
      center: Float32Array.from([1.5, -2.5, 3.5]),
      radii: Float32Array.from([4, 5, 6]),
    },
    predicate: RoiPredicate.ANY_SEGMENT,
    operator: RoiOperator.AND,
  });
  state.addRoi(id, {
    shape: {
      kind: "box",
      lower: Float32Array.from([0, 0, 0]),
      upper: Float32Array.from([10, 20, 30]),
    },
    predicate: RoiPredicate.EITHER_ENDPOINT,
    operator: RoiOperator.ANDNOT,
  });
  return groupToJson(state.groups[0]);
}

function buildDocument(name: string, id?: string) {
  return makeRoiGroupDocument({
    group: buildGroupJson(name),
    source: { url: "zarr-vectors://https://example.com/tracts.zarr" },
    createdBy: "test@example.com",
    id,
  });
}

describe("RoiGroupStore", () => {
  it("round-trips a group through save and read", async () => {
    const store = makeStore();
    const doc = buildDocument("Arcuate L");
    await store.save(doc);

    const loaded = await store.read(doc.id);
    expect(loaded.id).toEqual(doc.id);
    expect(loaded.schemaVersion).toEqual(1);
    expect(loaded.createdBy).toEqual("test@example.com");
    expect(loaded.source.url).toEqual(
      "zarr-vectors://https://example.com/tracts.zarr",
    );
    // The group JSON must survive unchanged: it is the same serialisation the
    // URL hash uses.
    expect(loaded.group).toEqual(doc.group);
  });

  it("stores the document under groups/<id>.json", async () => {
    const store = makeStore();
    const doc = buildDocument("Naming");
    await store.save(doc);
    expect(server.objects.has(`groups/${doc.id}.json`)).toBe(true);
  });

  it("restores a loaded group into a live filter state", async () => {
    const store = makeStore();
    const doc = buildDocument("Corticospinal R");
    await store.save(doc);
    const loaded = await store.read(doc.id);

    // Loading appends via insertGroup, which assigns a fresh session id rather
    // than clobbering the ids the filter tab is holding.
    const state = new RoiFilterState();
    const existing = state.addGroup();
    const { id: _discarded, ...group } = groupFromJson(loaded.group, 0);
    const newId = state.insertGroup(group);

    expect(newId).not.toEqual(existing);
    expect(state.groups).toHaveLength(2);
    const restored = state.groups[1];
    expect(restored.name).toEqual("Corticospinal R");
    expect(restored.opacity).toBeCloseTo(0.42);
    expect(restored.rois).toHaveLength(2);
    expect(restored.rois[0].shape.kind).toEqual("ellipsoid");
    expect(restored.rois[0].predicate).toEqual(RoiPredicate.ANY_SEGMENT);
    expect(restored.rois[1].shape.kind).toEqual("box");
    expect(restored.rois[1].operator).toEqual(RoiOperator.ANDNOT);
    expect(Array.from((restored.rois[1].shape as any).upper)).toEqual([
      10, 20, 30,
    ]);
    // And it re-serialises to exactly what was stored.
    expect(groupToJson(restored)).toEqual(doc.group);
  });

  it("lists saved groups with names from object metadata", async () => {
    const store = makeStore();
    const doc = buildDocument("Uncinate L");
    await store.save(doc);

    const entry = (await store.list()).find((s) => s.id === doc.id);
    expect(entry).toBeDefined();
    // Name comes from custom metadata, so the list renders without fetching
    // every document.
    expect(entry!.name).toEqual("Uncinate L");
    expect(entry!.createdBy).toEqual("test@example.com");
    expect(entry!.sourceUrl).toEqual(
      "zarr-vectors://https://example.com/tracts.zarr",
    );
  });

  it("replaces a document saved under the same id", async () => {
    const store = makeStore();
    const first = buildDocument("Before");
    await store.save(first);
    await store.save(buildDocument("After", first.id));

    expect((await store.read(first.id)).group.name).toEqual("After");
    const matching = (await store.list()).filter((s) => s.id === first.id);
    expect(matching).toHaveLength(1);
    expect(matching[0].name).toEqual("After");
  });

  it("deletes a saved group", async () => {
    const store = makeStore();
    const doc = buildDocument("Temporary");
    await store.save(doc);
    expect((await store.list()).some((s) => s.id === doc.id)).toBe(true);

    await store.delete(doc.id);
    expect((await store.list()).some((s) => s.id === doc.id)).toBe(false);
  });

  it("ignores objects under groups/ that are not documents", async () => {
    const store = makeStore();
    server.putRaw("groups/README.txt", "notes");
    server.putRaw("groups/nested/x.json", "{}");

    const listed = await store.list();
    expect(listed.some((s) => s.name === "README.txt")).toBe(false);
    expect(listed.some((s) => s.id.includes("/"))).toBe(false);
  });

  it("falls back to the id when an object carries no name metadata", async () => {
    const store = makeStore();
    const doc = buildDocument("Unlisted");
    // Written without custom metadata, as another tool might.
    server.putRaw(`groups/${doc.id}.json`, JSON.stringify(doc));
    const entry = (await store.list()).find((s) => s.id === doc.id);
    expect(entry?.name).toEqual(doc.id);
  });

  it("refuses a document from a newer schema version", async () => {
    const store = makeStore();
    const doc = buildDocument("From the future");
    server.putRaw(
      `groups/${doc.id}.json`,
      JSON.stringify({ ...doc, schemaVersion: 99 }),
    );
    // Better to refuse than to silently drop fields this build cannot see.
    await expect(store.read(doc.id)).rejects.toThrow(/newer/);
  });

  it("refuses to save without sign-in", async () => {
    // Constructed directly: passing `undefined` to makeStore would select its
    // default parameter and hand back the stub auth.
    const store = new RoiGroupStore({
      bucket: server.bucket,
      endpoint: server.url,
    });
    await expect(store.save(buildDocument("Anonymous"))).rejects.toThrow(
      /sign-in/,
    );
  });

  it("discards a rejected token and retries once", async () => {
    let invalidated = 0;
    const tokens = ["stale-token", "fresh-token"];
    const auth = {
      getAccessToken: async () => tokens[Math.min(invalidated, 1)],
      invalidate: () => {
        ++invalidated;
      },
    };
    const store = new RoiGroupStore({
      bucket: server.bucket,
      endpoint: server.url,
      auth,
    });

    server.authRequests.length = 0;
    server.failNextAuth(1, 401);
    const doc = buildDocument("Retried");
    await store.save(doc);

    expect(invalidated).toEqual(1);
    expect(server.authRequests.map((r) => r.token)).toEqual([
      "stale-token",
      "fresh-token",
    ]);
    expect(server.objects.has(`groups/${doc.id}.json`)).toBe(true);
  });

  it("gives up after a second rejection rather than looping", async () => {
    let invalidated = 0;
    const auth = {
      getAccessToken: async () => "always-bad",
      invalidate: () => {
        ++invalidated;
      },
    };
    const store = new RoiGroupStore({
      bucket: server.bucket,
      endpoint: server.url,
      auth,
    });

    server.authRequests.length = 0;
    server.failNextAuth(5, 401);
    await expect(store.save(buildDocument("Doomed"))).rejects.toThrow();
    // One retry, not an unbounded re-prompt loop.
    expect(server.authRequests).toHaveLength(2);
    expect(invalidated).toEqual(1);
    server.failNextAuth(0);
  });

  it("lists anonymously when the bucket allows it", async () => {
    // The happy path: no Authorization header is sent at all, so a public
    // bucket needs no sign-in to browse.
    const store = makeStore();
    const doc = buildDocument("Public");
    await store.save(doc);
    server.authRequests.length = 0;
    await store.list();
    expect(server.authRequests).toHaveLength(0);
  });

  it("escalates to a held token when anonymous listing is refused", async () => {
    // A bucket can serve objects publicly yet withhold storage.objects.list.
    const paged = await startFakeGcsSubset("locked-bucket");
    try {
      const auth = {
        getAccessToken: async () => "held-token",
        invalidate: () => {},
        cachedAccessToken: "held-token" as string | undefined,
      };
      const store = new RoiGroupStore({
        bucket: paged.bucket,
        endpoint: paged.url,
        auth,
      });
      paged.putRaw("groups/a.json", "{}", { roiGroupName: "A" });
      paged.rejectAnonymousList(true);

      const listed = await store.list();
      expect(listed.map((s) => s.name)).toEqual(["A"]);
    } finally {
      await paged.close();
    }
  });

  it("reports a specific error when listing is refused and nobody is signed in", async () => {
    const paged = await startFakeGcsSubset("locked-bucket-2");
    try {
      // No auth at all: the remedy is to sign in or open the bucket up, and
      // the message has to say so rather than surfacing a bare 401.
      const store = new RoiGroupStore({
        bucket: paged.bucket,
        endpoint: paged.url,
      });
      paged.rejectAnonymousList(true);
      await expect(store.list()).rejects.toThrow(/anonymous listing/i);
    } finally {
      await paged.close();
    }
  });

  it("never prompts for sign-in from a listing", async () => {
    // getAccessToken would open a popup; a listing must only ever use a token
    // that is already held.
    const paged = await startFakeGcsSubset("locked-bucket-3");
    try {
      let prompted = 0;
      const store = new RoiGroupStore({
        bucket: paged.bucket,
        endpoint: paged.url,
        auth: {
          getAccessToken: async () => {
            ++prompted;
            return "fresh";
          },
          invalidate: () => {},
          cachedAccessToken: undefined,
        },
      });
      paged.rejectAnonymousList(true);
      await expect(store.list()).rejects.toThrow(/anonymous listing/i);
      expect(prompted).toEqual(0);
    } finally {
      await paged.close();
    }
  });

  it("follows pagination when listing", async () => {
    const paged = await startFakeGcsSubset("paged-bucket", /*pageSize=*/ 2);
    try {
      const store = new RoiGroupStore({
        bucket: paged.bucket,
        endpoint: paged.url,
        auth: stubAuth,
      });
      const ids: string[] = [];
      for (let i = 0; i < 5; ++i) {
        const doc = buildDocument(`Group ${i}`);
        ids.push(doc.id);
        await store.save(doc);
      }
      const listed = await store.list();
      expect(listed).toHaveLength(5);
      expect(new Set(listed.map((s) => s.id))).toEqual(new Set(ids));
    } finally {
      await paged.close();
    }
  });
});
