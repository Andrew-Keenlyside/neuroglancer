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
 * @file Reading and writing saved ROI groups in a Google Cloud Storage bucket.
 *
 * The bucket is public-read, so listing and loading are ANONYMOUS: browsing the
 * shared library never prompts for sign-in, and works identically in the plain
 * and cross-origin isolated builds.  Only writes carry a token.
 *
 * Talks to the GCS JSON API directly rather than going through `KvStore`.
 * `KvStore` is read-only by construction (`src/kvstore/index.ts`) and is built
 * for byte-range reads of large chunked data, not small whole documents; the
 * existing precedent for a remote write is `src/datasource/state_share.ts`,
 * which likewise bypasses it.
 *
 * Bucket and endpoint are constructor arguments rather than being read from the
 * build define inside each method, so a test can point a store at a local
 * emulator.  `getRoiGroupStore()` supplies the configured instance.
 */

import { getRoiStoreConfig } from "#src/roi_store/config.js";
import { getRoiStoreAuth } from "#src/roi_store/credentials.js";
import type {
  RoiGroupDocument,
  RoiGroupSummary,
} from "#src/roi_store/schema.js";
import {
  parseRoiGroupDocument,
  roiGroupCustomMetadata,
  roiGroupIdFromObjectName,
  ROI_GROUP_PREFIX,
  roiGroupObjectName,
} from "#src/roi_store/schema.js";
import { HttpError, fetchOk } from "#src/util/http_request.js";
import { NullarySignal } from "#src/util/signal.js";

/**
 * Fires after the store's contents change, so open listings can refresh.
 *
 * Module-level rather than per-store instance: a save made from one layer's
 * Filter tab must refresh the pickers in every other layer's tab too, and they
 * all address the same bucket.
 */
export const roiGroupStoreChanged = new NullarySignal();

/**
 * The part of the sign-in state this module needs.  Narrow so tests can supply
 * a stub without standing up an OAuth flow; `RoiStoreAuth` satisfies it.
 */
export interface RoiStoreTokenSource {
  getAccessToken(signal?: AbortSignal): Promise<string>;
  invalidate(): void;
  /** A token if one is already held; must NOT prompt for sign-in. */
  readonly cachedAccessToken?: string | undefined;
}

/**
 * The bucket refused an anonymous listing.
 *
 * Distinguished from a generic failure because the remedy is specific and
 * worth telling the user: either grant `allUsers` the `objectViewer` role
 * (which is what carries `storage.objects.list`), or sign in. A bucket can
 * serve objects publicly while still denying listing — reads succeed and only
 * the browse UI breaks — so this is worth naming rather than surfacing as a
 * bare 401.
 */
export class RoiStoreListForbiddenError extends Error {
  constructor(readonly signedIn: boolean) {
    super(
      signedIn
        ? "This account is not allowed to list the ROI group bucket."
        : "This bucket does not allow anonymous listing. Sign in, or grant " +
            "allUsers the Storage Object Viewer role on the bucket.",
    );
    this.name = "RoiStoreListForbiddenError";
  }
}

export interface RoiGroupStoreOptions {
  bucket: string;
  endpoint: string;
  /** Needed only for writes; listing and reading are anonymous. */
  auth?: RoiStoreTokenSource;
}

export class RoiGroupStore {
  constructor(private options: RoiGroupStoreOptions) {}

  private get objectApiUrl(): string {
    const { endpoint, bucket } = this.options;
    return `${endpoint}/storage/v1/b/${encodeURIComponent(bucket)}/o`;
  }

  private get uploadApiUrl(): string {
    const { endpoint, bucket } = this.options;
    return `${endpoint}/upload/storage/v1/b/${encodeURIComponent(bucket)}/o`;
  }

  private objectUrl(id: string): URL {
    const url = new URL(this.objectApiUrl);
    // The object name contains a slash, which must stay encoded: GCS addresses
    // objects by name, not by path.
    url.pathname += `/${encodeURIComponent(roiGroupObjectName(id))}`;
    return url;
  }

  /**
   * Lists saved groups.
   *
   * Attempted anonymously first, because a public-read bucket needs no
   * credential and opening a panel must not raise a sign-in popup. A bucket
   * can serve objects publicly while still withholding `storage.objects.list`
   * — reads work and only listing 401s — so if a token is ALREADY held the
   * request is retried with it before giving up.
   *
   * Requests only the fields the browse list renders and reads the name and
   * author from each object's custom metadata, so the list costs one request
   * per page rather than one per document.
   */
  async list(signal?: AbortSignal): Promise<RoiGroupSummary[]> {
    const summaries: RoiGroupSummary[] = [];
    let pageToken: string | undefined;
    do {
      const url = new URL(this.objectApiUrl);
      url.searchParams.set("prefix", ROI_GROUP_PREFIX);
      url.searchParams.set(
        "fields",
        "nextPageToken,items(name,updated,metadata)",
      );
      if (pageToken !== undefined) {
        url.searchParams.set("pageToken", pageToken);
      }
      const response = await this.fetchListPage(url.toString(), signal);
      const page = await response.json();
      for (const item of page?.items ?? []) {
        const id = roiGroupIdFromObjectName(item?.name ?? "");
        if (id === undefined) continue;
        const metadata = item.metadata ?? {};
        summaries.push({
          id,
          // Fall back to the id so a document written before this metadata
          // existed, or by another tool, still appears in the list.
          name: metadata.roiGroupName ?? id,
          createdBy: metadata.createdBy,
          sourceUrl: metadata.sourceUrl,
          updated: item.updated,
        });
      }
      pageToken = page?.nextPageToken ?? undefined;
    } while (pageToken !== undefined);
    return summaries;
  }

  /** One page of a listing: anonymous, escalating to a held token if refused. */
  private async fetchListPage(
    url: string,
    signal?: AbortSignal,
  ): Promise<Response> {
    try {
      return await fetchOk(url, { signal });
    } catch (error) {
      const refused =
        error instanceof HttpError &&
        (error.status === 401 || error.status === 403);
      if (!refused) throw error;
      // Only a token already in hand — never prompt from a listing.
      const token = this.options.auth?.cachedAccessToken;
      if (token === undefined) {
        throw new RoiStoreListForbiddenError(/*signedIn=*/ false);
      }
      try {
        return await fetchOk(url, {
          signal,
          headers: { Authorization: `Bearer ${token}` },
        });
      } catch (authedError) {
        if (
          authedError instanceof HttpError &&
          (authedError.status === 401 || authedError.status === 403)
        ) {
          this.options.auth?.invalidate();
          throw new RoiStoreListForbiddenError(/*signedIn=*/ true);
        }
        throw authedError;
      }
    }
  }

  /** Loads one saved group.  Anonymous: the bucket is public-read. */
  async read(id: string, signal?: AbortSignal): Promise<RoiGroupDocument> {
    const url = this.objectUrl(id);
    url.searchParams.set("alt", "media");
    const response = await fetchOk(url.toString(), { signal });
    return parseRoiGroupDocument(await response.json());
  }

  /** Creates or replaces a saved group.  Requires write access to the bucket. */
  async save(doc: RoiGroupDocument, signal?: AbortSignal): Promise<void> {
    const url = new URL(this.uploadApiUrl);
    url.searchParams.set("uploadType", "multipart");
    const { body, contentType } = multipartUploadBody(doc);
    await this.fetchWithToken(
      url.toString(),
      { method: "POST", body, headers: { "Content-Type": contentType } },
      signal,
    );
    roiGroupStoreChanged.dispatch();
  }

  /** Deletes a saved group.  Requires delete access to the bucket. */
  async delete(id: string, signal?: AbortSignal): Promise<void> {
    await this.fetchWithToken(
      this.objectUrl(id).toString(),
      { method: "DELETE" },
      signal,
    );
    roiGroupStoreChanged.dispatch();
  }

  /**
   * Runs an authenticated request, retrying once if the token is refused.
   *
   * A token can stop working before it expires — revoked, or the account lost
   * its grant on the bucket — which the local expiry check cannot detect, so a
   * 401/403 is what tells us to discard it and ask again.  Retrying only once
   * keeps a genuinely unauthorised account from looping on the popup.
   */
  private async fetchWithToken(
    url: string,
    init: RequestInit,
    signal?: AbortSignal,
  ): Promise<Response> {
    const { auth } = this.options;
    if (auth === undefined) {
      throw new Error("Saving ROI groups requires sign-in");
    }
    for (let attempt = 0; ; ++attempt) {
      const token = await auth.getAccessToken(signal);
      const headers = new Headers(init.headers);
      headers.set("Authorization", `Bearer ${token}`);
      try {
        return await fetchOk(url, { ...init, headers, signal });
      } catch (error) {
        const rejected =
          error instanceof HttpError &&
          (error.status === 401 || error.status === 403);
        if (!rejected || attempt > 0) throw error;
        auth.invalidate();
      }
    }
  }
}

/**
 * Builds a multipart upload body, so the document and its custom metadata are
 * written in a single request.
 */
function multipartUploadBody(doc: RoiGroupDocument): {
  body: string;
  contentType: string;
} {
  const boundary = `roi-group-boundary-${doc.id}`;
  const metadata = {
    name: roiGroupObjectName(doc.id),
    contentType: "application/json",
    metadata: roiGroupCustomMetadata(doc),
  };
  const body =
    `--${boundary}\r\n` +
    "Content-Type: application/json; charset=UTF-8\r\n\r\n" +
    `${JSON.stringify(metadata)}\r\n` +
    `--${boundary}\r\n` +
    "Content-Type: application/json; charset=UTF-8\r\n\r\n" +
    `${JSON.stringify(doc)}\r\n` +
    `--${boundary}--`;
  return { body, contentType: `multipart/related; boundary=${boundary}` };
}

let singleton: RoiGroupStore | undefined;

/** The store described by the `ROI_STORE` build define. */
export function getRoiGroupStore(): RoiGroupStore {
  if (singleton === undefined) {
    const { bucket, endpoint } = getRoiStoreConfig();
    singleton = new RoiGroupStore({
      bucket,
      endpoint,
      auth: getRoiStoreAuth(),
    });
  }
  return singleton;
}
