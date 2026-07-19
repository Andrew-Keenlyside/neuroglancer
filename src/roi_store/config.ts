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
 * @file Build-time configuration for the shared ROI group store.
 *
 * Follows the `STATE_SERVERS` precedent in `src/datasource/state_share.ts`:
 * the define is optional, and when it is absent the whole feature — sign-in
 * chip, save action, browse dialog — is omitted from the UI.
 */

declare const ROI_STORE:
  | {
      /** Name of the public-read GCS bucket holding `groups/*.json`. */
      bucket: string;
      /**
       * Which sign-in backs writes.  Defaults to `google`.
       *
       * `google` — a Google OAuth2 token with a storage scope, written straight
       * to the GCS JSON API. The only option that works in the cross-origin
       * isolated (pyodide) build, because our own redirect page can broadcast
       * the response past COOP.
       *
       * `middleauth` — reuse neuroglancer's existing CAVE/middleauth login (the
       * same one `state_share` uses), so no separate OAuth client is needed.
       * TWO constraints: the middleauth token is a CAVE bearer token, so the
       * `endpoint` MUST be a server that accepts it (not raw GCS); and the
       * login popup gets its response from the CAVE server's own page via
       * `window.opener`, which COOP severs — so this does NOT work in the
       * pyodide build.
       */
      provider?: "google" | "middleauth";
      /**
       * Sign in eagerly at viewer start rather than lazily on first save.
       *
       * This is the "authenticate on the wasm before adding the url" path: the
       * token is ready before any listing, so a bucket that denies anonymous
       * listing populates the picker without a later prompt.
       */
      eager?: boolean;
      /**
       * Storage endpoint override, for pointing development builds at a local
       * stand-in server instead of Google.  Defaults to Google Cloud Storage.
       * For the `middleauth` provider this must be an endpoint that accepts a
       * CAVE bearer token.
       */
      endpoint?: string;

      // --- google provider ---
      /** OAuth2 client ID authorised for this origin and the storage scope. */
      clientId?: string;
      /**
       * OAuth scopes to request.  Defaults to the storage scopes.
       *
       * FOR LOCAL DEVELOPMENT ONLY.  `devstorage.read_write` is a sensitive
       * scope, so a new OAuth client cannot grant it to anyone outside the
       * project until Google has verified it.  Against a stand-in server that
       * does not check tokens, dropping to `["openid","email"]` lets the whole
       * sign-in → save → load path be exercised with an existing client while
       * that verification is pending.  A build pointed at real Google Cloud
       * Storage MUST leave this unset.
       */
      scopes?: string[];

      // --- middleauth provider ---
      /**
       * middleauth login server, e.g. `https://global.daf-apis.com`.  Required
       * when `provider` is `middleauth`.
       */
      authServer?: string;
    }
  | undefined;

export const DEFAULT_STORAGE_ENDPOINT = "https://storage.googleapis.com";

/**
 * `devstorage.read_write` permits writing objects; `openid`+`email` are what
 * make the id token carry an email to attribute saves to.
 */
export const DEFAULT_SCOPES = [
  "openid",
  "email",
  "https://www.googleapis.com/auth/devstorage.read_write",
];

export interface RoiStoreConfig {
  bucket: string;
  endpoint: string;
  eager: boolean;
  provider: "google" | "middleauth";
  /** Set when `provider` is `google`. */
  clientId?: string;
  scopes: string[];
  /** Set when `provider` is `middleauth`. */
  authServer?: string;
}

function readConfig(): RoiStoreConfig | undefined {
  if (typeof ROI_STORE === "undefined" || ROI_STORE === undefined) {
    return undefined;
  }
  const { bucket, clientId, endpoint, scopes, provider, eager, authServer } =
    ROI_STORE;
  if (typeof bucket !== "string" || bucket.length === 0) return undefined;

  const resolvedProvider = provider === "middleauth" ? "middleauth" : "google";
  // Each provider has a required field; without it the feature cannot work, so
  // treat it as unconfigured (feature off) rather than half-mounting a UI that
  // fails on first use.
  if (resolvedProvider === "google") {
    if (typeof clientId !== "string" || clientId.length === 0) return undefined;
  } else if (typeof authServer !== "string" || authServer.length === 0) {
    return undefined;
  }

  const validScopes =
    Array.isArray(scopes) && scopes.every((s) => typeof s === "string")
      ? scopes.filter((s) => s.length > 0)
      : [];
  // Trailing slashes would double up when joined with a path.  Type-checked
  // like the others rather than assumed: a non-string here would throw at
  // module scope, taking the whole viewer down rather than just disabling the
  // feature, and a slash-only value would strip to "" and silently produce
  // same-origin request URLs.
  const trimmedEndpoint =
    typeof endpoint === "string" ? endpoint.replace(/\/+$/, "") : "";
  return {
    bucket,
    provider: resolvedProvider,
    eager: eager === true,
    clientId: resolvedProvider === "google" ? clientId : undefined,
    authServer:
      resolvedProvider === "middleauth"
        ? (authServer as string).replace(/\/+$/, "")
        : undefined,
    endpoint:
      trimmedEndpoint.length > 0 ? trimmedEndpoint : DEFAULT_STORAGE_ENDPOINT,
    scopes: validScopes.length > 0 ? validScopes : DEFAULT_SCOPES,
  };
}

const config = readConfig();

export const roiStoreEnabled = config !== undefined;

export function getRoiStoreConfig(): RoiStoreConfig {
  if (config === undefined) {
    throw new Error("ROI store is not configured for this build");
  }
  return config;
}
