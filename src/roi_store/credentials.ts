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
 * @file Google sign-in for the shared ROI group store.
 *
 * Saved ROI groups live in a public-read Google Cloud Storage bucket, so
 * browsing and loading need no credentials at all — a token is required only
 * to write.  Sign-in is therefore lazy: nothing here runs until the user first
 * saves.
 *
 * Deliberately NOT a `CredentialsProvider`.  That interface is built around
 * `makeCredentialsGetter`, which memoises the in-flight promise and exposes no
 * way to invalidate the cache other than handing back the stale credentials it
 * previously issued.  Sign-out has no such handle, and the sign-in chip needs
 * an observable signed-in state that `CredentialsProvider` does not provide.
 * The valuable parts of the existing machinery — `authenticateGoogleOAuth2`
 * and the `getCredentialsWithStatus` status-bar UI — are reused directly.
 */

import { getDefaultCredentialsManager } from "#src/credentials_provider/default_manager.js";
import type {
  CredentialsProvider,
  CredentialsWithGeneration,
} from "#src/credentials_provider/index.js";
import { getCredentialsWithStatus } from "#src/credentials_provider/interactive_credentials_provider.js";
import type { MiddleAuthToken } from "#src/kvstore/middleauth/credentials_provider.js";
import { getRoiStoreConfig } from "#src/roi_store/config.js";
import { raceWithAbort } from "#src/util/abort.js";
import {
  authenticateGoogleOAuth2,
  immediateAuthSupported,
  type OAuth2Token,
} from "#src/util/google_oauth2.js";
import { NullarySignal } from "#src/util/signal.js";

// Exported for testing.
export const LOCAL_STORAGE_KEY = "neuroglancer_roi_store_token_v1";

/**
 * Treat a token as expired this long before it actually expires, so a save
 * that is about to start does not die holding a token that lapses in flight.
 */
export const EXPIRY_MARGIN_MS = 5 * 60 * 1000;

interface StoredToken {
  accessToken: string;
  tokenType: string;
  email: string | undefined;
  scope: string;
  /** Epoch milliseconds at which the token stops being usable. */
  expiresAt: number;
}

function isUsable(token: StoredToken | undefined): token is StoredToken {
  return token !== undefined && token.expiresAt - EXPIRY_MARGIN_MS > Date.now();
}

function toStoredToken(token: OAuth2Token): StoredToken {
  // `expires_in` is a string count of seconds; treat anything unparseable as
  // already expired rather than trusting it indefinitely.
  const seconds = Number.parseInt(token.expiresIn, 10);
  return {
    accessToken: token.accessToken,
    tokenType: token.tokenType,
    email: token.email,
    scope: token.scope,
    expiresAt: Number.isFinite(seconds) ? Date.now() + seconds * 1000 : 0,
  };
}

// localStorage throws in private-browsing modes and when storage is disabled;
// caching the token is an optimisation, so failures degrade to re-signing in.
function readStoredToken(): StoredToken | undefined {
  let raw: string | null;
  try {
    raw = localStorage.getItem(LOCAL_STORAGE_KEY);
  } catch {
    return undefined;
  }
  if (raw === null) return undefined;
  try {
    const parsed = JSON.parse(raw);
    if (
      typeof parsed?.accessToken !== "string" ||
      typeof parsed?.expiresAt !== "number"
    ) {
      return undefined;
    }
    return parsed as StoredToken;
  } catch {
    return undefined;
  }
}

function writeStoredToken(token: StoredToken | undefined) {
  try {
    if (token === undefined) {
      localStorage.removeItem(LOCAL_STORAGE_KEY);
    } else {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(token));
    }
  } catch {
    // Not fatal: the token simply will not survive a reload.
  }
}

/**
 * The identity used to write to the ROI group store.
 *
 * Reads are anonymous, so this is only consulted on save (and on a listing that
 * a private bucket refuses anonymously).  Two implementations back it — a
 * Google OAuth2 token and a reused middleauth/CAVE token — behind one interface
 * so the sign-in chip, the store client and the picker are provider-agnostic.
 */
export interface RoiStoreAuth {
  /** Fires whenever the signed-in identity changes. */
  readonly changed: NullarySignal;
  readonly signedIn: boolean;
  /** A human label for the signed-in identity, if the provider exposes one. */
  readonly email: string | undefined;
  /** A token if one is already held; never prompts. */
  readonly cachedAccessToken: string | undefined;
  getAccessToken(signal?: AbortSignal): Promise<string>;
  signIn(signal?: AbortSignal): Promise<void>;
  signOut(): void;
  invalidate(): void;
}

/**
 * Holds the Google OAuth2 token used to write to the ROI group store.
 */
export class GoogleRoiStoreAuth implements RoiStoreAuth {
  /** Fires whenever the signed-in identity changes. */
  changed = new NullarySignal();

  private token: StoredToken | undefined;
  private pending: Promise<StoredToken> | undefined;
  private pendingController: AbortController | undefined;
  /**
   * Bumped whenever the identity is deliberately discarded.  An attempt that
   * started before the bump must not install its result: otherwise a sign-out
   * during an in-flight sign-in is silently undone when that sign-in lands.
   */
  private generation = 0;
  private expiryTimer: ReturnType<typeof setTimeout> | undefined;

  constructor() {
    const stored = readStoredToken();
    // Drop an expired token at construction so the chip does not show a stale
    // identity that would fail on first use.
    if (isUsable(stored)) {
      this.token = stored;
      this.scheduleExpiry();
    } else if (stored !== undefined) {
      writeStoredToken(undefined);
    }
  }

  get signedIn(): boolean {
    return isUsable(this.token);
  }

  get email(): string | undefined {
    return this.signedIn ? this.token?.email : undefined;
  }

  /**
   * The current token if one is already held, WITHOUT prompting for sign-in.
   *
   * For requests that should be attempted anonymously first and only escalate
   * when a credential happens to be available — listing a bucket that denies
   * anonymous `objects.list`, say. Opening a panel must never raise a popup on
   * its own.
   */
  get cachedAccessToken(): string | undefined {
    return isUsable(this.token) ? this.token.accessToken : undefined;
  }

  /**
   * Returns a usable access token, prompting for sign-in if necessary.
   *
   * Concurrent callers share one sign-in: without this, saving two groups at
   * once would open two popups.
   */
  async getAccessToken(signal?: AbortSignal): Promise<string> {
    if (isUsable(this.token)) return this.token.accessToken;
    if (this.pending === undefined) this.startAuth();
    const token = await raceWithAbort(this.pending!, signal);
    return token.accessToken;
  }

  /**
   * Starts a fresh interactive sign-in, replacing any current identity.
   *
   * Deliberately does NOT join an in-flight attempt.  Two reasons: joining
   * would hand back the very identity the user is trying to switch away from,
   * and a previous attempt can be wedged — `getCredentialsWithStatus` keeps its
   * promise pending after a failed login (it shows a Retry button instead of
   * rejecting), and under cross-origin isolation closing the popup raises no
   * error at all because the severed handle cannot be monitored.  Cancelling
   * and restarting is what makes the sign-in affordance always work.
   */
  async signIn(signal?: AbortSignal): Promise<void> {
    this.cancelPending(new Error("Superseded by a new sign-in"));
    this.clearToken();
    // `select_account` forces the chooser even when Google already has a
    // session, which is the only way to switch account.
    this.startAuth("select_account");
    await raceWithAbort(this.pending!, signal);
  }

  signOut() {
    this.cancelPending(new Error("Signed out"));
    this.clearToken();
  }

  /**
   * Discards the current token after the server rejected it, so the next call
   * re-authenticates.  A token can be refused before `expiresAt` (revoked, or
   * the account lost bucket access), which the expiry check alone cannot catch.
   */
  invalidate() {
    this.cancelPending(new Error("Credentials were rejected"));
    this.clearToken();
  }

  private startAuth(prompt?: "select_account"): void {
    const controller = new AbortController();
    const generation = this.generation;
    this.pendingController = controller;
    const promise = this.authenticate(controller.signal, prompt)
      .then((stored) => {
        if (generation !== this.generation) {
          // Signed out (or invalidated) while this was in flight; do not
          // resurrect an identity the user has already discarded.
          throw new Error("Sign-in was cancelled");
        }
        this.setToken(stored);
        return stored;
      })
      .finally(() => {
        // Only retire this attempt: a newer one may already have replaced it.
        if (this.pending === promise) {
          this.pending = undefined;
          this.pendingController = undefined;
        }
      });
    this.pending = promise;
  }

  private cancelPending(reason: unknown): void {
    // Aborting the signal that `getCredentialsWithStatus` was given is what
    // makes it reject, so the stuck promise settles and its `finally` runs.
    this.pendingController?.abort(reason);
    this.pendingController = undefined;
    this.pending = undefined;
  }

  private setToken(token: StoredToken): void {
    this.token = token;
    writeStoredToken(token);
    this.scheduleExpiry();
    this.changed.dispatch();
  }

  private clearToken(): void {
    ++this.generation;
    this.clearExpiryTimer();
    const had = this.token !== undefined;
    this.token = undefined;
    writeStoredToken(undefined);
    if (had) this.changed.dispatch();
  }

  /**
   * Expiry is otherwise a silent transition: `signedIn` flips to false with
   * nothing to tell the chip, which would go on displaying an identity that no
   * longer works until something happened to re-render it.
   */
  private scheduleExpiry(): void {
    this.clearExpiryTimer();
    if (this.token === undefined) return;
    const remaining = this.token.expiresAt - EXPIRY_MARGIN_MS - Date.now();
    if (remaining <= 0) return;
    this.expiryTimer = setTimeout(() => {
      this.expiryTimer = undefined;
      this.clearToken();
    }, remaining);
  }

  private clearExpiryTimer(): void {
    if (this.expiryTimer !== undefined) {
      clearTimeout(this.expiryTimer);
      this.expiryTimer = undefined;
    }
  }

  private async authenticate(
    signal: AbortSignal,
    prompt?: "select_account",
  ): Promise<StoredToken> {
    const { clientId, scopes } = getRoiStoreConfig();
    // The factory only constructs this class for the google provider, where
    // clientId is required and validated in config.ts.
    if (clientId === undefined) {
      throw new Error("ROI store google provider has no client ID");
    }
    const oauthToken = await getCredentialsWithStatus<OAuth2Token>(
      {
        description: "ROI group store",
        requestDescription: "sign-in",
        // Silent renewal uses a hidden cross-origin iframe, which COEP blocks
        // in the cross-origin isolated build.
        supportsImmediate: prompt === undefined && immediateAuthSupported(),
        get: (getSignal, immediate) =>
          authenticateGoogleOAuth2(
            { clientId, scopes, immediate, authUser: 0, prompt },
            getSignal,
          ),
      },
      signal,
    );
    return toStoredToken(oauthToken);
  }
}

/**
 * Reuses neuroglancer's existing middleauth/CAVE login for the ROI store.
 *
 * Delegates to the shared `MiddleAuthCredentialsProvider` from the default
 * credentials manager, so a user who has already signed in to reach a
 * middleauth-protected data source (or to share state) is signed in here too,
 * with no separate OAuth client.
 *
 * Constraints, both enforced by the deployment rather than here: the token is a
 * CAVE bearer token, so `endpoint` must accept it (not raw GCS); and the login
 * popup receives its response from the CAVE server's page via `window.opener`,
 * which COOP severs, so this cannot complete in the pyodide build.
 */
export class MiddleAuthRoiStoreAuth implements RoiStoreAuth {
  changed = new NullarySignal();

  private provider: CredentialsProvider<MiddleAuthToken>;
  private current: CredentialsWithGeneration<MiddleAuthToken> | undefined;
  /** Creds to present as stale on the next get, to force a refresh. */
  private invalidated: CredentialsWithGeneration<MiddleAuthToken> | undefined;
  private pending: Promise<string> | undefined;

  constructor(private authServer: string) {
    this.provider =
      getDefaultCredentialsManager().getCredentialsProvider<MiddleAuthToken>(
        "middleauth",
        authServer,
      );
  }

  private get storageKey(): string {
    // Mirrors the key the middleauth provider persists under, so `signedIn`
    // can be answered synchronously without driving the provider.
    return `auth_token_v2_${this.authServer}`;
  }

  get cachedAccessToken(): string | undefined {
    if (this.current !== undefined) return this.current.credentials.accessToken;
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (raw === null) return undefined;
      const parsed = JSON.parse(raw);
      return typeof parsed?.accessToken === "string"
        ? parsed.accessToken
        : undefined;
    } catch {
      return undefined;
    }
  }

  get signedIn(): boolean {
    return this.cachedAccessToken !== undefined;
  }

  // Middleauth tokens carry no email; the chip falls back to a generic label.
  get email(): string | undefined {
    return undefined;
  }

  async getAccessToken(signal?: AbortSignal): Promise<string> {
    const cached = this.cachedAccessToken;
    if (cached !== undefined && this.invalidated === undefined) return cached;
    if (this.pending === undefined) {
      const invalid = this.invalidated;
      this.invalidated = undefined;
      this.pending = this.provider
        .get(invalid, signal ? { signal } : undefined)
        .then((result) => {
          this.current = result;
          this.changed.dispatch();
          return result.credentials.accessToken;
        })
        .finally(() => {
          this.pending = undefined;
        });
    }
    return raceWithAbort(this.pending, signal);
  }

  async signIn(signal?: AbortSignal): Promise<void> {
    this.signOut();
    await this.getAccessToken(signal);
  }

  signOut(): void {
    this.invalidated = this.current;
    this.current = undefined;
    try {
      // Clear the persisted token so a reload does not resurrect the identity.
      localStorage.removeItem(this.storageKey);
    } catch {
      // Not fatal.
    }
    this.changed.dispatch();
  }

  invalidate(): void {
    this.signOut();
  }
}

let singleton: RoiStoreAuth | undefined;

/**
 * The process-wide ROI store identity, per the configured provider.  A
 * singleton so the sign-in chip and every save action share one token and one
 * popup.
 */
export function getRoiStoreAuth(): RoiStoreAuth {
  if (singleton === undefined) {
    const config = getRoiStoreConfig();
    singleton =
      config.provider === "middleauth"
        ? new MiddleAuthRoiStoreAuth(config.authServer!)
        : new GoogleRoiStoreAuth();
  }
  return singleton;
}
