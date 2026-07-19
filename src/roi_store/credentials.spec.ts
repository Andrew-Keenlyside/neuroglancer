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
 * @file Token caching for the ROI group store sign-in.
 *
 * Covers only what does not require an OAuth round trip: how a cached token is
 * accepted, rejected or repaired at construction.  Getting a token in the
 * first place is interactive and is exercised by hand.
 */

import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import {
  EXPIRY_MARGIN_MS,
  LOCAL_STORAGE_KEY,
  GoogleRoiStoreAuth,
} from "#src/roi_store/credentials.js";

function seed(value: unknown) {
  localStorage.setItem(
    LOCAL_STORAGE_KEY,
    typeof value === "string" ? value : JSON.stringify(value),
  );
}

function validToken(overrides: Record<string, unknown> = {}) {
  return {
    accessToken: "cached-token",
    tokenType: "Bearer",
    email: "test@example.com",
    scope: "openid email",
    // Comfortably beyond the early-expiry margin.
    expiresAt: Date.now() + EXPIRY_MARGIN_MS + 60_000,
    ...overrides,
  };
}

describe("GoogleRoiStoreAuth token cache", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("restores a valid cached token", () => {
    seed(validToken());
    const auth = new GoogleRoiStoreAuth();
    expect(auth.signedIn).toBe(true);
    expect(auth.email).toEqual("test@example.com");
  });

  it("starts signed out when nothing is cached", () => {
    const auth = new GoogleRoiStoreAuth();
    expect(auth.signedIn).toBe(false);
    expect(auth.email).toBeUndefined();
  });

  it("discards an expired token and clears it from storage", () => {
    seed(validToken({ expiresAt: Date.now() - 1000 }));
    const auth = new GoogleRoiStoreAuth();
    expect(auth.signedIn).toBe(false);
    expect(localStorage.getItem(LOCAL_STORAGE_KEY)).toBeNull();
  });

  it("treats a token inside the expiry margin as unusable", () => {
    // Not yet expired, but too close to risk starting a save with it.
    seed(validToken({ expiresAt: Date.now() + EXPIRY_MARGIN_MS - 1000 }));
    const auth = new GoogleRoiStoreAuth();
    expect(auth.signedIn).toBe(false);
  });

  it("survives corrupt or partial cached data", () => {
    for (const bad of [
      "not json at all",
      JSON.stringify({ accessToken: "x" }), // no expiresAt
      JSON.stringify({ expiresAt: Date.now() + 3_600_000 }), // no accessToken
      JSON.stringify({ accessToken: 5, expiresAt: "soon" }), // wrong types
    ]) {
      localStorage.clear();
      seed(bad);
      const auth = new GoogleRoiStoreAuth();
      expect(auth.signedIn).toBe(false);
    }
  });

  it("signOut clears storage and notifies", () => {
    seed(validToken());
    const auth = new GoogleRoiStoreAuth();
    let notified = 0;
    auth.changed.add(() => {
      ++notified;
    });
    auth.signOut();
    expect(auth.signedIn).toBe(false);
    expect(auth.email).toBeUndefined();
    expect(localStorage.getItem(LOCAL_STORAGE_KEY)).toBeNull();
    expect(notified).toEqual(1);
  });

  it("signOut on an already signed-out instance does not notify", () => {
    const auth = new GoogleRoiStoreAuth();
    let notified = 0;
    auth.changed.add(() => {
      ++notified;
    });
    auth.signOut();
    expect(notified).toEqual(0);
  });

  it("invalidate drops a token the server has rejected", () => {
    // Expiry alone cannot catch revocation or a lost bucket grant.
    seed(validToken());
    const auth = new GoogleRoiStoreAuth();
    expect(auth.signedIn).toBe(true);
    auth.invalidate();
    expect(auth.signedIn).toBe(false);
    expect(localStorage.getItem(LOCAL_STORAGE_KEY)).toBeNull();
  });

  it("does not reuse a failed sign-in attempt", async () => {
    // The store is unconfigured in tests, so authentication always fails. What
    // matters is that the failed attempt is retired: a memoised in-flight
    // promise that is never cleared would leave every later call awaiting a
    // dead sign-in with no way to recover.
    const auth = new GoogleRoiStoreAuth();
    await expect(auth.getAccessToken()).rejects.toThrow(/not configured/);
    await expect(auth.getAccessToken()).rejects.toThrow(/not configured/);
    await expect(auth.signIn()).rejects.toThrow(/not configured/);
  });
});

describe("GoogleRoiStoreAuth expiry", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("notifies when a cached token reaches the expiry margin", () => {
    // Otherwise expiry is silent: `signedIn` flips but the chip keeps showing
    // an identity that no longer works.
    seed(validToken({ expiresAt: Date.now() + EXPIRY_MARGIN_MS + 1000 }));
    const auth = new GoogleRoiStoreAuth();
    expect(auth.signedIn).toBe(true);

    let notified = 0;
    auth.changed.add(() => {
      ++notified;
    });

    vi.advanceTimersByTime(999);
    expect(auth.signedIn).toBe(true);
    expect(notified).toEqual(0);

    vi.advanceTimersByTime(2);
    expect(auth.signedIn).toBe(false);
    expect(auth.email).toBeUndefined();
    expect(notified).toEqual(1);
    expect(localStorage.getItem(LOCAL_STORAGE_KEY)).toBeNull();
  });

  it("does not fire the expiry timer after signOut", () => {
    seed(validToken({ expiresAt: Date.now() + EXPIRY_MARGIN_MS + 1000 }));
    const auth = new GoogleRoiStoreAuth();
    let notified = 0;
    auth.changed.add(() => {
      ++notified;
    });
    auth.signOut();
    expect(notified).toEqual(1);
    // The pending timer must have been cancelled, not left to fire a second
    // spurious notification.
    vi.advanceTimersByTime(5000);
    expect(notified).toEqual(1);
  });
});
