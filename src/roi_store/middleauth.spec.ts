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
 * @file The synchronous surface of the middleauth-backed ROI store identity.
 *
 * Covers what does not require a login round trip: reading an existing
 * middleauth token from localStorage, and signing out. The interactive get is
 * exercised by hand, since it drives the CAVE server's popup.
 */

// Register the "middleauth" credentials provider so the default manager can
// resolve it during construction.
import "#src/kvstore/middleauth/register_credentials_provider.js";
import { beforeEach, describe, it, expect } from "vitest";
import { MiddleAuthRoiStoreAuth } from "#src/roi_store/credentials.js";

const SERVER = "https://global.daf-apis.com";
const KEY = `auth_token_v2_${SERVER}`;

function seedToken(accessToken: string) {
  localStorage.setItem(
    KEY,
    JSON.stringify({
      tokenType: "Bearer",
      accessToken,
      url: SERVER,
      app_urls: [],
    }),
  );
}

describe("MiddleAuthRoiStoreAuth", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("is signed in when the shared middleauth token is present", () => {
    // The whole point: reuse the login the user already has, keyed exactly the
    // way MiddleAuthCredentialsProvider persists it.
    seedToken("cave-token");
    const auth = new MiddleAuthRoiStoreAuth(SERVER);
    expect(auth.signedIn).toBe(true);
    expect(auth.cachedAccessToken).toEqual("cave-token");
  });

  it("is signed out when no token is present", () => {
    const auth = new MiddleAuthRoiStoreAuth(SERVER);
    expect(auth.signedIn).toBe(false);
    expect(auth.cachedAccessToken).toBeUndefined();
  });

  it("exposes no email (CAVE tokens carry none)", () => {
    seedToken("cave-token");
    expect(new MiddleAuthRoiStoreAuth(SERVER).email).toBeUndefined();
  });

  it("signOut clears the shared token and notifies", () => {
    seedToken("cave-token");
    const auth = new MiddleAuthRoiStoreAuth(SERVER);
    let notified = 0;
    auth.changed.add(() => {
      ++notified;
    });
    auth.signOut();
    expect(auth.signedIn).toBe(false);
    expect(localStorage.getItem(KEY)).toBeNull();
    expect(notified).toEqual(1);
  });

  it("ignores a malformed persisted token", () => {
    localStorage.setItem(KEY, "not json");
    expect(new MiddleAuthRoiStoreAuth(SERVER).signedIn).toBe(false);
  });
});
