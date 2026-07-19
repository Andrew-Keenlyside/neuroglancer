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
 * @file Tests the OAuth2 response transports.
 *
 * A browser test rather than a jsdom one because BroadcastChannel delivery
 * semantics are the thing under test, and they need a real browser.
 *
 * The broadcast path exists because `COOP: same-origin` severs the popup's
 * `window.opener`, so in the cross-origin isolated build postMessage cannot
 * deliver the response at all.  These headers cannot be set on the vitest
 * browser runner, so what is verified here is the transport wiring; that it
 * genuinely survives cross-origin isolation was established separately against
 * a COOP/COEP server.
 */

import { describe, it, expect } from "vitest";
import {
  authResponseChannelName,
  parseAuthResponse,
  waitForAuthResponseMessage,
} from "#src/util/google_oauth2.js";

function makeIdToken(email: string): string {
  const payload = btoa(JSON.stringify({ email }));
  return `header.${payload}.signature`;
}

function makeResponse(state: string, email = "test@example.com") {
  return {
    access_token: "test-access-token",
    token_type: "Bearer",
    expires_in: "3599",
    scope: "openid email",
    state,
    id_token: makeIdToken(email),
  };
}

describe("parseAuthResponse", () => {
  it("parses a well-formed response", () => {
    const token = parseAuthResponse(makeResponse("s1"), "s1");
    expect(token).toBeDefined();
    expect(token!.accessToken).toEqual("test-access-token");
    expect(token!.tokenType).toEqual("Bearer");
    expect(token!.expiresIn).toEqual("3599");
    expect(token!.email).toEqual("test@example.com");
  });

  it("ignores a response for a different attempt", () => {
    // Must not throw: a mismatched state means the message is simply not ours,
    // and rejecting would let an unrelated broadcast kill a live sign-in.
    expect(parseAuthResponse(makeResponse("other"), "s1")).toBeUndefined();
  });

  it("ignores messages that are not auth responses", () => {
    // None of these may throw: an unrelated same-origin message must not be
    // able to abort a live sign-in.
    expect(parseAuthResponse(undefined, "s1")).toBeUndefined();
    expect(parseAuthResponse(null, "s1")).toBeUndefined();
    expect(parseAuthResponse("a string", "s1")).toBeUndefined();
    expect(parseAuthResponse(42, "s1")).toBeUndefined();
    expect(parseAuthResponse({ unrelated: true }, "s1")).toBeUndefined();
    expect(parseAuthResponse([], "s1")).toBeUndefined();
    expect(parseAuthResponse([1, 2, 3], "s1")).toBeUndefined();
    expect(parseAuthResponse({ state: 42 }, "s1")).toBeUndefined();
  });

  it("throws when the matching response is malformed", () => {
    // State matched, so this IS our response — a missing access token is a
    // real error and must not be silently swallowed.
    expect(() => parseAuthResponse({ state: "s1" }, "s1")).toThrow(/id_token/);
  });
});

describe("waitForAuthResponseMessage", () => {
  it("resolves from a broadcast when the popup handle is severed", async () => {
    const controller = new AbortController();
    const promise = waitForAuthResponseMessage(
      /*source=*/ undefined,
      "s1",
      controller.signal,
    );
    const channel = new BroadcastChannel(authResponseChannelName("s1"));
    channel.postMessage(makeResponse("s1"));
    const token = await promise;
    channel.close();
    controller.abort();
    expect(token.accessToken).toEqual("test-access-token");
    expect(token.email).toEqual("test@example.com");
  });

  it("does not listen on another attempt's channel", async () => {
    // The channel name embeds the attempt's random state, so a token meant for
    // a different attempt is never even delivered here — the payload is a
    // bearer token and any same-origin context can open a channel it can name.
    const controller = new AbortController();
    let settled = false;
    waitForAuthResponseMessage(undefined, "s1", controller.signal).then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );
    const other = new BroadcastChannel(authResponseChannelName("s2"));
    other.postMessage(makeResponse("s2"));
    await new Promise((resolve) => setTimeout(resolve, 250));
    other.close();
    expect(settled).toBe(false);
    controller.abort();
  });

  it("ignores a stale payload on its own channel", async () => {
    // Defence in depth behind the channel scoping: a replayed or mismatched
    // payload must be ignored rather than rejecting a live sign-in.
    const controller = new AbortController();
    let settled = false;
    const promise = waitForAuthResponseMessage(
      undefined,
      "s1",
      controller.signal,
    ).then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );
    const channel = new BroadcastChannel(authResponseChannelName("s1"));
    channel.postMessage(makeResponse("a-different-attempt"));
    channel.postMessage([1, 2, 3]);
    channel.postMessage({ unrelated: true });
    await new Promise((resolve) => setTimeout(resolve, 250));
    expect(settled).toBe(false);

    // The right one still gets through afterwards.
    channel.postMessage(makeResponse("s1"));
    await promise;
    channel.close();
    controller.abort();
    expect(settled).toBe(true);
  });

  it("still accepts a same-window postMessage", async () => {
    // Regression guard for the non-isolated builds (brainmaps, boss), where
    // the opener link is intact and postMessage remains the delivery path.
    const controller = new AbortController();
    const promise = waitForAuthResponseMessage(window, "s1", controller.signal);
    window.postMessage(makeResponse("s1"), location.origin);
    const token = await promise;
    controller.abort();
    expect(token.accessToken).toEqual("test-access-token");
  });

  it("rejects rather than hanging when the signal is already aborted", async () => {
    // Nothing can arrive on an aborted attempt, so it must settle: a caller
    // awaiting it outside raceWithAbort would otherwise wait forever.
    const controller = new AbortController();
    controller.abort(new Error("already gone"));
    await expect(
      waitForAuthResponseMessage(undefined, "s1", controller.signal),
    ).rejects.toThrow(/already gone/);
  });
});
