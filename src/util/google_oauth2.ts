/**
 * @license
 * Copyright 2016 Google Inc.
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

import {
  CredentialsProvider,
  makeCredentialsGetter,
} from "#src/credentials_provider/index.js";
import {
  getCredentialsWithStatus,
  monitorAuthPopupWindow,
} from "#src/credentials_provider/interactive_credentials_provider.js";
import { raceWithAbort } from "#src/util/abort.js";
import { removeFromParent } from "#src/util/dom.js";
import {
  verifyObject,
  verifyObjectProperty,
  verifyString,
} from "#src/util/json.js";
import { ProgressSpan } from "#src/util/progress_listener.js";
import { getRandomHexString } from "#src/util/random.js";

export const EMAIL_SCOPE = "email";
export const OPENID_SCOPE = "openid";

export const AUTH_SERVER = "https://accounts.google.com/o/oauth2/v2/auth";

/**
 * Prefix of the BroadcastChannel over which the redirect page also delivers
 * the OAuth2 response.
 *
 * When the initiating document is cross-origin isolated, `COOP: same-origin`
 * places the popup in a separate browsing context group: the popup's
 * `window.opener` is null, so it cannot `postMessage` the response back.
 * BroadcastChannel is same-origin by construction but crosses browsing context
 * groups, so it still reaches the initiating window.
 *
 * Keep in sync with `google_oauth2_redirect.html`, which cannot import this.
 */
export const AUTH_RESPONSE_CHANNEL_PREFIX = "neuroglancer-oauth2-";

/**
 * The channel for one authentication attempt.
 *
 * Scoped by the attempt's random `state` rather than being a single well-known
 * name, because a BroadcastChannel is readable by ANY same-origin context and
 * the payload is a bearer token.  A listener must know the channel name up
 * front — there is no enumeration API — so deriving it from a value that only
 * the initiating window and the redirect page possess keeps the token away
 * from unrelated same-origin code.  That matters most in the Pyodide build,
 * where user-supplied Python runs in the same origin with `js` access.
 */
export function authResponseChannelName(state: string): string {
  return AUTH_RESPONSE_CHANNEL_PREFIX + state;
}

/** Whether the document is cross-origin isolated, safe where the global is absent. */
function isCrossOriginIsolated(): boolean {
  return globalThis.crossOriginIsolated === true;
}

/**
 * OAuth2 Token
 */
export interface OAuth2Token {
  accessToken: string;
  expiresIn: string;
  tokenType: string;
  scope: string;
  email: string | undefined;
}

function extractEmailFromIdToken(idToken: string): string {
  const idTokenParts = idToken.split(".");
  try {
    if (idTokenParts.length !== 3) throw new Error("Invalid JWT format");
    const decoded = atob(idTokenParts[1]);
    const parsed = JSON.parse(decoded);
    verifyObject(parsed);
    return verifyObjectProperty(parsed, "email", verifyString);
  } catch (e) {
    throw new Error(`Failed to decode id token: ${e.message}`);
  }
}

/**
 * Parses an OAuth2 response, or returns `undefined` if it does not belong to
 * the attempt identified by `state`.
 *
 * A mismatched state is ignored rather than treated as an error: the response
 * is simply not ours (a stale broadcast from an earlier attempt, or another
 * tab's login), and rejecting on it would let an unrelated message abort a
 * legitimate flow.  Only a matching state is ever accepted, so this remains
 * the CSRF check.
 */
// Exported for testing.
export function parseAuthResponse(
  data: unknown,
  state: string,
): OAuth2Token | undefined {
  // Everything before the state match must be non-throwing: an unrelated
  // same-origin message (an array, a bare object, another library's postMessage)
  // must be ignored, not turned into a rejection that kills a live sign-in.
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return undefined;
  }
  const obj = data as Record<string, unknown>;
  if (typeof obj.state !== "string" || obj.state !== state) return undefined;
  // Past this point the response IS ours, so a malformed one is a real error.
  verifyObject(obj);
  const idToken = verifyObjectProperty(obj, "id_token", verifyString);
  return {
    accessToken: verifyObjectProperty(obj, "access_token", verifyString),
    tokenType: verifyObjectProperty(obj, "token_type", verifyString),
    expiresIn: verifyObjectProperty(obj, "expires_in", verifyString),
    scope: verifyObjectProperty(obj, "scope", verifyString),
    email: extractEmailFromIdToken(idToken),
  };
}

// Note: `signal` is guaranteed to be aborted once the operation completes.
//
// Listens on both delivery transports.  `source` is undefined when the popup
// handle has been severed by COOP, in which case only the broadcast can arrive
// and `state` alone ties the response to this attempt.
// Exported for testing.
export function waitForAuthResponseMessage(
  source: Window | undefined,
  state: string,
  signal: AbortSignal,
): Promise<OAuth2Token> {
  return new Promise((resolve, reject) => {
    const handleResponse = (data: unknown) => {
      let token: OAuth2Token | undefined;
      try {
        token = parseAuthResponse(data, state);
      } catch (parseError) {
        reject(
          new Error(
            `Received unexpected authentication response: ${parseError.message}`,
          ),
        );
        console.error("Response received: ", data);
        return;
      }
      if (token !== undefined) {
        resolve(token);
      }
    };

    window.addEventListener(
      "message",
      (event: MessageEvent) => {
        if (event.origin !== location.origin) {
          return;
        }
        if (source !== undefined && event.source !== source) return;
        handleResponse(event.data);
      },
      { signal: signal },
    );

    if (signal.aborted) {
      // Nothing can arrive, so settle rather than leaving the caller waiting on
      // a promise that never resolves.
      reject(signal.reason);
      return;
    }
    // BroadcastChannel only ever delivers same-origin messages, so unlike the
    // `message` path there is no origin to check here.  The channel name is
    // derived from `state`, so only this attempt's participants know it.
    const channel = new BroadcastChannel(authResponseChannelName(state));
    channel.addEventListener(
      "message",
      (event: MessageEvent) => handleResponse(event.data),
      { signal: signal },
    );
    signal.addEventListener("abort", () => channel.close(), { once: true });
  });
}

function makeAuthRequestUrl(options: {
  clientId: string;
  scopes: string[];
  nonce?: string;
  approvalPrompt?: "force" | "auto";
  state?: string;
  loginHint?: string;
  authUser?: number;
  includeGrantedScopes?: boolean;
  immediate?: boolean;
  /**
   * `select_account` shows the account chooser even when Google already has a
   * session, which is the only way a signed-in user can switch identity.
   */
  prompt?: "select_account" | "consent";
}) {
  let url = `${AUTH_SERVER}?client_id=${encodeURIComponent(options.clientId)}`;
  const redirectUri = new URL("./google_oauth2_redirect.html", import.meta.url)
    .href;
  url += `&redirect_uri=${redirectUri}`;
  let responseType = "token";
  const { scopes } = options;
  if (scopes.includes("email") && scopes.includes("openid")) {
    responseType = "token%20id_token";
  }
  url += `&response_type=${responseType}`;
  if (options.includeGrantedScopes === true) {
    url += "&include_granted_scopes=true";
  }
  url += `&scope=${encodeURIComponent(scopes.join(" "))}`;
  if (options.state) {
    url += `&state=${options.state}`;
  }
  if (options.loginHint) {
    url += `&login_hint=${encodeURIComponent(options.loginHint)}`;
  }
  if (options.immediate) {
    url += "&immediate=true";
  }
  if (options.nonce !== undefined) {
    url += `&nonce=${options.nonce}`;
  }
  if (options.authUser !== undefined) {
    url += `&authuser=${options.authUser}`;
  }
  if (options.prompt !== undefined) {
    url += `&prompt=${encodeURIComponent(options.prompt)}`;
  }
  return url;
}

function createAuthIframe(
  url: string,
  abortController: AbortController,
): Window {
  const iframe = document.createElement("iframe");
  iframe.src = url;
  iframe.style.display = "none";
  iframe.addEventListener(
    "load",
    () => {
      if (iframe.contentDocument == null) {
        // Error received
        abortController.abort(new Error("Immediate authentication failed"));
      }
    },
    { signal: abortController.signal },
  );
  document.body.appendChild(iframe);
  abortController.signal.addEventListener("abort", () => {
    removeFromParent(iframe);
  });
  return iframe.contentWindow!;
}

/**
 * Obtain a Google OAuth2 authentication token.
 * @return A Promise that resolves to an authentication token.
 */
export async function authenticateGoogleOAuth2(
  options: {
    clientId: string;
    scopes: string[];
    approvalPrompt?: "force" | "auto";
    loginHint?: string;
    immediate?: boolean;
    authUser?: number;
    prompt?: "select_account" | "consent";
  },
  signal: AbortSignal,
) {
  const state = getRandomHexString();
  const nonce = getRandomHexString();
  const url = makeAuthRequestUrl({
    state,
    nonce,
    clientId: options.clientId,
    scopes: options.scopes,
    approvalPrompt: options.approvalPrompt,
    loginHint: options.loginHint,
    immediate: options.immediate,
    authUser: options.authUser,
    prompt: options.prompt,
  });
  const abortController = new AbortController();
  signal = AbortSignal.any([abortController.signal, signal]);
  try {
    let source: Window | undefined;
    if (options.immediate) {
      if (!immediateAuthSupported()) {
        throw new Error(
          "Silent authentication is unavailable in a cross-origin isolated context",
        );
      }
      source = createAuthIframe(url, abortController);
    } else {
      const newWindow = open(url);
      if (newWindow === null) {
        throw new Error("Failed to create authentication popup window");
      }
      if (isCrossOriginIsolated()) {
        // The popup handle has been severed by `COOP: same-origin`.  It cannot
        // be monitored: `closed` reads true within a few hundred milliseconds
        // of the popup navigating to the auth server, while the popup is still
        // very much open, so `monitorAuthPopupWindow` would abort the flow
        // while the user is still logging in.  Leave `source` undefined; the
        // response arrives over the broadcast channel instead.
        //
        // Nothing here can close the popup either — `close()` on a severed
        // handle is a silent no-op, not a throw — so the redirect page closes
        // itself once it has delivered the response.
        source = undefined;
      } else {
        monitorAuthPopupWindow(newWindow, abortController);
        source = newWindow;
      }
    }
    return await raceWithAbort(
      waitForAuthResponseMessage(source, state, abortController.signal),
      signal,
    );
  } finally {
    abortController.abort();
  }
}

/**
 * Whether the hidden-iframe "immediate" (silent re-authentication) mode can
 * work.  `COEP: require-corp` blocks the cross-origin auth iframe outright, so
 * in a cross-origin isolated context a token can only be obtained via a popup.
 */
export function immediateAuthSupported(): boolean {
  return !isCrossOriginIsolated();
}

export class GoogleOAuth2CredentialsProvider extends CredentialsProvider<OAuth2Token> {
  constructor(
    public options: { clientId: string; scopes: string[]; description: string },
  ) {
    super();
  }

  get = makeCredentialsGetter(async (options) => {
    using _span = new ProgressSpan(options.progressListener, {
      message: `Requesting ${this.options.description} OAuth2 access token`,
    });
    return await getCredentialsWithStatus(
      {
        description: this.options.description,
        supportsImmediate: immediateAuthSupported(),
        get: (signal, immediate) =>
          authenticateGoogleOAuth2(
            {
              clientId: this.options.clientId,
              scopes: this.options.scopes,
              immediate: immediate,
              authUser: 0,
            },
            signal,
          ),
      },
      options.signal,
    );
  });
}
