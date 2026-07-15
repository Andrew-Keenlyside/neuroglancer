/**
 * @license
 * Copyright 2024 Google Inc.
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
 * @file Service Worker for the Pyodide (browser-only) neuroglancer deployment.
 *
 * Intercepts HTTP requests from the neuroglancer JS frontend that would
 * normally go to the Python Tornado server, and forwards them to the Pyodide
 * Web Worker via a MessageChannel port.  Also maintains Server-Sent Event
 * (SSE) streams for state synchronization.
 *
 * Communication protocol (with pyodide_worker.ts via MessageChannel):
 *   SW → Worker:  { type: 'request', requestId, url, method, body: ArrayBuffer | null }
 *   Worker → SW:  { type: 'response', requestId, status, contentType, body: ArrayBuffer }
 *   Worker → SW:  { type: 'sse_event', token, data: string }
 */

/// <reference lib="webworker" />

declare const self: ServiceWorkerGlobalScope;

// Port connected to the Pyodide Web Worker, set up by the main thread.
let pyodidePort: MessagePort | null = null;

// Pending fetch requests awaiting a response from Pyodide.
const pendingRequests = new Map<
  string,
  (result: { status: number; contentType: string; body: ArrayBuffer }) => void
>();

// Open SSE stream controllers keyed by viewer token.
const sseStreams = new Map<string, ReadableStreamDefaultController<Uint8Array>>();

const encoder = new TextEncoder();
let requestCounter = 0;

// ---------------------------------------------------------------------------
// Service Worker lifecycle
// ---------------------------------------------------------------------------

self.addEventListener("install", () => {
  // Activate immediately without waiting for existing clients to unload.
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  // Take control of all clients immediately.
  event.waitUntil(self.clients.claim());
});

// ---------------------------------------------------------------------------
// Main-thread → Service Worker initialisation message
// ---------------------------------------------------------------------------

self.addEventListener("message", (event) => {
  if (event.data?.type === "init") {
    pyodidePort = event.ports[0];
    pyodidePort.onmessage = handlePyodideMessage;
    pyodidePort.start();
  }
});

function handlePyodideMessage(event: MessageEvent) {
  const { type } = event.data;
  if (type === "response") {
    const { requestId, status, contentType, body } = event.data as {
      requestId: string;
      status: number;
      contentType: string;
      body: ArrayBuffer;
    };
    const resolve = pendingRequests.get(requestId);
    if (resolve) {
      pendingRequests.delete(requestId);
      resolve({ status, contentType, body });
    }
  } else if (type === "sse_event") {
    const { token, data } = event.data as { token: string; data: string };
    const controller = sseStreams.get(token);
    if (controller) {
      try {
        controller.enqueue(encoder.encode(`data: ${data}\n\n`));
      } catch {
        // Stream may have been cancelled.
        sseStreams.delete(token);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Fetch interception
// ---------------------------------------------------------------------------

/** Returns true if this request should be handled by Pyodide. */
function shouldIntercept(pathname: string): boolean {
  return (
    pathname.startsWith("/neuroglancer/") ||
    pathname.startsWith("/events/") ||
    pathname.startsWith("/state/") ||
    pathname.startsWith("/action/") ||
    pathname.startsWith("/volume_response/") ||
    pathname.startsWith("/credentials/")
  );
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (!shouldIntercept(url.pathname)) {
    // Let the browser handle static assets normally.
    return;
  }

  if (url.pathname.startsWith("/events/")) {
    event.respondWith(handleSSERequest(url, event.request));
  } else {
    event.respondWith(handleAPIRequest(event.request));
  }
});

// ---------------------------------------------------------------------------
// SSE (EventSource) handler
// ---------------------------------------------------------------------------

async function handleSSERequest(
  url: URL,
  request: Request,
): Promise<Response> {
  const token = url.pathname.split("/")[2];

  // Create a ReadableStream that stays open and receives events pushed by
  // Pyodide via the pyodidePort message handler above.
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
      sseStreams.set(token, c);
      // Flush an initial comment so the browser registers the connection.
      c.enqueue(encoder.encode(": connected\n\n"));
    },
    cancel() {
      sseStreams.delete(token);
    },
  });

  // Notify Pyodide that a client connected so it can push the current state.
  // We intentionally do not await this — the SSE stream is already open.
  forwardToPyodide(request.url, "GET", null).catch(() => {});

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

// ---------------------------------------------------------------------------
// Generic API request handler
// ---------------------------------------------------------------------------

async function handleAPIRequest(request: Request): Promise<Response> {
  const body =
    request.method !== "GET" ? await request.arrayBuffer() : null;
  try {
    const result = await forwardToPyodide(request.url, request.method, body);
    return new Response(result.body, {
      status: result.status,
      headers: { "Content-Type": result.contentType },
    });
  } catch (e) {
    return new Response(String(e), { status: 503 });
  }
}

// ---------------------------------------------------------------------------
// Forward a request to the Pyodide Web Worker and await its response
// ---------------------------------------------------------------------------

function forwardToPyodide(
  url: string,
  method: string,
  body: ArrayBuffer | null,
): Promise<{ status: number; contentType: string; body: ArrayBuffer }> {
  return new Promise((resolve, reject) => {
    if (!pyodidePort) {
      reject(new Error("Pyodide port not initialised"));
      return;
    }
    const requestId = String(requestCounter++);
    pendingRequests.set(requestId, resolve);
    const transferables: Transferable[] = body ? [body] : [];
    pyodidePort.postMessage(
      { type: "request", requestId, url, method, body },
      transferables,
    );
  });
}
