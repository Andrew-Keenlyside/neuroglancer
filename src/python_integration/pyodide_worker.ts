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
 * @file Dedicated Web Worker that hosts the Pyodide Python runtime.
 *
 * Startup sequence (triggered by the main thread):
 *   1. Receive an `init` message containing:
 *        - `port`: MessageChannel port connected to the Service Worker
 *        - `neuroglancerZipUrl`: URL of the bundled neuroglancer Python package
 *        - `userScript`: Python source to execute after initialisation
 *   2. Load Pyodide and install numpy / scipy / pillow.
 *   3. Unpack the neuroglancer Python zip into the Pyodide virtual filesystem.
 *   4. Set up the JS↔Python bridge:
 *        - `globalThis.pyodide_push_sse(token, data)` — called by Python to
 *          push SSE events; forwarded to the Service Worker port.
 *   5. Run the user Python script.
 *   6. Post `{ type: 'ready' }` back to the main thread.
 *
 * Request handling (once ready):
 *   Service Worker → Worker: { type: 'request', requestId, url, method, body }
 *   Python `pyodide_handle_request(url, method, body)` is called synchronously.
 *   Worker → Service Worker: { type: 'response', requestId, status, contentType, body }
 */

/// <reference lib="webworker" />

// Pyodide is loaded via importScripts — declare its global.
declare function loadPyodide(options?: Record<string, unknown>): Promise<any>;

// Port to the Service Worker, established during init.
let swPort: MessagePort | null = null;
// Reference to the loaded Pyodide instance.
let pyodide: any = null;
// User Python script, stored between init and run phases.
let pendingUserScript: string | null = null;

// ---------------------------------------------------------------------------
// Message handler — two-phase protocol:
//   init  → load Pyodide + packages → post packages_ready
//   run   → (optionally) set starting URL → run script → post ready
// ---------------------------------------------------------------------------

self.addEventListener("message", async (event) => {
  const msgType = event.data?.type;

  if (msgType === "init") {
    const {
      port,
      neuroglancerZipUrl,
      userScript,
      pyodideIndexUrl,
    }: {
      port: MessagePort;
      neuroglancerZipUrl: string;
      userScript: string;
      pyodideIndexUrl: string;
    } = event.data;

    pendingUserScript = userScript;
    swPort = port;
    swPort.onmessage = handleSwRequest;
    swPort.start();

    // Set up the SSE bridge before Python is loaded so Python can call it
    // immediately in response to setup code.
    (globalThis as any).pyodide_push_sse = (token: string, data: string) => {
      swPort!.postMessage({ type: "sse_event", token, data });
    };

    try {
      await setupPyodide(pyodideIndexUrl, neuroglancerZipUrl);
      self.postMessage({ type: "packages_ready" });
    } catch (e) {
      self.postMessage({ type: "error", message: String(e) });
    }

  } else if (msgType === "run") {
    const { startingUrl }: { startingUrl: string | null } = event.data;

    // Expose the optional starting URL to Python before running the script.
    if (startingUrl) {
      (globalThis as any).neuroglancer_starting_url = startingUrl;
    }

    try {
      self.postMessage({ type: "progress", message: "Running Python setup…" });
      await pyodide.runPythonAsync(pendingUserScript!);
      self.postMessage({ type: "ready" });
    } catch (e) {
      self.postMessage({ type: "error", message: String(e) });
    }
  }
});

async function setupPyodide(
  pyodideIndexUrl: string,
  neuroglancerZipUrl: string,
) {
  // Load Pyodide runtime.
  importScripts(pyodideIndexUrl);
  pyodide = await loadPyodide({ indexURL: pyodideIndexUrl.replace(/[^/]*$/, "") });

  self.postMessage({ type: "progress", message: "Loading Python packages…" });

  // Install scientific packages.
  await pyodide.loadPackage(["numpy", "scipy", "pillow", "micropip"]);

  self.postMessage({ type: "progress", message: "Loading neuroglancer Python package…" });

  // Unpack the bundled neuroglancer Python zip into the VFS.
  const zipResp = await fetch(neuroglancerZipUrl);
  if (!zipResp.ok) {
    throw new Error(`Failed to fetch neuroglancer zip: HTTP ${zipResp.status} from ${neuroglancerZipUrl}`);
  }
  const zipBuf = await zipResp.arrayBuffer();
  // unpackArchive requires a TypedArray, not a raw ArrayBuffer.
  // Use Python's site module to get the correct versioned site-packages path.
  const sitePackages: string = pyodide.runPython(
    "import site; site.getsitepackages()[0]"
  );
  pyodide.unpackArchive(new Uint8Array(zipBuf), "zip", { extractDir: sitePackages });
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

async function handleSwRequest(event: MessageEvent) {
  if (event.data?.type !== "request") return;

  const { requestId, url, method, body } = event.data as {
    requestId: string;
    url: string;
    method: string;
    body: ArrayBuffer | null;
  };

  if (!pyodide) {
    swPort!.postMessage({
      type: "response",
      requestId,
      status: 503,
      contentType: "text/plain",
      body: new TextEncoder().encode("Pyodide not ready"),
    });
    return;
  }

  try {
    // Call the Python request handler registered by browser_server.py.
    // browser_server._setup_js_bridge() stores it in Python's __main__ globals
    // so we can retrieve it reliably via pyodide.globals.get().
    const handleRequest = pyodide.globals.get("pyodide_handle_request");

    if (!handleRequest) {
      // Python hasn't finished setting up yet — tell the client to retry.
      swPort!.postMessage(
        {
          type: "response",
          requestId,
          status: 503,
          contentType: "text/plain",
          body: new TextEncoder().encode("Pyodide not ready").buffer,
        },
        [],
      );
      return;
    }

    // Pass the Uint8Array directly as a JsProxy so Python can call .to_py() on it.
    // Do NOT call pyodide.toPy() here: that converts it to a Python memoryview which
    // Pyodide then unwraps before passing to the Python function, leaving no .to_py().
    const bodyBytes = body ? new Uint8Array(body) : null;
    const result = handleRequest(url, method, bodyBytes ?? null);

    const status: number = result.status;
    const contentType: string = result.contentType;
    let responseBody: Uint8Array;

    const rawBody = result.body;
    if (rawBody instanceof Uint8Array) {
      responseBody = rawBody;
    } else if (rawBody && typeof rawBody.toJs === "function") {
      responseBody = rawBody.toJs({ create_proxies: false });
    } else if (rawBody instanceof ArrayBuffer) {
      responseBody = new Uint8Array(rawBody);
    } else {
      responseBody = new TextEncoder().encode(String(rawBody ?? ""));
    }

    if (status >= 500) {
      console.error(`[pyodide_worker] Python returned ${status} for ${method} ${url}:`, new TextDecoder().decode(responseBody));
    }

    swPort!.postMessage(
      { type: "response", requestId, status, contentType, body: responseBody.buffer },
      [responseBody.buffer],
    );
  } catch (e) {
    const errBytes = new TextEncoder().encode(String(e));
    swPort!.postMessage(
      {
        type: "response",
        requestId,
        status: 500,
        contentType: "text/plain",
        body: errBytes.buffer,
      },
      [errBytes.buffer],
    );
  }
}
