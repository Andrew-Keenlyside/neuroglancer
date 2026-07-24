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

interface PyodideModule {
  loadPyodide(options?: Record<string, unknown>): Promise<any>;
}

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
  // Load the Pyodide runtime. `pyodideIndexUrl` points at pyodide.mjs; import
  // it dynamically rather than via importScripts(pyodide.js), because pyodide
  // >=314 refuses to run in a classic worker and importScripts is not
  // available in a module worker. The webpackIgnore comment keeps rspack from
  // trying to resolve the CDN URL at build time.
  const mod: PyodideModule = await import(
    /* webpackIgnore: true */ pyodideIndexUrl
  );
  pyodide = await mod.loadPyodide({
    indexURL: pyodideIndexUrl.replace(/[^/]*$/, ""),
  });

  self.postMessage({ type: "progress", message: "Loading Python packages…" });

  // Install scientific packages. `zarr` comes from the pyodide distribution
  // (314.0.2 ships a WASM-patched zarr 3.2.1 that runs sync() on a WebLoop, not
  // a thread) so the tractography reader can go through zarr-vectors-py rather
  // than its built-in decoder. loadPackage pulls zarr's deps (numcodecs,
  // google-crc32c) automatically; never micropip zarr, which is the unpatched
  // PyPI wheel. Absent zarr, or absent the bundled zarr_vectors package, the
  // reader falls back cleanly.
  await pyodide.loadPackage(["numpy", "scipy", "pillow", "micropip", "zarr"]);

  self.postMessage({
    type: "progress",
    message: "Loading neuroglancer Python package…",
  });

  // Unpack the bundled neuroglancer Python zip into the VFS.
  const zipResp = await fetch(neuroglancerZipUrl);
  if (!zipResp.ok) {
    throw new Error(
      `Failed to fetch neuroglancer zip: HTTP ${zipResp.status} from ${neuroglancerZipUrl}`,
    );
  }
  const zipBuf = await zipResp.arrayBuffer();
  // unpackArchive requires a TypedArray, not a raw ArrayBuffer.
  // Use Python's site module to get the correct versioned site-packages path.
  const sitePackages: string = pyodide.runPython(
    "import site; site.getsitepackages()[0]",
  );
  pyodide.unpackArchive(new Uint8Array(zipBuf), "zip", {
    extractDir: sitePackages,
  });
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

/** True for the tract-export route (the only one that may need JSPI). */
function isExportPath(url: string): boolean {
  try {
    return /\/neuroglancer\/export\//.test(new URL(url).pathname);
  } catch {
    return false;
  }
}

/**
 * Whether this runtime has WebAssembly stack switching (JSPI).
 *
 * `callPromising` / zarr's synchronous `sync()` need it. When it is absent
 * (older/other browsers), export falls back to a plain async handler awaited on
 * the WebLoop: TRK works there (async read + pure write), and `.zvf` -- which
 * reaches zarr's synchronous writer -- returns a clear "needs JSPI" error.
 */
let jspiCache: boolean | undefined;
function jspiAvailable(): boolean {
  if (jspiCache === undefined) {
    jspiCache =
      typeof (WebAssembly as any).Suspending === "function" ||
      typeof (WebAssembly as any).promising === "function";
  }
  return jspiCache;
}

/**
 * A one-at-a-time gate for the promising export entrypoint.
 *
 * `callPromising` enables JSPI stack switching so the `.zvf` writer's
 * synchronous zarr `sync()` can suspend -- and two concurrent promising entries
 * into that path wedge this worker permanently (see the long note at the
 * synchronous `handleRequest` call below). Serialising every promising entry is
 * the proven mitigation: 100% completion at hanging scale. Ordinary requests do
 * not take this path and keep their full concurrency.
 */
let exportTail: Promise<unknown> = Promise.resolve();
function runExclusive<T>(fn: () => Promise<T>): Promise<T> {
  const result = exportTail.then(fn);
  // Swallow settle state on the tail so a failed export cannot reject the next.
  exportTail = result.then(
    () => {},
    () => {},
  );
  return result;
}

/** Convert the Python handler's `{status, contentType, body}` into transferable bytes. */
function marshalPythonResponse(result: any): {
  status: number;
  contentType: string;
  responseBody: Uint8Array;
} {
  const status: number = result.status;
  const contentType: string = result.contentType;
  const rawBody = result.body;
  let responseBody: Uint8Array;
  if (rawBody instanceof Uint8Array) {
    responseBody = rawBody;
  } else if (rawBody && typeof rawBody.toJs === "function") {
    responseBody = rawBody.toJs({ create_proxies: false });
  } else if (rawBody instanceof ArrayBuffer) {
    responseBody = new Uint8Array(rawBody);
  } else {
    responseBody = new TextEncoder().encode(String(rawBody ?? ""));
  }
  return { status, contentType, responseBody };
}

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

  // Export is the one route that needs JSPI (its .zvf writer calls zarr sync()).
  // Everything else stays on the synchronous, fully-concurrent path.
  const isExport = isExportPath(url);

  try {
    // Call the Python handler registered by browser_server.py, retrieved from
    // Python's __main__ globals via pyodide.globals.get(). Export uses a
    // separate promising-capable entrypoint.
    const handlerName = !isExport
      ? "pyodide_handle_request"
      : jspiAvailable()
        ? "pyodide_handle_export"
        : "pyodide_handle_export_async";
    const handler = pyodide.globals.get(handlerName);

    if (!handler) {
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

    let result: any;
    if (isExport && jspiAvailable()) {
      // Export via callPromising so the .zvf writer's synchronous zarr call can
      // suspend (JSPI), under runExclusive so only ONE promising entry is ever
      // live -- the proven guard against the deadlock documented on the
      // synchronous path below. The async read (zarr_vectors/core/aio) never
      // reaches sync(); only the .zvf writer does.
      result = await runExclusive(() =>
        handler.callPromising(url, method, bodyBytes ?? null),
      );
    } else if (isExport) {
      // No JSPI: the handler is an async coroutine; await it on the WebLoop.
      // TRK completes here (async read + pure write). .zvf hits zarr's
      // synchronous writer and returns a clear "needs JSPI" error. Still under
      // runExclusive for parity with the promising path.
      result = await runExclusive(() =>
        handler(url, method, bodyBytes ?? null),
      );
    } else {
      // This call MUST stay synchronous. Do not "fix" it to
      // `await handler.callPromising(...)`.
      //
      // Being synchronous is what makes each request handler ATOMIC. The service
      // worker assigns a requestId and posts immediately, with no queue
      // (pyodide_service_worker.ts) -- concurrency is the design, and it is only
      // safe because handlers cannot interleave.
      //
      // Making this promising enables JSPI stack switching, which is what a
      // synchronous zarr call needs in order to suspend. It also turns atomic
      // handlers into interleaved coroutines, and two concurrent promising entries
      // into zarr-vectors' sync read path wedge this worker PERMANENTLY.
      // Independently reproduced 2026-07-19 (headless chromium 141, pyodide
      // 314.0.2), deterministic over 8 runs:
      //
      //   - concurrency 2 at ~530-730 store keys per reader and above -> hang.
      //     Below that it interleaves fine, so a small-store smoke test passes and
      //     production hangs.
      //   - the same store read concurrently hangs too; not a two-store artifact.
      //   - sequential at ~7x the work (12,096 keys x2) completes in 25.5s, so it
      //     is not memory and not total work.
      //   - blast radius is THIS worker's event loop, not the page's: the UI stays
      //     alive while Python request handling is dead and unrecoverable.
      //
      // Root cause is still open. Two internal-corruption hypotheses
      // (validSuspender save/restore crossing, Module.stackStop clobbering) were
      // tested and refuted; what is established is that resuming a JSPI-suspended
      // stack needs the macrotask queue, and the queue stops being serviced.
      //
      // The export route above IS promising, so it is confined to runExclusive:
      // serialising every promising entry is the proven mitigation (100%
      // completion at hanging scale), and it must cover EVERY entry -- any other
      // promising entrypoint racing a request reintroduces the identical bug.
      //
      // The read avoids JSPI entirely via zarr_vectors/core/aio.py, which awaits
      // the I/O up front then replays the ordinary sync readers against a primed
      // Group so they never reach sync(). See upstream_prompts/zarr-vectors-py.md.
      result = handler(url, method, bodyBytes ?? null);
    }

    const { status, contentType, responseBody } = marshalPythonResponse(result);

    if (status >= 500) {
      console.error(
        `[pyodide_worker] Python returned ${status} for ${method} ${url}:`,
        new TextDecoder().decode(responseBody),
      );
    }

    swPort!.postMessage(
      {
        type: "response",
        requestId,
        status,
        contentType,
        body: responseBody.buffer,
      },
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
