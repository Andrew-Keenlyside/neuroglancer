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
 * @file Entry point for the Pyodide (browser-only, serverless) neuroglancer viewer.
 *
 * Startup sequence:
 *   1. Patch the URL to `/v/pyodide/` so api.ts can parse the viewer token.
 *   2. Register the Service Worker and establish a MessageChannel to it.
 *   3. Start the Pyodide Web Worker, which loads Python and runs the user script.
 *   4. Once Python signals "ready", initialize the neuroglancer viewer.
 *
 * The Service Worker (`pyodide_sw.js`) intercepts all HTTP requests that the
 * neuroglancer frontend normally sends to the Python/Tornado server, and
 * forwards them to the Pyodide Web Worker via the MessageChannel.
 */

// ---------------------------------------------------------------------------
// Step 0: Patch the URL immediately (before any neuroglancer code reads it).
// getServerUrls() in api.ts parses window.location.pathname for /v/{token}.
// ---------------------------------------------------------------------------
if (!window.location.pathname.includes("/v/")) {
  history.replaceState(null, "", "/v/pyodide/");
}

import "#src/util/polyfills.js";
import "#src/layer/enabled_frontend_modules.js";
import "#src/datasource/enabled_frontend_modules.js";
import "#src/kvstore/enabled_frontend_modules.js";
import { debounce } from "lodash-es";
import { CachingCredentialsManager } from "#src/credentials_provider/index.js";
import type { PythonDataSource } from "#src/datasource/python/frontend.js";
import {
  Client,
  ClientStateReceiver,
  ClientStateSynchronizer,
} from "#src/python_integration/api.js";
import { PythonCredentialsManager } from "#src/python_integration/credentials_provider.js";
import { TrackableBasedEventActionMap } from "#src/python_integration/event_action_map.js";
import { PrefetchManager } from "#src/python_integration/prefetch.js";
import { RemoteActionHandler } from "#src/python_integration/remote_actions.js";
import { TrackableBasedStatusMessages } from "#src/python_integration/remote_status_messages.js";
import { ScreenshotHandler } from "#src/python_integration/screenshots.js";
import { VolumeRequestHandler } from "#src/python_integration/volume.js";
import { TrackableValue } from "#src/trackable_value.js";
import {
  bindDefaultCopyHandler,
  bindDefaultPasteHandler,
} from "#src/ui/default_clipboard_handling.js";
import { setDefaultInputEventBindings } from "#src/ui/default_input_event_bindings.js";
import { makeDefaultViewer } from "#src/ui/default_viewer.js";
import { bindTitle } from "#src/ui/title.js";
import { UrlHashBinding } from "#src/ui/url_hash_binding.js";
import { parseFixedLengthArray, verifyInt } from "#src/util/json.js";
import type { Trackable } from "#src/util/trackable.js";
import { CompoundTrackable } from "#src/util/trackable.js";
import type { InputEventBindings } from "#src/viewer.js";
import { VIEWER_UI_CONFIG_OPTIONS } from "#src/viewer.js";

// ---------------------------------------------------------------------------
// Helpers (copied from main_python.ts)
// ---------------------------------------------------------------------------

function makeTrackableBasedEventActionMaps(
  inputEventBindings: InputEventBindings,
) {
  const config = new CompoundTrackable();
  const globalMap = new TrackableBasedEventActionMap();
  config.add("viewer", globalMap);
  inputEventBindings.global.addParent(globalMap.eventActionMap, 1000);

  const sliceViewMap = new TrackableBasedEventActionMap();
  config.add("sliceView", sliceViewMap);
  inputEventBindings.sliceView.addParent(sliceViewMap.eventActionMap, 1000);

  const perspectiveViewMap = new TrackableBasedEventActionMap();
  config.add("perspectiveView", perspectiveViewMap);
  inputEventBindings.perspectiveView.addParent(
    perspectiveViewMap.eventActionMap,
    1000,
  );

  const dataViewMap = new TrackableBasedEventActionMap();
  config.add("dataView", dataViewMap);
  inputEventBindings.perspectiveView.addParent(dataViewMap.eventActionMap, 999);
  inputEventBindings.sliceView.addParent(dataViewMap.eventActionMap, 999);

  return config;
}

function makeTrackableBasedSourceGenerationHandler(
  pythonDataSource: PythonDataSource,
) {
  const state = new TrackableValue<{ [key: string]: number }>({}, (x) => {
    for (const key of Object.keys(x)) {
      const value = x[key];
      if (typeof value !== "number") {
        throw new Error(
          `Expected key ${JSON.stringify(key)} to have a numeric value, but received: ${JSON.stringify(value)}.`,
        );
      }
    }
    return x;
  });
  state.changed.add(
    debounce(() => {
      const generations = state.value;
      for (const key of Object.keys(generations)) {
        pythonDataSource.setSourceGeneration(key, generations[key]);
      }
      for (const key of pythonDataSource.sourceGenerations.keys()) {
        if (!Object.prototype.hasOwnProperty.call(generations, key)) {
          pythonDataSource.deleteSourceGeneration(key);
        }
      }
    }, 0),
  );
  return state;
}

// ---------------------------------------------------------------------------
// Loading overlay helpers
// ---------------------------------------------------------------------------

function setLoadingMessage(msg: string) {
  const el = document.getElementById("neuroglancer-pyodide-loading");
  if (el) el.textContent = msg;
}

function hideLoadingOverlay() {
  const el = document.getElementById("neuroglancer-pyodide-overlay");
  if (el) el.style.display = "none";
}

/** Show the URL-input form and resolve with the entered URL (or null). */
function promptForStartingUrl(): Promise<string | null> {
  return new Promise((resolve) => {
    const spinner = document.getElementById("neuroglancer-pyodide-spinner")!;
    const loading = document.getElementById("neuroglancer-pyodide-loading")!;
    const note = document.getElementById("neuroglancer-pyodide-note")!;
    const section = document.getElementById("neuroglancer-pyodide-url-section")!;
    const input = document.getElementById("neuroglancer-pyodide-url-input") as HTMLInputElement;
    const btnUrl = document.getElementById("neuroglancer-pyodide-btn-url")!;
    const btnDemo = document.getElementById("neuroglancer-pyodide-btn-demo")!;

    spinner.style.display = "none";
    loading.style.display = "none";
    note.style.display = "none";
    section.style.display = "flex";
    input.focus();

    const submit = () => {
      const url = input.value.trim();
      section.style.display = "none";
      spinner.style.display = "";
      loading.style.display = "";
      setLoadingMessage("Running Python setup…");
      resolve(url || null);
    };

    btnUrl.addEventListener("click", submit);
    btnDemo.addEventListener("click", () => {
      input.value = "";
      submit();
    });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
  });
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function bootstrap() {
  // ------------------------------------------------------------------
  // 1. Register the Service Worker
  // ------------------------------------------------------------------
  setLoadingMessage("Registering Service Worker…");
  if (!("serviceWorker" in navigator)) {
    setLoadingMessage("Error: Service Workers are not supported in this browser.");
    return;
  }

  let swRegistration: ServiceWorkerRegistration;
  try {
    // pyodide_sw.js must be served from the root (or the same scope as the page).
    swRegistration = await navigator.serviceWorker.register("/pyodide_sw.js", {
      scope: "/",
    });
  } catch (e) {
    setLoadingMessage(`Error registering Service Worker: ${e}`);
    throw e;
  }

  // Wait for the Service Worker to become active.
  await navigator.serviceWorker.ready;
  const sw = swRegistration.active!;

  // ------------------------------------------------------------------
  // 2. Create a MessageChannel to bridge the SW and the Pyodide Worker
  // ------------------------------------------------------------------
  const channel = new MessageChannel();

  // Send port1 to the Service Worker.
  sw.postMessage({ type: "init" }, [channel.port1]);

  // ------------------------------------------------------------------
  // 3. Start the Pyodide Web Worker
  // ------------------------------------------------------------------
  setLoadingMessage("Starting Python runtime (Pyodide)…");

  // The worker URL is resolved relative to this module at build time by rspack.
  const pyodideWorker = new Worker(
    new URL(
      "./python_integration/pyodide_worker.ts",
      import.meta.url,
    ),
    { type: "module" },
  );

  // Phase 1: load Pyodide + packages.  Resolves on 'packages_ready'.
  const packagesReadyPromise = new Promise<void>((resolve, reject) => {
    pyodideWorker.onmessage = (event) => {
      const { type, message } = event.data;
      if (type === "progress") {
        setLoadingMessage(message);
      } else if (type === "packages_ready") {
        resolve();
      } else if (type === "error") {
        reject(new Error(message));
      }
    };
    pyodideWorker.onerror = reject;
  });

  // Send port2 and configuration to the Pyodide Worker (does NOT run script yet).
  const neuroglancerZipUrl = new URL("/neuroglancer_pyodide.zip", window.location.href).href;
  const userScriptUrl = new URL("/example_linear_registration_pyodide.py", window.location.href).href;
  const userScript = await fetch(userScriptUrl).then((r) => r.text());

  // Pyodide CDN — using 0.27.x (update as needed).
  const pyodideIndexUrl =
    "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/pyodide.js";

  pyodideWorker.postMessage(
    {
      type: "init",
      port: channel.port2,
      neuroglancerZipUrl,
      userScript,
      pyodideIndexUrl,
    },
    [channel.port2],
  );

  setLoadingMessage("Loading Python packages (numpy, scipy, pillow)…");
  await packagesReadyPromise;

  // Phase 2: prompt the user for an optional starting Neuroglancer URL.
  const startingUrl = await promptForStartingUrl();

  // Phase 3: tell the worker to run the script, then wait for 'ready'.
  const readyPromise = new Promise<void>((resolve, reject) => {
    pyodideWorker.onmessage = (event) => {
      const { type, message } = event.data;
      if (type === "progress") {
        setLoadingMessage(message);
      } else if (type === "ready") {
        resolve();
      } else if (type === "error") {
        reject(new Error(message));
      }
    };
  });
  pyodideWorker.postMessage({ type: "run", startingUrl });
  await readyPromise;

  // Handle download requests posted by Python (workers lack document).
  pyodideWorker.addEventListener("message", (event) => {
    if (event.data?.type === "download") {
      const { filename, content, mimeType } = event.data as {
        filename: string;
        content: string;
        mimeType: string;
      };
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  });

  // ------------------------------------------------------------------
  // 4. Initialize the neuroglancer viewer (mirrors main_python.ts)
  // ------------------------------------------------------------------
  hideLoadingOverlay();

  const configState = new CompoundTrackable();
  const client = new Client();

  const credentialsManager = new PythonCredentialsManager(client);

  const viewer = ((window as any).viewer = makeDefaultViewer({
    showLayerDialog: false,
    resetStateWhenEmpty: false,
    credentialsManager: new CachingCredentialsManager(credentialsManager),
  }));

  const pythonDataSource = viewer.dataSourceProvider.dataSources.get(
    "python",
  ) as PythonDataSource;
  configState.add(
    "sourceGenerations",
    makeTrackableBasedSourceGenerationHandler(pythonDataSource),
  );
  setDefaultInputEventBindings(viewer.inputEventBindings);
  configState.add(
    "inputEventBindings",
    makeTrackableBasedEventActionMaps(viewer.inputEventBindings),
  );

  const remoteActionHandler = new RemoteActionHandler(viewer);
  (window as any).remoteActionHandler = remoteActionHandler;
  configState.add("actions", remoteActionHandler.actionSet);

  configState.add("statusMessages", new TrackableBasedStatusMessages());

  const screenshotHandler = new ScreenshotHandler(viewer);
  configState.add("screenshot", screenshotHandler.requestState);

  const volumeHandler = new VolumeRequestHandler(viewer);
  configState.add("volumeRequests", volumeHandler.requestState);

  let sharedState: Trackable | undefined = viewer.state;

  if (window.location.hash) {
    const hashBinding = viewer.registerDisposer(
      new UrlHashBinding(
        viewer.state,
        viewer.dataSourceProvider.sharedKvStoreContext,
      ),
    );
    hashBinding.updateFromUrlHash();
    sharedState = undefined;
  }

  const prefetchManager = new PrefetchManager(
    viewer.display,
    viewer.dataSourceProvider,
    viewer.dataContext.addRef(),
    viewer.uiConfiguration,
  );
  configState.add("prefetch", prefetchManager);

  for (const key of VIEWER_UI_CONFIG_OPTIONS) {
    configState.add(key, viewer.uiConfiguration[key]);
  }
  configState.add("scaleBarOptions", viewer.scaleBarOptions);

  const size = new TrackableValue<[number, number] | undefined>(
    undefined,
    (x) =>
      x == null
        ? undefined
        : parseFixedLengthArray(<[number, number]>[0, 0], x, verifyInt),
  );
  configState.add("viewerSize", size);

  const updateSize = () => {
    const element = viewer.display.container;
    const value = size.value;
    if (value === undefined) {
      element.style.position = "relative";
      element.style.width = "";
      element.style.height = "";
      element.style.transform = "";
      element.style.transformOrigin = "";
    } else {
      element.style.position = "absolute";
      element.style.width = `${value[0]}px`;
      element.style.height = `${value[1]}px`;
      const screenWidth = document.documentElement!.clientWidth;
      const screenHeight = document.documentElement!.clientHeight;
      const scaleX = screenWidth / value[0];
      const scaleY = screenHeight / value[1];
      const scale = Math.min(scaleX, scaleY);
      element.style.transform = `scale(${scale})`;
      element.style.transformOrigin = "top left";
    }
  };
  updateSize();
  window.addEventListener("resize", updateSize);
  size.changed.add(debounce(() => updateSize(), 0));

  const states = new Map<string, ClientStateSynchronizer>();
  states.set("c", new ClientStateSynchronizer(client, configState, null));
  if (sharedState !== undefined) {
    states.set("s", new ClientStateSynchronizer(client, sharedState, 100));
  }
  new ClientStateReceiver(client, states);

  remoteActionHandler.sendActionRequested.add((action, state) =>
    client.sendActionNotification(action, state),
  );
  screenshotHandler.sendScreenshotRequested.add((state) =>
    client.sendActionNotification("screenshot", state),
  );
  screenshotHandler.sendStatisticsRequested.add((state) =>
    client.sendActionNotification("screenshotStatistics", state),
  );

  volumeHandler.sendVolumeInfoResponseRequested.add((requestId, info) =>
    client.sendVolumeInfoNotification(requestId, info),
  );

  volumeHandler.sendVolumeChunkResponseRequested.add((requestId, info) =>
    client.sendVolumeChunkNotification(requestId, info),
  );

  bindDefaultCopyHandler(viewer);
  bindDefaultPasteHandler(viewer);
  viewer.registerDisposer(bindTitle(viewer.title));
}

bootstrap().catch((err) => {
  console.error("Pyodide bootstrap failed:", err);
  setLoadingMessage(`Failed to start: ${err}`);
});
