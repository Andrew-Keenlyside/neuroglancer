# Neuroglancer Python — Pyodide (Serverless) Deployment

This directory contains everything needed to build and deploy a **fully static,
serverless** version of a Neuroglancer Python viewer.  Python runs entirely in
the browser via [Pyodide](https://pyodide.org/) (Python compiled to WebAssembly)
— no server, no installation, no backend required.

One shared bundle is built once and reused across use cases; each use case is a
separate Python script that ships alongside it and is chosen at runtime via the
`?script=<same-origin-path>` URL parameter (same server, same bundle — only the
URL differs). The build copies every `*.py` in this directory into
`dist/pyodide/`. Two ship today:

- `user_script.py` — **default** (no `?script=`): the Streamline Filter demo,
  which boots the HCP1065 whole-brain tractogram (zarr-vectors) into a viewer
  whose layer has a **Filter** tab for drawing ROIs.
- `example_linear_registration_pyodide.py` — the interactive linear-registration
  workflow; open with `?script=/example_linear_registration_pyodide.py`.

## How it works

```
Browser Tab
├── Main Thread
│   ├── Neuroglancer JS frontend (WebGL renderer, unchanged)
│   └── Bootstrap: registers SW, starts Pyodide Worker, shows loading overlay
│
├── Service Worker  (pyodide_sw.js)
│   ├── Intercepts all API fetch() calls that normally go to the Tornado server
│   └── Forwards requests to the Pyodide Worker; streams SSE state updates back
│
└── Pyodide Web Worker  (pyodide_worker.[hash].js)
    ├── Loads Pyodide + numpy + scipy + pillow from CDN (~40 MB, cached after first load)
    ├── Loads the neuroglancer Python package (from bundled neuroglancer_pyodide.zip)
    └── Runs the user Python script; handles HTTP requests from the Service Worker
```

The neuroglancer Python API (`neuroglancer.Viewer`, `neuroglancer.LocalVolume`,
etc.) works without changes — in Pyodide mode the Tornado server is replaced
by an in-process request router that the Service Worker calls directly.

## Prerequisites

- **Node.js** ≥ 18 (for the build)
- **Python 3** (only needed to create the zip bundle at build time)
- `npm install` already run in the repo root

## Building

```bash
npm run build-pyodide
```

Output is written to `dist/pyodide/`:

| File | Description |
|------|-------------|
| `index.html` | Entry page with loading overlay |
| `main_pyodide.[hash].js` | Main bundle (neuroglancer + bootstrap) |
| `pyodide_sw.js` | Service Worker (stable filename, no hash) |
| `pyodide_worker.[hash].js` | Pyodide Web Worker |
| `neuroglancer_pyodide.zip` | Bundled neuroglancer Python package |
| `user_script.py` | Default script (Streamline Filter demo; override with `?script=`) |
| `example_linear_registration_pyodide.py` | Linear-registration workflow (`?script=/…`) |

For development with watch mode:

```bash
npm run build-pyodide:watch
```

This rebuilds JS/TS automatically on file changes.  The Python zip and example
script are only regenerated on a full (non-watch) build.

## In-browser tract export (zarr-vectors)

The Streamline Filter layer's **Export** tab runs entirely in the browser here:
it re-evaluates the dissection at level 0 through `zarr-vectors-py`'s async read
path and writes a **TrackVis `.trk`** or a **zipped zarr-vectors store** in
Pyodide's memory, then either **downloads** it (no auth) or **uploads** it to the
configured GCS bucket (`Save to GCS`, middleauth sign-in). Both **selected
groups** and **whole store** scopes are supported. No native exporter or sidecar
is needed.

This needs the (currently unreleased) `zarr-vectors-py` bundled into the Python
zip via `NEUROGLANCER_PYODIDE_PACKAGES` — a comma/`;`-separated list of package
directories added to `neuroglancer_pyodide.zip` at build time:

```bash
NEUROGLANCER_PYODIDE_PACKAGES="/path/to/zarr-vectors-py/zarr_vectors" \
  npm run build-pyodide
```

Notes:
- `zarr` itself is already loaded from the Pyodide distribution; only the
  `zarr_vectors` package tree needs bundling. `zarr-vectors-tools` is **not**
  required (the TRK writer is inlined; the ZVF writer uses core `write_polylines`).
- `nibabel` (TRK only) is pure-Python and installed on first export via
  `micropip` — the first `.trk` export therefore needs network.
- **Graceful degradation:** if `zarr_vectors` is not bundled, the export route
  returns a clear error and the tab falls back to "Download job spec" (run it
  with `python -m neuroglancer.tract_export`). The ordinary `dist/client` build
  and CI are unaffected — the imports are Pyodide-only and guarded.
- The `.zvf` writer drives zarr's synchronous path under JSPI; the export
  request runs on a serialized promising entrypoint (`pyodide_worker.ts` +
  `browser_server.py`). Verify a real store round-trips in the target browser
  before relying on ZVF export (the read/TRK paths are covered by CPython tests;
  the JSPI write path is not).

## Local testing

Use the included dev server:

```bash
python python/examples/pyodide/dev_server.py
```

Then open <http://localhost:8080/>.

> **Cross-origin isolation (COOP/COEP) is deliberately NOT set.** This runtime is
> single-threaded — it loads Pyodide with no pthreads and uses no
> `SharedArrayBuffer` anywhere — so cross-origin isolation buys nothing, and its
> `COOP: same-origin` would sever the `window.opener` the middleauth sign-in
> popup needs to save exports to GCS. The dev server and the `ng-pyodide`
> Firebase deploy both omit the headers to keep sign-in working. (If you ever
> reintroduce Pyodide threading, you will need the headers back — and will lose
> middleauth GCS save in this build unless you move it to the `google` OAuth
> provider, which broadcasts past COOP.) Verify a fresh build still loads in your
> target browsers after any change here.

Options:

```
python python/examples/pyodide/dev_server.py --port 9000
python python/examples/pyodide/dev_server.py --dir /path/to/dist/pyodide
```

**First load** downloads ~40 MB of Python packages from the Pyodide CDN.
Subsequent loads use the browser cache and start in a few seconds.

## Sharing a filter session via URL

The whole viewer state — layers **and** the ROI filter groups — lives in the
address bar as a standard Neuroglancer `#!{…}` link that updates live as you
work. The link carries the ROI **positions** (sphere centers/radii, box corners,
half-space planes, label-mask label ids); it never carries the materialised list
of passing streamline IDs, which the viewer recomputes on load.

Round-trip:

1. Open the app and click **Use test data (HCP1065)** (or open any share link).
2. Draw/commit ROI filter groups. The address bar keeps updating —
   `…/v/pyodide/#!{…}`.
3. **Copy the browser URL** and send it. Opening it (in this app) restores the
   scene and the committed groups and re-runs the dissection, so the recipient
   continues exactly where you left off. It auto-loads — no start screen.

You can also paste a share link into the **Load URL** box on the start screen; a
link containing a `#!` fragment is applied as viewer state (the same as opening
it directly) rather than passed to the Python script.

> Share links live under the `/v/pyodide/` path (the app moves itself there at
> boot so the export route has a token). The `ng-pyodide` Firebase target and the
> local `dev_server.py` both rewrite non-file paths to `index.html`, so those
> links load instead of 404ing. A bare `…/#!{…}` link (no `/v/…` path) would have
> its fragment stripped at boot — always share the address-bar URL as-is.

## Using a custom Python script

To deploy a different neuroglancer Python script instead of the default:

1. Write your script using `neuroglancer.Viewer()` (or
   `neuroglancer.UnsynchronizedViewer()`).  Adapt it for Pyodide:
   - Replace `threading.Timer`-based debounce with `js.setTimeout` /
     `js.clearTimeout`.
   - Replace file writes with browser downloads via the JS Blob API.
   - Remove `argparse`, `webbrowser`, `neuroglancer.cli`, and blocking `input()`
     calls.
   - Execute at module level — no `if __name__ == "__main__":` guard is needed.

2. Serve it alongside the bundle and select it with the `?script=` query
   parameter, e.g. <http://localhost:8080/?script=/my_script.py>.  The path must
   be same-origin: the script is fetched and executed with full access to the
   viewer, so an absolute URL in a shared link must not be able to run code.

   To change the default instead, replace `python/examples/pyodide/user_script.py`
   (the build copies it to `dist/pyodide/user_script.py`, which
   `DEFAULT_USER_SCRIPT_PATH` in `src/main_pyodide.ts` points at).

3. Rebuild with `npm run build-pyodide`.

**Available packages:** numpy, scipy, pillow are loaded by default.  Add more
via `micropip` inside your script or by extending `loadPackage(...)` in
`src/python_integration/pyodide_worker.ts`.

**Limitations:**
- The C++ `_neuroglancer` extension (mesh generation via marching cubes) is not
  available in Pyodide.  Segmentation meshes return a 501 error; the rest of
  the viewer works normally.
- TensorStore and cloudvolume are not available.  Only `LocalVolume` (numpy
  arrays) and CORS-enabled remote data sources (e.g. precomputed on GCS) work.
- Memory is limited to the browser heap (~2 GB).

## Production deployment

The `dist/pyodide/` directory is a fully static site, and — because this build
sets **no** COOP/COEP headers (see the Local testing note) — any static host
works, including ones that cannot set custom response headers. The
header-setting examples below are legacy; **do not** add COOP/COEP if you want
middleauth GCS save to work (it severs the sign-in popup). The canonical deploy
is the `ng-pyodide` Firebase target (see `src/roi_store/README.md`).

### Netlify

Add a `_headers` file to `dist/pyodide/` before uploading:

```
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
```

Then drag-and-drop the folder in the Netlify dashboard, or use the CLI:

```bash
npx netlify deploy --prod --dir dist/pyodide
```

### Vercel

Add `vercel.json` to the repo root (or `dist/pyodide/`):

```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" },
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" }
      ]
    }
  ]
}
```

### AWS S3 + CloudFront

Upload `dist/pyodide/` to an S3 bucket with static website hosting enabled,
then configure a CloudFront **Response Headers Policy** that adds the two COOP /
COEP headers to all responses.

### GitHub Pages

GitHub Pages cannot set custom response headers, but this build no longer needs
any (it is single-threaded, with no `SharedArrayBuffer`), so it works there.

### Requirements for all hosts

- HTTPS is required for Service Workers (localhost is the only exception).
- `pyodide_sw.js` must be served from the root path `/` (which it is, given
  `publicPath: "/"` in the build config).
- No server-side compute or database is required.
