# Neuroglancer Python — Pyodide (Serverless) Deployment

This directory contains everything needed to build and deploy a **fully static,
serverless** version of a Neuroglancer Python viewer.  Python runs entirely in
the browser via [Pyodide](https://pyodide.org/) (Python compiled to WebAssembly)
— no server, no installation, no backend required.

The included example (`example_linear_registration_pyodide.py`) demonstrates
interactive affine registration using numpy arrays and scipy.

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
| `example_linear_registration_pyodide.py` | User Python script |

For development with watch mode:

```bash
npm run build-pyodide:watch
```

This rebuilds JS/TS automatically on file changes.  The Python zip and example
script are only regenerated on a full (non-watch) build.

## Local testing

Pyodide requires [`SharedArrayBuffer`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer),
which in turn requires two HTTP response headers that a plain `python -m http.server`
cannot set:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Use the included dev server instead:

```bash
python python/examples/pyodide/dev_server.py
```

Then open <http://localhost:8080/>.

Options:

```
python python/examples/pyodide/dev_server.py --port 9000
python python/examples/pyodide/dev_server.py --dir /path/to/dist/pyodide
```

**First load** downloads ~40 MB of Python packages from the Pyodide CDN.
Subsequent loads use the browser cache and start in a few seconds.

## Using a custom Python script

To deploy a different neuroglancer Python script instead of the registration
example:

1. Write your script using `neuroglancer.Viewer()` (or
   `neuroglancer.UnsynchronizedViewer()`).  Adapt it for Pyodide:
   - Replace `threading.Timer`-based debounce with `js.setTimeout` /
     `js.clearTimeout` (see `example_linear_registration_pyodide.py` for the
     pattern).
   - Replace file writes with browser downloads via the JS Blob API (same file
     has a `_browser_download` helper you can copy).
   - Remove `argparse`, `webbrowser`, `neuroglancer.cli`, and blocking `input()`
     calls.
   - Execute at module level — no `if __name__ == "__main__":` guard is needed.

2. Update `copyExampleScript()` in `build_tools/build_pyodide.ts` to copy your
   script instead of (or in addition to) the registration example.

3. Update `userScriptUrl` in `src/main_pyodide.ts` to point at your script's
   filename.

4. Rebuild with `npm run build-pyodide`.

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

The `dist/pyodide/` directory is a fully static site.  Any host that supports
custom response headers works.

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

GitHub Pages does not support custom response headers, so `SharedArrayBuffer`
cannot be enabled there.  Use Netlify, Vercel, or Cloudflare Pages instead.

### Requirements for all hosts

- HTTPS is required for Service Workers (localhost is the only exception).
- `pyodide_sw.js` must be served from the root path `/` (which it is, given
  `publicPath: "/"` in the build config).
- No server-side compute or database is required.
