#!/usr/bin/env python3
"""Simple local development server for the Pyodide neuroglancer deployment.

Serves the dist/pyodide/ directory on http://localhost:8080, mirroring the
`ng-pyodide` Firebase deploy.

COOP/COEP (cross-origin isolation) are deliberately NOT set: the runtime is
single-threaded (no SharedArrayBuffer), JSPI does not need isolation, and
COOP:same-origin would sever the `window.opener` the middleauth sign-in popup
uses to save exports to GCS. Keeping this in step with `firebase.json` means the
sign-in flow behaves the same locally as in production.

Non-file paths fall back to `index.html`, mirroring the `ng-pyodide` Firebase
`rewrites`. The app moves itself to `/v/pyodide/` at boot, so a pasted share link
is `/v/pyodide/#!{state}`; that path has no file and would otherwise 404.

Usage:
    python python/examples/pyodide/dev_server.py [--port 8080] [--dir dist/pyodide]
"""

import argparse
import http.server
import os
import pathlib


class PyodideDevHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the Pyodide build with no-cache, and no cross-origin isolation.

    See the module docstring: isolation buys nothing single-threaded and breaks
    the middleauth popup, so it is intentionally absent here and in the deploy.
    """

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _rewrite_to_shell_if_missing(self):
        """SPA fallback: serve index.html for paths that map to no real file.

        Mirrors the `ng-pyodide` Firebase `rewrites`. Static bundle files
        (pyodide_worker.js, *.py, the zip, ...) still resolve to themselves; a
        share link like /v/pyodide/#!{state} has no file, so it falls back to the
        app shell. The URL fragment/query stays in the browser -- only the
        server-side file lookup is rewritten -- so client-side `?script=` and the
        `#!` hash still work.
        """
        fs_path = self.translate_path(self.path)
        if os.path.isdir(fs_path):
            if not os.path.isfile(os.path.join(fs_path, "index.html")):
                self.path = "/index.html"
        elif not os.path.isfile(fs_path):
            self.path = "/index.html"

    def do_GET(self):
        self._rewrite_to_shell_if_missing()
        super().do_GET()

    def do_HEAD(self):
        self._rewrite_to_shell_if_missing()
        super().do_HEAD()

    def log_message(self, fmt, *args):
        # Keep output tidy
        print(f"  {self.address_string()} {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--dir",
        default=str(
            pathlib.Path(__file__).parent.parent.parent.parent / "dist" / "pyodide"
        ),
        help="Directory to serve (default: dist/pyodide/)",
    )
    args = parser.parse_args()

    serve_dir = os.path.abspath(args.dir)
    if not os.path.isdir(serve_dir):
        print(f"Error: directory does not exist: {serve_dir}")
        print("Run 'npm run build-pyodide' first.")
        raise SystemExit(1)

    os.chdir(serve_dir)

    handler = PyodideDevHTTPRequestHandler

    with http.server.HTTPServer(("localhost", args.port), handler) as httpd:
        print(f"Serving {serve_dir}")
        print(f"Open http://localhost:{args.port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
