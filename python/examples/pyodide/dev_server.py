#!/usr/bin/env python3
"""Simple local development server for the Pyodide neuroglancer deployment.

Serves the dist/pyodide/ directory on http://localhost:8080 with the
COOP and COEP headers required by Pyodide's SharedArrayBuffer usage.

Usage:
    python python/examples/pyodide/dev_server.py [--port 8080] [--dir dist/pyodide]
"""

import argparse
import http.server
import os
import pathlib


class CoopCoepHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Adds Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers.

    These headers are required for Pyodide to use SharedArrayBuffer, which
    it needs for efficient memory operations.
    """

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

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

    handler = CoopCoepHTTPRequestHandler

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
