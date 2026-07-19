# @license
# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A standalone local exporter the browser build can reach.

The viewer's own Python server already carries this route, but the browser and
Pyodide builds are served as static files and never talk to it. Those builds
cannot do the work themselves either: a TRK writer would only need numpy, but a
zarr-vectors store goes through ``zarr``'s synchronous API, which drives a
dedicated IO thread (``zarr/core/sync.py``) that Pyodide cannot start. So to get
both formats out of a statically served viewer, the exporter runs beside it and
the page calls in.

Security. This writes a file at a path the request chooses, so it is bound to
loopback and gated on a token minted per run and printed at startup. The token
is in the URL the user pastes into the Export tab, so a page that was not given
it cannot drive the exporter -- which matters, because any site the user visits
can otherwise reach ``127.0.0.1``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets

import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.web

from neuroglancer.tract_export.job import JobSpecError, parse_job
from neuroglancer.tract_export.run import ExportRunError, run_job

DEFAULT_PORT = 9944
PATH_REGEX = r"^/export/tract/(?P<token>[^/]+)$"


class _ExportHandler(tornado.web.RequestHandler):
    def initialize(self, token: str, allow_origin: str) -> None:
        self.token = token
        self.allow_origin = allow_origin

    def prepare(self) -> None:
        # Not `set_default_headers`: tornado calls that from `clear()` inside
        # `__init__`, which runs BEFORE `initialize()`, so `self.allow_origin`
        # would not exist yet and every request would die with an AttributeError
        # before writing a response. `prepare()` runs after initialize.
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", self.allow_origin)
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        # The Pyodide build is served cross-origin-isolated (COEP:
        # require-corp), which blocks any cross-origin response that does not
        # opt in. Without this the fetch fails before the handler is reached.
        self.set_header("Cross-Origin-Resource-Policy", "cross-origin")

    def options(self, token: str) -> None:
        self.set_status(204)
        self.finish()

    async def post(self, token: str) -> None:
        # compare_digest so a wrong token cannot be recovered by timing.
        if not secrets.compare_digest(token, self.token):
            self.set_status(404)
            self.finish(json.dumps({"error": "Unknown exporter token."}))
            return
        try:
            spec = json.loads(self.request.body)
        except ValueError as e:
            self.set_status(400)
            self.finish(json.dumps({"error": f"Invalid JSON: {e}"}))
            return
        try:
            job = parse_job(spec)
        except JobSpecError as e:
            self.set_status(400)
            self.finish(json.dumps({"error": str(e)}))
            return
        try:
            # On a thread: a real export is seconds to minutes of CPU and I/O.
            summary = await asyncio.to_thread(run_job, job)
        except ExportRunError as e:
            self.set_status(422)
            self.finish(json.dumps({"error": str(e)}))
            return
        except Exception as e:
            logging.exception("tract export failed")
            self.set_status(500)
            self.finish(json.dumps({"error": f"Export failed: {e}"}))
            return
        logging.info(
            "exported %s streamline(s) to %s",
            summary.get("streamline_count"),
            summary.get("output_path"),
        )
        self.finish(json.dumps(summary))


def make_app(token: str, allow_origin: str = "*") -> tornado.web.Application:
    return tornado.web.Application(
        [(PATH_REGEX, _ExportHandler, {"token": token, "allow_origin": allow_origin})]
    )


def serve(
    port: int = DEFAULT_PORT,
    bind: str = "127.0.0.1",
    token: str | None = None,
    allow_origin: str = "*",
) -> tuple[str, int, str]:
    """Start the exporter. Returns (url, port, token); does not block."""
    token = token or secrets.token_urlsafe(24)
    sockets = tornado.netutil.bind_sockets(port, bind)
    server = tornado.httpserver.HTTPServer(make_app(token, allow_origin))
    server.add_sockets(sockets)
    actual = sockets[0].getsockname()[1]
    return f"http://{bind}:{actual}/export/tract/{token}", actual, token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m neuroglancer.tract_export --serve",
        description="Run a local tract exporter for a statically served viewer.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Loopback by default. Binding wider exposes arbitrary file writes.",
    )
    parser.add_argument(
        "--allow-origin",
        default="*",
        help="Origin allowed to call the exporter; the token is the real gate.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    url, _, _ = serve(args.port, args.bind, allow_origin=args.allow_origin)
    print("\nLocal tract exporter running.\n")
    print("Paste this into the Export tab's “Exporter URL” field:\n")
    print(f"    {url}\n")
    print("The token is per-run: restarting mints a new URL. Ctrl-C to stop.\n")
    try:
        # The canonical Tornado blocking call. `asyncio.get_event_loop()` here
        # raised on Python 3.14 (no running loop), and `IOLoop.current()` is
        # what `serve()`'s `add_sockets` already scheduled on regardless.
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
