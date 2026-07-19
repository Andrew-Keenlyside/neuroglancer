# @license
# Copyright 2024 Google Inc.
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

"""Browser-compatible virtual HTTP server for Pyodide/WASM deployments.

Replaces the Tornado-based server.py when running inside a Pyodide Web Worker.
Instead of binding to a TCP port, this module:

  1. Registers a request handler on ``js.globalThis`` so that the Service Worker
     (pyodide_service_worker.ts) can forward intercepted HTTP requests directly
     to Python.
  2. Pushes Server-Sent Events (SSE) via ``js.globalThis.pyodide_push_sse``
     whenever viewer state changes, so the SSE stream maintained by the Service
     Worker is updated in real time.

Usage (from Python side):
    server = get_browser_server()
    server.register_viewer(viewer)

The JavaScript side (pyodide_worker.ts) calls ``js.globalThis.pyodide_handle_request``
for every intercepted request and passes the response back to the Service Worker.
"""

import json
import re

import numpy as np

from . import local_volume, skeleton
from .json_utils import encode_json, json_encoder_default
from .trackable_state import ConcurrentModificationError

# Path regexes (same patterns as server.py)
_INFO_RE = re.compile(r"^/neuroglancer/info/(?P<token>[^/?]+)")
_SKELETON_INFO_RE = re.compile(r"^/neuroglancer/skeletoninfo/(?P<token>[^/?]+)")
_DATA_RE = re.compile(
    r"^/neuroglancer/(?P<data_format>[^/]+)/(?P<token>[^/]+)"
    r"/(?P<scale_key>[^/]+)/(?P<start>[0-9]+(?:,[0-9]+)*)/(?P<end>[0-9]+(?:,[0-9]+)*)"
)
_MESH_RE = re.compile(r"^/neuroglancer/mesh/(?P<key>[^/?]+)/(?P<object_id>[0-9]+)")
_SKELETON_RE = re.compile(
    r"^/neuroglancer/skeleton/(?P<key>[^/?]+)/(?P<object_id>[0-9]+)"
)
_ROI_FILTER_RE = re.compile(r"^/neuroglancer/roi_filter/(?P<scope>[^/?]+)")
_ACTION_RE = re.compile(r"^/action/(?P<viewer_token>[^/?]+)")
_VOLUME_INFO_RESP_RE = re.compile(
    r"^/volume_response/(?P<viewer_token>[^/]+)/(?P<request_id>[^/]+)/info"
)
_VOLUME_CHUNK_RESP_RE = re.compile(
    r"^/volume_response/(?P<viewer_token>[^/]+)/(?P<request_id>[^/]+)/chunk"
)
_EVENTS_RE = re.compile(r"^/events/(?P<viewer_token>[^/?]+)")
_SET_STATE_RE = re.compile(r"^/state/(?P<viewer_token>[^/?]+)")
_CREDENTIALS_RE = re.compile(r"^/credentials/(?P<viewer_token>[^/?]+)")


class BrowserViewerServer:
    """Virtual HTTP server that runs entirely within a Pyodide Web Worker.

    Exposes ``pyodide_handle_request`` on ``js.globalThis`` so the Service
    Worker can synchronously call Python for each intercepted request.
    Pushes SSE events via ``js.globalThis.pyodide_push_sse``.
    """

    def __init__(self):
        self.viewers: dict = {}
        # Built on first use: importing the tractography package pulls in numpy,
        # which most viewers never need.
        self._roi_filter_service = None
        self._setup_js_bridge()

    def _setup_js_bridge(self):
        import js  # type: ignore[import]
        from pyodide.ffi import create_proxy  # type: ignore[import]

        import __main__

        # Keep a Python reference to the proxy so it is not garbage collected.
        self._handle_request_proxy = create_proxy(self._js_handle_request)
        # Expose in Python's __main__ globals so the JS worker can retrieve it
        # via pyodide.globals.get("pyodide_handle_request").  Writing to
        # js.globalThis from Python is unreliable for this direction.
        __main__.pyodide_handle_request = self._handle_request_proxy
        js.globalThis.pyodide_server_ready = True

    def _js_handle_request(self, url: str, method: str, body_js) -> dict:
        """Called from JavaScript for each intercepted HTTP request.

        Parameters
        ----------
        url:
            Full URL of the intercepted request.
        method:
            HTTP method (``"GET"`` or ``"POST"``).
        body_js:
            JS ArrayBuffer (or ``None``) containing the request body.

        Returns
        -------
        dict with keys ``status`` (int), ``contentType`` (str), ``body`` (bytes).
        """
        try:
            # Pyodide <=0.27 converted JS `null` to Python `None`; newer
            # versions hand over a `JsNull` sentinel instead, so that `null`
            # and `undefined` stay distinguishable. `JsNull` is not `None` and
            # has no `to_py`, so test for convertibility rather than identity.
            if body_js is None:
                body_bytes = b""
            elif isinstance(body_js, bytes | bytearray | memoryview):
                body_bytes = bytes(body_js)
            elif not hasattr(body_js, "to_py"):
                body_bytes = b""
            else:
                converted = body_js.to_py()
                body_bytes = b"" if converted is None else bytes(converted)
            status, content_type, response_body = self._route_request(
                url, method, body_bytes
            )
        except BaseException:
            import traceback

            tb = traceback.format_exc()
            print(f"[browser_server] ERROR handling {method} {url}:\n{tb}")
            status, content_type, response_body = 500, "text/plain", tb.encode()

        import js as _js

        result = _js.Object.new()
        result.status = status
        result.contentType = content_type
        result.body = response_body
        return result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_request(
        self, url: str, method: str, body: bytes
    ) -> tuple[int, str, bytes]:
        # Strip the origin and query string from the URL to get just the path.
        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            path = parsed.path
            query = parse_qs(parsed.query)
        except Exception:
            return 400, "text/plain", b"Bad URL"

        # Checked before _DATA_RE: both live under /neuroglancer/, and although
        # the shapes do not currently collide, the specific route should win.
        m = _ROI_FILTER_RE.match(path)
        if m and method == "POST":
            return self._handle_roi_filter(m.group("scope"), body, query)

        m = _INFO_RE.match(path)
        if m:
            return self._handle_volume_info(m.group("token"))

        m = _SKELETON_INFO_RE.match(path)
        if m:
            return self._handle_skeleton_info(m.group("token"))

        m = _DATA_RE.match(path)
        if m:
            return self._handle_subvolume(
                m.group("data_format"),
                m.group("token"),
                m.group("scale_key"),
                m.group("start"),
                m.group("end"),
            )

        m = _MESH_RE.match(path)
        if m:
            return self._handle_mesh(m.group("key"), int(m.group("object_id")))

        m = _SKELETON_RE.match(path)
        if m:
            return self._handle_skeleton(m.group("key"), int(m.group("object_id")))

        m = _ACTION_RE.match(path)
        if m and method == "POST":
            return self._handle_action(m.group("viewer_token"), body)

        m = _VOLUME_INFO_RESP_RE.match(path)
        if m and method == "POST":
            return self._handle_volume_info_response(
                m.group("viewer_token"), m.group("request_id"), body
            )

        m = _VOLUME_CHUNK_RESP_RE.match(path)
        if m and method == "POST":
            return self._handle_volume_chunk_response(
                m.group("viewer_token"), m.group("request_id"), body, query
            )

        m = _EVENTS_RE.match(path)
        if m:
            return self._handle_events(m.group("viewer_token"), query)

        m = _SET_STATE_RE.match(path)
        if m and method == "POST":
            return self._handle_set_state(m.group("viewer_token"), body)

        m = _CREDENTIALS_RE.match(path)
        if m:
            return 403, "text/plain", b"Credentials not supported in Pyodide mode"

        return 404, "text/plain", b"Not found"

    # ------------------------------------------------------------------
    # Viewer / volume helpers
    # ------------------------------------------------------------------

    def register_viewer(self, viewer):
        self.viewers[viewer.token] = viewer

    def get_volume(self, key: str):
        """Resolve ``{viewerToken}.{volumeToken}`` → LocalVolume/SkeletonSource."""
        dot = key.find(".")
        if dot == -1:
            return None
        viewer_token = key[:dot]
        volume_token = key[dot + 1 :]
        viewer = self.viewers.get(viewer_token)
        if viewer is None:
            return None
        return viewer.volume_manager.volumes.get(volume_token)

    def push_sse_event(self, token: str, data: str):
        """Push a Server-Sent Event to the frontend via the Service Worker."""
        import js  # type: ignore[import]

        js.globalThis.pyodide_push_sse(token, data)

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    def _handle_volume_info(self, token: str) -> tuple[int, str, bytes]:
        vol = self.get_volume(token)
        if vol is None or not isinstance(vol, local_volume.LocalVolume):
            return 404, "application/json", b"null"
        body = json.dumps(vol.info(), default=json_encoder_default).encode()
        return 200, "application/json", body

    def _handle_skeleton_info(self, token: str) -> tuple[int, str, bytes]:
        vol = self.get_volume(token)
        if vol is None or not isinstance(vol, skeleton.SkeletonSource):
            return 404, "application/json", b"null"
        body = json.dumps(vol.info(), default=json_encoder_default).encode()
        return 200, "application/json", body

    def _handle_subvolume(
        self,
        data_format: str,
        token: str,
        scale_key: str,
        start: str,
        end: str,
    ) -> tuple[int, str, bytes]:
        start_pos = np.array(start.split(","), dtype=np.int64)
        end_pos = np.array(end.split(","), dtype=np.int64)
        vol = self.get_volume(token)
        if vol is None or not isinstance(vol, local_volume.LocalVolume):
            return 404, "text/plain", b"Volume not found"
        try:
            data, content_type = vol.get_encoded_subvolume(
                data_format=data_format,
                start=start_pos,
                end=end_pos,
                scale_key=scale_key,
            )
        except ValueError as e:
            return 400, "text/plain", str(e).encode()
        return 200, content_type, data

    def _handle_mesh(self, key: str, object_id: int) -> tuple[int, str, bytes]:
        vol = self.get_volume(key)
        if vol is None or not isinstance(vol, local_volume.LocalVolume):
            return 404, "text/plain", b"Volume not found"
        try:
            encoded = vol.get_object_mesh(object_id)
        except local_volume.MeshImplementationNotAvailable:
            return 501, "text/plain", b"Mesh implementation not available"
        except local_volume.MeshesNotSupportedForVolume:
            return 405, "text/plain", b"Meshes not supported for this volume"
        except local_volume.InvalidObjectIdForMesh:
            return 404, "text/plain", b"Mesh not available for object id"
        except ValueError as e:
            return 400, "text/plain", str(e).encode()
        return 200, "application/octet-stream", encoded

    def _handle_skeleton(self, key: str, object_id: int) -> tuple[int, str, bytes]:
        vol = self.get_volume(key)
        if vol is None or not isinstance(vol, skeleton.SkeletonSource):
            return 404, "text/plain", b"Skeleton source not found"
        try:
            skel = vol.get_skeleton(object_id)
            if skel is None:
                return 404, "text/plain", b"Skeleton not found"
            encoded = skel.encode(vol)
        except Exception as e:
            return 500, "text/plain", str(e).encode()
        return 200, "application/octet-stream", encoded

    def _handle_roi_filter(
        self, scope: str, body: bytes, query: dict
    ) -> tuple[int, str, bytes]:
        """Evaluate one ROI dissection for the viewer's chunk backend worker.

        Called directly from that worker via the Service Worker, so it never
        touches viewer state and nothing here reaches the URL. Synchronous, like
        every route here: it holds the Python thread for the duration of the
        dissection, which is why the geometry is cached rather than re-uploaded
        (the arithmetic is milliseconds; re-parsing megabytes would not be).
        """
        from .tractography.service import RoiFilterService
        from .tractography.wire import decode_request

        service = self._roi_filter_service
        if service is None:
            service = self._roi_filter_service = RoiFilterService()
        try:
            params = query.get("p", ["{}"])[0]
            request = decode_request(params, body)
        except Exception as e:
            return 400, "text/plain", f"bad roi_filter request: {e}".encode()
        if request.scope != scope:
            return 400, "text/plain", b"roi_filter scope mismatch"
        try:
            return 200, "application/octet-stream", service.handle(request)
        except KeyError as e:
            # Geometry this process no longer has -- the worker restarted under
            # a viewer that still believes it is cached. Naming the chunks lets
            # the viewer re-upload them rather than silently mis-filtering.
            missing = e.args[0] if e.args else []
            return (
                409,
                "application/json",
                encode_json({"missing": list(missing)}).encode(),
            )
        except Exception as e:
            return 500, "text/plain", f"roi_filter failed: {e}".encode()

    def _handle_action(self, viewer_token: str, body: bytes) -> tuple[int, str, bytes]:
        viewer = self.viewers.get(viewer_token)
        if viewer is None:
            return 404, "text/plain", b"Viewer not found"
        try:
            action = json.loads(body)
            viewer.actions.invoke(action["action"], action["state"])
        except Exception as e:
            return 400, "text/plain", str(e).encode()
        return 200, "text/plain", b""

    def _handle_volume_info_response(
        self, viewer_token: str, request_id: str, body: bytes
    ) -> tuple[int, str, bytes]:
        viewer = self.viewers.get(viewer_token)
        if viewer is None:
            return 404, "text/plain", b"Viewer not found"
        info = json.loads(body)
        viewer._handle_volume_info_reply(request_id, info)
        return 200, "text/plain", b""

    def _handle_volume_chunk_response(
        self,
        viewer_token: str,
        request_id: str,
        body: bytes,
        query: dict,
    ) -> tuple[int, str, bytes]:
        viewer = self.viewers.get(viewer_token)
        if viewer is None:
            return 404, "text/plain", b"Viewer not found"
        p_list = query.get("p", [])
        if not p_list:
            return 400, "text/plain", b"Missing p parameter"
        params = json.loads(p_list[0])
        viewer._handle_volume_chunk_reply(request_id, params, body)
        return 200, "text/plain", b""

    def _handle_events(self, viewer_token: str, query: dict) -> tuple[int, str, bytes]:
        """Send current state immediately when the SSE connection is established.

        The actual SSE stream is maintained by the Service Worker.  This handler
        only pushes the initial state so the frontend doesn't need to wait for
        the next change event.
        """
        viewer = self.viewers.get(viewer_token)
        if viewer is None:
            return 404, "text/plain", b"Viewer not found"

        client_id = (query.get("c") or [""])[0]

        states = [("c", viewer.config_state)]
        if hasattr(viewer, "shared_state"):
            states.append(("s", viewer.shared_state))

        for key, state in states:
            last_gen = (query.get(f"g{key}") or [""])[0]
            raw_state, generation = state.raw_state_and_generation
            if generation != last_gen and not generation.startswith(client_id + "/"):
                msg = {"k": key, "s": raw_state, "g": generation}
                self.push_sse_event(viewer_token, encode_json(msg))

        return 200, "text/event-stream", b""

    def _handle_set_state(
        self, viewer_token: str, body: bytes
    ) -> tuple[int, str, bytes]:
        viewer = self.viewers.get(viewer_token)
        if viewer is None:
            return 404, "text/plain", b"Viewer not found"
        msg = json.loads(body)
        prev_generation = msg["pg"]
        generation = msg["g"]
        state = msg["s"]
        client_id = msg["c"]
        try:
            new_generation = viewer.set_state(
                state,
                f"{client_id}/{generation}",
                existing_generation=prev_generation,
            )
            return 200, "application/json", json.dumps({"g": new_generation}).encode()
        except ConcurrentModificationError:
            return 412, "text/plain", b""


# Module-level singleton
_browser_server: BrowserViewerServer | None = None


def get_browser_server() -> BrowserViewerServer:
    """Return (and lazily create) the module-level BrowserViewerServer."""
    global _browser_server
    if _browser_server is None:
        _browser_server = BrowserViewerServer()
    return _browser_server
