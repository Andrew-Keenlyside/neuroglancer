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

"""Pyodide-specific Viewer implementations.

These replace the Tornado-backed ``Viewer`` and ``UnsynchronizedViewer`` when
running inside a Pyodide Web Worker.  Instead of registering with a TCP server,
they register with :mod:`browser_server` and push state changes as SSE events
via ``js.globalThis.pyodide_push_sse``.
"""

from . import browser_server, viewer_base
from .json_utils import encode_json


class _PyodideViewerHelper:
    """Mixin that wires a viewer into the Pyodide browser-server infrastructure.

    Responsibilities:
    - Register with :class:`~browser_server.BrowserViewerServer`.
    - Attach changed-callbacks on ``config_state`` (and ``shared_state`` if
      present) that push SSE events to the JS frontend whenever state changes.
    - Provide a ``defer_callback`` implementation that uses ``js.setTimeout``.

    Note: ``ViewerCommonBase.__init__`` does not call ``super().__init__()``, so
    this mixin's ``__init__`` is never reached via the MRO chain.  Subclasses
    must call ``_pyodide_setup()`` explicitly after their base ``__init__``.
    """

    def _pyodide_setup(self):
        """Wire up the Pyodide browser-server.  Call after ViewerBase.__init__."""
        self._bs = browser_server.get_browser_server()
        self._bs.register_viewer(self)
        # Wire up state-change → SSE push
        self.config_state.add_changed_callback(
            lambda: self._push_state_changed("c", self.config_state)
        )
        if hasattr(self, "shared_state"):
            self.shared_state.add_changed_callback(
                lambda: self._push_state_changed("s", self.shared_state)
            )

    def _push_state_changed(self, key: str, state):
        raw_state, generation = state.raw_state_and_generation
        # Client-originated updates have generation "{client_id}/{gen}" (contains "/").
        # Echoing those back causes mid-gesture state interference (e.g. drag duplication).
        # Only push when Python itself modified the state (random token, no "/").
        if "/" in generation:
            return
        msg = {"k": key, "s": raw_state, "g": generation}
        self._bs.push_sse_event(self.token, encode_json(msg))

    def defer_callback(self, callback, *args, **kwargs):
        import js  # type: ignore[import]
        from pyodide.ffi import create_proxy  # type: ignore[import]

        js.setTimeout(create_proxy(lambda: callback(*args, **kwargs)), 0)

    def __repr__(self):
        return f"PyodideViewer(token={self.token!r})"

    def _repr_html_(self):
        return f"<b>PyodideViewer</b> (token: {self.token})"

    def get_viewer_url(self):
        # The page itself is the viewer; no external URL is needed.
        return ""


class PyodideViewer(viewer_base.ViewerBase, _PyodideViewerHelper):
    """Neuroglancer viewer backed by the Pyodide browser-server (synchronized).

    Equivalent to :class:`~viewer.Viewer` but works without a Tornado server.
    The token defaults to ``"pyodide"`` to match the fixed URL ``/v/pyodide/``
    used by the JS frontend.
    """

    def __init__(self, token: str = "pyodide", **kwargs):
        super().__init__(token=token, **kwargs)
        self._pyodide_setup()


class PyodideUnsynchronizedViewer(
    viewer_base.UnsynchronizedViewerBase, _PyodideViewerHelper
):
    """Neuroglancer viewer backed by the Pyodide browser-server (unsynchronized).

    Equivalent to :class:`~viewer.UnsynchronizedViewer` but works without a
    Tornado server.
    The token defaults to ``"pyodide"`` to match the fixed URL ``/v/pyodide/``
    used by the JS frontend.
    """

    def __init__(self, token: str = "pyodide", **kwargs):
        super().__init__(token=token, **kwargs)
        self._pyodide_setup()
