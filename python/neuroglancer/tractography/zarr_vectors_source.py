# @license
# Copyright 2026 Allen Institute for Brain Science
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

"""Read a coarse pyramid level of a zarr-vectors store into a `TractIndex`.

**Not used by the streamline filter.** The viewer already has the geometry
decoded, at the pyramid level it is actually drawing, and hands it over -- see
`service.py`. This is a standalone utility for using the dissection geometry
from a Python script.

The read goes through the real ``zarr-vectors-py`` library via its async API
(``zarr_vectors.core.aio.read_async``): it builds a fetch-backed zarr Store
(``browser_fetch_store.py``), opens it with ``open_store_async``, and runs
``read_polylines`` through ``read_async``. That is the canonical decoder -- one
copy of the format, with the compression and sharding support the old
hand-rolled decoder (deleted; see git history and ``zvf_pure.py`` for the pure
byte codecs it left behind) never had.

Availability: the async path needs ``zarr`` and the bundled ``zarr_vectors``
package, which are present under Pyodide but not this repo's plain CPython env.
There is no longer a built-in fallback -- :func:`load_tract_index` raises when
the library is absent. The pure byte helpers re-exported below (from
``zvf_pure``) remain usable anywhere.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

import numpy as np

from .index import TractIndex

# Re-exported for back-compat: these moved to `zvf_pure` (zarr-free, testable
# offline) but callers and tests still import them from here.
from .zvf_pure import (  # noqa: F401
    FRAGMENT_INDEX_MAGIC,
    FRAGMENT_INDEX_VERSION,
    _polylines_to_tract_index,
    apply_byte_range,
    decode_fragment_index,
    decode_vlen_bytes,
)

__all__ = [
    "load_tract_index",
    "decode_vlen_bytes",
    "decode_fragment_index",
    "apply_byte_range",
]


# -- fetching ---------------------------------------------------------------


async def _pyodide_fetch(url: str) -> bytes | None:
    from pyodide.http import pyfetch  # type: ignore

    try:
        response = await pyfetch(url)
    except AttributeError:
        # pyodide.http raises AbortError(reason) on a failed/aborted fetch, and
        # AbortError.__init__ reads `reason.message` -- which itself throws a bare
        # AttributeError when the abort carries no reason (network error, blocked
        # request, an oversized response). Surface something usable rather than
        # letting that masking error propagate.
        raise OSError(
            f"fetch of {url} failed (network error, CORS, or the response was "
            f"too large to load into browser memory)"
        ) from None
    if response.status == 404:
        return None
    if response.status >= 400:
        raise OSError(f"{response.status} fetching {url}")
    try:
        return await response.bytes()
    except AttributeError:
        raise OSError(
            f"reading the body of {url} failed -- the fetch was aborted, often "
            f"because the object is too large to hold in browser memory. Exporting "
            f"a whole-brain level 0 in-browser is not feasible; use the native "
            f"exporter (Download job spec)."
        ) from None


async def _urllib_fetch(url: str) -> bytes | None:
    """CPython-only fallback, for running this outside the browser.

    Never selected under Pyodide -- `asyncio.to_thread` would fail there, since
    Pyodide cannot start threads. `_default_fetch` picks `_pyodide_fetch`
    whenever `pyodide.http` imports, so this is only reachable off-platform.
    """
    import urllib.error
    import urllib.request

    def get():
        try:
            with urllib.request.urlopen(url) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    return await asyncio.to_thread(get)


def _default_fetch() -> Callable[[str], Awaitable[bytes | None]]:
    try:
        import pyodide.http  # noqa: F401
    except ImportError:
        return _urllib_fetch
    return _pyodide_fetch


# -- reading through zarr-vectors-py -----------------------------------------


async def _load_via_zarr_vectors(
    base: str,
    level: int,
    fetch: Callable[[str], Awaitable[bytes | None]],
) -> TractIndex | None:
    """Read the level through zarr-vectors-py, or return None if unavailable.

    Returns None (rather than raising) when the library, zarr, or the async read
    is unavailable, so :func:`load_tract_index` can raise a single clear error.
    """
    try:
        from zarr_vectors.core.aio import open_store_async, read_async
        from zarr_vectors.types.polylines import read_polylines

        from .browser_fetch_store import make_browser_fetch_store
    except Exception:
        return None
    try:
        store = make_browser_fetch_store(base, fetch)
        root = await open_store_async(store)
        result = await read_async(read_polylines, root, level=level)
        return _polylines_to_tract_index(result)
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).info(
            "zarr-vectors read of %s level %s failed (%s)", base, level, exc
        )
        return None


def _finish_index(
    index: TractIndex,
    max_vertices: int | None,
    scale: Iterable[float] | None,
    translation: Iterable[float] | None,
) -> TractIndex:
    """Apply the coordinate transform and decimation a caller may request."""
    if scale is not None or translation is not None:
        scaled = index.positions.astype(np.float32, copy=True)
        if scale is not None:
            scaled *= np.asarray(list(scale), dtype=np.float32)
        if translation is not None:
            scaled += np.asarray(list(translation), dtype=np.float32)
        index = TractIndex(scaled, index.offsets, index.row_ids)
    if max_vertices is not None:
        index = index.decimate(max_vertices)
    return index


# -- loading ----------------------------------------------------------------


async def load_tract_index(
    url: str,
    level: int = 3,
    *,
    fetch: Callable[[str], Awaitable[bytes | None]] | None = None,
    max_vertices: int | None = None,
    scale: Iterable[float] | None = None,
    translation: Iterable[float] | None = None,
) -> TractIndex:
    """Load one pyramid level of a zarr-vectors store as a `TractIndex`.

    `url` is the store root. `level` selects the pyramid level; it should be the
    one the viewer is drawing, since ids present in the index but absent from
    the drawn level simply never appear, while the converse under-selects.

    `scale` and `translation` map stored coordinates into the space the viewer's
    annotations live in -- they must match the transform the datasource applies,
    or regions will not line up with the tracts they appear to touch.

    `max_vertices` decimates each fragment, trading a little exactness in the
    `ANY_SEGMENT` test for a smaller index. See `TractIndex.decimate`.

    Raises `RuntimeError` if `zarr-vectors` / `zarr` are not importable.
    """
    if fetch is None:
        fetch = _default_fetch()
    base = url if url.endswith("/") else url + "/"

    index = await _load_via_zarr_vectors(base, level, fetch)
    if index is None:
        raise RuntimeError(
            f"Could not read {base!r}: reading a zarr-vectors store needs the "
            f"`zarr-vectors` package and zarr, which are available under "
            f"Pyodide (with the package bundled) but not in this environment."
        )
    return _finish_index(index, max_vertices, scale, translation)
