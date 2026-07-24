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
"""In-browser tract export under Pyodide: select, then write, from one async read.

The native exporter (:mod:`neuroglancer.tract_export.run`) reads and writes
through zarr's **synchronous** API, which is fine under CPython. In the browser
that path deadlocks -- a synchronous zarr call needs JSPI stack switching, and
two of those wedge the worker (see the ``handleRequest`` note in
``src/python_integration/pyodide_worker.ts``). So this module drives the read
through :func:`zarr_vectors.core.aio.read_async` -- the async prime-and-replay
path that never reaches ``sync()`` -- and returns the finished file as *bytes*
for the page to download.

Selection is normally done *before* export by the viewer's on-screen dissection:
a v3 job carries each group's passing object ids (``objectIds``), and this module
reads exactly those objects through ``read_polylines``'s selective branch -- no
whole-level read, no fold. That is what makes an in-browser export fast and
convergent even on a whole-brain store. A legacy spec without ``objectIds`` still
folds via :func:`neuroglancer.tract_export.run.select_from` over a whole-level
read, choosing the same streamlines a native export would.

The ``.zvf`` writer (``write_polylines``) still calls zarr ``sync()``, so a ZVF
export must be driven from JS through a single *promising* entrypoint held under
a host mutex (``browser_server`` + ``pyodide_worker``); nothing here starts a
thread or lists the store. TRK needs no ``sync()`` and only an ``await`` for the
read.

Imports of zarr / zarr-vectors / nibabel are lazy, so this module loads even
where they are absent; :func:`export_async` then raises a clear
:class:`~neuroglancer.tract_export.run.ExportRunError`.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from neuroglancer.tract_export.job import ExportJob
from neuroglancer.tract_export.run import (
    ExportRunError,
    _uses_explicit_ids,
    per_group_from_object_ids,
    select_from,
    store_path_from_url,
    union_ids,
)

FetchFn = Callable[[str], Awaitable["bytes | None"]]

_MISSING_LIB = (
    "In-browser export needs the `zarr-vectors` package bundled into the "
    "Pyodide runtime. Rebuild with NEUROGLANCER_PYODIDE_PACKAGES pointing at the "
    "zarr-vectors-py checkout, or use the native exporter."
)

#: Where the writers stage their output before it is read back as bytes. Under
#: Pyodide this is the in-memory MEMFS; the file never touches a real disk.
_TRK_TMP = "/tmp/ng_export.trk"
_ZVF_TMP = "/tmp/ng_export.zvf"


def _http_base(base: str) -> str:
    """Map a ``gs://`` store URL to the HTTPS endpoint the browser can fetch.

    The render path reads through neuroglancer's GCS driver, but the in-browser
    exporter fetches over plain HTTPS via ``pyfetch``, which cannot read
    ``gs://``. ``gs://bucket/path`` becomes
    ``https://storage.googleapis.com/bucket/path`` -- the same object endpoint an
    ``https://storage.googleapis.com/...`` source already uses. Non-``gs://``
    URLs pass through unchanged.
    """
    if base.startswith("gs://"):
        return "https://storage.googleapis.com/" + base[len("gs://") :]
    return base


async def export_async(
    job: ExportJob, fetch: FetchFn | None = None
) -> tuple[bytes, str, dict[str, Any]]:
    """Carry out ``job`` in the browser.

    Returns ``(body, content_type, summary)``:

    * a successful TRK export -> the ``.trk`` bytes, ``application/octet-stream``;
    * a successful ZVF export -> the zipped store bytes, ``application/zip``;
    * an empty selection (or nothing to write) -> the summary encoded as JSON,
      ``application/json`` -- there is no file, and the tab reports the message.

    ``summary`` is always returned as the third element for callers/tests even
    when it is also the body.
    """
    if fetch is None:
        from neuroglancer.tractography.zarr_vectors_source import _default_fetch

        fetch = _default_fetch()

    base = _http_base(store_path_from_url(job.source_url))
    if not base.endswith("/"):
        base = base + "/"

    per_group: list[tuple[str, np.ndarray]]
    if job.scope == "whole":
        # Every object at the level, no ROI fold. Inherently a whole-level read
        # (and memory-gated in-browser), so this path is unchanged.
        result = await _read_polylines_async(base, job.level, fetch)
        object_ids = list(result["object_ids"])
        ids = np.unique(np.asarray(object_ids, dtype=np.uint64))
        considered = len(object_ids)
        per_group = []
        streamlines = (
            _streamlines_for(ids, object_ids, result["polylines"])
            if ids.size
            else []
        )
    elif _uses_explicit_ids(job.groups):
        # v3 fast path: read ONLY the objects the viewer selected on screen.
        # `object_ids=` takes read_polylines's selective (O(subset)) branch, so
        # the read_async discovery loop converges in a few rounds instead of
        # re-running the whole-level cross-chunk reassembly every round. This is
        # the fix for "export is painfully slow / doesn't converge in the wasm".
        per_group = per_group_from_object_ids(job.groups)
        ids = union_ids(per_group)
        considered = int(ids.size)
        if ids.size:
            result = await _read_polylines_async(
                base, job.level, fetch, object_ids=[int(i) for i in ids]
            )
            streamlines = _streamlines_from_result(result)
        else:
            streamlines = []
    else:
        # Legacy fold path (spec without objectIds): read the whole level, then
        # fold rois in-memory. release=False: the geometry is reused to build the
        # output from this same read rather than fetching the store again.
        result = await _read_polylines_async(base, job.level, fetch)
        polylines = result["polylines"]
        object_ids = list(result["object_ids"])
        considered, per_group = select_from(
            polylines, object_ids, job.groups, release=False
        )
        ids = union_ids(per_group)
        streamlines = (
            _streamlines_for(ids, object_ids, polylines) if ids.size else []
        )

    summary: dict[str, Any] = {
        "object_count": int(ids.size),
        "candidate_object_count": considered,
        "groups": [{"name": name, "count": int(v.size)} for name, v in per_group],
        "output_path": job.destination.path,
    }

    if ids.size == 0:
        summary.update(
            written=False,
            streamline_count=0,
            vertex_count=0,
            message="No streamlines passed this dissection; nothing written.",
        )
        return json.dumps(summary).encode("utf-8"), "application/json", summary

    summary.update(
        written=True,
        streamline_count=len(streamlines),
        vertex_count=int(sum(len(s) for s in streamlines)),
    )

    if job.format == "trk":
        await _ensure_nibabel()
        body, content_type = _write_trk_bytes(streamlines, job.affine)
    else:
        chunk_shape = await _fetch_root_chunk_shape(base, fetch)
        try:
            body, content_type = _write_zvf_zip_bytes(streamlines, chunk_shape)
        except ExportRunError:
            raise
        except Exception as e:  # noqa: BLE001
            # write_polylines drives zarr's SYNCHRONOUS writer, which under
            # Pyodide can only suspend via JSPI. Where the runtime lacks
            # WebAssembly stack switching it fails here -- TRK export (async
            # read + pure write) and the native exporter still work.
            msg = str(e)
            if any(k in msg.lower() for k in ("stack switching", "suspend", "jspi")):
                raise ExportRunError(
                    "ZVF export needs WebAssembly stack switching (JSPI), which "
                    "this browser does not support. Export as TRK instead, or "
                    "use the native exporter (Download job spec)."
                ) from e
            raise ExportRunError(f"ZVF export failed: {msg}") from e
    return body, content_type, summary


async def _ensure_nibabel() -> None:
    """Make ``nibabel`` importable, installing it via micropip under Pyodide.

    nibabel is pure-Python and not in the Pyodide distribution, so it is
    installed on first TRK export rather than bundled. A no-op where it is
    already present (CPython, or a warm Pyodide session).
    """
    try:
        import nibabel  # noqa: F401

        return
    except ImportError:
        pass
    try:
        import micropip  # type: ignore[import]
    except ImportError as e:
        raise ExportRunError(
            "TRK export needs `nibabel`, which is not installed and micropip is "
            "unavailable to fetch it."
        ) from e
    await micropip.install("nibabel")


async def _read_polylines_async(
    base: str,
    level: int,
    fetch: FetchFn,
    object_ids: list[int] | None = None,
) -> dict:
    """Read a level through the async (never-``sync()``) path.

    With ``object_ids=None`` this reads the whole level once (the whole-store /
    legacy-fold path). With an explicit ``object_ids`` subset it takes
    ``read_polylines``'s selective branch -- reading only those objects' manifests
    and the chunks they reference -- which is O(subset). That matters under
    Pyodide: ``read_async`` re-runs the reader against a growing offline snapshot
    each round, so a whole-level read reassembles every tract every round and can
    take many passes, whereas the subset read settles quickly.
    """
    try:
        from zarr_vectors.core.aio import open_store_async, read_async
        from zarr_vectors.types.polylines import read_polylines

        from neuroglancer.tractography.browser_fetch_store import (
            make_browser_fetch_store,
        )
    except Exception as e:  # noqa: BLE001 - any import failure means "not bundled"
        raise ExportRunError(_MISSING_LIB) from e

    store = make_browser_fetch_store(base, fetch)
    root = await open_store_async(store)
    kwargs: dict[str, Any] = {"level": level}
    if object_ids is not None:
        kwargs["object_ids"] = object_ids
    result = await read_async(read_polylines, root, **kwargs)
    if result.get("object_ids") is None:
        raise ExportRunError(
            "read_polylines did not report object_ids, so the dissection cannot "
            "be mapped back to objects. A newer zarr-vectors is needed."
        )
    return result


def _streamlines_from_result(result: dict) -> list[np.ndarray]:
    """One concatenated polyline per returned object, in the read's order.

    For the selective (``object_ids=``) read the result already holds exactly the
    requested objects -- each a list of fragment arrays -- so concatenating each
    object's fragments yields whole tracts, matching ``_streamlines_for`` and the
    native ``export_trk``.
    """
    return [
        np.concatenate(fragments, axis=0).astype(np.float32, copy=False)
        for fragments in result["polylines"]
    ]


def _streamlines_for(
    ids: np.ndarray, object_ids: list, polylines: list
) -> list[np.ndarray]:
    """One concatenated polyline per selected id, in ``ids`` order.

    Fragments are joined per object -- the same thing ``run.build_index`` and
    the native ``export_trk`` do -- so endpoint predicates and the written
    geometry see whole tracts, not chunk fragments.
    """
    first: dict[int, int] = {}
    for i, oid in enumerate(object_ids):
        first.setdefault(int(oid), i)
    out: list[np.ndarray] = []
    for oid in ids:
        fragments = polylines[first[int(oid)]]
        out.append(np.concatenate(fragments, axis=0).astype(np.float32, copy=False))
    return out


def _write_trk_bytes(
    streamlines: list[np.ndarray], affine: np.ndarray | None
) -> tuple[bytes, str]:
    """Assemble a TrackVis ``.trk`` in memory, with a header tools can place.

    The streamlines are in the store's own (voxel) coordinates and ``affine`` is
    the voxel→RAS matrix **in millimetres** (the Export tab pre-fills it from the
    layer's scales × 1000; TRK is a millimetre format). We must fill the header's
    voxel size, dimensions and voxel→RAS explicitly: nibabel otherwise leaves them
    at ``(1,1,1)`` / identity and just bakes the affine into the points, which
    both loses the physical frame (freeview then refuses to place the tract) and,
    when the affine carries the metre→millimetre factor, is the only thing that
    keeps a whole brain from reading as a sub-millimetre speck.
    """
    try:
        import nibabel as nib
        from nibabel.streamlines.trk import TrkFile
    except ImportError as e:
        raise ExportRunError(
            "TRK export needs `nibabel`, which is not available in this Pyodide "
            "runtime (it installs via micropip)."
        ) from e

    aff = (
        np.eye(4, dtype=np.float32)
        if affine is None
        else np.asarray(affine, dtype=np.float32)
    )
    header = _trk_header(aff, streamlines)
    tractogram = nib.streamlines.Tractogram(
        streamlines=streamlines,
        affine_to_rasmm=aff,
    )
    buf = io.BytesIO()
    TrkFile(tractogram=tractogram, header=header).save(buf)
    return buf.getvalue(), "application/octet-stream"


def _trk_header(aff: np.ndarray, streamlines: list[np.ndarray]) -> dict:
    """A TRK header with a real voxel size / dimensions / voxel→RAS.

    ``voxel_sizes`` are the norms of the affine's direction columns;
    ``dimensions`` is a reference grid that bounds the exported vertices (they are
    in voxel space), so freeview has a non-degenerate volume to place the tract
    against. TRK stores dimensions as int16, so the grid is clamped to that range;
    the tracts still render in RAS regardless of the exact grid.
    """
    from nibabel.streamlines import Field

    voxel_sizes = np.linalg.norm(aff[:3, :3], axis=0).astype(np.float32)
    if streamlines:
        all_pts = np.concatenate(streamlines, axis=0)
        maxc = np.ceil(np.abs(all_pts).max(axis=0)).astype(np.int64) + 1
        dims = np.clip(maxc, 1, 32767).astype(np.int16)
    else:
        dims = np.array([1, 1, 1], dtype=np.int16)
    return {
        Field.VOXEL_SIZES: tuple(float(v) for v in voxel_sizes),
        Field.DIMENSIONS: tuple(int(v) for v in dims),
        Field.VOXEL_TO_RASMM: aff,
    }


def _write_zvf_zip_bytes(
    streamlines: list[np.ndarray], chunk_shape: tuple[float, ...]
) -> tuple[bytes, str]:
    """Write a fresh zarr-vectors store to MEMFS, return it zipped.

    A ``.zvf`` is a directory store; the browser can only download a single
    file, so it is zipped (stored, not deflated -- the chunks may already be
    compressed). ``write_polylines`` drives zarr's synchronous writer, so this
    must run under the promising export entrypoint (see the module docstring).
    """
    try:
        from zarr_vectors.types.polylines import write_polylines
    except Exception as e:  # noqa: BLE001
        raise ExportRunError(_MISSING_LIB) from e

    _remove(_ZVF_TMP)  # write_polylines refuses to write into a non-empty dir
    write_polylines(_ZVF_TMP, streamlines, chunk_shape=tuple(chunk_shape))
    try:
        return _zip_dir_bytes(_ZVF_TMP), "application/zip"
    finally:
        _remove(_ZVF_TMP)


async def _fetch_root_chunk_shape(base: str, fetch: FetchFn) -> tuple[float, ...]:
    """The store's spatial ``chunk_shape``, read from the root ``zarr.json``.

    Carried into the new store so the export re-chunks the geometry the way the
    source was chunked, matching the native ``_export_zvf``. Read by a direct
    async fetch rather than the sync metadata helpers, which cannot run over
    HTTP under Pyodide.
    """
    raw = await fetch(base + "zarr.json")
    if raw is None:
        raise ExportRunError(
            f"Could not read root metadata at {base}zarr.json to determine the "
            f"ZVF chunk shape."
        )
    meta = json.loads(raw.decode("utf-8"))
    # zarr v3 nests group attributes under "attributes"; tolerate a bare dict.
    attrs = meta.get("attributes", meta)
    try:
        chunk_shape = attrs["zarr_vectors"]["chunk_shape"]
    except (KeyError, TypeError) as e:
        raise ExportRunError(
            "Root metadata has no zarr_vectors.chunk_shape; cannot write a ZVF "
            "store with a matching grid."
        ) from e
    if not chunk_shape:
        raise ExportRunError("Root metadata zarr_vectors.chunk_shape is empty.")
    return tuple(chunk_shape)


def _zip_dir_bytes(path: str) -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(path):
            for name in sorted(files):
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, path))
    return buf.getvalue()


def _remove(path: str) -> None:
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
