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
"""Execute an export job: re-evaluate the dissection, then write the output.

The selection is made **here**, not taken from the viewer. The viewer's passing
set covers only the chunks resident at the level of detail currently on screen,
so exporting it would quietly drop tracts. This reads the store at the job's
level (0 by default) and re-runs the same fold, which is why the exported count
normally exceeds the count shown in the Filter tab.

One consequence worth knowing: the viewer necessarily tests `either_endpoint` /
`both_endpoints` against *fragment* ends, because its geometry arrives chunked
and a tract cut across a chunk boundary has interior ends that look like
endpoints (see the note in ``tractography/roi.py``). This reads whole objects and
concatenates their fragments before testing, so endpoints here are the tract's
real ends. An endpoint dissection can therefore legitimately select a different
-- and more correct -- set on export than on screen.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from neuroglancer.tract_export.job import ExportGroup, ExportJob
from neuroglancer.tractography.index import TractIndex
from neuroglancer.tractography.roi import streamlines_pass_rois

#: Datasource prefixes neuroglancer puts on a URL that a store path must not carry.
_SCHEME_PREFIX = re.compile(r"^(zarr-vectors|zarr3?)://")


class ExportRunError(RuntimeError):
    """A job that parsed but could not be carried out."""


def store_path_from_url(url: str) -> str:
    """Strip the neuroglancer datasource scheme from a layer's source URL.

    The viewer addresses a store as ``zarr-vectors://gs://bucket/x.zvf``; the
    Python API wants the underlying ``gs://bucket/x.zvf``. Anything without a
    recognised prefix is passed through untouched, so a plain local path written
    into a hand-edited job spec still works.
    """
    return _SCHEME_PREFIX.sub("", url, count=1)


def _require_zarr_vectors():
    try:
        from zarr_vectors.types.polylines import read_polylines
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ExportRunError(
            "Exporting needs the `zarr-vectors` package, which is not "
            "installed in this interpreter. Install it with "
            "`pip install zarr-vectors`."
        ) from e
    return read_polylines


def build_index(polylines: list, object_ids: list) -> TractIndex:
    """Build a ragged index from ``read_polylines`` output.

    Each polyline becomes **one row**, its fragments concatenated -- the same
    thing ``export_trk`` does before writing. Leaving fragments as separate rows
    would make the endpoint predicates test chunk boundaries rather than tract
    ends.
    """
    if len(polylines) != len(object_ids):
        raise ExportRunError(
            f"read_polylines returned {len(polylines)} polylines but "
            f"{len(object_ids)} object ids; they must correspond"
        )
    if not polylines:
        return TractIndex(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(1, dtype=np.intp),
            np.zeros(0, dtype=np.uint64),
        )
    rows = [np.concatenate(fragments, axis=0) for fragments in polylines]
    positions = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    offsets = np.zeros(len(rows) + 1, dtype=np.intp)
    offsets[1:] = np.cumsum(np.fromiter((len(r) for r in rows), dtype=np.intp))
    return TractIndex(positions, offsets, np.asarray(object_ids, dtype=np.uint64))


#: Objects folded per batch. `TractIndex` costs ~44 bytes per vertex, of which
#: only 12 are the positions -- the rest are the derived vertex/segment arrays
#: the ROI tests need. A whole-brain tractogram runs ~200 vertices per tract, so
#: a 500k-tract level would need ~4.5 GB if indexed in one go. Batching the fold
#: holds the derived arrays for this many tracts at a time instead.
OBJECT_BATCH = 25_000


def _read_level(job: ExportJob) -> tuple[list, list]:
    """Read the whole level once. Returns `(polylines, object_ids)`.

    Deliberately ONE call with no `object_ids=`. `read_polylines` picks its
    strategy from that argument: an explicit subset makes it read each object's
    manifest individually (O(subset) round trips), while no subset makes it
    prefetch every vertex chunk in one gather (O(store)). Asking for the whole
    level object-by-object therefore takes the pathological branch -- measurably
    so over HTTP, where each manifest is its own request.

    Nor `bbox=` or `chunks=`. `bbox=` keeps a polyline only if one of its stored
    *fragments* lands in an intersecting chunk, and a fragment's chunk comes
    from where its vertices fall -- so a tract whose long segment leaps from
    chunk A to chunk C, with no vertex in the B that holds the ROI, is dropped
    before the fold ever sees it. That is the "leap-across artifact" the default
    `ANY_SEGMENT` predicate exists to avoid, reintroduced at the read. `chunks=`
    is worse: it crops at segment level and splits one tract into several.
    """
    read_polylines = _require_zarr_vectors()
    store = store_path_from_url(job.source_url)
    try:
        result = read_polylines(store, level=job.level)
    except Exception as e:
        raise ExportRunError(f"Could not read {store!r}: {e}") from e
    object_ids = result.get("object_ids")
    if object_ids is None:
        raise ExportRunError(
            "This build of `zarr-vectors` does not report `object_ids` from "
            "read_polylines, so the dissection cannot be mapped back to "
            "objects. Upgrade `zarr-vectors`."
        )
    return result["polylines"], list(object_ids)


def load_index(job: ExportJob) -> TractIndex:
    """The whole level as one index. Convenience for small levels and tests.

    Prefer :func:`select`, which folds in batches: this materialises the derived
    arrays for every vertex at once (see :data:`OBJECT_BATCH`).
    """
    polylines, object_ids = _read_level(job)
    return build_index(polylines, object_ids)


def passing_object_ids(index: TractIndex, group: ExportGroup) -> np.ndarray:
    """The object ids of `index` that survive `group`'s fold, ascending."""
    if len(index) == 0:
        return np.zeros(0, dtype=np.uint64)
    return index.object_ids[streamlines_pass_rois(index, list(group.rois))]


def select(
    job: ExportJob, batch: int | None = None
) -> tuple[int, list[tuple[str, np.ndarray]]]:
    """Evaluate every group over the level.

    The read is one pass (see :func:`_read_level`); the *fold* is batched, since
    that is where the memory goes -- 32 of `TractIndex`'s 44 bytes per vertex
    are derived arrays that exist only to answer the ROI tests. Each batch's raw
    fragments are released once indexed, so peak use falls as the pass proceeds
    rather than holding raw geometry and derived arrays simultaneously.

    Batching cannot change a verdict: the fold is decided per object, with no
    reference to any other object.

    Returns the number of objects considered and each group's passing ids. The
    per-group results are a **list** keyed by position, not a dict keyed by
    name: group names are user-editable display text with no uniqueness
    guarantee, and the viewer makes collisions easy -- it names new groups
    ``Group ${length + 1}``, so deleting a middle group and adding another
    reproduces an existing name. Keying by name silently dropped every colliding
    group but the last from the exported union.
    """
    # Resolved here rather than as a default argument, which would bind at def
    # time and make `OBJECT_BATCH` impossible to override.
    batch = OBJECT_BATCH if batch is None else batch
    polylines, object_ids = _read_level(job)

    per_group: list[list[np.ndarray]] = [[] for _ in job.groups]
    considered = 0
    for start in range(0, len(polylines), batch):
        stop = min(start + batch, len(polylines))
        index = build_index(polylines[start:stop], object_ids[start:stop])
        considered += len(index)
        for i, group in enumerate(job.groups):
            hit = passing_object_ids(index, group)
            if hit.size:
                per_group[i].append(hit)
        # `build_index` concatenated these into the index's own buffers, so the
        # originals are dead weight from here on.
        for k in range(start, stop):
            polylines[k] = None
        del index

    return considered, [
        (
            g.name,
            np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64),
        )
        for g, parts in zip(job.groups, per_group)
    ]


def _export_trk(job: ExportJob, ids: np.ndarray, output_path: str) -> dict[str, Any]:
    try:
        from zarr_vectors.exceptions import ExportError
        from zarr_vectors_tools.export.trk import export_trk
    except ImportError as e:
        # `[trk]` rather than `[streamlines]`: the extras were split so the
        # nibabel half can be installed without `trx-python`, which has no
        # pyemscripten wheel. `[streamlines]` still resolves to both and keeps
        # working. Note `[trk]` is slightly broader than its name -- it also
        # covers TCK/TRK *ingest*, which needs nibabel too.
        raise ExportRunError(
            "TRK export needs `zarr-vectors-tools` with its `trk` extra "
            "(nibabel). Install it with "
            "`pip install 'zarr-vectors-tools[trk]'`."
        ) from e

    try:
        summary = export_trk(
            store_path_from_url(job.source_url),
            output_path,
            level=job.level,
            # `object_ids`, never `chunks`: the chunk whitelist crops at the
            # segment level and would split every tract that leaves the ROI box
            # into several shorter streamlines.
            object_ids=[int(i) for i in ids],
            affine=job.affine,
        )
    except ExportError as e:
        raise ExportRunError(str(e)) from e
    return {**summary, "written": True}


def _export_zvf(job: ExportJob, ids: np.ndarray, output_path: str) -> dict[str, Any]:
    try:
        from zarr_vectors.core.metadata import get_level_chunk_shape
        from zarr_vectors.core.store import (
            open_store,
            read_level_metadata,
            read_root_metadata,
        )
        from zarr_vectors.types.polylines import read_polylines, write_polylines
    except ImportError as e:
        raise ExportRunError(
            "ZVF export needs the `zarr-vectors` package. Install it with "
            "`pip install zarr-vectors`."
        ) from e

    # Exporting creates a store; it never appends to one. `write_polylines`
    # routes through `_create_or_open_store`, which *silently ignores*
    # chunk_shape/bounds/ndim when the path already exists and keeps the old
    # store's metadata authoritative -- while still splitting the new geometry
    # on the chunk_shape passed here. The result is a store whose declared grid
    # contradicts its own chunk keys, so a later bbox read looks for
    # coordinates that were never written and silently returns nothing.
    if os.path.exists(output_path) and os.listdir(output_path):
        raise ExportRunError(
            f"{output_path!r} already exists and is not empty. Exporting "
            f"creates a new store; remove the destination or choose another."
        )

    store = store_path_from_url(job.source_url)
    root = open_store(store)
    root_meta = read_root_metadata(root)
    try:
        level_meta = read_level_metadata(root, job.level)
    except Exception:
        level_meta = None
    # Carry the source's chunk shape rather than inventing one: a mismatched
    # grid would rechunk the geometry as a side effect of exporting it.
    chunk_shape = get_level_chunk_shape(root_meta, level_meta)

    # Re-read just the selected objects rather than holding the whole level in
    # memory through selection -- the same thing `export_trk` does internally.
    # Fragments are concatenated per object, matching how the fold saw them.
    selected = read_polylines(store, level=job.level, object_ids=[int(i) for i in ids])
    polylines = [
        np.concatenate(fragments, axis=0) for fragments in selected["polylines"]
    ]

    write_polylines(output_path, polylines, chunk_shape=chunk_shape)
    return {
        "streamline_count": len(polylines),
        "vertex_count": int(sum(len(p) for p in polylines)),
        "written": True,
    }


def union_ids(per_group: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """The ascending union of every group's passing ids."""
    arrays = [ids for _, ids in per_group if ids.size]
    if not arrays:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(arrays))


def run_job(job: ExportJob) -> dict[str, Any]:
    """Carry out `job`, returning a summary suitable for the viewer."""
    considered, per_group = select(job)
    ids = union_ids(per_group)

    if job.destination.kind != "local":
        raise ExportRunError(
            f"destination.kind {job.destination.kind!r} is not supported yet; "
            f"export locally and upload the result."
        )
    output_path = job.destination.path

    if ids.size == 0:
        # Neither writer produces a file for an empty selection -- export_trk
        # raises before it saves, and there is nothing to hand write_polylines.
        # Say so, rather than reporting success at a path that holds nothing
        # (or worse, a stale file from an earlier export that would then be
        # read as this one's result).
        summary: dict[str, Any] = {
            "streamline_count": 0,
            "vertex_count": 0,
            "written": False,
            "message": "No streamlines passed this dissection; nothing written.",
        }
    elif job.format == "trk":
        summary = _export_trk(job, ids, output_path)
    else:
        summary = _export_zvf(job, ids, output_path)

    summary = dict(summary)
    summary["output_path"] = output_path
    summary["object_count"] = int(ids.size)
    summary["groups"] = [{"name": name, "count": int(v.size)} for name, v in per_group]
    # Objects *considered*: every object the fold was actually evaluated
    # against. Equal to the level's object count minus any the store skipped
    # (no manifest, no readable fragment), so it is a report of the work done
    # rather than a claim about the dataset's size.
    summary["candidate_object_count"] = considered
    return summary
