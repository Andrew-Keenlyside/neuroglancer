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

"""The ROI dissection service: evaluate streamline filters for the viewer.

The viewer's chunk backend worker owns the decoded tract geometry and decides
*which* chunks to filter and *when*. It calls in here for the one thing that
moved out of TypeScript: the geometry itself -- does this streamline cross this
region. Everything above that line (pyramid-level selection, chunk residency,
recompute scheduling, set diffing) stays where it was.

The transport is the same Service-Worker-intercepted `fetch` the viewer already
uses to reach Python for local volumes and skeletons, so the backend worker
talks to this directly, without the main thread and without touching viewer
state. Nothing here is persisted, and nothing reaches the URL.

**Geometry is uploaded once per chunk.** A decoded chunk is immutable for the
life of its `(scope, key)` -- the pyramid level is part of the key -- so after
the first upload an ROI edit sends only the region definitions, a few hundred
bytes, and gets back a set of ids. That is what makes dragging a region cheap:
the alternative, re-sending a couple of megabytes per frame, would cost more
than the arithmetic it feeds.

See `wire.py` for the byte layout, which the TypeScript side mirrors in
`src/datasource/zarr-vectors/roi_filter_service.ts`.
"""

from __future__ import annotations

import numpy as np

from .index import TractIndex
from .roi import streamlines_pass_rois
from .wire import RoiFilterRequest, encode_response

__all__ = ["RoiFilterService", "evaluate_groups"]


def evaluate_groups(index: TractIndex, groups) -> tuple[np.ndarray, dict, np.ndarray]:
    """Evaluate several ROI groups against one batch of geometry.

    Each visible group is an independent dissection (its own include/or/exclude
    fold), so groups are evaluated separately -- flattening them into one fold
    would wrongly let one group's exclusion drop another group's tracts.

    Returns the union of all visible groups' passing objects, the packed colour
    of the *first* visible group each object belongs to, and the subset of
    passing objects in groups marked high-detail. This mirrors
    `computeGroupedPassingSet` in the TypeScript implementation it replaced,
    including the first-group-wins colour rule.
    """
    n = len(index)
    passing = np.zeros(n, dtype=bool)
    high_detail = np.zeros(n, dtype=bool)
    claimed_color = np.zeros(n, dtype=np.uint32)
    claimed = np.zeros(n, dtype=bool)

    for group in groups:
        if not group.visible or not group.rois:
            continue
        member = streamlines_pass_rois(index, group.rois)
        # First group in list order wins the colour (topmost, deterministic).
        fresh = member & ~claimed
        claimed_color[fresh] = group.color_packed
        claimed |= fresh
        passing |= member
        if group.high_detail:
            high_detail |= member

    ids = index.object_ids
    colors = {int(i): int(c) for i, c in zip(ids[claimed], claimed_color[claimed])}
    return ids[passing], colors, ids[high_detail]


class RoiFilterService:
    """Holds the per-chunk geometry the viewer has uploaded, and evaluates against it.

    Two caches, both keyed by data rather than by time:

    `_chunks`   (scope, key) -> TractIndex for that one chunk.
    `_combined` (scope, chunk-key-set) -> the concatenated index for a whole
                recompute. Rebuilt only when chunk residency changes, so an ROI
                drag -- which changes the regions but not the chunk set -- reuses
                it and does no concatenation at all.

    Concatenating chunks into one index is what recombines a tract that spans
    them: rows sharing an object id become fragments of one streamline, and the
    per-region crossings are OR-ed across them before the include/or/exclude
    fold is applied. Folding per chunk instead would drop a tract that crosses
    region A in one chunk and region B in another.
    """

    def __init__(self, max_scopes: int = 8):
        self._chunks: dict[tuple[str, str], TractIndex] = {}
        self._combined: dict[str, tuple[tuple[str, ...], TractIndex]] = {}
        self._max_scopes = max_scopes
        # Scopes in least-recently-used order. A scope is a render layer, and a
        # layer that is removed and re-added gets a new one, so nothing on the
        # viewer side can ever ask for the old one's geometry back. Without this
        # bound, layer churn would strand it.
        self._scope_order: list[str] = []

    # -- cache maintenance -------------------------------------------------

    def drop(self, scope: str, keys) -> None:
        """Forget chunks the viewer has evicted."""
        for key in keys:
            self._chunks.pop((scope, key), None)
        self._combined.pop(scope, None)

    def forget_scope(self, scope: str) -> None:
        """Drop everything for one render layer (it went away)."""
        for cache_key in [k for k in self._chunks if k[0] == scope]:
            del self._chunks[cache_key]
        self._combined.pop(scope, None)

    def _combined_index(self, scope: str, keys: tuple[str, ...]) -> TractIndex:
        cached = self._combined.get(scope)
        if cached is not None and cached[0] == keys:
            return cached[1]
        parts = [self._chunks[(scope, k)] for k in keys]
        index = _concatenate(parts)
        self._combined[scope] = (keys, index)
        return index

    # -- the request -------------------------------------------------------

    def handle(self, request: RoiFilterRequest) -> bytes:
        """Run one recompute. Returns the encoded response body.

        Raises `KeyError` naming the chunks whose geometry this process does not
        have -- the caller turns that into a 409 so the viewer can re-upload.
        That happens when the Python worker restarted underneath a viewer that
        still believes its geometry is cached; it is recoverable, not fatal.
        """
        scope = request.scope
        if scope in self._scope_order:
            self._scope_order.remove(scope)
        self._scope_order.append(scope)
        while len(self._scope_order) > self._max_scopes:
            self.forget_scope(self._scope_order.pop(0))

        for key, index in request.uploads.items():
            self._chunks[(scope, key)] = index
        if request.drop:
            self.drop(scope, request.drop)

        missing = [k for k in request.chunk_keys if (scope, k) not in self._chunks]
        if missing:
            raise KeyError(missing)

        if not request.chunk_keys or not request.groups:
            return encode_response(
                np.zeros(0, dtype=np.uint64), {}, np.zeros(0, dtype=np.uint64)
            )

        index = self._combined_index(scope, request.chunk_keys)
        passing, colors, high_detail = evaluate_groups(index, request.groups)
        return encode_response(passing, colors, high_detail)


def _concatenate(parts: list[TractIndex]) -> TractIndex:
    """Concatenate per-chunk indices into one, preserving row (fragment) identity."""
    if len(parts) == 1:
        return parts[0]
    if not parts:
        return TractIndex(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(1, dtype=np.intp),
            np.zeros(0, dtype=np.uint64),
        )
    positions = np.concatenate([p.positions for p in parts])
    row_ids = np.concatenate([p.row_ids for p in parts])
    counts = np.concatenate([p.counts for p in parts])
    offsets = np.zeros(counts.size + 1, dtype=np.intp)
    np.cumsum(counts, out=offsets[1:])
    return TractIndex(positions, offsets, row_ids)
