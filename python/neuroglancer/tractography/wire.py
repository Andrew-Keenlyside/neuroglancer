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

"""Byte layout of the ROI dissection request and response.

Mirrored on the viewer side by `src/datasource/zarr-vectors/roi_filter_service.ts`.
If either side changes, both change together.

Small, variable-shape values (the region definitions, which chunks to evaluate,
which to forget) travel as JSON in the query string. Bulk geometry travels as
the raw request body, and the resulting id sets as the raw response body -- the
same split the viewer's existing volume-chunk endpoint uses, and for the same
reason: base64 in JSON would cost a third more bytes and a parse on both ends.

All integers are little-endian. Every array starts at an offset that is a
multiple of its element size, so both sides can view the buffer in place rather
than copying it.

Request body -- the geometry of chunks not already cached, concatenated, each
blob padded to a multiple of 8 bytes::

    uint32  rank
    uint32  numFragments        F
    uint32  numVertices         V, summed over fragments
    uint32  reserved            (0; keeps the u64 below 8-aligned)
    uint64  rowIds[F]           the object id each fragment belongs to
    uint32  rowOffsets[F + 1]   fragment i is vertices[rowOffsets[i]:rowOffsets[i+1]]
    float32 positions[V * rank]

`positions` is *gathered*: the viewer emits each fragment's vertices
contiguously, in fragment order, rather than the chunk's raw vertex array. That
is not an optimisation for its own sake -- the viewer must copy anyway, since
handing over (transferring) its buffers would detach them and break both the
render path and every later refilter. Gathering during that unavoidable copy
also drops the chunk's ghost vertices, which belong to no fragment and must not
be tested.

Response body::

    uint32  magic = 'RFLT'
    uint32  numPassing          P
    uint32  numHighDetail       H
    uint32  numColors           C
    uint64  passing[P]
    uint64  highDetail[H]
    uint64  colorIds[C]
    uint32  colorValues[C]      packed RGBA, matching the viewer's packColor
"""

from __future__ import annotations

import json
import struct
from typing import NamedTuple

import numpy as np

from .index import TractIndex
from .roi import Box, Ellipsoid, Halfspace, Roi, RoiOperator, RoiPredicate

__all__ = [
    "RESPONSE_MAGIC",
    "RoiGroupSpec",
    "RoiFilterRequest",
    "decode_request",
    "encode_response",
]

RESPONSE_MAGIC = 0x52464C54  # 'RFLT'
_CHUNK_HEADER = struct.Struct("<IIII")


class RoiGroupSpec(NamedTuple):
    """One dissection, as the viewer's `RoiGroupConfig` sends it."""

    rois: tuple[Roi, ...]
    color_packed: int
    visible: bool
    high_detail: bool


class RoiFilterRequest(NamedTuple):
    scope: str
    """Cache scope -- one per tract render layer."""
    chunk_keys: tuple[str, ...]
    """The chunks to evaluate, in the viewer's order."""
    uploads: dict[str, TractIndex]
    """Geometry carried by this request, by chunk key."""
    drop: tuple[str, ...]
    """Chunks the viewer has evicted; forget their geometry."""
    groups: tuple[RoiGroupSpec, ...]


def _shape_of(spec: dict):
    kind = spec["kind"]
    if kind == "ellipsoid":
        return Ellipsoid(
            np.asarray(spec["center"], dtype=np.float64),
            np.asarray(spec["radii"], dtype=np.float64),
        )
    if kind == "box":
        return Box(
            np.asarray(spec["lower"], dtype=np.float64),
            np.asarray(spec["upper"], dtype=np.float64),
        )
    if kind == "halfspace":
        return Halfspace(
            np.asarray(spec["origin"], dtype=np.float64),
            np.asarray(spec["normal"], dtype=np.float64),
        )
    raise ValueError(f"unknown ROI shape kind: {kind!r}")


def _group_of(spec: dict) -> RoiGroupSpec:
    return RoiGroupSpec(
        rois=tuple(
            Roi(
                shape=_shape_of(r["shape"]),
                predicate=RoiPredicate(int(r.get("predicate", 0))),
                operator=RoiOperator(int(r.get("operator", 0))),
            )
            for r in spec.get("rois", ())
        ),
        color_packed=int(spec.get("colorPacked", 0)) & 0xFFFFFFFF,
        visible=bool(spec.get("visible", True)),
        high_detail=bool(spec.get("highDetail", False)),
    )


def _decode_chunk(body: memoryview, offset: int) -> tuple[TractIndex, int]:
    """Decode one geometry blob. Returns the index and the next blob's offset."""
    rank, num_fragments, num_vertices, _reserved = _CHUNK_HEADER.unpack_from(
        body, offset
    )
    pos = offset + _CHUNK_HEADER.size

    row_ids = np.frombuffer(body, dtype="<u8", count=num_fragments, offset=pos)
    pos += num_fragments * 8

    row_offsets = np.frombuffer(body, dtype="<u4", count=num_fragments + 1, offset=pos)
    pos += (num_fragments + 1) * 4

    positions = np.frombuffer(
        body, dtype="<f4", count=num_vertices * rank, offset=pos
    ).reshape(num_vertices, rank)
    pos += num_vertices * rank * 4

    if int(row_offsets[-1]) != num_vertices:
        raise ValueError(
            f"chunk rowOffsets end at {int(row_offsets[-1])}, header says "
            f"{num_vertices} vertices"
        )

    # `frombuffer` views the request body, and these arrays are cached across
    # requests, so every one of them must be copied -- `np.array`, never
    # `np.asarray`. The distinction is not cosmetic: `asarray` copies only when
    # the dtype changes, so `row_offsets` (<u4 -> intp) would copy by accident
    # while `row_ids` (already <u8) would not, leaving a view that pins the
    # WHOLE multi-chunk request body for as long as that one chunk is cached --
    # and silently defeating eviction, since dropping 511 of 512 chunks would
    # then free nothing.
    index = TractIndex(
        np.array(positions, dtype=np.float32),
        np.array(row_offsets, dtype=np.intp),
        np.array(row_ids, dtype=np.uint64),
    )
    return index, (pos + 7) & ~7


def decode_request(query_json: str, body: bytes) -> RoiFilterRequest:
    spec = json.loads(query_json)
    view = memoryview(body)
    uploads: dict[str, TractIndex] = {}
    offset = 0
    for upload in spec.get("uploads", ()):
        index, offset = _decode_chunk(view, offset)
        uploads[upload["key"]] = index
    return RoiFilterRequest(
        scope=str(spec["scope"]),
        chunk_keys=tuple(spec.get("chunkKeys", ())),
        uploads=uploads,
        drop=tuple(spec.get("drop", ())),
        groups=tuple(_group_of(g) for g in spec.get("groups", ())),
    )


def encode_response(
    passing: np.ndarray, colors: dict, high_detail: np.ndarray
) -> bytes:
    color_ids = np.fromiter(colors.keys(), dtype=np.uint64, count=len(colors))
    color_values = np.fromiter(colors.values(), dtype=np.uint32, count=len(colors))
    parts = [
        struct.pack(
            "<IIII",
            RESPONSE_MAGIC,
            passing.size,
            high_detail.size,
            color_ids.size,
        ),
        np.ascontiguousarray(passing, dtype="<u8").tobytes(),
        np.ascontiguousarray(high_detail, dtype="<u8").tobytes(),
        np.ascontiguousarray(color_ids, dtype="<u8").tobytes(),
        np.ascontiguousarray(color_values, dtype="<u4").tobytes(),
    ]
    return b"".join(parts)
