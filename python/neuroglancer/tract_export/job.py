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
"""The export job spec: what the viewer asks the exporter to produce.

One spec drives both entry points -- the HTTP handler the Export tab POSTs to,
and ``python -m neuroglancer.tract_export job.json``. That is deliberate: the
viewer falls back to *downloading* the spec when no service is reachable, so a
downloaded file and a posted body must mean exactly the same thing.

Groups arrive in the viewer's **persistence** JSON form, i.e. ``groupToJson``
output from ``src/datasource/zarr-vectors/roi_filter_state.ts``: shapes keyed by
``type`` with *string* predicate/operator names. Note this is a different
encoding from the one ``neuroglancer.tractography.wire`` speaks, which keys
shapes by ``kind`` and uses the raw enum integers. Reusing the persistence form
means the spec, the URL hash and a saved ROI-group document all carry
byte-identical group JSON, so there is a single serialisation to keep correct.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from neuroglancer.tractography.roi import (
    Box,
    Ellipsoid,
    Halfspace,
    Roi,
    RoiOperator,
    RoiPredicate,
    RoiShape,
)

#: Bumped when a change would make an older exporter misread a newer spec.
JOB_SCHEMA_VERSION = 1

FORMATS = ("trk", "zvf")
DESTINATION_KINDS = ("local", "gcs")

_PREDICATE_FROM_JSON = {
    "any_segment": RoiPredicate.ANY_SEGMENT,
    "any_vertex": RoiPredicate.ANY_VERTEX,
    "either_endpoint": RoiPredicate.EITHER_ENDPOINT,
    "both_endpoints": RoiPredicate.BOTH_ENDPOINTS,
}
_OPERATOR_FROM_JSON = {
    "and": RoiOperator.AND,
    "or": RoiOperator.OR,
    "andnot": RoiOperator.ANDNOT,
}


class JobSpecError(ValueError):
    """A malformed job spec. The message is meant to be shown to the user."""


@dataclasses.dataclass(frozen=True)
class ExportGroup:
    """One dissection to export, named so the output can be labelled."""

    name: str
    rois: tuple[Roi, ...]


@dataclasses.dataclass(frozen=True)
class Destination:
    kind: str
    path: str


@dataclasses.dataclass(frozen=True)
class ExportJob:
    source_url: str
    level: int
    groups: tuple[ExportGroup, ...]
    format: str
    destination: Destination
    #: 4x4 voxel-to-RAS matrix for TRK, or None for identity.
    affine: np.ndarray | None = None


def _require(obj: Any, key: str, where: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise JobSpecError(f"{where}: missing required property {key!r}")
    return obj[key]


def _floats(value: Any, where: str) -> np.ndarray:
    if not isinstance(value, list | tuple):
        raise JobSpecError(f"{where}: expected an array of numbers")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise JobSpecError(f"{where}: expected an array of numbers ({e})") from e
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise JobSpecError(f"{where}: expected a flat array of finite numbers")
    return arr


def _parse_shape(obj: Any, where: str) -> RoiShape:
    kind = _require(obj, "type", where)
    if kind == "ellipsoid":
        return Ellipsoid(
            _floats(_require(obj, "center", where), f"{where}.center"),
            _floats(_require(obj, "radii", where), f"{where}.radii"),
        )
    if kind == "box":
        return Box(
            _floats(_require(obj, "lower", where), f"{where}.lower"),
            _floats(_require(obj, "upper", where), f"{where}.upper"),
        )
    if kind == "halfspace":
        return Halfspace(
            _floats(_require(obj, "origin", where), f"{where}.origin"),
            _floats(_require(obj, "normal", where), f"{where}.normal"),
        )
    raise JobSpecError(f"{where}: unknown ROI shape type {kind!r}")


def _parse_roi(obj: Any, where: str) -> Roi:
    shape = _parse_shape(_require(obj, "shape", where), f"{where}.shape")
    predicate_name = _require(obj, "predicate", where)
    operator_name = _require(obj, "operator", where)
    if predicate_name not in _PREDICATE_FROM_JSON:
        raise JobSpecError(f"{where}: unknown ROI predicate {predicate_name!r}")
    if operator_name not in _OPERATOR_FROM_JSON:
        raise JobSpecError(f"{where}: unknown ROI operator {operator_name!r}")
    # `name` is display-only and deliberately dropped: it never affects which
    # streamlines are selected, so carrying it into the fold would be noise.
    return Roi(
        shape,
        _PREDICATE_FROM_JSON[predicate_name],
        _OPERATOR_FROM_JSON[operator_name],
    )


def _parse_group(obj: Any, where: str) -> ExportGroup:
    name = _require(obj, "name", where)
    if not isinstance(name, str):
        raise JobSpecError(f"{where}.name: expected a string")
    rois = _require(obj, "rois", where)
    if not isinstance(rois, list):
        raise JobSpecError(f"{where}.rois: expected an array")
    # An empty ROI list passes *everything*, which as an export would silently
    # write the whole dataset under a group's name. Refuse rather than surprise.
    if not rois:
        raise JobSpecError(
            f"{where}: group {name!r} has no ROIs, which would select every "
            f"streamline in the dataset"
        )
    return ExportGroup(
        name=name,
        rois=tuple(_parse_roi(r, f"{where}.rois[{i}]") for i, r in enumerate(rois)),
    )


def _parse_affine(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise JobSpecError(f"affine: expected a 4x4 matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise JobSpecError("affine: expected finite numbers")
    return arr


def parse_job(obj: Any) -> ExportJob:
    """Validate a job spec, raising :class:`JobSpecError` with a usable message."""
    if not isinstance(obj, dict):
        raise JobSpecError("Job spec must be a JSON object")

    version = obj.get("schemaVersion", JOB_SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise JobSpecError("schemaVersion: expected an integer")
    # Reject newer rather than silently ignoring fields this build cannot honour
    # -- an exporter that quietly drops a future `subsample` key would produce a
    # plausible file that is not what was asked for.
    if version > JOB_SCHEMA_VERSION:
        raise JobSpecError(
            f"Job schema version {version} is newer than this exporter "
            f"supports ({JOB_SCHEMA_VERSION})"
        )

    source = _require(obj, "source", "spec")
    source_url = _require(source, "url", "source")
    if not isinstance(source_url, str) or not source_url:
        raise JobSpecError("source.url: expected a non-empty string")
    level = source.get("level", 0)
    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise JobSpecError("source.level: expected a non-negative integer")

    fmt = _require(obj, "format", "spec")
    if fmt not in FORMATS:
        raise JobSpecError(f"format: expected one of {FORMATS}, got {fmt!r}")

    destination = _require(obj, "destination", "spec")
    kind = _require(destination, "kind", "destination")
    if kind not in DESTINATION_KINDS:
        raise JobSpecError(
            f"destination.kind: expected one of {DESTINATION_KINDS}, got {kind!r}"
        )
    # Rejected here, before `run_job` reads the whole level and folds the
    # dissection: only `local` is implemented, and a hand-written `gcs` spec
    # should fail fast with the workaround rather than after minutes of work.
    if kind != "local":
        raise JobSpecError(
            f"destination.kind {kind!r} is not supported yet; export locally "
            f"and upload the result."
        )
    path = _require(destination, "path", "destination")
    if not isinstance(path, str) or not path:
        raise JobSpecError("destination.path: expected a non-empty string")

    groups = _require(obj, "groups", "spec")
    if not isinstance(groups, list) or not groups:
        raise JobSpecError("groups: expected a non-empty array")

    affine = obj.get("affine")
    return ExportJob(
        source_url=source_url,
        level=level,
        groups=tuple(_parse_group(g, f"groups[{i}]") for i, g in enumerate(groups)),
        format=fmt,
        destination=Destination(kind=kind, path=path),
        affine=None if affine is None else _parse_affine(affine),
    )


def roi_bounds(rois: Sequence[Roi]) -> tuple[np.ndarray, np.ndarray] | None:
    """The axis-aligned box enclosing `rois`, or None if it is unbounded.

    Used only to narrow which chunks are read before evaluating the fold; it is
    never the selection itself. A halfspace is unbounded, and an exclusion
    region's *complement* is what passes, so in both cases there is no safe
    finite box and the caller must read everything. Returning None rather than a
    best-effort box keeps that decision explicit at the call site.
    """
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    for roi in rois:
        if roi.operator == RoiOperator.ANDNOT:
            return None
        shape = roi.shape
        if isinstance(shape, Ellipsoid):
            lowers.append(np.asarray(shape.center) - np.abs(shape.radii))
            uppers.append(np.asarray(shape.center) + np.abs(shape.radii))
        elif isinstance(shape, Box):
            lowers.append(np.minimum(shape.lower, shape.upper))
            uppers.append(np.maximum(shape.lower, shape.upper))
        else:
            return None
    if not lowers:
        return None
    # An OR widens the selection, so the union is the only sound envelope; it is
    # also correct (merely looser) for a pure AND chain.
    return (
        np.min(np.stack(lowers), axis=0),
        np.max(np.stack(uppers), axis=0),
    )
