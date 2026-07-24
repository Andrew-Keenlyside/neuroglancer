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

"""Streamline (tractography) dissection geometry, evaluated in Python/WASM.

This package answers one question for the viewer: given some streamlines and
some regions, which streamlines pass. It is the part of the ROI streamline
filter that moved out of TypeScript.

Everything *around* that question stays in the viewer, where it already was and
where it belongs: the Filter tab and its group model, the ROI overlays, the
persisted layer state, the shader tiers, and the render-layer backend that
decides which pyramid level to evaluate, which chunks are resident, and when a
recompute is warranted. That backend calls in here for the arithmetic and
applies the answer itself.

- `roi`     the geometry: does this polyline cross this region, and the
            include/or/exclude fold that composes several regions.
- `index`   the flat, ragged form the tests evaluate against.
- `wire`    the byte layout the viewer sends and receives.
- `service` the request handler and its geometry cache.

The viewer reaches this through `/neuroglancer/roi_filter`, served by
`neuroglancer.browser_server`. Nothing here is persisted and nothing reaches
the URL.
"""

from .index import TractIndex
from .roi import (
    Box,
    Ellipsoid,
    Halfspace,
    LabelMask,
    Roi,
    RoiOperator,
    RoiPredicate,
    RoiShape,
    combine_roi_verdicts,
    streamlines_pass_roi,
    streamlines_pass_rois,
)
from .service import RoiFilterService, evaluate_groups

__all__ = [
    "TractIndex",
    "Box",
    "Ellipsoid",
    "Halfspace",
    "LabelMask",
    "Roi",
    "RoiOperator",
    "RoiPredicate",
    "RoiShape",
    "RoiFilterService",
    "combine_roi_verdicts",
    "evaluate_groups",
    "streamlines_pass_roi",
    "streamlines_pass_rois",
]
