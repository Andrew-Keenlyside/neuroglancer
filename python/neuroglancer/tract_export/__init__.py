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
"""Export an ROI dissection as a TRK file or a new zarr-vectors dataset.

The viewer cannot do this itself: the writers live in ``zarr-vectors-tools``.
So the Export tab describes the work as a :mod:`job` spec and this package
carries it out natively, either behind an HTTP route or from the command line.

That split is the right design regardless of what imports where -- exporting
re-evaluates the dissection at pyramid level 0 across the whole store, which is
not browser work. But the dependency argument for it was overstated, and was
corrected on 2026-07-19: under pyodide 314.0.2, ``zarr>=3`` ships in the
distribution, ``numcodecs`` does too, and ``nibabel`` is pure Python and
micropip-installable. The one genuinely unavailable package is ``trx-python``
(no pyemscripten wheel), which is bundled with ``nibabel`` in the same
``[streamlines]`` extra -- see ``upstream_prompts/zarr-vectors-tools.md`` for
splitting them.

Which streamlines get exported is decided **here**, at pyramid level 0, by
re-evaluating the dissection against the store. The viewer's own passing set
covers only the chunks currently resident at the current level of detail, so
exporting from it would quietly drop tracts.
"""

from neuroglancer.tract_export.job import (
    JOB_SCHEMA_VERSION,
    Destination,
    ExportGroup,
    ExportJob,
    JobSpecError,
    parse_job,
    roi_bounds,
)

__all__ = [
    "Destination",
    "ExportGroup",
    "ExportJob",
    "JobSpecError",
    "JOB_SCHEMA_VERSION",
    "parse_job",
    "roi_bounds",
]
