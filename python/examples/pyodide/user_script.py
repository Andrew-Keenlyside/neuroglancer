# @license
# Copyright 2026 The Neuroglancer Authors
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

"""Default script for the Streamline Filter (Pyodide) build of neuroglancer.

Loads the HCP1065 whole-brain tractogram (zarr-vectors) as the test dataset, so
"Use test data" on the start screen boots straight into a filterable tractogram.
Open the layer's **Filter** tab to draw inclusion/exclusion ROIs.

To use your own data, either edit `TEST_TRACTOGRAM` below and rebuild, or paste
a Neuroglancer URL on the start screen (which takes over the state) instead of
clicking "Use test data".

Runs inside the Pyodide worker; `neuroglancer.Viewer` resolves to
`PyodideViewer`, which talks to the page through the service worker instead of a
tornado server. Select a different script with `?script=<same-origin-path>`.

Pyodide scripts execute at module level -- there is no `__main__` guard, and
`argparse`, `webbrowser`, `neuroglancer.cli`, and blocking `input()` are all
unavailable. Use `js.setTimeout` rather than `threading.Timer`.
"""

import neuroglancer

# A zarr-vectors source is `<kvstore-url>|zarr-vectors:`. Any gs:// zarr-vectors
# store works; the `https://storage.googleapis.com/...` form is used here so the
# public bucket resolves without extra GCS auth.
TEST_TRACTOGRAM = (
    "https://storage.googleapis.com/hip_ct_zarr_vector_03987646472fethdsvdvdfg/"
    "zarr_vectors_test/hcp1065_whole_brain.zarrvectors/|zarr-vectors:"
)

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    s.layers["tracts"] = neuroglancer.SegmentationLayer(source=TEST_TRACTOGRAM)
    s.layout = "4panel"

print("Viewer ready:", viewer.get_viewer_url())
