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
    tracts = neuroglancer.SegmentationLayer(source=TEST_TRACTOGRAM)
    s.layers["tracts"] = tracts
    s.layout = "4panel"

    # Render the streamlines as thin lines. In the 2-d slices default to lines
    # only (no per-node dots, which clutter a dense tractogram), and draw
    # passing streamlines fully opaque there (the slice default is 0.5). The
    # ROI filter dims non-passing streamlines to `DEFAULT_GHOST_ALPHA` (0.3).
    tracts.skeleton_rendering.mode2d = "lines"
    tracts.skeleton_rendering.line_width2d = 2
    tracts.selected_alpha = 1  # passing ("on") streamline opacity in 2-d

    # Load sparse-first. The pyramid holds ~503k / 50k / 5k / 503 / 50 whole
    # streamlines at levels 0..4; the GPU-memory budget picks the finest level
    # that fits. Keep it small so a coarse *whole-brain* level loads (raise it
    # to show more, lower it to show fewer):
    #   ~20 MB -> level 3 (~503 tracts, fastest),  ~50 MB -> level 2 (~5k),
    #   large budgets pick the finest level (10^8 vertices) and stall.
    s.gpu_memory_limit = 50_000_000

    # In the 3-d view, the cross-section planes otherwise draw as opaque grey
    # quads that hide the tracts. Discard their empty background so the tracts
    # show through, and outline each plane so its position is still visible.
    s.hide_cross_section_background_3d = True
    s.show_cross_section_outline_3d = True

    # The 2-d slice panels default to a grey background (meant to sit behind a
    # volume). A tract-only layer has nothing volumetric there, so make the
    # slice background black to match the 3-d view.
    s.cross_section_background_color = "#000000"

print("Viewer ready:", viewer.get_viewer_url())
