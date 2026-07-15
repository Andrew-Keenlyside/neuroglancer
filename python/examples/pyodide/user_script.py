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

"""Default script for the Pyodide build of neuroglancer.

Runs inside the Pyodide worker; `neuroglancer.Viewer` resolves to
`PyodideViewer`, which talks to the page through the service worker instead of
a tornado server. Select a different script with `?script=<same-origin-path>`.

Pyodide scripts execute at module level -- there is no `__main__` guard, and
`argparse`, `webbrowser`, `neuroglancer.cli`, and blocking `input()` are all
unavailable. Use `js.setTimeout` rather than `threading.Timer`.
"""

import neuroglancer

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    s.layers["image"] = neuroglancer.ImageLayer(
        source="precomputed://gs://neuroglancer-public-data/flyem_fib-25/image",
    )
    s.layers["segmentation"] = neuroglancer.SegmentationLayer(
        source="precomputed://gs://neuroglancer-public-data/flyem_fib-25/ground_truth",
    )

print("Viewer ready:", viewer.get_viewer_url())
