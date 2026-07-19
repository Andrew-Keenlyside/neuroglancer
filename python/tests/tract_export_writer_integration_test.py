# @license
# Copyright 2026 Google Inc.
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

"""End-to-end tract export against a REAL zarr-vectors store.

Unlike ``tract_export_run_test`` (which injects geometry to test selection
without zarr), this drives the whole path: write a store, export a dissection to
TRK, read it back. It also exercises the async read path
(``zarr_vectors.core.aio.read_async`` + the adapter) that the Pyodide viewer
uses.

Skipped unless ``zarr-vectors``, ``zarr-vectors-tools`` (with nibabel) and
``zarr`` are all importable -- the neuroglancer CI env has none of them, so this
is a no-op there and runs on a machine that has installed them.
"""

import asyncio

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
polylines_mod = pytest.importorskip("zarr_vectors.types.polylines")
pytest.importorskip("zarr_vectors.core.aio")
pytest.importorskip("zarr_vectors_tools.export.trk")
nib = pytest.importorskip("nibabel")

from neuroglancer.tract_export import parse_job  # noqa: E402
from neuroglancer.tract_export.run import run_job  # noqa: E402
from neuroglancer.tractography.zarr_vectors_source import (  # noqa: E402
    _polylines_to_tract_index,
)

read_polylines = polylines_mod.read_polylines
write_polylines = polylines_mod.write_polylines


@pytest.fixture
def store(tmp_path):
    """A three-streamline store: two clustered near (10,10,10), one far."""
    path = str(tmp_path / "tracts.zvf")
    s0 = np.array([[10, 10, 10], [12, 10, 10], [14, 10, 10]], dtype=np.float32)
    s1 = np.array([[11, 11, 11], [13, 11, 11]], dtype=np.float32)
    s2 = np.array([[200, 200, 200], [210, 200, 200]], dtype=np.float32)
    write_polylines(
        path,
        [s0, s1, s2],
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0.0, 0.0, 0.0], [300.0, 300.0, 300.0]),
    )
    return path


def _spec(store_path, out_trk):
    return {
        "schemaVersion": 1,
        "source": {"url": f"zarr-vectors://{store_path}", "level": 0},
        "groups": [
            {
                "name": "near",
                "color": "#ff0000",
                "rois": [
                    {
                        # A sphere around the clustered pair; the far tract misses.
                        "shape": {
                            "type": "ellipsoid",
                            "center": [11, 11, 11],
                            "radii": [8, 8, 8],
                        },
                        "predicate": "any_segment",
                        "operator": "and",
                    }
                ],
            }
        ],
        "format": "trk",
        "destination": {"kind": "local", "path": out_trk},
    }


def test_exports_the_dissection_to_a_trk(store, tmp_path):
    out_trk = str(tmp_path / "out.trk")
    summary = run_job(parse_job(_spec(store, out_trk)))

    # The dissection selects the two near tracts; the far one misses the sphere.
    assert summary["streamline_count"] == 2
    assert summary["object_count"] == 2
    assert summary["candidate_object_count"] == 3

    trk = nib.streamlines.load(out_trk)
    assert len(trk.streamlines) == 2
    # Geometry survived: 3 + 2 vertices across the two tracts.
    assert sum(len(s) for s in trk.streamlines) == 5


def test_async_read_path_matches_the_store(store):
    async def read_via_aio():
        from zarr_vectors.core.aio import open_store_async, read_async

        root = await open_store_async(store)
        result = await read_async(read_polylines, root, level=0)
        return _polylines_to_tract_index(result)

    index = asyncio.run(read_via_aio())
    # One row per fragment; each of the three tracts is a single fragment.
    assert len(index) == 3
    assert {int(x) for x in index.row_ids} == {0, 1, 2}

    # The async path must agree with a direct synchronous read.
    direct = read_polylines(store, level=0)
    assert {int(x) for x in direct["object_ids"]} == {0, 1, 2}
