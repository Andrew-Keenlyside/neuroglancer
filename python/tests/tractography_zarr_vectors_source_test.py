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

"""The zarr-free parts of the zarr-vectors reader: byte ranges and the
read_polylines -> TractIndex adapter.

The zarr-backed path (the fetch Store + aio) needs a Pyodide runtime and is not
exercised here; these cover the pure logic that feeds it and that a broken
adapter would corrupt silently.
"""

from dataclasses import dataclass

import numpy as np
from neuroglancer.tractography.zarr_vectors_source import (
    _polylines_to_tract_index,
    apply_byte_range,
)


@dataclass
class _Range:
    start: int
    end: int


@dataclass
class _Offset:
    offset: int


@dataclass
class _Suffix:
    suffix: int


class TestApplyByteRange:
    def test_none_returns_whole(self):
        assert apply_byte_range(b"abcdef", None) == b"abcdef"

    def test_range(self):
        assert apply_byte_range(b"abcdef", _Range(1, 4)) == b"bcd"

    def test_offset(self):
        assert apply_byte_range(b"abcdef", _Offset(2)) == b"cdef"

    def test_suffix(self):
        assert apply_byte_range(b"abcdef", _Suffix(2)) == b"ef"

    def test_range_to_end(self):
        # A RangeByteRequest whose end is None runs to the end of the data.
        r = _Range(2, 6)
        r.end = None  # type: ignore[assignment]
        assert apply_byte_range(b"abcdef", r) == b"cdef"


class TestPolylinesToTractIndex:
    def test_maps_fragments_to_rows(self):
        # Two objects: object 10 has two fragments, object 20 has one.
        f_a0 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        f_a1 = np.array([[2, 2, 2]], dtype=np.float32)
        f_b0 = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]], dtype=np.float32)
        result = {
            "polylines": [[f_a0, f_a1], [f_b0]],
            "object_ids": [10, 20],
        }
        index = _polylines_to_tract_index(result)

        # One row per fragment, tagged with its object id.
        assert list(index.row_ids) == [10, 10, 20]
        # Offsets span the concatenated vertices exactly.
        assert list(index.offsets) == [0, 2, 3, 6]
        assert index.positions.shape == (6, 3)
        np.testing.assert_array_equal(index.positions[0], [0, 0, 0])
        np.testing.assert_array_equal(index.positions[5], [5, 5, 5])

    def test_skips_empty_fragments(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        f = np.array([[1, 2, 3]], dtype=np.float32)
        result = {"polylines": [[empty, f]], "object_ids": [7]}
        index = _polylines_to_tract_index(result)
        assert list(index.row_ids) == [7]
        assert list(index.offsets) == [0, 1]

    def test_empty_read_raises(self):
        import pytest

        with pytest.raises(ValueError, match="no vertices"):
            _polylines_to_tract_index({"polylines": [], "object_ids": []})
