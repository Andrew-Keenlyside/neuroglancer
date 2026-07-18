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

"""Tests for the ROI dissection service and its wire format.

The encoder here stands in for the viewer's TypeScript side
(`src/datasource/zarr-vectors/roi_filter_service.ts`). Keeping an independent
implementation of the *writing* half is deliberate: a round-trip through the
same code proves very little, whereas these tests fail if either side's idea of
the byte layout drifts.
"""

import json
import struct

import numpy as np
import pytest

from neuroglancer.tractography import Ellipsoid, Roi, RoiOperator, RoiPredicate, TractIndex
from neuroglancer.tractography.service import RoiFilterService, evaluate_groups
from neuroglancer.tractography.wire import (
    RESPONSE_MAGIC,
    RoiGroupSpec,
    decode_request,
    encode_response,
)


def encode_chunk(fragments, row_ids, rank=3) -> bytes:
    """Encode one chunk's gathered geometry, as the viewer will.

    `fragments` is a list of (n, rank) arrays -- each already gathered, i.e. the
    vertices of one fragment in order, ghost vertices excluded.
    """
    counts = [len(f) for f in fragments]
    offsets = np.zeros(len(fragments) + 1, dtype="<u4")
    np.cumsum(counts, out=offsets[1:])
    positions = (
        np.concatenate(fragments).astype("<f4")
        if fragments
        else np.zeros((0, rank), dtype="<f4")
    )
    blob = b"".join(
        [
            struct.pack("<IIII", rank, len(fragments), int(offsets[-1]), 0),
            np.asarray(row_ids, dtype="<u8").tobytes(),
            offsets.tobytes(),
            positions.tobytes(),
        ]
    )
    return blob + b"\0" * (-len(blob) % 8)


def make_request(scope, chunks, groups, drop=(), cached=()):
    """Build (query_json, body) for a set of chunks. `cached` keys send no geometry."""
    uploads, body = [], b""
    for key, fragments, row_ids in chunks:
        if key in cached:
            continue
        uploads.append({"key": key})
        body += encode_chunk(fragments, row_ids)
    spec = {
        "scope": scope,
        "chunkKeys": [c[0] for c in chunks],
        "uploads": uploads,
        "drop": list(drop),
        "groups": groups,
    }
    return json.dumps(spec), body


def group(rois, color=0xFF00FF00, visible=True, high_detail=False):
    return {
        "rois": rois,
        "colorPacked": color,
        "visible": visible,
        "highDetail": high_detail,
    }


def sphere_roi(center, radius, operator=0, predicate=0):
    return {
        "shape": {"kind": "ellipsoid", "center": list(center), "radii": [radius] * 3},
        "operator": operator,
        "predicate": predicate,
    }


def decode_response(body: bytes):
    magic, n_pass, n_hd, n_col = struct.unpack_from("<IIII", body, 0)
    assert magic == RESPONSE_MAGIC
    pos = 16
    passing = np.frombuffer(body, dtype="<u8", count=n_pass, offset=pos)
    pos += n_pass * 8
    high_detail = np.frombuffer(body, dtype="<u8", count=n_hd, offset=pos)
    pos += n_hd * 8
    color_ids = np.frombuffer(body, dtype="<u8", count=n_col, offset=pos)
    pos += n_col * 8
    color_values = np.frombuffer(body, dtype="<u4", count=n_col, offset=pos)
    pos += n_col * 4
    assert pos == len(body), f"{len(body) - pos} trailing bytes"
    return (
        sorted(int(x) for x in passing),
        sorted(int(x) for x in high_detail),
        {int(i): int(v) for i, v in zip(color_ids, color_values)},
    )


def line(x0, x1, y=0.0, n=5):
    xs = np.linspace(x0, x1, n)
    return np.stack([xs, np.full(n, y), np.zeros(n)], axis=1)


class TestRouteHandler:
    """The HTTP surface `BrowserViewerServer` exposes to the viewer's worker.

    Constructed without `__init__` on purpose: the real one installs a Pyodide
    JS bridge that does not exist off-platform, and none of it is needed to
    exercise the handler.
    """

    @staticmethod
    def server():
        from neuroglancer.browser_server import BrowserViewerServer

        instance = object.__new__(BrowserViewerServer)
        instance.viewers = {}
        instance._roi_filter_service = None
        return instance

    def call(self, server, query, body, scope="s"):
        return server._handle_roi_filter(scope, body, {"p": [query]})

    def test_evaluates_and_returns_octet_stream(self):
        query, body = make_request(
            "s",
            [("c0", [line(-1, 1)], [4])],
            [group([sphere_roi([0, 0, 0], 1)])],
        )
        status, content_type, payload = self.call(self.server(), query, body)
        assert status == 200 and content_type == "application/octet-stream"
        assert decode_response(payload)[0] == [4]

    def test_service_is_built_once_and_caches_between_calls(self):
        server = self.server()
        query, body = make_request(
            "s", [("c0", [line(-1, 1)], [4])], [group([sphere_roi([0, 0, 0], 1)])]
        )
        self.call(server, query, body)
        service = server._roi_filter_service
        assert service is not None
        # A second call sending no geometry must still resolve, via the cache.
        cached_query, cached_body = make_request(
            "s",
            [("c0", [line(-1, 1)], [4])],
            [group([sphere_roi([0, 0, 0], 1)])],
            cached=("c0",),
        )
        status, _, payload = self.call(server, cached_query, cached_body)
        assert status == 200 and decode_response(payload)[0] == [4]
        assert server._roi_filter_service is service

    def test_missing_geometry_reports_409_with_the_keys(self):
        query, body = make_request(
            "s",
            [("c0", [line(-1, 1)], [4])],
            [group([sphere_roi([0, 0, 0], 1)])],
            cached=("c0",),
        )
        status, content_type, payload = self.call(self.server(), query, body)
        assert status == 409 and content_type == "application/json"
        assert json.loads(payload)["missing"] == ["c0"]

    def test_scope_mismatch_is_rejected(self):
        query, body = make_request(
            "s", [("c0", [line(-1, 1)], [4])], [group([sphere_roi([0, 0, 0], 1)])]
        )
        status, _, _ = self.call(self.server(), query, body, scope="other")
        assert status == 400

    def test_malformed_request_is_a_400_not_a_crash(self):
        status, _, payload = self.call(self.server(), "{not json", b"")
        assert status == 400 and b"bad roi_filter request" in payload

    def test_truncated_body_is_a_400(self):
        query, body = make_request(
            "s", [("c0", [line(-1, 1)], [4])], [group([sphere_roi([0, 0, 0], 1)])]
        )
        status, _, _ = self.call(self.server(), query, body[:20])
        assert status == 400


class TestCrossLanguageGolden:
    """Decode bytes produced by the viewer's actual TypeScript encoder.

    `GOLDEN_CHUNK_HEX` is asserted byte-for-byte by
    `src/datasource/zarr-vectors/roi_filter_service.spec.ts`. Testing both sides
    against the same literal is what makes the wire format a contract rather
    than two implementations that happen to agree today. The encoder that
    produced it saw six vertices; the sixth is a ghost, appended for rendering
    continuity across chunk borders, belonging to no fragment — it must not
    appear here.
    """

    GOLDEN_CHUNK_HEX = (
        "030000000200000005000000"  # rank 3, 2 fragments, 5 vertices
        "00000000"  # reserved
        "07000000000000000900000000000000"  # rowIds: 7, 9
        "000000000300000005000000"  # rowOffsets: 0, 3, 5
        "000000000000000000000000"  # (0, 0, 0)
        "0000803f0000000000000000"  # (1, 0, 0)
        "000000400000000000000000"  # (2, 0, 0)
        "000020410000000000000000"  # (10, 0, 0)
        "000030410000000000000000"  # (11, 0, 0)
    )

    def test_decodes_the_typescript_encoders_output(self):
        body = bytes.fromhex(self.GOLDEN_CHUNK_HEX)
        assert len(body) % 8 == 0, "blobs must stay 8-aligned for the next uint64"
        query = json.dumps(
            {"scope": "s", "chunkKeys": ["c0"], "uploads": [{"key": "c0"}], "groups": []}
        )
        index = decode_request(query, body).uploads["c0"]

        assert index.n_rows == 2
        assert list(index.object_ids) == [7, 9]
        # Five vertices, not six: the ghost is absent.
        assert index.positions.shape == (5, 3)
        np.testing.assert_allclose(
            index.positions,
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0], [11, 0, 0]],
        )
        assert list(index.counts) == [3, 2]

    def test_decoded_arrays_own_their_memory(self):
        # A decoded chunk is cached across requests, so it must not view the
        # request body. `np.asarray` copies only on a dtype change, which would
        # have left `row_ids` (already <u8) as a view pinning the WHOLE
        # multi-chunk body -- and silently defeating eviction, since dropping
        # every other chunk would then free nothing.
        body = bytes.fromhex(self.GOLDEN_CHUNK_HEX) * 2
        query = json.dumps(
            {
                "scope": "s",
                "chunkKeys": ["c0", "c1"],
                "uploads": [{"key": "c0"}, {"key": "c1"}],
                "groups": [],
            }
        )
        request = decode_request(query, body)
        for key, index in request.uploads.items():
            for name in ("positions", "offsets", "row_ids"):
                array = getattr(index, name)
                assert array.flags.owndata, f"{key}.{name} views the request body"
                assert not np.shares_memory(array, np.frombuffer(body, dtype=np.uint8))

    def test_the_golden_geometry_filters_as_expected(self):
        body = bytes.fromhex(self.GOLDEN_CHUNK_HEX)
        query = json.dumps(
            {
                "scope": "s",
                "chunkKeys": ["c0"],
                "uploads": [{"key": "c0"}],
                "groups": [group([sphere_roi([10.5, 0, 0], 1)])],
            }
        )
        passing, _, _ = decode_response(
            RoiFilterService().handle(decode_request(query, body))
        )
        assert passing == [9]


class TestWireRoundTrip:
    def test_geometry_survives_encoding(self):
        frags = [line(-10, 10), line(0, 5, y=20.0, n=3)]
        query, body = make_request(
            "s", [("c0", frags, [7, 9])], [group([sphere_roi([0, 0, 0], 1)])]
        )
        request = decode_request(query, body)
        assert request.scope == "s"
        assert request.chunk_keys == ("c0",)
        index = request.uploads["c0"]
        assert index.n_rows == 2
        assert list(index.object_ids) == [7, 9]
        assert index.positions.shape == (8, 3)
        np.testing.assert_allclose(index.positions[0], [-10, 0, 0])

    def test_multiple_chunks_decode_at_the_right_offsets(self):
        # The second blob only decodes correctly if the first was padded to 8.
        chunks = [
            ("c0", [line(-10, 10), line(-1, 1, y=5.0, n=3)], [1, 2]),
            ("c1", [line(50, 60, y=1.0, n=7)], [3]),
        ]
        query, body = make_request("s", chunks, [group([sphere_roi([0, 0, 0], 1)])])
        request = decode_request(query, body)
        assert list(request.uploads["c0"].object_ids) == [1, 2]
        assert list(request.uploads["c1"].object_ids) == [3]
        np.testing.assert_allclose(request.uploads["c1"].positions[0], [50, 1, 0])

    def test_group_spec_decodes(self):
        query, body = make_request(
            "s",
            [("c0", [line(-1, 1)], [1])],
            [
                group(
                    [
                        sphere_roi([0, 0, 0], 1),
                        sphere_roi([9, 0, 0], 1, operator=2, predicate=2),
                    ],
                    color=0x11223344,
                    high_detail=True,
                )
            ],
        )
        (g,) = decode_request(query, body).groups
        assert isinstance(g, RoiGroupSpec)
        assert g.color_packed == 0x11223344 and g.high_detail and g.visible
        assert g.rois[1].operator is RoiOperator.ANDNOT
        assert g.rois[1].predicate is RoiPredicate.EITHER_ENDPOINT

    def test_response_round_trip(self):
        body = encode_response(
            np.array([5, 9], dtype=np.uint64),
            {5: 0xAABBCCDD},
            np.array([9], dtype=np.uint64),
        )
        passing, high_detail, colors = decode_response(body)
        assert passing == [5, 9] and high_detail == [9]
        assert colors == {5: 0xAABBCCDD}

    def test_empty_response_round_trip(self):
        body = encode_response(
            np.zeros(0, dtype=np.uint64), {}, np.zeros(0, dtype=np.uint64)
        )
        assert decode_response(body) == ([], [], {})


class TestService:
    def chunks(self):
        # Tract 1 spans both chunks: left half in c0, right half in c1.
        return [
            ("c0", [line(-11, -9), line(-1, 1, y=20.0)], [1, 2]),
            ("c1", [line(9, 11)], [1]),
        ]

    def run(self, service, groups, chunks=None, cached=(), drop=()):
        query, body = make_request(
            "s", chunks if chunks is not None else self.chunks(), groups, drop, cached
        )
        return decode_response(service.handle(decode_request(query, body)))

    def test_single_region(self):
        passing, _, colors = self.run(
            RoiFilterService(), [group([sphere_roi([0, 20, 0], 2)])]
        )
        assert passing == [2]
        assert colors == {2: 0xFF00FF00}

    def test_and_fold_spans_two_chunks(self):
        # Tract 1 crosses x=-10 (in c0) and x=10 (in c1). Folding per chunk
        # would drop it; OR-ing crossings across chunks first keeps it.
        passing, _, _ = self.run(
            RoiFilterService(),
            [
                group(
                    [
                        sphere_roi([-10, 0, 0], 1),
                        sphere_roi([10, 0, 0], 1, operator=0),
                    ]
                )
            ],
        )
        assert passing == [1]

    def test_first_group_wins_the_colour(self):
        rois = [sphere_roi([0, 20, 0], 2)]
        passing, _, colors = self.run(
            RoiFilterService(),
            [group(rois, color=0x111), group(rois, color=0x222)],
        )
        assert passing == [2] and colors == {2: 0x111}

    def test_high_detail_is_the_union_of_marked_groups(self):
        passing, high_detail, _ = self.run(
            RoiFilterService(),
            [
                group([sphere_roi([0, 20, 0], 2)], high_detail=False),
                group([sphere_roi([-10, 0, 0], 1)], high_detail=True),
            ],
        )
        assert passing == [1, 2] and high_detail == [1]

    def test_invisible_group_contributes_nothing(self):
        passing, _, _ = self.run(
            RoiFilterService(),
            [group([sphere_roi([0, 20, 0], 2)], visible=False)],
        )
        assert passing == []

    def test_no_groups_passes_nothing(self):
        assert self.run(RoiFilterService(), []) == ([], [], {})

    def test_geometry_is_cached_across_requests(self):
        service = RoiFilterService()
        first = self.run(service, [group([sphere_roi([0, 20, 0], 2)])])
        # Second request uploads NO geometry -- this is the ROI-drag case.
        second = self.run(
            service,
            [group([sphere_roi([0, 20, 0], 2)])],
            cached=("c0", "c1"),
        )
        assert first == second

    def test_missing_geometry_is_reported_not_guessed(self):
        service = RoiFilterService()
        with pytest.raises(KeyError) as excinfo:
            self.run(service, [group([sphere_roi([0, 0, 0], 1)])], cached=("c0", "c1"))
        assert sorted(excinfo.value.args[0]) == ["c0", "c1"]

    def test_drop_evicts(self):
        service = RoiFilterService()
        self.run(service, [group([sphere_roi([0, 20, 0], 2)])])
        with pytest.raises(KeyError):
            self.run(
                service,
                [group([sphere_roi([0, 20, 0], 2)])],
                cached=("c0", "c1"),
                drop=("c0",),
            )

    def test_combined_index_is_rebuilt_when_the_chunk_set_changes(self):
        service = RoiFilterService()
        rois = [group([sphere_roi([10, 0, 0], 1)])]
        both = self.run(service, rois)
        assert both == ([1], [], {1: 0xFF00FF00})
        # Now evaluate only c0, which does not contain x=10.
        only_c0 = self.run(
            service, rois, chunks=[self.chunks()[0]], cached=("c0",)
        )
        assert only_c0[0] == []
        # ...and back again, to prove the first result was not cached over.
        assert self.run(service, rois, cached=("c0", "c1")) == both

    def test_scopes_are_bounded(self):
        # A scope is a render layer. Removing and re-adding a layer mints a new
        # one, and nothing on the viewer side can ever ask for the old one's
        # geometry back, so layer churn would otherwise strand it forever.
        service = RoiFilterService(max_scopes=3)
        for i in range(10):
            query, body = make_request(
                f"scope{i}",
                [("c0", [line(-1, 1)], [1])],
                [group([sphere_roi([0, 0, 0], 1)])],
            )
            service.handle(decode_request(query, body))
        assert len(service._scope_order) == 3
        assert {k[0] for k in service._chunks} == {"scope7", "scope8", "scope9"}

    def test_a_live_scope_is_not_evicted_by_its_own_traffic(self):
        service = RoiFilterService(max_scopes=2)
        for _ in range(10):
            query, body = make_request(
                "live",
                [("c0", [line(-1, 1)], [1])],
                [group([sphere_roi([0, 0, 0], 1)])],
                cached=("c0",) if service._chunks else (),
            )
            status = service.handle(decode_request(query, body))
            assert status  # no KeyError: its geometry survived
        assert ("live", "c0") in service._chunks

    def test_matches_the_direct_engine(self):
        # The service path must agree with evaluating a hand-built index.
        service = RoiFilterService()
        groups = [group([sphere_roi([-10, 0, 0], 1)], color=0x99)]
        via_service = self.run(service, groups)

        frags = [line(-11, -9), line(-1, 1, y=20.0), line(9, 11)]
        offsets = np.zeros(4, dtype=np.intp)
        np.cumsum([len(f) for f in frags], out=offsets[1:])
        index = TractIndex(
            np.concatenate(frags), offsets, np.array([1, 2, 1], dtype=np.uint64)
        )
        spec = RoiGroupSpec(
            rois=(Roi(Ellipsoid(np.array([-10.0, 0, 0]), np.array([1.0, 1, 1])),
                      RoiPredicate.ANY_SEGMENT, RoiOperator.AND),),
            color_packed=0x99,
            visible=True,
            high_detail=False,
        )
        passing, colors, _ = evaluate_groups(index, [spec])
        assert via_service[0] == sorted(int(x) for x in passing)
        assert via_service[2] == {int(k): int(v) for k, v in colors.items()}
