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
"""Job-spec parsing for the tract exporter."""

import copy

import numpy as np
import pytest
from neuroglancer.tract_export import (
    JOB_SCHEMA_VERSION,
    JobSpecError,
    parse_job,
    roi_bounds,
)
from neuroglancer.tractography.roi import RoiOperator, RoiPredicate

# A spec in exactly the shape the Export tab posts: groups are `groupToJson`
# output, i.e. shapes keyed by `type` with string predicate/operator names.
#
# CONTRACT: this literal is duplicated, deliberately, as the expected value in
# `src/datasource/zarr-vectors/export_job.spec.ts`. The two languages have no
# shared schema, so the pairing is what keeps them honest -- change one and the
# other suite fails, rather than the feature breaking silently at runtime.
#
# One asymmetry the pairing alone does not catch, hence the assertion below:
# `parse_job` accepts any version <= JOB_SCHEMA_VERSION, so `VALID` would keep
# parsing after a Python-side bump while the TS suite -- which asserts the
# literal 1 in its *output* -- also stays green. The bump would then ship with
# the two sides silently disagreeing about the current version.
VALID = {
    "schemaVersion": 1,
    "source": {"url": "zarr-vectors://gs://bucket/tracts.zvf", "level": 0},
    "groups": [
        {
            "name": "Motor CST",
            "color": "#ff0000",
            "rois": [
                {
                    "shape": {
                        "type": "ellipsoid",
                        "center": [10, 20, 30],
                        "radii": [5, 5, 5],
                    },
                    "predicate": "any_segment",
                    "operator": "and",
                    "name": "seed",
                },
                {
                    "shape": {
                        "type": "box",
                        "lower": [0, 0, 0],
                        "upper": [100, 100, 100],
                    },
                    "predicate": "either_endpoint",
                    "operator": "andnot",
                },
            ],
        }
    ],
    "format": "trk",
    "destination": {"kind": "local", "path": "out.trk"},
}


def test_fixture_pins_the_current_schema_version():
    """The fixture must track the version, not merely be accepted by it.

    Without this, bumping `JOB_SCHEMA_VERSION` on the Python side leaves both
    suites green -- see the CONTRACT note above. Bumping should force this
    fixture, and therefore its TypeScript twin, to be updated in the same
    change.
    """
    assert VALID["schemaVersion"] == JOB_SCHEMA_VERSION


def spec(**overrides):
    s = copy.deepcopy(VALID)
    s.update(overrides)
    return s


class TestParseJob:
    def test_parses_a_valid_spec(self):
        job = parse_job(VALID)
        assert job.source_url == "zarr-vectors://gs://bucket/tracts.zvf"
        assert job.level == 0
        assert job.format == "trk"
        assert job.destination.kind == "local"
        assert job.destination.path == "out.trk"
        assert job.affine is None
        assert len(job.groups) == 1
        assert job.groups[0].name == "Motor CST"

    def test_maps_string_enums_onto_the_fold_enums(self):
        rois = parse_job(VALID).groups[0].rois
        assert rois[0].predicate == RoiPredicate.ANY_SEGMENT
        assert rois[0].operator == RoiOperator.AND
        assert rois[1].predicate == RoiPredicate.EITHER_ENDPOINT
        assert rois[1].operator == RoiOperator.ANDNOT

    def test_drops_the_display_only_roi_name(self):
        # `name` exists in the persisted JSON but must not reach the fold.
        roi = parse_job(VALID).groups[0].rois[0]
        assert not hasattr(roi, "name")

    def test_level_defaults_to_zero(self):
        s = spec()
        del s["source"]["level"]
        assert parse_job(s).level == 0

    def test_accepts_a_4x4_affine(self):
        s = spec(affine=np.eye(4).tolist())
        assert parse_job(s).affine.shape == (4, 4)

    def test_all_three_shape_types(self):
        for shape in (
            {"type": "ellipsoid", "center": [0, 0, 0], "radii": [1, 1, 1]},
            {"type": "box", "lower": [0, 0, 0], "upper": [1, 1, 1]},
            {"type": "halfspace", "origin": [0, 0, 0], "normal": [0, 0, 1]},
        ):
            s = spec()
            s["groups"][0]["rois"] = [
                {"shape": shape, "predicate": "any_segment", "operator": "and"}
            ]
            assert len(parse_job(s).groups[0].rois) == 1


class TestRejections:
    def test_rejects_a_newer_schema_version(self):
        with pytest.raises(JobSpecError, match="newer than this exporter"):
            parse_job(spec(schemaVersion=2))

    def test_rejects_an_unknown_format(self):
        with pytest.raises(JobSpecError, match="format"):
            parse_job(spec(format="obj"))

    def test_rejects_an_unknown_destination_kind(self):
        with pytest.raises(JobSpecError, match="destination.kind"):
            parse_job(spec(destination={"kind": "s3", "path": "x"}))

    def test_rejects_gcs_at_parse_before_the_read(self):
        # A known-but-unimplemented kind fails fast with the workaround, rather
        # than after run_job reads and folds the whole level.
        with pytest.raises(JobSpecError, match="not supported yet"):
            parse_job(spec(destination={"kind": "gcs", "path": "gs://b/x.trk"}))

    def test_rejects_a_missing_source_url(self):
        with pytest.raises(JobSpecError, match="url"):
            parse_job(spec(source={"level": 0}))

    def test_rejects_empty_groups(self):
        with pytest.raises(JobSpecError, match="groups"):
            parse_job(spec(groups=[]))

    def test_rejects_a_group_with_no_rois(self):
        # An empty fold passes everything, so this would export the entire
        # dataset under the group's name rather than a dissection.
        s = spec()
        s["groups"][0]["rois"] = []
        with pytest.raises(JobSpecError, match="every streamline"):
            parse_job(s)

    def test_rejects_an_unknown_shape_type(self):
        s = spec()
        s["groups"][0]["rois"][0]["shape"]["type"] = "cone"
        with pytest.raises(JobSpecError, match="cone"):
            parse_job(s)

    def test_rejects_an_unknown_operator(self):
        s = spec()
        s["groups"][0]["rois"][0]["operator"] = "nand"
        with pytest.raises(JobSpecError, match="nand"):
            parse_job(s)

    def test_names_the_offending_roi(self):
        s = spec()
        del s["groups"][0]["rois"][1]["predicate"]
        with pytest.raises(JobSpecError, match=r"groups\[0\].rois\[1\]"):
            parse_job(s)

    def test_rejects_a_non_4x4_affine(self):
        with pytest.raises(JobSpecError, match="4x4"):
            parse_job(spec(affine=[[1, 0], [0, 1]]))

    def test_rejects_a_wire_format_shape(self):
        # The wire encoding keys shapes by `kind`, not `type`. Accepting it here
        # would silently diverge the two encodings.
        s = spec()
        s["groups"][0]["rois"][0]["shape"] = {
            "kind": "ellipsoid",
            "center": [0, 0, 0],
            "radii": [1, 1, 1],
        }
        with pytest.raises(JobSpecError, match="type"):
            parse_job(s)


class TestRoiBounds:
    def _roi(self, shape, operator="and"):
        s = spec()
        s["groups"][0]["rois"] = [
            {"shape": shape, "predicate": "any_segment", "operator": operator}
        ]
        return parse_job(s).groups[0].rois

    def test_encloses_an_ellipsoid(self):
        rois = self._roi(
            {"type": "ellipsoid", "center": [10, 10, 10], "radii": [2, 3, 4]}
        )
        lower, upper = roi_bounds(rois)
        assert list(lower) == [8, 7, 6]
        assert list(upper) == [12, 13, 14]

    def test_unions_several_regions(self):
        rois = (
            parse_job(
                spec(
                    groups=[
                        {
                            "name": "g",
                            "color": "#ffffff",
                            "rois": [
                                {
                                    "shape": {
                                        "type": "box",
                                        "lower": [0, 0, 0],
                                        "upper": [1, 1, 1],
                                    },
                                    "predicate": "any_segment",
                                    "operator": "and",
                                },
                                {
                                    "shape": {
                                        "type": "box",
                                        "lower": [5, 5, 5],
                                        "upper": [7, 7, 7],
                                    },
                                    "predicate": "any_segment",
                                    "operator": "or",
                                },
                            ],
                        }
                    ]
                )
            )
            .groups[0]
            .rois
        )
        lower, upper = roi_bounds(rois)
        assert list(lower) == [0, 0, 0]
        assert list(upper) == [7, 7, 7]

    def test_unbounded_for_a_halfspace(self):
        rois = self._roi(
            {"type": "halfspace", "origin": [0, 0, 0], "normal": [0, 0, 1]}
        )
        assert roi_bounds(rois) is None

    def test_unbounded_when_any_region_excludes(self):
        # What passes an exclusion is its *complement*, which no finite box
        # encloses -- narrowing the read to the region itself would drop exactly
        # the tracts that should survive.
        rois = self._roi(
            {"type": "ellipsoid", "center": [0, 0, 0], "radii": [1, 1, 1]},
            operator="andnot",
        )
        assert roi_bounds(rois) is None
