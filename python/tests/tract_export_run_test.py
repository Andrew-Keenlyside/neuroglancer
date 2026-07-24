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
"""Selection logic for the tract exporter.

Geometry is injected rather than read from a store: what needs testing is the
index construction, the fold, and the read-narrowing decision -- none of which
depend on zarr, and all of which are where a wrong export comes from.
"""

import copy

import numpy as np
import pytest
from neuroglancer.tract_export import JobSpecError, parse_job
from neuroglancer.tract_export import run as run_mod
from neuroglancer.tract_export.run import (
    ExportRunError,
    build_index,
    passing_object_ids,
    per_group_from_object_ids,
    store_path_from_url,
)

BASE = {
    "schemaVersion": 1,
    "source": {"url": "zarr-vectors://gs://bucket/tracts.zvf", "level": 0},
    "groups": [],
    "format": "trk",
    "destination": {"kind": "local", "path": "out.trk"},
}


def job_with(rois, **overrides):
    spec = copy.deepcopy(BASE)
    spec["groups"] = [{"name": "g", "color": "#ff0000", "rois": rois}]
    spec.update(overrides)
    return parse_job(spec)


def sphere(center, radius, operator="and", predicate="any_segment"):
    return {
        "shape": {
            "type": "ellipsoid",
            "center": list(center),
            "radii": [radius] * 3,
        },
        "predicate": predicate,
        "operator": operator,
    }


def line(*points):
    """One polyline as a single fragment."""
    return [np.array(points, dtype=np.float32)]


class TestStorePathFromUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("zarr-vectors://gs://b/x.zvf", "gs://b/x.zvf"),
            ("zarr://gs://b/x.zvf", "gs://b/x.zvf"),
            ("zarr3://gs://b/x.zvf", "gs://b/x.zvf"),
            # A hand-written spec may carry a bare path; pass it through.
            ("/data/tracts.zvf", "/data/tracts.zvf"),
            ("gs://b/x.zvf", "gs://b/x.zvf"),
        ],
    )
    def test_strips_only_the_datasource_scheme(self, url, expected):
        assert store_path_from_url(url) == expected

    def test_strips_at_most_one_prefix(self):
        # The inner scheme belongs to the store and must survive.
        assert store_path_from_url("zarr-vectors://zarr://x") == "zarr://x"

    @pytest.mark.parametrize(
        "url,expected",
        [
            # The viewer's KvStore pipeline form: store URL then format adapter.
            (
                "https://h/b/store.zarrvectors/|zarr-vectors:",
                "https://h/b/store.zarrvectors/",
            ),
            ("gs://b/store/|zarr-vectors:", "gs://b/store/"),
            # A parameterised adapter is stripped too.
            ("https://h/b/store/|zarr-vectors:foo=1", "https://h/b/store/"),
        ],
    )
    def test_strips_the_viewer_pipeline_suffix(self, url, expected):
        assert store_path_from_url(url) == expected


class TestBuildIndex:
    def test_one_row_per_polyline_with_fragments_concatenated(self):
        # Two fragments of one object must become a single row, or the endpoint
        # predicates would test the chunk boundary between them.
        polylines = [[np.zeros((3, 3), np.float32), np.ones((2, 3), np.float32)]]
        index = build_index(polylines, [7])
        assert index.n_rows == 1
        assert index.offsets.tolist() == [0, 5]
        assert index.object_ids.tolist() == [7]

    def test_rows_align_with_object_ids(self):
        index = build_index(
            [line((0, 0, 0), (1, 0, 0)), line((5, 0, 0), (6, 0, 0))], [11, 4]
        )
        assert index.n_rows == 2
        # `object_ids` is the ascending unique set, not the input order.
        assert index.object_ids.tolist() == [4, 11]

    def test_empty_input_yields_an_empty_index(self):
        index = build_index([], [])
        assert len(index) == 0
        assert index.n_rows == 0

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ExportRunError, match="must correspond"):
            build_index([line((0, 0, 0), (1, 0, 0))], [1, 2])


class TestPassingObjectIds:
    def _index(self):
        # id 1 crosses the origin; id 2 sits out at x=50; id 3 crosses both.
        return build_index(
            [
                line((-5, 0, 0), (5, 0, 0)),
                line((50, 0, 0), (55, 0, 0)),
                line((-5, 0, 0), (55, 0, 0)),
            ],
            [1, 2, 3],
        )

    def test_include_selects_crossing_objects(self):
        ids = passing_object_ids(
            self._index(), job_with([sphere((0, 0, 0), 2)]).groups[0]
        )
        assert ids.tolist() == [1, 3]

    def test_leading_exclusion_selects_the_complement(self):
        # The Stage 1 semantics, end to end through the job spec.
        group = job_with([sphere((0, 0, 0), 2, operator="andnot")]).groups[0]
        assert passing_object_ids(self._index(), group).tolist() == [2]

    def test_include_then_exclude(self):
        group = job_with(
            [sphere((0, 0, 0), 2), sphere((52, 0, 0), 5, operator="andnot")]
        ).groups[0]
        assert passing_object_ids(self._index(), group).tolist() == [1]

    def test_empty_index_selects_nothing(self):
        group = job_with([sphere((0, 0, 0), 2)]).groups[0]
        assert passing_object_ids(build_index([], []), group).size == 0


class TestReadStrategy:
    """One unfiltered read, then a batched fold.

    `read_polylines` picks its strategy from `object_ids=`: an explicit subset
    reads each object's manifest individually, while no subset prefetches every
    chunk in one gather. Asking for a whole level object-by-object takes the
    pathological branch -- badly so over HTTP. And geometric narrowing is unsafe
    at any speed: `bbox=` drops a tract whose long segment leaps the chunk
    holding the ROI, `chunks=` splits one tract into several.
    """

    def _capture(self, monkeypatch, total=3):
        calls = []

        def fake_read_polylines(store, *, level, **kwargs):
            # Faithful to the real signature's optional kwargs, so reintroducing
            # `chunks=`, `bbox=` or per-object reads is caught here.
            calls.append({"store": store, "level": level, "kwargs": kwargs})
            return {
                "polylines": [line((0, 0, 0), (1, 0, 0)) for _ in range(total)],
                "object_ids": list(range(total)),
                "polyline_count": total,
                "vertex_count": 2 * total,
            }

        monkeypatch.setattr(
            run_mod, "_require_zarr_vectors", lambda: fake_read_polylines
        )
        return calls

    def test_reads_once_with_no_filtering_kwargs(self, monkeypatch):
        calls = self._capture(monkeypatch)
        run_mod.select(job_with([sphere((0, 0, 0), 5)]))
        assert len(calls) == 1, f"expected a single read, got {len(calls)}"
        assert calls[0]["kwargs"] == {}, calls[0]["kwargs"]

    def test_does_not_read_per_object(self, monkeypatch):
        # The regression that made a level-3 export take minutes instead of
        # seconds: object_ids= forces a manifest read per object.
        calls = self._capture(monkeypatch, total=500)
        run_mod.select(job_with([sphere((0, 0, 0), 5)]))
        assert len(calls) == 1
        assert "object_ids" not in calls[0]["kwargs"]

    def test_never_passes_bbox_even_for_bounded_regions(self, monkeypatch):
        calls = self._capture(monkeypatch)
        run_mod.select(job_with([sphere((10, 10, 10), 2), sphere((0, 0, 0), 1, "or")]))
        assert "bbox" not in calls[0]["kwargs"]

    def test_never_passes_chunks(self, monkeypatch):
        calls = self._capture(monkeypatch)
        run_mod.select(job_with([sphere((0, 0, 0), 2, operator="andnot")]))
        assert "chunks" not in calls[0]["kwargs"]

    def test_batch_size_does_not_change_the_selection(self, monkeypatch):
        self._capture(monkeypatch, total=40)
        whole, g_whole = run_mod.select(job_with([sphere((0, 0, 0), 5)]), batch=10_000)
        split, g_split = run_mod.select(job_with([sphere((0, 0, 0), 5)]), batch=7)
        assert whole == split == 40
        assert g_whole[0][1].tolist() == g_split[0][1].tolist()

    def test_passes_the_stripped_store_path_and_level(self, monkeypatch):
        calls = self._capture(monkeypatch)
        run_mod.select(
            job_with(
                [sphere((0, 0, 0), 1)],
                source={"url": "zarr-vectors://gs://b/x.zvf", "level": 2},
            )
        )
        assert calls[0]["store"] == "gs://b/x.zvf"
        assert calls[0]["level"] == 2

    def test_fails_clearly_when_object_ids_are_absent(self, monkeypatch):
        def old_build(store, *, level, **kwargs):
            return {"polylines": [], "polyline_count": 0, "vertex_count": 0}

        monkeypatch.setattr(run_mod, "_require_zarr_vectors", lambda: old_build)
        with pytest.raises(ExportRunError, match="does not report `object_ids`"):
            run_mod.select(job_with([sphere((0, 0, 0), 1)]))


class TestMultiGroup:
    """Regressions for group-level bugs found by adversarial review."""

    def _patch(self, monkeypatch):
        def fake(store, *, level, **kwargs):
            return {
                # id 1 near the origin, id 2 out at x=50.
                "polylines": [
                    line((-5, 0, 0), (5, 0, 0)),
                    line((45, 0, 0), (55, 0, 0)),
                ],
                "object_ids": [1, 2],
                "polyline_count": 2,
                "vertex_count": 4,
            }

        monkeypatch.setattr(run_mod, "_require_zarr_vectors", lambda: fake)

    def _two_groups(self, name_a, name_b):
        spec = copy.deepcopy(BASE)
        spec["groups"] = [
            {"name": name_a, "color": "#ff0000", "rois": [sphere((0, 0, 0), 2)]},
            {"name": name_b, "color": "#00ff00", "rois": [sphere((50, 0, 0), 2)]},
        ]
        return parse_job(spec)

    def test_unions_ids_across_groups(self, monkeypatch):
        self._patch(monkeypatch)
        _, per_group = run_mod.select(self._two_groups("A", "B"))
        assert [name for name, _ in per_group] == ["A", "B"]
        assert run_mod.union_ids(per_group).tolist() == [1, 2]

    def test_duplicate_group_names_do_not_collapse(self, monkeypatch):
        # The viewer names groups `Group ${n+1}`, so deleting a middle group and
        # adding another reproduces an existing name. Keying results by name
        # silently dropped every colliding group but the last from the union.
        self._patch(monkeypatch)
        _, per_group = run_mod.select(self._two_groups("Group 3", "Group 3"))
        assert len(per_group) == 2
        assert run_mod.union_ids(per_group).tolist() == [1, 2]

    def test_union_of_no_groups_is_empty(self):
        assert run_mod.union_ids([]).tolist() == []

    def test_union_ignores_empty_groups(self):
        import numpy as _np

        per_group = [
            ("a", _np.array([], dtype=_np.uint64)),
            ("b", _np.array([7], dtype=_np.uint64)),
        ]
        assert run_mod.union_ids(per_group).tolist() == [7]


class TestEmptySelection:
    """An empty dissection must not report a file it never wrote."""

    def test_reports_not_written_and_calls_no_writer(self, monkeypatch, tmp_path):
        def fake(store, *, level, **kwargs):
            return {
                "polylines": [line((100, 100, 100), (101, 100, 100))],
                "object_ids": [1],
                "polyline_count": 1,
                "vertex_count": 2,
            }

        monkeypatch.setattr(run_mod, "_require_zarr_vectors", lambda: fake)

        def explode(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("writer called for an empty selection")

        monkeypatch.setattr(run_mod, "_export_trk", explode)
        monkeypatch.setattr(run_mod, "_export_zvf", explode)

        out = tmp_path / "out.trk"
        spec = copy.deepcopy(BASE)
        spec["groups"] = [
            {"name": "g", "color": "#ff0000", "rois": [sphere((0, 0, 0), 2)]}
        ]
        spec["destination"] = {"kind": "local", "path": str(out)}
        summary = run_mod.run_job(parse_job(spec))

        assert summary["written"] is False
        assert summary["streamline_count"] == 0
        assert summary["object_count"] == 0
        assert not out.exists()

    def test_rejects_a_non_local_destination(self):
        # gcs is now rejected at parse -- before the expensive read+fold -- so
        # no monkeypatched reader or run is needed. See job.py.
        spec = copy.deepcopy(BASE)
        spec["groups"] = [
            {"name": "g", "color": "#ff0000", "rois": [sphere((0, 0, 0), 2)]}
        ]
        spec["destination"] = {"kind": "gcs", "path": "gs://b/out.trk"}
        with pytest.raises(JobSpecError, match="not supported yet"):
            parse_job(spec)


class TestWholeScope:
    """`scope="whole"` exports every object at the level with no ROI fold."""

    def _patch(self, monkeypatch):
        def fake(store, *, level, **kwargs):
            return {
                "polylines": [line((-5, 0, 0), (5, 0, 0)), line((45, 0, 0), (55, 0, 0))],
                "object_ids": [2, 1],  # unsorted, to prove the union sorts
                "polyline_count": 2,
                "vertex_count": 4,
            }

        monkeypatch.setattr(run_mod, "_require_zarr_vectors", lambda: fake)

    def test_selects_every_object_and_calls_the_writer_with_all_ids(
        self, monkeypatch, tmp_path
    ):
        self._patch(monkeypatch)
        captured = {}

        def fake_trk(job, ids, output_path):
            captured["ids"] = [int(i) for i in ids]
            return {"streamline_count": len(ids), "vertex_count": 0, "written": True}

        monkeypatch.setattr(run_mod, "_export_trk", fake_trk)

        s = copy.deepcopy(BASE)
        s["scope"] = "whole"
        s["groups"] = []  # whole ignores groups
        s["destination"] = {"kind": "local", "path": str(tmp_path / "out.trk")}
        summary = run_mod.run_job(parse_job(s))

        assert captured["ids"] == [1, 2]
        assert summary["object_count"] == 2
        assert summary["candidate_object_count"] == 2
        assert summary["groups"] == []
        assert summary["written"] is True


class TestExplicitObjectIds:
    """v3 fast path: the viewer hands over the passing ids; nothing is read/folded.

    This is the fix for a sparse export that read (and, under Pyodide, re-read to
    convergence) the whole level. With ids in the spec, `run_job` reads no level
    and folds nothing -- it just writes the union by id.
    """

    def _job(self, groups, **overrides):
        s = copy.deepcopy(BASE)
        s["schemaVersion"] = 3
        s["groups"] = groups
        s.update(overrides)
        return parse_job(s)

    def test_per_group_from_object_ids_dedups_and_sorts(self):
        job = self._job(
            [{"name": "A", "color": "#ff0000", "objectIds": ["5", "5", "3"]}]
        )
        per_group = per_group_from_object_ids(job.groups)
        assert [name for name, _ in per_group] == ["A"]
        assert per_group[0][1].tolist() == [3, 5]

    def test_run_job_reads_no_level_and_writes_the_id_union(
        self, monkeypatch, tmp_path
    ):
        def explode(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("whole-level read/fold ran for an id-based job")

        monkeypatch.setattr(run_mod, "_read_level", explode)
        monkeypatch.setattr(run_mod, "select", explode)

        captured = {}

        def fake_trk(job, ids, output_path):
            captured["ids"] = sorted(int(i) for i in ids)
            return {"streamline_count": len(ids), "vertex_count": 0, "written": True}

        monkeypatch.setattr(run_mod, "_export_trk", fake_trk)

        # A large id proves the string ids survive as exact uint64.
        job = self._job(
            [
                {"name": "A", "color": "#ff0000", "objectIds": ["1", "2"]},
                {
                    "name": "B",
                    "color": "#00ff00",
                    "objectIds": ["2", "9007199254740993"],
                },
            ],
            destination={"kind": "local", "path": str(tmp_path / "out.trk")},
        )
        summary = run_mod.run_job(job)

        assert captured["ids"] == [1, 2, 9007199254740993]
        assert summary["object_count"] == 3  # union deduplicates the shared id 2
        assert summary["groups"] == [
            {"name": "A", "count": 2},
            {"name": "B", "count": 2},
        ]
        assert summary["written"] is True

    def test_empty_id_selection_writes_nothing(self, monkeypatch, tmp_path):
        def explode(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("read or writer ran for an empty id selection")

        monkeypatch.setattr(run_mod, "_read_level", explode)
        monkeypatch.setattr(run_mod, "_export_trk", explode)
        monkeypatch.setattr(run_mod, "_export_zvf", explode)

        job = self._job(
            [{"name": "A", "color": "#ff0000", "objectIds": []}],
            destination={"kind": "local", "path": str(tmp_path / "out.trk")},
        )
        summary = run_mod.run_job(job)
        assert summary["written"] is False
        assert summary["object_count"] == 0

    def test_mixed_id_and_fold_groups_are_rejected(self, tmp_path):
        # The viewer sends ids for all selected groups or none; a mixed spec would
        # drag the whole level back in for the fold groups. Fail fast.
        job = self._job(
            [
                {"name": "A", "color": "#ff0000", "objectIds": ["1"]},
                {"name": "B", "color": "#00ff00", "rois": [sphere((0, 0, 0), 2)]},
            ],
            destination={"kind": "local", "path": str(tmp_path / "out.trk")},
        )
        with pytest.raises(ExportRunError, match="objectIds for all groups or none"):
            run_mod.run_job(job)


class TestCliSummaryShape:
    """The CLI builds its dry-run summary by hand, so it can drift from select().

    It did: when `select` changed from returning a TractIndex to returning a
    count, `--dry-run` kept calling `len(index)` and died with a TypeError that
    no unit test covered.
    """

    def test_dry_run_reports_the_same_keys_as_a_real_run(self, monkeypatch, tmp_path):
        import json

        from neuroglancer.tract_export.__main__ import main

        def fake(store, *, level, **kwargs):
            return {
                "polylines": [
                    line((-5, 0, 0), (5, 0, 0)),
                    line((50, 0, 0), (55, 0, 0)),
                ],
                "object_ids": [1, 2],
                "polyline_count": 2,
                "vertex_count": 4,
            }

        monkeypatch.setattr(run_mod, "_require_zarr_vectors", lambda: fake)

        spec_file = tmp_path / "job.json"
        s = copy.deepcopy(BASE)
        s["groups"] = [
            {"name": "g", "color": "#ff0000", "rois": [sphere((0, 0, 0), 2)]}
        ]
        s["destination"] = {"kind": "local", "path": str(tmp_path / "out.trk")}
        spec_file.write_text(json.dumps(s), encoding="utf-8")

        out = tmp_path / "captured.json"
        with out.open("w", encoding="utf-8") as fh:
            monkeypatch.setattr("sys.stdout", fh)
            assert main([str(spec_file), "--dry-run"]) == 0
        summary = json.loads(out.read_text(encoding="utf-8"))

        assert summary["dryRun"] is True
        assert summary["written"] is False
        assert summary["candidate_object_count"] == 2
        assert summary["object_count"] == 1
        assert summary["groups"] == [{"name": "g", "count": 1}]
        assert not (tmp_path / "out.trk").exists()
