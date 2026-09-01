#!/usr/bin/env python
"""Split one object into two by removing a single link, IN PLACE.

The companion script (`split_skeleton.py`) proves the result by rebuilding a
store. This one performs the edit as an actual mutation of an existing ZVF
store: `EditSession.remove_link(ref, update_objects=True)` drops one branch link
and re-attributes the two sides to two objects, leaving every other object
untouched. That is the primitive a proofreading UI would eventually call.

Constraints that are real, not incidental (each was measured, see README):

  * Only an EXPLICIT link can be removed. In the `implicit_sequential_with_branches`
    convention the edges inside a fragment are implied by adjacency and have no
    row to delete, so the cut must land on a branch link.
  * `atomic=True` (the default) is required. The in-place variant empties the
    whole link fragment, destroying the edges of every other object sharing it.
  * The second output object's id is never reported, so it is recovered by
    diffing the set of present objects across the edit.

Usage:
    python inplace_split.py --store /tmp/zv_poc/single_axon.zv --object 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import zarr_vectors as zv
from zarr_vectors.api.edit import EditPlan
from zarr_vectors.core.arrays import (
    read_link_fragment_index,
    read_object_manifest,
    read_fragment,
)
from zarr_vectors.types.skeletons import read_chunk_link_fragment
from zarr_vectors.core.store import get_resolution_level, open_store


def present_objects(store_path: str, limit: int = 64) -> dict[int, int]:
    """{object id: vertex count} for the objects that actually hold geometry."""
    root = open_store(store_path)
    lg = get_resolution_level(root, 0)
    out: dict[int, int] = {}
    for oid in range(limit):
        try:
            manifest = read_object_manifest(lg, oid)
        except Exception:
            break
        if not manifest:
            continue
        total = 0
        for cc, fi in manifest:
            total += len(np.asarray(read_fragment(lg, tuple(cc), fi, ndim=3)))
        out[oid] = total
    return out


def find_intra_chunk_link(store_path: str, object_id: int):
    """A removable link belonging to `object_id`: explicit and intra-chunk.

    Links are addressed as `(chunk, link fragment, row)`, not by a flat index:
    the skeleton writer emits one link fragment per vertex fragment, each
    holding the single branch link that reattaches that path to its parent.
    Fragment 0 is the root path, so it is empty -- which is why a naive
    `fragment=0, row=0` reference fails with "row 0 out of range".
    """
    root = open_store(store_path)
    lg = get_resolution_level(root, 0)
    manifest = read_object_manifest(lg, object_id)
    chunks = sorted({tuple(cc) for cc, _ in manifest})
    for cc in chunks:
        index = read_link_fragment_index(lg, cc)
        for fragment in range(len(index)):
            rows = np.asarray(read_chunk_link_fragment(lg, cc, fragment))
            if rows.size == 0:
                continue
            return cc, fragment, 0, rows[0].tolist()
    return None, None, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, help="store to COPY and edit")
    ap.add_argument("--object", type=int, default=1)
    ap.add_argument("--out", default=None, help="where to put the edited copy")
    args = ap.parse_args()

    src = Path(args.store)
    dst = Path(args.out or (str(src).rstrip("/") + ".inplace_split.zv"))
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    print(f"copied {src} -> {dst}")

    before = present_objects(str(dst))
    print("before:", before)

    chunk, fragment, row, record = find_intra_chunk_link(str(dst), args.object)
    if chunk is None:
        raise SystemExit(
            f"object {args.object} has no intra-chunk explicit link to remove; "
            "in this convention only branch links are removable"
        )
    print(f"removing link: chunk {chunk} fragment {fragment} row {row} "
          f"(endpoints {record})")

    # zv.aopen() has no mode parameter and always opens read-only; the writable
    # door is zv.open(path, mode="r+").
    ds = zv.open(str(dst), mode="r+")
    with EditPlan(ds) as plan:                     # atomic=True by default
        session = plan.session
        from zarr_vectors.ops.refs import LinkRef

        ref = LinkRef(level=0, chunk=chunk, fragment=fragment, row=row,
                      delta=0, offsets=((0, 0, 0),))
        session.remove_link(ref, update_objects=True)
    print("renamed():", plan.renamed())

    after = present_objects(str(dst))
    print("after: ", after)
    new_ids = sorted(set(after) - set(before))
    print(f"new object ids: {new_ids}")
    total_before = sum(before.values())
    total_after = sum(v for k, v in after.items() if k in new_ids) or sum(after.values())
    print(f"vertices before {total_before}, in new objects {total_after}")
    (dst.parent / "inplace_report.json").write_text(
        json.dumps({"store": str(dst), "before": before, "after": after,
                    "new_ids": new_ids,
                    "removed_link": {"fragment": fragment, "row": row,
                                     "endpoints": record},
                    "chunk": list(chunk)}, indent=2)
    )


if __name__ == "__main__":
    main()
