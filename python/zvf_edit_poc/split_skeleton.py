#!/usr/bin/env python
"""Proof of concept: extract one skeleton from a ZVF store and cut it in two.

Two stores are written:

  ``<out>/single_axon.zv``  the selected object on its own, one object
  ``<out>/split_axon.zv``   the same geometry with one edge removed, so the
                            two halves are two independent objects

The cut is the atomic edit: removing a single edge from a tree partitions it,
and the two connected components become object 0 and object 1. Nothing else
about the geometry changes -- same vertices, same coordinates, one fewer edge.

Why the geometry is rebuilt rather than edited in place: a ZVF store's
per-object membership lives in ``object_index/manifests``, which is positional,
and splitting an object adds one. The library's in-place edit path
(``zarr_vectors.ops.edit.EditSession``) owns those invariants, but its object
ids churn on write -- see ``zarr_vectors/api/edit.py``, which says a polyline
edit reallocates under both atomic settings. For a single-axon store the whole
rewrite is well under a second, so the proof of concept takes the path whose
result is unambiguous.

Reading one object is manifest-driven and NOT ``read_graph``: that function
refuses an ``object_ids`` filter outright, and its synthesised edge list chains
across object boundaries (measured on this store: 209,416 edges longer than
10 um, the longest 868 um, against a real median edge of 1.9 um).

Usage:
    python split_skeleton.py --store /path/to/skeletons.zv --object 281 \
        --out /tmp/zv_poc
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import read_fragment, read_object_manifest
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.encoding.fragments import read_fragment_index
from zarr_vectors.types.skeletons import _read_attr_fragment as read_attr_fragment
from zarr_vectors.core.arrays import read_links
from zarr_vectors.core.arrays import write_object_index
from zarr_vectors.types.skeletons import (
    finalize_skeleton_store,
    init_skeleton_store,
    write_skeleton_chunk,
)


# ---------------------------------------------------------------------------
# Read one object
# ---------------------------------------------------------------------------


def read_raw_zv_attrs(store_path: str) -> dict:
    """The root `zarr_vectors` attrs as written.

    Not `RootMetadata`: its `from_dict` picks out the keys it knows by name, and
    `coordinate_offset` is not among them, so it silently reads as absent. The
    offset is the difference between this store's local frame and the absolute
    nanometre frame its sibling synapse/mesh stores use, so losing it puts the
    output in the wrong place by a quarter of a millimetre.
    """
    import json as _json

    return _json.loads(
        (Path(store_path) / "zarr.json").read_text()
    ).get("attributes", {}).get("zarr_vectors", {})


def extract_object(store_path: str, object_id: int, level: int = 0):
    """One object's vertices and edges, in the store's own coordinates.

    Returns ``(positions, edges, info)``. Edges are indices into ``positions``.
    When the store declares a `vertex_id_attribute`, `info["node_ids"]` carries
    that column, parallel to the positions -- the stable identity the viewer
    picks by.

    A ``implicit_sequential_with_branches`` skeleton stores its geometry as
    fragments -- each a path, contributing ``count - 1`` sequential edges -- plus
    explicit branch records in ``links/0`` that rejoin them. Both halves are
    needed: fragments alone leave the object as disconnected pieces.
    """
    root = open_store(store_path)
    meta = read_root_metadata(root)
    lg = get_resolution_level(root, level)
    ndim = meta.sid_ndim

    manifest = read_object_manifest(lg, object_id)
    if not manifest:
        raise SystemExit(f"object {object_id} has no manifest in {store_path}")

    id_attribute = read_raw_zv_attrs(store_path).get("vertex_id_attribute")
    node_ids: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    row_keys: list[tuple] = []          # (chunk_coords, row) per output vertex
    spans: list[tuple[int, int]] = []   # (start, count) per fragment
    index_cache: dict[tuple, object] = {}
    total = 0
    for chunk_coords, fragment_index in manifest:
        key = tuple(chunk_coords)
        fidx = index_cache.get(key)
        if fidx is None:
            fidx = index_cache[key] = read_fragment_index(lg, "vertex_fragments", key)
        if fidx.is_range(fragment_index):
            start, count = fidx.range(fragment_index)
            rows = np.arange(start, start + count, dtype=np.int64)
        else:
            rows = np.asarray(fidx.indices(fragment_index), dtype=np.int64)
        pts = np.asarray(read_fragment(lg, key, fragment_index, ndim=ndim))
        if id_attribute is not None:
            column = read_attr_fragment(lg, id_attribute, key, fragment_index)
            node_ids.append(
                np.asarray(column).reshape(-1)
                if column is not None
                else np.zeros(len(pts), dtype=np.int64)
            )
        if len(pts) != len(rows):
            raise SystemExit(
                f"fragment {key}/{fragment_index}: {len(pts)} vertices but "
                f"{len(rows)} rows in the fragment index"
            )
        spans.append((total, len(pts)))
        total += len(pts)
        positions.append(pts)
        row_keys.extend((key, int(r)) for r in rows)

    P = np.concatenate(positions).astype(np.float32)

    sequential = [
        np.stack([np.arange(s, s + c - 1), np.arange(s + 1, s + c)], axis=1)
        for s, c in spans
        if c >= 2
    ]
    E_seq = (
        np.concatenate(sequential).astype(np.int64)
        if sequential
        else np.zeros((0, 2), np.int64)
    )

    # Branch links are stored level-wide; keep the records whose BOTH endpoints
    # land in this object. Endpoints are (chunk coords, row within chunk), the
    # same address `row_keys` was built from.
    index_of = {rk: i for i, rk in enumerate(row_keys)}
    branch = []
    for record in read_links(lg, delta=0, include_intra=True):
        if len(record) != 2:
            continue
        a = index_of.get((tuple(record[0][0]), int(record[0][1])))
        if a is None:
            continue
        b = index_of.get((tuple(record[1][0]), int(record[1][1])))
        if b is not None:
            branch.append((a, b))
    E_branch = np.asarray(branch, dtype=np.int64).reshape(-1, 2)

    E = np.concatenate([E_seq, E_branch]) if len(E_branch) else E_seq
    info = {
        "node_ids": (
            np.concatenate(node_ids).astype(np.int64)
            if id_attribute is not None and node_ids
            else None
        ),
        "vertex_id_attribute": id_attribute,
        "object_id": object_id,
        "vertices": int(len(P)),
        "fragments": len(spans),
        "sequential_edges": int(len(E_seq)),
        "branch_edges": int(len(E_branch)),
        "links_convention": meta.links_convention,
        "bounds_of_object": [P.min(0).tolist(), P.max(0).tolist()],
        "coordinate_offset": list(
            read_raw_zv_attrs(store_path).get("coordinate_offset", []) or []
        ),
        "store_bounds": [list(meta.bounds[0]), list(meta.bounds[1])],
        "store_chunk_shape": list(meta.chunk_shape),
    }
    return P, E, info


# ---------------------------------------------------------------------------
# Graph helpers -- deliberately dependency-free (this venv has no scipy)
# ---------------------------------------------------------------------------


def connected_components(num_vertices: int, edges: np.ndarray) -> np.ndarray:
    """Component label per vertex, via union-find."""
    parent = np.arange(num_vertices, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(num_vertices)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def choose_edge(P: np.ndarray, E: np.ndarray, args) -> int:
    """Which edge to break, and why."""
    if args.edge_index is not None:
        if not 0 <= args.edge_index < len(E):
            raise SystemExit(f"--edge-index must be in [0, {len(E)})")
        return args.edge_index
    if args.near is not None:
        target = np.asarray([float(v) for v in args.near.split(",")], dtype=np.float64)
        if target.shape != (3,):
            raise SystemExit("--near wants three comma-separated numbers")
        mid = (P[E[:, 0]] + P[E[:, 1]]) / 2.0
        return int(np.argmin(np.linalg.norm(mid - target, axis=1)))
    # Default: the cut that splits the tree most evenly, which is what makes a
    # demonstration legible. Every edge of a tree is a valid cut, so this is a
    # presentation choice, not a correctness one.
    best_index, best_score = 0, -1
    n = len(P)
    for i in range(len(E)):
        keep = np.delete(E, i, axis=0)
        labels = connected_components(n, keep)
        if labels.max() != 1:
            continue
        smaller = min(int((labels == 0).sum()), int((labels == 1).sum()))
        if smaller > best_score:
            best_index, best_score = i, smaller
        if best_score >= n // 2:
            break
    return best_index


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


# Object ids are written 1-based. Neuroglancer's segmentation layer hides
# segment 0 by default (`hideSegmentZero`, `layer/segmentation/index.ts`), so an
# object written as id 0 loads, resolves, and then renders nothing at all --
# which looks exactly like a broken store.
FIRST_OBJECT_ID = 1


def root_at(adjacency: list[list[int]], members: list[int]) -> int:
    """Root a component at its lowest-indexed leaf, else its lowest vertex."""
    for v in members:
        if len(adjacency[v]) == 1:
            return v
    return members[0]


def child_parent_edges(n: int, edges: np.ndarray, members: np.ndarray):
    """Re-express one component as `[child, parent]` pairs, root omitted.

    `write_skeleton_chunk` wants a rooted tree: the fragment decomposition it
    runs (`decompose_tree_to_paths`) is what makes consecutive vertices in a
    fragment implicitly connected, which is the whole point of the
    `implicit_sequential_with_branches` convention. Handing it an undirected
    edge list instead is what makes a store that loads but draws chords across
    the arbor.
    """
    local_of = {int(g): i for i, g in enumerate(members)}
    adjacency: list[list[int]] = [[] for _ in range(len(members))]
    for a, b in edges:
        la, lb = local_of.get(int(a)), local_of.get(int(b))
        if la is None or lb is None:
            continue
        adjacency[la].append(lb)
        adjacency[lb].append(la)
    for nbrs in adjacency:
        nbrs.sort()
    root = root_at(adjacency, list(range(len(members))))
    parent = np.full(len(members), -1, dtype=np.int64)
    seen = np.zeros(len(members), dtype=bool)
    seen[root] = True
    queue = [root]
    head = 0
    # NOT `for head in range(len(queue))`: Python fixes that range at the
    # initial length, so the walk stops after the root and every other vertex
    # stays parentless -- which decomposes into one fragment per vertex.
    while head < len(queue):
        v = queue[head]
        head += 1
        for w in adjacency[v]:
            if seen[w]:
                continue
            seen[w] = True
            parent[w] = v
            queue.append(w)
    pairs = [(c, int(parent[c])) for c in range(len(members)) if parent[c] >= 0]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def single_chunk_grid(P: np.ndarray):
    """A chunk shape big enough that the whole object lives in ONE cell.

    A proof of concept does not need spatial subdivision, and one cell removes
    the cross-chunk link bookkeeping entirely -- every edge stays intra-chunk.
    The cell must genuinely contain every vertex: neuroglancer derives chunk
    membership from world coordinates (`floor(world / chunk_shape)`), so a
    declared cell that disagrees with that arithmetic renders nothing.
    """
    lo, hi = P.min(0).astype(np.float64), P.max(0).astype(np.float64)
    chunk = float(np.max(hi - lo)) * 2.0 or 1.0
    for _ in range(64):
        if np.array_equal(np.floor(lo / chunk), np.floor(hi / chunk)):
            break
        chunk *= 2.0
    else:
        raise SystemExit("could not find a single chunk containing the object")
    return chunk, tuple(int(c) for c in np.floor(lo / chunk))


def write_store(path: Path, P: np.ndarray, E: np.ndarray, object_ids, info,
                pad=8000.0, node_ids=None):
    """Write positions+edges as a skeleton store, one object per component.

    Uses the skeleton-native builder rather than `write_graph`: the latter
    records nearly every edge as an explicit link while the store still declares
    `implicit_sequential_with_branches`, so a reader that also synthesises the
    implicit edges -- which is what the format says to do, and what both this
    script and the neuroglancer fork do -- sees each edge about twice, plus
    chords between vertices that are merely adjacent in storage order. Measured
    on this axon: 1074 edges in, 1992 edges back, 105 of them spurious, the
    longest 53 um against a true median of 1.8 um.
    """
    shutil.rmtree(path, ignore_errors=True)
    object_ids = np.asarray(object_ids, dtype=np.int64)
    lower = (P.min(0) - pad).tolist()
    upper = (P.max(0) + pad).tolist()
    chunk, cc = single_chunk_grid(P)

    root, lg = init_skeleton_store(
        str(path),
        chunk_shape=(chunk, chunk, chunk),
        bounds=(lower, upper),
        ndim=3,
        attribute_dtypes={NODE_ID_ATTRIBUTE: "uint32"},
    )

    # A stable per-vertex id, so the viewer can pick a NODE rather than just the
    # object it belongs to. Ids are 1-based and unique across the whole store:
    # the edit overlay keys one map by node id across every segment it holds
    # (src/skeleton/segment_overlay.ts), so per-object numbering would collide.
    # `node_id` here is simply the vertex's index in the source array + 1, which
    # keeps it stable for as long as the extraction is stable.
    ids = (
        np.asarray(node_ids, dtype=np.uint32)
        if node_ids is not None
        else (np.arange(len(P)) + 1).astype(np.uint32)
    )
    pieces = []
    for oid in sorted({int(v) for v in object_ids}):
        members = np.nonzero(object_ids == oid)[0]
        pieces.append(
            {
                "segment_id": int(oid),
                "positions": P[members].astype(np.float32),
                "edges": child_parent_edges(len(P), E, members),
                "attributes": {NODE_ID_ATTRIBUTE: ids[members]},
            }
        )
    records, _ = write_skeleton_chunk(lg, cc, pieces)

    manifests: dict[int, list] = {}
    for segment_id, chunk_coords, fragment_index in records:
        manifests.setdefault(int(segment_id), []).append(
            (tuple(chunk_coords), int(fragment_index))
        )
    write_object_index(lg, manifests, 3, total_objects=max(manifests) + 1)
    finalize_skeleton_store(root)
    stamp_axis_units(path)
    stamp_vertex_id_attribute(path)
    return {
        "node_count": int(len(P)),
        "edge_count": int(len(E)),
        "objects": len(pieces),
        "fragments": len(records),
        "chunk_shape": chunk,
        "chunk_coords": cc,
    }


def stamp_axis_units(path: Path, unit: str = "nanometer") -> None:
    """Declare the axis unit on a freshly written store.

    `write_graph` writes NGFF axes with no `unit`, which leaves the store
    dimensionless. Neuroglancer then reconciles the layer against a viewer whose
    global dimensions ARE nanometres, and the geometry lands at ~1e14 nm --
    about a billion times further out than the camera, so the layer loads,
    reports correct bounds, and draws nothing at all. The source store declares
    "nanometer"; matching it is what makes the output viewable next to it.
    """
    meta_path = path / "zarr.json"
    meta = json.loads(meta_path.read_text())
    multiscales = meta.get("attributes", {}).get("multiscales") or []
    changed = False
    for entry in multiscales:
        for axis in entry.get("axes", []):
            if axis.get("type") == "space" and "unit" not in axis:
                axis["unit"] = unit
                changed = True
    if changed:
        meta_path.write_text(json.dumps(meta, indent=2))


NODE_ID_ATTRIBUTE = "node_id"


def stamp_vertex_id_attribute(path: Path, name: str = NODE_ID_ATTRIBUTE) -> None:
    """Declare which per-vertex column carries node identity.

    The fork reads `zarr_vectors.vertex_id_attribute` from the root attrs and
    decodes that column from raw bytes rather than the float32 render column,
    because the values become picking ids and must be exact
    (`resolveVertexIdColumn`, geometry_chunk_download.ts). The key is a reader
    convention -- it is absent from the ZVF schema, and `RootMetadata.from_dict`
    ignores unknown keys, so writing it costs nothing elsewhere.
    """
    meta_path = path / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("attributes", {}).setdefault("zarr_vectors", {})[
        "vertex_id_attribute"
    ] = name
    meta_path.write_text(json.dumps(meta, indent=2))


def read_back(path: Path) -> dict:
    """Reopen a written store and report what each object holds."""
    root = open_store(str(path))
    meta = read_root_metadata(root)
    lg = get_resolution_level(root, 0)
    out = {"geometry_types": list(meta.geometry_types),
           "links_convention": meta.links_convention,
           "objects": {}}
    for oid in range(0, 8):
        try:
            manifest = read_object_manifest(lg, oid)
        except Exception:
            break
        if not manifest:
            continue
        count = 0
        for cc, fi in manifest:
            count += len(np.asarray(read_fragment(lg, tuple(cc), fi, ndim=3)))
        out["objects"][oid] = {"fragments": len(manifest), "vertices": count}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default="/hdd/ZV_MICRONS_Conversion/minnie65_v1822/skeletons.zv")
    ap.add_argument("--object", type=int, default=281, help="dense object index")
    ap.add_argument("--out", default="/tmp/zv_poc")
    ap.add_argument("--edge-index", type=int, default=None, help="break this edge")
    ap.add_argument("--near", default=None, help="break the edge nearest X,Y,Z")
    ap.add_argument(
        "--absolute",
        action="store_true",
        help="add the store's coordinate_offset so the output sits in the same "
             "absolute nanometre frame as the sibling synapse/mesh stores",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    P, E, info = extract_object(args.store, args.object)
    info["extract_seconds"] = round(time.time() - t0, 2)

    labels = connected_components(len(P), E)
    info["components_before_cut"] = int(labels.max() + 1)
    if info["components_before_cut"] != 1:
        print(
            f"! object {args.object} is not connected "
            f"({info['components_before_cut']} components) -- cutting one edge "
            "will not produce exactly two objects"
        )

    if args.absolute:
        if not info["coordinate_offset"]:
            raise SystemExit(
                "--absolute was given but the source store declares no "
                "coordinate_offset, so there is nothing to apply"
            )
        P = P + np.asarray(info["coordinate_offset"], dtype=np.float32)
        info["applied_coordinate_offset"] = info["coordinate_offset"]
    else:
        info["applied_coordinate_offset"] = None

    print(json.dumps({k: v for k, v in info.items() if k != "node_ids"}, indent=2))

    single = out / "single_axon.zv"
    s1 = write_store(
        single, P, E, np.full(len(P), FIRST_OBJECT_ID, np.int64), info
    )
    print(f"\nwrote {single}: {s1.get('node_count', len(P))} nodes, "
          f"{s1.get('edge_count', len(E))} edges, 1 object")

    edge_index = choose_edge(P, E, args)
    a, b = int(E[edge_index, 0]), int(E[edge_index, 1])
    length = float(np.linalg.norm(P[a] - P[b]))
    E_cut = np.delete(E, edge_index, axis=0)
    labels = connected_components(len(P), E_cut)
    n_components = int(labels.max() + 1)
    sizes = [int((labels == k).sum()) for k in range(n_components)]

    print(f"\ncut edge #{edge_index}: vertex {a} -> {b}, length {length:.0f} nm")
    print(f"  midpoint {(P[a] + P[b]) / 2}")
    print(f"  components after cut: {n_components}  sizes {sizes}")
    if n_components != 2:
        raise SystemExit(
            f"expected 2 components after the cut, got {n_components} -- the "
            "chosen edge was not a bridge, so the object was not a tree there"
        )

    split = out / "split_axon.zv"
    s2 = write_store(split, P, E_cut, labels + FIRST_OBJECT_ID, info)
    ids = [k + FIRST_OBJECT_ID for k in range(n_components)]
    print(f"wrote {split}: {s2['node_count']} nodes, {s2['edge_count']} edges, "
          f"{s2['objects']} objects (ids {ids}), {s2['fragments']} fragments")

    print("\n--- read back ---")
    for path in (single, split):
        print(f"{path.name}: {json.dumps(read_back(path))}")

    info.pop("node_ids", None)
    report = {
        "source_store": args.store,
        "source_object": args.object,
        "cut": {"edge_index": edge_index, "vertex_a": a, "vertex_b": b,
                "length_nm": length,
                "midpoint": ((P[a] + P[b]) / 2).tolist(),
                "component_sizes": sizes,
                "object_ids": [k + FIRST_OBJECT_ID for k in range(len(sizes))]},
        **info,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
