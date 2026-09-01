#!/usr/bin/env python
"""A loopback edit service for ZVF skeleton stores.

One operation: split an object at a node. The viewer's Split tool posts the
picked `(segmentId, nodeId)`; this rewrites the store so the node's subtree
becomes an object of its own, and answers with the two resulting object ids.

Whole-store rewrite, deliberately. The in-place path
(`EditSession.remove_link`) re-attributes objects correctly but leaves the
store declaring `links_convention="explicit"` with only the branch links
materialised, so a conforming reader draws a fraction of the edges (measured:
34 of 1073). For a single-axon store the rewrite is well under a second and the
result is exactly what the reader expects.

Bound to 127.0.0.1 and gated on a token printed at startup — the same shape as
`python/neuroglancer/tract_export/serve.py`, and for the same reason: it writes
to local disk on request.

    python edit_service.py --root /tmp/zv_poc --port 9099
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_skeleton import (  # noqa: E402
    connected_components,
    extract_object,
    read_raw_zv_attrs,
    write_store,
)
from zarr_vectors.core.arrays import read_object_manifest  # noqa: E402
from zarr_vectors.core.store import get_resolution_level, open_store  # noqa: E402

LOCK = threading.Lock()


def present_object_ids(store_path: str, limit: int = 4096) -> list[int]:
    root = open_store(store_path)
    lg = get_resolution_level(root, 0)
    out = []
    for oid in range(limit):
        try:
            manifest = read_object_manifest(lg, oid)
        except Exception:
            break
        if manifest:
            out.append(oid)
    return out


def rooted_parent(P: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Parent index per vertex under the same rooting rule used when writing."""
    adjacency: list[list[int]] = [[] for _ in range(len(P))]
    for a, b in E:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    for nbrs in adjacency:
        nbrs.sort()
    root = next((v for v in range(len(P)) if len(adjacency[v]) == 1), 0)
    parent = np.full(len(P), -1, dtype=np.int64)
    seen = np.zeros(len(P), dtype=bool)
    seen[root] = True
    queue = [root]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        for w in adjacency[v]:
            if not seen[w]:
                seen[w] = True
                parent[w] = v
                queue.append(w)
    return parent


def split_at_node(store_path: str, segment_id: int, node_id: int) -> dict:
    """Detach `node_id`'s subtree from its parent, as a new object."""
    objects = {}
    for oid in present_object_ids(store_path):
        P, E, info = extract_object(store_path, oid)
        objects[oid] = (P, E, info)
    if segment_id not in objects:
        raise ValueError(f"object {segment_id} is not in {store_path}")

    P, E, info = objects[segment_id]
    ids = info.get("node_ids")
    if ids is None:
        raise ValueError(
            "this store declares no vertex_id_attribute, so a node cannot be "
            "addressed by id"
        )
    match = np.nonzero(ids == int(node_id))[0]
    if len(match) == 0:
        raise ValueError(f"node {node_id} is not in object {segment_id}")
    v = int(match[0])

    parent = rooted_parent(P, E)
    if parent[v] < 0:
        raise ValueError(
            f"node {node_id} is the root of object {segment_id}; there is no "
            "edge above it to cut"
        )
    keep = [
        i
        for i, (a, b) in enumerate(E)
        if not ({int(a), int(b)} == {v, int(parent[v])})
    ]
    if len(keep) == len(E):
        raise ValueError("internal: the parent edge was not found in the edge list")
    E_cut = E[keep]
    labels = connected_components(len(P), E_cut)
    if labels.max() != 1:
        raise ValueError(
            f"cutting above node {node_id} produced {labels.max() + 1} pieces, "
            "expected 2"
        )

    # Renumber: every other object keeps its id; this one becomes two, taking
    # its own id plus the next free one, so the viewer only has to learn one id.
    free = max(objects) + 1
    new_ids = [segment_id, free]

    all_P, all_E, all_obj, all_node = [], [], [], []
    base = 0
    for oid, (Q, F, meta) in sorted(objects.items()):
        if oid == segment_id:
            continue
        all_P.append(Q)
        all_E.append(F + base)
        all_obj.append(np.full(len(Q), oid, dtype=np.int64))
        all_node.append(meta["node_ids"])
        base += len(Q)
    all_P.append(P)
    all_E.append(E_cut + base)
    all_obj.append(np.where(labels == 0, new_ids[0], new_ids[1]).astype(np.int64))
    all_node.append(ids)

    P_all = np.concatenate(all_P)
    E_all = np.concatenate([e for e in all_E if len(e)]) if any(len(e) for e in all_E) else np.zeros((0, 2), np.int64)
    obj_all = np.concatenate(all_obj)
    node_all = np.concatenate(all_node)

    write_store(
        Path(store_path), P_all, E_all, obj_all, info, node_ids=node_all
    )
    sizes = [int((obj_all == i).sum()) for i in new_ids]
    return {
        "ids": new_ids,
        "sizes": sizes,
        "cutNode": int(node_id),
        "parentNode": int(ids[int(parent[v])]),
        "objects": sorted({int(x) for x in obj_all}),
    }


class Handler(BaseHTTPRequestHandler):
    root_dir = Path("/tmp/zv_poc")
    token = ""

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802 - http.server naming
        self._send(200, {})

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/split"):
            self._send(404, {"error": "only /split is implemented"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"bad JSON: {e}"})
            return
        if self.token and body.get("token") != self.token:
            self._send(403, {"error": "bad or missing token"})
            return
        store = body.get("store")
        if not store:
            self._send(400, {"error": "store is required"})
            return
        path = (self.root_dir / store).resolve()
        if self.root_dir.resolve() not in path.parents and path != self.root_dir.resolve():
            self._send(400, {"error": "store must be inside the served root"})
            return
        try:
            with LOCK:
                result = split_at_node(
                    str(path), int(body["segmentId"]), int(body["nodeId"])
                )
        except Exception as e:  # surfaced verbatim in the viewer's status line
            self._send(400, {"error": f"{type(e).__name__}: {e}"})
            return
        self._send(200, result)

    def log_message(self, fmt, *args):
        sys.stderr.write("edit_service: " + (fmt % args) + "\n")
        sys.stderr.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/tmp/zv_poc")
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    Handler.root_dir = Path(args.root)
    Handler.token = args.token if args.token is not None else secrets.token_hex(8)
    print(f"edit service on http://127.0.0.1:{args.port}  root={args.root}")
    print(f"token={Handler.token}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
