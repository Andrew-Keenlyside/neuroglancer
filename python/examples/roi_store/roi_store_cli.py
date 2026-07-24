#!/usr/bin/env python3
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

"""Local push/pull/list/delete for the shared ROI-group store.

Why this exists
---------------
The in-browser "Save to store" writes to the bucket with a cross-origin POST,
which the browser only allows if the bucket has a write-CORS rule for the app
origin.  When you can *upload/delete objects* but cannot *edit bucket CORS or
config*, that in-browser save is blocked -- but a plain command-line upload is
not, because CORS is a browser rule and has nothing to do with `gcloud`.

So the workflow splits cleanly:

  1. In Neuroglancer: run a coarse ROI selection to build your groups, then copy
     the viewer's JSON state (the ``{}`` button in the top toolbar) to a file.
  2. Here: `push` lifts each group out of that state, wraps it in the exact store
     document the browser would have written, and uploads it with `gcloud
     storage` using your existing `gcloud` login.  BROWSE/LOAD in Neuroglancer
     then reads it back anonymously -- no CORS, no bucket-config change.

This mirrors, and must stay in step with, ``src/roi_store/schema.ts`` and
``src/roi_store/gcs_client.ts``:

  * documents live at ``groups/<id>.json`` (ROI_GROUP_PREFIX + id + ".json");
  * ``id`` is a 128-bit hex string (getRandomHexString(128) -> 32 hex chars);
  * the browse list reads ``roiGroupName`` / ``createdBy`` / ``sourceUrl`` from
    each object's *custom metadata* (roiGroupCustomMetadata), falling back to the
    id for the name when absent -- so metadata is nice-to-have, not required for
    a load to work;
  * a group's JSON in the layer state (``layers[i].roiFilter.groups[j]``) is
    byte-identical to a document's ``group`` field (both are groupToJson output),
    which is what lets this tool avoid touching Neuroglancer at all.

Auth
----
Uses the gcloud CLI's own credentials -- whatever ``gcloud auth login`` already
set up.  No Application Default Credentials and no service-account key needed.
Permissions required: object create (push), get+list (list/pull), delete (rm).
No bucket-level (IAM/CORS) permission is used.

Usage
-----
    # Push every ROI group found in a saved viewer state:
    python roi_store_cli.py push --state state.json

    # Push only one layer's groups, attributing them to you:
    python roi_store_cli.py push --state state.json \
        --layer tracts --created-by you@example.com

    python roi_store_cli.py list
    python roi_store_cli.py pull --id <id> --out group.json
    python roi_store_cli.py rm   --id <id>

``--bucket`` defaults to the demo bucket; override it for your own store.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep in sync with src/roi_store/schema.ts.
ROI_GROUP_PREFIX = "groups/"
ROI_GROUP_SCHEMA_VERSION = 1
# Keep in sync with the ROI_STORE define in rspack.config.ts / build_pyodide.ts.
DEFAULT_BUCKET = "hip_ct_zarr_vector_03987646472fethdsvdvdfg"


def _now_iso() -> str:
    """An ISO-8601 UTC timestamp in the ``...Z`` shape Date.toISOString() uses."""
    return (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )


def _new_id() -> str:
    """128 random bits as 32 hex chars, matching getRandomHexString(128)."""
    return secrets.token_hex(16)


def _run_gcloud(args: list[str], *, capture: bool = False) -> str:
    """Run a `gcloud` subcommand, surfacing a clean error if it is missing."""
    try:
        result = subprocess.run(
            ["gcloud", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError:
        sys.exit(
            "error: `gcloud` not found on PATH. Install the Google Cloud SDK "
            "and run `gcloud auth login`."
        )
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip()
        sys.exit(f"error: gcloud {' '.join(args)} failed:\n{detail}")
    return result.stdout if capture else ""


def _gcloud_account() -> str | None:
    """The email `gcloud` is logged in as, for default provenance."""
    out = _run_gcloud(
        ["config", "get-value", "account"], capture=True
    ).strip()
    # `gcloud` prints "(unset)" to stdout when no account is configured.
    return out if out and out != "(unset)" else None


def _source_url(source: Any) -> str | None:
    """Extract a data-source URL from a layer's ``source`` (str | list | dict).

    Neuroglancer serialises a layer source as a bare URL string, an object with
    a ``url`` field, or a list of either (the first is the primary data source).
    The store only records one URL, so take the first resolvable one.
    """
    if isinstance(source, str):
        return source
    if isinstance(source, dict):
        url = source.get("url")
        return url if isinstance(url, str) else None
    if isinstance(source, list):
        for entry in source:
            url = _source_url(entry)
            if url is not None:
                return url
    return None


def _iter_layers(state: dict) -> list[tuple[str, Any, Any]]:
    """Yield ``(name, source, roiFilter)`` for every layer carrying an roiFilter.

    Handles both layer encodings: the modern list (each layer a dict with a
    ``name``) and the legacy name-keyed object.
    """
    layers = state.get("layers")
    out: list[tuple[str, Any, Any]] = []
    if isinstance(layers, list):
        items = [
            (layer.get("name"), layer)
            for layer in layers
            if isinstance(layer, dict)
        ]
    elif isinstance(layers, dict):
        items = [
            (name, layer)
            for name, layer in layers.items()
            if isinstance(layer, dict)
        ]
    else:
        return out
    for name, layer in items:
        roi_filter = layer.get("roiFilter")
        if isinstance(roi_filter, dict) and roi_filter.get("groups"):
            out.append((name, layer.get("source"), roi_filter))
    return out


def _load_state(path: str) -> dict:
    """Read the viewer-state JSON from a file or ``-`` for stdin."""
    raw = sys.stdin.read() if path == "-" else Path(path).read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"error: {path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        sys.exit("error: expected a viewer-state object at the top level.")
    return data


def _make_document(
    group: dict,
    *,
    source_url: str | None,
    layer_name: str | None,
    viewer_url: str | None,
    created_by: str | None,
) -> dict:
    """Wrap a groupToJson object in an RoiGroupDocument (parseRoiGroupDocument)."""
    now = _now_iso()
    doc: dict[str, Any] = {
        "schemaVersion": ROI_GROUP_SCHEMA_VERSION,
        "id": _new_id(),
        "group": group,
        # `source.url` is required by the schema; the loader shows a mismatch
        # warning if it does not line up with the layer it is dropped onto.
        "source": {"url": source_url or ""},
        "createdAt": now,
        "updatedAt": now,
    }
    scene: dict[str, str] = {}
    if viewer_url:
        scene["url"] = viewer_url
    if layer_name:
        scene["layerName"] = layer_name
    if scene:
        doc["scene"] = scene
    if created_by:
        doc["createdBy"] = created_by
    return doc


def _custom_metadata_arg(doc: dict) -> list[str]:
    """Build the ``--custom-metadata`` flag mirroring roiGroupCustomMetadata().

    gcloud packs the whole dict into one comma-delimited argument, so a value
    containing the delimiter would corrupt it -- and group names and zarr-vectors
    URLs (``...|zarr-vectors:``) routinely contain commas and pipes.  gcloud's
    ``^DELIM^`` escape lets us pick a delimiter absent from every value; if none
    of the candidates is safe (astronomically unlikely) we omit the metadata
    rather than write a corrupt object -- the group still loads, it just lists
    under its id until re-described.
    """
    meta: dict[str, str] = {}
    name = doc.get("group", {}).get("name")
    if isinstance(name, str):
        meta["roiGroupName"] = name
    if doc.get("createdBy"):
        meta["createdBy"] = doc["createdBy"]
    source_url = doc.get("source", {}).get("url")
    if source_url:
        meta["sourceUrl"] = source_url
    if not meta:
        return []
    pairs = [f"{k}={v}" for k, v in meta.items()]
    blob = "".join(pairs)
    for delim in (",", ";", "#", "~", "!", "@"):
        if delim not in blob:
            joined = delim.join(pairs)
            # A leading "^DELIM^" tells gcloud to split on DELIM, not commas.
            value = joined if delim == "," else f"^{delim}^{joined}"
            return [f"--custom-metadata={value}"]
    return []


def cmd_push(args: argparse.Namespace) -> None:
    state = _load_state(args.state)
    layers = _iter_layers(state)
    if args.layer is not None:
        layers = [entry for entry in layers if entry[0] == args.layer]
        if not layers:
            sys.exit(f"error: no layer named {args.layer!r} carries ROI groups.")
    if not layers:
        sys.exit(
            "error: no ROI groups found in that state. Draw at least one group "
            "in the Filter tab, then copy the viewer JSON ({} button)."
        )

    created_by = args.created_by or _gcloud_account()
    if created_by is None:
        print(
            "note: no --created-by and `gcloud` has no active account; "
            "documents will record no author.",
            file=sys.stderr,
        )

    pushed = 0
    for layer_name, source, roi_filter in layers:
        source_url = args.source_url or _source_url(source)
        if source_url is None:
            print(
                f"warning: layer {layer_name!r} has no resolvable source URL; "
                "writing an empty source.url (load will warn about a mismatch).",
                file=sys.stderr,
            )
        for group in roi_filter["groups"]:
            if not isinstance(group, dict) or "name" not in group:
                continue
            doc = _make_document(
                group,
                source_url=source_url,
                layer_name=layer_name,
                viewer_url=args.viewer_url,
                created_by=created_by,
            )
            name = f"{ROI_GROUP_PREFIX}{doc['id']}.json"
            dest = f"gs://{args.bucket}/{name}"
            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False
            ) as fh:
                json.dump(doc, fh)
                tmp_path = fh.name
            try:
                _run_gcloud(
                    [
                        "storage",
                        "cp",
                        "--content-type=application/json",
                        *_custom_metadata_arg(doc),
                        tmp_path,
                        dest,
                    ]
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            print(f"pushed {group['name']!r}  ->  {dest}")
            pushed += 1
    print(f"done: {pushed} group(s) uploaded to gs://{args.bucket}/{ROI_GROUP_PREFIX}")


def cmd_list(args: argparse.Namespace) -> None:
    # `ls` needs storage.objects.list; a bucket that withholds it anonymously
    # still lists here because gcloud presents your credentials.
    out = _run_gcloud(
        ["storage", "ls", f"gs://{args.bucket}/{ROI_GROUP_PREFIX}"],
        capture=True,
    )
    ids = [
        line.rsplit("/", 1)[-1][: -len(".json")]
        for line in out.splitlines()
        if line.endswith(".json")
    ]
    if not ids:
        print(f"(no groups under gs://{args.bucket}/{ROI_GROUP_PREFIX})")
        return
    for group_id in ids:
        print(group_id)
    print(f"\n{len(ids)} group(s). Names/authors show in the browser's browse list.")


def cmd_pull(args: argparse.Namespace) -> None:
    src = f"gs://{args.bucket}/{ROI_GROUP_PREFIX}{args.id}.json"
    dest = args.out or f"{args.id}.json"
    _run_gcloud(["storage", "cp", src, dest])
    print(f"pulled {src}  ->  {dest}")


def cmd_rm(args: argparse.Namespace) -> None:
    target = f"gs://{args.bucket}/{ROI_GROUP_PREFIX}{args.id}.json"
    _run_gcloud(["storage", "rm", target])
    print(f"deleted {target}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"GCS bucket holding the store (default: {DEFAULT_BUCKET}).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_push = sub.add_parser("push", help="upload ROI groups from a viewer state")
    p_push.add_argument(
        "--state",
        required=True,
        help="viewer JSON state file (the {} editor), or - for stdin.",
    )
    p_push.add_argument(
        "--layer", help="only push groups from this layer (default: all layers)."
    )
    p_push.add_argument(
        "--source-url",
        help="override the recorded source URL (default: the layer's source).",
    )
    p_push.add_argument(
        "--viewer-url",
        help="viewer URL to record in scene.url, so the group reopens in context.",
    )
    p_push.add_argument(
        "--created-by",
        help="author email for provenance (default: the active gcloud account).",
    )
    p_push.set_defaults(func=cmd_push)

    p_list = sub.add_parser("list", help="list group ids in the store")
    p_list.set_defaults(func=cmd_list)

    p_pull = sub.add_parser("pull", help="download one group document")
    p_pull.add_argument("--id", required=True, help="group id (object name stem).")
    p_pull.add_argument("--out", help="output path (default: <id>.json).")
    p_pull.set_defaults(func=cmd_pull)

    p_rm = sub.add_parser("rm", help="delete one group document")
    p_rm.add_argument("--id", required=True, help="group id (object name stem).")
    p_rm.set_defaults(func=cmd_rm)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
