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
"""``python -m neuroglancer.tract_export <job.json>``.

The counterpart to the HTTP route: when the Export tab cannot reach a service it
offers the job spec as a download instead, and this runs that exact file. Both
paths go through the same :func:`parse_job` / :func:`run_job`, so a downloaded
spec and a posted body cannot drift.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from neuroglancer.tract_export.job import JobSpecError, parse_job
from neuroglancer.tract_export.run import ExportRunError, run_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m neuroglancer.tract_export",
        description="Export an ROI dissection to TRK or a new zarr-vectors store.",
    )
    parser.add_argument(
        "job",
        nargs="?",
        help="Path to a job spec written by the viewer's Export tab, or - for stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Override destination.path from the spec.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate the dissection and report the counts without writing.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Instead of running one job, stay up as a local exporter that a "
            "statically served viewer (browser or Pyodide build) can call."
        ),
    )
    parser.add_argument("--port", type=int, default=None, help="Port for --serve.")
    parser.add_argument(
        "--bind",
        default=None,
        help="Bind address for --serve (default loopback).",
    )
    parser.add_argument(
        "--allow-origin",
        default=None,
        help="Origin allowed to call the exporter, for --serve.",
    )
    args = parser.parse_args(argv)

    if args.serve:
        from neuroglancer.tract_export import serve as serve_mod

        # Forward every serve-relevant flag; previously only --port reached the
        # sub-command, so --bind and --allow-origin were silently ignored.
        serve_argv: list[str] = []
        if args.port is not None:
            serve_argv += ["--port", str(args.port)]
        if args.bind is not None:
            serve_argv += ["--bind", args.bind]
        if args.allow_origin is not None:
            serve_argv += ["--allow-origin", args.allow_origin]
        return serve_mod.main(serve_argv)
    if args.job is None:
        parser.error("a job spec is required unless --serve is given")

    if args.job == "-":
        raw = sys.stdin.read()
    else:
        with open(args.job, encoding="utf-8") as f:
            raw = f.read()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"error: {args.job} is not valid JSON: {e}", file=sys.stderr)
        return 2

    try:
        job = parse_job(spec)
    except JobSpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.output is not None:
        job = dataclasses.replace(
            job, destination=dataclasses.replace(job.destination, path=args.output)
        )

    try:
        if args.dry_run:
            from neuroglancer.tract_export.run import select, union_ids

            # Same keys a real run emits, so a caller can read one summary
            # shape; only the write-dependent fields differ.
            considered, per_group = select(job)
            ids = union_ids(per_group)
            summary = {
                "dryRun": True,
                "written": False,
                "streamline_count": int(ids.size),
                "object_count": int(ids.size),
                "candidate_object_count": considered,
                "output_path": job.destination.path,
                "groups": [
                    {"name": name, "count": int(v.size)} for name, v in per_group
                ],
            }
        else:
            summary = run_job(job)
    except ExportRunError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
