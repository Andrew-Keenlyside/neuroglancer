# zarr-vectors-tools: make the streamline export reachable from Pyodide

Handoff prompt for work in `zarr-vectors-tools`. Written from an investigation
in the neuroglancer fork on 2026-07-19.

Read `zarr-vectors-py.md` alongside this — the core library change is the
larger piece, and this one is small by comparison.

---

## Why

The neuroglancer fork's Export tab describes an ROI dissection as a **job spec**
and hands it to a native process to execute
(`python/neuroglancer/tract_export/`), because this package "needs `zarr>=3`,
`numcodecs` and `nibabel` — none of which import under Pyodide". Two of those
three claims are now false, and the third is only half true, so it is worth
re-checking how much of the export could run in the browser.

## What is actually true

- **`zarr>=3` is fine.** Pyodide 314.0.2 ships a WASM-patched zarr 3.2.1 as a
  first-party package.
- **`numcodecs` is fine.** It is in the Pyodide distribution (0.15.1).
- **`nibabel` is fine.** Pure Python, installable by `micropip`.
- **`trx-python` is NOT fine.** No pyemscripten wheel. Same for `DracoPy`
  (optional in core) and `pyfqmr` (the `mesh` extra).
- **`pandas` is fine** — in the Pyodide distribution.

So the only genuine blocker in the `[streamlines]` extra is `trx-python`, and
it is bundled with `nibabel` in one extra.

## Change 1 — split the `streamlines` extra

Today:

```toml
streamlines = ["nibabel>=5.0", "trx-python>=0.3"]
```

Anyone wanting TRK export has to take `trx-python` too, which makes the whole
extra uninstallable under Pyodide even though TRK itself only needs `nibabel`.

Split it so the installable half can be installed alone:

```toml
trk = ["nibabel>=5.0"]
trx = ["trx-python>=0.3"]
streamlines = ["zarr-vectors-tools[trk,trx]"]   # keep as an alias, unchanged for existing users
```

Then make the TRX import lazy and function-local (the pattern core already uses
for `obstore`), so importing the TRK exporter never transitively pulls
`trx-python`. Verify with a clean env: `pip install zarr-vectors-tools[trk]`
followed by `from zarr_vectors_tools.export.trk import export_trk` must succeed
with `trx-python` absent.

The fork already imports exactly that symbol —
`python/neuroglancer/tract_export/run.py:216` — and its error message tells the
user to `pip install 'zarr-vectors-tools[streamlines]'`. Once split, that hint
should become `[trk]`.

> **Status 2026-07-19: Change 1 is DONE.** `pyproject.toml` now has separate
> `trk` / `trx` extras with `streamlines` kept as a back-compat alias resolving
> to both. No code change was needed — `export/__init__.py` is empty, and
> `trx.py:48` / `trk.py:50` already import function-locally. Verified by
> blocking `trx` at `sys.meta_path`: `export_trk` imported clean, `trx` never
> entered `sys.modules`, and 4 streamlines / 28 vertices round-tripped.
> The fork's hint at `run.py:216` has been updated to `[trk]`.
> Caveat recorded: `[trk]` also covers TCK/TRK **ingest**, so the extra is
> slightly broader than its name suggests.

## Change 2 — audit the write path for threads

`zarr-vectors-py`'s batch writer spawns a `ThreadPoolExecutor`
(`core/_batch_writer.py:198`), and `threading.Thread.start()` raises
`RuntimeError: can't start new thread` under Pyodide. If this package's write
path goes through that, browser-side export is blocked until the core fix lands.

Please confirm whether the export/write path:

1. uses zarr's **sync** API (which under Pyodide needs JSPI — see the deadlock
   warning in the core prompt), and
2. spawns threads anywhere of its own.

Report what you find even if the answer is "yes to both" — that is a legitimate
result and tells the fork to keep the native-job design.

> **Status 2026-07-19: audited — yes to both, on the READ path.** Nothing in
> `zarr_vectors_tools/export/` spawns threads; every pool is ingest-side behind
> an explicit `workers` argument. But a cold `export_trk` in a fresh CPython
> process goes from `['MainThread']` to
> `['MainThread', 'zarr_io', 'asyncio_0', 'asyncio_1']`, because
> `read_polylines` → `batched_reads` → `sync(_gather_plan(...))` at core's
> `_batch_reader.py:175` lazily starts zarr's global `zarr_io` daemon thread and
> fans out to `async.concurrency=10`. No knob in the export API suppresses it.
> This is the **read** path, so it is independent of `_batch_writer.py:198`.
>
> **Caveat on extrapolating this to Pyodide:** that audit was CPython. Pyodide's
> zarr is WASM-patched to run `sync()` on a WebLoop rather than a thread, so
> `zarr_io` should not appear there and the predicted
> `RuntimeError: can't start new thread` is probably not the actual failure
> mode. The concurrency fan-out is the more likely problem — see the
> `_sync_fallback` lever now written up in `zarr-vectors-py.md`, which is a core
> change, not one for this package.

## What is NOT being asked

**Do not port the whole export pipeline to the browser.** Exporting
re-evaluates the dissection at pyramid level 0 across the full store, which is
the right work for a native process regardless of whether it _could_ run in
WASM. The fork's job-spec split is likely correct on its merits.

The narrow goal is: make the **TRK** path installable and importable under
Pyodide, so a browser-side TRK export becomes _possible_ if wanted, and so the
dependency story stops being a blanket "none of this works in the browser" when
only one optional package is at fault.

## How to verify

1. Clean env, `pip install zarr-vectors-tools[trk]`, import `export_trk`,
   round-trip a small TRK. Must not require `trx-python`.
2. Existing `[streamlines]` installs keep working unchanged.
3. Under Pyodide headlessly: `micropip.install("nibabel")`, put this package on
   `sys.path`, import the TRK exporter. Report where it fails if it does — a
   precise failure is a useful result.

## Reference

- Fork's consumer: `python/neuroglancer/tract_export/run.py:216`
- Fork's rationale, now partly stale:
  `python/neuroglancer/tract_export/__init__.py:17-21`
- Dependency lists: `zarr-vectors-tools/pyproject.toml`
