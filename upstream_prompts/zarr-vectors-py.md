# zarr-vectors-py: an async read API for Pyodide

Handoff prompt for work in `zarr-vectors-py` (PyPI name `zarr-vectors`).
Written from an investigation in the neuroglancer fork on 2026-07-19; every
file:line below was verified against the working tree at that date.

---

## Why

The neuroglancer fork ships a Pyodide (browser/WASM) deployment that renders
zarr-vectors tractography. It currently cannot call this library, so it carries
a 380-line hand-rolled reimplementation of the ZVF 0.9 read path at
`python/neuroglancer/tractography/zarr_vectors_source.py`.

That reimplementation is a liability:

- Its fragment-index decode is **byte-for-byte identical** to
  `zarr_vectors/encoding/fragments.py` (same magic `0x5A564647`, same
  `<IHHII>` header, same bitmap padding, same CSR layout). Two copies of one
  binary format, in two languages, with no shared fixture pinning them.
- It is a strict **subset**: it rejects any compressor, rejects sharding, and
  brute-forces the whole chunk grid absorbing 404s where this library reads the
  `nonempty_chunks` manifest.
- It has **zero callers** and says so in its own docstring.

The goal is to delete it and use this library directly, so the format lives in
exactly one place.

## What is NOT the problem

Two blockers the fork's docstring asserts turned out to be false, so don't
design around them:

- **obstore / fsspec / aiohttp are not blockers.** They are optional extras
  here (`pyproject.toml` `[project.optional-dependencies]`), imported
  function-locally inside `try/except` (`core/store.py:99-113`, `:188-192`),
  and reached only when a caller passes a URL _string_. An import census over
  all 73 `.py` files found zero `aiohttp` and zero `fsspec` imports. Pyodide
  314.0.2 ships both anyway.
- **Compression is not a blocker.** `asyncio.to_thread` works under Pyodide
  (it runs inline), so zstd/blosc/gzip codecs decode fine.

## The actual problem

Every public read entry point is synchronous, and under Pyodide a synchronous
zarr call can only work via WebAssembly stack switching (JSPI), invoked from JS
through a _promising_ entrypoint. Two concurrent promising entries into that
path **wedge the Pyodide worker permanently** — see the reproduction below for
the measured thresholds. Small stores pass, so a smoke test goes green and
production hangs.

So the sync+JSPI route is closed for the fork's normal operation, and the
library needs a path that never blocks on `sync()`.

### Reproduced independently, 2026-07-19

The deadlock was confirmed from scratch by a second harness (headless Chromium
141, pyodide 314.0.2, real zarr-vectors 0.2.1.dev66), deterministic across 8
process invocations. Three corrections to the first report:

- **The threshold is lower.** Not "~1700 keys" — the hang appears between
  **~530 and ~730 store keys per reader at concurrency 2**. Small stores (102
  keys) genuinely interleave and both complete, so concurrency per se is fine;
  scale is what kills it. Concurrent reads of the **same** store hang too, so it
  is not a two-store artifact.
- **The blast radius is narrower.** It does not kill "the whole JS event loop".
  In the real topology (pyodide in a dedicated Web Worker) the **worker's** loop
  dies while the page main thread stays fully alive. The UI survives; Python
  request handling is permanently and unrecoverably dead. Still fatal for the
  feature.
- **Not memory or total work.** Sequential at ~7× the work (12,096 keys × 2)
  completes fine in 25.5s.

**The root cause remains open.** What is established: zarr's WASM `sync()`
becomes `run_sync(future)`, suspending the WASM stack via JSPI, and resuming it
requires the JS macrotask queue because `WebLoop.call_soon` routes through
`setTimeout`. Instrumentation shows two `sync()` frames from two promising
entries live at once; the first suspends and never resumes, the second runs
~130ms then stops dead. Two specific internal-corruption hypotheses
(`validSuspender` save/restore crossing, `Module.stackStop` clobbering) were
tested directly and **refuted**. Treat the mechanism as unexplained.

### The `_sync_fallback` idea does not work — do not spend time on it

An earlier draft of this prompt suggested widening `_sync_fallback`'s predicate
(`_batch_reader.py:180`) to catch Pyodide, on the theory that `_gather_plan`'s
`async.concurrency=10` fan-out was the trigger. **The reproduction does not
support that.** The trigger is two concurrent _promising entrypoints_, each
holding a suspended WASM stack — internal fan-out within one call is not the
variable, and the failure scales with key count, i.e. with the number of
suspend/resume cycles. A serial inner path would still suspend just as often.

The mitigation that **is** proven is serialising every promising entry into
Python at the host: 4 and 6 concurrent requests at hanging scale all completed,
100%. That is the fork's bridge, not a library change.

## What to build

**`zarr_vectors/core/aio.py` already has the right shape — continue that.**
It appeared during this investigation and supersedes what this section
originally specified.

Its docstring makes the argument better than this prompt did: an async mirror of
each `read_*` "is the wrong one… duplicating eight of them would fork the format
logic in two, which is exactly the failure mode this package exists to avoid."
That is correct, and an earlier version of this document asked for exactly that
mistake. The prime-and-replay design — `await` the I/O up front, then run the
ordinary sync reader against a Group primed via `offline_reads` so it never
reaches `sync()`, recording misses and going round again — keeps one copy of the
format logic and works for every reader present and future.

The sync-call inventory below is retained only as a **coverage checklist** for
what the priming pass must satisfy. It is not a list of functions to mirror.
Instrumenting a real `read_polylines()` under Pyodide recorded **12 `sync()`
invocations**:

| Site                                                       | Count | What it does              |
| ---------------------------------------------------------- | ----- | ------------------------- |
| `core/group.py:797` `_lookup_node`                         | 4     | `node = self._zarr[path]` |
| `core/arrays.py:3693`, `:3696` `read_all_object_manifests` | 3     | manifest reads            |
| `core/group.py:157` `__contains__`                         | 2     | `group_name in root`      |
| `core/group.py:144` `__getitem__`                          | 2     | node access               |
| `core/_batch_reader.py:175`                                | 1     | `sync(_gather_plan(...))` |

Note for anyone who has seen an earlier draft of this analysis: **the "it's one
line at `_batch_reader.py:175`" claim is wrong.** That is 1 of 12, and
`_gather_plan` is only a prefetch **cache filler** — its own docstring
(`_batch_reader.py:116-117`) says that on a cache miss "the caller's sync
`Group.read_bytes` then raises `StoreError`". Promoting it does not give you a
read path.

Two constraints the priming pass must respect regardless of design:

1. **Accept a pre-built Store.** The injection point exists and is test-covered
   — `core/store.py:181-185` (`if isinstance(path, _ZStore): return path, None`),
   and `tests/test_batched_reads.py:94-103` already round-trips a
   `zarr.storage.MemoryStore` through `write_points`/`read_points`. The browser
   host supplies a fetch-backed Store.
2. **Avoid needing `list`.** `get_resolution_level` (`core/store.py:1289-1298`)
   uses only metadata GETs, but `list_resolution_levels` (`:1301-1320`) iterates
   the group, which requires store listing — and a browser-fetch Store cannot
   list. Prefer enumerating levels from the root `multiscales` attributes.

Keep the sync API exactly as it is — this is additive. CPython users should see
no change.

## Two small independent fixes

- **`lazy.open_zv` blocks Store injection.** It is annotated
  `path: str | Path` and calls `str(path)`, so a pre-built Store cannot get
  through the lazy API even though `open_store` accepts one. Widen to
  `StoreLike` and drop the coercion — `lazy/store.py:200` and `:227`,
  about two lines.
- **The batch writer spawns threads.** `core/_batch_writer.py:198` creates a
  `ThreadPoolExecutor`; `threading.Thread.start()` raises
  `RuntimeError: can't start new thread` under Pyodide. A serial branch already
  exists — reuse it when `zarr._constants.IS_WASM` (or
  `sys.platform == "emscripten"`), or clamp `_FLUSH_MAX_WORKERS` to 1. Only
  affects writes, so it does not block the read work, but it is cheap.

## Packaging

The fork needs the version carrying the **merged links / format 0.9.0** layout,
which is unreleased: `setuptools-scm` reports `0.2.1.dev66+gfedb231fd`, and
`zarr-vectors-tools`'s own pyproject comment says "the real guarantee is an
editable install of the core working tree". So `micropip.install("zarr-vectors")`
cannot reach it today.

The package is **pure Python** (setuptools + setuptools-scm, no `setup.py`, no
`ext_modules`, no Cython/cffi), so it builds a `py3-none-any` wheel and needs no
Pyodide-specific build. To unblock the fork, either cut a release, or publish a
built wheel the fork can host and `micropip.install(<url>)`.

## How to verify

The consumer is a browser, so CPython tests alone are not sufficient:

1. Keep the existing sync tests green — this work is additive.
2. Add async equivalents against `zarr.storage.MemoryStore`, mirroring
   `tests/test_batched_reads.py`.
3. Prove it under Pyodide, headlessly: load Pyodide 314.0.2, put the package on
   `sys.path`, supply a fetch-backed Store, and read a real store **through a
   plain (non-promising) synchronous entrypoint**. If that works, JSPI is not
   needed and the deadlock is designed out rather than worked around. Include a
   **concurrent** case — the failure is load-dependent and invisible on small
   stores.
4. Assert `sync()` is never reached on the async path. Monkeypatching
   `zarr.core.sync.sync` to raise is a blunt but effective guard.

## Reference

- Fork's reimplementation to be deleted:
  `neuroglancer/python/neuroglancer/tractography/zarr_vectors_source.py`
- Fork's stale rationale, to be corrected: same file lines 29-38
- Byte-identical decode: `zarr_vectors/encoding/fragments.py:12-38`, `:54-64`,
  `:348-400`
