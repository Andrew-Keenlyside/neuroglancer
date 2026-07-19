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

"""Read a coarse pyramid level of a zarr-vectors store into a `TractIndex`.

**Not used by the streamline filter.** The viewer already has the geometry
decoded, at the pyramid level it is actually drawing, and hands it over -- see
`service.py`. This module is a standalone reader for using the dissection
geometry from a Python script, and it earns its place only because
`zarr-vectors-py` cannot run under Pyodide (see below); if you are working in
CPython, prefer that package.

Only what an ROI dissection needs is read -- vertex positions, the fragment
index that cuts them into runs, and each fragment's object id -- and only at one
coarse level. On the HCP whole-brain tractogram that is a couple of megabytes,
against roughly a gigabyte for the finest level.

**Why this does not use `zarr-vectors-py`.** That package is the natural home for
this, and the format handling below is a direct restatement of its layout (ZVF
0.9). This module exists only because its read path cannot yet be driven from
the browser -- and the reason is narrower than it once appeared. Re-checked
2026-07-19 against pyodide 314.0.2:

* `zarr` itself is FINE. Pyodide ships zarr 3.2.1 with upstream's WASM patch, so
  its sync API runs on a WebLoop rather than a dedicated thread. See the pin
  comment in `src/main_pyodide.ts`.
* `obstore`/`fsspec`/`aiohttp` are NOT a blocker, contrary to what this file
  used to say. They are optional extras in both zarr and zarr-vectors, imported
  function-locally, and reached only when a caller passes a URL *string* rather
  than a pre-built store. Pyodide ships aiohttp and fsspec anyway.
* Compression is NOT a blocker either: `asyncio.to_thread` runs inline under
  Pyodide, so the zstd/blosc/gzip codecs decode.

What actually blocks it: every public read entry point in zarr-vectors-py is
synchronous, so calling it from JS needs a *promising* entrypoint and JSPI stack
switching -- and that makes the worker's request handlers interleave, which
deadlocks the event loop under concurrent reads. See the comment at the
`handleRequest` call in `src/python_integration/pyodide_worker.ts`.

So the fix is an async read API upstream, not more code here; see
`upstream_prompts/zarr-vectors-py.md`. Until that lands, reading the bytes with
the browser's own `fetch` and decoding them with numpy keeps the dependency
footprint at numpy alone.

**This module is a stopgap and should be deleted when that API exists.** It is a
strict subset of the real reader -- no compressors, no sharding, and it walks the
whole chunk grid absorbing 404s where core reads the `nonempty_chunks` manifest
-- and its fragment-index decode is byte-for-byte identical to
`zarr_vectors/encoding/fragments.py`. Two copies of one binary format is exactly
the drift risk it was written to avoid elsewhere.

The two decoders here mirror `src/datasource/zarr-vectors/vlen_bytes.ts` and
`src/datasource/zarr-vectors/fragment_index.ts`, which read the same bytes for
rendering. They are the specification's §7.3 layout; if that changes, all three
change together.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import struct
from collections.abc import Awaitable, Callable, Iterable

import numpy as np

from .index import TractIndex

__all__ = ["load_tract_index", "decode_vlen_bytes", "decode_fragment_index"]

FRAGMENT_INDEX_MAGIC = 0x5A564647  # 'ZVFG' little-endian
FRAGMENT_INDEX_VERSION = 1
_FRAGMENT_HEADER_BYTES = 16


# -- codecs ----------------------------------------------------------------


def decode_vlen_bytes(raw: bytes) -> list[bytes]:
    """Decode a zarr v3 `vlen-bytes` chunk into its list of blobs.

    Layout, all little-endian::

        uint32 num_elements N
        for each i in [0, N):
            uint32 byte_length L_i
            byte[L_i] data
    """
    if len(raw) < 4:
        raise ValueError(f"vlen-bytes chunk too short: {len(raw)} < 4 (header)")
    (n,) = struct.unpack_from("<I", raw, 0)
    out = []
    offset = 4
    for i in range(n):
        if len(raw) < offset + 4:
            raise ValueError(f"vlen-bytes chunk truncated at element {i} header")
        (length,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        if len(raw) < offset + length:
            raise ValueError(
                f"vlen-bytes chunk truncated in element {i} payload "
                f"(declared {length}, have {len(raw) - offset})"
            )
        out.append(raw[offset : offset + length])
        offset += length
    if offset != len(raw):
        raise ValueError(
            f"vlen-bytes chunk has {len(raw) - offset} trailing bytes after {n} elements"
        )
    return out


def decode_fragment_index(raw: bytes) -> list[np.ndarray]:
    """Decode a v1 fragment-index blob into one row-index array per fragment.

    Layout, all little-endian::

        HEADER (16 bytes)
            uint32 magic = 0x5A564647, uint16 version = 1, uint16 flags,
            uint32 num_fragments F, uint32 num_range_fragments R
        RANGE BITMAP    ceil(F/8) bytes padded to an 8-byte boundary;
                        bit f (LSB-first) set iff fragment f is a range
        RANGE TABLE     R x (int64 start, int64 count), in fragment order
        EXPLICIT CSR    uint32 offsets[E+1] then int64 indices[T], E = F - R

    Each fragment is either a contiguous range of row indices into the chunk's
    vertex array, or an explicit list of them.
    """
    if len(raw) < _FRAGMENT_HEADER_BYTES:
        raise ValueError(f"fragment index too short: {len(raw)}")
    magic, version, _flags, num_fragments, num_ranges = struct.unpack_from(
        "<IHHII", raw, 0
    )
    if magic != FRAGMENT_INDEX_MAGIC:
        raise ValueError(f"bad fragment-index magic: 0x{magic:08x}")
    if version != FRAGMENT_INDEX_VERSION:
        raise ValueError(f"unsupported fragment-index version {version}")

    offset = _FRAGMENT_HEADER_BYTES
    bitmap_raw = (num_fragments + 7) >> 3
    bitmap_padded = (bitmap_raw + 7) & ~7
    bitmap = np.frombuffer(raw, dtype=np.uint8, count=bitmap_raw, offset=offset)
    offset += bitmap_padded

    is_range = (
        np.unpackbits(bitmap, bitorder="little")[:num_fragments].astype(bool)
        if num_fragments
        else np.zeros(0, dtype=bool)
    )
    if int(is_range.sum()) != num_ranges:
        raise ValueError(
            f"fragment-index bitmap has {int(is_range.sum())} ranges, "
            f"header declares {num_ranges}"
        )

    range_table = np.frombuffer(
        raw, dtype="<i8", count=num_ranges * 2, offset=offset
    ).reshape(-1, 2)
    offset += num_ranges * 16

    num_explicit = num_fragments - num_ranges
    csr_offsets = np.frombuffer(raw, dtype="<u4", count=num_explicit + 1, offset=offset)
    offset += (num_explicit + 1) * 4
    total = int(csr_offsets[-1]) if num_explicit else 0
    csr_indices = np.frombuffer(raw, dtype="<i8", count=total, offset=offset)

    # Rank within each family, so fragment f maps to its own table row.
    range_rank = np.cumsum(is_range) - is_range
    explicit_rank = np.cumsum(~is_range) - ~is_range

    fragments = []
    for f in range(num_fragments):
        if is_range[f]:
            start, count = range_table[range_rank[f]]
            fragments.append(np.arange(start, start + count, dtype=np.intp))
        else:
            e = explicit_rank[f]
            fragments.append(
                csr_indices[csr_offsets[e] : csr_offsets[e + 1]].astype(np.intp)
            )
    return fragments


# -- fetching ---------------------------------------------------------------


async def _pyodide_fetch(url: str) -> bytes | None:
    from pyodide.http import pyfetch  # type: ignore

    response = await pyfetch(url)
    if response.status == 404:
        return None
    if response.status >= 400:
        raise OSError(f"{response.status} fetching {url}")
    return await response.bytes()


async def _urllib_fetch(url: str) -> bytes | None:
    """CPython-only fallback, for running this outside the browser.

    Never selected under Pyodide -- `asyncio.to_thread` would fail there, since
    Pyodide cannot start threads. `_default_fetch` picks `_pyodide_fetch`
    whenever `pyodide.http` imports, so this is only reachable off-platform.
    """
    import urllib.error
    import urllib.request

    def get():
        try:
            with urllib.request.urlopen(url) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    return await asyncio.to_thread(get)


def _default_fetch() -> Callable[[str], Awaitable[bytes | None]]:
    try:
        import pyodide.http  # noqa: F401
    except ImportError:
        return _urllib_fetch
    return _pyodide_fetch


# -- store layout -----------------------------------------------------------


def _chunk_key(prefix: str, coords: Iterable[int], separator: str = "/") -> str:
    return prefix + "c" + separator + separator.join(str(int(c)) for c in coords)


def _check_codecs(meta: dict, path: str) -> None:
    """Fail loudly on anything but a bare vlen-bytes chain.

    The coarse levels this reads are stored uncompressed, so supporting only
    that keeps the reader dependency-free. A compressed level is a clear error
    rather than a silent misparse.
    """
    names = [c.get("name", "") for c in meta.get("codecs", [])]
    unsupported = [n for n in names if n not in ("vlen-bytes", "bytes")]
    if unsupported:
        raise NotImplementedError(
            f"{path}: unsupported codec(s) {unsupported}. This reader handles "
            f"uncompressed vlen-bytes only; a level written with a compressor "
            f"needs numcodecs (unavailable under Pyodide by default)."
        )


async def _read_array_meta(fetch, base: str, path: str) -> dict:
    raw = await fetch(f"{base}{path}/zarr.json")
    if raw is None:
        raise FileNotFoundError(f"{base}{path}/zarr.json not found")
    meta = json.loads(raw.decode("utf-8"))
    _check_codecs(meta, path)
    return meta


# -- loading ----------------------------------------------------------------


async def load_tract_index(
    url: str,
    level: int = 3,
    *,
    fetch: Callable[[str], Awaitable[bytes | None]] | None = None,
    concurrency: int = 32,
    max_vertices: int | None = None,
    scale: Iterable[float] | None = None,
    translation: Iterable[float] | None = None,
) -> TractIndex:
    """Load one pyramid level of a zarr-vectors store as a `TractIndex`.

    `url` is the store root, with a trailing slash. `level` selects the pyramid
    level; it should be the one the viewer is drawing, since ids present in the
    index but absent from the drawn level simply never appear, while the
    converse under-selects.

    `scale` and `translation` map stored coordinates into the space the viewer's
    annotations live in -- they must match the transform the datasource applies,
    or regions will not line up with the tracts they appear to touch.

    `max_vertices` decimates each fragment, trading a little exactness in the
    `ANY_SEGMENT` test for a smaller index. See `TractIndex.decimate`.
    """
    if fetch is None:
        fetch = _default_fetch()
    base = url if url.endswith("/") else url + "/"
    prefix = f"{level}/"

    vertices_meta = await _read_array_meta(fetch, base, f"{prefix}vertices")
    grid_shape = [int(x) for x in vertices_meta["shape"]]
    separator = (
        vertices_meta.get("chunk_key_encoding", {})
        .get("configuration", {})
        .get("separator", "/")
    )
    chunk_shape = [
        int(x) for x in vertices_meta["chunk_grid"]["configuration"]["chunk_shape"]
    ]
    if any(c != 1 for c in chunk_shape):
        raise NotImplementedError(
            f"expected one cell per chunk, got chunk_shape {chunk_shape}"
        )
    await _read_array_meta(fetch, base, f"{prefix}vertex_fragments")
    await _read_array_meta(fetch, base, f"{prefix}fragment_attributes/segment_id")

    semaphore = asyncio.Semaphore(concurrency)

    async def read_cell(coords):
        key = _chunk_key("", coords, separator)
        async with semaphore:
            raw_vertices = await fetch(f"{base}{prefix}vertices/{key}")
            if raw_vertices is None:
                return None
            raw_fragments = await fetch(f"{base}{prefix}vertex_fragments/{key}")
            raw_ids = await fetch(f"{base}{prefix}fragment_attributes/segment_id/{key}")
        if raw_fragments is None or raw_ids is None:
            # Vertices without their fragment index or ids cannot be attributed
            # to a tract, so they are unusable rather than merely incomplete.
            raise ValueError(
                f"cell {tuple(coords)} has vertices but is missing "
                f"{'vertex_fragments' if raw_fragments is None else 'segment_id'}"
            )
        return coords, raw_vertices, raw_fragments, raw_ids

    cells = await asyncio.gather(
        *(read_cell(c) for c in itertools.product(*(range(n) for n in grid_shape)))
    )

    position_blocks: list[np.ndarray] = []
    counts: list[int] = []
    row_ids: list[np.ndarray] = []

    for cell in cells:
        if cell is None:
            continue
        coords, raw_vertices, raw_fragments, raw_ids = cell
        vertices_blob = decode_vlen_bytes(raw_vertices)[0]
        fragments_blob = decode_vlen_bytes(raw_fragments)[0]
        ids_blob = decode_vlen_bytes(raw_ids)[0]
        if not vertices_blob:
            continue

        vertices = np.frombuffer(vertices_blob, dtype="<f4").reshape(-1, 3)
        fragments = decode_fragment_index(fragments_blob)
        segment_ids = np.frombuffer(ids_blob, dtype="<u8")
        if len(segment_ids) != len(fragments):
            raise ValueError(
                f"cell {tuple(coords)}: {len(fragments)} fragments but "
                f"{len(segment_ids)} segment ids"
            )

        for rows, object_id in zip(fragments, segment_ids):
            if rows.size == 0:
                continue
            position_blocks.append(vertices[rows])
            counts.append(int(rows.size))
            row_ids.append(object_id)

    if not position_blocks:
        raise ValueError(
            f"no streamlines found at level {level} of {base}. Check the level "
            f"exists and the store is readable."
        )

    positions = np.concatenate(position_blocks)
    offsets = np.zeros(len(counts) + 1, dtype=np.intp)
    np.cumsum(np.asarray(counts, dtype=np.intp), out=offsets[1:])
    index = TractIndex(positions, offsets, np.array(row_ids, dtype=np.uint64))

    if scale is not None or translation is not None:
        scaled = index.positions.astype(np.float32, copy=True)
        if scale is not None:
            scaled *= np.asarray(list(scale), dtype=np.float32)
        if translation is not None:
            scaled += np.asarray(list(translation), dtype=np.float32)
        index = TractIndex(scaled, index.offsets, index.row_ids)

    if max_vertices is not None:
        index = index.decimate(max_vertices)
    return index
