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

"""A read-only zarr Store backed by the browser's ``fetch``.

This is the piece ``zarr_vectors.core.aio`` asks the host to supply: "the only
way in is a fetch-backed Store the caller constructs itself." With it, the
Pyodide viewer can read a zarr-vectors store through the real library rather
than a hand-rolled decoder -- see ``zarr_vectors_source.load_tract_index``.

**Imports zarr**, so it can only be imported where zarr is available -- i.e.
under Pyodide (which ships a WASM-patched zarr 3), not this repo's plain CPython
env. Callers import it lazily inside a ``try`` and fall back when it is absent.

Read-only, and deliberately non-listing: the async read path in ``aio`` never
lists (it fetches metadata and chunks by key), and a plain HTTP object store
cannot enumerate keys anyway. Any attempt to list raises.

The byte-range interpretation lives in ``zvf_pure.apply_byte_range`` so it can
be unit-tested without zarr present.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype

from .zvf_pure import apply_byte_range

FetchFn = Callable[[str], Awaitable[bytes | None]]


class BrowserFetchStore(Store):
    """Read-only zarr v3 store over an async ``fetch(url) -> bytes | None``.

    ``base_url`` is the store root (with or without a trailing slash); a zarr
    key is appended to it directly. ``fetch`` returns ``None`` for a 404, which
    zarr reads as an absent key.
    """

    def __init__(self, base_url: str, fetch: FetchFn):
        super().__init__(read_only=True)
        self._base = base_url if base_url.endswith("/") else base_url + "/"
        self._fetch = fetch

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BrowserFetchStore) and other._base == self._base

    def __hash__(self) -> int:
        return hash(("BrowserFetchStore", self._base))

    def __repr__(self) -> str:
        return f"BrowserFetchStore({self._base!r})"

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_partial_writes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return False

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        data = await self._fetch(self._base + key)
        if data is None:
            return None
        return prototype.buffer.from_bytes(apply_byte_range(data, byte_range))

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # One fetch per key. The stores this reads are coarse levels, so the
        # simplicity is worth more than pipelining here.
        return [await self.get(key, prototype, br) for key, br in key_ranges]

    async def exists(self, key: str) -> bool:
        return (await self._fetch(self._base + key)) is not None

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("BrowserFetchStore is read-only")

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("BrowserFetchStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("BrowserFetchStore is read-only")

    # An HTTP object store cannot enumerate keys, and the aio read path never
    # needs to. Each of these is an async generator that raises when iterated.
    async def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("BrowserFetchStore cannot list")
        yield  # pragma: no cover - marks this an async generator

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("BrowserFetchStore cannot list")
        yield  # pragma: no cover

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("BrowserFetchStore cannot list")
        yield  # pragma: no cover


def make_browser_fetch_store(base_url: str, fetch: FetchFn) -> Any:
    """Construct a :class:`BrowserFetchStore`.

    A thin factory so callers can obtain a store without importing this module
    at their top level (it imports zarr, which is not always present).
    """
    return BrowserFetchStore(base_url, fetch)
