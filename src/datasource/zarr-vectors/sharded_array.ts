/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Shared helpers for reading zarr-vectors-py 0.8.1 native-sharded
 * per-chunk arrays: `vertices`, `vertex_fragments`, `link_fragments`,
 * `links/<delta>`, `vertex_attributes/<name>`, `fragment_attributes/<name>`.
 *
 * Each such array is ONE Zarr v3 `variable_length_bytes` array whose
 * shape equals the level's spatial chunk grid, coded with a single
 * `sharding_indexed` outer codec (see
 * `zarr_vectors.core.group.Group.create_sharded_chunk_array`):
 *
 * - Outer chunk ("shard") shape = `chunk_grid.configuration.chunk_shape`
 *   on the array's own `zarr.json` (NOT a bespoke attribute — unlike the
 *   pre-0.8.1 `cross_chunk_links` kK arrays, which stamped `shard_shape`/
 *   `chunk_origin` attributes; per-chunk arrays here have no origin
 *   shift, chunk coord 0 is always the grid's own origin).
 * - Inner chunk shape is all-1s: one array ELEMENT per zarr sub-chunk,
 *   i.e. one spatial chunk's whole payload per "cell".
 * - The sharding codec's inner codec chain is `["vlen-bytes"]` (cells
 *   uncompressed — `zarr_vectors_layout`'s "raw" capability, stamped as
 *   `attributes.cell_codec === "raw"` on the array; used for `vertices`/
 *   `vertex_attributes/<name>` so a reader can byte-range a fragment's
 *   rows within a cell) or `["vlen-bytes", "zstd"]` (the default — cells
 *   zstd-compressed; used for `vertex_fragments`/`links`/
 *   `fragment_attributes`, which are read whole per chunk anyway).
 * - Shard file layout: cell payloads concatenated, followed by a flat
 *   `(numCellsInShard, 2)` row-major index of `(offset, length)` `uint64`
 *   little-endian pairs (absolute byte offsets within the shard file),
 *   followed by a 4-byte `crc32c` checksum (not verified by this reader),
 *   at `index_location` (`"start"` or `"end"`; this writer uses `"end"`).
 *   A pair of all-1 bits (`2^64 - 1`) marks an empty cell.
 * - Each cell's raw payload (post any zstd decompression) is exactly one
 *   numcodecs `vlen-bytes` item: `uint32` count (always 1, since the
 *   sharding codec's inner chunk shape is all-1s), `uint32` length, then
 *   that many raw bytes — the array element's actual encoded bytes
 *   (float32 vertex positions, a ZVFG fragment index, etc.).
 */

import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import { asyncMemoize, type AsyncMemoize } from "#src/util/memoize.js";

export const MISSING_SHARD_ENTRY = 0xffffffffffffffffn;

/** A byte-range request: an absolute range, or "the last N bytes" (no upfront file-size lookup needed — the kvstore layer resolves it). */
export type ByteRangeRequest =
  | { offset: number; length: number }
  | { suffixLength: number };

export interface ShardedArrayMeta {
  /** Shape of the array = the level's spatial chunk grid, one entry per axis. */
  readonly gridShape: number[];
  /** Outer "shard" chunk shape, same rank as `gridShape`. */
  readonly shardShape: number[];
  readonly indexLocation: "start" | "end";
  /** True iff cell payloads are uncompressed (`attributes.cell_codec === "raw"`) — byte-range-readable within a cell. */
  readonly cellsAreRaw: boolean;
}

/** Parse + validate one per-chunk array's `zarr.json` (native-sharded 0.8.1 layout). */
export function parseShardedArrayMeta(
  bytes: Uint8Array,
  arrayPath: string,
): ShardedArrayMeta {
  let meta: any;
  try {
    meta = JSON.parse(new TextDecoder().decode(bytes));
  } catch (e) {
    throw new Error(
      `${arrayPath}/zarr.json: invalid JSON: ${(e as Error).message}`,
    );
  }
  const gridShape = meta?.shape;
  if (!Array.isArray(gridShape)) {
    throw new Error(`${arrayPath}/zarr.json: missing 'shape'`);
  }
  const codecs = meta?.codecs;
  const shardingCodec = Array.isArray(codecs)
    ? codecs.find((c: any) => c?.name === "sharding_indexed")
    : undefined;
  if (shardingCodec === undefined) {
    throw new Error(
      `${arrayPath}/zarr.json: expected a 'sharding_indexed' codec, got ` +
        `${JSON.stringify(codecs)}`,
    );
  }
  // The shard shape is the ARRAY's own outer chunk grid (each regular
  // chunk of the array IS one shard file) — NOT the sharding codec's
  // `configuration.chunk_shape`, which is the inner per-cell sub-chunk
  // shape within a shard (always all-1s per the writer, see the module
  // docstring above).
  const shardShape = meta?.chunk_grid?.configuration?.chunk_shape;
  if (!Array.isArray(shardShape)) {
    throw new Error(
      `${arrayPath}/zarr.json: missing 'chunk_grid.configuration.chunk_shape'`,
    );
  }
  const innerChunkShape = shardingCodec.configuration?.chunk_shape;
  if (
    !Array.isArray(innerChunkShape) ||
    innerChunkShape.some((n: any) => n !== 1)
  ) {
    throw new Error(
      `${arrayPath}/zarr.json: expected sharding_indexed inner 'chunk_shape' ` +
        `to be all-1s, got ${JSON.stringify(innerChunkShape)}`,
    );
  }
  const indexLocation = shardingCodec.configuration?.index_location;
  if (indexLocation !== "start" && indexLocation !== "end") {
    throw new Error(
      `${arrayPath}/zarr.json: unsupported sharding index_location ` +
        `${JSON.stringify(indexLocation)}`,
    );
  }
  const innerCodecs = shardingCodec.configuration?.codecs;
  const hasVlenBytes =
    Array.isArray(innerCodecs) &&
    innerCodecs.some((c: any) => c?.name === "vlen-bytes");
  if (!hasVlenBytes) {
    throw new Error(
      `${arrayPath}/zarr.json: expected inner codec chain to include ` +
        `'vlen-bytes', got ${JSON.stringify(innerCodecs)}`,
    );
  }
  const cellsAreRaw = meta?.attributes?.cell_codec === "raw";
  const hasZstd =
    Array.isArray(innerCodecs) && innerCodecs.some((c: any) => c?.name === "zstd");
  if (cellsAreRaw === hasZstd) {
    throw new Error(
      `${arrayPath}/zarr.json: 'cell_codec' attribute (${JSON.stringify(
        meta?.attributes?.cell_codec,
      )}) is inconsistent with the inner codec chain ` +
        `${JSON.stringify(innerCodecs)}`,
    );
  }
  return {
    gridShape: gridShape.map(Number),
    shardShape: shardShape.map(Number),
    indexLocation,
    cellsAreRaw,
  };
}

/**
 * Decode one numcodecs ``vlen-bytes``-framed buffer that is known to
 * hold exactly one item (the convention for a zarr sub-chunk of shape
 * all-``1``s).  Layout: 4-byte little-endian item count, then per item
 * a 4-byte little-endian length followed by that many raw bytes.  See
 * https://github.com/zarr-developers/numcodecs/blob/main/src/numcodecs/vlen.pyx
 */
export function decodeVlenBytesSingleItem(buf: Uint8Array): Uint8Array {
  if (buf.byteLength < 8) {
    throw new Error(
      `sharded_array: vlen-bytes cell buffer too short (${buf.byteLength} bytes)`,
    );
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const numItems = dv.getUint32(0, true);
  if (numItems !== 1) {
    throw new Error(
      `sharded_array: expected exactly 1 vlen-bytes item per cell, got ${numItems}`,
    );
  }
  const length = dv.getUint32(4, true);
  if (8 + length > buf.byteLength) {
    throw new Error(
      `sharded_array: vlen-bytes cell buffer truncated ` +
        `(declared length ${length}, have ${buf.byteLength - 8} bytes)`,
    );
  }
  return buf.subarray(8, 8 + length);
}

/** Byte offset, within a raw cell's decoded payload, where the actual element bytes start (past the vlen-bytes single-item header). */
export const RAW_CELL_PAYLOAD_HEADER_LEN = 8;

function flattenRowMajor(
  coords: readonly number[],
  shape: readonly number[],
): number {
  let flat = 0;
  for (let d = 0; d < coords.length; ++d) {
    flat = flat * shape[d] + coords[d];
  }
  return flat;
}

/** Resolve one global chunk coordinate's shard coordinate + within-shard (row-major) sub-chunk coordinate. */
export function chunkToShardLocation(
  chunkCoords: readonly number[],
  shardShape: readonly number[],
): { shardCoord: number[]; subChunk: number[] } {
  const ndim = chunkCoords.length;
  const shardCoord = new Array<number>(ndim);
  const subChunk = new Array<number>(ndim);
  for (let d = 0; d < ndim; ++d) {
    shardCoord[d] = Math.floor(chunkCoords[d] / shardShape[d]);
    subChunk[d] = chunkCoords[d] % shardShape[d];
  }
  return { shardCoord, subChunk };
}

function shardIndexBodyLength(shardShape: readonly number[]): number {
  return shardShape.reduce((a, b) => a * b, 1) * 16;
}

/** Total byte length of a shard's index region (entries + trailing crc32c). */
export function shardIndexRegionLength(shardShape: readonly number[]): number {
  return shardIndexBodyLength(shardShape) + 4;
}

/** The byte-range request that fetches exactly one shard's index region (no whole-shard fetch). */
export function shardIndexByteRangeRequest(
  shardShape: readonly number[],
  indexLocation: "start" | "end",
): ByteRangeRequest {
  const len = shardIndexRegionLength(shardShape);
  return indexLocation === "end" ? { suffixLength: len } : { offset: 0, length: len };
}

/**
 * Parse one specific cell's byte range (absolute within its shard file)
 * out of that shard's already-fetched index region bytes. Returns
 * `undefined` if the cell is empty (the `MISSING_SHARD_ENTRY` sentinel).
 */
export function findCellByteRangeInIndex(
  indexBytes: Uint8Array,
  subChunk: readonly number[],
  shardShape: readonly number[],
): { offset: number; length: number } | undefined {
  const bodyLen = shardIndexBodyLength(shardShape);
  if (indexBytes.byteLength < bodyLen) {
    throw new Error(
      `sharded_array: index region (${indexBytes.byteLength} bytes) is ` +
        `smaller than expected (${bodyLen} bytes for shard shape ` +
        `${JSON.stringify(shardShape)})`,
    );
  }
  const flatCell = flattenRowMajor(subChunk, shardShape);
  const dv = new DataView(indexBytes.buffer, indexBytes.byteOffset, bodyLen);
  const offsetVal = dv.getBigUint64(flatCell * 16, true);
  const lengthVal = dv.getBigUint64(flatCell * 16 + 8, true);
  if (offsetVal === MISSING_SHARD_ENTRY && lengthVal === MISSING_SHARD_ENTRY) {
    return undefined;
  }
  return { offset: Number(offsetVal), length: Number(lengthVal) };
}

/**
 * Iterate every populated cell in an already-fetched WHOLE shard buffer
 * (used only by tests / whole-table enumeration — the per-chunk reader
 * below never fetches a whole shard).
 */
export function* iteratePopulatedCellsInShard(
  shardBytes: Uint8Array,
  shardShape: readonly number[],
  indexLocation: "start" | "end",
): Iterable<{ subChunk: number[]; byteRange: { offset: number; length: number } }> {
  const indexRegionLen = shardIndexRegionLength(shardShape);
  if (shardBytes.byteLength < indexRegionLen) {
    throw new Error(
      `sharded_array: shard (${shardBytes.byteLength} bytes) is smaller ` +
        `than its own index region (${indexRegionLen} bytes)`,
    );
  }
  const indexStart =
    indexLocation === "end" ? shardBytes.byteLength - indexRegionLen : 0;
  const indexBytes = shardBytes.subarray(
    indexStart,
    indexStart + shardIndexBodyLength(shardShape),
  );
  const numCells = shardShape.reduce((a, b) => a * b, 1);
  for (let flatCell = 0; flatCell < numCells; ++flatCell) {
    const subChunk = unflattenRowMajor(flatCell, shardShape);
    const range = findCellByteRangeInIndex(indexBytes, subChunk, shardShape);
    if (range === undefined) continue;
    yield { subChunk, byteRange: range };
  }
}

function unflattenRowMajor(
  flatIndex: number,
  shape: readonly number[],
): number[] {
  const ndim = shape.length;
  const out = new Array<number>(ndim);
  let rem = flatIndex;
  for (let d = ndim - 1; d >= 0; --d) {
    out[d] = rem % shape[d];
    rem = Math.floor(rem / shape[d]);
  }
  return out;
}

/** Decode one cell's fetched bytes (raw or zstd-compressed) into its raw element bytes. */
export async function decodeShardedArrayCellPayload(
  cellBytes: Uint8Array,
  cellsAreRaw: boolean,
  signal: AbortSignal,
): Promise<Uint8Array> {
  let framed: Uint8Array;
  if (cellsAreRaw) {
    framed = cellBytes;
  } else {
    // `requestAsyncComputation` transfers its buffer argument — copy
    // first so a cached/shared reference elsewhere isn't detached.
    const owned = new Uint8Array(cellBytes);
    framed = await requestAsyncComputation(
      decodeZstd,
      signal,
      [owned.buffer as ArrayBuffer],
      owned as Uint8Array<ArrayBuffer>,
    );
  }
  return decodeVlenBytesSingleItem(framed);
}

/** Reads bytes from a per-chunk array's shard files. */
export type ShardedArrayReadRange = (
  subpath: string,
  byteRange: ByteRangeRequest,
  signal: AbortSignal,
) => Promise<Uint8Array | undefined>;

export interface ShardedArrayReadContext {
  readonly kvStoreReadRange: ShardedArrayReadRange;
  /** Array logical path relative to the level base, e.g. `"vertices"` or `"vertex_attributes/radius"`. */
  readonly arrayPath: string;
  readonly meta: ShardedArrayMeta;
}

/**
 * Cache of resolved shard index-region bytes, keyed by
 * `"<arrayPath>/c/<shardCoord.join('/')>"`. Share one instance across
 * every chunk query for a given level so chunks that land in the same
 * shard (common — a shard typically covers many spatial chunks) reuse
 * the index fetch instead of re-fetching it.
 *
 * Each entry is an {@link AsyncMemoize}, not a bare `Promise`: many
 * concurrent chunk loads (e.g. everything visible while panning) can
 * want the same shard's index at once, each with its OWN `AbortSignal`
 * tied to its own chunk's lifetime. A bare cached `Promise` created
 * from whichever caller happened to arrive first would tie the shared
 * fetch to THAT caller's signal alone — if that caller's chunk got
 * evicted/cancelled first, every other concurrent caller still waiting
 * on the same cached promise would see an unrelated `AbortError` and
 * fail their entire chunk load, even though their own signal never
 * fired. `asyncMemoize` fixes this: the underlying fetch runs on an
 * internal `SharedAbortController` that only aborts once EVERY
 * registered caller's signal has fired, while each caller's own
 * `await` still resolves/rejects promptly on their own signal via
 * `raceWithAbort` — so one caller losing interest never poisons the
 * others.
 */
export type ShardIndexCache = Map<
  string,
  AsyncMemoize<Uint8Array | undefined>
>;

async function resolveShardIndexBytes(
  ctx: ShardedArrayReadContext,
  shardPath: string,
  signal: AbortSignal,
  cache?: ShardIndexCache,
): Promise<Uint8Array | undefined> {
  const fetchIt = (options: { signal: AbortSignal }) =>
    ctx.kvStoreReadRange(
      shardPath,
      shardIndexByteRangeRequest(ctx.meta.shardShape, ctx.meta.indexLocation),
      options.signal,
    );
  if (cache === undefined) return fetchIt({ signal });
  let memoized = cache.get(shardPath);
  if (memoized === undefined) {
    memoized = asyncMemoize(fetchIt);
    cache.set(shardPath, memoized);
  }
  return memoized({ signal });
}

/** Resolve one chunk's cell byte range (absolute within its shard file), or `undefined` if the chunk is empty/absent. */
export async function resolveShardedArrayCellRange(
  ctx: ShardedArrayReadContext,
  chunkCoords: readonly number[],
  signal: AbortSignal,
  cache?: ShardIndexCache,
): Promise<{ shardPath: string; offset: number; length: number } | undefined> {
  const { shardCoord, subChunk } = chunkToShardLocation(
    chunkCoords,
    ctx.meta.shardShape,
  );
  const shardPath = `${ctx.arrayPath}/c/${shardCoord.join("/")}`;
  const indexBytes = await resolveShardIndexBytes(ctx, shardPath, signal, cache);
  if (indexBytes === undefined) return undefined;
  const range = findCellByteRangeInIndex(
    indexBytes,
    subChunk,
    ctx.meta.shardShape,
  );
  if (range === undefined) return undefined;
  return { shardPath, ...range };
}

/** Fetch + decode one chunk's whole payload for a sharded per-chunk array. */
export async function readShardedArrayChunk(
  ctx: ShardedArrayReadContext,
  chunkCoords: readonly number[],
  signal: AbortSignal,
  cache?: ShardIndexCache,
): Promise<Uint8Array | undefined> {
  const cell = await resolveShardedArrayCellRange(ctx, chunkCoords, signal, cache);
  if (cell === undefined) return undefined;
  const cellBytes = await ctx.kvStoreReadRange(
    cell.shardPath,
    { offset: cell.offset, length: cell.length },
    signal,
  );
  if (cellBytes === undefined) return undefined;
  return decodeShardedArrayCellPayload(cellBytes, ctx.meta.cellsAreRaw, signal);
}

/**
 * Fetch a byte sub-range WITHIN one chunk's raw (uncompressed) payload —
 * the vlen-bytes single-item header is skipped automatically, so
 * `innerByteRange` addresses the decoded element bytes directly (e.g.
 * `rowIndex * bytesPerVertex`). Only valid when `ctx.meta.cellsAreRaw`;
 * throws otherwise (callers should check `cellsAreRaw` and fall back to
 * {@link readShardedArrayChunk} for compressed arrays).
 */
export async function readShardedArrayChunkScoped(
  ctx: ShardedArrayReadContext,
  chunkCoords: readonly number[],
  innerByteRange: { offset: number; length: number },
  signal: AbortSignal,
  cache?: ShardIndexCache,
): Promise<Uint8Array | undefined> {
  if (!ctx.meta.cellsAreRaw) {
    throw new Error(
      `${ctx.arrayPath}: scoped reads require raw (uncompressed) cells`,
    );
  }
  const cell = await resolveShardedArrayCellRange(ctx, chunkCoords, signal, cache);
  if (cell === undefined) return undefined;
  const payloadLen = cell.length - RAW_CELL_PAYLOAD_HEADER_LEN;
  if (innerByteRange.offset + innerByteRange.length > payloadLen) {
    throw new Error(
      `${ctx.arrayPath}: scoped range [${innerByteRange.offset}, ` +
        `${innerByteRange.offset + innerByteRange.length}) exceeds cell ` +
        `payload size ${payloadLen}`,
    );
  }
  return ctx.kvStoreReadRange(
    cell.shardPath,
    {
      offset: cell.offset + RAW_CELL_PAYLOAD_HEADER_LEN + innerByteRange.offset,
      length: innerByteRange.length,
    },
    signal,
  );
}

/**
 * Higher-level convenience layer: wires the primitives above into
 * `readArrayChunk` / `readArrayChunkScoped`-shaped closures (see
 * `skeleton_chunk_download.ts`), resolving + caching each array's
 * `ShardedArrayMeta` (from `<arrayPath>/zarr.json`) lazily on first use.
 * One `PerChunkArrayCaches` instance should be created per resolution
 * level and reused across every chunk download for that level.
 */
export interface PerChunkArraySource {
  /** Reads one small whole blob (e.g. an array's `zarr.json`), relative to the level base URL. */
  readonly kvStoreRead: (
    subpath: string,
    signal: AbortSignal,
  ) => Promise<Uint8Array | undefined>;
  readonly kvStoreReadRange: ShardedArrayReadRange;
}

export interface PerChunkArrayCaches {
  readonly metaCache: Map<string, Promise<ShardedArrayMeta | undefined>>;
  readonly shardIndexCache: ShardIndexCache;
}

export function createPerChunkArrayCaches(): PerChunkArrayCaches {
  return { metaCache: new Map(), shardIndexCache: new Map() };
}

async function resolveShardedArrayMetaCached(
  source: PerChunkArraySource,
  caches: PerChunkArrayCaches,
  arrayPath: string,
  signal: AbortSignal,
): Promise<ShardedArrayMeta | undefined> {
  let pending = caches.metaCache.get(arrayPath);
  if (pending === undefined) {
    pending = (async () => {
      const bytes = await source.kvStoreRead(`${arrayPath}/zarr.json`, signal);
      if (bytes === undefined) return undefined;
      return parseShardedArrayMeta(bytes, arrayPath);
    })().catch((error) => {
      caches.metaCache.delete(arrayPath);
      throw error;
    });
    caches.metaCache.set(arrayPath, pending);
  }
  return pending;
}

/** Build a `readArrayChunk`-shaped function (see `skeleton_chunk_download.ts`) backed by one level's per-chunk-array caches. */
export function makeReadArrayChunk(
  source: PerChunkArraySource,
  caches: PerChunkArrayCaches,
): (
  arrayPath: string,
  chunkCoords: readonly number[],
  signal: AbortSignal,
) => Promise<Uint8Array | undefined> {
  return async (arrayPath, chunkCoords, signal) => {
    const meta = await resolveShardedArrayMetaCached(
      source,
      caches,
      arrayPath,
      signal,
    );
    if (meta === undefined) return undefined;
    return readShardedArrayChunk(
      { kvStoreReadRange: source.kvStoreReadRange, arrayPath, meta },
      chunkCoords,
      signal,
      caches.shardIndexCache,
    );
  };
}

/**
 * Build a `readArrayChunkScoped`-shaped function; resolves to `undefined`
 * if the named array's cells aren't raw (uncompressed) — callers should
 * treat that the same as "range read unavailable" and fall back to
 * {@link makeReadArrayChunk}'s whole-chunk reader.
 */
export function makeReadArrayChunkScoped(
  source: PerChunkArraySource,
  caches: PerChunkArrayCaches,
): (
  arrayPath: string,
  chunkCoords: readonly number[],
  innerByteRange: { offset: number; length: number },
  signal: AbortSignal,
) => Promise<Uint8Array | undefined> {
  return async (arrayPath, chunkCoords, innerByteRange, signal) => {
    const meta = await resolveShardedArrayMetaCached(
      source,
      caches,
      arrayPath,
      signal,
    );
    if (meta === undefined || !meta.cellsAreRaw) return undefined;
    return readShardedArrayChunkScoped(
      { kvStoreReadRange: source.kvStoreReadRange, arrayPath, meta },
      chunkCoords,
      innerByteRange,
      signal,
      caches.shardIndexCache,
    );
  };
}
