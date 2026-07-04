/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Read and decode the ``cross_chunk_links/<delta>/`` tree emitted by
 * zarr-vectors 0.8 writers under the ``"explicit_links"`` cross-chunk
 * strategy.
 *
 * **v0.8 sharded layout** (see
 * ``zarr-vectors-py/docs/spec/object_model/cross_chunk_links.md`` and
 * ``zarr-vectors-py/zarr_vectors/core/arrays.py:write_cross_chunk_links``):
 * cross-chunk records are partitioned by ``K`` (the number of distinct
 * chunks a record touches, ``1 <= K <= link_width``) into a family of
 * ``cross_chunk_links/<delta>/kK/`` arrays.  Each ``kK`` array is a zarr
 * v3 ``sharding_indexed``-coded, variable-length-bytes array whose
 * logical shape is ``(Cx, Cy, Cz)`` repeated ``K`` times; the cell at
 * coordinate ``(chunk_0 - origin) ⧺ … ⧺ (chunk_{K-1} - origin)`` (chunks
 * in lex order) holds a ragged byte blob encoding every record whose
 * sorted-unique-chunks tuple equals that cell's chunks.
 *
 * Each cell's *decoded* (post inner-codec) payload is:
 * ``[ci_0..ci_{L-1} (L * uint8), vi_0..vi_{L-1} (L * int64 little-endian)]``
 * repeated back-to-back per record, where ``L = link_width``.  ``ci_i``
 * indexes into the cell's K sorted chunks to say which chunk endpoint
 * ``i`` lives in; ``vi_i`` is its local vertex index in that chunk.
 *
 * The inner (per-cell) codec chain the writer uses is
 * ``vlen-bytes`` (numcodecs' length-prefixed framing — see
 * https://github.com/zarr-developers/numcodecs/blob/main/src/numcodecs/vlen.pyx)
 * then ``zstd``.  Since each cell's zarr *sub-chunk* holds exactly one
 * array element (inner ``chunk_shape`` is all-``1``s), the vlen-bytes
 * frame always has exactly one item; decoding it recovers the raw
 * record bytes above.
 *
 * The outer shard's on-disk layout (zarr v3 ``sharding_indexed``, see
 * the zarr v3 spec) is: concatenated cell byte blobs, followed by a
 * flat ``(num_cells_in_shard, 2)`` (row-major) index of
 * ``(offset, length)`` ``uint64`` little-endian pairs, followed by a
 * 4-byte ``crc32c`` checksum over the index — with the index block
 * placed at ``index_location`` (``"start"`` or ``"end"``, this writer
 * uses ``"end"``).  A pair of all-``1`` bits (``2^64 - 1``) marks an
 * absent cell.
 *
 * v0.7 stores (single flat blob at ``cross_chunk_links/<delta>/data``)
 * are not supported here — see the "Migration from 0.7" section of the
 * spec page above.
 */

import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import { arraysEqual } from "#src/util/array.js";

/** One endpoint of a cross-chunk record. */
export interface CrossChunkLinkEndpoint {
  /** Length-``sidNdim`` chunk-grid coordinates of the endpoint's chunk. */
  readonly chunkCoords: number[];
  /** 0-based vertex index inside that chunk's ``vertices/`` array. */
  readonly vertexIndex: number;
}

/** One ``link_width``-arity cross-chunk record. */
export interface CrossChunkLinkRecord {
  readonly endpoints: CrossChunkLinkEndpoint[];
}

/** Whole table for one ``(level, delta)`` pair. */
export interface CrossChunkLinksTable {
  readonly linkWidth: number;
  readonly sidNdim: number;
  readonly records: CrossChunkLinkRecord[];
}

/** Directory-listing result for one prefix, as narrow bare-name arrays. */
export interface CrossChunkLinksListResult {
  /** Immediate subdirectory names (no trailing separator), relative to the prefix. */
  readonly directories: string[];
  /** Immediate file names, relative to the prefix. */
  readonly files: string[];
}

/**
 * Configuration for {@link readCrossChunkLinks}.  Lets callers
 * substitute a kvstore reader/lister at the test boundary.
 */
export interface CrossChunkLinksReaderOptions {
  /**
   * Reads one byte blob relative to the resolution-level base URL.
   * Returns ``undefined`` for missing keys.  **Must return raw
   * (non-decompressed) bytes** — this module manages zstd
   * decompression itself at the per-cell granularity; a shard file's
   * overall bytes are a multi-cell concatenation, not a single zstd
   * frame, so passing a blob through a magic-byte-sniffing
   * auto-decompressor (as used for plain vertex/attribute reads
   * elsewhere in this datasource) would corrupt shard parsing.
   */
  readonly kvStoreRead: (
    subpath: string,
    signal: AbortSignal,
  ) => Promise<Uint8Array | undefined>;
  /**
   * Lists the immediate children of one directory-like prefix
   * (relative to the resolution-level base URL).  Required to discover
   * which shards of a ``kK`` array are populated — sharded arrays have
   * no way to enumerate written cells other than listing the store.
   * Omit only in contexts (e.g. tests targeting a single known shard)
   * that don't need whole-table enumeration.
   */
  readonly kvStoreList?: (
    prefix: string,
    signal: AbortSignal,
  ) => Promise<CrossChunkLinksListResult>;
  /** Level delta; 0 for intra-level, ±N for cross-level pyramids. */
  readonly delta?: number;
}

/**
 * A whole-level `cross_chunk_links/<delta>/` table can hold tens of
 * millions of records (a real ~100k-streamline store's level 0 has been
 * measured at 21.6M) — decoding all of them into `CrossChunkLinksTable`'s
 * nested-object representation costs roughly 15-20x the packed on-disk
 * size once V8's per-object/per-array overhead is counted (record →
 * endpoints array → endpoint objects → chunkCoords arrays), multiple
 * gigabytes for a real dataset.  Worse, holding that table in a plain
 * field (as callers here do) makes it invisible to neuroglancer's
 * `ChunkState`-tracked GPU/system memory budget — it's never evicted
 * under memory pressure.
 *
 * `readCrossChunkLinksForChunk` decodes only the records touching one
 * specific chunk (the actual per-`download()` need — see
 * `skeleton_backend.ts`), reusing two caches so repeated chunk queries
 * amortize the discovery/fetch cost without re-decoding the world:
 *
 * - `shardCoords`: populated-shard-coordinate lists are cheap to keep
 *   (arrays of small coordinate tuples, not decoded records) and avoid
 *   re-walking the `kvStoreList` discovery tree on every chunk query.
 * - `shardBytes`: raw (still-compressed) whole-shard byte blobs, bounded
 *   by {@link BoundedByteCache} so a huge dataset's shard bytes don't
 *   accumulate without limit — nearby chunks' queries typically land in
 *   the same or adjacent shards, so caching pays off without needing to
 *   retain every shard ever touched.
 *
 * Chunk coordinates are NOT guaranteed to be spatially adjacent to be
 * linked — a heavily-decimated coarse pyramid level's boundary-crossing
 * points can be many original samples apart, so a cross-chunk link can
 * legitimately span chunks that aren't neighbors.  That rules out a
 * "only check nearby chunk pairs" shortcut; the safe optimization is to
 * skip *decoding* a cell once its chunk coordinates are known (cheap —
 * derived from the cell's position within the shard, no decompression
 * needed) to not include the target chunk, not to skip *discovering*
 * cells in the first place.
 */
export interface CrossChunkLinksCaches {
  /** `kBase` path -> populated shard coordinate list. */
  readonly shardCoords: Map<string, number[][]>;
  /** `"kBase/c/shardCoord.join('/')"` -> raw (compressed) shard bytes. */
  readonly shardBytes: BoundedByteCache;
}

/** Fresh, empty cache pair for one `(store, delta)` — share across every chunk query for that pair. */
export function createCrossChunkLinksCaches(
  maxShardBytesCached = 256 * 1024 * 1024,
): CrossChunkLinksCaches {
  return {
    shardCoords: new Map(),
    shardBytes: new BoundedByteCache(maxShardBytesCached),
  };
}

/**
 * Byte-budgeted LRU cache.  Insertion order in a `Map` doubles as
 * recency order: `get` re-inserts the hit key so it moves to the "most
 * recently used" (last) position; `set` evicts from the "least recently
 * used" (first) position while the tracked byte total exceeds budget.
 */
export class BoundedByteCache {
  private readonly entries = new Map<string, Uint8Array>();
  private totalBytes = 0;

  constructor(private readonly maxBytes: number) {}

  get(key: string): Uint8Array | undefined {
    const value = this.entries.get(key);
    if (value === undefined) return undefined;
    this.entries.delete(key);
    this.entries.set(key, value);
    return value;
  }

  set(key: string, value: Uint8Array): void {
    const existing = this.entries.get(key);
    if (existing !== undefined) {
      this.totalBytes -= existing.byteLength;
      this.entries.delete(key);
    }
    this.entries.set(key, value);
    this.totalBytes += value.byteLength;
    while (this.totalBytes > this.maxBytes && this.entries.size > 1) {
      const oldestKey = this.entries.keys().next().value!;
      const oldestValue = this.entries.get(oldestKey)!;
      this.entries.delete(oldestKey);
      this.totalBytes -= oldestValue.byteLength;
    }
  }
}

const MISSING_SHARD_ENTRY = 0xffffffffffffffffn;

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
      `cross_chunk_links: vlen-bytes cell buffer too short (${buf.byteLength} bytes)`,
    );
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const numItems = dv.getUint32(0, true);
  if (numItems !== 1) {
    throw new Error(
      `cross_chunk_links: expected exactly 1 vlen-bytes item per cell, got ${numItems}`,
    );
  }
  const length = dv.getUint32(4, true);
  if (8 + length > buf.byteLength) {
    throw new Error(
      `cross_chunk_links: vlen-bytes cell buffer truncated ` +
        `(declared length ${length}, have ${buf.byteLength - 8} bytes)`,
    );
  }
  return buf.subarray(8, 8 + length);
}

/**
 * Decode one cell's raw record payload (post vlen-bytes framing) into
 * structured records.  ``sortedChunks`` is the cell's K sorted-unique
 * chunk-coordinate tuples (length K, each of length ``sidNdim``), in
 * the same order the writer used to compute each record's ``ci``.
 * Exported so tests can drive the decoder with hand-crafted byte
 * fixtures.
 */
export function decodeCrossChunkLinkCell(
  payload: Uint8Array,
  sortedChunks: readonly (readonly number[])[],
  linkWidth: number,
): CrossChunkLinkRecord[] {
  if (linkWidth < 1) {
    throw new Error(
      `cross_chunk_links: link_width must be >= 1; got ${linkWidth}`,
    );
  }
  const K = sortedChunks.length;
  const recordStride = linkWidth * 1 + linkWidth * 8; // L uint8 + L int64
  if (payload.byteLength % recordStride !== 0) {
    throw new Error(
      `cross_chunk_links: cell payload ${payload.byteLength} bytes is not ` +
        `a multiple of one record (${recordStride} bytes for link_width=${linkWidth})`,
    );
  }
  const numRecords = payload.byteLength / recordStride;
  const dv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const records: CrossChunkLinkRecord[] = [];
  for (let r = 0; r < numRecords; ++r) {
    const recBase = r * recordStride;
    const ciBase = recBase;
    const viBase = recBase + linkWidth;
    const endpoints: CrossChunkLinkEndpoint[] = [];
    for (let i = 0; i < linkWidth; ++i) {
      const ci = dv.getUint8(ciBase + i);
      if (ci < 0 || ci >= K) {
        throw new Error(
          `cross_chunk_links: record ${r} endpoint ${i} has ci=${ci} out of ` +
            `range [0, ${K})`,
        );
      }
      const vi = dv.getBigInt64(viBase + i * 8, true);
      endpoints.push({
        chunkCoords: sortedChunks[ci].slice(),
        vertexIndex: Number(vi),
      });
    }
    records.push({ endpoints });
  }
  return records;
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

/**
 * Walk one shard's trailing/leading index and yield every populated
 * cell's local (within-shard) coordinate and byte range.
 */
function* iteratePopulatedCellsInShard(
  shardBytes: Uint8Array,
  shardShape: readonly number[],
  indexLocation: "start" | "end",
): Iterable<{ subChunk: number[]; byteRange: { offset: number; length: number } }> {
  const numCells = shardShape.reduce((a, b) => a * b, 1);
  const indexBodyLen = numCells * 2 * 8;
  const CHECKSUM_LEN = 4;
  const indexRegionLen = indexBodyLen + CHECKSUM_LEN;
  if (shardBytes.byteLength < indexRegionLen) {
    throw new Error(
      `cross_chunk_links: shard (${shardBytes.byteLength} bytes) is smaller ` +
        `than its own index region (${indexRegionLen} bytes for ${numCells} cells)`,
    );
  }
  const indexStart =
    indexLocation === "end"
      ? shardBytes.byteLength - indexRegionLen
      : 0;
  const dv = new DataView(
    shardBytes.buffer,
    shardBytes.byteOffset + indexStart,
    indexBodyLen,
  );
  for (let flatCell = 0; flatCell < numCells; ++flatCell) {
    const offsetVal = dv.getBigUint64(flatCell * 16, true);
    const lengthVal = dv.getBigUint64(flatCell * 16 + 8, true);
    if (offsetVal === MISSING_SHARD_ENTRY && lengthVal === MISSING_SHARD_ENTRY) {
      continue;
    }
    yield {
      subChunk: unflattenRowMajor(flatCell, shardShape),
      byteRange: { offset: Number(offsetVal), length: Number(lengthVal) },
    };
  }
}

/**
 * Upper bound on simultaneously in-flight ``kvStoreList`` requests during
 * shard discovery.  The shard-coordinate space is ``ndim``-D (``sid_ndim *
 * K`` for a ``kK`` array), so a naive fully-parallel recursive walk over a
 * wide chunk grid can fan out into tens of thousands of concurrent HTTP
 * requests and exhaust the browser's connection pool
 * (``net::ERR_INSUFFICIENT_RESOURCES``). Capping concurrency keeps total
 * request count the same but bounds how many are ever in flight at once.
 */
const MAX_CONCURRENT_SHARD_LIST_REQUESTS = 8;

/**
 * Minimal concurrency limiter: at most ``maxConcurrent`` wrapped calls run
 * at once; additional calls queue and start as running ones settle.
 */
function createConcurrencyLimiter(maxConcurrent: number) {
  let active = 0;
  const queue: Array<() => void> = [];
  function next(): void {
    if (active >= maxConcurrent || queue.length === 0) return;
    active++;
    queue.shift()!();
  }
  return function limit<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      queue.push(() => {
        fn().then(
          (value) => {
            active--;
            resolve(value);
            next();
          },
          (error) => {
            active--;
            reject(error);
            next();
          },
        );
      });
      next();
    });
  };
}

/**
 * Recursively list every populated shard's coordinate tuple under
 * ``<kKBasePath>/c``.  zarr v3's default chunk-key encoding lays a
 * ``ndim``-D chunk coordinate out as ``ndim`` nested path segments
 * (``c/<i0>/<i1>/…/<i_{ndim-1}>``); this walks that tree one level at a
 * time via ``kvStoreList``, descending into every reported subdirectory
 * and treating leaves at depth ``ndim`` as populated shard files.
 *
 * Requests are routed through a shared concurrency limiter (see
 * ``MAX_CONCURRENT_SHARD_LIST_REQUESTS``) so a wide/deep shard grid can't
 * fire an unbounded burst of simultaneous ``kvStoreList`` calls.
 */
async function listPopulatedShardCoords(
  kvStoreList: (
    prefix: string,
    signal: AbortSignal,
  ) => Promise<CrossChunkLinksListResult>,
  cBasePath: string,
  ndim: number,
  signal: AbortSignal,
): Promise<number[][]> {
  const results: number[][] = [];
  const limit = createConcurrencyLimiter(MAX_CONCURRENT_SHARD_LIST_REQUESTS);
  async function walk(prefix: string, coordsSoFar: number[]): Promise<void> {
    if (coordsSoFar.length === ndim) {
      results.push(coordsSoFar);
      return;
    }
    const { directories, files } = await limit(() =>
      kvStoreList(`${prefix}/`, signal),
    );
    const isLastLevel = coordsSoFar.length === ndim - 1;
    const names = isLastLevel ? [...directories, ...files] : directories;
    await Promise.all(
      names.map((name) => {
        const idx = Number(name);
        if (!Number.isInteger(idx)) return Promise.resolve();
        return walk(`${prefix}/${name}`, [...coordsSoFar, idx]);
      }),
    );
  }
  await walk(cBasePath, []);
  return results;
}

interface KArrayMeta {
  shardShape: number[];
  chunkOrigin: number[];
  indexLocation: "start" | "end";
}

/**
 * Parse + validate one ``kK`` array's ``zarr.json``, asserting the
 * exact codec chain this reader knows how to decode (defensive: a
 * differently-configured store should fail loudly, not silently
 * mis-decode).
 */
function parseKArrayMeta(bytes: Uint8Array, kBase: string): KArrayMeta {
  let meta: any;
  try {
    meta = JSON.parse(new TextDecoder().decode(bytes));
  } catch (e) {
    throw new Error(
      `${kBase}/zarr.json: invalid JSON: ${(e as Error).message}`,
    );
  }
  const attrs = meta?.attributes ?? {};
  const shardShape = attrs.shard_shape;
  const chunkOrigin = attrs.chunk_origin;
  if (!Array.isArray(shardShape) || !Array.isArray(chunkOrigin)) {
    throw new Error(
      `${kBase}/zarr.json: missing 'shard_shape' / 'chunk_origin' attributes`,
    );
  }
  const codecs = meta?.codecs;
  const shardingCodec = Array.isArray(codecs)
    ? codecs.find((c: any) => c?.name === "sharding_indexed")
    : undefined;
  if (shardingCodec === undefined) {
    throw new Error(
      `${kBase}/zarr.json: expected a 'sharding_indexed' codec, got ` +
        `${JSON.stringify(codecs)}`,
    );
  }
  const indexLocation = shardingCodec.configuration?.index_location;
  if (indexLocation !== "start" && indexLocation !== "end") {
    throw new Error(
      `${kBase}/zarr.json: unsupported sharding index_location ` +
        `${JSON.stringify(indexLocation)}`,
    );
  }
  const innerCodecs = shardingCodec.configuration?.codecs;
  const hasVlenBytes =
    Array.isArray(innerCodecs) &&
    innerCodecs.some((c: any) => c?.name === "vlen-bytes");
  if (!hasVlenBytes) {
    throw new Error(
      `${kBase}/zarr.json: expected inner codec chain to include ` +
        `'vlen-bytes', got ${JSON.stringify(innerCodecs)}`,
    );
  }
  return {
    shardShape: shardShape.map(Number),
    chunkOrigin: chunkOrigin.map(Number),
    indexLocation,
  };
}

async function decodeCellPayload(
  compressedBytes: Uint8Array,
  signal: AbortSignal,
): Promise<Uint8Array> {
  // `compressedBytes` is a view into a shared per-shard buffer that other
  // cells in the same shard still need to read from — `requestAsyncComputation`
  // transfers its buffer argument (postMessage transfer semantics), which
  // would detach that shared buffer out from under the rest of the shard's
  // cells.  Copy into a freshly-allocated, standalone buffer first.
  const owned = new Uint8Array(compressedBytes);
  const decompressed = await requestAsyncComputation(
    decodeZstd,
    signal,
    [owned.buffer as ArrayBuffer],
    owned as Uint8Array<ArrayBuffer>,
  );
  return decodeVlenBytesSingleItem(decompressed);
}

/**
 * Shared walk: parses the group + per-`K` array metadata, discovers
 * shards, and iterates every populated cell — exactly the fixed
 * structure both {@link readCrossChunkLinks} (whole table, no caching)
 * and {@link readCrossChunkLinksForChunk} (one chunk's records, cached
 * discovery/shard-bytes) need.  The two differ only in:
 *
 * - `getShardCoords`: fresh `listPopulatedShardCoords` walk each time,
 *   vs. a cached lookup.
 * - `getShardBytes`: fresh `kvStoreRead` each time, vs. a cached lookup.
 * - `shouldDecodeCell`: always `true`, vs. cheaply checking the cell's
 *   already-known chunk coordinates (no decompression needed for this
 *   check) against a target chunk before paying for `decodeCellPayload`
 *   + `decodeCrossChunkLinkCell`.
 *
 * Keeping one implementation means a decode/validation fix made here
 * can't accidentally apply to only one of the two entry points.
 */
async function readCrossChunkLinksImpl(
  options: CrossChunkLinksReaderOptions,
  signal: AbortSignal,
  shouldDecodeCell: (sortedChunks: readonly (readonly number[])[]) => boolean,
  getShardCoords: (
    kBase: string,
    ndim: number,
  ) => Promise<number[][]>,
  getShardBytes: (shardKey: string) => Promise<Uint8Array | undefined>,
): Promise<CrossChunkLinksTable | undefined> {
  const { kvStoreRead, kvStoreList, delta = 0 } = options;
  const base = `cross_chunk_links/${delta}`;

  const groupMetaBytes = await kvStoreRead(`${base}/zarr.json`, signal);
  if (groupMetaBytes === undefined) {
    return undefined;
  }
  let groupMeta: any;
  try {
    groupMeta = JSON.parse(new TextDecoder().decode(groupMetaBytes));
  } catch (e) {
    throw new Error(
      `${base}/zarr.json: invalid JSON: ${(e as Error).message}`,
    );
  }
  const attrs = groupMeta?.attributes;
  if (attrs === undefined) {
    throw new Error(`${base}/zarr.json: missing 'attributes' object`);
  }
  const layout = attrs.layout;
  // A genuine pre-0.8 (single-blob) store stamps `num_links` on the group
  // meta — the v0.8 rewrite dropped that field entirely (record counts are
  // implicit per-cell).  Some 0.8 writers predate a fix that stamps
  // `layout: "sharded_v1"` explicitly (zarr-vectors-py's
  // create_cross_chunk_links_array / write_cross_chunk_links originally
  // documented but didn't actually write it) — tolerate a missing `layout`
  // as "assume 0.8" as long as the tell-tale v0.7 `num_links` field isn't
  // present, so already-ingested stores don't need re-ingesting.
  if (layout !== "sharded_v1" && layout !== undefined) {
    throw new Error(
      `${base}: unsupported cross_chunk_links layout ${JSON.stringify(layout)} ` +
        `(expected "sharded_v1")`,
    );
  }
  if (layout === undefined && attrs.num_links !== undefined) {
    throw new Error(
      `${base}: pre-0.8 single-blob cross_chunk_links layout detected ` +
        `('num_links' present, no 'layout' field) — run the zarr-vectors ` +
        `0.7→0.8 migration helper`,
    );
  }
  const linkWidth = Number(attrs.link_width);
  const sidNdim = Number(attrs.sid_ndim);
  if (!Number.isInteger(linkWidth) || linkWidth < 1) {
    throw new Error(`${base}: invalid link_width ${attrs.link_width}`);
  }
  if (!Number.isInteger(sidNdim) || sidNdim < 1) {
    throw new Error(`${base}: invalid sid_ndim ${attrs.sid_ndim}`);
  }

  const records: CrossChunkLinkRecord[] = [];

  for (let K = 1; K <= linkWidth; ++K) {
    const kBase = `${base}/k${K}`;
    const kMetaBytes = await kvStoreRead(`${kBase}/zarr.json`, signal);
    if (kMetaBytes === undefined) {
      // Lazy per-K allocation: no records ever landed in this bucket.
      continue;
    }
    const { shardShape, chunkOrigin, indexLocation } = parseKArrayMeta(
      kMetaBytes,
      kBase,
    );
    const ndim = sidNdim * K;
    if (shardShape.length !== ndim) {
      throw new Error(
        `${kBase}: shard_shape length ${shardShape.length} != ` +
          `sid_ndim*K (${ndim})`,
      );
    }
    if (kvStoreList === undefined) {
      throw new Error(
        `${kBase}: reading a populated cross_chunk_links kK array requires ` +
          `a kvStoreList callback to discover its shards`,
      );
    }
    const shardCoords = await getShardCoords(kBase, ndim);
    for (const shardCoord of shardCoords) {
      const shardKey = `${kBase}/c/${shardCoord.join("/")}`;
      const shardBytes = await getShardBytes(shardKey);
      if (shardBytes === undefined) continue;
      for (const { subChunk, byteRange } of iteratePopulatedCellsInShard(
        shardBytes,
        shardShape,
        indexLocation,
      )) {
        const cellCoordFull = subChunk.map(
          (v, d) => v + shardCoord[d] * shardShape[d],
        );
        const sortedChunks: number[][] = [];
        for (let k = 0; k < K; ++k) {
          const chunk: number[] = [];
          for (let a = 0; a < sidNdim; ++a) {
            chunk.push(cellCoordFull[k * sidNdim + a] + chunkOrigin[a]);
          }
          sortedChunks.push(chunk);
        }
        // Decided purely from the cell's position within the shard —
        // no decompression needed, so skipping here is nearly free and
        // avoids the (potentially very large) decode/object-allocation
        // cost for every cell that doesn't touch the caller's target.
        if (!shouldDecodeCell(sortedChunks)) continue;
        const compressedCell = shardBytes.subarray(
          byteRange.offset,
          byteRange.offset + byteRange.length,
        );
        const payload = await decodeCellPayload(compressedCell, signal);
        // Loop rather than `records.push(...cellRecords)`: a single cell can
        // hold far more than V8's ~65,536-argument call limit on large
        // datasets, and spreading into a call throws "Maximum call stack
        // size exceeded" past that limit.
        for (const record of decodeCrossChunkLinkCell(
          payload,
          sortedChunks,
          linkWidth,
        )) {
          records.push(record);
        }
      }
    }
  }

  return { linkWidth, sidNdim, records };
}

/**
 * Fetch + decode ``cross_chunk_links/<delta>/`` for one resolution
 * level.  Returns ``undefined`` if the group is absent entirely (no
 * ``explicit_links`` cross-chunk strategy at this level/delta).
 *
 * Decodes every record in the level into memory — for a real dataset
 * this can be tens of millions of records / multiple gigabytes (see
 * {@link CrossChunkLinksCaches}'s docstring).  Prefer
 * {@link readCrossChunkLinksForChunk} when only one chunk's records are
 * actually needed, which is the common case (per-chunk `download()`).
 */
export async function readCrossChunkLinks(
  options: CrossChunkLinksReaderOptions,
  signal: AbortSignal,
): Promise<CrossChunkLinksTable | undefined> {
  const { kvStoreRead, kvStoreList } = options;
  return readCrossChunkLinksImpl(
    options,
    signal,
    () => true,
    (kBase, ndim) =>
      listPopulatedShardCoords(kvStoreList!, `${kBase}/c`, ndim, signal),
    (shardKey) => kvStoreRead(shardKey, signal),
  );
}

/**
 * Fetch + decode only the ``cross_chunk_links/<delta>/`` records whose
 * sorted-chunks tuple includes ``targetChunkCoords`` — the actual need
 * of one chunk's `download()`.  ``caches`` should be created once per
 * ``(store, delta)`` via {@link createCrossChunkLinksCaches} and reused
 * across every chunk query for that pair, so repeated queries amortize
 * shard discovery/fetch instead of re-paying it (and never decode more
 * than the records the caller asked for).
 *
 * NOTE: a chunk's linked partners are not guaranteed to be spatially
 * adjacent (see {@link CrossChunkLinksCaches}'s docstring), so this
 * still discovers every shard the way {@link readCrossChunkLinks} does
 * — the savings are in skipping decode/object-allocation for
 * non-matching cells and in the two caches, not in visiting fewer
 * shards for a single isolated query.  The caches are what make repeat
 * queries (i.e. every chunk actually rendered) cheap.
 */
export async function readCrossChunkLinksForChunk(
  options: CrossChunkLinksReaderOptions,
  targetChunkCoords: readonly number[],
  caches: CrossChunkLinksCaches,
  signal: AbortSignal,
): Promise<CrossChunkLinksTable | undefined> {
  const { kvStoreRead, kvStoreList } = options;
  return readCrossChunkLinksImpl(
    options,
    signal,
    (sortedChunks) =>
      sortedChunks.some((c) => arraysEqual(c, targetChunkCoords)),
    async (kBase, ndim) => {
      let coords = caches.shardCoords.get(kBase);
      if (coords === undefined) {
        coords = await listPopulatedShardCoords(
          kvStoreList!,
          `${kBase}/c`,
          ndim,
          signal,
        );
        caches.shardCoords.set(kBase, coords);
      }
      return coords;
    },
    async (shardKey) => {
      let bytes = caches.shardBytes.get(shardKey);
      if (bytes === undefined) {
        const fetched = await kvStoreRead(shardKey, signal);
        if (fetched !== undefined) {
          bytes = fetched;
          caches.shardBytes.set(shardKey, bytes);
        }
      }
      return bytes;
    },
  );
}
