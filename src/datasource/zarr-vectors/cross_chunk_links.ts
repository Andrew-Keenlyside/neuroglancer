/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Read and decode the ``cross_chunk_links/<delta>/`` tree emitted by
 * zarr-vectors-py 0.8.1 (``main``) writers under the ``"explicit_links"``
 * cross-chunk strategy.
 *
 * **0.8.1 flat-cell-key layout** (see
 * ``zarr_vectors.core.arrays.write_cross_chunk_link_cells`` /
 * ``zarr_vectors.spatial.boundary.canonical_sort`` /
 * ``zarr_vectors.core.paths.format_cell_key``): each cross-chunk record
 * is a list of ``link_width`` ``(chunk_coords, vertex_idx)`` endpoints.
 * The record's endpoints are stored under a cell keyed by the dotted
 * concatenation of the endpoints' chunk coordinates —
 * ``cross_chunk_links/<delta>/<cell_key>`` where ``cell_key`` has
 * ``link_width * sid_ndim`` dotted integer components (``L`` chunk-coord
 * tuples of ``sid_ndim`` ints each, back to back). Each such path is
 * itself a standard (non-sharded) zarr v3 array — the same "one tiny
 * ``bytes``+``zstd``-coded array per key" convention used everywhere else
 * in this format for small per-key blobs (see e.g.
 * ``vertex_fragments``/``fragment_attributes`` chunk downloads in
 * ``skeleton_chunk_download.ts``) — at ``<cell_key>/c/0``.
 *
 * For an **undirected** (``store="canonical"``) family the cell key is
 * the endpoints' chunks in canonical (lexicographic) order. For a
 * **directed** family the cell key is the *literal input order* — endpoint
 * 0 is the "owning" side, so e.g. a streamline's predecessor→successor
 * transition between chunk A and chunk B files under ``A.B``, and the
 * reverse transition under ``B.A`` (see
 * ``zarr_vectors.spatial.boundary._cell_placements``).
 *
 * A cell's decompressed payload is an "inline-header ragged blob"
 * (``zarr_vectors.encoding.ragged.encode_ragged_blob``): an ``int64``
 * count ``k`` of records, then ``k`` ``int64`` byte-offsets into the data
 * section that follows, one per record (records are written back-to-back
 * so offsets increase by exactly ``(1 + link_width) * 8`` bytes each, but
 * this reader honors the offsets rather than assuming that). Each
 * record is ``(1 + link_width)`` ``int64`` little-endian values:
 * ``[perm_idx, vi_0, ..., vi_{link_width-1}]``.
 *
 * ``perm_idx`` is the *Lehmer code* of the permutation that maps the
 * cell's canonical (stored) endpoint order back to the record's original
 * input order — see ``zarr_vectors.spatial.boundary._lehmer_encode`` /
 * ``_lehmer_decode`` / ``apply_perm_inverse``. This applies
 * unconditionally (both directed and undirected families): for a
 * directed family ``perm_idx`` is always 0 (the identity permutation,
 * since the writer stores endpoints in their literal input order
 * already); for an undirected family it recovers the pre-canonicalization
 * order (needed e.g. for mesh-face winding). Concretely: let ``sorted[i]
 * = (cellChunks[i], vi_i)``; decode ``sortedIdx = lehmerDecode(perm_idx,
 * link_width)``; then ``original[sortedIdx[i]] = sorted[i]``.
 *
 * The cell's zstd frame wraps the ragged blob directly — there is no
 * vlen-bytes framing layer for cross-chunk-link cells (unlike the
 * sharded per-chunk arrays read in ``skeleton_chunk_download.ts``, which
 * do use vlen-bytes since they're elements of a ``variable_length_bytes``
 * sharded array; a CCL cell is a plain scalar ``bytes``-array chunk).
 *
 * The group's own ``cross_chunk_links/<delta>/zarr.json`` carries
 * ``link_width``, ``sid_ndim``, ``directed``, and ``store``
 * (``"canonical"`` or ``"duplicate"``) attributes, and no per-``K``
 * sub-arrays (the pre-0.8.1 ``sharded_v1`` kK-array layout this reader
 * used to support is gone; this reader targets 0.8.1 stores only).
 *
 * **Two physical layouts** (discriminated by the ``layout`` attribute on
 * the family's ``zarr.json``; this reader dispatches PER-FAMILY, so a
 * single store may mix them across levels):
 *
 * 1. ``"flat_cells"`` (default; also the meaning when ``layout`` is
 *    ABSENT): ``cross_chunk_links/<delta>/`` is a GROUP and each populated
 *    cell is its own tiny non-sharded ``bytes``+``zstd`` array at
 *    ``<cell_key>/c/0`` (everything described above). Cells are discovered
 *    by listing the group (``kvStoreList``).
 *
 * 2. ``"packed_sharded"`` (opt-in): ``cross_chunk_links/<delta>/`` is a
 *    SINGLE native-sharded 1-D ``variable_length_bytes`` Zarr array of
 *    shape ``(N,)`` (``N`` = number of populated cells), read with the
 *    same machinery as the per-chunk arrays in ``sharded_array.ts``. Its
 *    ``zarr.json`` attributes additionally carry ``layout:
 *    "packed_sharded"`` and ``cell_keys`` — a sorted JSON list of the
 *    ``<cell_key>`` strings, where flat array index ``i`` holds the cell
 *    for ``cell_keys[i]`` (so cells are enumerated from ``cell_keys``, no
 *    ``kvStoreList``). Element ``i``'s DECODED bytes (post vlen-unframe +
 *    zstd) are the EXACT SAME ragged-int64 blob a flat cell's decoded
 *    payload holds — only the storage location differs, so every layout-
 *    independent decoder below is reused unchanged.
 */

import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import {
  parseShardedArrayMeta,
  readShardedArrayChunk,
  type ShardedArrayMeta,
  type ShardedArrayReadContext,
  type ShardedArrayReadRange,
  type ShardIndexCache,
} from "#src/datasource/zarr-vectors/sharded_array.js";
import { arraysEqual } from "#src/util/array.js";
import { asyncMemoize, type AsyncMemoize } from "#src/util/memoize.js";

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
   * Returns ``undefined`` for missing keys. **Must return raw
   * (non-decompressed) bytes** — this module manages zstd
   * decompression itself; a magic-byte-sniffing auto-decompressor (as
   * used for plain vertex/attribute reads elsewhere in this datasource)
   * would work here too since each cell is its own independent zstd
   * frame, but this module doesn't rely on that behavior.
   */
  readonly kvStoreRead: (
    subpath: string,
    signal: AbortSignal,
  ) => Promise<Uint8Array | undefined>;
  /**
   * Lists the immediate children of one directory-like prefix
   * (relative to the resolution-level base URL). Required to discover
   * which ``<cell_key>`` entries exist under
   * ``cross_chunk_links/<delta>/`` — a flat namespace has no way to
   * enumerate written cells other than listing the store. Omit only in
   * contexts (e.g. tests, or {@link readCrossChunkLinksForOwnedChunks}'s
   * go-direct path) that don't need whole-table enumeration.
   */
  readonly kvStoreList?: (
    prefix: string,
    signal: AbortSignal,
  ) => Promise<CrossChunkLinksListResult>;
  /**
   * Byte-range reader (rooted like {@link kvStoreRead}, returning RAW
   * non-decompressed bytes) used ONLY by the ``"packed_sharded"`` layout
   * to read one element of the native-sharded array via
   * {@link readShardedArrayChunk}. Omit for ``"flat_cells"`` families
   * (never consulted there). When a ``"packed_sharded"`` family is
   * encountered and this is absent, the reader falls back to fetching the
   * whole shard file via {@link kvStoreRead} and slicing it in-process
   * (correct but wastes bandwidth — production call sites should thread a
   * real byte-range reader through).
   */
  readonly kvStoreReadRange?: ShardedArrayReadRange;
  /** Level delta; 0 for intra-level, ±N for cross-level pyramids. */
  readonly delta?: number;
}

/**
 * A whole-level `cross_chunk_links/<delta>/` table can hold tens of
 * millions of records — decoding all of them into `CrossChunkLinksTable`'s
 * nested-object representation costs many times the packed on-disk size
 * once V8's per-object/per-array overhead is counted (record → endpoints
 * array → endpoint objects → chunkCoords arrays). Worse, holding that
 * table in a plain field (as callers here do) makes it invisible to
 * neuroglancer's `ChunkState`-tracked GPU/system memory budget — it's
 * never evicted under memory pressure.
 *
 * `readCrossChunkLinksForChunk` decodes only the records touching one
 * specific chunk (the actual per-`download()` need — see
 * `skeleton_backend.ts`), reusing caches so repeated chunk queries
 * amortize the discovery/fetch cost without re-decoding the world:
 *
 * - `headers`: resolved group metadata (`link_width`/`sid_ndim`/
 *   `directed`/`store`), keyed by the `cross_chunk_links/<delta>` base
 *   path — a per-level constant, fetched once.
 * - `cellKeys`: the full list of populated `<cell_key>` names under one
 *   `cross_chunk_links/<delta>/`, keyed by that same base path. The 0.8.1
 *   layout has no per-chunk discovery query (unlike the pre-0.8.1
 *   sharded kK layout's shard-coordinate pruning) — one flat listing per
 *   level, cached, means a repeat query for a *different* target chunk
 *   at the same level costs zero additional network requests (only a
 *   client-side filter over the already-fetched key list).
 * - `cellBytes`: raw (still zstd-compressed) per-cell byte blobs, bounded
 *   by {@link BoundedByteCache} so a huge dataset's cell bytes don't
 *   accumulate without limit.
 *
 * Values in `headers` and `cellKeys` are {@link AsyncMemoize}s, not bare
 * `Promise`s, so that concurrent queries for the same level (e.g. every
 * visible chunk's `download()` firing at once) share one resolve/list
 * call instead of each independently repeating it — AND so that one
 * caller's chunk getting cancelled (panned out of view, evicted for a
 * higher-priority download) doesn't reject the shared in-flight
 * promise for every OTHER still-active caller awaiting the same entry.
 * A bare `Promise` cached from whichever caller's `signal` happened to
 * start the fetch would tie the shared work to THAT caller's
 * cancellation alone; `asyncMemoize` instead routes it through an
 * internal `SharedAbortController` that only aborts once EVERY
 * registered caller has walked away (see `sharded_array.ts`'s
 * `ShardIndexCache`, which has the identical shape for the same
 * reason).
 */
export interface CrossChunkLinksCaches {
  /** `"cross_chunk_links/<delta>"` -> resolved group metadata. */
  readonly headers: Map<string, AsyncMemoize<ResolvedCclHeader | undefined>>;
  /** `"cross_chunk_links/<delta>"` -> every populated `<cell_key>` name at that level. */
  readonly cellKeys: Map<string, AsyncMemoize<string[]>>;
  /** `"cross_chunk_links/<delta>/<cell_key>"` -> raw (compressed) cell bytes. */
  readonly cellBytes: BoundedByteCache;
  /**
   * ``"packed_sharded"`` layout only: `"cross_chunk_links/<delta>"` ->
   * that family's sharded-array {@link ShardIndexCache}, so element reads
   * that land in the same shard reuse one shard-index fetch. One entry per
   * family; created lazily on first packed read.
   */
  readonly shardIndexCaches: Map<string, ShardIndexCache>;
  /**
   * ``"packed_sharded"`` layout only: `"cross_chunk_links/<delta>"` ->
   * that family's `<cell_key>` -> flat-array-index lookup, built once from
   * the header's ``cell_keys`` and reused across every go-direct probe.
   */
  readonly cellKeyIndexMaps: Map<string, Map<string, number>>;
}

/** Fresh, empty caches for one `(store, delta)` — share across every chunk query for that pair. */
export function createCrossChunkLinksCaches(
  maxCellBytesCached = 256 * 1024 * 1024,
): CrossChunkLinksCaches {
  return {
    headers: new Map(),
    cellKeys: new Map(),
    cellBytes: new BoundedByteCache(maxCellBytesCached),
    shardIndexCaches: new Map(),
    cellKeyIndexMaps: new Map(),
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

/**
 * Decode one "inline-header ragged blob"
 * (``zarr_vectors.encoding.ragged.encode_ragged_blob``) of fixed-stride
 * ``int64`` records: an ``int64`` count ``k``, then ``k`` ``int64`` byte
 * offsets into the data section that follows, then the data itself.
 * Each record occupies ``[offsets[i], offsets[i+1] or end)`` and must be
 * exactly ``ncols * 8`` bytes. Returns one `BigInt64Array` of length
 * `ncols` per record. Exported so tests can drive the decoder with
 * hand-crafted byte fixtures.
 */
export function decodeRaggedBlobInt64Records(
  bytes: Uint8Array,
  ncols: number,
): BigInt64Array[] {
  if (bytes.byteLength < 8) return [];
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const k = Number(dv.getBigInt64(0, true));
  if (k === 0) return [];
  const headerLen = 8 * (1 + k);
  if (bytes.byteLength < headerLen) {
    throw new Error(
      `cross_chunk_links: ragged blob header declares ${k} records ` +
        `(needs ${headerLen} header bytes) but buffer is only ` +
        `${bytes.byteLength} bytes`,
    );
  }
  const offsets = new Array<number>(k);
  for (let i = 0; i < k; ++i) {
    offsets[i] = Number(dv.getBigInt64(8 + i * 8, true));
  }
  const dataStart = headerLen;
  const dataLen = bytes.byteLength - dataStart;
  const rows: BigInt64Array[] = new Array(k);
  for (let i = 0; i < k; ++i) {
    const start = offsets[i];
    const end = i + 1 < k ? offsets[i + 1] : dataLen;
    const rowByteLength = end - start;
    if (rowByteLength !== ncols * 8) {
      throw new Error(
        `cross_chunk_links: ragged blob record ${i} is ${rowByteLength} ` +
          `bytes; expected ${ncols * 8} (ncols=${ncols})`,
      );
    }
    const row = new BigInt64Array(ncols);
    const rowDv = new DataView(
      bytes.buffer,
      bytes.byteOffset + dataStart + start,
      rowByteLength,
    );
    for (let c = 0; c < ncols; ++c) {
      row[c] = rowDv.getBigInt64(c * 8, true);
    }
    rows[i] = row;
  }
  return rows;
}

/**
 * Decode the Lehmer code `code` of a permutation of `range(L)` back into
 * that permutation — the inverse of
 * ``zarr_vectors.spatial.boundary._lehmer_encode``. `L` is small in
 * practice (`link_width`, typically 2-4), so plain `number` arithmetic
 * (not `BigInt`) is used throughout; `code` must already fit in a safe
 * integer (guaranteed since `code < L!`, which is tiny for realistic
 * `L`).
 */
export function lehmerDecode(code: number, L: number): number[] {
  if (L < 1) {
    throw new Error(`cross_chunk_links: lehmerDecode requires L >= 1, got ${L}`);
  }
  const available: number[] = [];
  for (let i = 0; i < L; ++i) available.push(i);
  const perm = new Array<number>(L);
  let fact = 1;
  for (let i = 2; i <= L; ++i) fact *= i; // L!
  let remaining = code;
  if (remaining < 0 || remaining >= fact) {
    throw new Error(
      `cross_chunk_links: perm_idx ${code} out of range [0, ${fact}) for L=${L}`,
    );
  }
  for (let i = 0; i < L; ++i) {
    fact = fact / (L - i);
    const idx = Math.floor(remaining / fact);
    remaining = remaining % fact;
    perm[i] = available[idx];
    available.splice(idx, 1);
  }
  return perm;
}

/**
 * Decode one cell's decompressed payload (the ragged-blob-of-fixed-stride-
 * records format described in this module's docstring) into structured
 * records. ``cellChunks`` is the cell's `link_width` chunk-coordinate
 * tuples in the exact order encoded in the cell's key (canonical order
 * for an undirected family, literal input order for a directed one — see
 * this module's docstring). Exported so tests can drive the decoder with
 * hand-crafted byte fixtures.
 */
export function decodeCrossChunkLinkCell(
  payload: Uint8Array,
  cellChunks: readonly (readonly number[])[],
  linkWidth: number,
): CrossChunkLinkRecord[] {
  if (linkWidth < 1) {
    throw new Error(
      `cross_chunk_links: link_width must be >= 1; got ${linkWidth}`,
    );
  }
  if (cellChunks.length !== linkWidth) {
    throw new Error(
      `cross_chunk_links: cell has ${cellChunks.length} chunks; expected ` +
        `link_width=${linkWidth}`,
    );
  }
  const recordLen = 1 + linkWidth;
  const rows = decodeRaggedBlobInt64Records(payload, recordLen);
  const records: CrossChunkLinkRecord[] = [];
  for (const row of rows) {
    const permIdx = Number(row[0]);
    const sortedEndpoints: CrossChunkLinkEndpoint[] = new Array(linkWidth);
    for (let i = 0; i < linkWidth; ++i) {
      sortedEndpoints[i] = {
        chunkCoords: cellChunks[i].slice(),
        vertexIndex: Number(row[1 + i]),
      };
    }
    const sortedIdx = lehmerDecode(permIdx, linkWidth);
    const endpoints = new Array<CrossChunkLinkEndpoint>(linkWidth);
    for (let i = 0; i < linkWidth; ++i) {
      endpoints[sortedIdx[i]] = sortedEndpoints[i];
    }
    records.push({ endpoints });
  }
  return records;
}

/** Concatenate `link_width` pre-ordered chunk-coord tuples into a dotted cell key. */
function formatCellKey(chunks: readonly (readonly number[])[]): string {
  const parts: string[] = [];
  for (const c of chunks) {
    for (const x of c) parts.push(String(x));
  }
  return parts.join(".");
}

/** Inverse of {@link formatCellKey}: split a cell key into `linkWidth` chunk-coord tuples. */
function parseCellKey(
  key: string,
  sidNdim: number,
  linkWidth: number,
): number[][] {
  const parts = key.split(".");
  const expected = sidNdim * linkWidth;
  if (parts.length !== expected) {
    throw new Error(
      `cross_chunk_links: cell key ${JSON.stringify(key)} has ` +
        `${parts.length} components; expected ${expected} ` +
        `(sid_ndim=${sidNdim} * link_width=${linkWidth})`,
    );
  }
  const nums = parts.map(Number);
  const chunks: number[][] = [];
  for (let i = 0; i < linkWidth; ++i) {
    chunks.push(nums.slice(i * sidNdim, (i + 1) * sidNdim));
  }
  return chunks;
}

/** Resolved `cross_chunk_links/<delta>` family metadata. `undefined` when the family is absent. */
interface ResolvedCclHeader {
  readonly linkWidth: number;
  readonly sidNdim: number;
  readonly directed: boolean;
  readonly store: "canonical" | "duplicate";
  /** Physical layout (defaults to `"flat_cells"` when the `layout` attribute is absent). */
  readonly layout: "flat_cells" | "packed_sharded";
  /**
   * ``"packed_sharded"`` only: the sorted `<cell_key>` list, where flat
   * array index `i` holds `cellKeys[i]`. `undefined` for `"flat_cells"`.
   */
  readonly cellKeys?: string[];
  /**
   * ``"packed_sharded"`` only: parsed native-sharded-array metadata for
   * the single ``cross_chunk_links/<delta>`` array. `undefined` for
   * `"flat_cells"`.
   */
  readonly shardedMeta?: ShardedArrayMeta;
}

async function resolveCclHeaderUncached(
  options: CrossChunkLinksReaderOptions,
  signal: AbortSignal,
): Promise<ResolvedCclHeader | undefined> {
  const { kvStoreRead, delta = 0 } = options;
  const base = `cross_chunk_links/${delta}`;
  const groupMetaBytes = await kvStoreRead(`${base}/zarr.json`, signal);
  if (groupMetaBytes === undefined) return undefined;
  let groupMeta: any;
  try {
    groupMeta = JSON.parse(new TextDecoder().decode(groupMetaBytes));
  } catch (e) {
    throw new Error(`${base}/zarr.json: invalid JSON: ${(e as Error).message}`);
  }
  const attrs = groupMeta?.attributes;
  if (attrs === undefined) {
    throw new Error(`${base}/zarr.json: missing 'attributes' object`);
  }
  const linkWidth = Number(attrs.link_width);
  const sidNdim = Number(attrs.sid_ndim);
  if (!Number.isInteger(linkWidth) || linkWidth < 1) {
    throw new Error(`${base}: invalid link_width ${attrs.link_width}`);
  }
  if (!Number.isInteger(sidNdim) || sidNdim < 1) {
    throw new Error(`${base}: invalid sid_ndim ${attrs.sid_ndim}`);
  }
  const directed = attrs.directed === true;
  const store = attrs.store === "duplicate" ? "duplicate" : "canonical";
  // `layout` absent => the original flat-cell layout (back-compat).
  const layout = attrs.layout === "packed_sharded" ? "packed_sharded" : "flat_cells";
  if (layout === "flat_cells") {
    return { linkWidth, sidNdim, directed, store, layout };
  }
  // Packed: the family's own `zarr.json` IS the sharded array's `zarr.json`
  // (the group node collapses into a single array), so reuse the same
  // bytes to parse the sharded metadata and read the `cell_keys` list.
  const cellKeysAttr = attrs.cell_keys;
  if (!Array.isArray(cellKeysAttr)) {
    throw new Error(
      `${base}/zarr.json: layout="packed_sharded" requires a 'cell_keys' ` +
        `array attribute, got ${JSON.stringify(cellKeysAttr)}`,
    );
  }
  const cellKeys = cellKeysAttr.map(String);
  const shardedMeta = parseShardedArrayMeta(groupMetaBytes, base);
  if (shardedMeta.gridShape.length !== 1) {
    throw new Error(
      `${base}/zarr.json: layout="packed_sharded" expects a 1-D array, got ` +
        `shape ${JSON.stringify(shardedMeta.gridShape)}`,
    );
  }
  if (shardedMeta.gridShape[0] !== cellKeys.length) {
    throw new Error(
      `${base}/zarr.json: layout="packed_sharded" array length ` +
        `${shardedMeta.gridShape[0]} != cell_keys length ${cellKeys.length}`,
    );
  }
  return { linkWidth, sidNdim, directed, store, layout, cellKeys, shardedMeta };
}

/**
 * Cached wrapper around {@link resolveCclHeaderUncached}: the group
 * ``zarr.json`` read is a per-level constant, so a caller that queries
 * many chunks/objects at one level resolves it ONCE instead of
 * re-fetching on every query. `cache` is keyed by the
 * `cross_chunk_links/<delta>` base path; omit it for a one-shot read.
 */
async function resolveCclHeader(
  options: CrossChunkLinksReaderOptions,
  signal: AbortSignal,
  cache?: Map<string, AsyncMemoize<ResolvedCclHeader | undefined>>,
): Promise<ResolvedCclHeader | undefined> {
  if (cache === undefined) return resolveCclHeaderUncached(options, signal);
  const base = `cross_chunk_links/${options.delta ?? 0}`;
  let memoized = cache.get(base);
  if (memoized === undefined) {
    memoized = asyncMemoize(({ signal }) =>
      resolveCclHeaderUncached(options, signal),
    );
    cache.set(base, memoized);
  }
  return memoized({ signal });
}

/**
 * List every populated `<cell_key>` name directly under
 * `cross_chunk_links/<delta>/` — a single flat `kvStoreList` call (each
 * cell_key is its own top-level directory, unlike the pre-0.8.1 sharded
 * layout's recursive per-axis shard-coordinate tree). Cached per level
 * when `cache` is supplied.
 */
async function listAllCellKeys(
  options: CrossChunkLinksReaderOptions,
  signal: AbortSignal,
  cache?: Map<string, AsyncMemoize<string[]>>,
): Promise<string[]> {
  const { kvStoreList, delta = 0 } = options;
  const base = `cross_chunk_links/${delta}`;
  const doList = async (listSignal: AbortSignal): Promise<string[]> => {
    if (kvStoreList === undefined) {
      throw new Error(
        `${base}: reading a populated cross_chunk_links table requires ` +
          `a kvStoreList callback to discover its cells`,
      );
    }
    const { directories } = await kvStoreList(`${base}/`, listSignal);
    return directories.filter((d) => d !== "");
  };
  if (cache === undefined) return doList(signal);
  let memoized = cache.get(base);
  if (memoized === undefined) {
    memoized = asyncMemoize(({ signal }) => doList(signal));
    cache.set(base, memoized);
  }
  return memoized({ signal });
}

async function decodeCellPayload(
  compressedBytes: Uint8Array,
  signal: AbortSignal,
): Promise<Uint8Array> {
  // `requestAsyncComputation` transfers its buffer argument (postMessage
  // transfer semantics) — copy into a freshly-allocated, standalone
  // buffer first so a cached/shared reference elsewhere isn't detached.
  const owned = new Uint8Array(compressedBytes);
  return await requestAsyncComputation(
    decodeZstd,
    signal,
    [owned.buffer as ArrayBuffer],
    owned as Uint8Array<ArrayBuffer>,
  );
}

/**
 * Byte-range reader used when a ``"packed_sharded"`` family is read
 * WITHOUT an `options.kvStoreReadRange`: fetch the whole shard file via
 * `kvStoreRead` and slice the requested range in-process. Correct but
 * wasteful (re-fetches the whole shard per range read) — production call
 * sites thread a real byte-range reader through and never hit this path.
 */
function wholeShardFallbackReadRange(
  kvStoreRead: CrossChunkLinksReaderOptions["kvStoreRead"],
): ShardedArrayReadRange {
  return async (subpath, byteRange, signal) => {
    const whole = await kvStoreRead(subpath, signal);
    if (whole === undefined) return undefined;
    if ("suffixLength" in byteRange) {
      const start = Math.max(0, whole.byteLength - byteRange.suffixLength);
      return whole.subarray(start);
    }
    return whole.subarray(
      byteRange.offset,
      byteRange.offset + byteRange.length,
    );
  };
}

/**
 * Build the {@link ShardedArrayReadContext} for a ``"packed_sharded"``
 * family — the array path is the family base itself, and the byte-range
 * reader is the caller-provided one (or the whole-shard fallback).
 */
function packedReadContext(
  options: CrossChunkLinksReaderOptions,
  header: ResolvedCclHeader,
  base: string,
): ShardedArrayReadContext {
  return {
    kvStoreReadRange:
      options.kvStoreReadRange ?? wholeShardFallbackReadRange(options.kvStoreRead),
    arrayPath: base,
    meta: header.shardedMeta!,
  };
}

/** Get (or lazily create) the per-family {@link ShardIndexCache} for one base path. */
function getShardIndexCache(
  caches: CrossChunkLinksCaches,
  base: string,
): ShardIndexCache {
  let cache = caches.shardIndexCaches.get(base);
  if (cache === undefined) {
    cache = new Map();
    caches.shardIndexCaches.set(base, cache);
  }
  return cache;
}

/** Get (or lazily build) the per-family `<cell_key>` -> flat-index map for a packed family. */
function getCellKeyIndexMap(
  caches: CrossChunkLinksCaches,
  base: string,
  cellKeys: readonly string[],
): Map<string, number> {
  let map = caches.cellKeyIndexMaps.get(base);
  if (map === undefined) {
    map = new Map();
    for (let i = 0; i < cellKeys.length; ++i) map.set(cellKeys[i], i);
    caches.cellKeyIndexMaps.set(base, map);
  }
  return map;
}

/**
 * Shared walk over every populated cell in `cross_chunk_links/<delta>/`,
 * decoding only the ones `shouldDecodeCell` accepts (checked cheaply from
 * the cell's key-derived chunk tuple, before paying for decompression).
 * Used by both {@link readCrossChunkLinks} (whole table, no caching) and
 * {@link readCrossChunkLinksForChunk} (one chunk's records, cached).
 * Dispatches on the family's stored `layout`.
 */
async function readCrossChunkLinksImpl(
  options: CrossChunkLinksReaderOptions,
  signal: AbortSignal,
  shouldDecodeCell: (cellChunks: readonly (readonly number[])[]) => boolean,
  headerCache?: Map<string, AsyncMemoize<ResolvedCclHeader | undefined>>,
  cellKeysCache?: Map<string, AsyncMemoize<string[]>>,
  cellBytesCache?: BoundedByteCache,
  shardIndexCache?: ShardIndexCache,
): Promise<CrossChunkLinksTable | undefined> {
  const { kvStoreRead, delta = 0 } = options;
  const base = `cross_chunk_links/${delta}`;
  const header = await resolveCclHeader(options, signal, headerCache);
  if (header === undefined) return undefined;
  const { linkWidth, sidNdim } = header;

  if (header.layout === "packed_sharded") {
    // Cells are enumerated from the header's sorted `cell_keys` (index `i`
    // holds `cell_keys[i]`) — no listing needed. A locally-created shard-
    // index cache (when none is supplied, e.g. the whole-table read) keeps
    // the single shard-index fetch amortized across every element.
    const idxCache = shardIndexCache ?? (new Map() as ShardIndexCache);
    const ctx = packedReadContext(options, header, base);
    const keys = header.cellKeys!;
    const records: CrossChunkLinkRecord[] = [];
    for (let i = 0; i < keys.length; ++i) {
      const cellChunks = parseCellKey(keys[i], sidNdim, linkWidth);
      if (!shouldDecodeCell(cellChunks)) continue;
      const payload = await readShardedArrayChunk(ctx, [i], signal, idxCache);
      if (payload === undefined) continue;
      for (const record of decodeCrossChunkLinkCell(
        payload,
        cellChunks,
        linkWidth,
      )) {
        records.push(record);
      }
    }
    return { linkWidth, sidNdim, records };
  }

  const cellKeys = await listAllCellKeys(options, signal, cellKeysCache);
  const records: CrossChunkLinkRecord[] = [];
  for (const cellKey of cellKeys) {
    const cellChunks = parseCellKey(cellKey, sidNdim, linkWidth);
    if (!shouldDecodeCell(cellChunks)) continue;
    const cellPath = `${base}/${cellKey}`;
    let bytes = cellBytesCache?.get(cellPath);
    if (bytes === undefined) {
      const fetched = await kvStoreRead(`${cellPath}/c/0`, signal);
      if (fetched === undefined) continue;
      bytes = fetched;
      cellBytesCache?.set(cellPath, bytes);
    }
    const payload = await decodeCellPayload(bytes, signal);
    // Loop rather than `records.push(...cellRecords)`: a single cell can
    // (in principle) hold more than V8's ~65,536-argument call limit on
    // large datasets, and spreading into a call throws past that limit.
    for (const record of decodeCrossChunkLinkCell(
      payload,
      cellChunks,
      linkWidth,
    )) {
      records.push(record);
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
  return readCrossChunkLinksImpl(options, signal, () => true);
}

/**
 * Fetch + decode only the ``cross_chunk_links/<delta>/`` records whose
 * chunk tuple includes ``targetChunkCoords`` — the actual need of one
 * chunk's `download()`. ``caches`` should be created once per
 * ``(store, delta)`` via {@link createCrossChunkLinksCaches} and reused
 * across every chunk query for that pair, so repeated queries amortize
 * the one-time cell-key listing instead of re-listing on every call (and
 * never decode more than the records the caller asked for).
 *
 * NOTE: a chunk's linked partners are not guaranteed to be spatially
 * adjacent, and the 0.8.1 flat-cell-key layout has no per-chunk discovery
 * query (unlike the pre-0.8.1 sharded kK layout's shard-coordinate
 * pruning) — every query walks the level's full (cached) cell-key list
 * and filters client-side.
 */
export async function readCrossChunkLinksForChunk(
  options: CrossChunkLinksReaderOptions,
  targetChunkCoords: readonly number[],
  caches: CrossChunkLinksCaches,
  signal: AbortSignal,
): Promise<CrossChunkLinksTable | undefined> {
  const base = `cross_chunk_links/${options.delta ?? 0}`;
  return readCrossChunkLinksImpl(
    options,
    signal,
    (cellChunks) => cellChunks.some((c) => arraysEqual(c, targetChunkCoords)),
    caches.headers,
    caches.cellKeys,
    caches.cellBytes,
    getShardIndexCache(caches, base),
  );
}

/** Lexicographic compare of two chunk-coordinate tuples. */
function lexCompareChunk(
  a: readonly number[],
  b: readonly number[],
): number {
  for (let i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

/**
 * Sorted, distinct K-combinations of `chunks` (each combination's chunks
 * ordered lexicographically). Used to enumerate the candidate cross-
 * chunk-link cells whose endpoints are ALL owned by one object.
 */
function sortedKCombinations(
  chunks: readonly (readonly number[])[],
  K: number,
): number[][][] {
  // Sort + dedup so a chunk revisited by the manifest doesn't spawn
  // duplicate combinations.
  const sorted = [...chunks].sort(lexCompareChunk);
  const uniq: (readonly number[])[] = [];
  for (const c of sorted) {
    if (uniq.length === 0 || lexCompareChunk(uniq[uniq.length - 1], c) !== 0) {
      uniq.push(c);
    }
  }
  const out: number[][][] = [];
  const combo: (readonly number[])[] = [];
  const rec = (start: number): void => {
    if (combo.length === K) {
      out.push(combo.map((c) => [...c]));
      return;
    }
    for (let i = start; i < uniq.length; ++i) {
      combo.push(uniq[i]);
      rec(i + 1);
      combo.pop();
    }
  };
  rec(0);
  return out;
}

/** Every permutation of `items` (small `items.length` only — used for `link_width` <= ~4). */
function permutations<T>(items: readonly T[]): T[][] {
  if (items.length <= 1) return [items.slice()];
  const out: T[][] = [];
  for (let i = 0; i < items.length; ++i) {
    const rest = [...items.slice(0, i), ...items.slice(i + 1)];
    for (const p of permutations(rest)) out.push([items[i], ...p]);
  }
  return out;
}

/**
 * "Go-direct" cross-chunk-links read for pass-2: given the set of chunks a
 * single object owns, read ONLY the records whose endpoints are all among
 * those chunks — computing the exact candidate cell keys directly rather
 * than listing and filtering the whole level's cell-key table.
 *
 * Contrast with {@link readCrossChunkLinksForChunk} (used by pass-1's
 * ghost fetch), which needs one level-wide listing + client-side filter
 * (the 0.8.1 flat-cell-key layout has no cheaper way to answer "records
 * touching chunk X, any partner"). Here the partner set is bounded to the
 * object's own owned chunks, so candidate cell keys can be constructed
 * directly with zero listing: they are exactly the sorted K-combinations
 * of the owned chunk set (K = `link_width`), formatted as cell keys. For
 * a **directed** family the writer keys by literal input order (not
 * canonical order), and a reader can't predict which of the `K!`
 * orderings of a given combination was actually written, so every
 * ordering is tried as a candidate key — cheap since `K = link_width` is
 * small in practice (2 for streamlines' cross-chunk edges).
 *
 * Only `store: "canonical"` families are supported here (the only mode
 * this codebase's writers produce) — a `"duplicate"`-store family uses a
 * different cell-keying scheme entirely (one placement per distinct
 * endpoint chunk, not per K-combination) and would need a different
 * candidate-enumeration strategy; callers with such a store should fall
 * back to {@link readCrossChunkLinksForChunk} per owned chunk instead.
 */
export async function readCrossChunkLinksForOwnedChunks(
  options: CrossChunkLinksReaderOptions,
  ownedChunks: readonly (readonly number[])[],
  caches: CrossChunkLinksCaches,
  signal: AbortSignal,
): Promise<CrossChunkLinksTable | undefined> {
  const { kvStoreRead, delta = 0 } = options;
  if (ownedChunks.length === 0) return undefined;
  const header = await resolveCclHeader(options, signal, caches.headers);
  if (header === undefined) return undefined;
  const { linkWidth, sidNdim, directed, store } = header;
  if (store !== "canonical") {
    throw new Error(
      `cross_chunk_links: readCrossChunkLinksForOwnedChunks only supports ` +
        `store="canonical" families, got ${JSON.stringify(store)}`,
    );
  }
  const base = `cross_chunk_links/${delta}`;
  const ownedKeys = new Set(ownedChunks.map((c) => c.join(",")));
  const combos = sortedKCombinations(ownedChunks, linkWidth);
  const candidateKeys = new Set<string>();
  for (const combo of combos) {
    if (!directed) {
      candidateKeys.add(formatCellKey(combo));
    } else {
      for (const perm of permutations(combo)) {
        candidateKeys.add(formatCellKey(perm));
      }
    }
  }
  const records: CrossChunkLinkRecord[] = [];

  if (header.layout === "packed_sharded") {
    // Same candidate keys, but resolve each to its flat array index via
    // the header's `cell_keys` (built once, cached) and read that element
    // — a candidate absent from the map simply isn't a written cell.
    const cellKeyIndex = getCellKeyIndexMap(caches, base, header.cellKeys!);
    const ctx = packedReadContext(options, header, base);
    const idxCache = getShardIndexCache(caches, base);
    for (const cellKey of candidateKeys) {
      const flatIndex = cellKeyIndex.get(cellKey);
      if (flatIndex === undefined) continue;
      const cellChunks = parseCellKey(cellKey, sidNdim, linkWidth);
      // Defensive: candidates are owned-only by construction, but keep the
      // check cheap and explicit rather than trusting it silently.
      if (!cellChunks.every((c) => ownedKeys.has(c.join(",")))) continue;
      const payload = await readShardedArrayChunk(
        ctx,
        [flatIndex],
        signal,
        idxCache,
      );
      if (payload === undefined) continue;
      for (const record of decodeCrossChunkLinkCell(
        payload,
        cellChunks,
        linkWidth,
      )) {
        records.push(record);
      }
    }
    return { linkWidth, sidNdim, records };
  }

  for (const cellKey of candidateKeys) {
    const cellPath = `${base}/${cellKey}`;
    let bytes = caches.cellBytes.get(cellPath);
    if (bytes === undefined) {
      const fetched = await kvStoreRead(`${cellPath}/c/0`, signal);
      if (fetched === undefined) continue;
      bytes = fetched;
      caches.cellBytes.set(cellPath, bytes);
    }
    const cellChunks = parseCellKey(cellKey, sidNdim, linkWidth);
    // Defensive: candidates are owned-only by construction, but keep the
    // check cheap and explicit rather than trusting it silently.
    if (!cellChunks.every((c) => ownedKeys.has(c.join(",")))) continue;
    const payload = await decodeCellPayload(bytes, signal);
    for (const record of decodeCrossChunkLinkCell(
      payload,
      cellChunks,
      linkWidth,
    )) {
      records.push(record);
    }
  }
  return { linkWidth, sidNdim, records };
}
