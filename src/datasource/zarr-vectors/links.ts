/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Reader for the ZVF 0.9 links family (`links/<delta>/<offsets>/`).
 *
 * Connectivity is one family per resolution level: each `<offsets>` array is a
 * plain rank-D vlen-bytes zarr grid over the chunk grid, one cell per **source**
 * chunk. A record's other endpoint sits in `source + offset`, where `offset`
 * comes from the array's path/metadata — not from the record bytes. That is the
 * structural change from ZVF ≤0.8, whose `cross_chunk_links/<delta>/kK/` sharded
 * layout (and its per-record `ci` index) is gone; this reader replaces it.
 *
 * The output contract ({@link CrossChunkLinksTable} of records with
 * `{chunkCoords, vertexIndex}` endpoints) is unchanged, so the render path
 * (`skeleton_backend.ts` `buildBridgeRequests` / ghost-vertex insertion) is
 * untouched. `endpoint[0]` is the record's source (predecessor); `endpoint[1]`
 * is `source + offset` (successor).
 *
 * Discovery is GET-first: the offsets that exist under `links/<delta>/` are
 * enumerated by listing when the store permits it, else by probing a bounded
 * neighbourhood (this bucket denies object LIST). Each offset array's own
 * `zarr.json` carries a `nonempty_chunks` manifest, so once an array is known
 * only its populated cells are fetched — no listing per cell.
 */

import {
  formatOffsets,
  intraOffsets,
  isIntra,
  type LinkOffset,
  linksGroupPath,
  linksHasPerm,
  linksPath,
  parseOffsets,
} from "#src/datasource/zarr-vectors/links_paths.js";

/** One endpoint of a link: which chunk, and which vertex within it. */
export interface CrossChunkLinkEndpoint {
  /** Length-`sidNdim` chunk-grid coordinates of the endpoint's chunk. */
  readonly chunkCoords: number[];
  /** 0-based vertex index inside that chunk's `vertices/` array. */
  readonly vertexIndex: number;
}

/** One `linkWidth`-arity link record. */
export interface CrossChunkLinkRecord {
  readonly endpoints: CrossChunkLinkEndpoint[];
}

/** Records for one `(level, delta)` pair, in the reader's traversal order. */
export interface CrossChunkLinksTable {
  readonly linkWidth: number;
  readonly sidNdim: number;
  readonly records: CrossChunkLinkRecord[];
}

/**
 * Configuration for the links reader. Lets callers substitute a kvstore
 * reader/lister at the test boundary.
 */
export interface CrossChunkLinksReaderOptions {
  /**
   * Reads one byte blob relative to the resolution-level base URL, returning
   * `undefined` for a missing key. Returns the key's **raw stored bytes**: for
   * a `zarr.json` that is its UTF-8 text; for an `<offsets>/c/...` cell it is
   * the vlen-bytes container (0.9 links arrays declare a `vlen-bytes`-only
   * codec chain, so there is no separate decompression step).
   */
  readonly kvStoreRead: (
    subpath: string,
    signal: AbortSignal,
  ) => Promise<Uint8Array | undefined>;
  /**
   * Lists the immediate children of one directory-like prefix (relative to the
   * level base URL). When available it enumerates every `<offsets>` array under
   * `links/<delta>/` — the correct, unbounded discovery. When omitted (or it
   * throws, e.g. a bucket that denies LIST), the reader falls back to probing a
   * bounded offset neighbourhood by GET.
   */
  readonly kvStoreList?: (
    prefix: string,
    signal: AbortSignal,
  ) => Promise<{ directories: string[]; files: string[] }>;
  /** Level delta; 0 for intra-level, ±N for cross-level pyramids. */
  readonly delta?: number;
  /**
   * Chebyshev radius for the GET-only offset probe when listing is unavailable.
   * The store's real offsets are all radius-1 (`{-1,0,+1}³`), so the default
   * covers it. A larger radius costs `(2r+1)^sidNdim - 1` speculative
   * `zarr.json` GETs per query set.
   */
  readonly fallbackOffsetRadius?: number;
}

const DEFAULT_FALLBACK_OFFSET_RADIUS = 1;

/** Marker written on the `links/<delta>/` family group. */
const LINKS_FAMILY_MARKER = "links_family";
/** Marker written on each `links/<delta>/<offsets>/` array. */
const LINKS_ARRAY_MARKER = "links";

/** zstd frame magic — links arrays must not be compressed (see file header). */
const ZSTD_MAGIC = [0x28, 0xb5, 0x2f, 0xfd];

/** Family-wide policy read from the `links/<delta>/` group. */
interface LinksFamilyMeta {
  readonly linkWidth: number;
  readonly sidNdim: number;
  readonly directed: boolean;
  readonly store: string;
}

/** One resolved `<offsets>` array: its policy plus which cells are populated. */
interface OffsetArrayDescriptor {
  /** The `linkWidth - 1` non-source offsets (parsed). */
  readonly offsets: LinkOffset[];
  /** Path of the array, relative to the level base. */
  readonly arrayPath: string;
  /** Whether rows carry a leading `perm_idx` column. */
  readonly hasPerm: number;
  /** Set of populated source-chunk keys (`"i.j.k"`), or `undefined` if the array
   * did not stamp `nonempty_chunks` (then every cell is probed by GET). */
  readonly nonemptyChunks: Set<string> | undefined;
}

/** Per-delta discovery cache: repeated per-chunk queries reuse the descriptors. */
export interface CrossChunkLinksCaches {
  /** delta → resolved descriptors (or `null` = family absent). */
  readonly byDelta: Map<number, Promise<DeltaDiscovery | null>>;
  /** One-time warnings already emitted, so noise stays bounded. */
  readonly warned: Set<string>;
}

interface DeltaDiscovery {
  readonly family: LinksFamilyMeta;
  readonly descriptors: OffsetArrayDescriptor[];
  /** True when the bounded GET-only fallback was used (may miss long offsets). */
  readonly usedFallback: boolean;
}

export function createCrossChunkLinksCaches(): CrossChunkLinksCaches {
  return { byDelta: new Map(), warned: new Set() };
}

function warnOnce(caches: CrossChunkLinksCaches, key: string, message: string) {
  if (caches.warned.has(key)) return;
  caches.warned.add(key);
  console.warn(message);
}

// ---------------------------------------------------------------------------
// Cell decoding
// ---------------------------------------------------------------------------

/**
 * Decode one numcodecs `vlen-bytes`-framed buffer known to hold exactly one
 * item (the convention for an all-`1`s-shaped zarr sub-chunk). Layout: 4-byte
 * LE item count, then per item a 4-byte LE length followed by that many bytes.
 */
export function decodeVlenBytesSingleItem(buf: Uint8Array): Uint8Array {
  if (buf.byteLength < 8) {
    throw new Error(
      `links: vlen-bytes cell buffer too short (${buf.byteLength} bytes)`,
    );
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const numItems = dv.getUint32(0, true);
  if (numItems !== 1) {
    throw new Error(
      `links: expected exactly 1 vlen-bytes item per cell, got ${numItems}`,
    );
  }
  const length = dv.getUint32(4, true);
  if (8 + length > buf.byteLength) {
    throw new Error(
      `links: vlen-bytes cell buffer truncated (declared ${length}, have ${buf.byteLength - 8})`,
    );
  }
  return buf.subarray(8, 8 + length);
}

/**
 * Decode an inline-header ragged blob (`encode_ragged_blob`) into a flat list
 * of `ncols`-wide int64 rows across all groups. Layout:
 * `[int64 k][int64 groupByteOffset × k][data]`, where each group's slice of
 * `data` is a contiguous `(M_k, ncols)` int64 matrix. Group boundaries carry no
 * meaning for edge rendering, so they are flattened away.
 */
export function decodeRaggedBlobRows(
  blob: Uint8Array,
  ncols: number,
): number[][] {
  if (ncols < 1) throw new Error(`links: ncols must be >= 1, got ${ncols}`);
  if (blob.byteLength < 8) return [];
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const k = Number(dv.getBigInt64(0, true));
  if (k === 0) return [];
  if (k < 0)
    throw new Error(`links: ragged blob has negative group count ${k}`);
  const headerLen = 8 * (1 + k);
  if (headerLen > blob.byteLength) {
    throw new Error(
      `links: ragged blob header (${headerLen} bytes) exceeds cell (${blob.byteLength})`,
    );
  }
  const groupOffsets: number[] = [];
  for (let i = 0; i < k; ++i) {
    groupOffsets.push(Number(dv.getBigInt64(8 + 8 * i, true)));
  }
  const dataLen = blob.byteLength - headerLen;
  const rowBytes = ncols * 8;
  const rows: number[][] = [];
  for (let g = 0; g < k; ++g) {
    const start = groupOffsets[g];
    const end = g + 1 < k ? groupOffsets[g + 1] : dataLen;
    if (start < 0 || end > dataLen || start > end) {
      throw new Error(
        `links: ragged blob group ${g} range [${start},${end}) invalid for data length ${dataLen}`,
      );
    }
    const segLen = end - start;
    if (segLen % rowBytes !== 0) {
      throw new Error(
        `links: ragged blob group ${g} length ${segLen} is not a multiple of one row ` +
          `(${rowBytes} bytes for ncols=${ncols})`,
      );
    }
    const numRows = segLen / rowBytes;
    for (let r = 0; r < numRows; ++r) {
      const rowBase = headerLen + start + r * rowBytes;
      const row = new Array<number>(ncols);
      for (let c = 0; c < ncols; ++c) {
        const v = dv.getBigInt64(rowBase + c * 8, true);
        // Vertex/permutation indices are small (chunk-local); Number is exact.
        if (
          v > BigInt(Number.MAX_SAFE_INTEGER) ||
          v < BigInt(Number.MIN_SAFE_INTEGER)
        ) {
          throw new Error(`links: link index ${v} exceeds safe integer range`);
        }
        row[c] = Number(v);
      }
      rows.push(row);
    }
  }
  return rows;
}

/**
 * Decode one cell of a `links/<delta>/<offsets>/` array into records, given the
 * cell's own source chunk and the array's (parsed) offsets. Endpoint `k`'s
 * chunk is `sourceChunk + offsets[k-1]` (endpoint 0 is the source); its vertex
 * index is column `k` (after the `perm_idx` column when present).
 */
export function decodeLinkCell(
  rawCell: Uint8Array,
  sourceChunk: readonly number[],
  offsets: readonly LinkOffset[],
  linkWidth: number,
  hasPerm: boolean,
): CrossChunkLinkRecord[] {
  if (rawCell.byteLength >= 4 && ZSTD_MAGIC.every((b, i) => rawCell[i] === b)) {
    throw new Error(
      "links: cell is zstd-compressed, but 0.9 links arrays declare a vlen-bytes-only " +
        "codec chain; a compressed links array needs the declared-codec reader (P2).",
    );
  }
  const blob = decodeVlenBytesSingleItem(rawCell);
  const permCol = hasPerm ? 1 : 0;
  const ncols = permCol + linkWidth;
  const rows = decodeRaggedBlobRows(blob, ncols);
  const records: CrossChunkLinkRecord[] = [];
  for (const row of rows) {
    const endpoints: CrossChunkLinkEndpoint[] = [];
    for (let k = 0; k < linkWidth; ++k) {
      // offset for endpoint 0 is zero (the source); offsets[] holds 1..L-1.
      const offset = k === 0 ? undefined : offsets[k - 1];
      const chunkCoords = sourceChunk.map(
        (c, d) => c + (offset ? offset[d] : 0),
      );
      endpoints.push({ chunkCoords, vertexIndex: row[permCol + k] });
    }
    records.push({ endpoints });
  }
  return records;
}

// ---------------------------------------------------------------------------
// Metadata parsing
// ---------------------------------------------------------------------------

function parseJson(bytes: Uint8Array): any {
  return JSON.parse(new TextDecoder().decode(bytes));
}

/** Read `links/<delta>/zarr.json`; `undefined` = the family is absent. */
async function readFamilyMeta(
  options: CrossChunkLinksReaderOptions,
  delta: number,
  signal: AbortSignal,
): Promise<LinksFamilyMeta | undefined> {
  const bytes = await options.kvStoreRead(
    `${linksGroupPath(delta)}/zarr.json`,
    signal,
  );
  if (bytes === undefined) return undefined;
  const attrs = parseJson(bytes)?.attributes ?? {};
  if (attrs.zv_array !== LINKS_FAMILY_MARKER) {
    throw new Error(
      `links: ${linksGroupPath(delta)} is not a links family ` +
        `(zv_array=${JSON.stringify(attrs.zv_array)})`,
    );
  }
  const linkWidth = Number(attrs.link_width ?? 2);
  const sidNdim = Number(attrs.sid_ndim);
  if (!Number.isInteger(linkWidth) || linkWidth < 1) {
    throw new Error(
      `links: invalid link_width ${JSON.stringify(attrs.link_width)}`,
    );
  }
  if (!Number.isInteger(sidNdim) || sidNdim < 1) {
    throw new Error(
      `links: family group is missing/invalid sid_ndim ${JSON.stringify(attrs.sid_ndim)}`,
    );
  }
  return {
    linkWidth,
    sidNdim,
    directed: Boolean(attrs.directed ?? false),
    store: String(attrs.store ?? "canonical"),
  };
}

/** Read `links/<delta>/<offsets>/zarr.json`; `undefined` = that array is absent. */
async function readOffsetArrayDescriptor(
  options: CrossChunkLinksReaderOptions,
  arrayPath: string,
  fallbackOffsets: LinkOffset[],
  family: LinksFamilyMeta,
  delta: number,
  signal: AbortSignal,
): Promise<OffsetArrayDescriptor | undefined> {
  const bytes = await options.kvStoreRead(`${arrayPath}/zarr.json`, signal);
  if (bytes === undefined) return undefined;
  const attrs = parseJson(bytes)?.attributes ?? {};
  if (attrs.zv_array !== LINKS_ARRAY_MARKER) {
    throw new Error(
      `links: ${arrayPath} is not a links array (zv_array=${JSON.stringify(attrs.zv_array)})`,
    );
  }
  const offsets: LinkOffset[] = Array.isArray(attrs.offsets)
    ? attrs.offsets.map((o: any[]) => o.map((c) => Number(c)))
    : fallbackOffsets;
  const hasPerm =
    attrs.has_perm !== undefined
      ? Number(Boolean(attrs.has_perm))
      : Number(
          linksHasPerm(offsets, {
            delta,
            directed: family.directed,
            store: family.store,
          }),
        );
  const nonemptyChunks = Array.isArray(attrs.nonempty_chunks)
    ? new Set<string>(attrs.nonempty_chunks.map((s: any) => String(s)))
    : undefined;
  return { offsets, arrayPath, hasPerm, nonemptyChunks };
}

// ---------------------------------------------------------------------------
// Offset-array discovery
// ---------------------------------------------------------------------------

/** All non-zero offset vectors within a Chebyshev radius, near-first. */
function boundedOffsetCandidates(
  sidNdim: number,
  radius: number,
): LinkOffset[] {
  const out: LinkOffset[] = [];
  const rec = (dim: number, acc: number[]) => {
    if (dim === sidNdim) {
      if (acc.some((c) => c !== 0)) out.push(acc.slice());
      return;
    }
    for (let c = -radius; c <= radius; ++c) rec(dim + 1, [...acc, c]);
  };
  rec(0, []);
  const cheby = (o: LinkOffset) => Math.max(...o.map((c) => Math.abs(c)));
  return out.sort((a, b) => cheby(a) - cheby(b));
}

async function discoverDelta(
  options: CrossChunkLinksReaderOptions,
  delta: number,
  caches: CrossChunkLinksCaches,
  signal: AbortSignal,
): Promise<DeltaDiscovery | null> {
  const family = await readFamilyMeta(options, delta, signal);
  if (family === undefined) return null; // legitimately absent

  const descriptors: OffsetArrayDescriptor[] = [];
  let usedFallback = false;

  const collect = async (offsetSegments: string[]) => {
    for (const segment of offsetSegments) {
      const arrayPath = `${linksGroupPath(delta)}/${segment}`;
      // parseOffsets is validated inside readOffsetArrayDescriptor via the array's
      // own `offsets` field; the path segment is a fallback only.
      const descriptor = await readOffsetArrayDescriptor(
        options,
        arrayPath,
        parseSegmentSafe(segment, family),
        family,
        delta,
        signal,
      );
      if (descriptor === undefined) continue;
      // The all-zero (intra) array is handled by the per-chunk skeleton reader,
      // not the cross-chunk bridge path; skip it here.
      if (isIntra(descriptor.offsets)) continue;
      descriptors.push(descriptor);
    }
  };

  // Primary: LIST the family group's children (correct and unbounded).
  let listed: string[] | undefined;
  if (options.kvStoreList !== undefined) {
    try {
      const result = await options.kvStoreList(
        `${linksGroupPath(delta)}/`,
        signal,
      );
      listed = result.directories.filter((name) => name.length > 0);
    } catch {
      listed = undefined; // listing denied/unavailable -> fall back
    }
  }

  if (listed !== undefined) {
    await collect(listed);
  } else {
    // Fallback: probe a bounded offset neighbourhood by GET only. Correct for
    // adjacent-chunk links (all this store has); may miss long-range offsets at
    // decimated coarse levels.
    usedFallback = true;
    if (family.linkWidth !== 2) {
      warnOnce(
        caches,
        `links-fallback-width-${delta}`,
        `zarr-vectors links: cannot enumerate link_width=${family.linkWidth} offsets ` +
          `without directory listing; cross-chunk links for delta=${delta} will not load. ` +
          `Enable object listing on the store.`,
      );
    } else {
      const radius =
        options.fallbackOffsetRadius ?? DEFAULT_FALLBACK_OFFSET_RADIUS;
      // link_width=2 here, so each candidate is a single-offset segment.
      const segments = boundedOffsetCandidates(family.sidNdim, radius).map(
        (o) => formatOffsets([o]),
      );
      await collect(segments);
      warnOnce(
        caches,
        `links-fallback-${delta}`,
        `zarr-vectors links: object listing unavailable; discovered cross-chunk link ` +
          `offsets by probing a radius-${radius} neighbourhood. Links spanning more than ` +
          `${radius} chunk(s) (possible at decimated coarse levels) may be missed. Enable ` +
          `object listing on the store for complete discovery.`,
      );
    }
  }

  return { family, descriptors, usedFallback };
}

/**
 * Parse a listed segment against the family policy for validation, deferring to
 * the array's own `offsets` metadata (which `readOffsetArrayDescriptor` prefers)
 * when the segment is malformed.
 */
function parseSegmentSafe(
  segment: string,
  family: LinksFamilyMeta,
): LinkOffset[] {
  try {
    return parseOffsets(segment, {
      sidNdim: family.sidNdim,
      linkWidth: family.linkWidth,
    });
  } catch {
    return [];
  }
}

async function getDiscovery(
  options: CrossChunkLinksReaderOptions,
  delta: number,
  caches: CrossChunkLinksCaches,
  signal: AbortSignal,
): Promise<DeltaDiscovery | null> {
  const existing = caches.byDelta.get(delta);
  if (existing !== undefined) return existing;
  const promise = discoverDelta(options, delta, caches, signal);
  caches.byDelta.set(delta, promise);
  return promise;
}

// ---------------------------------------------------------------------------
// Public readers
// ---------------------------------------------------------------------------

function chunkKeyDot(coords: readonly number[]): string {
  return coords.join(".");
}
function cellPath(arrayPath: string, coords: readonly number[]): string {
  return `${arrayPath}/c/${coords.join("/")}`;
}

async function readCellRecords(
  options: CrossChunkLinksReaderOptions,
  descriptor: OffsetArrayDescriptor,
  sourceChunk: readonly number[],
  family: LinksFamilyMeta,
  signal: AbortSignal,
): Promise<CrossChunkLinkRecord[]> {
  // Skip the GET when the array's manifest says this cell is empty.
  if (
    descriptor.nonemptyChunks !== undefined &&
    !descriptor.nonemptyChunks.has(chunkKeyDot(sourceChunk))
  ) {
    return [];
  }
  const raw = await options.kvStoreRead(
    cellPath(descriptor.arrayPath, sourceChunk),
    signal,
  );
  if (raw === undefined || raw.byteLength === 0) return [];
  return decodeLinkCell(
    raw,
    sourceChunk,
    descriptor.offsets,
    family.linkWidth,
    descriptor.hasPerm !== 0,
  );
}

/**
 * Read every link record incident on `targetChunkCoords`.
 *
 * For each `<offsets>` array a chunk C participates in two ways: as the record
 * **source** (cell C, endpoint 0 in C) and as a **target** (cell `C - offset`,
 * endpoint `k` in C). Both are read so the caller's per-chunk ghost insertion
 * sees the bridge from whichever side it is drawing.
 *
 * Returns `undefined` **only** when the links family is absent (no `links/<delta>/`
 * group) — the caller may then legitimately skip links. A family that exists but
 * cannot be read throws, so the caller can warn without permanently disabling a
 * store that does have links.
 */
export async function readCrossChunkLinksForChunk(
  options: CrossChunkLinksReaderOptions,
  targetChunkCoords: readonly number[],
  caches: CrossChunkLinksCaches,
  signal: AbortSignal,
): Promise<CrossChunkLinksTable | undefined> {
  const delta = options.delta ?? 0;
  const discovery = await getDiscovery(options, delta, caches, signal);
  if (discovery === null) return undefined;
  const { family, descriptors } = discovery;

  const records: CrossChunkLinkRecord[] = [];
  for (const descriptor of descriptors) {
    // Source side: records filed under C.
    for (const rec of await readCellRecords(
      options,
      descriptor,
      targetChunkCoords,
      family,
      signal,
    )) {
      records.push(rec);
    }
    // Target side: records whose successor is in C are filed under C - offset.
    // (link_width=2 has one offset; wider records are handled only via the
    // source side here, which is all the bridge renderer consumes.)
    if (descriptor.offsets.length === 1) {
      const mirror = targetChunkCoords.map(
        (c, d) => c - descriptor.offsets[0][d],
      );
      if (mirror.every((c) => c >= 0)) {
        for (const rec of await readCellRecords(
          options,
          descriptor,
          mirror,
          family,
          signal,
        )) {
          records.push(rec);
        }
      }
    }
  }
  return { linkWidth: family.linkWidth, sidNdim: family.sidNdim, records };
}

/**
 * Read the whole links table for one `(level, delta)` — every record under
 * every `<offsets>` array. Used by the object-keyed (pass-2) backend, which
 * aggregates a whole object into one merged geometry. Prefer
 * {@link readCrossChunkLinksForChunk} when only one chunk's records are needed.
 *
 * Requires either object listing or per-array `nonempty_chunks` manifests to
 * enumerate populated cells; returns `undefined` when the family is absent.
 */
export async function readCrossChunkLinks(
  options: CrossChunkLinksReaderOptions,
  signal: AbortSignal,
): Promise<CrossChunkLinksTable | undefined> {
  const delta = options.delta ?? 0;
  const caches = createCrossChunkLinksCaches();
  const discovery = await getDiscovery(options, delta, caches, signal);
  if (discovery === null) return undefined;
  const { family, descriptors } = discovery;

  const records: CrossChunkLinkRecord[] = [];
  for (const descriptor of descriptors) {
    if (descriptor.nonemptyChunks === undefined) {
      throw new Error(
        `links: whole-table read of ${descriptor.arrayPath} needs a nonempty_chunks ` +
          `manifest or object listing to enumerate cells`,
      );
    }
    for (const key of descriptor.nonemptyChunks) {
      const sourceChunk = key.split(".").map((s) => Number(s));
      for (const rec of await readCellRecords(
        options,
        descriptor,
        sourceChunk,
        family,
        signal,
      )) {
        records.push(rec);
      }
    }
  }
  return { linkWidth: family.linkWidth, sidNdim: family.sidNdim, records };
}

/** The intra-chunk (all-zero offsets) array path, for the per-chunk skeleton
 * reader that renders same-chunk connectivity for explicit stores. */
export function intraLinksArrayPath(
  delta: number,
  sidNdim: number,
  linkWidth: number,
): string {
  return linksPath(delta, intraOffsets(sidNdim, linkWidth));
}
