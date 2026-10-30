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
 * @file Whole objects, for Neuroglancer's own skeleton and mesh layers:
 * segment id -> row -> manifest -> the fragments it owns in each chunk.
 * Skeletons are read whole at the finest level, meshes one chunk (octree
 * node) at a time, at any level. Where the cells allow it only the object's
 * own rows are read, by byte range; otherwise its chunks are decoded whole,
 * and those decoded chunks, like the manifest chunks, are shared between
 * the objects that pass through them.
 */

import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/base.js";
import type { DecodedChunk } from "#src/datasource/zarr-vectors/chunk_decode.js";
import {
  decodeChunk,
  edgeTangents,
  forEachFragmentVertex,
} from "#src/datasource/zarr-vectors/chunk_decode.js";
import {
  fillObjectValues,
  prefetchObjectValues,
} from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import type { ElementType } from "#src/datasource/zarr-vectors/dtype.js";
import {
  attributeDtype,
  decodeFloat32,
  decodeIndices,
  ELEMENT_BYTES,
  isElementType,
} from "#src/datasource/zarr-vectors/dtype.js";
import type { FragmentRows } from "#src/datasource/zarr-vectors/fragment_index.js";
import {
  decodeFragmentHeader,
  decodeFragments,
  FRAGMENT_HEADER_BYTES,
  FRAGMENT_RANGE_ENTRY_BYTES,
  FragmentIndex,
} from "#src/datasource/zarr-vectors/fragment_index.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import {
  LevelCells,
  MAX_ROW_READS,
  rangeSpans,
  ROW_GAP,
} from "#src/datasource/zarr-vectors/level_cells.js";
import type { CrossChunkLinkRecord } from "#src/datasource/zarr-vectors/links.js";
import {
  CrossChunkLinks,
  intraLinksPath,
  ownedLinksInRows,
} from "#src/datasource/zarr-vectors/links.js";
import type { ManifestBlock } from "#src/datasource/zarr-vectors/object_manifest.js";
import {
  decodeObjectManifest,
  resolveFragmentRef,
} from "#src/datasource/zarr-vectors/object_manifest.js";
import {
  readObjectTable,
  SegmentIdIndex,
} from "#src/datasource/zarr-vectors/objects.js";
import type {
  ZarrVectorsLevel,
  ZarrVectorsStoreAccess,
} from "#src/datasource/zarr-vectors/store.js";
import {
  AsyncLru,
  mapConcurrent,
  SHARED_SIGNAL,
  warnOnce,
} from "#src/datasource/zarr-vectors/util.js";
import { decodeVlenElements } from "#src/datasource/zarr-vectors/zarr_array.js";

export interface ObjectSkeleton {
  positions: Float32Array;
  edges: Uint32Array;
  /** One array per exposed attribute, then the tangent if the kind has one. */
  attributes: Float32Array[];
}

/**
 * One skeleton from parts read at several levels (an additive pyramid keeps
 * each part of an object at one level): vertices concatenated, edges
 * renumbered, attributes (and the tangent) concatenated alike.
 */
export function mergeSkeletons(
  parts: readonly ObjectSkeleton[],
): ObjectSkeleton {
  const used = parts.filter((p) => p.positions.length > 0);
  if (used.length === 1) return used[0];
  if (used.length === 0) return parts[0];
  const numVertices = used.reduce((n, p) => n + p.positions.length / 3, 0);
  const positions = new Float32Array(3 * numVertices);
  const edges = new Uint32Array(used.reduce((n, p) => n + p.edges.length, 0));
  const attributes = used[0].attributes.map(
    (_, k) =>
      new Float32Array(
        used.reduce((n, p) => n + (p.attributes[k]?.length ?? 0), 0),
      ),
  );
  let vertexAt = 0;
  let edgeAt = 0;
  const attributeAt = attributes.map(() => 0);
  for (const p of used) {
    positions.set(p.positions, 3 * vertexAt);
    for (let i = 0; i < p.edges.length; ++i) {
      edges[edgeAt + i] = p.edges[i] + vertexAt;
    }
    edgeAt += p.edges.length;
    p.attributes.forEach((a, k) => {
      if (k >= attributes.length) return;
      attributes[k].set(a, attributeAt[k]);
      attributeAt[k] += a.length;
    });
    vertexAt += p.positions.length / 3;
  }
  return { positions, edges, attributes };
}

/**
 * The rows of one chunk an object's read has to hand: every row (the decoded
 * cell) or only the object's own (read by byte range).
 */
interface ChunkRows {
  fragments: FragmentRows;
  /** x, y, z per held row. */
  positions: Float32Array;
  /** One array per exposed attribute, `components` values per held row. */
  attributes: Float32Array[];
  /**
   * The chunk's own edges, as pairs of chunk rows; none when only the
   * object's rows were read (a curve's edges follow from its fragments).
   */
  edges: Uint32Array;
  /** Index of chunk row `v` into `positions` and `attributes`, or -1. */
  indexOf(v: number): number;
}

/** The attribute layouts measured against one chunk's cells. */
interface MeasuredLayout {
  chunkKey: string;
  layout: AttributeLayout[];
  /** Every attribute measured (none lacked a cell in that chunk). */
  complete: boolean;
}

/** How one exposed attribute is read by row range at a level. */
interface AttributeLayout {
  components: number;
  /**
   * The level's per-vertex array for it; `undefined` reads as NaN (an
   * object's value is filled in afterwards, as for a decoded chunk).
   */
  stored?: { path: string; type: ElementType; rowBytes: number };
}

/**
 * Past this share of a chunk's rows, an object is read from the whole cell:
 * one request, and the decoded chunk serves the next object through it.
 */
const WHOLE_CELL_FRACTION = 0.5;

/**
 * How much of a fragment-index cell is read first: its header and the range
 * bitmap of up to ~130k fragments. A cell no larger is thereby read whole; a
 * larger one (a dense chunk's index runs to a megabyte) is finished by
 * reading only the asked-for fragments' range entries.
 */
const FRAGMENT_PREFIX_BYTES = 16 * 1024;

/** Range entries of one index this close together are read together. */
const ENTRY_MERGE_GAP = 16 * 1024;

/** A fragment index read in part: the range entries of some fragments. */
class PartialFragments implements FragmentRows {
  constructor(
    readonly numFragments: number,
    private readonly ranges: ReadonlyMap<
      number,
      { start: number; count: number }
    >,
  ) {}

  isRange(f: number) {
    return this.ranges.has(f);
  }

  range(f: number) {
    const range = this.ranges.get(f);
    if (range === undefined) throw new Error(`fragment ${f} was not read`);
    return range;
  }

  indices(f: number) {
    const { start, count } = this.range(f);
    return Uint32Array.from({ length: count }, (_, i) => start + i);
  }
}

/** A large fragment index's head: enough to address its range entries. */
interface FragmentHead {
  numFragments: number;
  /** Offset of the range table in the cell's payload. */
  tableOffset: number;
  /** Bit `f`: fragment `f` is a range. */
  bitmap: Uint8Array;
  /** Range fragments before each fragment: its row in the range table. */
  prefix: Int32Array;
}

export interface ObjectReaderOptions {
  /** See {@link FRAGMENT_PREFIX_BYTES}; tests shrink it. */
  fragmentPrefixBytes?: number;
}

/**
 * Row reads of one cell this close together are merged into one request
 * (`coalesceRangeReads`): objects loading together usually own neighbouring
 * rows of a chunk (a writer lays out consecutive objects consecutively), and
 * a request costs about what transferring this many bytes does.
 */
const ROW_MERGE_GAP = 256 * 1024;

/** One object's vertices in each chunk it visits, in manifest (walk) order. */
interface ObjectParts {
  order: { chunkKey: string; vertices: number[] }[];
  chunks: Map<string, ChunkRows>;
}

/** Faces sampled to measure a level's edge length. */
const SAMPLE_FACES = 5000;

/** Rows `[start, start + count)` of a chunk and their positions. */
interface RowSpan {
  start: number;
  count: number;
  /** x, y, z per row. */
  positions: Float32Array;
}

/** One object's part of a chunk: its faces there and its vertices. */
interface LocalPart {
  /**
   * Chunk-local vertex triples. A face with a corner outside `spans` is not
   * the object's and is left out.
   */
  faces: Uint32Array;
  /** The object's rows of the chunk, ascending by start. */
  spans: RowSpan[];
}

function decodedBytes(chunk: DecodedChunk | undefined) {
  if (chunk === undefined) return 0;
  let bytes =
    chunk.positions.byteLength +
    chunk.edges.byteLength +
    // Twice: meshes also keep the faces grouped by fragment.
    2 * chunk.faces.byteLength +
    chunk.fragments.byteLength +
    (chunk.tangents?.byteLength ?? 0);
  for (const a of chunk.attributes) bytes += a.byteLength;
  return bytes;
}

/**
 * A decoded chunk's faces grouped by the fragment of their first corner,
 * built once per chunk, so each object's faces are found without scanning
 * every object's.
 */
const facesByFragment = new WeakMap<
  DecodedChunk,
  { offsets: Uint32Array; faces: Uint32Array }
>();

function fragmentFaces(decoded: DecodedChunk, fragment: number) {
  let index = facesByFragment.get(decoded);
  if (index === undefined) {
    const { fragments, faces, numVertices } = decoded;
    const owner = new Int32Array(numVertices).fill(-1);
    for (let f = 0; f < fragments.numFragments; ++f) {
      forEachFragmentVertex(fragments, f, (v) => {
        if (v < numVertices) owner[v] = f;
      });
    }
    const offsets = new Uint32Array(fragments.numFragments + 1);
    for (let i = 0; i + 3 <= faces.length; i += 3) {
      const f = owner[faces[i]];
      if (f >= 0) offsets[f + 1] += 3;
    }
    for (let f = 0; f < fragments.numFragments; ++f) {
      offsets[f + 1] += offsets[f];
    }
    const next = offsets.slice(0, fragments.numFragments);
    const grouped = new Uint32Array(offsets[fragments.numFragments]);
    for (let i = 0; i + 3 <= faces.length; i += 3) {
      const f = owner[faces[i]];
      if (f < 0) continue;
      grouped.set(faces.subarray(i, i + 3), next[f]);
      next[f] += 3;
    }
    index = { offsets, faces: grouped };
    facesByFragment.set(decoded, index);
  }
  return index.faces.subarray(
    index.offsets[fragment],
    index.offsets[fragment + 1],
  );
}

interface StoreCaches {
  chunks: AsyncLru<DecodedChunk | undefined>;
  /** Raw vertex cells, shared by chunk decodes and faces reaching across. */
  vertexCells: AsyncLru<Uint8Array | undefined>;
}

const caches = new WeakMap<ZarrVectorsStoreAccess, StoreCaches>();

/**
 * Decoded chunks are large (tens of MB at a whole-brain store's finest
 * level) and not charged to the chunk manager, so the caches are bounded by
 * bytes -- and shared by every reader of a store (one per level of detail,
 * per kind of source), so the bound is per store, not per reader. Objects
 * drawn together mostly share chunks.
 */
function storeCaches(access: ZarrVectorsStoreAccess): StoreCaches {
  let out = caches.get(access);
  if (out === undefined) {
    out = {
      chunks: new AsyncLru<DecodedChunk | undefined>(
        Infinity,
        384 * 1024 * 1024,
        decodedBytes,
      ),
      vertexCells: new AsyncLru<Uint8Array | undefined>(
        Infinity,
        128 * 1024 * 1024,
        (b) => b?.byteLength ?? 0,
      ),
    };
    caches.set(access, out);
  }
  return out;
}

export class ObjectReader {
  private cells: LevelCells;
  private links: CrossChunkLinks | undefined;
  private index:
    | Promise<{ rows: SegmentIdIndex; chunkRows: number } | undefined>
    | undefined;
  /** Caches shared with the store's other readers, and this one's keys. */
  private caches: StoreCaches;
  private cacheKey: string;
  private manifestChunks = new AsyncLru<Uint8Array[]>(16);
  private fragmentIndexes = new AsyncLru<FragmentIndex | undefined>(1024);
  /** Heads of large fragment indexes, per chunk. */
  private fragmentHeads = new AsyncLru<FragmentHead | undefined>(1024);
  private readonly fragmentPrefixBytes: number;
  /** Attribute layouts measured per chunk, until one chunk settles them all. */
  private layouts = new AsyncLru<MeasuredLayout | undefined>(16);
  private layout_: Promise<AttributeLayout[] | undefined> | undefined;
  /** The measurement in flight, shared by every chunk asking meanwhile. */
  private measuring: Promise<MeasuredLayout | undefined> | undefined;

  constructor(
    private access: ZarrVectorsStoreAccess,
    private description: ZarrVectorsGeometryDescription,
    private level: ZarrVectorsLevel,
    options: ObjectReaderOptions = {},
  ) {
    this.fragmentPrefixBytes =
      options.fragmentPrefixBytes ?? FRAGMENT_PREFIX_BYTES;
    this.cells = LevelCells.forLevel(access, level, description);
    this.caches = storeCaches(access);
    // A decoded chunk depends on the level and the attributes selected.
    this.cacheKey = `${level.path}|${description.attributes
      .map((a) => a.name)
      .join(",")}|`;
    if (KIND_CAPABILITIES[description.geometryKind].primitive !== "points") {
      this.links = new CrossChunkLinks({
        cells: this.cells,
        listDirectories: (path) =>
          access.listDirectories(`${level.path}/${path}`),
        warn: warnOnce,
      });
    }
    // Needed by the first object, so fetched now rather than in turn: the
    // table its manifest is found through, and the values every vertex gets.
    if (description.hasObjects) {
      this.objectIndex().catch(() => {});
      prefetchObjectValues(access, description);
    }
  }

  private objectIndex() {
    if (this.index === undefined) {
      const promise = (async () => {
        const table = await readObjectTable(this.access, this.cells.levelPath);
        if (table === undefined) return undefined;
        const manifests = await this.cells.reader("object_index/manifests");
        return {
          rows: new SegmentIdIndex(table.segmentIds),
          chunkRows: manifests?.array.readChunkShape[0] ?? 1,
        };
      })();
      this.index = promise;
      promise.catch(() => {
        if (this.index === promise) this.index = undefined;
      });
    }
    return this.index;
  }

  /** The manifest of the object shown as `segmentId`. */
  async manifest(segmentId: bigint): Promise<ManifestBlock[]> {
    const index = await this.objectIndex();
    if (index === undefined) throw new Error("this store has no object index");
    const row = index.rows.rowOf(segmentId);
    if (row === undefined) return [];
    const reader = await this.cells.reader("object_index/manifests");
    if (reader === undefined)
      throw new Error("object_index/manifests is missing");
    const chunk = Math.floor(row / index.chunkRows);
    const elements = await this.manifestChunks.get(String(chunk), async () => {
      const bytes = await reader.readChunk([chunk], SHARED_SIGNAL);
      return bytes === undefined ? [] : decodeVlenElements(bytes);
    });
    const blob = elements[row - chunk * index.chunkRows];
    return blob === undefined || blob.byteLength === 0
      ? []
      : decodeObjectManifest(blob, 3);
  }

  private async parts(
    segmentId: bigint,
    signal: AbortSignal,
  ): Promise<ObjectParts> {
    const blocks = await this.manifest(segmentId);
    const fragmentIds = new Map<string, number[]>();
    for (const block of blocks) {
      const key = block.chunkCoords.join(".");
      let ids = fragmentIds.get(key);
      if (ids === undefined) fragmentIds.set(key, (ids = []));
      for (const f of resolveFragmentRef(block.fragmentRef)) ids.push(f);
    }
    const keys = [...fragmentIds.keys()];
    const rows = await Promise.all(
      keys.map((key) => this.chunkRows(key, fragmentIds.get(key)!, signal)),
    );
    signal.throwIfAborted();
    const chunks = new Map<string, ChunkRows>();
    keys.forEach((key, i) => {
      if (rows[i] !== undefined) chunks.set(key, rows[i]!);
    });
    const order: ObjectParts["order"] = [];
    for (const block of blocks) {
      const chunkKey = block.chunkCoords.join(".");
      const chunk = chunks.get(chunkKey);
      if (chunk === undefined) continue;
      for (const f of resolveFragmentRef(block.fragmentRef)) {
        if (f >= chunk.fragments.numFragments) continue;
        const vertices: number[] = [];
        forEachFragmentVertex(chunk.fragments, f, (v) => vertices.push(v));
        order.push({ chunkKey, vertices });
      }
    }
    return { order, chunks };
  }

  async readSkeleton(
    segmentId: bigint,
    signal: AbortSignal,
  ): Promise<ObjectSkeleton> {
    const { description } = this;
    const { order, chunks } = await this.parts(segmentId, signal);
    // One output vertex per (chunk, local vertex).
    const indexOf = new Map<string, Map<number, number>>();
    const sources: { chunk: ChunkRows; vertex: number }[] = [];
    const add = (chunkKey: string, vertex: number) => {
      let map = indexOf.get(chunkKey);
      if (map === undefined) indexOf.set(chunkKey, (map = new Map()));
      let i = map.get(vertex);
      if (i === undefined) {
        i = sources.length;
        map.set(vertex, i);
        sources.push({ chunk: chunks.get(chunkKey)!, vertex });
      }
      return i;
    };
    const edges: number[] = [];
    const primitive = KIND_CAPABILITIES[description.geometryKind].primitive;
    if (
      primitive === "lines" &&
      description.linksConvention === "implicit_sequential"
    ) {
      // A curve: its fragments, in manifest order, continue one another.
      let tail: number | undefined;
      for (const { chunkKey, vertices } of order) {
        for (const v of vertices) {
          const i = add(chunkKey, v);
          if (tail !== undefined) edges.push(tail, i);
          tail = i;
        }
      }
    } else {
      for (const { chunkKey, vertices } of order) {
        for (const v of vertices) add(chunkKey, v);
      }
      // The links between its chunks...
      const linked = description.skeletonLayout === "linked";
      // A stored link replaces a row's implied parent only where parents
      // are implied: explicit stores list every edge.
      const relinks =
        linked &&
        description.linksConvention === "implicit_sequential_with_branches";
      const relinked = new Set<string>();
      const chunkKeys = [...indexOf.keys()];
      const linkSets = new Array<CrossChunkLinkRecord[]>(chunkKeys.length);
      await mapConcurrent(chunkKeys, 16, async (chunkKey, i) => {
        const coords = chunkKey.split(".").map(Number);
        linkSets[i] =
          (await (linked
            ? this.links?.linksTouching(coords)
            : this.links?.linksOwnedBy(coords))) ?? [];
      });
      signal.throwIfAborted();
      for (const [i, chunkKey] of chunkKeys.entries()) {
        for (const { endpoints } of linkSets[i]) {
          if (endpoints.length !== 2) continue;
          const keys = endpoints.map((e) => e.chunkCoords.join("."));
          const [a, b] = endpoints.map((e, i) =>
            indexOf.get(keys[i])?.get(e.vertexIndex),
          );
          if (a === undefined || b === undefined) continue;
          if (keys[0] !== chunkKey && keys[1] !== chunkKey) continue;
          if (linked && keys[0] !== chunkKey) continue; // the child's chunk keeps it
          edges.push(a, b);
          // In a linked skeleton the link replaces the child's implied parent.
          if (relinks) relinked.add(`${keys[0]}:${endpoints[0].vertexIndex}`);
        }
      }
      // ...and each chunk's own edges between its vertices.
      for (const [chunkKey, map] of indexOf) {
        const e = chunks.get(chunkKey)!.edges;
        for (let k = 0; k < e.length; k += 2) {
          const a = map.get(e[k]);
          const b = map.get(e[k + 1]);
          if (a === undefined || b === undefined) continue;
          const implied = e[k] + 1 === e[k + 1];
          if (implied && relinked.has(`${chunkKey}:${e[k + 1]}`)) continue;
          edges.push(a, b);
        }
      }
    }
    const n = sources.length;
    const positions = new Float32Array(n * 3);
    const attributes = description.attributes.map(
      (a) => new Float32Array(n * a.components),
    );
    sources.forEach(({ chunk, vertex }, i) => {
      const row = chunk.indexOf(vertex);
      if (row < 0) return; // a row the cell does not hold stays zero
      positions.set(chunk.positions.subarray(3 * row, 3 * row + 3), 3 * i);
      description.attributes.forEach(({ components: w }, k) => {
        attributes[k].set(
          chunk.attributes[k].subarray(w * row, w * (row + 1)),
          w * i,
        );
      });
    });
    // Object values are the object's own, on every vertex.
    if (n > 0 && description.objectValuesPath !== undefined) {
      const ids = new Uint32Array(2 * n);
      for (let v = 0; v < n; ++v) {
        ids[2 * v] = Number(segmentId & 0xffffffffn);
        ids[2 * v + 1] = Number(segmentId >> 32n);
      }
      await fillObjectValues(this.access, description, attributes, ids, n);
    }
    const edgeArray = Uint32Array.from(edges);
    if (KIND_CAPABILITIES[description.geometryKind].tangent !== undefined) {
      attributes.push(edgeTangents(positions, edgeArray));
    }
    return { positions, edges: edgeArray, attributes };
  }

  /** The chunks an object has geometry in at this level. */
  async chunksOf(segmentId: bigint): Promise<number[][]> {
    const blocks = await this.manifest(segmentId);
    const keys = new Set(blocks.map((b) => b.chunkCoords.join(".")));
    return [...keys].map((key) => key.split(".").map(Number));
  }

  /**
   * Whether an object can be read from its own rows alone: a curve's edges
   * join consecutive rows of its fragments and a point cloud has none, so
   * neither needs the chunk's link records. A skeleton's or graph's edges
   * are stored per chunk and still take the whole cell.
   */
  private get readsOwnRows() {
    const { geometryKind, linksConvention } = this.description;
    const { primitive } = KIND_CAPABILITIES[geometryKind];
    return (
      primitive === "points" ||
      (primitive === "lines" && linksConvention === "implicit_sequential")
    );
  }

  /**
   * The rows of `chunkKey` that fragments `fragmentIds` of one object need:
   * only those, by byte range, where the cells allow it and the object holds
   * a minority of the chunk's rows; else the whole decoded cell. `undefined`
   * when the chunk is empty.
   */
  private async chunkRows(
    chunkKey: string,
    fragmentIds: readonly number[],
    signal: AbortSignal,
  ): Promise<ChunkRows | undefined> {
    if (this.readsOwnRows) {
      const own = await this.rowsFromRanges(chunkKey, fragmentIds, signal);
      if (own !== undefined) return own;
    }
    const decoded = await this.decode(chunkKey);
    if (decoded === undefined) return undefined;
    const { numVertices } = decoded;
    return {
      fragments: decoded.fragments,
      positions: decoded.positions,
      attributes: decoded.attributes,
      edges: decoded.edges,
      indexOf: (v) => (v < numVertices ? v : -1),
    };
  }

  /**
   * How each exposed attribute is read by row range at this level: from its
   * array, with the element type the level stores and a row width checked
   * against the length of `chunkKey`'s cell (as the store's open measured
   * it; an attribute the level stores another way, or not at all, reads as
   * NaN, with the same warning as a decoded chunk gives). `undefined` when
   * some attribute's cells cannot be read by range, so the chunk is read
   * whole. Remembered once a chunk has settled every attribute; a chunk
   * that lacks some attribute's cell settles only itself.
   */
  private attributeLayout(
    chunkKey: string,
  ): Promise<AttributeLayout[] | undefined> {
    if (this.layout_ !== undefined) return this.layout_;
    const measure = (key: string) =>
      this.layouts.get(key, () => this.measureAttributeLayout(key));
    // One chunk's measurement serves the chunks asking at the same time:
    // the arrays are the same whichever chunk's header is read.
    const shared = (this.measuring ??= measure(chunkKey).finally(() => {
      this.measuring = undefined;
    }));
    return shared.then((measured) => {
      if (measured === undefined) return undefined;
      if (measured.complete) {
        this.layout_ ??= Promise.resolve(measured.layout);
        return measured.layout;
      }
      // The measured chunk lacks some attribute's cell: it settled only itself.
      if (measured.chunkKey === chunkKey) return measured.layout;
      return measure(chunkKey).then((own) => own?.layout);
    });
  }

  private async measureAttributeLayout(
    chunkKey: string,
  ): Promise<MeasuredLayout | undefined> {
    const { cells, description } = this;
    const vertexRow = 3 * ELEMENT_BYTES[this.vertexType];
    // The chunk's vertex count, from its cell's length, when first needed.
    let rows: Promise<number> | undefined;
    const chunkRows = () =>
      (rows ??= cells
        .cellPayloadLength("vertices", chunkKey, SHARED_SIGNAL)
        .then((length) =>
          length === null || length % vertexRow !== 0
            ? NaN
            : length / vertexRow,
        ));
    // `null`: not readable by range; `undefined`: no cell here, unmeasured.
    const measured = await Promise.all(
      description.attributes.map(
        async (a): Promise<AttributeLayout | null | undefined> => {
          const { components } = a;
          if (a.objectValue !== undefined) return { components };
          const path = `vertex_attributes/${a.name}`;
          const reader = await cells.reader(path);
          if (reader === undefined) {
            warnOnce(
              `${cells.levelPath} has no vertex attribute ${a.name}; it reads ` +
                "as NaN there",
            );
            return { components };
          }
          if (!reader.rangeAddressable) return null;
          const type = attributeDtype({ attributes: reader.array.attributes });
          if (!isElementType(type)) return null;
          const length = await cells.cellPayloadLength(
            path,
            chunkKey,
            SHARED_SIGNAL,
          );
          if (!length) return undefined;
          const n = await chunkRows();
          if (!(n > 0)) return null;
          const width = length / (n * ELEMENT_BYTES[type]);
          if (width !== components) {
            warnOnce(
              `${cells.levelPath}/${path} stores ${type} x${width} per vertex, ` +
                `not x${components}; it reads as NaN there`,
            );
            return { components };
          }
          return {
            components,
            stored: { path, type, rowBytes: components * ELEMENT_BYTES[type] },
          };
        },
      ),
    );
    if (measured.some((m) => m === null)) return undefined;
    return {
      chunkKey,
      layout: measured.map(
        (m, k) => m ?? { components: description.attributes[k].components },
      ),
      complete: measured.every((m) => m !== undefined),
    };
  }

  /**
   * The object's rows of a chunk read by byte range: its fragments' row
   * ranges from the chunk's fragment index, then those rows of `vertices`
   * and of each attribute array, merged into few reads. `undefined` where
   * the whole cell is the better read: cells that cannot be read by range
   * (compressed), a fragment listed row by row, an object holding most of
   * the chunk, or a read that fails.
   */
  private async rowsFromRanges(
    chunkKey: string,
    fragmentIds: readonly number[],
    signal: AbortSignal,
  ): Promise<ChunkRows | undefined> {
    const { cells } = this;
    if (!(await cells.rangeAddressable("vertices"))) return undefined;
    const fragments = await this.objectFragments(chunkKey, fragmentIds, signal);
    if (fragments === undefined) return undefined;
    const ranges: [number, number][] = [];
    let needed = 0;
    for (const f of new Set(fragmentIds)) {
      if (f >= fragments.numFragments) continue;
      if (!fragments.isRange(f)) return undefined;
      const { start, count } = fragments.range(f);
      if (count <= 0) continue;
      ranges.push([start, start + count - 1]);
      needed += count;
    }
    const empty = new Uint32Array(0);
    if (needed === 0) {
      return {
        fragments,
        positions: new Float32Array(0),
        attributes: this.description.attributes.map(() => new Float32Array(0)),
        edges: empty,
        indexOf: () => -1,
      };
    }
    // A whole index says how many rows the chunk has; a part read of a
    // large one does not, and such a chunk is rarely mostly one object.
    if (
      fragments instanceof FragmentIndex &&
      needed > WHOLE_CELL_FRACTION * fragments.rowBound()
    ) {
      return undefined;
    }
    const layout = await this.attributeLayout(chunkKey).catch((e) => {
      warnOnce(
        `${cells.levelPath}: measuring the attribute arrays failed; objects ` +
          "are read from whole chunks",
      );
      console.debug(e);
      return undefined;
    });
    if (layout === undefined) return undefined;
    const vertexType = this.vertexType;
    const vertexRow = 3 * ELEMENT_BYTES[vertexType];
    const spans = rangeSpans(
      ranges.sort((a, b) => a[0] - b[0]),
      ROW_GAP,
      MAX_ROW_READS,
    );
    let parts: { vertices: Uint8Array; attributes: (Uint8Array | null)[] }[];
    try {
      const read = (path: string, first: number, count: number, row: number) =>
        cells.readCellRange(
          path,
          chunkKey,
          first * row,
          count * row,
          signal,
          ROW_MERGE_GAP,
        );
      const read_ = await Promise.all(
        spans.map(async ([first, last]) => {
          const count = last - first + 1;
          const [vertices, ...attributes] = await Promise.all([
            read("vertices", first, count, vertexRow),
            ...layout.map(({ stored }) =>
              stored === undefined
                ? null
                : read(stored.path, first, count, stored.rowBytes),
            ),
          ]);
          // An empty cell here is an empty chunk: the whole-cell read says
          // so without a request.
          if (vertices == null) return undefined;
          return { vertices, attributes: attributes.map((b) => b ?? null) };
        }),
      );
      if (read_.some((p) => p === undefined)) return undefined;
      parts = read_ as NonNullable<(typeof read_)[number]>[];
    } catch (e) {
      signal.throwIfAborted();
      warnOnce(
        `${cells.levelPath}: reading an object's own rows failed; its ` +
          "chunks are read whole",
      );
      console.debug(e);
      return undefined;
    }
    const total = spans.reduce((n, [first, last]) => n + last - first + 1, 0);
    const positions = new Float32Array(3 * total);
    const attributes = layout.map(
      ({ components }) => new Float32Array(total * components),
    );
    const offsets = new Array<number>(spans.length);
    let at = 0;
    spans.forEach(([first, last], s) => {
      const count = last - first + 1;
      offsets[s] = at;
      const part = parts[s];
      positions.set(
        decodeFloat32(part.vertices, vertexType, 3 * count),
        3 * at,
      );
      layout.forEach(({ components, stored }, k) => {
        const bytes = part.attributes[k];
        const out = attributes[k].subarray(
          at * components,
          (at + count) * components,
        );
        if (stored === undefined || bytes === null) out.fill(NaN);
        else out.set(decodeFloat32(bytes, stored.type, count * components));
      });
      at += count;
    });
    const indexOf = (v: number) => {
      for (let s = 0; s < spans.length; ++s) {
        const [first, last] = spans[s];
        if (v >= first && v <= last) return offsets[s] + v - first;
      }
      return -1;
    };
    return { fragments, positions, attributes, edges: empty, indexOf };
  }

  private vertexCell(chunkKey: string) {
    return this.caches.vertexCells.get(`${this.level.path}|${chunkKey}`, () =>
      this.cells.readCell("vertices", chunkKey, SHARED_SIGNAL),
    );
  }

  /** A chunk decoded once for every object and level of detail that needs it. */
  private decode(chunkKey: string) {
    return this.caches.chunks.get(this.cacheKey + chunkKey, () =>
      decodeChunk(this.cells, this.description, chunkKey, SHARED_SIGNAL, {
        skipSegmentIds: true,
        // Skeletons get tangents once they are assembled; mesh nodes use
        // positions alone.
        skipTangents: true,
        skipAttributes:
          KIND_CAPABILITIES[this.description.geometryKind].primitive ===
          "triangles",
        vertices: this.vertexCell(chunkKey),
      }),
    );
  }

  private get vertexType(): ElementType {
    return this.level.arrays.vertices.attributes?.dtype ?? "float32";
  }

  /**
   * The level's intra-chunk face array, when the level is stamped
   * `fragment_link_groups`: one face group per vertex fragment, in fragment
   * order, as zarr-vectors-py's mesh writers and `zvtools index-faces` leave
   * it.
   */
  private get groupedFaces() {
    const { level, description } = this;
    const array = level.arrays.intraLinks;
    if (
      !level.fragmentLinkGroups ||
      array === undefined ||
      KIND_CAPABILITIES[description.geometryKind].primitive !== "triangles" ||
      // Rows are read as triangles; wider faces go through the whole cell.
      description.linkWidth !== 3
    ) {
      return undefined;
    }
    const type = array.attributes?.dtype ?? "int64";
    return isElementType(type)
      ? { path: intraLinksPath(description.linkWidth), type }
      : undefined;
  }

  /** A chunk's vertex fragments or face groups, decoded. */
  private fragmentIndex(path: string, chunkKey: string) {
    return this.fragmentIndexes.get(`${path}|${chunkKey}`, async () => {
      const bytes = await this.cells.readCell(path, chunkKey, SHARED_SIGNAL);
      return bytes === undefined ? undefined : decodeFragments(bytes);
    });
  }

  /**
   * The vertex fragments of a chunk that `fragmentIds` name, as cheaply as
   * the cell allows: the whole index (decoded once per chunk) when the cell
   * is small or cannot be read by range; else its head once per chunk, then
   * only the named fragments' range entries. `undefined` when the chunk has
   * no index or a named fragment is not a row range: the whole decoded
   * chunk handles those.
   */
  private async objectFragments(
    chunkKey: string,
    fragmentIds: readonly number[],
    signal: AbortSignal,
  ): Promise<FragmentRows | undefined> {
    const { cells } = this;
    const path = "vertex_fragments";
    const whole = this.fragmentIndexes.peek(`${path}|${chunkKey}`);
    if (whole !== undefined) return whole;
    if (!(await cells.rangeAddressable(path))) {
      return this.fragmentIndex(path, chunkKey);
    }
    const head = await this.fragmentHeads.get(chunkKey, () =>
      this.readFragmentHead(chunkKey),
    );
    // No head: the cell was small and is decoded whole, or is empty.
    if (head === undefined) return this.fragmentIndex(path, chunkKey);
    const wanted = [...new Set(fragmentIds)].filter(
      (f) => f < head.numFragments,
    );
    const rowOf = new Map<number, number>();
    for (const f of wanted) {
      if (((head.bitmap[f >> 3] >> (f & 7)) & 1) === 0) return undefined;
      rowOf.set(head.prefix[f], f);
    }
    const rows = [...rowOf.keys()].sort((a, b) => a - b);
    const spans = rangeSpans(
      rows.map((r) => [r, r] as const),
      64,
      MAX_ROW_READS,
    );
    const entrySize = FRAGMENT_RANGE_ENTRY_BYTES;
    const ranges = new Map<number, { start: number; count: number }>();
    await Promise.all(
      spans.map(async ([first, last]) => {
        const bytes = await cells.readCellRange(
          path,
          chunkKey,
          head.tableOffset + entrySize * first,
          entrySize * (last - first + 1),
          signal,
          ENTRY_MERGE_GAP,
        );
        if (bytes == null) throw new Error(`${path}/${chunkKey}: index gone`);
        const view = new DataView(
          bytes.buffer,
          bytes.byteOffset,
          bytes.byteLength,
        );
        for (let r = first; r <= last; ++r) {
          const f = rowOf.get(r);
          if (f === undefined) continue;
          const at = entrySize * (r - first);
          ranges.set(f, {
            start: Number(view.getBigInt64(at, true)),
            count: Number(view.getBigInt64(at + 8, true)),
          });
        }
      }),
    );
    return new PartialFragments(head.numFragments, ranges);
  }

  /**
   * The head of a chunk's fragment index: `undefined` when the cell turned
   * out small enough to decode whole (then {@link fragmentIndex} has it), or
   * is empty, or could not be read by range.
   */
  private async readFragmentHead(
    chunkKey: string,
  ): Promise<FragmentHead | undefined> {
    const { cells } = this;
    const path = "vertex_fragments";
    const max = this.fragmentPrefixBytes;
    let bytes: Uint8Array | undefined | null;
    try {
      bytes = await cells.readCellPrefix(path, chunkKey, max, SHARED_SIGNAL);
    } catch (e) {
      // A server refusing the range: the whole cell is read instead.
      console.debug(e);
      return undefined;
    }
    if (bytes === null) return undefined;
    if (bytes === undefined || bytes.length < max) {
      // The whole cell came back: decoded once, for every object through it.
      const whole = bytes;
      this.fragmentIndexes
        .get(`${path}|${chunkKey}`, async () =>
          whole === undefined || whole.length === 0
            ? undefined
            : decodeFragments(whole),
        )
        .catch(() => {});
      return undefined;
    }
    const { numFragments, tableOffset } = decodeFragmentHeader(bytes);
    const bitmapBytes = (numFragments + 7) >> 3;
    let bitmap = bytes.subarray(
      FRAGMENT_HEADER_BYTES,
      FRAGMENT_HEADER_BYTES + bitmapBytes,
    );
    if (bitmap.length < bitmapBytes) {
      const rest = await cells.readCellRange(
        path,
        chunkKey,
        bytes.length,
        FRAGMENT_HEADER_BYTES + bitmapBytes - bytes.length,
        SHARED_SIGNAL,
      );
      if (rest == null) return undefined;
      const full = new Uint8Array(bitmapBytes);
      full.set(bitmap);
      full.set(rest, bitmap.length);
      bitmap = full;
    }
    const prefix = new Int32Array(numFragments + 1);
    let n = 0;
    for (let f = 0; f < numFragments; ++f) {
      prefix[f] = n;
      n += (bitmap[f >> 3] >> (f & 7)) & 1;
    }
    prefix[numFragments] = n;
    return { numFragments, tableOffset, bitmap, prefix };
  }

  private inconsistent(why: string) {
    warnOnce(
      `${this.cells.levelPath}: is stamped fragment_link_groups, ` +
        `but ${why}; those chunks are read whole`,
    );
  }

  /**
   * The object's part of a chunk read from its own rows only: its vertex
   * rows and its face group, by byte range. `undefined` unless the level
   * is stamped `fragment_link_groups` and the chunk bears that out
   * (uncompressed, one group per fragment, each a row range whose faces
   * index only its fragment) -- the caller then reads the whole cell.
   */
  private async partFromRows(
    chunkKey: string,
    fragmentIds: readonly number[],
    signal: AbortSignal,
  ): Promise<LocalPart | undefined> {
    const faceArray = this.groupedFaces;
    if (faceArray === undefined) return undefined;
    try {
      return await this.readRowsPart(chunkKey, fragmentIds, faceArray, signal);
    } catch (e) {
      // An unreadable sidecar, or one pointing past its cell.
      signal.throwIfAborted();
      this.inconsistent("reading a chunk's own rows failed");
      console.debug(e);
      return undefined;
    }
  }

  private async readRowsPart(
    chunkKey: string,
    fragmentIds: readonly number[],
    faceArray: { path: string; type: ElementType },
    signal: AbortSignal,
  ): Promise<LocalPart | undefined> {
    const { cells } = this;
    if (
      !(await cells.rangeAddressable("vertices")) ||
      !(await cells.rangeAddressable(faceArray.path))
    ) {
      return undefined;
    }
    const [fragments, groups, mayHaveFaces] = await Promise.all([
      this.fragmentIndex("vertex_fragments", chunkKey),
      this.fragmentIndex("link_fragments", chunkKey),
      cells.mayHaveCell(faceArray.path, chunkKey),
    ]);
    if (fragments === undefined) return undefined;
    // No face cell where the level lists its cells: every face here
    // crosses into a neighbour, so each object's group is empty.
    const noFaces = groups === undefined && !mayHaveFaces;
    if (groups === undefined && !noFaces) return undefined;
    if (
      groups !== undefined &&
      groups.numFragments !== fragments.numFragments
    ) {
      this.inconsistent("a chunk's face groups do not match its fragments");
      return undefined;
    }
    // Every fragment a row range with a row-range group, before any read.
    for (const f of fragmentIds) {
      if (
        f >= fragments.numFragments ||
        !fragments.isRange(f) ||
        (groups !== undefined && !groups.isRange(f))
      ) {
        return undefined;
      }
    }
    const vertexType = this.vertexType;
    const vertexRow = 3 * ELEMENT_BYTES[vertexType];
    const faceRow = 3 * ELEMENT_BYTES[faceArray.type];
    const parts = await Promise.all(
      fragmentIds.map(async (f) => {
        const { start, count } = fragments.range(f);
        const group = groups?.range(f) ?? { start: 0, count: 0 };
        const [vertexBytes, faceBytes] = await Promise.all([
          count === 0
            ? new Uint8Array(0)
            : cells.readCellRange(
                "vertices",
                chunkKey,
                start * vertexRow,
                count * vertexRow,
                signal,
              ),
          group.count === 0
            ? new Uint8Array(0)
            : cells.readCellRange(
                faceArray.path,
                chunkKey,
                group.start * faceRow,
                group.count * faceRow,
                signal,
              ),
        ]);
        if (vertexBytes == null || faceBytes == null) return undefined;
        const faces = decodeIndices(faceBytes, faceArray.type, 3 * group.count);
        for (const v of faces) {
          if (v < start || v >= start + count) {
            this.inconsistent(
              "a face group indexes another fragment's vertices",
            );
            return undefined;
          }
        }
        const positions = decodeFloat32(vertexBytes, vertexType, 3 * count);
        return { span: { start, count, positions }, faces };
      }),
    );
    if (parts.some((p) => p === undefined)) return undefined;
    const done = parts as NonNullable<(typeof parts)[number]>[];
    const faces = new Uint32Array(done.reduce((n, p) => n + p.faces.length, 0));
    let at = 0;
    for (const p of done) {
      faces.set(p.faces, at);
      at += p.faces.length;
    }
    return {
      faces,
      spans: done.map((p) => p.span).sort((a, b) => a.start - b.start),
    };
  }

  /** The object's part of a chunk, from the whole decoded cell. */
  private async partFromCell(
    chunkKey: string,
    fragmentIds: readonly number[],
  ): Promise<LocalPart | undefined> {
    const decoded = await this.decode(chunkKey);
    if (decoded === undefined) return undefined;
    const { fragments, positions, numVertices } = decoded;
    const spans: RowSpan[] = [];
    const faceParts: Uint32Array[] = [];
    for (const f of new Set(fragmentIds)) {
      if (f >= fragments.numFragments) continue;
      const rows = (start: number, count: number) => {
        count = Math.min(count, numVertices - start);
        if (count > 0) {
          spans.push({
            start,
            count,
            positions: positions.subarray(3 * start, 3 * (start + count)),
          });
        }
      };
      if (fragments.isRange(f)) {
        const { start, count } = fragments.range(f);
        rows(start, count);
      } else {
        forEachFragmentVertex(fragments, f, (v) => rows(v, 1));
      }
      faceParts.push(fragmentFaces(decoded, f));
    }
    const faces = new Uint32Array(faceParts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of faceParts) {
      faces.set(p, at);
      at += p.length;
    }
    return { faces, spans: spans.sort((a, b) => a.start - b.start) };
  }

  /** Positions of some rows of a neighbour, decoded already or read. */
  private async farPositions(
    chunkKey: string,
    rows: readonly number[],
    signal: AbortSignal,
  ): Promise<Map<number, Float32Array>> {
    const cached = this.caches.chunks.peek(this.cacheKey + chunkKey);
    const decoded = cached && (await cached.catch(() => undefined));
    if (decoded !== undefined) {
      const out = new Map<number, Float32Array>();
      for (const v of rows) {
        if (3 * v + 3 <= decoded.positions.length) {
          out.set(v, decoded.positions.subarray(3 * v, 3 * v + 3));
        }
      }
      return out;
    }
    return this.cells.vertexRows(chunkKey, this.vertexType, rows, signal, () =>
      this.vertexCell(chunkKey),
    );
  }

  /**
   * An object's triangles stored in one chunk: its faces inside the chunk,
   * and the faces the chunk stores across its faces (their other corners are
   * read from the neighbours).  Every face of the object is in exactly one
   * chunk's set.  Reads only the object's rows where the store allows it
   * (`partFromRows`), else the whole cell.
   */
  async readMeshNode(
    segmentId: bigint,
    chunk: readonly number[],
    signal: AbortSignal,
  ): Promise<{ positions: Float32Array; indices: Uint32Array }> {
    const empty = {
      positions: new Float32Array(0),
      indices: new Uint32Array(0),
    };
    const chunkKey = chunk.join(".");
    const fragmentIds = [
      ...new Set(
        (await this.manifest(segmentId))
          .filter((b) => b.chunkCoords.join(".") === chunkKey)
          .flatMap((b) => [...resolveFragmentRef(b.fragmentRef)]),
      ),
    ];
    if (fragmentIds.length === 0) return empty;
    const part =
      (await this.partFromRows(chunkKey, fragmentIds, signal)) ??
      (await this.partFromCell(chunkKey, fragmentIds));
    signal.throwIfAborted();
    if (part === undefined) return empty;

    // The object's rows here, numbered in row order: output vertices.
    const { spans, faces } = part;
    let lo = Infinity;
    let hi = -Infinity;
    let total = 0;
    for (const { start, count } of spans) {
      if (count === 0) continue;
      lo = Math.min(lo, start);
      hi = Math.max(hi, start + count);
      total += count;
    }
    if (total === 0) return empty;
    const remap = new Int32Array(hi - lo).fill(-1);
    const local = new Float32Array(3 * total);
    let numLocal = 0;
    for (const { start, count, positions } of spans) {
      for (let i = 0; i < count; ++i) {
        if (remap[start + i - lo] !== -1) continue;
        remap[start + i - lo] = numLocal;
        local.set(positions.subarray(3 * i, 3 * i + 3), 3 * numLocal);
        ++numLocal;
      }
    }
    const localIndex = (v: number) => (v >= lo && v < hi ? remap[v - lo] : -1);

    const intra = new Uint32Array(faces.length);
    let numIntra = 0;
    for (let i = 0; i + 3 <= faces.length; i += 3) {
      const a = localIndex(faces[i]);
      const b = localIndex(faces[i + 1]);
      const c = localIndex(faces[i + 2]);
      if (a < 0 || b < 0 || c < 0) continue;
      intra[numIntra++] = a;
      intra[numIntra++] = b;
      intra[numIntra++] = c;
    }

    // Faces this chunk stores across its faces. One object's: its corners
    // here say which, so they are looked up by the object's rows; the other
    // corners are read from the neighbours.
    const isLocal = (coords: readonly number[]) =>
      coords.every((c, d) => c === chunk[d]);
    const crossing: CrossChunkLinkRecord[] = [];
    if (this.links !== undefined) {
      const owned = await this.links.ownedByLocalVertex(chunk);
      // Merged row ranges, so no record is found twice.
      const ranges: [number, number][] = [];
      for (const { start, count } of spans) {
        const last = ranges[ranges.length - 1];
        if (last !== undefined && start <= last[1]) {
          last[1] = Math.max(last[1], start + count);
        } else if (count > 0) {
          ranges.push([start, start + count]);
        }
      }
      for (const [first, end] of ranges) {
        for (const record of ownedLinksInRows(owned, first, end)) {
          if (record.endpoints.length < 3) continue;
          if (
            record.endpoints.every(
              (e) => !isLocal(e.chunkCoords) || localIndex(e.vertexIndex) >= 0,
            )
          ) {
            crossing.push(record);
          }
        }
      }
    }
    const farRows = new Map<string, Set<number>>();
    for (const { endpoints } of crossing) {
      for (const { chunkCoords, vertexIndex } of endpoints) {
        if (isLocal(chunkCoords)) continue;
        const key = chunkCoords.join(".");
        let rows = farRows.get(key);
        if (rows === undefined) farRows.set(key, (rows = new Set()));
        rows.add(vertexIndex);
      }
    }
    const far = new Map(
      await Promise.all(
        [...farRows].map(
          async ([key, rows]) =>
            [
              key,
              await this.farPositions(
                key,
                [...rows].sort((a, b) => a - b),
                signal,
              ),
            ] as const,
        ),
      ),
    );
    const farIndex = new Map<string, number>();
    const farPositions: number[] = [];
    const across: number[] = [];
    for (const { endpoints } of crossing) {
      const ids: number[] = [];
      for (const { chunkCoords, vertexIndex } of endpoints) {
        if (isLocal(chunkCoords)) {
          ids.push(localIndex(vertexIndex));
          continue;
        }
        const key = `${chunkCoords.join(".")}:${vertexIndex}`;
        let id = farIndex.get(key);
        if (id === undefined) {
          const p = far.get(chunkCoords.join("."))?.get(vertexIndex);
          if (p === undefined) break;
          id = numLocal + farPositions.length / 3;
          farPositions.push(p[0], p[1], p[2]);
          farIndex.set(key, id);
        }
        ids.push(id);
      }
      if (ids.length !== endpoints.length) continue;
      for (let k = 1; k + 1 < ids.length; ++k) {
        across.push(ids[0], ids[k], ids[k + 1]);
      }
    }
    signal.throwIfAborted();
    const positions = new Float32Array(3 * numLocal + farPositions.length);
    positions.set(local.subarray(0, 3 * numLocal));
    positions.set(farPositions, 3 * numLocal);
    const indices = new Uint32Array(numIntra + across.length);
    indices.set(intra.subarray(0, numIntra));
    indices.set(across, numIntra);
    return { positions, indices };
  }

  /**
   * The first faces of a chunk's face cell and their corners: read by byte
   * range (a few thousand faces and just their vertices) where the cell
   * allows it, else from the whole decoded cell.
   */
  private async faceSample(chunkKey: string): Promise<
    | {
        faces: ArrayLike<number>;
        position(vertex: number): ArrayLike<number> | undefined;
      }
    | undefined
  > {
    const path = intraLinksPath(this.description.linkWidth);
    const type =
      (await this.cells.reader(path))?.array.attributes?.dtype ?? "int64";
    const length = isElementType(type)
      ? await this.cells.cellPayloadLength(path, chunkKey, SHARED_SIGNAL)
      : null;
    if (length === null || !isElementType(type)) {
      const decoded = await this.decode(chunkKey);
      if (decoded === undefined || decoded.faces.length < 3) return undefined;
      const { positions } = decoded;
      return {
        faces: decoded.faces.subarray(0, 3 * SAMPLE_FACES),
        position: (v) => positions.subarray(3 * v, 3 * v + 3),
      };
    }
    const rowBytes = 3 * ELEMENT_BYTES[type];
    const rows = Math.min(SAMPLE_FACES, Math.floor(length / rowBytes));
    if (rows === 0) return undefined;
    const bytes = await this.cells.readCellRange(
      path,
      chunkKey,
      0,
      rows * rowBytes,
      SHARED_SIGNAL,
    );
    if (bytes == null) return undefined;
    const faces = decodeIndices(bytes, type, 3 * rows);
    const found = await this.cells.vertexRows(
      chunkKey,
      this.vertexType,
      [...new Set(faces)].sort((a, b) => a - b),
      SHARED_SIGNAL,
      () => this.vertexCell(chunkKey),
    );
    return { faces, position: (v) => found.get(v) };
  }

  /**
   * Mean edge length of some faces in one populated chunk: the level's
   * resolution, for choosing between levels of detail.
   */
  async meanEdgeLength(): Promise<number | undefined> {
    const reader = await this.cells.reader("vertices");
    if (reader === undefined) return undefined;
    const { nonemptyCells, shape, origin } = reader.array;
    const candidates = nonemptyCells
      ? [...nonemptyCells].slice(0, 8)
      : Array.from(
          { length: Math.min(64, shape[0] * shape[1] * shape[2]) },
          (_, i) =>
            [
              Math.floor(i / (shape[1] * shape[2])) + origin[0],
              (Math.floor(i / shape[2]) % shape[1]) + origin[1],
              (i % shape[2]) + origin[2],
            ].join("."),
        );
    const measure = async (key: string) => {
      const sample = await this.faceSample(key);
      if (sample === undefined) return undefined;
      const { faces, position } = sample;
      let total = 0;
      let count = 0;
      for (let i = 0; i + 3 <= faces.length; i += 3) {
        for (let k = 0; k < 3; ++k) {
          const a = position(faces[i + k]);
          const b = position(faces[i + ((k + 1) % 3)]);
          if (a === undefined || b === undefined) continue;
          total += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
          ++count;
        }
      }
      const mean = total / count;
      return count > 0 && Number.isFinite(mean) && mean > 0 ? mean : undefined;
    };
    // Listed cells hold data, so the first usually answers; unlisted ones
    // are guesses, mostly empty, so several are tried at once.
    const batch = nonemptyCells ? 1 : 8;
    for (let i = 0; i < candidates.length; i += batch) {
      const means = await Promise.all(
        candidates.slice(i, i + batch).map(measure),
      );
      const found = means.find((m) => m !== undefined);
      if (found !== undefined) return found;
    }
    return undefined;
  }
}
