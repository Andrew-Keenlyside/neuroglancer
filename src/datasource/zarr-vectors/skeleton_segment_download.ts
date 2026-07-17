/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Aggregate one object's skeleton geometry across the chunks that the
 * `object_index/manifests` reader reports for it.
 *
 * The pass-2 chunk-source backend calls `downloadSegmentSkeleton(oid,
 * ...)`: this resolves the manifest, fetches every named source chunk,
 * filters each one to just the fragments the object owns, and emits a
 * single merged geometry — `vertexPositions`, `indices`, and
 * `vertexAttributes` — ready to drop into a per-segment `SkeletonChunk`.
 */

import type { CrossChunkLinksTable } from "#src/datasource/zarr-vectors/cross_chunk_links.js";
import {
  hasSynthesisedTangent,
  KIND_CAPABILITIES,
} from "#src/datasource/zarr-vectors/geometry_kind.js";
import { resolveFragmentRef } from "#src/datasource/zarr-vectors/object_manifest.js";
import {
  readObjectManifest,
  type ObjectManifestReaderOptions,
} from "#src/datasource/zarr-vectors/object_manifest_reader.js";
import type {
  AttributeTypedArray,
  LinksConvention,
  SkeletonChunk,
  SkeletonGeometryKind,
} from "#src/datasource/zarr-vectors/skeleton_chunk.js";
import {
  computeTangentsFromEdges,
  orientTangentSignsAcrossEdges,
} from "#src/datasource/zarr-vectors/skeleton_chunk.js";
import {
  downloadSkeletonChunk,
  downloadSkeletonChunkScoped,
  type AttributeDtype,
  type LinkDtype,
  type ReadArrayChunk,
  type ReadArrayChunkScoped,
} from "#src/datasource/zarr-vectors/skeleton_chunk_download.js";

/**
 * The merged geometry for one object.  Shapes match the per-segment
 * `SkeletonChunk` fields that the render layer consumes.
 */
export interface AggregatedSegmentSkeleton {
  /** `(numVertices * rank)` floats. */
  readonly vertexPositions: Float32Array;
  /** `(numEdges * 2)` chunk-local-then-global vertex indices. */
  readonly indices: Uint32Array;
  /**
   * Per-vertex attributes, in the order the render layer will reference.
   * For streamline / polyline geometry kinds, index 0 is the synthesised
   * `tangent` (vec3); subsequent entries are the user-declared
   * attributes from `attributeNames` in declaration order.  For
   * skeleton geometry, only user-declared attributes are present.
   */
  readonly vertexAttributes: AttributeTypedArray[];
}

/**
 * Pure function: filter one decoded `SkeletonChunk` to just the
 * vertices and edges named by `fragmentIndices`.  Returns chunk-local
 * geometry (positions in float-flat layout, edges as chunk-local
 * vertex indices into the filtered output), plus the filtered attribute
 * arrays parallel to the positions.
 *
 * The render layer's vertex-attribute ordering convention is mirrored
 * here: when `chunk.tangents` is present (streamline/polyline) it is
 * emitted as the first attribute; user attributes follow.
 *
 * Vertices that don't belong to any of the named fragments are
 * dropped; edges with at least one endpoint dropped are also dropped
 * (no dangling references).
 */
export function filterChunkByFragments(
  chunk: SkeletonChunk,
  fragmentIndices: Uint32Array,
): {
  positions: Float32Array;
  edges: Uint32Array;
  attributes: AttributeTypedArray[];
  /** Map from source chunk-local vertex index → position in the filtered output. */
  vertexRemap: Int32Array;
} {
  const { rank, numVertices, positions, edges, vertexAttributes, tangents } =
    chunk;

  // Collect chunk-local vertex indices owned by the named fragments.
  // Use a `seen` mask to dedupe (the same vertex can be referenced by
  // multiple fragments — e.g. a branch point at level 0).  Walk-order
  // is preserved for the first occurrence so attribute lookups stay
  // deterministic.
  const seen = new Uint8Array(numVertices);
  const owned: number[] = [];
  for (let i = 0; i < fragmentIndices.length; ++i) {
    const f = fragmentIndices[i];
    const fragVerts = chunk.fragmentIndex.indices(f);
    for (let j = 0; j < fragVerts.length; ++j) {
      const v = fragVerts[j];
      if (seen[v] === 0) {
        seen[v] = 1;
        owned.push(v);
      }
    }
  }
  const numOwned = owned.length;

  // Build the source→filtered vertex remap.  -1 means "not in output".
  const vertexRemap = new Int32Array(numVertices).fill(-1);
  for (let i = 0; i < numOwned; ++i) vertexRemap[owned[i]] = i;

  // Gather positions for owned vertices, in walk order.
  const filteredPositions = new Float32Array(numOwned * rank);
  for (let i = 0; i < numOwned; ++i) {
    const v = owned[i];
    for (let d = 0; d < rank; ++d) {
      filteredPositions[i * rank + d] = positions[v * rank + d];
    }
  }

  // Filter edges: keep only those whose endpoints are both in the owned
  // set, and remap to filtered-output indices.
  const keptEdges: number[] = [];
  for (let e = 0; e < edges.length; e += 2) {
    const a = edges[e];
    const b = edges[e + 1];
    if (seen[a] === 1 && seen[b] === 1) {
      keptEdges.push(vertexRemap[a]);
      keptEdges.push(vertexRemap[b]);
    }
  }
  const filteredEdges = new Uint32Array(keptEdges);

  // Filter attribute arrays (tangents first if present, then user attrs)
  // in the same conventional order the spatially-indexed backend uses.
  const filteredAttrs: AttributeTypedArray[] = [];
  if (tangents !== undefined) {
    const t = new Float32Array(numOwned * 3);
    for (let i = 0; i < numOwned; ++i) {
      const v = owned[i];
      t[i * 3] = tangents[v * 3];
      t[i * 3 + 1] = tangents[v * 3 + 1];
      t[i * 3 + 2] = tangents[v * 3 + 2];
    }
    filteredAttrs.push(t);
  }
  for (const src of vertexAttributes) {
    // Each `src` is a per-vertex array of length `numVertices` (scalar
    // attribute) — the higher-level zarr-vectors writer paths don't
    // currently emit multi-component vertex attributes via the
    // ZarrVectorsAttributeDtype enum, so a 1:1 element copy suffices.
    const Ctor = src.constructor as new (n: number) => AttributeTypedArray;
    const dst = new Ctor(numOwned);
    for (let i = 0; i < numOwned; ++i) dst[i] = src[owned[i]] as never;
    filteredAttrs.push(dst);
  }

  return {
    positions: filteredPositions,
    edges: filteredEdges,
    attributes: filteredAttrs,
    vertexRemap,
  };
}

export interface DownloadSegmentSkeletonOptions {
  /** Manifest reader configuration (numObjects, chunkSize, sidNdim, kvStoreRead). */
  readonly manifestReader: ObjectManifestReaderOptions;
  /** Spatial-chunk download parameters (rank, dtypes, links convention, etc.). */
  readonly rank: number;
  readonly linkDtype: LinkDtype;
  readonly attributeNames: readonly string[];
  readonly attributeDtypes: readonly AttributeDtype[];
  readonly linksConvention: LinksConvention;
  readonly geometryKind: SkeletonGeometryKind;
  /**
   * Optional decoded ``cross_chunk_links/0/`` table for the level.  When
   * present, ``downloadSegmentSkeleton`` appends one edge per record
   * whose two endpoints both land on vertices the current object owns
   * (i.e. survived the per-block fragment filter).  Records of
   * ``linkWidth !== 2`` are ignored — they're for meshes / metanode
   * pyramids, not streamlines.
   *
   * Prefer {@link queryCrossChunkLinksForChunks} for ``implicit_sequential``
   * (streamline/polyline) stores — a real dataset's whole-level table can
   * be tens of millions of records / multiple gigabytes, and decoding it
   * just to resolve one object's handful of cross-chunk edges can OOM the
   * tab. This field remains for callers that already have a cheap
   * pre-fetched table (e.g. graphs/skeletons, or tests).
   */
  readonly crossChunkLinks?: CrossChunkLinksTable;
  /**
   * Scoped alternative to {@link crossChunkLinks}: given the list of
   * chunk-coordinate keys the object's manifest actually touches (known
   * only after the manifest walk completes), returns a table containing
   * just the records incident on those chunks — via
   * {@link readCrossChunkLinksForChunk} per chunk, sharing one
   * shard-discovery/byte cache across the whole query and across every
   * object download for the level. Checked only when
   * {@link crossChunkLinks} is not supplied.
   */
  readonly queryCrossChunkLinksForChunks?: (
    chunkCoordsList: readonly (readonly number[])[],
    signal: AbortSignal,
  ) => Promise<CrossChunkLinksTable | undefined>;
  /** Fetch + decode one per-chunk array's whole payload — see `sharded_array.ts`. */
  readonly readArrayChunk: ReadArrayChunk;
  /**
   * Optional byte-range-scoped reader. When supplied (and `linksConvention
   * === "implicit_sequential"`), each chunk is fetched via
   * {@link downloadSkeletonChunkScoped} — reading only the selected
   * object's fragments' vertices/attributes from the (uncompressed,
   * `cell_codec: "raw"`) `vertices`/`vertex_attributes` arrays, instead of
   * the whole chunk. The backend should only pass this when the level's
   * kvstore supports offset reads AND those arrays are raw; a missing/
   * failed range read for a given block falls back to the whole-chunk
   * {@link downloadSkeletonChunk}.
   */
  readonly readArrayChunkScoped?: ReadArrayChunkScoped;
  /** Whether to fetch `fragment_attributes/segment_id` per chunk. */
  readonly hasFragmentSegmentIds?: boolean;
}

/**
 * Per-source-chunk bookkeeping kept while {@link downloadSegmentSkeleton}
 * processes a manifest.  Used after concatenation to translate
 * cross-chunk endpoint references into the merged-output vertex index
 * space.
 */
interface OwnedChunkInfo {
  /** Map from chunk-local vertex index → filtered-output position (-1 = dropped). */
  readonly vertexRemap: Int32Array;
  /** Cumulative merged-output index at which this chunk's vertices start. */
  readonly vertexOffset: number;
}

/**
 * Pure helper: given a decoded cross-chunk table and the per-chunk
 * remap/offset info collected during fragment aggregation, emit the
 * subset of edges whose endpoints both land on owned vertices.
 *
 * Exported so unit tests can drive it with hand-crafted fixtures
 * without staging a whole manifest/chunk pipeline.
 */
/**
 * Per-block bookkeeping carried by {@link downloadSegmentSkeleton}'s
 * manifest walk.  Exposed via the helper signatures so unit tests can
 * drive {@link deriveImplicitSequentialCrossChunkEdges} without
 * staging a whole download.
 */
export interface OrderedManifestBlock {
  /** Joined chunk-coordinate string, e.g. ``"0.-1.2"``. */
  readonly chunkKey: string;
  /** Maps chunk-local vertex index → filtered-output position (-1 = dropped). */
  readonly vertexRemap: Int32Array;
  /** Cumulative merged-output index at which this block's vertices start. */
  readonly vertexOffset: number;
  /** Chunk-local first vertex of this block's single fragment (-1 if N/A). */
  readonly firstFragmentLocalVert: number;
  /** Chunk-local last vertex of this block's single fragment (-1 if N/A). */
  readonly lastFragmentLocalVert: number;
}

/**
 * Pure helper for the ``implicit_sequential`` inter-fragment bridge
 * *fallback* path — used only when no cross_chunk_links-derived edges
 * were available (see the call site in `downloadSegmentSkeleton`).
 * Walks the manifest-ordered blocks pairwise; for **every** consecutive
 * pair emits one edge bridging fragment k's last vertex with fragment
 * k+1's first vertex (both translated to the merged-output vertex index
 * space).
 *
 * Correct only when manifest order matches true path order. Guaranteed
 * for a store's source (finest) level, written directly by the ingest
 * pipeline in walk order — NOT guaranteed for a coarsened level: the
 * zarr-vectors-tools pyramid coarsener's per-object manifests are sorted
 * by chunk coordinate, not path order, so this function can bridge two
 * physically distant blocks that happen to be adjacent post-sort.
 *
 * Bridges connect *consecutive fragments*, not just *cross-chunk*
 * transitions.  Streamlines are partitioned by zarr-vectors' bin grid
 * (writer default: ``bin_shape`` = chunk_shape / 4), so a polyline can
 * generate multiple fragments **inside one chunk**.  Each fragment is
 * its own implicit-sequential edge run, so adjacent fragments — same
 * chunk or not — need an explicit bridge between fragment k's last
 * vertex and fragment k+1's first vertex.  Skipping same-chunk
 * transitions would leave intra-chunk bin-boundary gaps visible.
 *
 * Skips pairs where either side isn't a single-fragment block
 * (``firstFragmentLocalVert`` / ``lastFragmentLocalVert`` are -1) — the
 * endpoint identity becomes ambiguous in that case.  Also skips pairs
 * where the relevant chunk-local vertex was filtered out (remap < 0) —
 * same no-dangling rule the per-chunk filter applies.
 *
 * Exported so unit tests can drive it with hand-crafted block sequences
 * without staging a whole download.
 */
export function deriveImplicitSequentialCrossChunkEdges(
  orderedBlocks: readonly OrderedManifestBlock[],
): Uint32Array {
  const out: number[] = [];
  for (let i = 0; i + 1 < orderedBlocks.length; ++i) {
    const a = orderedBlocks[i];
    const b = orderedBlocks[i + 1];
    if (a.lastFragmentLocalVert < 0 || b.firstFragmentLocalVert < 0) continue;
    if (a.lastFragmentLocalVert >= a.vertexRemap.length) continue;
    if (b.firstFragmentLocalVert >= b.vertexRemap.length) continue;
    const aRemap = a.vertexRemap[a.lastFragmentLocalVert];
    const bRemap = b.vertexRemap[b.firstFragmentLocalVert];
    if (aRemap < 0 || bRemap < 0) continue;
    out.push(aRemap + a.vertexOffset);
    out.push(bRemap + b.vertexOffset);
  }
  return new Uint32Array(out);
}

/**
 * Per-chunk map from a chunk-local vertex index to that vertex's position
 * in the merged-output vertex array (the value already includes the owning
 * block's `vertexOffset`).  Accumulated across **every** manifest block of a
 * chunk — an object routinely owns many fragments in one chunk (mouselight:
 * up to ~150), each arriving as its own `single`-mode manifest block, so a
 * per-chunk last-write-wins `vertexRemap`/`vertexOffset` (the old
 * {@link OwnedChunkInfo} keying) would only resolve the *last* block's
 * vertices and silently drop every cross-chunk edge whose endpoint lives in
 * an earlier block. Cross-chunk endpoints are chunk-local (as written by
 * `write_skeleton_cross_chunk_links`), so this maps them straight to the
 * merged index space.
 */
export type ChunkLocalVertexIndexMap = Map<string, Map<number, number>>;

export function collectOwnedCrossChunkEdges(
  table: CrossChunkLinksTable,
  chunkVertexMaps: ChunkLocalVertexIndexMap,
): Uint32Array {
  // Only line-arity (linkWidth=2) records describe cross-chunk edges.
  // Triangle / metanode records aren't relevant to streamline rendering.
  if (table.linkWidth !== 2) return new Uint32Array(0);
  const out: number[] = [];
  for (const record of table.records) {
    const [a, b] = record.endpoints;
    const aMap = chunkVertexMaps.get(a.chunkCoords.join("."));
    const bMap = chunkVertexMaps.get(b.chunkCoords.join("."));
    if (aMap === undefined || bMap === undefined) continue;
    const aGlobal = aMap.get(a.vertexIndex);
    const bGlobal = bMap.get(b.vertexIndex);
    // A missing entry means the endpoint's vertex isn't owned by this object
    // (belongs to another object's fragment in the same chunk) — skip it, no
    // dangling edge.
    if (aGlobal === undefined || bGlobal === undefined) continue;
    out.push(aGlobal);
    out.push(bGlobal);
  }
  return new Uint32Array(out);
}

/**
 * Download and aggregate one object's skeleton geometry across all the
 * chunks the manifest reports for it.  Returns `undefined` when the
 * object is absent (no manifest, or every fragment chunk missing).
 *
 * Algorithm:
 *
 * 1. Resolve `oid` → `ManifestBlock[]` via `readObjectManifest`.
 * 2. For each block:
 *    a. Fetch + decode the spatial chunk via `downloadSkeletonChunk`.
 *    b. Resolve `block.fragmentRef` to a flat list of fragment indices
 *       within that chunk.
 *    c. Call `filterChunkByFragments` to extract just those fragments'
 *       vertices/edges/attributes.
 * 3. Concatenate the per-chunk filtered outputs, re-offsetting the edge
 *    indices so they reference the merged vertex array.
 */
export async function downloadSegmentSkeleton(
  oid: number | bigint,
  options: DownloadSegmentSkeletonOptions,
  signal: AbortSignal,
): Promise<AggregatedSegmentSkeleton | undefined> {
  const {
    manifestReader,
    rank,
    linkDtype,
    attributeNames,
    attributeDtypes,
    linksConvention,
    geometryKind,
    crossChunkLinks,
    queryCrossChunkLinksForChunks,
    readArrayChunk,
    readArrayChunkScoped,
    hasFragmentSegmentIds,
  } = options;
  const manifest = await readObjectManifest(oid, manifestReader, signal);
  if (manifest === undefined || manifest.length === 0) return undefined;

  const perChunkPositions: Float32Array[] = [];
  const perChunkEdges: Uint32Array[] = [];
  // Outer array: one slot per attribute.  Inner: one entry per source
  // chunk.  Every geometry kind with synthesised tangents (streamline,
  // polyline, graph) carries the tangent in slot 0 of
  // `filterChunkByFragments`'s output — see `hasSynthesisedTangent` in
  // `geometry_kind.ts` for the canonical per-kind capability table.
  const numAttrsExpected =
    (hasSynthesisedTangent(geometryKind) ? 1 : 0) + attributeNames.length;
  const perChunkAttrs: AttributeTypedArray[][] = Array.from(
    { length: numAttrsExpected },
    () => [] as AttributeTypedArray[],
  );
  // Per-chunk chunk-local-vertex → merged-output-index map. This is what the
  // blob-based cross-chunk edge resolver uses to translate a
  // cross_chunk_links endpoint (chunk, chunk-local vertex) into the merged
  // vertex array. Built once per chunk (all the object's fragments in a chunk
  // are filtered together), so it covers every owned vertex. See
  // `collectOwnedCrossChunkEdges` / `ChunkLocalVertexIndexMap`.
  const chunkVertexMaps: ChunkLocalVertexIndexMap = new Map();
  // Distinct chunk-coords this object's manifest touches, in first-seen
  // order — fed to `queryCrossChunkLinksForChunks` once the walk below
  // completes, so the caller can scope its cross_chunk_links query to
  // just these chunks instead of decoding a whole level.
  const ownedChunkCoordsList: number[][] = [];
  const seenChunkKeys = new Set<string>();
  // Per-block bookkeeping, in manifest order.  Drives the
  // implicit_sequential cross-chunk *fallback* path (see
  // `deriveImplicitSequentialCrossChunkEdges`'s docstring): consecutive
  // blocks in different chunks emit one bridging edge (last vertex of
  // fragment k → first vertex of fragment k+1). Only used when no
  // real cross_chunk_links data is available.
  interface OrderedBlock extends OwnedChunkInfo {
    readonly chunkKey: string;
    /** Chunk-local index of the first vertex of this block's single
     * fragment; -1 if the block has 0 or >1 fragments (cross-chunk edge
     * reconstruction skips those). */
    readonly firstFragmentLocalVert: number;
    /** Chunk-local index of the last vertex of this block's single
     * fragment; -1 if not single-fragment. */
    readonly lastFragmentLocalVert: number;
  }
  const orderedBlocks: OrderedBlock[] = [];

  // Cache whole-chunk downloads by chunk key.  An object's manifest can list
  // MANY fragment blocks in the SAME chunk — most extreme at coarsened pyramid
  // levels, where all of an object's fragments can collapse into a single
  // chunk (e.g. hundreds of fragments in one level-3 chunk).  Without this,
  // each block re-downloads — and for a native-sharded array, re-range-reads —
  // that chunk's ENTIRE cell, turning one object load into O(fragments)
  // whole-cell reads of the identical bytes (observed as thousands of
  // duplicate range requests / gigabytes transferred).  Reading each distinct
  // chunk once and filtering per block makes it O(distinct chunks).  Only the
  // whole-chunk path is cached; the `implicit_sequential` scoped path
  // byte-range-reads per fragment and isn't keyed by chunk alone.
  const wholeChunkCache = new Map<string, SkeletonChunk | undefined>();
  const readWholeChunkCached = async (
    key: string,
    options: Parameters<typeof downloadSkeletonChunk>[0],
  ): Promise<SkeletonChunk | undefined> => {
    if (wholeChunkCache.has(key)) return wholeChunkCache.get(key);
    const s = await downloadSkeletonChunk(options, signal);
    wholeChunkCache.set(key, s);
    return s;
  };

  // Group the manifest's blocks by chunk so every fragment an object owns
  // in a chunk is filtered TOGETHER, in a single `filterChunkByFragments`
  // call. This is essential for `implicit_sequential_with_branches`
  // skeletons: within-chunk connectivity between fragments lives entirely
  // in inter-fragment branch links (`links/0/<chunk>`), and
  // `filterChunkByFragments` keeps an edge only when BOTH endpoints are
  // owned. The on-disk manifest stores one `single`-mode block per fragment
  // (mouselight: up to ~150 per chunk), so filtering one fragment at a time
  // would drop every branch link (each has an endpoint in another fragment
  // not in that block's `seen` set), shattering each neuron into
  // per-fragment pieces that only cross-chunk links could partly rejoin —
  // leaving empty gaps. Grouping also gives `chunkVertexMaps` one whole-chunk
  // entry per chunk (no last-write-wins).
  interface ChunkGroup {
    readonly chunkCoords: number[];
    readonly fragmentIds: number[];
  }
  const chunkGroups = new Map<string, ChunkGroup>();
  const chunkOrder: string[] = [];
  for (const block of manifest) {
    const key = block.chunkCoords.join(".");
    const ids = resolveFragmentRef(block.fragmentRef);
    let group = chunkGroups.get(key);
    if (group === undefined) {
      group = { chunkCoords: block.chunkCoords, fragmentIds: [] };
      chunkGroups.set(key, group);
      chunkOrder.push(key);
    }
    for (const id of ids) group.fragmentIds.push(id);
  }

  // Per-chunk decode kept so the implicit_sequential fallback's per-fragment
  // ordering pass (below) can map each fragment's endpoints through its
  // chunk's shared remap/offset without re-downloading.
  const perChunkDecoded = new Map<
    string,
    {
      fragmentIndex: SkeletonChunk["fragmentIndex"];
      vertexRemap: Int32Array;
      vertexOffset: number;
    }
  >();

  let runningVertexOffset = 0;

  for (const chunkKey of chunkOrder) {
    const group = chunkGroups.get(chunkKey)!;
    // Dedup the owned fragment ids (a chunk revisited by the manifest, or an
    // overlapping range/explicit ref, could list one twice).
    const fragmentIds = Uint32Array.from(new Set(group.fragmentIds));
    const wholeChunkOptions = {
      chunkCoords: group.chunkCoords,
      rank,
      linkDtype,
      attributeNames,
      attributeDtypes,
      linksConvention,
      geometryKind,
      hasFragmentSegmentIds,
      readArrayChunk,
    };
    let skel: SkeletonChunk | undefined;
    if (
      readArrayChunkScoped !== undefined &&
      linksConvention === "implicit_sequential"
    ) {
      try {
        skel = await downloadSkeletonChunkScoped(
          {
            ...wholeChunkOptions,
            readArrayChunkScoped,
            restrictToFragments: fragmentIds,
          },
          signal,
        );
      } catch {
        // A range read that came back the wrong length (a store that
        // silently ignored the request) throws — fall back to the whole-
        // chunk read for this chunk so rendering still succeeds.
        skel = await readWholeChunkCached(chunkKey, wholeChunkOptions);
      }
    } else {
      skel = await readWholeChunkCached(chunkKey, wholeChunkOptions);
    }
    if (skel === undefined) continue;

    // Filter ALL the object's owned fragments in this chunk at once — so
    // `seen` covers them and inter-fragment branch links are retained.
    const filtered = filterChunkByFragments(skel, fragmentIds);
    if (filtered.positions.length === 0) continue;

    // Chunk-local vertex → merged-output index, for the blob-based
    // cross-chunk resolver. `filtered.vertexRemap` spans the whole chunk's
    // vertex count, so `vi` is the true chunk-local index that
    // cross_chunk_links endpoints reference.
    const chunkVertexMap = new Map<number, number>();
    chunkVertexMaps.set(chunkKey, chunkVertexMap);
    for (let vi = 0; vi < filtered.vertexRemap.length; ++vi) {
      const local = filtered.vertexRemap[vi];
      if (local >= 0) chunkVertexMap.set(vi, local + runningVertexOffset);
    }
    seenChunkKeys.add(chunkKey);
    ownedChunkCoordsList.push(group.chunkCoords);
    perChunkDecoded.set(chunkKey, {
      fragmentIndex: skel.fragmentIndex,
      vertexRemap: filtered.vertexRemap,
      vertexOffset: runningVertexOffset,
    });

    perChunkPositions.push(filtered.positions);
    // Shift edge indices into the merged-output coordinate space.
    if (filtered.edges.length > 0) {
      if (runningVertexOffset === 0) {
        perChunkEdges.push(filtered.edges);
      } else {
        const shifted = new Uint32Array(filtered.edges.length);
        for (let i = 0; i < filtered.edges.length; ++i) {
          shifted[i] = filtered.edges[i] + runningVertexOffset;
        }
        perChunkEdges.push(shifted);
      }
    }
    if (filtered.attributes.length !== numAttrsExpected) {
      throw new Error(
        `downloadSegmentSkeleton: chunk ${chunkKey} returned ` +
          `${filtered.attributes.length} attributes; expected ${numAttrsExpected}`,
      );
    }
    for (let i = 0; i < numAttrsExpected; ++i) {
      perChunkAttrs[i].push(filtered.attributes[i]);
    }

    runningVertexOffset += filtered.positions.length / rank;
  }

  // Build `orderedBlocks` per-fragment in manifest order — drives the
  // `implicit_sequential` cross-chunk *fallback*
  // (`deriveImplicitSequentialCrossChunkEdges`), which bridges consecutive
  // fragments. All fragments in a chunk share that chunk's remap/offset, so
  // a fragment's first/last chunk-local vertex maps to the merged space the
  // same way regardless of which fragment it is.
  for (const block of manifest) {
    const chunkKey = block.chunkCoords.join(".");
    const decoded = perChunkDecoded.get(chunkKey);
    if (decoded === undefined) continue;
    const ids = resolveFragmentRef(block.fragmentRef);
    let firstFragmentLocalVert = -1;
    let lastFragmentLocalVert = -1;
    if (ids.length === 1) {
      const fragVerts = decoded.fragmentIndex.indices(ids[0]);
      if (fragVerts.length > 0) {
        firstFragmentLocalVert = fragVerts[0];
        lastFragmentLocalVert = fragVerts[fragVerts.length - 1];
      }
    }
    orderedBlocks.push({
      vertexRemap: decoded.vertexRemap,
      vertexOffset: decoded.vertexOffset,
      chunkKey,
      firstFragmentLocalVert,
      lastFragmentLocalVert,
    });
  }

  // Inter-fragment bridge reconstruction.  Two strategies:
  //
  //  - Preferred, whenever the on-disk cross_chunk_links blob exists and
  //    yields edges: the blob-based resolver over `chunkVertexMaps`
  //    (`collectOwnedCrossChunkEdges`), using real chunk-local vertex
  //    indices for each endpoint. This is what `explicit` /
  //    `implicit_sequential_with_branches` (graphs, skeletons) always
  //    used; `implicit_sequential` (polylines/streamlines) now prefers
  //    it too — the current writer (zarr-vectors-py's
  //    `types/polylines.py`) stores real endpoints for cross-chunk
  //    transitions, not placeholders (an earlier revision of this
  //    comment claimed otherwise, based on stale information).
  //
  //  - Fallback, only when no usable blob-based edges were found (older
  //    writer, or a store that genuinely never recorded real endpoints
  //    for this convention): walk `orderedBlocks` pairwise; emit one
  //    edge per consecutive pair, connecting fragment k's last vertex to
  //    fragment k+1's first vertex. This is `implicit_sequential`-only —
  //    `deriveImplicitSequentialCrossChunkEdges` assumes manifest order
  //    equals path order, which the source-level writer guarantees but a
  //    multi-resolution pyramid's coarsened levels do NOT: the coarsener
  //    (`zarr_vectors_tools`'s `_reduce_object_index_shard`, shared by
  //    the polyline and skeleton coarseners) sorts each object's
  //    per-level manifest blocks by chunk coordinate, not path order, so
  //    this reconstruction can bridge two blocks that are adjacent in
  //    the (re-sorted) manifest but nowhere near each other on the
  //    actual path — producing a long spurious "ghost" edge. Prefer the
  //    blob-based path above whenever possible; this fallback exists for
  //    compatibility with stores lacking real link data at all.
  let crossChunkEdges: Uint32Array | undefined;
  let resolvedCrossChunkLinks = crossChunkLinks;
  if (
    resolvedCrossChunkLinks === undefined &&
    queryCrossChunkLinksForChunks !== undefined &&
    ownedChunkCoordsList.length > 0
  ) {
    // Cross-chunk links are OPTIONAL connectivity — a failure to read them
    // (missing table, layout quirk, a store that errors on a 404 probe)
    // must NOT abort the whole skeleton: without them the object still
    // renders as its per-chunk fragments (just not bridged across chunk
    // boundaries), which is far better than rendering nothing.
    try {
      resolvedCrossChunkLinks = await queryCrossChunkLinksForChunks(
        ownedChunkCoordsList,
        signal,
      );
    } catch (e) {
      console.warn(
        "zarr-vectors: cross-chunk link query failed; rendering fragments " +
          "without cross-chunk bridges",
        e,
      );
      resolvedCrossChunkLinks = undefined;
    }
  }
  if (resolvedCrossChunkLinks !== undefined && chunkVertexMaps.size > 0) {
    const edges = collectOwnedCrossChunkEdges(
      resolvedCrossChunkLinks,
      chunkVertexMaps,
    );
    if (edges.length > 0) crossChunkEdges = edges;
  }
  if (crossChunkEdges === undefined && linksConvention === "implicit_sequential") {
    const edges = deriveImplicitSequentialCrossChunkEdges(orderedBlocks);
    if (edges.length > 0) crossChunkEdges = edges;
  }

  if (runningVertexOffset === 0) return undefined;

  // Concatenate per-chunk arrays.
  const totalFloats = runningVertexOffset * rank;
  const vertexPositions = new Float32Array(totalFloats);
  {
    let cursor = 0;
    for (const p of perChunkPositions) {
      vertexPositions.set(p, cursor);
      cursor += p.length;
    }
  }
  let totalEdgeEntries = 0;
  for (const e of perChunkEdges) totalEdgeEntries += e.length;
  if (crossChunkEdges !== undefined) totalEdgeEntries += crossChunkEdges.length;
  const indices = new Uint32Array(totalEdgeEntries);
  {
    let cursor = 0;
    for (const e of perChunkEdges) {
      indices.set(e, cursor);
      cursor += e.length;
    }
    if (crossChunkEdges !== undefined) {
      indices.set(crossChunkEdges, cursor);
    }
  }
  const vertexAttributes: AttributeTypedArray[] = [];
  for (let a = 0; a < numAttrsExpected; ++a) {
    const parts = perChunkAttrs[a];
    // Each attribute keeps its dtype consistent across chunks because
    // the dtypes come from per-array zarr metadata, not per-chunk.
    // Use the first non-empty part's constructor to allocate the merged
    // buffer (fallback to Float32Array if all parts are zero-length).
    let totalLen = 0;
    for (const p of parts) totalLen += p.length;
    let merged: AttributeTypedArray;
    if (parts.length === 0 || totalLen === 0) {
      merged = new Float32Array(0);
    } else {
      const Ctor = parts[0].constructor as new (
        n: number,
      ) => AttributeTypedArray;
      merged = new Ctor(totalLen);
      let cursor = 0;
      for (const p of parts) {
        (
          merged as unknown as {
            set: (a: ArrayLike<number>, o: number) => void;
          }
        ).set(p as unknown as ArrayLike<number>, cursor);
        cursor += p.length;
      }
    }
    vertexAttributes.push(merged);
  }

  // Fix up the synthesised tangent (slot 0) across the fully merged graph so
  // cross-chunk bridges don't render as black bands under
  // `abs(prop_tangent())`. Two failure modes, both from per-chunk tangent
  // synthesis that never saw the bridge edges:
  //   1. Opposing SIGNS — each chunk sign-oriented independently, so a bridge
  //      can join `+t` to `-t`; the segment interpolates through zero.
  //   2. ZERO magnitude — a fragment that is a single vertex (or otherwise
  //      has no intra-chunk neighbour) is degree-0 in its chunk and gets a
  //      zero tangent; once bridged it interpolates from a real tangent to
  //      zero → a black half-segment.
  // For edge-adjacency kinds (skeleton/graph) we RECOMPUTE tangents from the
  // merged graph with the same `computeTangentsFromEdges` the per-chunk decode
  // used — now every boundary/singleton vertex has its bridge neighbour, so it
  // gets a real direction, and the built-in flood-fill orients signs across
  // the whole object (fixing both modes at once). Walk-order kinds
  // (streamline/polyline) keep their per-chunk walk-order tangents and only
  // need the cheaper sign re-orientation across bridges.
  if (
    hasSynthesisedTangent(geometryKind) &&
    vertexAttributes.length > 0 &&
    vertexAttributes[0] instanceof Float32Array &&
    vertexAttributes[0].length === runningVertexOffset * 3
  ) {
    if (KIND_CAPABILITIES[geometryKind].hasEdgeAdjacencyTangent) {
      vertexAttributes[0] = computeTangentsFromEdges(
        vertexPositions,
        rank,
        indices,
        runningVertexOffset,
      );
    } else {
      orientTangentSignsAcrossEdges(
        vertexAttributes[0],
        indices,
        runningVertexOffset,
      );
    }
  }

  return { vertexPositions, indices, vertexAttributes };
}
