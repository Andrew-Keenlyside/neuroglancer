/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Worker-side `SharedObject` chunk-source backends for zarr-vectors
 * skeleton / polyline / streamline rendering.  Provides:
 *
 * - `ZarrVectorsSpatiallyIndexedSkeletonSourceBackend` — the **pass-1**
 *   backing store.  Subclasses neuroglancer's existing
 *   `SpatiallyIndexedSkeletonSourceBackend` and overrides `download()`
 *   to fetch + decode zarr-vectors chunks via the
 *   `downloadSkeletonChunk()` orchestrator.
 *
 * - `ZarrVectorsObjectKeyedSkeletonSourceBackend` — the **pass-2**
 *   backing store, intentionally **not implemented in this slice**.
 *   Will subclass `SkeletonSource` once the `object_index/manifests`
 *   zarr-vlen-bytes reader is in place (slice 4b).
 *
 * Mirrors the CATMAID pattern at
 * `src/datasource/catmaid/backend.ts:40-83`.
 */

import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import { WithParameters } from "#src/chunk_manager/backend.js";
import {
  ZarrVectorsMultiscaleObjectKeyedSkeletonSourceParameters,
  ZarrVectorsObjectKeyedSkeletonSourceParameters,
  ZarrVectorsSpatiallyIndexedSkeletonSourceParameters,
  type ZarrVectorsLinkDtype,
  type ZarrVectorsLinksConvention,
  type ZarrVectorsSkeletonGeometryKind,
} from "#src/datasource/zarr-vectors/base.js";
import {
  createCrossChunkLinksCaches,
  readCrossChunkLinks,
  readCrossChunkLinksForChunk,
  type CrossChunkLinksCaches,
  type CrossChunkLinksTable,
} from "#src/datasource/zarr-vectors/cross_chunk_links.js";
import { hasSynthesisedTangent } from "#src/datasource/zarr-vectors/geometry_kind.js";
import {
  appendGhostVertices,
  appendIntraChunkEdges,
  recomputeTangentsForBridges,
  type ResolvedBridge,
} from "#src/datasource/zarr-vectors/skeleton_chunk.js";
import {
  downloadSkeletonChunk,
  fetchGhostVertices,
  type AttributeDtype,
  type GhostVertexRequest,
  type KvStoreReadRange,
  type LinkDtype,
} from "#src/datasource/zarr-vectors/skeleton_chunk_download.js";
import { downloadSegmentSkeleton } from "#src/datasource/zarr-vectors/skeleton_segment_download.js";
import { probeObjectAcrossLevels } from "#src/datasource/zarr-vectors/multiscale_manifest.js";
import { WithSharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import { joinBaseUrlAndPath } from "#src/kvstore/url.js";
import type {
  MultiscaleSkeletonFragmentChunk,
  MultiscaleSkeletonManifestChunk,
  SkeletonChunk,
  SpatiallyIndexedSkeletonChunk,
} from "#src/skeleton/backend.js";
import {
  MultiscaleSkeletonSource,
  SkeletonSource,
  SpatiallyIndexedSkeletonSourceBackend,
} from "#src/skeleton/backend.js";
import { registerSharedObject } from "#src/worker_rpc.js";

const ZSTD_MAGIC = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd]);

function looksLikeZstd(bytes: Uint8Array): boolean {
  if (bytes.byteLength < 4) return false;
  return (
    bytes[0] === ZSTD_MAGIC[0] &&
    bytes[1] === ZSTD_MAGIC[1] &&
    bytes[2] === ZSTD_MAGIC[2] &&
    bytes[3] === ZSTD_MAGIC[3]
  );
}

/**
 * Decompress a zstd-framed byte buffer; pass through other formats.
 *
 * Duplicated from the point-cloud backend.ts intentionally — keeps this
 * slice's diff scoped to new files only.  If a third caller appears,
 * promote both copies to a shared helper module.
 */
async function maybeDecompress(
  bytes: Uint8Array<ArrayBuffer>,
  signal: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  if (!looksLikeZstd(bytes)) return bytes;
  return await requestAsyncComputation(
    decodeZstd,
    signal,
    [bytes.buffer],
    bytes,
  );
}

/**
 * Build a `kvStoreRead` callback bound to a base URL and the worker-side
 * shared kvstore context.  Resolves to a decompressed `Uint8Array` (or
 * `undefined` for a missing key).
 */
function makeKvStoreRead(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      read: (url: string, options: { signal: AbortSignal }) => Promise<any>;
    };
  },
) {
  return async (
    subpath: string,
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined> => {
    const url = joinBaseUrlAndPath(baseUrl, subpath);
    const response = await sharedKvStoreContext.kvStoreContext.read(url, {
      signal,
    });
    if (response === undefined) return undefined;
    const bytes = new Uint8Array(
      (await response.response.arrayBuffer()) as ArrayBuffer,
    );
    return await maybeDecompress(bytes, signal);
  };
}

/**
 * Build a `kvStoreRead` callback like {@link makeKvStoreRead} but WITHOUT
 * the magic-byte-sniffing auto-decompress step.  Required by
 * {@link readCrossChunkLinks}: a v0.8 cross-chunk-links shard file is a
 * multi-cell concatenation (each cell independently zstd-framed), not a
 * single zstd stream, so auto-decompressing the whole shard blob would
 * corrupt the shard-index/cell-boundary parsing.
 */
function makeRawKvStoreRead(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      read: (url: string, options: { signal: AbortSignal }) => Promise<any>;
    };
  },
) {
  return async (
    subpath: string,
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined> => {
    const url = joinBaseUrlAndPath(baseUrl, subpath);
    const response = await sharedKvStoreContext.kvStoreContext.read(url, {
      signal,
    });
    if (response === undefined) return undefined;
    return new Uint8Array(
      (await response.response.arrayBuffer()) as ArrayBuffer,
    );
  };
}

/**
 * Whether the store backing `baseUrl` supports byte-offset reads. Only
 * such stores can serve the fragment-scoped vertex range reads
 * ({@link downloadSkeletonChunkScoped}); issuing a range read to a store
 * that can't honor it risks a hung request (a hang can't be caught, so
 * gate BEFORE issuing rather than catching after). Returns false on any
 * probe failure so the caller safely falls back to whole-chunk reads.
 *
 * Scoped reads are additionally gated on the writer having stamped
 * `vertices_layout: "raw_v1"` (surfaced as `verticesRangeAddressable`):
 * a compressed `vertices` chunk is not byte-range-addressable, so a range
 * read of it would decode to garbage. See the pass-2 backends' download
 * paths and `frontend.ts buildSkeletonMetadata`.
 *
 * NOTE: the HTTP driver returns `true` here unconditionally — it cannot
 * tell whether the *server* honors Range — so this is necessary but not
 * sufficient; the data server must return `206` for `Range` requests (and
 * answer the CORS preflight for the `Range` header on cross-origin loads).
 */
function storeSupportsOffsetReads(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      getKvStore: (url: string) => { store: { supportsOffsetReads?: boolean } };
    };
  },
): boolean {
  try {
    const { store } = sharedKvStoreContext.kvStoreContext.getKvStore(baseUrl);
    return store.supportsOffsetReads === true;
  } catch {
    return false;
  }
}

/**
 * Build a byte-range `kvStoreReadRange` callback (see {@link KvStoreReadRange}):
 * reads exactly `[offset, offset+length)` of a `raw`-encoded array chunk
 * with NO auto-decompress. `strictByteRange` makes the read throw if the
 * store didn't honor the range (e.g. a driver without offset-read
 * support returned the whole value) — the caller catches that and falls
 * back to a whole-chunk read. Returns `undefined` for an absent key.
 */
function makeKvStoreReadRange(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      read: (url: string, options: any) => Promise<any>;
    };
  },
): KvStoreReadRange {
  return async (
    subpath: string,
    byteRange: { offset: number; length: number },
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined> => {
    const url = joinBaseUrlAndPath(baseUrl, subpath);
    const response = await sharedKvStoreContext.kvStoreContext.read(url, {
      signal,
      byteRange,
      strictByteRange: true,
    });
    if (response === undefined) return undefined;
    return new Uint8Array(
      (await response.response.arrayBuffer()) as ArrayBuffer,
    );
  };
}

/**
 * Build a `kvStoreList` callback bound to a base URL and the worker-side
 * shared kvstore context.  Used by {@link readCrossChunkLinks} to
 * recursively discover which shards of a v0.8 ``kK`` array are
 * populated.  Mirrors ``listAttributeNames`` in `frontend.ts`, but
 * returns bare (no-trailing-slash) names for both directories and files
 * instead of a single merged list.
 */
function makeKvStoreList(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      list: (
        urlPrefix: string,
        options: { responseKeys: "suffix"; signal: AbortSignal },
      ) => Promise<{ entries: { key: string }[]; directories: string[] }>;
    };
  },
) {
  return async (
    prefix: string,
    signal: AbortSignal,
  ): Promise<{ directories: string[]; files: string[] }> => {
    const url = joinBaseUrlAndPath(baseUrl, prefix);
    const response = await sharedKvStoreContext.kvStoreContext.list(url, {
      responseKeys: "suffix",
      signal,
    });
    const stripTrailingSlash = (s: string) =>
      s.endsWith("/") ? s.slice(0, -1) : s;
    return {
      directories: response.directories.map(stripTrailingSlash),
      files: response.entries.map((e) => stripTrailingSlash(e.key)),
    };
  };
}

/**
 * The parameter types in `base.ts` declare `ZarrVectorsLinkDtype` and
 * `ZarrVectorsAttributeDtype` as union subsets of the orchestrator's
 * dtypes.  The orchestrator's `LinkDtype` / `AttributeDtype` are
 * structurally identical at the value level — the two type names exist
 * separately so the parameter classes can carry semantically-named
 * unions while the orchestrator stays decoupled from the parameter
 * surface.  Cast through here.
 */
function asLinkDtype(d: ZarrVectorsLinkDtype): LinkDtype {
  return d as LinkDtype;
}
function asAttributeDtype(d: string): AttributeDtype {
  return d as AttributeDtype;
}

/**
 * Spatially-indexed skeleton chunk source — the **pass-1** backing
 * store.  One chunk per `(chunkGridPosition, lod)` pair.
 *
 * For each chunk, `download()` fetches the relevant byte blobs and
 * decodes them into a `SpatiallyIndexedSkeletonChunk` whose
 * `vertexPositions`, `indices`, and `vertexAttributes` fields the render
 * layer consumes.
 *
 * Streamline / polyline geometry kinds prepend a synthesised
 * `tangent` vec3 attribute to `vertexAttributes` so the default shader's
 * `prop_tangent()` resolves to the per-vertex unit direction.
 * Skeleton geometry skips this — branching breaks the
 * "direction at this vertex" abstraction.
 */
@registerSharedObject()
export class ZarrVectorsSpatiallyIndexedSkeletonSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(SpatiallyIndexedSkeletonSourceBackend),
  ZarrVectorsSpatiallyIndexedSkeletonSourceParameters,
) {
  /**
   * Shared shard-discovery / shard-byte caches for this level's
   * ``cross_chunk_links/0/`` tree, reused across every chunk's
   * per-chunk-targeted query (see {@link getCrossChunkLinksForChunk}).
   *
   * ``false`` means "probed, store has no such table at all" (older
   * zarr-vectors stores written without
   * ``cross_chunk_strategy="explicit_links"``) — skip all future
   * queries.  ``undefined`` means "not yet probed".
   *
   * Deliberately NOT a cache of decoded records (that was the previous
   * design: a single ``CrossChunkLinksTable`` holding every record in
   * the level, read once and reused for every chunk).  For a real
   * dataset that whole-table decode can be tens of millions of records
   * — multiple gigabytes once V8's per-object overhead is counted for
   * the deeply-nested `record → endpoints[] → {chunkCoords[]}`
   * representation — and, being a plain field rather than a `Chunk`,
   * it was invisible to `ChunkState`'s GPU/system memory budget and
   * never evicted under memory pressure.  `CrossChunkLinksCaches` only
   * retains cheap shard-coordinate lists and byte-budgeted raw
   * (still-compressed) shard bytes; per-chunk queries decode only the
   * records actually needed.  See `cross_chunk_links.ts`'s
   * `CrossChunkLinksCaches` docstring for the full rationale.
   *
   * Mirror of the same field on
   * {@link ZarrVectorsObjectKeyedSkeletonSourceBackend}; the two
   * backends share a parameter type's ``baseUrl`` for the same store
   * level, but each holds its own cache instance.
   */
  private crossChunkLinksCaches_: CrossChunkLinksCaches | false | undefined;

  private async getCrossChunkLinksForChunk(
    targetChunkCoords: readonly number[],
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    if (this.crossChunkLinksCaches_ === false) return undefined;
    if (this.crossChunkLinksCaches_ === undefined) {
      this.crossChunkLinksCaches_ = createCrossChunkLinksCaches();
    }
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      targetChunkCoords,
      this.crossChunkLinksCaches_,
      signal,
    );
    if (table === undefined) {
      this.crossChunkLinksCaches_ = false;
    }
    return table;
  }

  /**
   * Filter the cross-chunk-link table down to records incident on the
   * named chunk.  Output has two kinds of items:
   *
   * - `ghostRequests`: cross-chunk records where one endpoint is in
   *   this chunk and the other is in a different chunk.  Each becomes
   *   a {@link GhostVertexRequest} so the backend can fetch the
   *   neighbor's boundary vertex and append it as a ghost.
   * - `intraChunkEdges`: records where BOTH endpoints are in this
   *   chunk.  At coarser pyramid levels the writer encodes intra-chunk
   *   metavertex-to-metavertex bridges (consecutive same-chunk
   *   fragments of one streamline) here too — no ghost vertex needed
   *   because both endpoints already live in this chunk's vertex
   *   texture.  Each record becomes one flat `(a, b)` pair of chunk-
   *   local vertex indices appended directly to the chunk's edge list.
   *
   * Triangle / metanode records (``linkWidth !== 2``) are skipped —
   * those describe mesh-style face stitching, not line edges.
   */
  private buildBridgeRequests(
    table: CrossChunkLinksTable,
    selfChunkCoords: Float32Array,
    selfNumVertices: number,
  ): {
    ghostRequests: GhostVertexRequest[];
    intraChunkEdges: Uint32Array;
    /**
     * Intra-chunk bridges resolved to chunk-local predecessor/successor
     * indices.  Cross-chunk bridges (one endpoint is a future ghost)
     * are appended later in `download()` once we know each ghost's
     * chunk-local index — see the comment on the ghost-append step.
     */
    intraChunkBridges: ResolvedBridge[];
    /**
     * Per-ghost-request side info needed to extend `intraChunkBridges`
     * after ghosts are appended.  `ghostIsPredecessor[i]` mirrors
     * `ghostRequests[i].isGhostPredecessor`.
     */
    ghostIsPredecessor: boolean[];
  } {
    if (table.linkWidth !== 2) {
      return {
        ghostRequests: [],
        intraChunkEdges: new Uint32Array(0),
        intraChunkBridges: [],
        ghostIsPredecessor: [],
      };
    }
    const selfCoords = Array.from(selfChunkCoords, (v) => Number(v));
    const ghostRequests: GhostVertexRequest[] = [];
    const intraEdges: number[] = [];
    const intraChunkBridges: ResolvedBridge[] = [];
    const ghostIsPredecessor: boolean[] = [];
    for (const record of table.records) {
      // Cross-chunk records encode walk order: endpoint[0] is the
      // PREDECESSOR (last vertex of fragment A) and endpoint[1] is
      // the SUCCESSOR (first vertex of fragment B).  See the polyline
      // writer at `zarr_vectors/types/polylines.py` and the boundary
      // helper at `zarr_vectors/spatial/boundary.py:75-114`.
      //
      // For cross-chunk records, the ghost's tangent must point in the
      // forward walk direction.  When this chunk matches endpoint[0],
      // the ghost is the successor — it sits AFTER the host.  When
      // this chunk matches endpoint[1], the ghost is the predecessor.
      // `appendGhostVertices` flips the synthesised ghost-tangent
      // sign based on `isGhostPredecessor` so both sides of the
      // bridge interpolate the same forward walk direction.
      const a = record.endpoints[0];
      const b = record.endpoints[1];
      const aMatches = endpointMatchesChunk(a.chunkCoords, selfCoords);
      const bMatches = endpointMatchesChunk(b.chunkCoords, selfCoords);
      if (aMatches && bMatches) {
        // Intra-chunk bridge: writer-emitted record connecting two
        // fragments inside the SAME chunk (coarser-pyramid-level
        // metavertex-to-metavertex transition).  Drop the record if
        // either endpoint is out of range.
        if (
          a.vertexIndex < 0 ||
          a.vertexIndex >= selfNumVertices ||
          b.vertexIndex < 0 ||
          b.vertexIndex >= selfNumVertices
        ) {
          continue;
        }
        intraEdges.push(a.vertexIndex, b.vertexIndex);
        intraChunkBridges.push({
          predecessorLocalIdx: a.vertexIndex,
          successorLocalIdx: b.vertexIndex,
        });
      } else if (aMatches) {
        ghostRequests.push({
          hostLocalVertex: a.vertexIndex,
          neighborChunkKey: b.chunkCoords.join("."),
          neighborLocalVertex: b.vertexIndex,
          isGhostPredecessor: false, // ghost is endpoint[1] = successor
        });
        ghostIsPredecessor.push(false);
      } else if (bMatches) {
        ghostRequests.push({
          hostLocalVertex: b.vertexIndex,
          neighborChunkKey: a.chunkCoords.join("."),
          neighborLocalVertex: a.vertexIndex,
          isGhostPredecessor: true, // ghost is endpoint[0] = predecessor
        });
        ghostIsPredecessor.push(true);
      }
      // Neither-match records (not ours) are silently ignored.
    }
    return {
      ghostRequests,
      intraChunkEdges: Uint32Array.from(intraEdges),
      intraChunkBridges,
      ghostIsPredecessor,
    };
  }

  async download(
    chunk: SpatiallyIndexedSkeletonChunk,
    signal: AbortSignal,
  ): Promise<void> {
    const {
      baseUrl,
      rank,
      attributeNames,
      attributeDtypes,
      linksConvention,
      geometryKind,
      linkDtype,
      hasFragmentSegmentIds,
    } = this.parameters;
    const { chunkGridPosition } = chunk;
    const chunkKey = Array.from(chunkGridPosition, (v) => String(v)).join(".");
    const kvStoreRead = makeKvStoreRead(baseUrl, this.sharedKvStoreContext);
    const rawKvStoreRead = makeRawKvStoreRead(baseUrl, this.sharedKvStoreContext);
    const kvStoreList = makeKvStoreList(baseUrl, this.sharedKvStoreContext);

    const decoded = await downloadSkeletonChunk(
      {
        chunkKey,
        rank,
        linkDtype: asLinkDtype(linkDtype),
        attributeNames,
        attributeDtypes: attributeDtypes.map(asAttributeDtype),
        linksConvention: linksConvention as ZarrVectorsLinksConvention,
        geometryKind: geometryKind as ZarrVectorsSkeletonGeometryKind,
        hasFragmentSegmentIds,
        kvStoreRead,
      },
      signal,
    );

    if (decoded === undefined) {
      // Sparse chunk presence — no vertices/<chunk> blob.  Set zero-
      // length buffers so the render layer's draw call short-circuits
      // safely.
      chunk.vertexPositions = new Float32Array(0);
      chunk.indices = new Uint32Array(0);
      chunk.vertexAttributes = attributeNames.map(() => new Float32Array(0));
      return;
    }

    // Pass-1 cross-chunk bridge insertion.  For each cross_chunk_links
    // record incident on this chunk, fetch ONE boundary vertex from the
    // neighbor and append it as a ghost vertex + bridge edge.  Each
    // chunk renders independently with its existing per-chunk-isolated
    // GPU resources, but the visible line is continuous across chunk
    // boundaries.  See the design plan at
    // ~/.claude/plans/i-wanted-you-to-spicy-candy.md (option 3) for
    // the full rationale.
    let withBridges = decoded;
    const table = await this.getCrossChunkLinksForChunk(
      Array.from(chunkGridPosition, (v) => Number(v)),
      rawKvStoreRead,
      kvStoreList,
      signal,
    );
    if (table !== undefined) {
      const {
        ghostRequests,
        intraChunkEdges,
        intraChunkBridges,
        ghostIsPredecessor,
      } = this.buildBridgeRequests(
        table,
        chunkGridPosition,
        decoded.numVertices,
      );
      // Intra-chunk bridges first: both endpoints already live in the
      // chunk's vertex texture, so we just append flat (a, b) pairs to
      // the edges array.  Affects coarser pyramid levels where the
      // writer encodes metavertex-to-metavertex transitions inside one
      // chunk via cross_chunk_links records with same-chunk endpoints.
      if (intraChunkEdges.length > 0) {
        withBridges = appendIntraChunkEdges(withBridges, intraChunkEdges);
      }
      // Cross-chunk bridges next: fetch neighbor boundary data and
      // append ghost vertices with bridge edges.
      const resolvedBridges: ResolvedBridge[] = [...intraChunkBridges];
      if (ghostRequests.length > 0) {
        const ghosts = await fetchGhostVertices(
          ghostRequests,
          {
            rank,
            attributeNames,
            attributeDtypes: attributeDtypes.map(asAttributeDtype),
            kvStoreRead,
          },
          signal,
        );
        if (ghosts.length > 0) {
          // Note: `fetchGhostVertices` drops requests whose neighbor
          // data is missing.  The remaining ghosts are appended in
          // order; ghost `g` lands at chunk-local index
          // `decodedNumVertices + intraChunkAppend + g`.
          //
          // We track each ghost's `isPredecessor` so the resolved
          // bridge points its predecessor/successor sides correctly
          // for tangent accumulation.  Drop alignment with the
          // original requests by matching `ghosts[g].bridgeFromLocalVertex`
          // back to `ghostRequests` — but in practice
          // `fetchGhostVertices` preserves request order; just skip
          // the dropped requests.
          const baseGhostIndex = withBridges.numVertices;
          withBridges = appendGhostVertices(withBridges, ghosts);
          // Walk ghosts and ghostRequests in parallel to build
          // resolved bridges.  Use bridgeFromLocalVertex to match
          // each surviving ghost back to its original request.
          let requestCursor = 0;
          for (let g = 0; g < ghosts.length; ++g) {
            const ghost = ghosts[g];
            // Advance requestCursor to the first request matching this
            // ghost's host vertex.  fetchGhostVertices is order-
            // preserving, so this is monotonic.
            while (
              requestCursor < ghostRequests.length &&
              ghostRequests[requestCursor].hostLocalVertex !==
                ghost.bridgeFromLocalVertex
            ) {
              requestCursor++;
            }
            if (requestCursor >= ghostRequests.length) break;
            const isPredecessor = ghostIsPredecessor[requestCursor];
            requestCursor++;
            const ghostIdx = baseGhostIndex + g;
            const hostIdx = ghost.bridgeFromLocalVertex;
            resolvedBridges.push({
              predecessorLocalIdx: isPredecessor ? ghostIdx : hostIdx,
              successorLocalIdx: isPredecessor ? hostIdx : ghostIdx,
            });
          }
        }
      }
      // Finally: refresh per-vertex tangents so the default RGB-by-
      // tangent shader gives non-black colors on metavertex centroids
      // at coarser pyramid levels (where `computeTangents` would
      // otherwise leave single-vertex-fragment tangents at zero).
      // Vertices not touched by any bridge keep their existing
      // tangent (correct at level 0 for multi-vertex fragments).
      if (resolvedBridges.length > 0 && withBridges.tangents !== undefined) {
        withBridges = recomputeTangentsForBridges(withBridges, resolvedBridges);
      }
    }

    chunk.vertexPositions = withBridges.positions;
    chunk.indices = withBridges.edges;
    // Order: synthesised tangent first (streamline/polyline only), then
    // user-declared attributes in declaration order, then the synthesised
    // per-vertex `"segment"` column last.  The frontend
    // (`buildZvSpatialVertexAttributes`) mirrors this exact ordering so
    // texture-unit indices line up and the render layer's
    // `findIndex(name === "segment")` resolves `segmentAttributeIndex`.
    const attrs: (
      | Float32Array
      | Uint8Array
      | Uint16Array
      | Uint32Array
      | Int8Array
      | Int16Array
      | Int32Array
    )[] = [];
    if (withBridges.tangents !== undefined) {
      attrs.push(withBridges.tangents);
    }
    for (const a of withBridges.vertexAttributes) attrs.push(a);
    if (withBridges.segmentIds !== undefined) {
      attrs.push(withBridges.segmentIds);
    }
    chunk.vertexAttributes = attrs;
  }
}

/**
 * Compare a cross-chunk endpoint's chunk-coordinates array to a
 * spatial chunk's grid position.  Both inputs are arrays of length
 * ``sid_ndim`` (3 for streamlines).  Returns ``true`` when the two
 * point at the same chunk.
 */
function endpointMatchesChunk(
  endpointCoords: readonly number[],
  selfCoords: readonly number[],
): boolean {
  if (endpointCoords.length !== selfCoords.length) return false;
  for (let i = 0; i < endpointCoords.length; ++i) {
    if (endpointCoords[i] !== selfCoords[i]) return false;
  }
  return true;
}

/**
 * Per-segment (object-keyed) skeleton chunk source — the **pass-2**
 * backing store.  One chunk per `objectId`.  The render layer (a
 * subclass of `SkeletonLayer`) iterates `forEachVisibleSegment` and
 * requests one chunk per visible object_id; this backend resolves each
 * `objectId` against the store's `object_index/manifests` array and
 * aggregates the named fragments across spatial chunks into one
 * merged geometry.
 *
 * Configuration of `numObjects` and `manifestChunkSize` is supplied
 * via the parameter object — the frontend dispatch (slice 4c) reads
 * `object_index/.zattrs` and `object_index/manifests/zarr.json` to fill
 * these fields before constructing the source.
 */
@registerSharedObject()
export class ZarrVectorsObjectKeyedSkeletonSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(SkeletonSource),
  ZarrVectorsObjectKeyedSkeletonSourceParameters,
) {
  /**
   * Cached decoded ``cross_chunk_links/0/`` table for this level.  Read
   * lazily on the first ``download()`` and reused across all subsequent
   * object downloads — the table is per-level, not per-object.
   *
   * ``null`` means "checked, store has no such table" (older
   * zarr-vectors stores written without ``cross_chunk_strategy =
   * "explicit_links"``).  ``undefined`` means "not yet probed".
   *
   * Only used for ``explicit`` / ``implicit_sequential_with_branches``
   * (graphs / skeletons), whose manifests span far fewer chunks per
   * object than a streamline pyramid — the whole-level decode cost
   * (tens of millions of records / multiple gigabytes for a real
   * dataset) is a known, currently-accepted tradeoff for those (see the
   * TODO below). ``implicit_sequential`` (streamline/polyline) uses
   * {@link crossChunkLinksCaches_} instead — the scoped, per-chunk
   * reader — because a whole-brain tractogram's level-0 table can be
   * large enough to OOM the tab (confirmed in practice) just to resolve
   * one selected segment's handful of cross-chunk edges.
   *
   * `downloadSegmentSkeleton` prefers real cross_chunk_links-derived
   * edges (`collectOwnedCrossChunkEdges`) over
   * `deriveImplicitSequentialCrossChunkEdges` (manifest-order
   * reconstruction), which is fragile against any coarsening pipeline
   * that doesn't preserve per-object manifest block order — the
   * zarr-vectors-tools pyramid coarsener's `_reduce_object_index_shard`
   * does not; it sorts blocks by chunk coordinate. The manifest-order
   * path remains as a fallback for stores where no cross_chunk_links
   * table exists at all.
   *
   * TODO: graphs/skeletons still pay the whole-table decode here since
   * one object's manifest can span many chunks (not the single target
   * a per-chunk query assumes) — a proper fix would move them onto the
   * same scoped-query mechanism as `implicit_sequential`, merging
   * per-chunk results the way `queryCrossChunkLinksForChunks` below
   * does. Out of scope for this pass since it hasn't caused a reported
   * crash.
   */
  private crossChunkLinks_: CrossChunkLinksTable | null | undefined;

  /**
   * Shared shard-discovery / shard-byte caches for this level's
   * ``cross_chunk_links/0/`` tree, used by ``implicit_sequential``
   * downloads via {@link queryCrossChunkLinksForChunks} — mirrors
   * {@link ZarrVectorsSpatiallyIndexedSkeletonSourceBackend
   * .crossChunkLinksCaches_} exactly (see that field's docstring for why
   * this holds only cheap shard-coordinate lists and byte-budgeted raw
   * shard bytes, not decoded records).
   */
  private crossChunkLinksCaches_: CrossChunkLinksCaches | false | undefined;

  private async getCrossChunkLinksForChunk(
    targetChunkCoords: readonly number[],
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    if (this.crossChunkLinksCaches_ === false) return undefined;
    if (this.crossChunkLinksCaches_ === undefined) {
      this.crossChunkLinksCaches_ = createCrossChunkLinksCaches();
    }
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      targetChunkCoords,
      this.crossChunkLinksCaches_,
      signal,
    );
    if (table === undefined) {
      this.crossChunkLinksCaches_ = false;
    }
    return table;
  }

  /**
   * Scoped alternative to {@link getCrossChunkLinks}, passed to
   * `downloadSegmentSkeleton` as `queryCrossChunkLinksForChunks` for
   * `implicit_sequential` downloads. Queries each of the object's owned
   * chunks individually (sharing {@link crossChunkLinksCaches_} across
   * both this call and every other object's downloads for the level)
   * and merges the results — never decodes more than the records
   * incident on chunks this specific object actually touches.
   *
   * A record whose both endpoints are among the queried chunks is
   * returned once per matching side and so may appear twice in the
   * merged table; `collectOwnedCrossChunkEdges` just emits the
   * (identical) edge twice in that case, which is harmless — cheaper
   * than deduplicating given how few cross-chunk records one object
   * has.
   */
  private async queryCrossChunkLinksForChunks(
    chunkCoordsList: readonly (readonly number[])[],
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    const records: CrossChunkLinksTable["records"] = [];
    let linkWidth: number | undefined;
    let sidNdim: number | undefined;
    for (const chunkCoords of chunkCoordsList) {
      const table = await this.getCrossChunkLinksForChunk(
        chunkCoords,
        kvStoreRead,
        kvStoreList,
        signal,
      );
      if (table === undefined) continue;
      linkWidth = table.linkWidth;
      sidNdim = table.sidNdim;
      for (const record of table.records) records.push(record);
    }
    if (linkWidth === undefined) return undefined;
    return { linkWidth, sidNdim: sidNdim!, records };
  }

  /**
   * Cached ``object_attributes/segment_id`` array (uint64, dense-OID
   * order → sorted ascending) mapping the dense object index ↔ the
   * original (e.g. flywire) segment id.  ``null`` = no such attribute
   * (selected id IS the dense index — identity); ``undefined`` = not yet
   * probed.
   */
  private segmentIds_: BigUint64Array | null | undefined;

  /**
   * Resolve a selected segment id to its dense object index.  When the
   * store carries ``object_attributes/segment_id`` (standard object-index
   * convention) binary-search the sorted ids; otherwise the id IS the
   * dense index.  Returns ``undefined`` when the id is absent.
   */
  private async resolveObjectIndex(
    objectId: number | bigint,
    numObjects: number,
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    signal: AbortSignal,
  ): Promise<number | undefined> {
    if (this.segmentIds_ === undefined) {
      const bytes = await kvStoreRead(
        "object_attributes/segment_id/data/c/0",
        signal,
      );
      if (bytes === undefined || bytes.byteLength < numObjects * 8) {
        this.segmentIds_ = null;
      } else {
        const copy = bytes.slice(0, numObjects * 8);
        this.segmentIds_ = new BigUint64Array(copy.buffer);
      }
    }
    const ids = this.segmentIds_;
    const target = BigInt(objectId);
    if (ids === null) {
      return target >= 0n && target < BigInt(numObjects)
        ? Number(target)
        : undefined;
    }
    let lo = 0;
    let hi = ids.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const v = ids[mid];
      if (v === target) return mid;
      if (v < target) lo = mid + 1;
      else hi = mid - 1;
    }
    return undefined;
  }

  private async getCrossChunkLinks(
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    if (this.crossChunkLinks_ !== undefined) {
      return this.crossChunkLinks_ ?? undefined;
    }
    const table = await readCrossChunkLinks({ kvStoreRead, kvStoreList }, signal);
    this.crossChunkLinks_ = table ?? null;
    return table;
  }

  async download(chunk: SkeletonChunk, signal: AbortSignal): Promise<void> {
    const {
      baseUrl,
      rank,
      attributeNames,
      attributeDtypes,
      linksConvention,
      geometryKind,
      linkDtype,
      hasFragmentSegmentIds,
      verticesRangeAddressable,
    } = this.parameters;
    const kvStoreRead = makeKvStoreRead(baseUrl, this.sharedKvStoreContext);
    const rawKvStoreRead = makeRawKvStoreRead(baseUrl, this.sharedKvStoreContext);
    // Fragment-scoped range reads: enabled when the store's `vertices`
    // array is uncompressed + range-addressable (`vertices_layout:
    // "raw_v1"`, surfaced as `verticesRangeAddressable`), the convention
    // is `implicit_sequential`, and the kvstore honors offset reads. A
    // compressed store falls through to the whole-chunk read path.
    const kvStoreReadRange =
      verticesRangeAddressable &&
      linksConvention === "implicit_sequential" &&
      storeSupportsOffsetReads(baseUrl, this.sharedKvStoreContext)
        ? makeKvStoreReadRange(baseUrl, this.sharedKvStoreContext)
        : undefined;
    const kvStoreList = makeKvStoreList(baseUrl, this.sharedKvStoreContext);

    // The manifests array's `numObjects` / `chunkSize` aren't carried
    // on the parameter blob (slice 4c will plumb them through from
    // `object_index/.zattrs.num_objects` and the array's `zarr.json`).
    // For now the backend reads them on each download — cheap because
    // the kvstore caches the metadata after the first fetch.
    const { numObjects, chunkSize } = await readManifestArrayShape(
      baseUrl,
      this.sharedKvStoreContext,
      signal,
    );

    // See `crossChunkLinks_`'s docstring: `implicit_sequential` uses the
    // scoped per-chunk query (avoids decoding a whole-level table that
    // can be large enough to OOM the tab for a streamline pyramid);
    // other conventions keep the whole-table fetch.
    const crossChunkLinks =
      linksConvention === "implicit_sequential"
        ? undefined
        : await this.getCrossChunkLinks(rawKvStoreRead, kvStoreList, signal);

    // Map the selected segment id (e.g. a flywire uint64) to the dense
    // object index via object_attributes/segment_id before the manifest
    // lookup.  Out-of-store ids yield an empty skeleton.
    const resolvedOid = await this.resolveObjectIndex(
      chunk.objectId,
      numObjects,
      kvStoreRead,
      signal,
    );
    if (resolvedOid === undefined) {
      chunk.vertexPositions = new Float32Array(0);
      chunk.indices = new Uint32Array(0);
      chunk.vertexAttributes = attributeNames.map(() => new Float32Array(0));
      if (
        hasSynthesisedTangent(geometryKind as ZarrVectorsSkeletonGeometryKind)
      ) {
        chunk.vertexAttributes = [
          new Float32Array(0),
          ...chunk.vertexAttributes,
        ];
      }
      return;
    }

    const aggregated = await downloadSegmentSkeleton(
      resolvedOid,
      {
        manifestReader: {
          numObjects,
          chunkSize,
          sidNdim: rank,
          kvStoreRead,
        },
        rank,
        linkDtype: asLinkDtype(linkDtype),
        attributeNames,
        attributeDtypes: attributeDtypes.map(asAttributeDtype),
        linksConvention: linksConvention as ZarrVectorsLinksConvention,
        geometryKind: geometryKind as ZarrVectorsSkeletonGeometryKind,
        crossChunkLinks,
        queryCrossChunkLinksForChunks:
          linksConvention === "implicit_sequential"
            ? (chunkCoordsList, sig) =>
                this.queryCrossChunkLinksForChunks(
                  chunkCoordsList,
                  rawKvStoreRead,
                  kvStoreList,
                  sig,
                )
            : undefined,
        // Byte-range-scoped vertex reads for streamline stores (see
        // `downloadSkeletonChunkScoped`): fetch only the selected object's
        // fragments, not the whole chunk. `undefined` unless the store is
        // `implicit_sequential` AND supports offset reads (gated above).
        kvStoreReadRange,
        hasFragmentSegmentIds,
      },
      signal,
    );

    if (aggregated === undefined) {
      // OID not in the manifest, or every named chunk is missing.
      chunk.vertexPositions = new Float32Array(0);
      chunk.indices = new Uint32Array(0);
      chunk.vertexAttributes = attributeNames.map(() => new Float32Array(0));
      // Every geometry kind with synthesised tangents (streamline,
      // polyline, graph) reserves a tangent slot at index 0 — mirror
      // that here so the render layer's attribute count is consistent
      // across passes even when an OID has no geometry.  See
      // `hasSynthesisedTangent` in `geometry_kind.ts` for the canonical
      // per-kind capability table.
      if (
        hasSynthesisedTangent(geometryKind as ZarrVectorsSkeletonGeometryKind)
      ) {
        chunk.vertexAttributes = [
          new Float32Array(0),
          ...chunk.vertexAttributes,
        ];
      }
      return;
    }

    chunk.vertexPositions = aggregated.vertexPositions;
    chunk.indices = aggregated.indices;
    chunk.vertexAttributes = aggregated.vertexAttributes;
  }
}

/**
 * Multi-resolution counterpart of {@link ZarrVectorsObjectKeyedSkeletonSourceBackend}
 * — see `ZarrVectorsMultiscaleObjectKeyedSkeletonSourceParameters`'s
 * docstring and the `MultiscaleSkeletonSource`/`MultiscaleSkeletonFragmentChunk`
 * two-tier chunk model in `src/skeleton/backend.ts`.
 *
 * `download()` (the manifest chunk) resolves the OID once against
 * `levels[0]` and probes every level's `object_index/manifests` for
 * presence/extent via `probeObjectAcrossLevels` — no vertex/edge data is
 * fetched here. `downloadFragment()` (one fragment chunk = one level) then
 * calls `downloadSegmentSkeleton` completely unchanged, exactly as the
 * single-level backend does, just pointed at that fragment's own level's
 * `baseUrl` instead of a single fixed one.
 */
@registerSharedObject()
export class ZarrVectorsMultiscaleObjectKeyedSkeletonSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(MultiscaleSkeletonSource),
  ZarrVectorsMultiscaleObjectKeyedSkeletonSourceParameters,
) {
  /**
   * Cached `object_attributes/segment_id` array, resolved against
   * `levels[0]` only — streamline/skeleton pyramids preserve object IDs
   * identically across levels, so this one resolution is reused for every
   * level's manifest lookup. Same semantics as the single-level backend's
   * `segmentIds_` (see that class's docstring).
   */
  private segmentIds_: BigUint64Array | null | undefined;

  /** Per-level `object_index/manifests` array shape, parallel to `levels`. */
  private manifestShapeByLevel_: (
    | { numObjects: number; chunkSize: number }
    | undefined
  )[] = [];

  /**
   * Per-level whole-table `cross_chunk_links/0/` cache, parallel to
   * `levels`. Only used for non-`implicit_sequential` conventions — see
   * {@link ZarrVectorsObjectKeyedSkeletonSourceBackend.crossChunkLinks_}'s
   * docstring for why `implicit_sequential` uses the scoped
   * {@link crossChunkLinksCachesByLevel_} instead.
   */
  private crossChunkLinksByLevel_: (CrossChunkLinksTable | null | undefined)[] =
    [];

  /**
   * Per-level scoped shard-discovery / shard-byte caches, parallel to
   * `levels`. Used for `implicit_sequential` downloads via
   * {@link queryCrossChunkLinksForChunksAtLevel} — mirrors
   * {@link ZarrVectorsObjectKeyedSkeletonSourceBackend.crossChunkLinksCaches_}.
   */
  private crossChunkLinksCachesByLevel_: (
    | CrossChunkLinksCaches
    | false
    | undefined
  )[] = [];

  private async getCrossChunkLinksForChunkAtLevel(
    level: number,
    targetChunkCoords: readonly number[],
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    let caches = this.crossChunkLinksCachesByLevel_[level];
    if (caches === false) return undefined;
    if (caches === undefined) {
      caches = createCrossChunkLinksCaches();
      this.crossChunkLinksCachesByLevel_[level] = caches;
    }
    const table = await readCrossChunkLinksForChunk(
      { kvStoreRead, kvStoreList },
      targetChunkCoords,
      caches,
      signal,
    );
    if (table === undefined) {
      this.crossChunkLinksCachesByLevel_[level] = false;
    }
    return table;
  }

  /**
   * Scoped alternative to {@link getCrossChunkLinksForLevel}, passed to
   * `downloadSegmentSkeleton` as `queryCrossChunkLinksForChunks` for
   * `implicit_sequential` fragment downloads — see
   * {@link ZarrVectorsObjectKeyedSkeletonSourceBackend
   * .queryCrossChunkLinksForChunks}'s docstring (identical merge
   * strategy, just parameterized by level).
   */
  private async queryCrossChunkLinksForChunksAtLevel(
    level: number,
    chunkCoordsList: readonly (readonly number[])[],
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    const records: CrossChunkLinksTable["records"] = [];
    let linkWidth: number | undefined;
    let sidNdim: number | undefined;
    for (const chunkCoords of chunkCoordsList) {
      const table = await this.getCrossChunkLinksForChunkAtLevel(
        level,
        chunkCoords,
        kvStoreRead,
        kvStoreList,
        signal,
      );
      if (table === undefined) continue;
      linkWidth = table.linkWidth;
      sidNdim = table.sidNdim;
      for (const record of table.records) records.push(record);
    }
    if (linkWidth === undefined) return undefined;
    return { linkWidth, sidNdim: sidNdim!, records };
  }

  private async resolveObjectIndex(
    objectId: number | bigint,
    numObjects: number,
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    signal: AbortSignal,
  ): Promise<number | undefined> {
    if (this.segmentIds_ === undefined) {
      const bytes = await kvStoreRead(
        "object_attributes/segment_id/data/c/0",
        signal,
      );
      if (bytes === undefined || bytes.byteLength < numObjects * 8) {
        this.segmentIds_ = null;
      } else {
        const copy = bytes.slice(0, numObjects * 8);
        this.segmentIds_ = new BigUint64Array(copy.buffer);
      }
    }
    const ids = this.segmentIds_;
    const target = BigInt(objectId);
    if (ids === null) {
      return target >= 0n && target < BigInt(numObjects)
        ? Number(target)
        : undefined;
    }
    let lo = 0;
    let hi = ids.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const v = ids[mid];
      if (v === target) return mid;
      if (v < target) lo = mid + 1;
      else hi = mid - 1;
    }
    return undefined;
  }

  private async getManifestShape(
    level: number,
    baseUrl: string,
    signal: AbortSignal,
  ): Promise<{ numObjects: number; chunkSize: number }> {
    let cached = this.manifestShapeByLevel_[level];
    if (cached === undefined) {
      cached = await readManifestArrayShape(
        baseUrl,
        this.sharedKvStoreContext,
        signal,
      );
      this.manifestShapeByLevel_[level] = cached;
    }
    return cached;
  }

  private async getCrossChunkLinksForLevel(
    level: number,
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    const cached = this.crossChunkLinksByLevel_[level];
    if (cached !== undefined) return cached ?? undefined;
    const table = await readCrossChunkLinks({ kvStoreRead, kvStoreList }, signal);
    this.crossChunkLinksByLevel_[level] = table ?? null;
    return table;
  }

  async download(
    chunk: MultiscaleSkeletonManifestChunk,
    signal: AbortSignal,
  ): Promise<void> {
    const { levels, rank, metersPerUnit } = this.parameters;
    const level0 = levels[0];
    const kvStoreRead0 = makeKvStoreRead(level0.baseUrl, this.sharedKvStoreContext);
    const { numObjects: numObjects0 } = await this.getManifestShape(
      0,
      level0.baseUrl,
      signal,
    );
    const resolvedOid = await this.resolveObjectIndex(
      chunk.objectId,
      numObjects0,
      kvStoreRead0,
      signal,
    );
    chunk.resolvedOid = resolvedOid;
    if (resolvedOid === undefined) {
      chunk.presentLevels = null;
      chunk.levelSpacings = null;
      return;
    }

    // A level whose `object_index/manifests` array doesn't exist at all
    // (as opposed to existing but having an empty manifest for this
    // particular object) must not fail the probe for every *other*
    // level — `getManifestShape`/`readManifestArrayShape` throws for a
    // missing array, so catch per-level and pass `undefined` through to
    // `probeObjectAcrossLevels`, which treats it as "not present" there.
    const levelProbeOptions = await Promise.all(
      levels.map(async (levelRef, level) => {
        try {
          const { numObjects, chunkSize } = await this.getManifestShape(
            level,
            levelRef.baseUrl,
            signal,
          );
          return {
            numObjects,
            chunkSize,
            kvStoreRead: makeKvStoreRead(levelRef.baseUrl, this.sharedKvStoreContext),
          };
        } catch {
          return undefined;
        }
      }),
    );
    const { presentLevels } = await probeObjectAcrossLevels(
      resolvedOid,
      levelProbeOptions,
      rank,
      signal,
    );
    chunk.presentLevels = presentLevels;
    // Per-level spacing in REAL-WORLD METERS, using the store's own
    // declared scale (`metersPerUnit`, from the NGFF coordinate space) —
    // NOT raw coordinate units. This is the same basis that calibrates
    // the "Resolution (skeleton grid 3D)" widget axis/slider and pass-1's
    // level selection (both derive from `getSpatialSkeletonGridSizes` =
    // `chunkShape * metersPerUnit`), so pass-2's requesting, drawing, and
    // histogram all agree with the slider. Computing it here (rather than
    // converting coordinate-unit spacings with
    // `getMetersPerUnit(projectionParameters)` at use sites) avoids the
    // display-space-vs-store-space scale mismatch that previously made
    // every level collapse to one bar.
    chunk.levelSpacings = Float32Array.from(
      levels.map((levelRef) => {
        const cs = levelRef.chunkShape;
        let minMeters = Number.POSITIVE_INFINITY;
        for (let a = 0; a < cs.length; ++a) {
          const m =
            metersPerUnit[a] ?? metersPerUnit[metersPerUnit.length - 1] ?? 1;
          minMeters = Math.min(minMeters, cs[a] * m);
        }
        return Math.max(minMeters, 1e-6);
      }),
    );
  }

  async downloadFragment(
    chunk: MultiscaleSkeletonFragmentChunk,
    signal: AbortSignal,
  ): Promise<void> {
    const { manifestChunk, level } = chunk;
    const {
      levels,
      rank,
      attributeNames,
      attributeDtypes,
      linksConvention,
      geometryKind,
      linkDtype,
    } = this.parameters;
    const levelRef = levels[level];

    const setEmpty = () => {
      chunk.vertexPositions = new Float32Array(0);
      chunk.indices = new Uint32Array(0);
      chunk.vertexAttributes = attributeNames.map(() => new Float32Array(0));
      if (
        hasSynthesisedTangent(geometryKind as ZarrVectorsSkeletonGeometryKind)
      ) {
        chunk.vertexAttributes = [
          new Float32Array(0),
          ...chunk.vertexAttributes,
        ];
      }
    };

    const resolvedOid = manifestChunk?.resolvedOid;
    if (resolvedOid === undefined) {
      // Shouldn't normally happen — the render layer only requests
      // fragment chunks for manifests with `presentLevels !== null` — but
      // guard defensively since `downloadFragment` is reachable directly.
      setEmpty();
      return;
    }

    const kvStoreRead = makeKvStoreRead(levelRef.baseUrl, this.sharedKvStoreContext);
    const rawKvStoreRead = makeRawKvStoreRead(
      levelRef.baseUrl,
      this.sharedKvStoreContext,
    );
    // Fragment-scoped range reads: enabled when this level's `vertices`
    // array is stored uncompressed + range-addressable (writer stamped
    // `vertices_layout: "raw_v1"`, surfaced as `verticesRangeAddressable`),
    // the convention is `implicit_sequential`, and the kvstore honors
    // offset reads. A compressed level (legacy) can't be range-read, so it
    // falls through to the whole-chunk path.
    const kvStoreReadRange =
      levelRef.verticesRangeAddressable &&
      linksConvention === "implicit_sequential" &&
      storeSupportsOffsetReads(levelRef.baseUrl, this.sharedKvStoreContext)
        ? makeKvStoreReadRange(levelRef.baseUrl, this.sharedKvStoreContext)
        : undefined;
    const kvStoreList = makeKvStoreList(levelRef.baseUrl, this.sharedKvStoreContext);
    const { numObjects, chunkSize } = await this.getManifestShape(
      level,
      levelRef.baseUrl,
      signal,
    );

    // See `ZarrVectorsObjectKeyedSkeletonSourceBackend.crossChunkLinks_`'s
    // docstring: `implicit_sequential` uses the scoped per-chunk query
    // (avoids decoding a whole-level table that can be large enough to
    // OOM the tab for a streamline pyramid); other conventions keep the
    // whole-table fetch.
    const crossChunkLinks =
      linksConvention === "implicit_sequential"
        ? undefined
        : await this.getCrossChunkLinksForLevel(
            level,
            rawKvStoreRead,
            kvStoreList,
            signal,
          );

    const aggregated = await downloadSegmentSkeleton(
      resolvedOid,
      {
        manifestReader: {
          numObjects,
          chunkSize,
          sidNdim: rank,
          kvStoreRead,
        },
        rank,
        linkDtype: asLinkDtype(linkDtype),
        attributeNames,
        attributeDtypes: attributeDtypes.map(asAttributeDtype),
        linksConvention: linksConvention as ZarrVectorsLinksConvention,
        geometryKind: geometryKind as ZarrVectorsSkeletonGeometryKind,
        crossChunkLinks,
        queryCrossChunkLinksForChunks:
          linksConvention === "implicit_sequential"
            ? (chunkCoordsList, sig) =>
                this.queryCrossChunkLinksForChunksAtLevel(
                  level,
                  chunkCoordsList,
                  rawKvStoreRead,
                  kvStoreList,
                  sig,
                )
            : undefined,
        // Byte-range-scoped vertex reads (see `downloadSkeletonChunkScoped`);
        // `undefined` unless implicit_sequential AND offset reads supported.
        kvStoreReadRange,
        hasFragmentSegmentIds: levelRef.hasFragmentSegmentIds,
      },
      signal,
    );

    if (aggregated === undefined) {
      setEmpty();
      return;
    }

    chunk.vertexPositions = aggregated.vertexPositions;
    chunk.indices = aggregated.indices;
    chunk.vertexAttributes = aggregated.vertexAttributes;
  }
}

/**
 * Read `numObjects` and the manifests array's chunk shape from
 * the store's `object_index/.zattrs` and `object_index/manifests/zarr.json`.
 *
 * Centralised here so the per-segment backend has one read path; slice
 * 4c will move this into the frontend dispatch so the values arrive
 * pre-resolved on the parameter blob.
 */
async function readManifestArrayShape(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      read: (url: string, options: { signal: AbortSignal }) => Promise<any>;
    };
  },
  signal: AbortSignal,
): Promise<{ numObjects: number; chunkSize: number }> {
  const arrayMetaUrl = joinBaseUrlAndPath(
    baseUrl,
    "object_index/manifests/zarr.json",
  );
  const response = await sharedKvStoreContext.kvStoreContext.read(
    arrayMetaUrl,
    {
      signal,
    },
  );
  if (response === undefined) {
    throw new Error(
      "zarr-vectors object-keyed skeleton: missing object_index/manifests/zarr.json",
    );
  }
  const text = new TextDecoder().decode(
    new Uint8Array((await response.response.arrayBuffer()) as ArrayBuffer),
  );
  const meta = JSON.parse(text);
  const shape = meta.shape;
  const chunkGrid = meta?.chunk_grid;
  if (
    !Array.isArray(shape) ||
    shape.length !== 1 ||
    typeof shape[0] !== "number"
  ) {
    throw new Error(
      "zarr-vectors object_index/manifests: shape must be a 1-D array of one integer",
    );
  }
  const numObjects = shape[0];
  // Zarr v3 regular chunk grid: chunk_grid.configuration.chunk_shape
  const chunkShape =
    chunkGrid?.configuration?.chunk_shape ?? chunkGrid?.chunk_shape;
  if (!Array.isArray(chunkShape) || chunkShape.length !== 1) {
    throw new Error(
      "zarr-vectors object_index/manifests: missing or non-1-D chunk_shape",
    );
  }
  const chunkSize = chunkShape[0];
  if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
    throw new Error(
      `zarr-vectors object_index/manifests: invalid chunk_shape ${JSON.stringify(chunkShape)}`,
    );
  }
  return { numObjects, chunkSize };
}
