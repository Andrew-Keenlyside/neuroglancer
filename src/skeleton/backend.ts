/**
 * @license
 * Copyright 2016 Google Inc.
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

import { debounce } from "lodash-es";
import type { ChunkManager } from "#src/chunk_manager/backend.js";
import {
  Chunk,
  ChunkRenderLayerBackend,
  ChunkSource,
  withChunkManager,
} from "#src/chunk_manager/backend.js";
import { ChunkState } from "#src/chunk_manager/base.js";
import type { Roi } from "#src/datasource/zarr-vectors/roi.js";
// Pure ROI geometry (no zarr/render deps) drives the streamline filter's
// backend recompute for zarr-vectors spatially-indexed skeleton (tract) layers.
// Inert for every other skeleton layer: the whole feature is guarded on the
// per-layer `roiPassingSegments` shared set being present (undefined here).
import {
  computePassingSet,
  diffPassingSet,
  type RoiFilterableChunk,
} from "#src/datasource/zarr-vectors/roi_filter_backend.js";
import { decodeVertexPositionsAndIndices } from "#src/mesh/backend.js";
import {
  type DisplayDimensionRenderInfo,
  validateDisplayDimensionRenderInfoProperty,
} from "#src/navigation_state.js";
import type {
  RenderLayerBackendAttachment,
  RenderedViewBackend,
} from "#src/render_layer_backend.js";
import { RenderLayerBackend } from "#src/render_layer_backend.js";
import { withSegmentationLayerBackendState } from "#src/segmentation_display_state/backend.js";
import {
  forEachVisibleSegment,
  getObjectKey,
} from "#src/segmentation_display_state/base.js";
import type { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type { SpatialSkeletonSourceState } from "#src/skeleton/api.js";
import {
  SKELETON_LAYER_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
} from "#src/skeleton/base.js";
import {
  freeSkeletonChunkSystemMemory,
  getVertexAttributeBytes,
  serializeSkeletonChunkData,
  type SkeletonChunkData,
} from "#src/skeleton/chunk_serialization.js";
import {
  getSpatiallyIndexedSkeletonGridIndex,
  selectSpatiallyIndexedSkeletonEntriesByGridWithFallback,
} from "#src/skeleton/source_selection.js";
import {
  BASE_PRIORITY,
  deserializeTransformedSources,
  SCALE_PRIORITY_MULTIPLIER,
  SliceViewChunk,
  SliceViewChunkSourceBackend,
} from "#src/sliceview/backend.js";
import {
  forEachVisibleVolumetricChunk,
  type SliceViewChunkSpecification,
  type SliceViewProjectionParameters,
  type TransformedSource,
} from "#src/sliceview/base.js";
import type { Uint64Set } from "#src/uint64_set.js";
import type { TypedNumberArray } from "#src/util/array.js";
import type { Endianness } from "#src/util/endian.js";
import { vec3 } from "#src/util/geom.js";
import { getObjectId } from "#src/util/object_id.js";
import {
  getBasePriority,
  getPriorityTier,
  withSharedVisibility,
} from "#src/visibility_priority/backend.js";

import type { RPC } from "#src/worker_rpc.js";
import { registerRPC, registerSharedObject } from "#src/worker_rpc.js";
export interface SpatiallyIndexedSkeletonChunkSpecification
  extends SliceViewChunkSpecification {
  chunkLayout: any;
}

const SKELETON_CHUNK_PRIORITY = 60;
const SPATIALLY_INDEXED_SKELETON_LOD_DEBOUNCE_MS = 300;
const tempCenter = vec3.create();
const tempChunkSize = vec3.create();
const tempCenterDataPosition = vec3.create();
const tempArbitrationChunkCenterWorld = vec3.create();
const tempArbitrationCandidateChunkPos = vec3.create();
const tempArbitrationLocalPoint = vec3.create();

function getChunkSpacing(size: Float32Array): number {
  return Math.max(Math.min(size[0], size[1], size[2]), 1e-6);
}

/**
 * Cheap djb2-style hash of a chunk key, XOR-combined across a chunk set to form
 * an order-independent signature for the ROI recompute (cheaper than sorting +
 * joining the keys each priority cycle). Collisions only cost a missed skip
 * (an extra recompute), never a wrong result.
 */
function cheapStringHash(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; ++i) {
    h = (((h << 5) + h) ^ s.charCodeAt(i)) | 0;
  }
  return h;
}

function computePhysicalUnitsPerScreenPixelAtPoint(
  modelViewProjection: Float32Array,
  viewportWidth: number,
  viewportHeight: number,
  worldPoint: Float32Array,
  displayDimensionScales?: Float64Array,
): number {
  const m = modelViewProjection;
  const m00 = m[0],
    m10 = m[1];
  const m01 = m[4],
    m11 = m[5];
  const m02 = m[8],
    m12 = m[9];
  const m30 = m[3],
    m31 = m[7],
    m32 = m[11],
    m33 = m[15];
  const w =
    m30 * worldPoint[0] + m31 * worldPoint[1] + m32 * worldPoint[2] + m33;
  if (!Number.isFinite(w) || w <= 0) return Number.POSITIVE_INFINITY;

  const sx =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 0 &&
    Number.isFinite(displayDimensionScales[0]) &&
    displayDimensionScales[0] > 0
      ? displayDimensionScales[0]
      : 1;
  const sy =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 1 &&
    Number.isFinite(displayDimensionScales[1]) &&
    displayDimensionScales[1] > 0
      ? displayDimensionScales[1]
      : sx;
  const sz =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 2 &&
    Number.isFinite(displayDimensionScales[2]) &&
    displayDimensionScales[2] > 0
      ? displayDimensionScales[2]
      : sy;

  const xScale = Math.sqrt(
    ((m00 / sx) * viewportWidth) ** 2 + ((m10 / sx) * viewportHeight) ** 2,
  );
  const yScale = Math.sqrt(
    ((m01 / sy) * viewportWidth) ** 2 + ((m11 / sy) * viewportHeight) ** 2,
  );
  const zScale = Math.sqrt(
    ((m02 / sz) * viewportWidth) ** 2 + ((m12 / sz) * viewportHeight) ** 2,
  );
  const scaleFactor = Math.max(xScale, yScale, zScale);
  if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  return w / scaleFactor;
}

function getChunkGridPositionForWorldPoint(
  tsource: TransformedSource<
    SpatiallyIndexedSkeletonRenderLayerBackend,
    SpatiallyIndexedSkeletonSourceBackend
  >,
  worldPoint: Float32Array,
  out: Float32Array,
): boolean {
  tsource.chunkLayout.globalToLocalSpatial(
    tempArbitrationLocalPoint,
    worldPoint as vec3,
  );
  const { size } = tsource.chunkLayout;
  const { lowerChunkBound, upperChunkBound } = tsource.source.spec;
  for (let i = 0; i < 3; ++i) {
    const dimSize = size[i];
    if (!Number.isFinite(dimSize) || dimSize <= 0) return false;
    const chunkCoord = Math.floor(tempArbitrationLocalPoint[i] / dimSize);
    if (
      Number.isFinite(lowerChunkBound[i]) &&
      Number.isFinite(upperChunkBound[i]) &&
      (chunkCoord < lowerChunkBound[i] || chunkCoord >= upperChunkBound[i])
    ) {
      return false;
    }
    out[i] = chunkCoord;
  }
  return true;
}

function getMetersPerUnit(projectionParameters: {
  displayDimensionRenderInfo?: { displayDimensionScales?: Float64Array };
}): number {
  const ddScales =
    projectionParameters.displayDimensionRenderInfo?.displayDimensionScales;
  if (ddScales === undefined || ddScales.length === 0) {
    return 1;
  }
  let metersPerUnit = Infinity;
  for (let i = 0; i < ddScales.length; ++i) {
    const s = ddScales[i];
    if (Number.isFinite(s) && s > 0) {
      metersPerUnit = Math.min(metersPerUnit, s);
    }
  }
  return Number.isFinite(metersPerUnit) ? metersPerUnit : 1;
}

function quantizeSpacingForArbitration(spacing: number): number {
  const clamped = Math.max(spacing, 1e-12);
  const log2Spacing = Math.log2(clamped);
  const quantizedLog = Math.round(log2Spacing * 4) / 4;
  return 2 ** quantizedLog;
}

export function getSpatiallyIndexedSkeletonChunkPriority(
  localCenter: Float32Array,
  chunkSize: Float32Array,
  positionInChunks: Float32Array,
) {
  let sum = 0;
  for (let i = 0; i < 3; ++i) {
    const delta = localCenter[i] - positionInChunks[i] * chunkSize[i];
    sum += delta * delta;
  }
  return -Math.sqrt(sum);
}

export function getSpatiallyIndexedSkeletonRenderPriority(
  basePriority: number,
  scaleIndex: number,
  localCenter: Float32Array,
  chunkSize: Float32Array,
  positionInChunks: Float32Array,
) {
  // No boost relative to other VISIBLE-tier sources (volume rendering,
  // annotations, meshes): spatially-indexed skeleton chunks compete for
  // the shared chunk memory budget purely on distance-to-view relevance,
  // like everything else.  A prior boost here
  // (SPATIALLY_INDEXED_SKELETON_PRIORITY_BOOST = -BASE_PRIORITY) exactly
  // canceled the shared BASE_PRIORITY baseline every VISIBLE-tier source
  // is anchored to, which meant EVERY skeleton chunk — even far ones —
  // unconditionally outranked EVERY other layer's visible chunks, letting
  // an actively-loading skeleton layer evict an unrelated image layer's
  // required chunks entirely.
  return (
    basePriority +
    SCALE_PRIORITY_MULTIPLIER * scaleIndex +
    getSpatiallyIndexedSkeletonChunkPriority(
      localCenter,
      chunkSize,
      positionInChunks,
    )
  );
}

export enum SpatiallyIndexedSkeletonChunkRequestOwner {
  NONE = 0,
  VIEW_2D = 1 << 0,
  VIEW_3D = 1 << 1,
}

export function markSpatiallyIndexedSkeletonChunkRequested(
  chunk: SpatiallyIndexedSkeletonChunk,
  currentGeneration: number,
  owner: SpatiallyIndexedSkeletonChunkRequestOwner,
) {
  if (
    owner === SpatiallyIndexedSkeletonChunkRequestOwner.NONE ||
    currentGeneration < 0
  ) {
    return;
  }
  if (chunk.requestGeneration !== currentGeneration) {
    chunk.requestGeneration = currentGeneration;
    chunk.requestOwners = owner;
    return;
  }
  chunk.requestOwners |= owner;
}

export function cancelStaleSpatiallyIndexedSkeletonDownloads(
  chunkManager: ChunkManager,
  sources: Iterable<SpatiallyIndexedSkeletonSourceBackend>,
  currentGeneration: number,
) {
  const queueManager = chunkManager.queueManager;
  for (const source of sources) {
    for (const chunk of source.chunks.values()) {
      const typedChunk = chunk as SpatiallyIndexedSkeletonChunk;
      if (typedChunk.state !== ChunkState.DOWNLOADING) continue;
      if (
        typedChunk.requestGeneration === currentGeneration &&
        typedChunk.requestOwners !==
          SpatiallyIndexedSkeletonChunkRequestOwner.NONE
      ) {
        continue;
      }
      const controller = typedChunk.downloadAbortController;
      if (controller === undefined) continue;
      typedChunk.downloadAbortController = undefined;
      controller.abort(
        new DOMException("stale spatial skeleton LOD download", "AbortError"),
      );
      queueManager.updateChunkState(typedChunk, ChunkState.QUEUED);
    }
  }
}

registerRPC(
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  function (x) {
    const view = this.get(x.view) as RenderedViewBackend;
    const layer = this.get(
      x.layer,
    ) as SpatiallyIndexedSkeletonRenderLayerBackend;
    const attachment = layer.attachments.get(
      view,
    )! as RenderLayerBackendAttachment<
      RenderedViewBackend,
      SpatiallyIndexedSkeletonRenderLayerAttachmentState
    >;
    attachment.state!.transformedSources = deserializeTransformedSources<
      SpatiallyIndexedSkeletonSourceBackend,
      SpatiallyIndexedSkeletonRenderLayerBackend
    >(this, x.sources, layer);
    attachment.state!.displayDimensionRenderInfo = x.displayDimensionRenderInfo;
    layer.chunkManager.scheduleUpdateChunkPriorities();
  },
);

// Chunk that contains the skeleton of a single object.
export class SkeletonChunk extends Chunk implements SkeletonChunkData {
  objectId: bigint = 0n;
  vertexPositions: Float32Array | null = null;
  vertexAttributes: TypedNumberArray[] | null = null;
  indices: Uint32Array | null = null;

  initializeSkeletonChunk(key: string, objectId: bigint) {
    super.initialize(key);
    this.objectId = objectId;
  }

  freeSystemMemory() {
    freeSkeletonChunkSystemMemory(this);
  }

  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    serializeSkeletonChunkData(this, msg, transfers);
    freeSkeletonChunkSystemMemory(this);
  }

  downloadSucceeded() {
    this.systemMemoryBytes = this.gpuMemoryBytes =
      this.indices!.byteLength + getVertexAttributeBytes(this);
    super.downloadSucceeded();
  }
}

export class SkeletonSource extends ChunkSource {
  declare chunks: Map<string, SkeletonChunk>;
  getChunk(objectId: bigint) {
    const key = getObjectKey(objectId);
    let chunk = this.chunks.get(key);
    if (chunk === undefined) {
      chunk = this.getNewChunk_(SkeletonChunk);
      chunk.initializeSkeletonChunk(key, objectId);
      this.addChunk(chunk);
    }
    return chunk;
  }
}

@registerSharedObject(SKELETON_LAYER_RPC_ID)
export class SkeletonLayer extends withSegmentationLayerBackendState(
  withSharedVisibility(withChunkManager(ChunkRenderLayerBackend)),
) {
  source: SkeletonSource;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.source = this.registerDisposer(
      rpc.getRef<SkeletonSource>(options.source),
    );
    this.registerDisposer(
      this.chunkManager.recomputeChunkPriorities.add(() => {
        this.updateChunkPriorities();
      }),
    );
  }

  private updateChunkPriorities() {
    const visibility = this.visibility.value;
    if (visibility === Number.NEGATIVE_INFINITY) {
      return;
    }
    this.chunkManager.registerLayer(this);
    const priorityTier = getPriorityTier(visibility);
    const basePriority = getBasePriority(visibility);
    const { source, chunkManager } = this;
    forEachVisibleSegment(this, (objectId) => {
      const chunk = source.getChunk(objectId);
      ++this.numVisibleChunksNeeded;
      if (chunk.state === ChunkState.GPU_MEMORY) {
        ++this.numVisibleChunksAvailable;
      }
      chunkManager.requestChunk(
        chunk,
        priorityTier,
        basePriority + SKELETON_CHUNK_PRIORITY,
      );
    });
  }
}

/**
 * Extracts vertex positions and edge vertex indices of the specified endianness from `data'.
 *
 * See documentation of decodeVertexPositionsAndIndices.
 */
export function decodeSkeletonVertexPositionsAndIndices(
  chunk: SkeletonChunk,
  data: ArrayBuffer,
  endianness: Endianness,
  vertexByteOffset: number,
  numVertices: number,
  indexByteOffset?: number,
  numEdges?: number,
) {
  const meshData = decodeVertexPositionsAndIndices(
    /*verticesPerPrimitive=*/ 2,
    data,
    endianness,
    vertexByteOffset,
    numVertices,
    indexByteOffset,
    numEdges,
  );
  chunk.vertexPositions = meshData.vertexPositions as Float32Array;
  chunk.indices = meshData.indices as Uint32Array;
}

export class SpatiallyIndexedSkeletonChunk
  extends SliceViewChunk
  implements SkeletonChunkData
{
  vertexPositions: Float32Array | null = null;
  vertexAttributes: TypedNumberArray[] | null = null;
  indices: Uint32Array | null = null;
  lod: number = 0;
  requestGeneration = -1;
  requestOwners = SpatiallyIndexedSkeletonChunkRequestOwner.NONE;
  nodeIds: Int32Array | undefined;
  nodeSourceStates: Array<SpatialSkeletonSourceState | undefined> | undefined;

  /**
   * Slim view of this chunk's decoded geometry retained for the ROI streamline
   * filter (zarr-vectors tract layers only; `undefined` for every other
   * source). Set by the zarr-vectors pass-1 `download()`. It aliases the same
   * `positions`/`segmentIds` buffers the chunk decoded: `serializeSkeletonChunkData`
   * packs a *copy* of those into the transferred message and only transfers
   * `indices.buffer`, so these references stay valid (un-detached) after
   * serialize + `freeSkeletonChunkSystemMemory`, letting the render-layer
   * backend re-filter within memory when the ROI list changes — no refetch.
   */
  roiFilterableChunk?: RoiFilterableChunk;

  freeSystemMemory() {
    freeSkeletonChunkSystemMemory(this);
    // Reclaim the ROI-filter retention too. This method is the chunk manager's
    // system-memory reclaim path (chunk data is gone → it must re-download to
    // display, and will re-establish roiFilterableChunk then). serialize() frees
    // via the free *function* directly, NOT this method, so the retention still
    // survives transfer to the frontend — which is what lets the render layer
    // re-filter without a refetch.
    this.roiFilterableChunk = undefined;
  }

  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    serializeSkeletonChunkData(this, msg, transfers);
    freeSkeletonChunkSystemMemory(this);
  }

  downloadSucceeded() {
    const attributeBytes =
      this.indices!.byteLength + getVertexAttributeBytes(this);
    this.gpuMemoryBytes = attributeBytes;
    // The retained roiFilterableChunk aliases the positions/segment buffers
    // already counted above (conservatively kept charged for GPU-state chunks);
    // its fragmentIndex is the one net-new retained allocation, so make it
    // visible to the system-memory budget.
    this.systemMemoryBytes =
      attributeBytes +
      (this.roiFilterableChunk?.fragmentIndex.byteLength ?? 0);
    super.downloadSucceeded();
  }
}

export class SpatiallyIndexedSkeletonSourceBackend extends SliceViewChunkSourceBackend<
  SpatiallyIndexedSkeletonChunkSpecification,
  SpatiallyIndexedSkeletonChunk
> {
  chunkConstructor = SpatiallyIndexedSkeletonChunk;
  currentLod: number = 0;
  currentRequestGeneration = -1;
  currentRequestOwner = SpatiallyIndexedSkeletonChunkRequestOwner.NONE;

  getChunk(chunkGridPosition: Float32Array) {
    const lodValue = this.currentLod;
    const key = `${chunkGridPosition.join()}:${lodValue}`;
    let chunk = this.chunks.get(key);
    if (chunk === undefined) {
      chunk = this.getNewChunk_(
        this.chunkConstructor,
      ) as SpatiallyIndexedSkeletonChunk;
      chunk.initializeVolumeChunk(key, chunkGridPosition);
      chunk.lod = lodValue;
      this.addChunk(chunk);
    }
    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      this.currentRequestGeneration,
      this.currentRequestOwner,
    );
    return chunk;
  }
}

interface SpatiallyIndexedSkeletonRenderLayerAttachmentState {
  displayDimensionRenderInfo: DisplayDimensionRenderInfo;
  transformedSources: TransformedSource<
    SpatiallyIndexedSkeletonRenderLayerBackend,
    SpatiallyIndexedSkeletonSourceBackend
  >[][];
}

@registerSharedObject(SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID)
export class SpatiallyIndexedSkeletonRenderLayerBackend extends withChunkManager(
  RenderLayerBackend,
) {
  localPosition: SharedWatchableValue<Float32Array>;
  renderScaleTarget: SharedWatchableValue<number>;
  skeletonLod: SharedWatchableValue<number>;
  skeletonGridLevel: SharedWatchableValue<number>;
  skeletonLod2d: SharedWatchableValue<number>;
  skeletonGridLevel2d: SharedWatchableValue<number>;
  skeletonGridResolutionTarget3d: SharedWatchableValue<number>;
  private pendingLodCleanup = false;

  // ROI streamline filter (zarr-vectors tract layers only). Both are set
  // together, or both left undefined for every other skeleton layer — in which
  // case the recompute is a no-op and this layer behaves exactly as before.
  //   - roiPassingSegments: shared set of object ids that pass the filter; this
  //     backend mutates it, the frontend's twin drives the ghosting shader.
  //   - roiConfig: the ordered ROI list (plain serialisable geometry).
  // The passing set is computed whenever ROIs exist, independent of whether the
  // user has the filter switched on — the *active* flag is purely a frontend
  // shader concern (uRoiFilterActive), so keeping the set current means enabling
  // the filter is instant rather than flashing the whole tractogram to
  // ghost-alpha while an async recompute catches up.
  roiPassingSegments?: Uint64Set;
  roiConfig?: SharedWatchableValue<readonly Roi[]>;
  /** Set when an ROI edit needs a recompute even if the resident chunk set is unchanged. */
  private roiRecomputePending = false;
  /** Signature of the last resident-chunk set filtered over, to skip redundant recomputes. */
  private roiLastChunkSignature = "";

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.renderScaleTarget = rpc.get(options.renderScaleTarget);
    this.localPosition = rpc.get(options.localPosition);
    this.skeletonLod = rpc.get(options.skeletonLod);
    this.skeletonGridLevel = rpc.get(options.skeletonGridLevel);
    this.skeletonLod2d = rpc.get(options.skeletonLod2d);
    this.skeletonGridLevel2d = rpc.get(options.skeletonGridLevel2d);
    this.skeletonGridResolutionTarget3d = rpc.get(
      options.skeletonGridResolutionTarget3d,
    );
    const scheduleUpdateChunkPriorities = () =>
      this.chunkManager.scheduleUpdateChunkPriorities();
    // Anything that can change the grid-anchor/level selection made in
    // recomputeChunkPriorities must also mark stale downloads for cleanup
    // (see `pendingLodCleanup` below) — otherwise in-flight downloads from
    // the superseded selection are never cancelled and just accumulate as
    // the camera moves/zooms, compounding with each recompute's marginally
    // different selection into an ever-growing, never-converging request
    // count. Previously only the skeletonLod slider did this.
    const scheduleUpdateAndMarkStaleCleanup = () => {
      this.pendingLodCleanup = true;
      scheduleUpdateChunkPriorities();
    };
    this.registerDisposer(
      this.localPosition.changed.add(scheduleUpdateAndMarkStaleCleanup),
    );
    this.registerDisposer(
      this.renderScaleTarget.changed.add(scheduleUpdateAndMarkStaleCleanup),
    );
    this.registerDisposer(
      this.skeletonGridLevel.changed.add(scheduleUpdateAndMarkStaleCleanup),
    );
    this.registerDisposer(
      this.skeletonGridLevel2d.changed.add(scheduleUpdateAndMarkStaleCleanup),
    );
    this.registerDisposer(
      this.skeletonGridResolutionTarget3d.changed.add(
        scheduleUpdateAndMarkStaleCleanup,
      ),
    );

    // Debounce LOD changes to avoid making requests for every slider value
    const debouncedLodUpdate = debounce(() => {
      scheduleUpdateChunkPriorities();
    }, SPATIALLY_INDEXED_SKELETON_LOD_DEBOUNCE_MS);
    this.registerDisposer(() => debouncedLodUpdate.cancel());

    const onLodChanged = () => {
      this.pendingLodCleanup = true;
      debouncedLodUpdate();
    };
    this.registerDisposer(this.skeletonLod.changed.add(onLodChanged));
    this.registerDisposer(this.skeletonLod2d.changed.add(onLodChanged));
    this.registerDisposer(
      this.chunkManager.recomputeChunkPriorities.add(() =>
        this.recomputeChunkPriorities(),
      ),
    );
    this.registerDisposer(
      this.chunkManager.recomputeChunkPrioritiesLate.add(() => {
        // Re-run the ROI filter after chunk priorities settle, so the passing
        // set reflects exactly the chunks now resident at the current level.
        // A no-op unless this is an ROI-filtered (zarr-vectors tract) layer.
        this.maybeRecomputeRoiPassingSet();
        if (!this.pendingLodCleanup) return;
        const sources = new Set<SpatiallyIndexedSkeletonSourceBackend>();
        for (const attachment of this.attachments.values()) {
          const attachmentState = attachment.state as
            | SpatiallyIndexedSkeletonRenderLayerAttachmentState
            | undefined;
          if (attachmentState === undefined) continue;
          for (const scales of attachmentState.transformedSources) {
            for (const tsource of scales) {
              sources.add(
                tsource.source as SpatiallyIndexedSkeletonSourceBackend,
              );
            }
          }
        }
        cancelStaleSpatiallyIndexedSkeletonDownloads(
          this.chunkManager,
          sources,
          this.chunkManager.recomputeChunkPriorities.count,
        );
        this.pendingLodCleanup = false;
      }),
    );

    // ROI streamline filter channel (present only for zarr-vectors tract
    // layers). An ROI edit needs a recompute even when the resident chunk set
    // is unchanged; schedule one and let the late-priorities hook run it.
    if (options.roiPassingSegments !== undefined) {
      this.roiPassingSegments = rpc.get(options.roiPassingSegments);
      const roiConfig = (this.roiConfig = rpc.get(options.roiConfig));
      const scheduleRoiRecompute = () => {
        this.roiRecomputePending = true;
        this.chunkManager.scheduleUpdateChunkPriorities();
      };
      this.registerDisposer(roiConfig.changed.add(scheduleRoiRecompute));
    }
  }

  /**
   * Object-id sources this layer can ROI-filter: the pass-1 skeleton sources
   * behind every attachment, de-duplicated (the same source backs both the 2-d
   * and 3-d views).
   */
  private *roiFilterableSources(): Iterable<SpatiallyIndexedSkeletonSourceBackend> {
    const seen = new Set<SpatiallyIndexedSkeletonSourceBackend>();
    for (const attachment of this.attachments.values()) {
      const attachmentState = attachment.state as
        | SpatiallyIndexedSkeletonRenderLayerAttachmentState
        | undefined;
      if (attachmentState === undefined) continue;
      for (const scales of attachmentState.transformedSources) {
        for (const tsource of scales) {
          const source = tsource.source as SpatiallyIndexedSkeletonSourceBackend;
          if (!seen.has(source)) {
            seen.add(source);
            yield source;
          }
        }
      }
    }
  }

  /**
   * Recompute which loaded streamlines pass the ROI filter and push the delta
   * to the shared passing set. A no-op for non-ROI (non-tract) layers.
   *
   * Evaluated at a SINGLE pyramid level: object ids are stable across levels, so
   * folding two levels' differently-decimated geometry for one id into one
   * verdict would let a coarse level pollute the fine view and vice versa. The
   * 3-d level is preferred; the recompute falls back to the 2-d level only when
   * no 3-d chunk is resident (a 2-d-only layout). One backend serves both views
   * of a tract layer, so its attachments span both — the recompute sees every
   * resident source.
   *
   * Crossings are OR-merged over that level's RESIDENT chunks. At the intended
   * coarse whole-brain operating point the level is fully resident, so a
   * streamline spanning several chunks is judged on its whole geometry. Under
   * partial residency (a zoomed / frustum-culled view) an object's out-of-view
   * fragments are not counted, so a multi-ROI spanning dissection can
   * under-select there — a known v1 limitation.
   *
   * Skipped when neither the ROI list nor the resident chunk set has changed
   * since the last run, so a plain camera move (which also fires this hook)
   * costs only a cheap, order-independent signature scan (no sort/alloc). The
   * delta is applied as one add batch and one remove batch, since the frontend
   * GPU hash table re-uploads wholesale per mutation.
   */
  private maybeRecomputeRoiPassingSet() {
    const passingSet = this.roiPassingSegments;
    if (passingSet === undefined || this.roiConfig === undefined) return;
    // Compute whenever ROIs exist, regardless of the (frontend-only) active
    // flag, so enabling the filter is instant. An empty ROI list yields an
    // empty passing set; the shader treats "active with no ROIs" as off, so an
    // empty set is never mistaken for "everything fails".
    const rois = this.roiConfig.value;
    const lod3d = this.skeletonLod.value;
    const lod2d = this.skeletonLod2d.value;

    // Bucket the resident, filterable chunks of the two live levels, tracking a
    // per-level count and an order-independent hash of their keys.
    const byLod = new Map<
      number,
      { chunks: RoiFilterableChunk[]; count: number; hash: number }
    >();
    if (rois.length !== 0) {
      for (const source of this.roiFilterableSources()) {
        for (const chunk of source.chunks.values()) {
          const c = chunk as SpatiallyIndexedSkeletonChunk;
          const data = c.roiFilterableChunk;
          if (data === undefined || (c.lod !== lod3d && c.lod !== lod2d)) {
            continue;
          }
          let bucket = byLod.get(c.lod);
          if (bucket === undefined) {
            bucket = { chunks: [], count: 0, hash: 0 };
            byLod.set(c.lod, bucket);
          }
          bucket.chunks.push(data);
          bucket.count++;
          bucket.hash = (bucket.hash ^ cheapStringHash(c.key ?? "")) | 0;
        }
      }
    }
    const targetLod = byLod.has(lod3d) ? lod3d : lod2d;
    const bucket = byLod.get(targetLod);
    const chunks = bucket?.chunks ?? [];

    const signature = `${targetLod}:${bucket?.count ?? 0}:${bucket?.hash ?? 0}`;
    if (!this.roiRecomputePending && signature === this.roiLastChunkSignature) {
      return;
    }
    this.roiRecomputePending = false;
    this.roiLastChunkSignature = signature;

    const target = computePassingSet(chunks, rois);
    const current = new Set<bigint>(passingSet.keys());
    const { added, removed } = diffPassingSet(target, current);
    if (removed.length !== 0) passingSet.delete(removed);
    if (added.length !== 0) passingSet.add(added);
  }

  attach(
    attachment: RenderLayerBackendAttachment<
      RenderedViewBackend,
      SpatiallyIndexedSkeletonRenderLayerAttachmentState
    >,
  ) {
    const scheduleUpdateChunkPriorities = () =>
      this.chunkManager.scheduleUpdateChunkPriorities();
    const { view } = attachment;
    attachment.registerDisposer(scheduleUpdateChunkPriorities);
    attachment.registerDisposer(
      view.projectionParameters.changed.add(scheduleUpdateChunkPriorities),
    );
    attachment.registerDisposer(
      view.visibility.changed.add(scheduleUpdateChunkPriorities),
    );
    attachment.state = {
      displayDimensionRenderInfo:
        view.projectionParameters.value.displayDimensionRenderInfo,
      transformedSources: [],
    };
  }

  private recomputeChunkPriorities() {
    this.chunkManager.registerLayer(this);
    const currentGeneration = this.chunkManager.recomputeChunkPriorities.count;
    for (const attachment of this.attachments.values()) {
      const { view } = attachment;
      const visibility = view.visibility.value;
      if (visibility === Number.NEGATIVE_INFINITY) {
        continue;
      }
      const attachmentState =
        attachment.state! as SpatiallyIndexedSkeletonRenderLayerAttachmentState;
      const { transformedSources } = attachmentState;
      if (
        transformedSources.length === 0 ||
        !validateDisplayDimensionRenderInfoProperty(
          attachmentState,
          view.projectionParameters.value.displayDimensionRenderInfo,
        )
      ) {
        continue;
      }
      const priorityTier = getPriorityTier(visibility);
      const basePriority = getBasePriority(visibility) + BASE_PRIORITY;
      const projectionParameters = view.projectionParameters.value;
      const { chunkManager } = this;
      const localCenter = tempCenter;
      const chunkSize = tempChunkSize;
      const centerDataPosition = tempCenterDataPosition;
      const {
        globalPosition,
        displayDimensionRenderInfo: { displayDimensionIndices },
      } = projectionParameters;
      for (let displayDim = 0; displayDim < 3; ++displayDim) {
        const globalDim = displayDimensionIndices[displayDim];
        centerDataPosition[displayDim] =
          globalDim === -1 ? 0 : globalPosition[globalDim];
      }
      const sliceProjectionParameters =
        projectionParameters as SliceViewProjectionParameters;
      const pixelSize =
        "pixelSize" in sliceProjectionParameters
          ? sliceProjectionParameters.pixelSize
          : undefined;
      let resolvedPixelSize = pixelSize;
      if (resolvedPixelSize === undefined) {
        const voxelPhysicalScales =
          projectionParameters.displayDimensionRenderInfo?.voxelPhysicalScales;
        if (voxelPhysicalScales) {
          let computedPixelSize = 0;
          const { invViewMatrix } = projectionParameters;
          for (let i = 0; i < 3; ++i) {
            const s = voxelPhysicalScales[i];
            const x = invViewMatrix[i];
            computedPixelSize += (s * x) ** 2;
          }
          resolvedPixelSize = Math.sqrt(computedPixelSize);
        }
      }
      const renderScaleTarget = this.renderScaleTarget.value;
      const is2dView = pixelSize !== undefined;
      const skeletonGridLevel = (
        is2dView ? this.skeletonGridLevel2d : this.skeletonGridLevel
      ).value;

      const selectScales = (
        scales: TransformedSource<
          SpatiallyIndexedSkeletonRenderLayerBackend,
          SpatiallyIndexedSkeletonSourceBackend
        >[],
      ): Array<{
        tsource: TransformedSource<
          SpatiallyIndexedSkeletonRenderLayerBackend,
          SpatiallyIndexedSkeletonSourceBackend
        >;
        scaleIndex: number;
      }> => {
        if (scales.length === 0) {
          return [];
        }
        if (
          scales.every(
            (scale) =>
              getSpatiallyIndexedSkeletonGridIndex(scale) !== undefined,
          )
        ) {
          // `...WithFallback` returns EVERY level in fallback-preference
          // order (preferred first, then progressively finer/coarser
          // alternatives) for a caller to try in sequence and stop at the
          // first viable one -- see its docstring. It is not a list of
          // levels to request/use simultaneously (the grid-anchor path a
          // few hundred lines below, in recomputeChunkPriorities's own
          // per-position candidate loop, uses it correctly this way). This
          // branch was instead returning the WHOLE list directly as "the
          // selected scales", so its caller looped over every pyramid
          // level and requested chunks from all of them -- the direct
          // cause of thousands of stray requests whenever grid indices are
          // present (which for zarr-vectors multi-resolution sources is
          // always true), including in 2D views where this is the only
          // code path (the grid-anchor arbitration path below only runs
          // for 3D views).
          const ordered =
            selectSpatiallyIndexedSkeletonEntriesByGridWithFallback(
              scales.map((tsource, scaleIndex) => ({ tsource, scaleIndex })),
              skeletonGridLevel,
              ({ tsource }) => getSpatiallyIndexedSkeletonGridIndex(tsource),
            );
          return ordered.length > 0 ? [ordered[0]] : [];
        }
        if (resolvedPixelSize === undefined) {
          return scales.map((tsource, scaleIndex) => ({
            tsource,
            scaleIndex,
          }));
        }
        const pixelSizeWithMargin = resolvedPixelSize * 1.1;
        const smallestVoxelSize = scales[0].effectiveVoxelSize;
        const canImproveOnVoxelSize = (voxelSize: Float32Array) => {
          const targetSize = pixelSizeWithMargin * renderScaleTarget;
          for (let i = 0; i < 3; ++i) {
            const size = voxelSize[i];
            if (size > targetSize && size > 1.01 * smallestVoxelSize[i]) {
              return true;
            }
          }
          return false;
        };
        const improvesOnPrevVoxelSize = (
          voxelSize: Float32Array,
          prevVoxelSize: Float32Array,
        ) => {
          const targetSize = pixelSizeWithMargin * renderScaleTarget;
          for (let i = 0; i < 3; ++i) {
            const size = voxelSize[i];
            const prevSize = prevVoxelSize[i];
            if (
              Math.abs(targetSize - size) < Math.abs(targetSize - prevSize) &&
              size < 1.01 * prevSize
            ) {
              return true;
            }
          }
          return false;
        };

        const selected: Array<{
          tsource: TransformedSource<
            SpatiallyIndexedSkeletonRenderLayerBackend,
            SpatiallyIndexedSkeletonSourceBackend
          >;
          scaleIndex: number;
        }> = [];
        let scaleIndex = scales.length - 1;
        let prevVoxelSize: Float32Array | undefined;
        while (true) {
          const tsource = scales[scaleIndex];
          const selectionVoxelSize = tsource.effectiveVoxelSize;
          if (
            prevVoxelSize !== undefined &&
            !improvesOnPrevVoxelSize(selectionVoxelSize, prevVoxelSize)
          ) {
            break;
          }
          selected.push({ tsource, scaleIndex });
          if (scaleIndex === 0) break;
          if (!canImproveOnVoxelSize(selectionVoxelSize)) break;
          prevVoxelSize = selectionVoxelSize;
          --scaleIndex;
        }
        return selected;
      };

      const lodValue = (is2dView ? this.skeletonLod2d : this.skeletonLod).value;
      for (const scales of transformedSources) {
        if (
          !is2dView &&
          scales.length > 1 &&
          scales.every(
            (scale) =>
              getSpatiallyIndexedSkeletonGridIndex(scale) !== undefined,
          )
        ) {
          // `fallbackRank` is the position in the returned preference order:
          // the selected grid level first, then the fallbacks. That order is
          // the ONLY thing carrying the level selection, and the arbitration
          // sort below would otherwise discard it -- see the tie-break there.
          const orderedCandidates =
            selectSpatiallyIndexedSkeletonEntriesByGridWithFallback(
              scales.map((tsource, scaleIndex) => ({ tsource, scaleIndex })),
              skeletonGridLevel,
              ({ tsource }) => getSpatiallyIndexedSkeletonGridIndex(tsource),
            ).map((candidate, fallbackRank) => ({
              ...candidate,
              fallbackRank,
            }));
          if (orderedCandidates.length > 0) {
            const metersPerUnit = getMetersPerUnit(projectionParameters);
            const spacingMeters = (candidate: {
              tsource: TransformedSource<
                SpatiallyIndexedSkeletonRenderLayerBackend,
                SpatiallyIndexedSkeletonSourceBackend
              >;
            }) =>
              getChunkSpacing(candidate.tsource.chunkLayout.size) *
              metersPerUnit;
            // Anchor the position-enumeration grid on whichever candidate's
            // spacing is CLOSEST to the desired resolution target, not
            // unconditionally the finest level: enumerating at the finest
            // level's cell density across the whole visible frustum for a
            // coarse/zoomed-out view is orders of magnitude more iteration
            // work than the view needs, and independently re-derives a
            // distance-based LOD per tiny finest-grid cell -- fragmenting
            // what should be one coarse-level selection into many different
            // levels scattered across the view.
            const fallbackAnchor = orderedCandidates.reduce(
              (best, candidate) =>
                spacingMeters(candidate) < spacingMeters(best)
                  ? candidate
                  : best,
            );
            const targetSpacingMeters =
              Number.isFinite(this.skeletonGridResolutionTarget3d.value) &&
              this.skeletonGridResolutionTarget3d.value > 0
                ? this.skeletonGridResolutionTarget3d.value
                : spacingMeters(fallbackAnchor);
            const anchor = orderedCandidates.reduce((best, candidate) =>
              Math.abs(spacingMeters(candidate) - targetSpacingMeters) <
              Math.abs(spacingMeters(best) - targetSpacingMeters)
                ? candidate
                : best,
            );
            const refPoint =
              projectionParameters.globalPosition.length >= 3
                ? projectionParameters.globalPosition
                : this.localPosition.value;
            const referencePixelSizeRaw =
              computePhysicalUnitsPerScreenPixelAtPoint(
                projectionParameters.viewProjectionMat,
                projectionParameters.width,
                projectionParameters.height,
                refPoint,
                projectionParameters.displayDimensionRenderInfo
                  ?.displayDimensionScales,
              );
            const referencePixelSize =
              Number.isFinite(referencePixelSizeRaw) &&
              referencePixelSizeRaw > 0
                ? referencePixelSizeRaw
                : 1;

            const emitted = new Set<string>();
            forEachVisibleVolumetricChunk(
              projectionParameters,
              this.localPosition.value,
              anchor.tsource,
              (anchorPosInChunks) => {
                tempArbitrationChunkCenterWorld[0] =
                  (anchorPosInChunks[0] + 0.5) *
                  anchor.tsource.chunkLayout.size[0];
                tempArbitrationChunkCenterWorld[1] =
                  (anchorPosInChunks[1] + 0.5) *
                  anchor.tsource.chunkLayout.size[1];
                tempArbitrationChunkCenterWorld[2] =
                  (anchorPosInChunks[2] + 0.5) *
                  anchor.tsource.chunkLayout.size[2];
                vec3.transformMat4(
                  tempArbitrationChunkCenterWorld,
                  tempArbitrationChunkCenterWorld,
                  anchor.tsource.chunkLayout.transform,
                );

                const chunkPixelSize =
                  computePhysicalUnitsPerScreenPixelAtPoint(
                    projectionParameters.viewProjectionMat,
                    projectionParameters.width,
                    projectionParameters.height,
                    tempArbitrationChunkCenterWorld,
                    projectionParameters.displayDimensionRenderInfo
                      ?.displayDimensionScales,
                  );
                const desiredSpacingRaw =
                  Number.isFinite(chunkPixelSize) && chunkPixelSize > 0
                    ? targetSpacingMeters *
                      (chunkPixelSize / referencePixelSize)
                    : targetSpacingMeters;
                const desiredSpacing =
                  quantizeSpacingForArbitration(desiredSpacingRaw);

                const candidatesByDesired = [...orderedCandidates].sort(
                  (a, b) => {
                    const da = Math.abs(spacingMeters(a) - desiredSpacing);
                    const db = Math.abs(spacingMeters(b) - desiredSpacing);
                    if (da !== db) return da - db;
                    // Ties broken by preference, matching the frontend's
                    // identical arbitration (`skeleton/frontend.ts`). Breaking
                    // on `scaleIndex` -- a position in the finest-first
                    // `getSources()` array -- silently discarded the level
                    // selection and always chose the finest level, because on
                    // an object-sparsity pyramid every level shares one
                    // chunk_shape, so every spacing is equal and this branch
                    // is always reached. The frontend then drew the selected
                    // level while the backend fetched the finest, so nothing
                    // ever fetched what was drawn.
                    return a.fallbackRank - b.fallbackRank;
                  },
                );

                let selected:
                  | {
                      tsource: TransformedSource<
                        SpatiallyIndexedSkeletonRenderLayerBackend,
                        SpatiallyIndexedSkeletonSourceBackend
                      >;
                      scaleIndex: number;
                      position: Float32Array;
                      key: string;
                    }
                  | undefined;
                for (const candidate of candidatesByDesired) {
                  if (
                    !getChunkGridPositionForWorldPoint(
                      candidate.tsource,
                      tempArbitrationChunkCenterWorld,
                      tempArbitrationCandidateChunkPos,
                    )
                  ) {
                    continue;
                  }
                  const key = `${tempArbitrationCandidateChunkPos.join()}:${lodValue}`;
                  const state = candidate.tsource.source.chunks.get(key)?.state;
                  // A failed candidate should not block fallback to the next
                  // ranked level, but loaded/system/queued candidates remain
                  // valid so target levels are still actively requested.
                  if (state === ChunkState.FAILED) {
                    continue;
                  }
                  const pos = vec3.fromValues(
                    tempArbitrationCandidateChunkPos[0],
                    tempArbitrationCandidateChunkPos[1],
                    tempArbitrationCandidateChunkPos[2],
                  );
                  selected = {
                    tsource: candidate.tsource,
                    scaleIndex: candidate.scaleIndex,
                    position: pos,
                    key,
                  };
                  break;
                }
                if (selected === undefined) {
                  return;
                }
                const emitKey = `${getObjectId(selected.tsource.source)}|${selected.key}`;
                if (emitted.has(emitKey)) {
                  return;
                }
                emitted.add(emitKey);

                const source = selected.tsource.source;
                source.currentLod = lodValue;
                source.currentRequestGeneration = currentGeneration;
                source.currentRequestOwner =
                  SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D;

                const { chunkLayout } = selected.tsource;
                chunkLayout.globalToLocalSpatial(
                  localCenter,
                  centerDataPosition,
                );
                const { size, finiteRank } = chunkLayout;
                vec3.copy(chunkSize, size);
                for (let i = finiteRank; i < 3; ++i) {
                  chunkSize[i] = 0;
                  localCenter[i] = 0;
                }

                const chunk = source.getChunk(selected.position);
                ++this.numVisibleChunksNeeded;
                if (chunk.state === ChunkState.GPU_MEMORY) {
                  ++this.numVisibleChunksAvailable;
                }
                chunkManager.requestChunk(
                  chunk,
                  priorityTier,
                  getSpatiallyIndexedSkeletonRenderPriority(
                    basePriority,
                    selected.scaleIndex,
                    localCenter,
                    chunkSize,
                    selected.position,
                  ),
                );
              },
            );
            continue;
          }
        }

        const selectedScales = selectScales(scales);
        for (const { tsource, scaleIndex } of selectedScales) {
          const source =
            tsource.source as SpatiallyIndexedSkeletonSourceBackend;
          const { chunkLayout } = tsource;
          chunkLayout.globalToLocalSpatial(localCenter, centerDataPosition);
          const { size, finiteRank } = chunkLayout;
          vec3.copy(chunkSize, size);
          for (let i = finiteRank; i < 3; ++i) {
            chunkSize[i] = 0;
            localCenter[i] = 0;
          }
          source.currentLod = lodValue;
          source.currentRequestGeneration = currentGeneration;
          source.currentRequestOwner = is2dView
            ? SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_2D
            : SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D;
          forEachVisibleVolumetricChunk(
            projectionParameters,
            this.localPosition.value,
            tsource,
            () => {
              const chunk = source.getChunk(tsource.curPositionInChunks);
              ++this.numVisibleChunksNeeded;
              if (chunk.state === ChunkState.GPU_MEMORY) {
                ++this.numVisibleChunksAvailable;
              }
              chunkManager.requestChunk(
                chunk,
                priorityTier,
                getSpatiallyIndexedSkeletonRenderPriority(
                  basePriority,
                  scaleIndex,
                  localCenter,
                  chunkSize,
                  tsource.curPositionInChunks,
                ),
              );
            },
          );
        }
      }
    }
  }
}
