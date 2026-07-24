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
import type {
  LabelSampler,
  Roi,
  RoiGroupConfig,
  RoiLabelField,
  RoiObjectAttrColumn,
} from "#src/datasource/zarr-vectors/roi.js";
import {
  makeLabelSampler,
  roiRegionBounds,
} from "#src/datasource/zarr-vectors/roi.js";
// Pure ROI geometry (no zarr/render deps) drives the streamline filter's
// backend recompute for zarr-vectors spatially-indexed skeleton (tract) layers.
// Inert for every other skeleton layer: the whole feature is guarded on the
// per-layer `roiPassingSegments` shared set being present (undefined here).
import {
  selectHighDetail,
  computeGroupedPassingSet,
  computePerGroupPassingSets,
  diffPassingSet,
  type RoiFilterableChunk,
} from "#src/datasource/zarr-vectors/roi_filter_backend.js";
// The dissection geometry itself evaluates in Python/WASM; this client is the
// bridge, and falls back to the TypeScript implementation above when no such
// service is present (an ordinary Neuroglancer build).
import {
  RoiFilterServiceClient,
  type RoiFilterChunkEntry,
  type RoiFilterResult,
} from "#src/datasource/zarr-vectors/roi_filter_service.js";
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
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_ROI_EXPORT_IDS_RPC_ID,
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
import type { Uint64Map } from "#src/uint64_map.js";
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
import {
  registerPromiseRPC,
  registerRPC,
  registerSharedObject,
} from "#src/worker_rpc.js";
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
const tempRoiChunkLower = new Float32Array(3);
const tempRoiChunkUpper = new Float32Array(3);
const tempRoiChunkPosition = new Float32Array(3);

/**
 * Cap on chunks the ROI-residency guarantee may pin.
 *
 * The guarantee exists so a small region's geometry is always present; a region
 * spanning the volume would instead pin the level and defeat the memory
 * ceiling. Past this the guarantee is dropped rather than honoured -- a
 * dissection that broad is not one the ceiling can serve anyway.
 */
const MAX_ROI_REGION_CHUNKS = 512;

/**
 * Priority boost for ROI-region chunks, above the frustum's own requests: the
 * filter cannot be correct without them, whereas a merely-visible chunk only
 * affects what is drawn.
 */
const ROI_REGION_CHUNK_PRIORITY = 10;

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

registerPromiseRPC(
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_ROI_EXPORT_IDS_RPC_ID,
  function (this: RPC, x: { layer: number; groups: RoiGroupConfig[] }) {
    const layer = this.get(
      x.layer,
    ) as SpatiallyIndexedSkeletonRenderLayerBackend;
    // The fold is synchronous once the resident chunks are in hand; wrap the
    // result in the {value, transfers} shape registerPromiseRPC expects.
    return Promise.resolve({
      value: layer.computeRoiExportIds(x.groups),
      transfers: [],
    });
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
      attributeBytes + (this.roiFilterableChunk?.fragmentIndex.byteLength ?? 0);
    super.downloadSucceeded();
    // Reaching SYSTEM_MEMORY_WORKER is what makes this chunk's geometry
    // filterable (roiFilterableChunk is set and deliberately retained past
    // serialize). But the ROI passing-set recompute only re-runs on
    // `recomputeChunkPrioritiesLate`, whose sole post-arrival trigger is
    // `gpuMemoryChanged` -- which fires on GPU promotion/eviction, NOT on
    // reaching worker memory. The ROI filter fetches region chunks regardless
    // of the camera and folds the FINEST resident level, so a region chunk the
    // view is not drawing lands here yet is never GPU-promoted; without this
    // its arrival never re-folds and the passing set stays computed over the
    // chunks resident at the last ROI edit (the "drag away and back to refresh"
    // symptom). Re-run the late hook: the added chunk changes the recompute
    // signature, so `maybeRecomputeRoiPassingSet` re-folds and includes it.
    //
    // THROTTLED, not per-arrival: a whole-brain load completes thousands of
    // chunks in separate macrotasks, and each un-throttled call would re-arm a
    // full multi-layer priority pass (`scheduleUpdateChunkPriorities` coalesces
    // only within one task). The throttle caps that at the 200 ms
    // `gpuMemoryChanged` cadence, trailing so the final arrival still lands a
    // refold. Gated on roiFilterableChunk so it is inert for every non-tract
    // skeleton layer; the recompute is itself one-at-a-time and cheap when no
    // ROIs exist (it early-returns on the `hasRois` guard).
    if (this.roiFilterableChunk !== undefined) {
      this.chunkManager.scheduleUpdateChunkPrioritiesThrottled();
    }
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
  //   - roiGroups: the ordered ROI groups (each an independent dissection).
  // The passing set is computed whenever ROIs exist, independent of whether the
  // user has the filter switched on — the *active* flag is purely a frontend
  // shader concern (uRoiFilterActive), so keeping the set current means enabling
  // the filter is instant rather than flashing the whole tractogram to
  // ghost-alpha while an async recompute catches up.
  roiPassingSegments?: Uint64Set;
  roiGroups?: SharedWatchableValue<readonly RoiGroupConfig[]>;
  /** Per-object numeric attribute columns (length, …) for the length filter and
   *  object-attribute colouring. */
  roiObjectAttrColumns?: SharedWatchableValue<
    ReadonlyMap<string, RoiObjectAttrColumn>
  >;
  /**
   * Dense anatomical label grid from a linked parcellation layer, sampled per
   * vertex to decide `labelMask` ROIs. Undefined when no parcellation is linked;
   * label-mask ROIs then select nothing until it loads. */
  roiLabelField?: SharedWatchableValue<RoiLabelField | undefined>;
  /** Shared id -> packed group colour for passing tracts (colour-by-group). */
  roiSegmentColors?: Uint64Map;
  /** Shared set of passing tracts of visible high-detail groups (drives pass-2). */
  roiHighDetailSegments?: Uint64Set;
  /**
   * Maximum streamlines pass 2 may load, or undefined for no cap.
   *
   * A count, not a byte figure: pass 2 fetches whole objects, so a tract that
   * merely clips the view costs its entire length.
   */
  roiHighDetailBudget?: SharedWatchableValue<number>;
  /** Set when an ROI edit needs a recompute even if the resident chunk set is unchanged. */
  private roiRecomputePending = false;
  /** An evaluation is outstanding; at most one runs at a time. */
  private roiRecomputeInFlight = false;
  /** Something asked for a recompute while one was in flight. */
  private roiRecomputeQueued = false;
  /** Client for the Python/WASM dissection service; built on first evaluation. */
  private roiFilterServiceClient: RoiFilterServiceClient | undefined;
  /** Signature of the last resident-chunk set filtered over, to skip redundant recomputes. */
  private roiLastChunkSignature = "";
  /** The colour attribution last pushed to `roiSegmentColors`, for diffing. */
  private roiLastColorById = new Map<bigint, number>();

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
      if (options.roiSegmentColors !== undefined) {
        this.roiSegmentColors = rpc.get(options.roiSegmentColors);
      }
      if (options.roiHighDetailSegments !== undefined) {
        this.roiHighDetailSegments = rpc.get(options.roiHighDetailSegments);
      }
      const roiGroups = (this.roiGroups = rpc.get(options.roiGroups));
      const scheduleRoiRecompute = () => {
        this.roiRecomputePending = true;
        this.chunkManager.scheduleUpdateChunkPriorities();
      };
      this.registerDisposer(roiGroups.changed.add(scheduleRoiRecompute));
      if (options.roiObjectAttrColumns !== undefined) {
        const columns = (this.roiObjectAttrColumns = rpc.get(
          options.roiObjectAttrColumns,
        ));
        // New per-object values (a length attribute finished loading, a group
        // switch) change what the length filter / attribute colour resolve to.
        this.registerDisposer(columns.changed.add(scheduleRoiRecompute));
      }
      if (options.roiLabelField !== undefined) {
        const field = (this.roiLabelField = rpc.get(options.roiLabelField));
        // A parcellation loading (or the linked layer changing) changes which
        // tracts a label-mask ROI selects, so it is a recompute trigger too.
        this.registerDisposer(field.changed.add(scheduleRoiRecompute));
      }
      if (options.roiHighDetailBudget !== undefined) {
        const budget = (this.roiHighDetailBudget = rpc.get(
          options.roiHighDetailBudget,
        ));
        // Lowering the cap must evict, raising it must fetch, so a change is a
        // recompute trigger exactly like an ROI edit. Registered after
        // `scheduleRoiRecompute` exists -- a `const` arrow referenced above its
        // declaration is a temporal dead zone, not a hoisted function.
        this.registerDisposer(budget.changed.add(scheduleRoiRecompute));
      }
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
          const source =
            tsource.source as SpatiallyIndexedSkeletonSourceBackend;
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
    if (passingSet === undefined || this.roiGroups === undefined) return;
    // Compute whenever any visible group has ROIs, regardless of the
    // (frontend-only) active flag, so enabling the filter is instant. No ROIs
    // yields an empty passing set; the shader treats "active with no ROIs" as
    // off, so an empty set is never mistaken for "everything fails".
    const groups = this.roiGroups.value;
    const hasRois = groups.some((g) => g.visible && g.rois.length !== 0);
    // Bucket by PYRAMID LEVEL (the source's `gridIndex`), not by `chunk.lod`.
    // `lod` records which render level a chunk was requested for; the level it
    // actually belongs to is a property of its source. Bucketing by `lod` tied
    // the dissection to whatever the camera had selected.
    const byLevel = new Map<
      number,
      { chunks: RoiFilterChunkEntry[]; count: number; hash: number }
    >();
    if (hasRois) {
      for (const source of this.roiFilterableSources()) {
        const gridIndex = getSpatiallyIndexedSkeletonGridIndex(source);
        if (gridIndex === undefined) continue;
        for (const chunk of source.chunks.values()) {
          const c = chunk as SpatiallyIndexedSkeletonChunk;
          const data = c.roiFilterableChunk;
          if (data === undefined) continue;
          let bucket = byLevel.get(gridIndex);
          if (bucket === undefined) {
            bucket = { chunks: [], count: 0, hash: 0 };
            byLevel.set(gridIndex, bucket);
          }
          bucket.chunks.push({ key: c.key ?? "", data });
          bucket.count++;
          bucket.hash = (bucket.hash ^ cheapStringHash(c.key ?? "")) | 0;
        }
      }
    }
    // Evaluate at the FINEST level with any resident geometry -- larger
    // gridIndex is finer (`frontend.ts`: levelPaths[0], the finest, is assigned
    // `numLevels - 1`).
    //
    // `requestRoiRegionChunks` guarantees the ROI regions' chunks at every
    // level, and a fragment outside every region cannot change a crossing test,
    // so the finest level is fully decidable from a bounded set of chunks even
    // though the level as a whole is nowhere near resident. That is what lets
    // the dissection see streamlines that exist ONLY at fine levels, rather
    // than being confined to whatever level the camera happens to draw.
    //
    // Still a SINGLE level: mixing differently-decimated geometry for one id
    // would let a coarse level pollute a fine verdict (see the note above
    // `roiFilterableSources`). Mixing happens over OBJECTS, not geometry.
    let targetLevel = -1;
    for (const level of byLevel.keys()) {
      if (level > targetLevel) targetLevel = level;
    }
    const bucket = targetLevel < 0 ? undefined : byLevel.get(targetLevel);
    const chunks = bucket?.chunks ?? [];

    const signature = `${targetLevel}:${bucket?.count ?? 0}:${bucket?.hash ?? 0}`;

    // While an evaluation is outstanding, ALWAYS queue and return. This check
    // must come before the signature comparison below, because during a flight
    // `roiLastChunkSignature` still holds the last COMMITTED signature, not the
    // in-flight one. Comparing against it would let a pass that happens to
    // observe the previously-committed world state early-return without
    // queueing, after which the in-flight result commits its own (now stale)
    // signature and nothing is left to correct it:
    //
    //   1. committed "lod4:A"; user moves to lod 3
    //   2. dispatch over "lod3:B"
    //   3. mid-flight, user moves back to lod 4 -> pass sees "lod4:A", which
    //      equals the committed signature -> returns, queueing nothing
    //   4. the lod-3 result lands and commits "lod3:B"
    //   5. drain finds nothing queued; the filter now shows lod-3 verdicts
    //      over lod-4 geometry until some unrelated event happens to fire
    //
    // Queueing unconditionally costs one extra pass per flight, whose own
    // signature check then early-returns if nothing really changed.
    if (this.roiRecomputeInFlight) {
      this.roiRecomputeQueued = true;
      return;
    }

    if (!this.roiRecomputePending && signature === this.roiLastChunkSignature) {
      return;
    }

    // The evaluation itself may be remote (the Python/WASM dissection service),
    // so it can no longer be committed at dispatch the way a synchronous call
    // could. Only ONE runs at a time (the guard above): the ROI sliders fire on
    // `input` with no debounce, so a drag would otherwise queue a recompute per
    // frame, each superseded before it landed. A drag coalesces into "the one
    // running" plus "one more afterwards". And `roiLastChunkSignature` /
    // `roiRecomputePending` commit on COMPLETION -- clearing them at dispatch
    // would let a failed or superseded request lose the layer's dirty state and
    // leave the passing set stale indefinitely.
    this.roiRecomputeInFlight = true;
    this.roiRecomputeQueued = false;
    this.roiRecomputePending = false;

    this.evaluateRoiGroups(chunks, groups).then(
      (result) => {
        this.roiRecomputeInFlight = false;
        // The evaluation can now outlive the layer, which was impossible while
        // it was synchronous. Writing to the shared sets after disposal would
        // resurrect RPC objects the frontend has already torn down.
        if (this.wasDisposed) return;
        this.roiLastChunkSignature = signature;
        this.applyRoiResult(result);
        this.drainRoiRecompute();
      },
      (e) => {
        this.roiRecomputeInFlight = false;
        if (this.wasDisposed) return;
        // Leave the layer dirty: the signature is not advanced and the pending
        // flag is re-armed, so the next priority pass retries this same work.
        this.roiRecomputePending = true;
        console.error("ROI streamline filter: recompute failed", e);
        this.drainRoiRecompute();
      },
    );
  }

  /** Re-enter the recompute if anything arrived while one was in flight. */
  private drainRoiRecompute() {
    if (this.roiRecomputeQueued || this.roiRecomputePending) {
      this.roiRecomputeQueued = false;
      this.chunkManager.scheduleUpdateChunkPriorities();
    }
  }

  /**
   * Evaluate the dissection, preferring the Python/WASM service.
   *
   * Falls back to the in-worker TypeScript implementation whenever there is no
   * service to talk to — an ordinary Neuroglancer build has no request
   * interceptor — so the filter behaves exactly as it did before in that case
   * rather than silently doing nothing.
   */
  private async evaluateRoiGroups(
    chunks: readonly RoiFilterChunkEntry[],
    groups: readonly RoiGroupConfig[],
  ): Promise<RoiFilterResult> {
    let client = this.roiFilterServiceClient;
    if (client === undefined) {
      client = this.roiFilterServiceClient = new RoiFilterServiceClient(
        `${this.rpcId}`,
      );
    }
    // Applied to BOTH paths. The Python/WASM service does its own fold and
    // returns early, so capping only the TypeScript fallback would make the
    // budget silently stop applying wherever the service is available -- which
    // is precisely the build the budget matters most in.
    const budget = this.roiHighDetailBudget?.value;
    const attrColumns = this.roiObjectAttrColumns?.value;
    const labelField = this.roiLabelField?.value;
    const sampleLabel: LabelSampler | undefined =
      labelField !== undefined ? makeLabelSampler(labelField) : undefined;
    // The dissection service knows nothing about per-object length filtering,
    // attribute colouring (or the unified colour-by model, where a group can opt
    // out of the flat colour override), OR label-mask ROIs (which need the dense
    // parcellation grid the service does not have). When any visible group needs
    // that, fall back to the in-worker TypeScript path, which has the values and
    // applies it per group. The common case (all groups flat "group" colour, no
    // length filter, only geometric ROIs) still takes the fast service path.
    const needsLocal = groups.some(
      (g) =>
        g.visible &&
        (g.lengthRange !== undefined ||
          (g.colorBy !== undefined && g.colorBy.kind !== "group") ||
          g.rois.some((r) => r.shape.kind === "labelMask")),
    );
    if (!client.isUnavailable && !needsLocal) {
      const remote = await client.compute(chunks, groups);
      if (remote !== undefined) {
        return {
          ...remote,
          highDetail: selectHighDetail(
            remote.passing,
            remote.highDetail,
            budget,
          ),
        };
      }
    }
    return computeGroupedPassingSet(
      chunks.map((c) => c.data),
      groups,
      budget,
      attrColumns,
      sampleLabel,
    );
  }

  /** Push one evaluation's result to the three shared objects, as minimal diffs. */
  private applyRoiResult({ passing, colorById, highDetail }: RoiFilterResult) {
    const passingSet = this.roiPassingSegments;
    if (passingSet === undefined) return;
    const current = new Set<bigint>(passingSet.keys());
    const { added, removed } = diffPassingSet(passing, current);
    if (removed.length !== 0) passingSet.delete(removed);
    if (added.length !== 0) passingSet.add(added);

    const highDetailSet = this.roiHighDetailSegments;
    if (highDetailSet !== undefined) {
      const currentHd = new Set<bigint>(highDetailSet.keys());
      const hdDiff = diffPassingSet(highDetail, currentHd);
      if (hdDiff.removed.length !== 0) highDetailSet.delete(hdDiff.removed);
      if (hdDiff.added.length !== 0) highDetailSet.add(hdDiff.added);
    }

    // Push the colour attribution to the shared map as a minimal diff (the
    // frontend mirrors it into segmentStatedColors when colour-by-group is on).
    const colors = this.roiSegmentColors;
    if (colors !== undefined) {
      const last = this.roiLastColorById;
      for (const [id, color] of colorById) {
        if (last.get(id) !== color) colors.set(id, BigInt(color));
      }
      for (const id of last.keys()) {
        if (!colorById.has(id)) colors.delete(id);
      }
      this.roiLastColorById = colorById;
    }
  }

  /**
   * Evaluate `groups` over the currently-resident chunks and return, per group,
   * its passing object ids as decimal strings. Drives the tract export: the tab
   * hands these ids to the exporter, which reads exactly those objects instead
   * of re-reading and re-folding the whole level.
   *
   * WYSIWYG by construction: it folds over the same finest-resident-level chunks
   * the on-screen passing set is computed from (see `maybeRecomputeRoiPassingSet`
   * for why a single level, and why the finest resident one). The result is
   * positional -- `perGroup[i]` corresponds to `groups[i]` -- because group
   * names are not unique.
   *
   * Ids are strings, not numbers: an object id is a uint64 and JSON would round
   * one past 2**53. Synchronous: the fold is pure once the chunks are in hand.
   */
  computeRoiExportIds(groups: readonly RoiGroupConfig[]): {
    perGroup: string[][];
  } {
    const byLevel = new Map<number, RoiFilterableChunk[]>();
    for (const source of this.roiFilterableSources()) {
      const gridIndex = getSpatiallyIndexedSkeletonGridIndex(source);
      if (gridIndex === undefined) continue;
      for (const chunk of source.chunks.values()) {
        const c = chunk as SpatiallyIndexedSkeletonChunk;
        const data = c.roiFilterableChunk;
        if (data === undefined) continue;
        let arr = byLevel.get(gridIndex);
        if (arr === undefined) {
          arr = [];
          byLevel.set(gridIndex, arr);
        }
        arr.push(data);
      }
    }
    // Finest level with any resident geometry (larger gridIndex is finer).
    let targetLevel = -1;
    for (const level of byLevel.keys()) {
      if (level > targetLevel) targetLevel = level;
    }
    const chunks = targetLevel < 0 ? [] : byLevel.get(targetLevel)!;
    const attrColumns = this.roiObjectAttrColumns?.value;
    const labelField = this.roiLabelField?.value;
    const sampleLabel: LabelSampler | undefined =
      labelField !== undefined ? makeLabelSampler(labelField) : undefined;
    const perGroupSets = computePerGroupPassingSets(
      chunks,
      groups,
      attrColumns,
      sampleLabel,
    );
    return {
      perGroup: perGroupSets.map((s) => Array.from(s, (id) => id.toString())),
    };
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

  /**
   * Request the chunks the ROI regions overlap, whatever the camera is showing.
   *
   * The dissection is folded over RESIDENT chunks, and residency is otherwise
   * driven by the frustum -- so zooming in silently dropped an object's
   * out-of-view fragments and a spanning dissection ("through here AND
   * terminating there") under-selected.
   *
   * The bound that makes this cheap: a fragment lying outside every region can
   * never change a crossing test, so the chunks overlapping the regions are not
   * merely necessary but SUFFICIENT to decide the whole fold. ROIs are small, so
   * this is a handful of chunks, and it holds the single-level evaluation intact
   * rather than folding differently-decimated geometry together.
   *
   * Skipped when any region is a halfspace: unbounded, so no finite chunk set
   * can be guaranteed and the camera's residency is all there is.
   */
  private requestRoiRegionChunks(
    tsource: TransformedSource<
      SpatiallyIndexedSkeletonRenderLayerBackend,
      SpatiallyIndexedSkeletonSourceBackend
    >,
    priorityTier: number,
    basePriority: number,
  ) {
    const groups = this.roiGroups?.value;
    if (groups === undefined) return;
    const rois: Roi[] = [];
    for (const group of groups) {
      if (group.visible && group.rois.length !== 0) rois.push(...group.rois);
    }
    if (rois.length === 0) return;
    const bounds = roiRegionBounds(rois);
    if (bounds === undefined) return;

    const source = tsource.source as SpatiallyIndexedSkeletonSourceBackend;
    const lower = tempRoiChunkLower;
    const upper = tempRoiChunkUpper;
    if (
      !getChunkGridPositionForWorldPoint(tsource, bounds.lower, lower) ||
      !getChunkGridPositionForWorldPoint(tsource, bounds.upper, upper)
    ) {
      return;
    }
    const { lowerChunkBound, upperChunkBound } = tsource.source.spec;
    let total = 1;
    for (let i = 0; i < 3; ++i) {
      const lo = Math.max(Math.min(lower[i], upper[i]), lowerChunkBound[i]);
      const hi = Math.min(Math.max(lower[i], upper[i]), upperChunkBound[i] - 1);
      if (hi < lo) return;
      lower[i] = lo;
      upper[i] = hi;
      total *= hi - lo + 1;
    }
    // A pathological ROI spanning the volume would otherwise pin the whole
    // level resident, defeating the memory ceiling entirely.
    if (total > MAX_ROI_REGION_CHUNKS) return;

    const pos = tempRoiChunkPosition;
    for (let z = lower[2]; z <= upper[2]; ++z) {
      for (let y = lower[1]; y <= upper[1]; ++y) {
        for (let x = lower[0]; x <= upper[0]; ++x) {
          pos[0] = x;
          pos[1] = y;
          pos[2] = z;
          // Same tier as the visible chunks: these ARE needed to draw a correct
          // dissection, so demoting them to prefetch would let the filter
          // under-select for as long as memory stayed tight.
          this.chunkManager.requestChunk(
            source.getChunk(pos),
            priorityTier,
            basePriority + ROI_REGION_CHUNK_PRIORITY,
          );
        }
      }
    }
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
        // Every level, before any frustum work: the dissection's geometry
        // requirement is set by the ROI regions and the budget, not by what is
        // on screen, so this must not sit inside a branch that has already
        // narrowed to one level.
        for (const tsource of scales) {
          this.requestRoiRegionChunks(
            tsource as TransformedSource<
              SpatiallyIndexedSkeletonRenderLayerBackend,
              SpatiallyIndexedSkeletonSourceBackend
            >,
            priorityTier,
            basePriority,
          );
        }
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
