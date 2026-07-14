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
import { ChunkPriorityTier, ChunkState } from "#src/chunk_manager/base.js";
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
  MULTISCALE_SKELETON_FRAGMENT_SOURCE_RPC_ID,
  MULTISCALE_SKELETON_LAYER_RPC_ID,
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
  pickFinestPresentLevelAtOrBelow,
  pickTargetLevelByScreenSize,
} from "#src/skeleton/multiscale_object_selection.js";
import {
  getSpatiallyIndexedSkeletonGridIndex,
  selectSpatiallyIndexedSkeletonEntriesByGridWithFallback,
} from "#src/skeleton/source_selection.js";
import {
  computePhysicalUnitsPerScreenPixelAtPoint,
  getChunkSpacing,
  getMetersPerUnit,
  quantizeSpacingForArbitration,
} from "#src/skeleton/screen_size.js";
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

const MULTISCALE_SKELETON_MANIFEST_CHUNK_PRIORITY = 100;
const MULTISCALE_SKELETON_FRAGMENT_CHUNK_PRIORITY = 50;

/**
 * Chunk describing which pyramid levels a single object is present at
 * (`presentLevels[level]`, index 0 = finest) and the per-level chunk
 * spacing shared by every object in the store (`levelSpacings`, copied
 * from the concrete data source's static per-level metadata). Mirrors
 * `MultiscaleManifestChunk` in `src/mesh/backend.ts`, adapted from an
 * octree+lodScales manifest to a flat per-level presence list, since a
 * skeleton/streamline object has no natural spatial octree of its own.
 *
 * Level selection is driven by a real-world resolution target
 * (`skeletonGridResolutionTarget3d`, shared with pass-1's identically
 * named control), not by this object's on-screen footprint — so unlike
 * precomputed meshes' manifests, no bounding box is tracked here.
 *
 * `levelSpacings[level]` is in REAL-WORLD METERS — the concrete source is
 * responsible for converting its native per-level chunk shape to meters
 * using the store's declared coordinate-space scale, so both the backend
 * priority computation and the frontend draw/histogram can compare it
 * directly against `skeletonGridResolutionTarget3d` (also meters) with no
 * further scale conversion. This deliberately does NOT go through
 * `getMetersPerUnit(projectionParameters)` at the use sites: that display-
 * space scalar can disagree with the store-space scale that calibrates
 * the "Resolution (skeleton grid 3D)" widget axis, which previously made
 * every level's bar collapse onto the same axis position.
 */
export class MultiscaleSkeletonManifestChunk extends Chunk {
  objectId: bigint = 0n;
  resolvedOid: number | undefined;
  presentLevels: boolean[] | null = null;
  levelSpacings: Float32Array | null = null;

  // We can't save a reference to objectId, because it may be a temporary
  // object.
  initializeManifestChunk(key: string, objectId: bigint) {
    super.initialize(key);
    this.objectId = objectId;
  }

  freeSystemMemory() {
    this.presentLevels = null;
    this.levelSpacings = null;
  }

  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    msg.presentLevels = this.presentLevels;
    msg.levelSpacings = this.levelSpacings;
  }

  downloadSucceeded() {
    // Cheap fixed estimate — the actual geometry lives in fragment chunks,
    // not here (mirrors mesh's `ManifestChunk`/`MultiscaleManifestChunk`).
    this.systemMemoryBytes = 100;
    this.gpuMemoryBytes = 0;
    super.downloadSucceeded();
    if (this.priorityTier < ChunkPriorityTier.RECENT) {
      this.source!.chunkManager.scheduleUpdateChunkPriorities();
    }
  }

  toString() {
    return this.objectId.toString();
  }
}

/**
 * Chunk holding one object's aggregated skeleton geometry at ONE pyramid
 * level — the whole-object equivalent of a single-level `SkeletonChunk`,
 * keyed by `${objectKey}/${level}` instead of just `${objectKey}`.
 */
export class MultiscaleSkeletonFragmentChunk
  extends Chunk
  implements SkeletonChunkData
{
  level = 0;
  manifestChunk: MultiscaleSkeletonManifestChunk | null = null;
  vertexPositions: Float32Array | null = null;
  vertexAttributes: TypedNumberArray[] | null = null;
  indices: Uint32Array | null = null;

  freeSystemMemory() {
    freeSkeletonChunkSystemMemory(this);
    this.manifestChunk = null;
  }

  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    msg.level = this.level;
    serializeSkeletonChunkData(this, msg, transfers);
    freeSkeletonChunkSystemMemory(this);
  }

  downloadSucceeded() {
    this.systemMemoryBytes = this.gpuMemoryBytes =
      this.indices!.byteLength + getVertexAttributeBytes(this);
    super.downloadSucceeded();
  }
}

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
export interface MultiscaleSkeletonSource {
  // TODO: Move this declaration to class definition below and declare abstract once
  // TypeScript supports mixins with abstract classes.
  downloadFragment(
    chunk: MultiscaleSkeletonFragmentChunk,
    signal: AbortSignal,
  ): Promise<void>;
}

/**
 * Object-keyed, multi-resolution skeleton source: a two-tier chunk model
 * (manifest chunk describing per-level presence/extent, fragment chunks
 * holding one level's aggregated geometry) mirroring
 * `MultiscaleMeshSource`/`MultiscaleFragmentSource` in `src/mesh/backend.ts`.
 * Data-source-agnostic — concrete subclasses (e.g. zarr-vectors) implement
 * `downloadFragment`/manifest-chunk `download`.
 */
// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
export class MultiscaleSkeletonSource extends ChunkSource {
  declare chunks: Map<string, MultiscaleSkeletonManifestChunk>;
  fragmentSource: MultiscaleSkeletonFragmentSource;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    const fragmentSource = (this.fragmentSource = this.registerDisposer(
      rpc.getRef<MultiscaleSkeletonFragmentSource>(options.fragmentSource),
    ));
    fragmentSource.meshSource = this;
  }

  getChunk(objectId: bigint) {
    const key = getObjectKey(objectId);
    let chunk = this.chunks.get(key);
    if (chunk === undefined) {
      chunk = this.getNewChunk_(MultiscaleSkeletonManifestChunk);
      chunk.initializeManifestChunk(key, objectId);
      this.addChunk(chunk);
    }
    return chunk;
  }

  getFragmentChunk(
    manifestChunk: MultiscaleSkeletonManifestChunk,
    level: number,
  ) {
    const key = `${manifestChunk.key}/${level}`;
    const fragmentSource = this.fragmentSource;
    let chunk = fragmentSource.chunks.get(
      key,
    ) as MultiscaleSkeletonFragmentChunk;
    if (chunk === undefined) {
      chunk = fragmentSource.getNewChunk_(MultiscaleSkeletonFragmentChunk);
      chunk.initialize(key);
      chunk.level = level;
      chunk.manifestChunk = manifestChunk;
      fragmentSource.addChunk(chunk);
    }
    return chunk;
  }
}

@registerSharedObject(MULTISCALE_SKELETON_FRAGMENT_SOURCE_RPC_ID)
export class MultiscaleSkeletonFragmentSource extends ChunkSource {
  declare chunks: Map<string, MultiscaleSkeletonFragmentChunk>;
  meshSource: MultiscaleSkeletonSource | null = null;
  download(chunk: MultiscaleSkeletonFragmentChunk, signal: AbortSignal) {
    return this.meshSource!.downloadFragment(chunk, signal);
  }
}

@registerSharedObject(MULTISCALE_SKELETON_LAYER_RPC_ID)
export class MultiscaleSkeletonRenderLayerBackend extends withSegmentationLayerBackendState(
  withSharedVisibility(withChunkManager(RenderLayerBackend)),
) {
  source: MultiscaleSkeletonSource;
  /**
   * Real-world (meters) target resolution, shared with pass-1's
   * identically named `skeletonGridResolutionTarget3d` control. Compared
   * directly against each manifest chunk's `levelSpacings` (also in
   * meters) to pick the target level — no camera/display-space dependency
   * (an object-keyed source has no natural screen footprint, and working
   * in real-world units keeps the target consistent with the widget
   * axis/slider and with pass-1).
   */
  skeletonGridResolutionTarget3d: SharedWatchableValue<number>;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.source = this.registerDisposer(
      rpc.getRef<MultiscaleSkeletonSource>(options.source),
    );
    this.skeletonGridResolutionTarget3d = rpc.get(
      options.skeletonGridResolutionTarget3d,
    );
    this.registerDisposer(
      this.skeletonGridResolutionTarget3d.changed.add(() =>
        this.chunkManager.scheduleUpdateChunkPriorities(),
      ),
    );
    this.registerDisposer(
      this.chunkManager.recomputeChunkPriorities.add(() => {
        this.updateChunkPriorities();
      }),
    );
  }

  attach(attachment: RenderLayerBackendAttachment<RenderedViewBackend>) {
    const scheduleUpdateChunkPriorities = () =>
      this.chunkManager.scheduleUpdateChunkPriorities();
    const { view } = attachment;
    attachment.registerDisposer(
      view.projectionParameters.changed.add(scheduleUpdateChunkPriorities),
    );
    attachment.registerDisposer(
      view.visibility.changed.add(scheduleUpdateChunkPriorities),
    );
    attachment.registerDisposer(scheduleUpdateChunkPriorities);
    scheduleUpdateChunkPriorities();
  }

  private updateChunkPriorities() {
    const maxVisibility = this.visibility.value;
    if (maxVisibility === Number.NEGATIVE_INFINITY) {
      return;
    }
    const manifestChunks = new Array<MultiscaleSkeletonManifestChunk>();
    this.chunkManager.registerLayer(this);
    {
      const priorityTier = getPriorityTier(maxVisibility);
      const basePriority = getBasePriority(maxVisibility);
      const { source, chunkManager } = this;
      forEachVisibleSegment(this, (objectId) => {
        const manifestChunk = source.getChunk(objectId);
        ++this.numVisibleChunksNeeded;
        chunkManager.requestChunk(
          manifestChunk,
          priorityTier,
          basePriority + MULTISCALE_SKELETON_MANIFEST_CHUNK_PRIORITY,
        );
        const state = manifestChunk.state;
        if (
          state === ChunkState.SYSTEM_MEMORY_WORKER ||
          state === ChunkState.SYSTEM_MEMORY ||
          state === ChunkState.GPU_MEMORY
        ) {
          ++this.numVisibleChunksAvailable;
          if (manifestChunk.presentLevels !== null) {
            manifestChunks.push(manifestChunk);
          }
        }
      });
    }
    if (manifestChunks.length === 0) return;
    // `skeletonGridResolutionTarget3d` is a real-world (meters) target,
    // independent of camera position, and `levelSpacings` is already in
    // real-world meters (the concrete source populates it from the
    // store's declared scale — see `MultiscaleSkeletonManifestChunk
    // .levelSpacings`), so the target level is the same for every attached
    // view and depends on no camera/display state. Compute it once per
    // manifest chunk.
    const targetSpacingMeters = this.skeletonGridResolutionTarget3d.value;
    const { source, chunkManager } = this;
    for (const manifestChunk of manifestChunks) {
      const presentLevels = manifestChunk.presentLevels!;
      const { levelSpacings } = manifestChunk;
      if (levelSpacings === null) continue;
      const target = pickTargetLevelByScreenSize(
        levelSpacings,
        targetSpacingMeters,
      );
      const targetActualLevel = pickFinestPresentLevelAtOrBelow(
        presentLevels,
        target,
      );
      if (targetActualLevel === undefined) continue;
      let coarsestPresentLevel = targetActualLevel;
      for (let level = presentLevels.length - 1; level >= 0; --level) {
        if (presentLevels[level]) {
          coarsestPresentLevel = level;
          break;
        }
      }
      for (const { view } of this.attachments.values()) {
        const visibility = view.visibility.value;
        if (visibility === Number.NEGATIVE_INFINITY) {
          continue;
        }
        const priorityTier = getPriorityTier(visibility);
        const basePriority = getBasePriority(visibility);
        for (
          let level = targetActualLevel;
          level <= coarsestPresentLevel;
          ++level
        ) {
          if (!presentLevels[level]) continue;
          const fragmentChunk = source.getFragmentChunk(manifestChunk, level);
          ++this.numVisibleChunksNeeded;
          chunkManager.requestChunk(
            fragmentChunk,
            priorityTier,
            basePriority +
              MULTISCALE_SKELETON_FRAGMENT_CHUNK_PRIORITY -
              coarsestPresentLevel +
              level,
          );
          if (fragmentChunk.state === ChunkState.GPU_MEMORY) {
            ++this.numVisibleChunksAvailable;
          }
        }
      }
    }
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
  numRealVertices: number | undefined;

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
          const ordered = selectSpatiallyIndexedSkeletonEntriesByGridWithFallback(
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
          const orderedCandidates =
            selectSpatiallyIndexedSkeletonEntriesByGridWithFallback(
              scales.map((tsource, scaleIndex) => ({ tsource, scaleIndex })),
              skeletonGridLevel,
              ({ tsource }) => getSpatiallyIndexedSkeletonGridIndex(tsource),
            );
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
            const fallbackAnchor = orderedCandidates.reduce((best, candidate) =>
              spacingMeters(candidate) < spacingMeters(best) ? candidate : best,
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
                    return a.scaleIndex - b.scaleIndex;
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
