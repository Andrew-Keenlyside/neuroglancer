/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * The edit half of the zarr-vectors spatial-skeleton source: what makes the
 * fork's Skeleton tools act on a ZVF store instead of reporting it read-only.
 *
 * The UI is reached entirely by duck typing -- a source that exposes
 * `readonly: false` plus the five required command factories
 * (`SPATIAL_SKELETON_EDIT_COMMAND_METADATA`) becomes editable, and the existing
 * Split/Merge/Edit tools drive it with no UI change. This module supplies those
 * factories.
 *
 * Only `splitSkeletons` does anything: it is the operation the proof of concept
 * is about. The other four are present because the duck type requires all five
 * -- a source missing one is not editable at all -- and they fail with a clear
 * message rather than pretending.
 *
 * A ZVF store cannot be written from the browser (neuroglancer's kvstore is
 * read-only by construction, `src/kvstore/index.ts`), so the edit is performed
 * by a loopback service that rewrites the store; see
 * `python/zvf_edit_poc/edit_service.py`.
 */

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  addSegmentToVisibleSets,
  removeSegmentFromVisibleSets,
} from "#src/segmentation_display_state/base.js";
import { SpatialSkeletonActions } from "#src/skeleton/actions.js";
import type {
  SpatialSkeletonCommandPayload,
  SpatialSkeletonEditCommandFactory,
} from "#src/skeleton/command_factories.js";
import type { SpatialSkeletonCommand } from "#src/skeleton/command_history.js";
import { StatusMessage } from "#src/status.js";

/** Where the edit service lives, and which store under it to edit. */
export interface ZarrVectorsEditTarget {
  readonly serviceUrl: string;
  readonly store: string;
  readonly token?: string;
}

interface SplitResult {
  ids: number[];
  sizes: number[];
  cutNode: number;
  parentNode: number;
  objects: number[];
}

async function postSplit(
  target: ZarrVectorsEditTarget,
  segmentId: number,
  nodeId: number,
): Promise<SplitResult> {
  const response = await fetch(
    `${target.serviceUrl.replace(/\/$/, "")}/split`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        store: target.store,
        segmentId,
        nodeId,
        token: target.token,
      }),
    },
  );
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      `zarr-vectors edit service: ${body.error ?? response.statusText}`,
    );
  }
  return body as SplitResult;
}

/**
 * Make the layer re-read a store whose bytes changed underneath it.
 *
 * Everything the viewer holds about this layer was decoded from the previous
 * contents: chunks in the worker, the per-segment node cache on the layer, and
 * the navigation state built from it. `invalidateCache()` is the whole-source
 * hammer (`chunk_manager/frontend.ts:494`); the narrower prefix invalidation is
 * no use here because a rewrite moves every fragment.
 */
function refreshLayerFromStore(layer: SegmentationUserLayer): void {
  for (const renderLayer of layer.renderLayers) {
    const sources = (
      renderLayer as unknown as {
        base?: {
          sources?: { chunkSource?: { invalidateCache?: () => void } }[];
        };
      }
    ).base?.sources;
    for (const entry of sources ?? []) {
      entry.chunkSource?.invalidateCache?.();
    }
  }
  for (const dataSource of layer.dataSources) {
    const loadState = dataSource.loadState;
    if (loadState === undefined || loadState.error !== undefined) continue;
    for (const subsource of loadState.subsources) {
      const source = subsource.subsourceEntry.subsource as {
        zarrVectors?: { invalidateCache?: () => void };
      };
      source.zarrVectors?.invalidateCache?.();
    }
  }
  // The node cache is keyed by segment id and was filled from the pre-edit
  // geometry; the split renames what those ids mean.
  layer.spatialSkeletonState.clearInspectedSkeletonCache();
  layer.markSpatialSkeletonNodeDataChanged({
    invalidateFullSkeletonCache: true,
  });
}

/** Show the two halves the split produced, and drop the pre-edit selection. */
function selectSplitResults(
  layer: SegmentationUserLayer,
  before: number,
  ids: number[],
): void {
  const group = layer.displayState.segmentationGroupState.value;
  // The split reuses the original id for one half, so only drop it when it is
  // genuinely gone -- otherwise the surviving half would be hidden.
  if (!ids.includes(before)) {
    removeSegmentFromVisibleSets(group, BigInt(before), { deselect: true });
  }
  for (const id of ids) {
    addSegmentToVisibleSets(group, BigInt(id));
  }
}

class ZarrVectorsSplitCommand implements SpatialSkeletonCommand {
  readonly label = "split skeleton";

  constructor(
    private readonly target: ZarrVectorsEditTarget,
    private readonly layer: SegmentationUserLayer,
    private readonly segmentId: number,
    private readonly nodeId: number,
  ) {}

  async execute(): Promise<void> {
    const result = await postSplit(this.target, this.segmentId, this.nodeId);
    refreshLayerFromStore(this.layer);
    selectSplitResults(this.layer, this.segmentId, result.ids);
    StatusMessage.showTemporaryMessage(
      `Split object ${this.segmentId} above node ${result.cutNode} into ` +
        `${result.ids.join(" and ")} (${result.sizes.join(" / ")} vertices).`,
      6000,
    );
  }

  async undo(): Promise<void> {
    // Honest refusal. Undo would be a merge, and the service implements only
    // the split; a silent no-op would leave the history claiming the store was
    // restored when it was not.
    throw new Error(
      "zarr-vectors: undo of a split is not implemented in this prototype -- " +
        "re-run the extraction to get back to a single object.",
    );
  }
}

function unsupported(action: string): SpatialSkeletonEditCommandFactory {
  return {
    action: action as SpatialSkeletonEditCommandFactory["action"],
    createCommand(): SpatialSkeletonCommand {
      throw new Error(
        `zarr-vectors: ${action} is not implemented in this prototype; only ` +
          "splitting a skeleton is wired to the edit service.",
      );
    },
  };
}

/**
 * The eleven factory slots the duck type inspects. All five REQUIRED ones must
 * be present or the source is not editable and every tool stays disabled --
 * which is why the unimplemented four are declared rather than omitted.
 */
export function makeZarrVectorsEditCommands(target: ZarrVectorsEditTarget) {
  return {
    splitSkeletonsCommand: {
      action: SpatialSkeletonActions.splitSkeletons,
      createCommand(
        layer: SegmentationUserLayer,
        payload: SpatialSkeletonCommandPayload,
      ): SpatialSkeletonCommand {
        const { nodeId, segmentId } = payload as {
          nodeId?: number;
          segmentId?: number;
        };
        if (
          !Number.isSafeInteger(nodeId) ||
          !Number.isSafeInteger(segmentId) ||
          nodeId! <= 0 ||
          segmentId! <= 0
        ) {
          throw new Error(
            "zarr-vectors: a split needs a picked node and its object; " +
              `got node ${nodeId}, object ${segmentId}. Nodes are only ` +
              "pickable when the layer draws points as well as lines.",
          );
        }
        return new ZarrVectorsSplitCommand(target, layer, segmentId!, nodeId!);
      },
    } satisfies SpatialSkeletonEditCommandFactory,
    addNodesCommand: unsupported(SpatialSkeletonActions.addNodes),
    moveNodesCommand: unsupported(SpatialSkeletonActions.moveNodes),
    deleteNodesCommand: unsupported(SpatialSkeletonActions.deleteNodes),
    mergeSkeletonsCommand: unsupported(SpatialSkeletonActions.mergeSkeletons),
  };
}
