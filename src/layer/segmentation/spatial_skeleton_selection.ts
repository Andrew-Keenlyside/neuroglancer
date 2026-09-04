/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use layer file except in compliance with the License.
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
 * The spatially-indexed skeleton node inspector: what the selection panel shows
 * when the picked thing is a node of a tract or cell rather than a segment.
 *
 * This is a DOM builder and nothing else. It reads the layer through its public
 * surface -- eight members, all already public -- so it lives beside
 * `index.ts` rather than inside it, which is what keeps SegmentationUserLayer
 * close to upstream. The `layer` parameter is what `layer` used to be.
 */

import "#src/layer/segmentation/style.css";
import "#src/layer/segmentation/spatial_skeleton.css";
import svg_circle from "ikonate/icons/circle.svg?raw";
import svg_flag from "ikonate/icons/flag.svg?raw";
import svg_minus from "ikonate/icons/minus.svg?raw";
import svg_origin from "ikonate/icons/origin.svg?raw";
import svg_share_android from "ikonate/icons/share-android.svg?raw";
import { debounce } from "lodash-es";

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  getNodeIdFromLayerSelectionState,
  getSegmentIdFromLayerSelectionValue,
} from "#src/layer/segmentation/selection.js";
import {
  executeSpatialSkeletonDeleteNode,
  executeSpatialSkeletonNodeConfidenceUpdate,
  executeSpatialSkeletonNodeDescriptionUpdate,
  executeSpatialSkeletonNodeRadiusUpdate,
  executeSpatialSkeletonNodeTrueEndUpdate,
  showSpatialSkeletonActionError,
} from "#src/layer/segmentation/spatial_skeleton_commands.js";

import { getCssColor } from "#src/segment_color.js";
import type { SegmentationDisplayState } from "#src/segmentation_display_state/frontend.js";
import { getBaseObjectColor } from "#src/segmentation_display_state/frontend.js";

import { SpatialSkeletonActions } from "#src/skeleton/actions.js";

import {
  classifySpatialSkeletonDisplayNodeType as getSpatialSkeletonDisplayNodeType,
  getSpatialSkeletonNodeFilterLabel,
  getSpatialSkeletonNodeIconFilterType,
  SpatialSkeletonDisplayNodeType,
  SpatialSkeletonNodeFilterType,
} from "#src/skeleton/node_types.js";
import { getEditableSpatiallyIndexedSkeletonSource } from "#src/skeleton/spatial_skeleton_manager.js";

import { StatusMessage } from "#src/status.js";

import * as matrix from "#src/util/matrix.js";

import { makeDeleteButton } from "#src/widget/delete_button.js";
import type { DependentViewContext } from "#src/widget/dependent_view_widget.js";
import { makeIcon } from "#src/widget/icon.js";


const SPATIAL_SKELETON_NODE_TYPE_ICONS: Record<
  SpatialSkeletonDisplayNodeType,
  string
> = {
  [SpatialSkeletonDisplayNodeType.ROOT]: svg_origin,
  [SpatialSkeletonDisplayNodeType.BRANCH_START]: svg_share_android,
  [SpatialSkeletonDisplayNodeType.REGULAR]: svg_minus,
  [SpatialSkeletonDisplayNodeType.VIRTUAL_END]: svg_circle,
};

function getSpatialSkeletonNodeTypeLabel(
  nodeType: SpatialSkeletonDisplayNodeType,
  nodeHasTrueEnd: boolean,
) {
  if (nodeHasTrueEnd) return "True end";
  switch (nodeType) {
    case SpatialSkeletonDisplayNodeType.ROOT:
      return "Root";
    case SpatialSkeletonDisplayNodeType.BRANCH_START:
      return "Branch point";
    case SpatialSkeletonDisplayNodeType.VIRTUAL_END:
      return "Leaf";
    default:
      return "Node";
  }
}

function formatSpatialSkeletonPosition(
  modelPosition: ArrayLike<number>,
  names?: readonly string[],
) {
  const x = Math.round(Number(modelPosition[0]));
  const y = Math.round(Number(modelPosition[1]));
  const z = Math.round(Number(modelPosition[2]));
  const n = names ?? ["x", "y", "z"];
  return {
    copyText: `${x}, ${y}, ${z}`,
    displayText: `${x} ${y} ${z}`,
    fullText: `${n[0]}: ${x} ${n[1]}: ${y} ${n[2]}: ${z}`,
    x,
    y,
    z,
  };
}

function formatSpatialSkeletonEditableNumber(
  value: number | undefined,
  fallback = "0",
) {
  return value === undefined ? fallback : `${value}`;
}

function getSpatialSkeletonSegmentChipColors(
  displayState: SegmentationDisplayState | undefined | null,
  segmentId: number,
) {
  const color = getBaseObjectColor(
    displayState,
    BigInt(segmentId),
    new Float32Array(4),
  );
  const r = Math.round(color[0] * 255);
  const g = Math.round(color[1] * 255);
  const b = Math.round(color[2] * 255);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return {
    background: getCssColor(color),
    foreground: luminance > 0.6 ? "#101010" : "#f5f5f5",
  };
}

function bindSpatialSkeletonSegmentSelection(
  element: HTMLElement,
  selectSegment: (id: bigint, pin: true | "force-unpin") => void,
  segmentId: number,
) {
  const id = BigInt(segmentId);
  const hasSegmentSelectionModifiers = (event: MouseEvent) =>
    event.ctrlKey && !event.altKey && !event.metaKey;
  element.addEventListener("mousedown", (event: MouseEvent) => {
    if (event.button !== 2 || !hasSegmentSelectionModifiers(event)) return;
    selectSegment(id, event.shiftKey ? "force-unpin" : true);
    event.preventDefault();
    event.stopPropagation();
  });
  element.addEventListener("contextmenu", (event: MouseEvent) => {
    if (!hasSegmentSelectionModifiers(event)) return;
    if (event.button !== 2) {
      selectSegment(id, event.shiftKey ? "force-unpin" : true);
    }
    event.preventDefault();
    event.stopPropagation();
  });
}

export function displaySpatialSkeletonSelection(
  layer: SegmentationUserLayer,
  state: SegmentationUserLayer["selectionState"],
  parent: HTMLElement,
  context: DependentViewContext,
): boolean {
  context.registerDisposer(
    layer.spatialSkeletonNodeDataVersion.changed.add(context.redraw),
  );
  context.registerDisposer(
    layer.selectedSpatialSkeletonNodeInfo.changed.add(context.redraw),
  );
  const nodeId = getNodeIdFromLayerSelectionState(state);
  if (nodeId === undefined) {
    return false;
  }

  const selectedSegmentId = getSegmentIdFromLayerSelectionValue(state);
  const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
  const cachedNodeInfo = layer.spatialSkeletonState.getCachedNode(nodeId);
  const completeNodeInfo = skeletonLayer?.getNode(nodeId) ?? cachedNodeInfo;
  const selectedNodeInfo = layer.selectedSpatialSkeletonNodeInfo.value;
  const previewNodeInfo =
    selectedNodeInfo !== undefined &&
    selectedNodeInfo.nodeId === nodeId &&
    selectedNodeInfo.segmentId === selectedSegmentId
      ? selectedNodeInfo
      : undefined;
  const nodeInfo = completeNodeInfo ?? previewNodeInfo;
  const container = document.createElement("div");
  container.classList.add("neuroglancer-spatial-skeleton-selection");
  parent.appendChild(container);

  const appendValue = (label: string, value: string | HTMLElement) => {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-annotation-property");
    const nameElement = document.createElement("div");
    nameElement.classList.add("neuroglancer-annotation-property-label");
    nameElement.textContent = label;
    const valueElement = document.createElement("div");
    valueElement.classList.add("neuroglancer-annotation-property-value");
    if (typeof value === "string") {
      valueElement.textContent = value;
    } else {
      valueElement.appendChild(value);
    }
    row.appendChild(nameElement);
    row.appendChild(valueElement);
    container.appendChild(row);
  };

  const appendSegmentAndNodeIds = (segmentId: number, nodeId: number) => {
    const segmentChipColors = getSpatialSkeletonSegmentChipColors(
      layer.displayState,
      segmentId,
    );
    const segmentIdChip = document.createElement("span");
    segmentIdChip.className = "neuroglancer-spatial-skeleton-node-segment-chip";
    segmentIdChip.textContent = `${segmentId}`;
    segmentIdChip.style.backgroundColor = segmentChipColors.background;
    segmentIdChip.style.color = segmentChipColors.foreground;
    segmentIdChip.title =
      `Segment ${segmentId}\n` +
      "Ctrl+right-click to pin selection\n" +
      "Ctrl+shift+right-click to unpin";
    bindSpatialSkeletonSegmentSelection(
      segmentIdChip,
      layer.selectSegment,
      segmentId,
    );
    appendValue("Segment ID", segmentIdChip);
    appendValue("Node ID", `${nodeId}`);
  };

  if (completeNodeInfo === undefined) {
    const segmentId = nodeInfo?.segmentId ?? selectedSegmentId;
    if (segmentId !== undefined) {
      appendSegmentAndNodeIds(segmentId, nodeId);
      return true;
    }
    const valueElement = document.createElement("div");
    valueElement.classList.add(
      "neuroglancer-selection-details-segment-description",
    );
    valueElement.textContent =
      "Selected node is not available in the current loaded or cached skeleton data.";
    container.appendChild(valueElement);
    return true;
  }

  const fullNodeInfo = completeNodeInfo;
  const segmentId = fullNodeInfo.segmentId;
  const nodePosition = fullNodeInfo.position;
  const segmentNodes =
    layer.spatialSkeletonState.getCachedSegmentNodes(segmentId);
  const directChildNodeIds =
    segmentNodes
      ?.filter((candidate) => candidate.parentNodeId === fullNodeInfo.nodeId)
      .map((candidate) => candidate.nodeId) ?? [];
  const nodeHasTrueEnd = fullNodeInfo.isTrueEnd ?? false;
  const nodeType = getSpatialSkeletonDisplayNodeType(
    fullNodeInfo,
    segmentNodes === undefined ? undefined : directChildNodeIds.length,
  );
  const nodeTypeLabel =
    nodeType === undefined
      ? "Unknown"
      : getSpatialSkeletonNodeTypeLabel(nodeType, nodeHasTrueEnd);
  const iconFilterType =
    nodeType === undefined
      ? undefined
      : getSpatialSkeletonNodeIconFilterType({
          nodeIsTrueEnd: nodeHasTrueEnd,
          nodeType,
        });
  const summaryRow = document.createElement("div");
  summaryRow.classList.add("neuroglancer-spatial-skeleton-selection-summary");
  container.appendChild(summaryRow);

  const editSource = getEditableSpatiallyIndexedSkeletonSource(skeletonLayer);
  const rerootDisabledReason =
    editSource?.rerootCommand === undefined
      ? "Unable to resolve a reroot-capable skeleton source for the active layer."
      : segmentNodes === undefined
        ? "Load the active skeleton in the Skeleton tab before rerooting from Selection."
        : fullNodeInfo.parentNodeId === undefined
          ? "Selected node is already root."
          : layer.getSpatialSkeletonActionsDisabledReason(
              SpatialSkeletonActions.reroot,
              {
                requireVisibleChunks: false,
              },
            );
  const rerootButton = document.createElement("button");
  rerootButton.type = "button";
  rerootButton.className = "neuroglancer-spatial-skeleton-selection-action";
  rerootButton.disabled = rerootDisabledReason !== undefined;
  rerootButton.title = rerootDisabledReason ?? "Set as root";
  rerootButton.appendChild(
    makeIcon({
      svg: svg_origin,
      title: rerootButton.title,
      clickable: false,
    }),
  );
  let rerootPending = false;
  rerootButton.addEventListener("click", () => {
    if (
      rerootButton.disabled ||
      rerootPending ||
      completeNodeInfo === undefined ||
      completeNodeInfo.parentNodeId === undefined
    ) {
      return;
    }
    rerootPending = true;
    rerootButton.disabled = true;
    void (async () => {
      try {
        await layer.rerootSpatialSkeletonNode(completeNodeInfo);
      } catch (error) {
        showSpatialSkeletonActionError("set node as root", error);
      } finally {
        rerootPending = false;
        context.redraw();
      }
    })();
  });
  const deleteDisabledReason =
    editSource === undefined
      ? "Unable to resolve editable skeleton source for the active layer."
      : segmentNodes === undefined
        ? "Load the active skeleton in the Skeleton tab before deleting from Selection."
        : fullNodeInfo.parentNodeId === undefined &&
            directChildNodeIds.length > 0
          ? "Reroot the skeleton manually before deleting the current root node."
          : layer.getSpatialSkeletonActionsDisabledReason(
              SpatialSkeletonActions.deleteNodes,
            );
  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "neuroglancer-spatial-skeleton-selection-action";
  deleteButton.disabled = deleteDisabledReason !== undefined;
  deleteButton.title = deleteDisabledReason ?? "Delete node";
  deleteButton.appendChild(
    makeDeleteButton({ title: deleteButton.title, clickable: false }),
  );
  let deletePending = false;
  deleteButton.addEventListener("click", () => {
    if (
      deleteButton.disabled ||
      editSource === undefined ||
      completeNodeInfo === undefined ||
      deletePending
    ) {
      return;
    }
    deletePending = true;
    void (async () => {
      try {
        await executeSpatialSkeletonDeleteNode(layer, completeNodeInfo);
      } catch (error) {
        showSpatialSkeletonActionError("delete node", error);
      } finally {
        deletePending = false;
      }
    })();
  });
  summaryRow.appendChild(rerootButton);
  summaryRow.appendChild(deleteButton);

  const icon = document.createElement("span");
  icon.className = "neuroglancer-spatial-skeleton-selection-summary-icon";
  const nodeTypeIconTitle =
    iconFilterType !== undefined
      ? getSpatialSkeletonNodeFilterLabel(iconFilterType)
      : nodeTypeLabel;
  icon.appendChild(
    makeIcon({
      svg:
        iconFilterType === SpatialSkeletonNodeFilterType.TRUE_END
          ? svg_flag
          : iconFilterType === SpatialSkeletonNodeFilterType.VIRTUAL_END
            ? svg_circle
            : nodeType === undefined
              ? svg_circle
              : SPATIAL_SKELETON_NODE_TYPE_ICONS[nodeType],
      title: nodeTypeIconTitle,
      clickable: false,
    }),
  );
  summaryRow.appendChild(icon);

  const skeletonDisplayTransform = skeletonLayer?.displayState.transform.value;
  let displayPosition: ArrayLike<number> = nodePosition;
  let displayNames: readonly string[] | undefined;
  if (
    skeletonDisplayTransform !== undefined &&
    skeletonDisplayTransform.error === undefined
  ) {
    const rank = skeletonDisplayTransform.rank;
    const modelPos = new Float32Array(rank);
    for (let i = 0; i < Math.min(nodePosition.length, rank); i++) {
      modelPos[i] = Number(nodePosition[i]);
    }
    const layerPos = new Float32Array(rank);
    matrix.transformPoint(
      layerPos,
      skeletonDisplayTransform.modelToRenderLayerTransform,
      rank + 1,
      modelPos,
      rank,
    );
    displayPosition = layerPos;
    displayNames = skeletonDisplayTransform.layerDimensionNames;
  }
  const position = formatSpatialSkeletonPosition(displayPosition, displayNames);
  const summaryCoordinates = document.createElement("span");
  summaryCoordinates.className =
    "neuroglancer-spatial-skeleton-selection-summary-coordinates";
  summaryCoordinates.textContent = position.displayText;
  summaryCoordinates.title = position.fullText;
  summaryRow.appendChild(summaryCoordinates);

  appendSegmentAndNodeIds(segmentId, fullNodeInfo.nodeId);
  const isLeaf = segmentNodes !== undefined && directChildNodeIds.length === 0;
  const leafTypeEditingDisabledReason = () =>
    editSource === undefined
      ? "Unable to resolve editable skeleton source for the active layer."
      : cachedNodeInfo === undefined || segmentNodes === undefined
        ? "Load the active skeleton in the Skeleton tab before changing leaf type."
        : layer.getSpatialSkeletonActionsDisabledReason(
            SpatialSkeletonActions.editNodeTrueEnd,
          );
  if (isLeaf || nodeHasTrueEnd) {
    let committedTrueEnd = nodeHasTrueEnd;
    let leafTypeSavePending = false;
    const leafTypeEditor = document.createElement("div");
    leafTypeEditor.className = "neuroglancer-spatial-skeleton-leaf-type";
    const leafTypeRadioName = `neuroglancer-spatial-skeleton-leaf-type-${segmentId}-${fullNodeInfo.nodeId}`;
    const leafTypeOptionElements: HTMLLabelElement[] = [];
    const makeLeafTypeOption = (options: {
      label: string;
      svg: string;
      trueEnd: boolean;
    }) => {
      const option = document.createElement("label");
      option.className = "neuroglancer-spatial-skeleton-leaf-type-option";
      const input = document.createElement("input");
      input.type = "radio";
      input.name = leafTypeRadioName;
      input.value = options.trueEnd ? "trueEnd" : "virtualEnd";
      input.className = "neuroglancer-spatial-skeleton-leaf-type-option-input";
      const icon = document.createElement("span");
      icon.className = "neuroglancer-spatial-skeleton-leaf-type-option-icon";
      icon.appendChild(
        makeIcon({
          svg: options.svg,
          title: options.label,
          clickable: false,
        }),
      );
      const text = document.createElement("span");
      text.className = "neuroglancer-spatial-skeleton-leaf-type-option-text";
      text.textContent = options.label;
      option.appendChild(input);
      option.appendChild(icon);
      option.appendChild(text);
      leafTypeOptionElements.push(option);
      leafTypeEditor.appendChild(option);
      return input;
    };
    const virtualEndInput = makeLeafTypeOption({
      label: "Virtual end",
      svg: svg_circle,
      trueEnd: false,
    });
    const trueEndInput = makeLeafTypeOption({
      label: "True end",
      svg: svg_flag,
      trueEnd: true,
    });
    const updateLeafTypeEditorState = () => {
      const disabledReason = leafTypeEditingDisabledReason();
      const editable = disabledReason === undefined && !leafTypeSavePending;
      virtualEndInput.checked = !committedTrueEnd;
      trueEndInput.checked = committedTrueEnd;
      for (const input of [virtualEndInput, trueEndInput]) {
        input.disabled = !editable;
        if (disabledReason !== undefined) {
          input.title = disabledReason;
        } else {
          input.removeAttribute("title");
        }
      }
      for (const option of leafTypeOptionElements) {
        option.classList.toggle(
          "neuroglancer-spatial-skeleton-leaf-type-option-disabled",
          !editable,
        );
        if (disabledReason !== undefined) {
          option.title = disabledReason;
        } else {
          option.removeAttribute("title");
        }
      }
    };
    const commitLeafType = (nextTrueEnd: boolean) => {
      if (leafTypeSavePending) return;
      const disabledReason = leafTypeEditingDisabledReason();
      if (disabledReason !== undefined) {
        StatusMessage.showTemporaryMessage(disabledReason);
        updateLeafTypeEditorState();
        return;
      }
      if (committedTrueEnd === nextTrueEnd) {
        updateLeafTypeEditorState();
        return;
      }
      const previousTrueEnd = committedTrueEnd;
      committedTrueEnd = nextTrueEnd;
      leafTypeSavePending = true;
      updateLeafTypeEditorState();
      void (async () => {
        try {
          const currentNode = layer.spatialSkeletonState.getCachedNode(
            fullNodeInfo.nodeId,
          );
          if (currentNode === undefined) {
            throw new Error(
              `Node ${fullNodeInfo.nodeId} is missing from the inspected skeleton cache.`,
            );
          }
          await executeSpatialSkeletonNodeTrueEndUpdate(layer, {
            node: currentNode,
            nextIsTrueEnd: nextTrueEnd,
          });
          committedTrueEnd = nextTrueEnd;
        } catch (error) {
          committedTrueEnd = previousTrueEnd;
          const message =
            error instanceof Error ? error.message : String(error);
          StatusMessage.showTemporaryMessage(
            `Failed to update leaf type: ${message}`,
          );
        } finally {
          leafTypeSavePending = false;
          updateLeafTypeEditorState();
        }
      })();
    };
    virtualEndInput.addEventListener("change", () => {
      if (!virtualEndInput.checked) return;
      commitLeafType(false);
    });
    trueEndInput.addEventListener("change", () => {
      if (!trueEndInput.checked) return;
      commitLeafType(true);
    });
    updateLeafTypeEditorState();
    appendValue("Node type", leafTypeEditor);
  } else {
    appendValue("Node type", nodeTypeLabel);
  }
  const confidenceConfiguration =
    editSource?.spatialSkeletonConfidenceConfiguration;
  const setPropertyInputValidity = (
    input: HTMLInputElement | HTMLSelectElement,
    valid: boolean,
    invalidTitle: string,
    disabledReason: string | undefined,
  ) => {
    input.classList.toggle(
      "neuroglancer-spatial-skeleton-properties-input-invalid",
      !valid,
    );
    if (disabledReason !== undefined) {
      input.title = disabledReason;
    } else if (!valid) {
      input.title = invalidTitle;
    } else {
      input.removeAttribute("title");
    }
  };
  const handlePropertyInputKeyDown = (event: KeyboardEvent) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    (event.currentTarget as HTMLElement | null)?.blur();
  };
  const getCachedNodeForPropertyEdit = () => {
    const currentNode = layer.spatialSkeletonState.getCachedNode(
      fullNodeInfo.nodeId,
    );
    if (currentNode === undefined) {
      throw new Error(
        `Node ${fullNodeInfo.nodeId} is missing from the inspected skeleton cache.`,
      );
    }
    return currentNode;
  };
  const radiusEditingDisabledReason = () =>
    editSource === undefined
      ? "Unable to resolve editable skeleton source for the active layer."
      : cachedNodeInfo === undefined
        ? "Load the active skeleton in the Skeleton tab before editing radius."
        : layer.getSpatialSkeletonActionsDisabledReason(
            SpatialSkeletonActions.editNodeRadius,
          );
  const confidenceEditingDisabledReason = () =>
    editSource === undefined
      ? "Unable to resolve editable skeleton source for the active layer."
      : cachedNodeInfo === undefined
        ? "Load the active skeleton in the Skeleton tab before editing confidence."
        : confidenceConfiguration === undefined
          ? "The active skeleton source does not provide confidence value configuration."
          : layer.getSpatialSkeletonActionsDisabledReason(
              SpatialSkeletonActions.editNodeConfidence,
            );

  if (radiusEditingDisabledReason() !== undefined) {
    appendValue(
      "Radius",
      formatSpatialSkeletonEditableNumber(fullNodeInfo.radius, "Unavailable"),
    );
  } else {
    let committedRadius = fullNodeInfo.radius ?? 0;
    const radiusInput = document.createElement("input");
    radiusInput.className = "neuroglancer-spatial-skeleton-properties-input";
    radiusInput.type = "number";
    radiusInput.step = "any";
    radiusInput.value = formatSpatialSkeletonEditableNumber(
      fullNodeInfo.radius,
    );
    appendValue("Radius", radiusInput);
    let radiusSavePending = false;
    const getParsedRadius = () => {
      const radius = Number(radiusInput.value);
      return {
        radius,
        radiusValid: Number.isFinite(radius),
      };
    };
    const updateRadiusEditorState = () => {
      const disabledReason = radiusEditingDisabledReason();
      const { radiusValid } = getParsedRadius();
      radiusInput.disabled = disabledReason !== undefined || radiusSavePending;
      setPropertyInputValidity(
        radiusInput,
        radiusValid,
        "Radius must be a finite number.",
        disabledReason,
      );
    };
    const resetRadiusInput = () => {
      radiusInput.value = formatSpatialSkeletonEditableNumber(committedRadius);
      updateRadiusEditorState();
    };
    const commitRadius = () => {
      if (radiusSavePending) return;
      const disabledReason = radiusEditingDisabledReason();
      if (disabledReason !== undefined) {
        StatusMessage.showTemporaryMessage(disabledReason);
        resetRadiusInput();
        return;
      }
      const { radius, radiusValid } = getParsedRadius();
      if (!radiusValid) {
        StatusMessage.showTemporaryMessage("Radius must be a finite number.");
        resetRadiusInput();
        return;
      }
      if (radius === committedRadius) {
        resetRadiusInput();
        return;
      }
      radiusSavePending = true;
      updateRadiusEditorState();
      void (async () => {
        try {
          await executeSpatialSkeletonNodeRadiusUpdate(layer, {
            node: getCachedNodeForPropertyEdit(),
            nextRadius: radius,
          });
          committedRadius = radius;
          resetRadiusInput();
        } catch (error) {
          showSpatialSkeletonActionError("update node radius", error);
          resetRadiusInput();
        } finally {
          radiusSavePending = false;
          updateRadiusEditorState();
        }
      })();
    };
    const debouncedCommitRadius = context.registerCancellable(
      debounce(commitRadius, 500),
    );
    radiusInput.addEventListener("input", updateRadiusEditorState);
    radiusInput.addEventListener("change", () => debouncedCommitRadius());
    radiusInput.addEventListener("blur", () => debouncedCommitRadius.flush());
    radiusInput.addEventListener("keydown", handlePropertyInputKeyDown);
    updateRadiusEditorState();
  }

  const confidenceConfigurationValues = confidenceConfiguration?.values;
  if (
    confidenceEditingDisabledReason() !== undefined ||
    confidenceConfigurationValues === undefined
  ) {
    appendValue(
      "Confidence level",
      formatSpatialSkeletonEditableNumber(
        fullNodeInfo.confidence,
        "Unavailable",
      ),
    );
  } else {
    let committedConfidence =
      fullNodeInfo.confidence !== undefined &&
      Number.isFinite(fullNodeInfo.confidence)
        ? Number(fullNodeInfo.confidence)
        : 0;
    const supportedConfidenceValues = Array.from(
      new Set([...confidenceConfigurationValues, committedConfidence]),
    ).filter((value): value is number => Number.isFinite(value));
    const confidenceSelectValues = Array.from(
      new Set([...supportedConfidenceValues, committedConfidence]),
    );
    const confidenceControl = document.createElement("select");
    confidenceControl.className =
      "neuroglancer-spatial-skeleton-properties-input";
    for (const value of confidenceSelectValues) {
      const option = document.createElement("option");
      option.value = value.toString();
      option.textContent = formatSpatialSkeletonEditableNumber(value);
      confidenceControl.appendChild(option);
    }
    confidenceControl.value = committedConfidence.toString();
    appendValue("Confidence level", confidenceControl);
    let confidenceSavePending = false;
    const getConfidenceValidationError = (confidence: number) => {
      if (!Number.isFinite(confidence)) {
        return "Confidence must be a finite number.";
      }
      return confidenceSelectValues.includes(confidence)
        ? undefined
        : "Confidence must use one of the supported values.";
    };
    const getParsedConfidence = () => {
      const confidence = Number(confidenceControl.value);
      const confidenceInvalidTitle = getConfidenceValidationError(confidence);
      return {
        confidence,
        confidenceValid: confidenceInvalidTitle === undefined,
        confidenceInvalidTitle,
      };
    };
    const updateConfidenceEditorState = () => {
      const confidenceDisabledReason = confidenceEditingDisabledReason();
      const { confidenceValid, confidenceInvalidTitle } = getParsedConfidence();
      confidenceControl.disabled =
        confidenceDisabledReason !== undefined || confidenceSavePending;
      setPropertyInputValidity(
        confidenceControl,
        confidenceValid,
        confidenceInvalidTitle ?? "Confidence is invalid.",
        confidenceDisabledReason,
      );
    };
    const resetConfidenceInput = () => {
      confidenceControl.value = committedConfidence.toString();
      updateConfidenceEditorState();
    };
    const commitConfidence = () => {
      if (confidenceSavePending) return;
      const disabledReason = confidenceEditingDisabledReason();
      if (disabledReason !== undefined) {
        StatusMessage.showTemporaryMessage(disabledReason);
        resetConfidenceInput();
        return;
      }
      const { confidence, confidenceValid, confidenceInvalidTitle } =
        getParsedConfidence();
      if (!confidenceValid) {
        StatusMessage.showTemporaryMessage(
          confidenceInvalidTitle ?? "Confidence is invalid.",
        );
        resetConfidenceInput();
        return;
      }
      const confidenceChanged = confidence !== committedConfidence;
      if (!confidenceChanged) {
        resetConfidenceInput();
        return;
      }
      confidenceSavePending = true;
      updateConfidenceEditorState();
      void (async () => {
        try {
          await executeSpatialSkeletonNodeConfidenceUpdate(layer, {
            node: getCachedNodeForPropertyEdit(),
            nextConfidence: confidence,
          });
          committedConfidence = confidence;
          resetConfidenceInput();
        } catch (error) {
          showSpatialSkeletonActionError("update node confidence", error);
          resetConfidenceInput();
        } finally {
          confidenceSavePending = false;
          updateConfidenceEditorState();
        }
      })();
    };
    confidenceControl.addEventListener("change", commitConfidence);
    updateConfidenceEditorState();
  }
  const descriptionText =
    cachedNodeInfo?.description ?? completeNodeInfo?.description ?? "";
  const descriptionEditingDisabledReason =
    editSource === undefined
      ? "Unable to resolve editable skeleton source for the active layer."
      : cachedNodeInfo === undefined
        ? "Load the active skeleton in the Skeleton tab before editing description."
        : layer.getSpatialSkeletonActionsDisabledReason(
            SpatialSkeletonActions.editNodeDescription,
          );
  if (descriptionEditingDisabledReason === undefined) {
    const descriptionElement = document.createElement("textarea");
    descriptionElement.classList.add(
      "neuroglancer-spatial-skeleton-selection-description",
    );
    descriptionElement.rows = 3;
    descriptionElement.placeholder = "Description";
    descriptionElement.value = descriptionText;
    descriptionElement.addEventListener("change", () => {
      if (editSource === undefined || cachedNodeInfo === undefined) {
        return;
      }
      const nextDescription = descriptionElement.value;
      if (descriptionText === nextDescription) {
        descriptionElement.value = nextDescription;
        return;
      }
      descriptionElement.disabled = true;
      void (async () => {
        try {
          const currentNode = layer.spatialSkeletonState.getCachedNode(
            fullNodeInfo.nodeId,
          );
          if (currentNode === undefined) {
            throw new Error(
              `Node ${fullNodeInfo.nodeId} is missing from the inspected skeleton cache.`,
            );
          }
          await executeSpatialSkeletonNodeDescriptionUpdate(layer, {
            node: currentNode,
            nextDescription,
          });
        } catch (error) {
          const message =
            error instanceof Error ? error.message : String(error);
          descriptionElement.value = descriptionText;
          StatusMessage.showTemporaryMessage(
            `Failed to update description: ${message}`,
          );
        } finally {
          descriptionElement.disabled = false;
        }
      })();
    });
    container.appendChild(descriptionElement);
  } else if (descriptionText.length > 0) {
    const descriptionElement = document.createElement("div");
    descriptionElement.classList.add(
      "neuroglancer-spatial-skeleton-selection-description",
    );
    descriptionElement.textContent = descriptionText;
    descriptionElement.title = descriptionEditingDisabledReason;
    container.appendChild(descriptionElement);
  } else if (completeNodeInfo === undefined) {
    appendValue("Description", "Unavailable");
  }
  return true;
}
