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

import "#src/layer/segmentation/style.css";
import "#src/layer/segmentation/spatial_skeleton.css";
import { displaySpatialSkeletonSelection } from "#src/layer/segmentation/spatial_skeleton_selection.js";

import type {
  Annotation,
  AnnotationPropertySpec,
  AnnotationReference,
} from "#src/annotation/index.js";
import {
  AnnotationType,
  LocalAnnotationSource,
} from "#src/annotation/index.js";
import type { CoordinateTransformSpecification } from "#src/coordinate_transform.js";
import { emptyValidCoordinateSpace } from "#src/coordinate_transform.js";
import type { DataSourceSpecification } from "#src/datasource/index.js";
import {
  LocalDataSource,
  localEquivalencesUrl,
} from "#src/datasource/local.js";
import { buildRoiLabelField } from "#src/datasource/zarr-vectors/label_field.js";
import type { ObjectAdmission } from "#src/datasource/zarr-vectors/object_admission.js";
import type {
  RoiBackgroundUniforms,
  RoiGroupConfig,
  RoiLabelField,
  RoiObjectAttrColumn,
} from "#src/datasource/zarr-vectors/roi.js";
import { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";
import { StreamlineFilterTab } from "#src/datasource/zarr-vectors/streamline_filter_tab.js";
import { StreamlineGuideTab } from "#src/datasource/zarr-vectors/streamline_guide_tab.js";
import { TractExportTab } from "#src/datasource/zarr-vectors/tract_export_tab.js";
import type {
  LayerActionContext,
  ManagedUserLayer,
  MouseSelectionState,
  UserLayerSelectionState,
} from "#src/layer/index.js";
import {
  LayerReference,
  LinkedLayerGroup,
  registerLayerType,
  registerLayerTypeDetector,
  registerVolumeLayerType,
  UserLayer,
} from "#src/layer/index.js";
import type { LoadedDataSubsource } from "#src/layer/layer_data_source.js";
import { layerDataSourceSpecificationFromJson } from "#src/layer/layer_data_source.js";
import * as json_keys from "#src/layer/segmentation/json_keys.js";
import { registerLayerControls } from "#src/layer/segmentation/layer_controls.js";
import {
  getNodeIdFromLayerSelectionState,
  getSegmentIdFromLayerSelectionValue,
  SpatialSkeletonHoverState,
} from "#src/layer/segmentation/selection.js";
import { executeSpatialSkeletonReroot } from "#src/layer/segmentation/spatial_skeleton_commands.js";
import {
  MeshLayer,
  MeshSource,
  MultiscaleMeshLayer,
  MultiscaleMeshSource,
} from "#src/mesh/frontend.js";
import { getRenderLayerTransform } from "#src/render_coordinate_transform.js";
import {
  RenderScaleHistogram,
  numRenderScaleHistogramBins,
  renderScaleHistogramBinSize,
  renderScaleHistogramOrigin,
  trackableRenderScaleTarget,
} from "#src/render_scale_statistics.js";
import { RenderLayerRole } from "#src/renderlayer.js";
import { getCssColor, SegmentColorHash } from "#src/segment_color.js";
import {
  addSegmentToVisibleSets,
  getVisibleSegments,
} from "#src/segmentation_display_state/base.js";
import type {
  SegmentationColorGroupState,
  SegmentationDisplayState,
  SegmentationGroupState,
} from "#src/segmentation_display_state/frontend.js";
import {
  augmentSegmentId,
  bindSegmentListWidth,
  getBaseObjectColor,
  makeSegmentWidget,
  maybeAugmentSegmentId,
  registerCallbackWhenSegmentationDisplayStateChanged,
  SegmentSelectionState,
  Uint64MapEntry,
} from "#src/segmentation_display_state/frontend.js";
import type {
  PreprocessedSegmentPropertyMap,
  SegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { getPreprocessedSegmentPropertyMap } from "#src/segmentation_display_state/property_map.js";
import { LocalSegmentationGraphSource } from "#src/segmentation_graph/local.js";
import { VisibleSegmentEquivalencePolicy } from "#src/segmentation_graph/segment_id.js";
import type {
  SegmentationGraphSource,
  SegmentationGraphSourceConnection,
} from "#src/segmentation_graph/source.js";
import { SegmentationGraphSourceTab } from "#src/segmentation_graph/source.js";
import { SharedDisjointUint64Sets } from "#src/shared_disjoint_sets.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import {
  DEFAULT_SPATIAL_SKELETON_EDIT_ACTIONS,
  getSpatialSkeletonActionSupportLabel,
  SpatialSkeletonActions,
  type SpatialSkeletonAction,
} from "#src/skeleton/actions.js";
import type {
  SpatiallyIndexedSkeletonNode,
  SpatialSkeletonSourceState,
} from "#src/skeleton/api.js";
import type { VertexAttrStats } from "#src/skeleton/spatial_base.js";
import { resolveSkeletonDefaultShader } from "#src/skeleton/default_shader.js";
import {
  PerspectiveViewSkeletonLayer,
  SkeletonLayer,
  SkeletonRenderingOptions,
  type SkeletonSource,
  DEFAULT_FRAGMENT_MAIN,
  SliceViewPanelSkeletonLayer,
} from "#src/skeleton/frontend.js";
import {
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
  SpatiallyIndexedSkeletonLayer,
  SpatiallyIndexedSkeletonSource,
  MultiscaleSpatiallyIndexedSkeletonSource,
  computeDiagonalModelToGlobalMetersScale,
} from "#src/skeleton/spatial_frontend.js";
import {
  findSpatiallyIndexedSkeletonNode,
  getSpatiallyIndexedSkeletonDirectChildren,
  getSpatiallyIndexedSkeletonNodeParent,
} from "#src/skeleton/node_traversal.js";
import { SpatialSkeletonNodeFilterType } from "#src/skeleton/node_types.js";
import {
  buildSpatialSkeletonGridLevels,
  getSpatialSkeletonGridSpacing,
  selectSpatialSkeletonGridLevelByBudget,
  SpatialSkeletonDetailFocus,
  type SpatialSkeletonGridLevel,
  type SpatialSkeletonGridSize,
} from "#src/skeleton/spatial_chunk_sizing.js";
import {
  editableSpatiallyIndexedSkeletonSourceSupportsAction,
  getEditableSpatiallyIndexedSkeletonSource,
  getSpatiallyIndexedSkeletonSource,
  isSpatiallyIndexedSkeletonSourceReadOnly,
  SpatialSkeletonState,
} from "#src/skeleton/spatial_skeleton_manager.js";
import { DataType, VolumeType } from "#src/sliceview/volume/base.js";
import { MultiscaleVolumeChunkSource } from "#src/sliceview/volume/frontend.js";
import { SegmentationRenderLayer } from "#src/sliceview/volume/segmentation_renderlayer.js";
import { StatusMessage } from "#src/status.js";
import { trackableAlphaValue } from "#src/trackable_alpha.js";
import { TrackableBoolean } from "#src/trackable_boolean.js";
import { trackableFiniteFloat } from "#src/trackable_finite_float.js";
import type {
  TrackableValueInterface,
  WatchableValueInterface,
} from "#src/trackable_value.js";
import {
  IndirectTrackableValue,
  IndirectWatchableValue,
  makeCachedDerivedWatchableValue,
  makeCachedLazyDerivedWatchableValue,
  observeWatchable,
  registerNestedSync,
  TrackableValue,
  WatchableValue,
} from "#src/trackable_value.js";
import { UserLayerWithAnnotationsMixin } from "#src/ui/annotations.js";
import { SegmentDisplayTab } from "#src/ui/segment_list.js";
import { registerSegmentSelectTools } from "#src/ui/segment_select_tools.js";
import { registerSegmentSplitMergeTools } from "#src/ui/segment_split_merge_tools.js";
import { DisplayOptionsTab } from "#src/ui/segmentation_display_options_tab.js";
import { SpatialSkeletonEditTab } from "#src/ui/spatial_skeleton_edit_tab.js";
import { registerSpatialSkeletonEditModeTool } from "#src/ui/spatial_skeleton_edit_tool.js";
import { Uint64Map } from "#src/uint64_map.js";
import { Uint64OrderedSet } from "#src/uint64_ordered_set.js";
import { Uint64Set } from "#src/uint64_set.js";
import { gatherUpdate } from "#src/util/array.js";
import {
  packColor,
  parseRGBColorSpecification,
  serializeColor,
  TrackableOptionalRGB,
  unpackRGB,
} from "#src/util/color.js";
import type { Borrowed, Owned } from "#src/util/disposable.js";
import { RefCounted } from "#src/util/disposable.js";
import { vec3, vec4 } from "#src/util/geom.js";
import {
  parseArray,
  parseUint64,
  verifyFiniteNonNegativeFloat,
  verifyFinitePositiveFloat,
  verifyNonnegativeInt,
  verifyObjectAsMap,
  verifyOptionalObjectProperty,
  verifyString,
} from "#src/util/json.js";
import * as matrix from "#src/util/matrix.js";
import { Signal } from "#src/util/signal.js";
import { TrackableEnum } from "#src/util/trackable_enum.js";
import { makeWatchableShaderError } from "#src/webgl/dynamic_shader.js";

import type { DependentViewContext } from "#src/widget/dependent_view_widget.js";

import { registerLayerShaderControlsTool } from "#src/widget/shader_controls.js";

const MAX_LAYER_BAR_UI_INDICATOR_COLORS = 6;

export class SegmentationUserLayerGroupState
  extends RefCounted
  implements SegmentationGroupState
{
  specificationChanged = new Signal();
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { specificationChanged } = this;
    this.hideSegmentZero.changed.add(specificationChanged.dispatch);
    this.segmentQuery.changed.add(specificationChanged.dispatch);

    const { selectedSegments } = this;
    const visibleSegments = (this.visibleSegments = this.registerDisposer(
      Uint64Set.makeWithCounterpart(layer.manager.rpc),
    ));
    this.segmentEquivalences = this.registerDisposer(
      SharedDisjointUint64Sets.makeWithCounterpart(
        layer.manager.rpc,
        layer.registerDisposer(
          makeCachedDerivedWatchableValue(
            (x) =>
              x?.visibleSegmentEquivalencePolicy ||
              VisibleSegmentEquivalencePolicy.MIN_REPRESENTATIVE,
            [this.graph],
          ),
        ),
      ),
    );

    this.temporaryVisibleSegments = layer.registerDisposer(
      Uint64Set.makeWithCounterpart(layer.manager.rpc),
    );
    this.temporarySegmentEquivalences = layer.registerDisposer(
      SharedDisjointUint64Sets.makeWithCounterpart(
        layer.manager.rpc,
        this.segmentEquivalences.disjointSets.visibleSegmentEquivalencePolicy,
      ),
    );
    this.useTemporaryVisibleSegments = layer.registerDisposer(
      SharedWatchableValue.make(layer.manager.rpc, false),
    );
    this.useTemporarySegmentEquivalences = layer.registerDisposer(
      SharedWatchableValue.make(layer.manager.rpc, false),
    );

    visibleSegments.changed.add(specificationChanged.dispatch);
    selectedSegments.changed.add(specificationChanged.dispatch);
    selectedSegments.changed.add((x, add) => {
      if (!add) {
        if (x) {
          visibleSegments.delete(x);
        } else {
          visibleSegments.clear();
        }
      }
    });
    visibleSegments.changed.add((x, add) => {
      if (add) {
        if (x) {
          selectedSegments.add(x);
        }
      }
    });
  }

  restoreState(specification: unknown) {
    verifyOptionalObjectProperty(
      specification,
      json_keys.HIDE_SEGMENT_ZERO_JSON_KEY,
      (value) => this.hideSegmentZero.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.EQUIVALENCES_JSON_KEY,
      (value) => {
        this.localGraph.restoreState(value);
      },
    );

    verifyOptionalObjectProperty(
      specification,
      json_keys.SEGMENTS_JSON_KEY,
      (segmentsValue) => {
        const { segmentEquivalences, selectedSegments, visibleSegments } = this;
        parseArray(segmentsValue, (value) => {
          let stringValue = String(value);
          const hidden = stringValue.startsWith("!");
          if (hidden) {
            stringValue = stringValue.substring(1);
          }
          const id = parseUint64(stringValue);
          const segmentId = segmentEquivalences.get(id);
          selectedSegments.add(segmentId);
          if (!hidden) {
            visibleSegments.add(segmentId);
          }
        });
      },
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.SEGMENT_QUERY_JSON_KEY,
      (value) => this.segmentQuery.restoreState(value),
    );
  }

  toJSON() {
    const x: any = {};
    x[json_keys.HIDE_SEGMENT_ZERO_JSON_KEY] = this.hideSegmentZero.toJSON();
    const { selectedSegments, visibleSegments } = this;
    if (selectedSegments.size > 0) {
      x[json_keys.SEGMENTS_JSON_KEY] = [...selectedSegments].map((segment) => {
        if (visibleSegments.has(segment)) {
          return segment.toString();
        }
        return "!" + segment.toString();
      });
    } else {
      x[json_keys.SEGMENTS_JSON_KEY] = [];
    }
    const { segmentEquivalences } = this;
    if (this.localSegmentEquivalences && segmentEquivalences.size > 0) {
      x[json_keys.EQUIVALENCES_JSON_KEY] = segmentEquivalences.toJSON();
    }
    x[json_keys.SEGMENT_QUERY_JSON_KEY] = this.segmentQuery.toJSON();
    return x;
  }

  assignFrom(other: SegmentationUserLayerGroupState) {
    this.maxIdLength.value = other.maxIdLength.value;
    this.hideSegmentZero.value = other.hideSegmentZero.value;
    this.selectedSegments.assignFrom(other.selectedSegments);
    this.visibleSegments.assignFrom(other.visibleSegments);
    this.segmentEquivalences.assignFrom(other.segmentEquivalences);
  }

  localGraph = new LocalSegmentationGraphSource();
  visibleSegments: Uint64Set;
  selectedSegments = this.registerDisposer(new Uint64OrderedSet());

  segmentPropertyMap = new WatchableValue<
    PreprocessedSegmentPropertyMap | undefined
  >(undefined);
  graph = new WatchableValue<SegmentationGraphSource | undefined>(undefined);
  segmentEquivalences: SharedDisjointUint64Sets;
  localSegmentEquivalences = false;
  maxIdLength = new WatchableValue(1);
  hideSegmentZero = new TrackableBoolean(true, true);
  segmentQuery = new TrackableValue<string>("", verifyString);

  temporaryVisibleSegments: Uint64Set;
  temporarySegmentEquivalences: SharedDisjointUint64Sets;
  useTemporaryVisibleSegments: SharedWatchableValue<boolean>;
  useTemporarySegmentEquivalences: SharedWatchableValue<boolean>;
}

export class SegmentationUserLayerColorGroupState
  extends RefCounted
  implements SegmentationColorGroupState
{
  specificationChanged = new Signal();
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { specificationChanged } = this;
    this.segmentColorHash.changed.add(specificationChanged.dispatch);
    this.segmentStatedColors.changed.add(specificationChanged.dispatch);
    this.tempSegmentStatedColors2d.changed.add(specificationChanged.dispatch);
    this.segmentDefaultColor.changed.add(specificationChanged.dispatch);
    this.tempSegmentDefaultColor2d.changed.add(specificationChanged.dispatch);
    this.highlightColor.changed.add(specificationChanged.dispatch);
  }

  restoreState(specification: unknown) {
    verifyOptionalObjectProperty(
      specification,
      json_keys.COLOR_SEED_JSON_KEY,
      (value) => this.segmentColorHash.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.SEGMENT_DEFAULT_COLOR_JSON_KEY,
      (value) => this.segmentDefaultColor.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.SEGMENT_STATED_COLORS_JSON_KEY,
      (y) => {
        const result = verifyObjectAsMap(y, (x) =>
          parseRGBColorSpecification(String(x)),
        );
        for (const [idStr, colorVec] of result) {
          const id = parseUint64(idStr);
          const color = BigInt(packColor(colorVec));
          this.segmentStatedColors.set(id, color);
        }
      },
    );
  }

  toJSON() {
    const x: any = {};
    x[json_keys.COLOR_SEED_JSON_KEY] = this.segmentColorHash.toJSON();
    x[json_keys.SEGMENT_DEFAULT_COLOR_JSON_KEY] =
      this.segmentDefaultColor.toJSON();
    const { segmentStatedColors } = this;
    if (segmentStatedColors.size > 0) {
      const j: any = (x[json_keys.SEGMENT_STATED_COLORS_JSON_KEY] = {});
      for (const [key, value] of segmentStatedColors) {
        j[key.toString()] = serializeColor(unpackRGB(Number(value)));
      }
    }
    return x;
  }

  assignFrom(other: SegmentationUserLayerColorGroupState) {
    this.segmentColorHash.value = other.segmentColorHash.value;
    this.segmentStatedColors.assignFrom(other.segmentStatedColors);
    this.tempSegmentStatedColors2d.assignFrom(other.tempSegmentStatedColors2d);
    this.segmentDefaultColor.value = other.segmentDefaultColor.value;
    this.highlightColor.value = other.highlightColor.value;
  }

  segmentColorHash = SegmentColorHash.getDefault();
  segmentStatedColors = this.registerDisposer(new Uint64Map());
  tempSegmentStatedColors2d = this.registerDisposer(new Uint64Map());
  segmentDefaultColor = new TrackableOptionalRGB();
  tempSegmentDefaultColor2d = new WatchableValue<vec3 | vec4 | undefined>(
    undefined,
  );
  highlightColor = new WatchableValue<vec4 | undefined>(undefined);
}

class LinkedSegmentationGroupState<
    State extends
      | SegmentationUserLayerGroupState
      | SegmentationUserLayerColorGroupState,
  >
  extends RefCounted
  implements WatchableValueInterface<State>
{
  private curRoot: SegmentationUserLayer | undefined;
  private curGroupState: Owned<State> | undefined;
  get changed() {
    return this.linkedGroup.root.changed;
  }
  get value() {
    const root = this.linkedGroup.root.value as SegmentationUserLayer;
    if (root !== this.curRoot) {
      this.curRoot = root;
      const groupState = root.displayState[this.propertyName] as State;
      if (root === this.linkedGroup.layer) {
        const { curGroupState } = this;
        if (curGroupState !== undefined) {
          groupState.assignFrom(curGroupState as any);
          curGroupState.dispose();
        }
      }
      this.curGroupState = groupState.addRef() as State;
    }
    return this.curGroupState!;
  }
  disposed() {
    this.curGroupState?.dispose();
  }
  constructor(
    public linkedGroup: LinkedLayerGroup,
    private propertyName: State extends SegmentationUserLayerGroupState
      ? "originalSegmentationGroupState"
      : "originalSegmentationColorGroupState",
  ) {
    super();
    this.value;
  }
}

function findClosestSpatialSkeletonGridLevelBySpacing(
  levels: SpatialSkeletonGridLevel[],
  spacing: number,
): number {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < levels.length; ++i) {
    const gridSpacing = getSpatialSkeletonGridSpacing(levels[i].size);
    const distance = Math.abs(gridSpacing - spacing);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function getSpatialSkeletonGridHistogramConfig(
  levels: SpatialSkeletonGridLevel[],
) {
  if (levels.length === 0) {
    return {
      origin: renderScaleHistogramOrigin,
      binSize: renderScaleHistogramBinSize,
    };
  }
  const logSpacings: number[] = [];
  let minLogSpacing = Number.POSITIVE_INFINITY;
  let maxLogSpacing = Number.NEGATIVE_INFINITY;
  for (const level of levels) {
    const spacing = Math.max(getSpatialSkeletonGridSpacing(level.size), 1e-6);
    const logSpacing = Math.log2(spacing);
    logSpacings.push(logSpacing);
    minLogSpacing = Math.min(minLogSpacing, logSpacing);
    maxLogSpacing = Math.max(maxLogSpacing, logSpacing);
  }
  if (!Number.isFinite(minLogSpacing) || !Number.isFinite(maxLogSpacing)) {
    return {
      origin: renderScaleHistogramOrigin,
      binSize: renderScaleHistogramBinSize,
    };
  }
  logSpacings.sort((a, b) => a - b);
  let minDelta = Number.POSITIVE_INFINITY;
  for (let i = 1; i < logSpacings.length; ++i) {
    const delta = logSpacings[i] - logSpacings[i - 1];
    if (delta > 0) minDelta = Math.min(minDelta, delta);
  }
  const span = maxLogSpacing - minLogSpacing;
  // Choose a bin size that spreads the levels across (most of) the widget
  // width.  Reserve a few bins of padding on each side so the extreme
  // levels aren't flush against the edges.  A single level (span 0) has no
  // meaningful spread — fall back to the default bin size.
  const coverageBinSize =
    span > 0
      ? span / Math.max(numRenderScaleHistogramBins - 4, 1)
      : renderScaleHistogramBinSize;
  // Never use a bin so large that two adjacent levels (minDelta apart in
  // log space) collapse into the same bin — that would merge distinct
  // scales into one bar.  When the coverage bin size already keeps them
  // distinct (the common case: few, well-separated pyramid levels), the
  // coverage value wins and the bars fan out across the full axis instead
  // of bunching into a narrow cluster in the middle.
  const maxBinSizeForDistinctBars = Number.isFinite(minDelta)
    ? minDelta * 0.9
    : Number.POSITIVE_INFINITY;
  let binSize = Math.max(
    0.05,
    Math.min(coverageBinSize, maxBinSizeForDistinctBars),
  );
  if (!Number.isFinite(binSize) || binSize <= 0) {
    binSize = renderScaleHistogramBinSize;
  }

  const range = numRenderScaleHistogramBins * binSize;
  const desiredPadding = binSize * 2;
  const minOrigin = maxLogSpacing + desiredPadding - range;
  const maxOrigin = minLogSpacing - desiredPadding;
  const centeredOrigin = (minLogSpacing + maxLogSpacing - range) / 2;
  const clampedOrigin = Math.min(
    Math.max(centeredOrigin, minOrigin),
    maxOrigin,
  );
  const roundedBinSize = Math.max(binSize, 1e-3);
  const roundedOrigin =
    Math.round(clampedOrigin / roundedBinSize) * roundedBinSize;
  return { origin: roundedOrigin, binSize: roundedBinSize };
}

/**
 * Flatten the persisted ROI groups into the plain, structured-clone-safe form
 * the worker consumes: each group's ROI list, its colour packed to an int, and
 * its visibility. The worker unions the visible groups' passing tracts (ghost
 * shader) and attributes each passing tract the colour of its group.
 */
function buildRoiGroupConfigs(roiFilter: RoiFilterState): RoiGroupConfig[] {
  // Include the live label-selection preview (if any) so a staged, not-yet-
  // committed selection ghosts/colours streamlines exactly like a real group.
  return roiFilter.groupsForWorker().map((g) => ({
    rois: g.rois,
    // Pack RGBA: rgb = group colour, a = group opacity. The colour-by-group RGB
    // override and the per-group "on" opacity both ride this single value.
    colorPacked: packColor(
      vec4.fromValues(g.color[0], g.color[1], g.color[2], g.opacity),
    ),
    visible: g.visible,
    // Per-group unified colour-by + attribute predicates (both settable like
    // opacity). The predicates are what let a group select by data rather than
    // by geometry, which for a point cloud is the only kind of group there is.
    colorBy: g.colorBy,
    ...(g.attrFilters.length !== 0 ? { attrFilters: g.attrFilters } : {}),
  }));
}

/**
 * Snapshot the loaded per-object numeric attributes as worker-shippable columns,
 * keyed by attribute name. Values are copied to a `Float32Array`; `ids` come
 * straight from the shared (read-only) inline id array.
 *
 * ID-space caveat: these are the segment-property map's ids. For a store with
 * `object_index_convention: "identity"` they equal the streamline segment ids
 * the passing set uses; a `"standard"` store would need re-keying through
 * `object_attributes/segment_id` first (a known follow-up — verify per dataset).
 */
function buildObjectAttrColumns(
  map: PreprocessedSegmentPropertyMap | undefined,
): Map<string, RoiObjectAttrColumn> {
  const columns = new Map<string, RoiObjectAttrColumn>();
  const inline = map?.segmentPropertyMap.inlineProperties;
  if (map === undefined || inline === undefined) return columns;
  const ids = inline.ids;
  for (const p of map.numericalProperties) {
    columns.set(p.id, {
      ids,
      values: Float32Array.from(p.values as ArrayLike<number>),
      min: Number(p.bounds[0]),
      max: Number(p.bounds[1]),
    });
  }
  return columns;
}

// The ROI overlay annotation shader: colour each region by its per-annotation
// `color` property (set to its group's colour). Box/plane ROIs render as a
// coloured wireframe; sphere ROIs as a translucent fill.
const ROI_OVERLAY_SHADER = "void main() {\n  setColor(prop_color());\n}\n";
// Same, but discard in the 2-d slice views (hide-overlays-in-2d toggle).
const ROI_OVERLAY_SHADER_HIDE_2D =
  "void main() {\n  if (!PROJECTION_VIEW) { discard; }\n  setColor(prop_color());\n}\n";

/**
 * Mirror the ROI groups into a local annotation source so the regions draw as
 * overlays, each in its group's colour (via the source's `color` property).
 * One-way — the `RoiFilterState` is the truth; this only reflects it.
 *
 * Updates annotations in place when the ROI count is unchanged (so a slider
 * drag moves a region without a delete/re-add flicker), and rebuilds from
 * scratch on a structural change. `refs` is the running annotation list.
 */
function rebuildRoiAnnotations(
  source: LocalAnnotationSource,
  roiFilter: RoiFilterState,
  refs: AnnotationReference[],
): void {
  const desired: Annotation[] = [];
  for (const group of roiFilter.groups) {
    const color = packColor(group.color);
    for (const roi of group.rois) {
      const shape = roi.shape;
      if (shape.kind === "box") {
        desired.push({
          type: AnnotationType.AXIS_ALIGNED_BOUNDING_BOX,
          id: "",
          pointA: Float32Array.from(shape.lower),
          pointB: Float32Array.from(shape.upper),
          properties: [color],
        });
      } else if (shape.kind === "ellipsoid") {
        desired.push({
          type: AnnotationType.ELLIPSOID,
          id: "",
          center: Float32Array.from(shape.center),
          radii: Float32Array.from(shape.radii),
          properties: [color],
        });
      }
      // halfspace ROIs are not drawn (axis-aligned regions only).
    }
  }
  if (refs.length === desired.length) {
    for (let i = 0; i < desired.length; ++i) {
      desired[i].id = refs[i].id;
      source.update(refs[i], desired[i]);
      source.commit(refs[i]);
    }
  } else {
    for (const ref of refs) source.delete(ref);
    refs.length = 0;
    for (const annotation of desired) refs.push(source.add(annotation));
  }
}

/**
 * Nothing to the object-keyed full-detail pass by default.
 *
 * It was an even-ish split while an ROI group could ask for its tracts at full
 * resolution. No group can now -- a dissection is a selection within the level
 * the layer draws -- so any reservation here is withheld from the only pass
 * that draws, which is a direct cut to how much of the tractogram fits.
 * Non-zero only if something is deliberately driving the pass-2 layer.
 */

let warnedAdmissionUnavailable = false;
function warnOnceAdmissionUnavailable(hasSource: boolean) {
  if (warnedAdmissionUnavailable) return;
  warnedAdmissionUnavailable = true;
  console.warn(
    "Object detail focus is selected but this store cannot be budgeted per " +
      (hasSource
        ? "object: its levels are not nested subsets of one object id space, " +
          "or it omits object_attributes/vertex_count at some level."
        : "object: no spatially-indexed tract source reported one.") +
      " Object focus stays in force -- one level everywhere, its whole volume" +
      " resident -- but the level is chosen by the whole-level memory ceiling" +
      " rather than by which objects fit, and the levels are not drawn" +
      " together.",
  );
}

class SegmentationUserLayerDisplayState implements SegmentationDisplayState {
  constructor(public layer: SegmentationUserLayer) {
    // Even though `SegmentationUserLayer` assigns this to its `displayState` property, redundantly
    // assign it here first in order to allow it to be accessed by `segmentationGroupState`.
    layer.displayState = this;

    this.linkedSegmentationGroup = layer.registerDisposer(
      new LinkedLayerGroup(
        layer.manager.rootLayers,
        layer,
        (userLayer) => userLayer instanceof SegmentationUserLayer,
        (userLayer) => {
          if (!(userLayer instanceof SegmentationUserLayer)) {
            throw new Error(
              "Expected a segmentation layer for the linked segmentation group.",
            );
          }
          return userLayer.displayState.linkedSegmentationGroup;
        },
      ),
    );

    this.linkedSegmentationColorGroup = this.layer.registerDisposer(
      new LinkedLayerGroup(
        layer.manager.rootLayers,
        layer,
        (userLayer) => userLayer instanceof SegmentationUserLayer,
        (userLayer) => {
          if (!(userLayer instanceof SegmentationUserLayer)) {
            throw new Error(
              "Expected a segmentation layer for the linked segmentation color group.",
            );
          }
          return userLayer.displayState.linkedSegmentationColorGroup;
        },
      ),
    );

    // A segmentation layer other than this one whose labels dissect the tracts.
    this.roiLabelLayer = layer.registerDisposer(
      new LayerReference(layer.manager.rootLayers.addRef(), (managed) => {
        const userLayer = managed.layer;
        return (
          userLayer === null ||
          (userLayer instanceof SegmentationUserLayer && userLayer !== layer)
        );
      }),
    );

    this.originalSegmentationGroupState = layer.registerDisposer(
      new SegmentationUserLayerGroupState(layer),
    );

    this.originalSegmentationColorGroupState = layer.registerDisposer(
      new SegmentationUserLayerColorGroupState(layer),
    );

    this.transparentPickEnabled = layer.pick;

    this.useTempSegmentStatedColors2d = layer.registerDisposer(
      SharedWatchableValue.make(layer.manager.rpc, false),
    );

    this.segmentationGroupState = this.layer.registerDisposer(
      new LinkedSegmentationGroupState<SegmentationUserLayerGroupState>(
        this.linkedSegmentationGroup,
        "originalSegmentationGroupState",
      ),
    );
    this.segmentationColorGroupState = this.layer.registerDisposer(
      new LinkedSegmentationGroupState<SegmentationUserLayerColorGroupState>(
        this.linkedSegmentationColorGroup,
        "originalSegmentationColorGroupState",
      ),
    );

    this.selectSegment = layer.selectSegment;
    this.filterBySegmentLabel = layer.filterBySegmentLabel;

    this.hideSegmentZero = this.layer.registerDisposer(
      new IndirectWatchableValue(
        this.segmentationGroupState,
        (group) => group.hideSegmentZero,
      ),
    );
    this.segmentColorHash = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.segmentColorHash,
      ),
    );
    this.segmentStatedColors = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.segmentStatedColors,
      ),
    );
    this.tempSegmentStatedColors2d = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.tempSegmentStatedColors2d,
      ),
    );
    this.segmentDefaultColor = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.segmentDefaultColor,
      ),
    );
    this.tempSegmentDefaultColor2d = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.tempSegmentDefaultColor2d,
      ),
    );
    this.highlightColor = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.highlightColor,
      ),
    );
    this.segmentQuery = this.layer.registerDisposer(
      new IndirectWatchableValue(
        this.segmentationGroupState,
        (group) => group.segmentQuery,
      ),
    );
    this.segmentPropertyMap = this.layer.registerDisposer(
      new IndirectWatchableValue(
        this.segmentationGroupState,
        (group) => group.segmentPropertyMap,
      ),
    );

    // LUT colors: apply the colors declared by an `rgb` segment property map by
    // overlaying the user's stated colors on top of the map's colors into a
    // derived, non-serialized map. Both the CPU color lookup
    // (`getBaseObjectColor`) and the GPU segmentation renderer read
    // `effectiveSegmentStatedColors`. When the map declares no colors it falls
    // through to the user's `segmentStatedColors` unchanged, so datasets without
    // a color LUT behave exactly as before.
    const self = this;
    this.derivedSegmentStatedColors = this.layer.registerDisposer(
      new Uint64Map(),
    );
    this.effectiveSegmentStatedColors = {
      changed: this.effectiveSegmentStatedColorsChanged,
      get value() {
        return self.effectiveStatedColorsUsesLut
          ? self.derivedSegmentStatedColors
          : self.segmentStatedColors.value;
      },
    };
    const recomputeEffectiveStatedColors = () => {
      const propertyMap = this.segmentPropertyMap.value;
      const colorProperty = propertyMap?.colors;
      if (colorProperty === undefined) {
        // No LUT colors: `effectiveSegmentStatedColors` mirrors the user map.
        this.effectiveStatedColorsUsesLut = false;
      } else {
        const derived = this.derivedSegmentStatedColors;
        derived.clear();
        const { ids } = propertyMap!.segmentPropertyMap.inlineProperties!;
        const { values } = colorProperty;
        for (let i = 0, n = ids.length; i < n; ++i) {
          const color = values[i];
          if (color >= 0) derived.set(ids[i], BigInt(color));
        }
        // A user-set stated color for an id overrides the LUT color.
        for (const [id, color] of this.segmentStatedColors.value) {
          derived.set(id, color);
        }
        this.effectiveStatedColorsUsesLut = true;
      }
      this.effectiveSegmentStatedColorsChanged.dispatch();
    };
    this.layer.registerDisposer(
      this.segmentPropertyMap.changed.add(recomputeEffectiveStatedColors),
    );
    this.layer.registerDisposer(
      this.segmentStatedColors.changed.add(recomputeEffectiveStatedColors),
    );
    recomputeEffectiveStatedColors();

    this.spatialSkeletonGridResolutionTarget2d.changed.add(() =>
      this.applySpatialSkeletonResolutionTarget(
        "2d",
        this.spatialSkeletonGridResolutionTarget2d.value,
      ),
    );
    this.spatialSkeletonGridResolutionTarget3d.changed.add(() =>
      this.applySpatialSkeletonResolutionTarget(
        "3d",
        this.spatialSkeletonGridResolutionTarget3d.value,
      ),
    );
  }

  /**
   * Move a view's grid level to the one closest to `target`, but no finer than
   * the memory budget allows.
   *
   * The budget is a CEILING, not a choice. It used to be a veto: while a
   * budget-driven level was in force these listeners stood down entirely, so
   * the resolution sliders moved, updated the URL, and changed nothing —
   * silently. Clamping instead keeps the whole-brain out-of-memory case fixed
   * (nothing can select a level that does not fit) while letting the slider do
   * what it appears to do, in both directions, at any zoom.
   *
   * `levels` is COARSEST-first (`getSpatialSkeletonGridSizes` ends in
   * `finestFirst.reverse()`), so a larger index is FINER, and
   * `spatialSkeletonBudgetLevel` -- the largest index whose cost fits the GPU
   * budget -- is an upper bound. Clamping is therefore a MINIMUM: coarser is
   * always available, finer only up to what fits.
   */
  private applySpatialSkeletonResolutionTarget(
    view: "2d" | "3d",
    target: number,
  ) {
    const levels = this.spatialSkeletonGridLevels.value;
    if (levels.length === 0) return;
    // Under OBJECT focus the level is a consequence of what the memory budget
    // admits, not of the camera; letting a resolution target write it too would
    // simply overwrite that choice on the next frame. See
    // {@link refreshSpatialSkeletonAdmission}.
    if (
      this.spatialSkeletonDetailFocus.value ===
        SpatialSkeletonDetailFocus.OBJECT &&
      this.spatialSkeletonComputeAdmission !== undefined
    ) {
      return;
    }
    const requested = findClosestSpatialSkeletonGridLevelBySpacing(
      levels,
      target,
    );
    // The whole-level ceiling is skipped once a per-cell budget is available.
    //
    // The two are rival answers to the same question, and the coarser one was
    // winning: the target `requested` comes from `targetSpacingForCellBudget`,
    // which divides the memory limit by the cells actually in view, while the
    // ceiling asks whether the level would fit if EVERY cell were resident.
    // Zooming in makes the first finer and leaves the second untouched, so
    // `Math.min` pinned the level at the whole-level answer no matter how far
    // the user zoomed -- the level simply never moved.
    //
    // Whole-volume residency would change that -- "if every cell were resident"
    // stops being hypothetical -- but that is requested only under the object
    // PARTITION, whose level is chosen by `refreshSpatialSkeletonAdmission` and
    // never reaches here (the early return above). Object focus without the
    // partition requests the frustum, like LOCAL, so it budgets like LOCAL.
    const ceiling =
      this.spatialSkeletonPerCellCostBytes.value.length > 0
        ? undefined
        : this.spatialSkeletonBudgetLevel;
    this.setSpatialSkeletonGridLevel(
      view,
      ceiling === undefined ? requested : Math.min(requested, ceiling),
    );
  }

  segmentSelectionState = new SegmentSelectionState();
  selectedAlpha = trackableAlphaValue(0.5);
  saturation = trackableAlphaValue(1.0);
  notSelectedAlpha = trackableAlphaValue(0);
  hoverHighlight = new TrackableBoolean(true, true);
  silhouetteRendering = new TrackableValue<number>(
    0,
    verifyFiniteNonNegativeFloat,
    0,
  );
  objectAlpha = trackableAlphaValue(1.0);
  hiddenObjectAlpha = trackableAlphaValue(0.5);
  skeletonLod = trackableFiniteFloat(0.0);
  /** TrackVis-style ROI streamline filter (zarr-vectors); persists to the URL. */
  roiFilter = new RoiFilterState();
  /**
   * ROI-filter data channel, created lazily by {@link ensureRoiFilterChannel}
   * only when a zarr-vectors spatially-indexed skeleton (tract) source loads;
   * left undefined otherwise, so the skeleton shader's ROI tier stays inert for
   * every other segmentation layer. Mirror the persisted {@link roiFilter}
   * (URL truth) into non-persisted watchables the render layer / backend read:
   * `roiGroups`/`roiFilterActive` feed the worker's passing-set recompute,
   * `roiFilterActive`/`roiGhostAlpha` drive the shader uniforms, and the worker
   * mutates `roiPassingSegments` to say which streamlines survive.
   */
  roiPassingSegments?: Uint64Set;
  roiFilterActive?: WatchableValue<boolean>;
  roiGhostAlpha?: WatchableValue<number>;
  roiGroups?: WatchableValue<readonly RoiGroupConfig[]>;
  /**
   * Evaluate the given groups over the currently-resident chunks and return each
   * group's passing object ids (WYSIWYG). Set once the tract render layer exists;
   * the Export tab calls it to select streamlines by id rather than re-folding
   * the whole level in the exporter. Positional: `result[i]` ↔ `groups[i]`.
   */
  computeRoiExportIds?: (
    groups: readonly RoiGroupConfig[],
  ) => Promise<bigint[][]>;
  /**
   * The per-vertex attribute names the geometry source loaded, in load order
   * (the on-disk `vertex_attributes/<name>` directory names). The Filter tab
   * offers these as filter targets; for a point cloud they are the ONLY tier it
   * can offer, since that kind has no per-object attributes at all. Set when a
   * zarr-vectors geometry source activates.
   */
  roiVertexAttributeNames?: readonly string[];
  /**
   * On-disk dtypes parallel to {@link roiVertexAttributeNames}. The Filter tab
   * needs them to tell a FLAG from a measurement when the loaded chunks happen
   * to show only one of the flag's two values.
   */
  roiVertexAttributeDtypes?: readonly string[];
  /**
   * Measure the named per-vertex attributes over the currently-resident chunks.
   * The values live in the worker (the frontend holds them as opaque packed
   * texture bytes), so a range for a slider has to be asked for. Positional:
   * `result[i]` ↔ `names[i]`.
   */
  computeRoiVertexAttrStats?: (
    names: readonly string[],
  ) => Promise<VertexAttrStats[]>;
  /**
   * Whether this layer's geometry is exportable as tracts. The Export tab's two
   * formats are streamline-shaped, so a point-cloud or mesh store gets the
   * Filter tab without it.
   */
  roiSupportsTractExport?: boolean;
  /**
   * What the filterable geometry draws as, so the Filter tab can name what it
   * is selecting: a point cloud's passing ids are single cells, not tracts.
   */
  roiGeometryPrimitive?: "points" | "lines" | "triangles";
  /**
   * The pass-2 (object-keyed, high-detail) skeleton source. Its resident chunks
   * hold whole tracts' geometry in frontend memory, which the Export tab reads
   * directly for the fast in-browser TRK export (no store re-read). Present only
   * once the tract layer's high-detail source is created.
   */
  roiHighDetailSkeletonSource?: SkeletonSource;
  roiSegmentColors?: Uint64Map;
  roiColorByGroup?: WatchableValue<boolean>;
  /**
   * Per-object numeric attribute columns (from the loaded segment-property map),
   * shipped to the worker so a group's length filter and object-attribute colour
   * can be evaluated. Keyed by attribute name; rebuilt when the property map
   * changes. `undefined`/empty until such a map loads.
   */
  roiObjectAttrColumns?: WatchableValue<
    ReadonlyMap<string, RoiObjectAttrColumn>
  >;
  /**
   * Dense anatomical label grid built from {@link roiLabelLayer}, shipped to the
   * worker so `labelMask` ROIs can be sampled per streamline vertex. Rebuilt when
   * the linked parcellation changes or finishes loading; undefined when none is
   * linked or it is still loading.
   */
  roiLabelField?: WatchableValue<RoiLabelField | undefined>;
  /**
   * Frontend-only per-object value map (id -> normalised attribute value, 16-bit
   * packed) for the background length filter + flat colour-by-attribute shader
   * tier, plus the resolved uniforms in {@link roiBackground}. Not sent to the
   * worker (the shader reads it directly).
   */
  roiObjectValues?: Uint64Map;
  roiBackground?: WatchableValue<RoiBackgroundUniforms | undefined>;
  /**
   * Shared set of object ids the pass-1 backend fills = union of visible +
   * high-detail groups' passing tracts. Drives the object-keyed pass-2 render
   * layer (its dedicated visible set), which redraws those at full detail.
   */
  roiHighDetailSegments?: Uint64Set;
  spatialSkeletonGridLevel2d = new TrackableValue<number>(
    0,
    verifyNonnegativeInt,
    0,
  );
  spatialSkeletonGridLevel3d = new TrackableValue<number>(
    0,
    verifyNonnegativeInt,
    0,
  );
  spatialSkeletonGridLevels = new WatchableValue<SpatialSkeletonGridLevel[]>(
    [],
  );
  spatialSkeletonGridResolutionTarget2d = new TrackableValue<number>(
    1,
    verifyFiniteNonNegativeFloat,
    1,
  );
  spatialSkeletonGridResolutionTarget3d = new TrackableValue<number>(
    1,
    verifyFiniteNonNegativeFloat,
    1,
  );
  // Persistent detail bias (1 = pure camera) folded in when the user
  // adjusts the render-scale widget under auto-LOD; persisted so the
  // calibration survives a refresh (the camera-derived target itself is
  // recomputed every frame, so it isn't meaningful to persist alone).
  spatialSkeletonGridResolutionBias2d = new TrackableValue<number>(
    1,
    verifyFinitePositiveFloat,
    1,
  );
  spatialSkeletonGridResolutionBias3d = new TrackableValue<number>(
    1,
    verifyFinitePositiveFloat,
    1,
  );
  // When true, the render layer overwrites
  // `spatialSkeletonGridResolutionTarget{2d,3d}` every frame with a
  // value derived from the current camera projection: world-units-per-
  // screen-pixel at the layer's localPosition.  This makes the level
  // picker auto-track camera zoom (like image LOD selection), so
  // zooming out switches to coarser pyramid levels and zooming in
  // restores finer ones.  Default off to preserve existing user-
  // controlled behavior for layers that don't want this (CATMAID).
  autoSpatialSkeletonGridLevel2d = new TrackableBoolean(false, false);
  autoSpatialSkeletonGridLevel3d = new TrackableBoolean(false, false);
  /**
   * Drop the memory ceiling on grid-level selection.
   *
   * The ceiling refuses any level whose *fully resident* estimate exceeds the
   * GPU budget -- for a whole-brain tractogram that is level 0 at ~4 GB against
   * 1 GB. But only chunks in view are ever fetched, so a tight crop would fetch
   * a small fraction of that and is refused for a cost it will never pay. Until
   * the estimate is view-scoped (which needs a per-level in-view chunk count the
   * store does not stamp), this lets the user take that judgement themselves.
   *
   * Off by default: with it on, a wide view of a dense level really can exhaust
   * GPU memory, which is the failure the ceiling exists to prevent.
   */
  ignoreSpatialSkeletonMemoryCeiling = new TrackableBoolean(false, false);
  /**
   * What the memory left over by the pyramid level being drawn is spent on.
   * See {@link SpatialSkeletonDetailFocus} for why a tractogram wants a
   * different answer here than an image pyramid does.
   *
   * Defaults to OBJECT: every source that reaches this code publishes per-level
   * costs, which today means zarr-vectors geometry, whose objects are long
   * enough that the local answer returns them in pieces.
   */
  // Explicit type argument: without it the initialiser narrows the generic to
  // the literal `SpatialSkeletonDetailFocus.OBJECT`, so assigning any other
  // member (see `refreshSpatialSkeletonAdmission`, which drops to LOCAL when
  // the store cannot be budgeted per object) fails to typecheck.
  spatialSkeletonDetailFocus = new TrackableEnum<SpatialSkeletonDetailFocus>(
    SpatialSkeletonDetailFocus,
    SpatialSkeletonDetailFocus.OBJECT,
  );
  /**
   * Whether the focus above was CHOSEN -- restored from the layer's JSON or
   * picked in the UI -- as opposed to being the class default.
   *
   * The default is OBJECT because that is what a tractogram wants, and a
   * tractogram is what the mode was built for. A store that cannot be budgeted
   * per object still honours object focus (one level everywhere, its whole
   * volume resident), but the sizing behind that is "the whole level fits the
   * GPU budget" -- which assumes the layer has the budget to itself. Three
   * geometry layers over one volume each assume that, and together they thrash:
   * every layer's whole level is requested, nothing stays resident, and the
   * viewer draws nothing at all. So on those stores the DEFAULT reverts to
   * LOCAL, once, when the source reports the capability
   * ({@link updateSpatialSkeletonSourceState}) -- and an explicit choice is
   * left alone.
   */
  spatialSkeletonDetailFocusExplicit = false;
  /** Guards our own writes to the focus, so they do not read as a choice. */
  private applyingDefaultDetailFocus = false;

  /**
   * Set the focus without marking it as the user's choice.
   *
   * Returns true if the value moved.
   */
  applyDefaultSpatialSkeletonDetailFocus(value: SpatialSkeletonDetailFocus) {
    if (this.spatialSkeletonDetailFocus.value === value) return false;
    this.applyingDefaultDetailFocus = true;
    try {
      this.spatialSkeletonDetailFocus.value = value;
    } finally {
      this.applyingDefaultDetailFocus = false;
    }
    return true;
  }

  /** See {@link spatialSkeletonDetailFocusExplicit}. */
  noteSpatialSkeletonDetailFocusChanged() {
    if (!this.applyingDefaultDetailFocus) {
      this.spatialSkeletonDetailFocusExplicit = true;
    }
  }

  /**
   * Bytes the level being drawn leaves unspent: the budget minus its
   * fully-resident estimate. Under OBJECT focus this is what buys whole
   * objects; under LOCAL focus nothing reads it.
   */
  /**
   * Share of the drawn level's NEW objects to decode, in [0, 1]. `1` means no
   * rationing. Derived, never set by the user; see
   * {@link refreshSpatialSkeletonAdmission}.
   */
  spatialSkeletonAdmissionFraction = new WatchableValue<number>(1);
  /**
   * Bytes ONE cell of each level costs on the GPU, coarsest-first. This is what
   * makes LOCAL focus respond to zoom: the budget divided by the number of cells
   * in view names the finest level each cell can afford.
   */
  spatialSkeletonPerCellCostBytes = new WatchableValue<number[]>([]);
  spatialSkeletonGridRenderScaleHistogram2d = new RenderScaleHistogram();
  spatialSkeletonGridRenderScaleHistogram3d = new RenderScaleHistogram();
  spatialSkeletonLod2d = new WatchableValue<number>(0);
  spatialSkeletonNodeQuery = new TrackableValue<string>("", verifyString);
  spatialSkeletonNodeFilter = new TrackableEnum(
    SpatialSkeletonNodeFilterType,
    SpatialSkeletonNodeFilterType.NONE,
  );
  ignoreNullVisibleSet = new TrackableBoolean(true, true);
  skeletonRenderingOptions = new SkeletonRenderingOptions();
  shaderError = makeWatchableShaderError();
  renderScaleHistogram = new RenderScaleHistogram();
  renderScaleTarget = trackableRenderScaleTarget(1);
  selectSegment: (id: bigint, pin: boolean | "toggle" | "force-unpin") => void;
  transparentPickEnabled: TrackableBoolean;
  baseSegmentColoring = new TrackableBoolean(false, false);
  baseSegmentHighlighting = new TrackableBoolean(false, false);
  useTempSegmentStatedColors2d: SharedWatchableValue<boolean>;
  hasVolume = new TrackableBoolean(false, false);

  filterBySegmentLabel: (id: bigint) => void;

  moveToSegment = (id: bigint) => {
    this.layer.moveToSegment(id);
  };

  /**
   * Publish the pyramid's levels, coarsest first, and choose one.
   *
   * `levelCostsBytes` (parallel to `gridSizes`) and `budgetBytes` opt a source
   * into budget-driven selection: the finest level that fits is chosen, rather
   * than the one closest to the camera-derived resolution target. That matters
   * where levels differ in *how many complete objects* they hold rather than
   * in resolution — detail-per-pixel then says nothing about whether a level
   * will fit, and on a whole-brain tractogram the camera target asks for the
   * finest level, ~10^8 vertices and several times the GPU budget. Sources
   * that omit them keep the camera-driven behaviour.
   */
  /**
   * Level chosen by memory budget, or `undefined` when the source did not opt
   * in. While set, the camera-driven resolution target does not re-select:
   * "the finest level that fits" and "the level matching the screen" are
   * different questions, and the camera's answer would otherwise win every
   * frame.
   */
  private spatialSkeletonBudgetLevel: number | undefined;

  setSpatialSkeletonGridSizes(
    gridSizes: SpatialSkeletonGridSize[],
    levelCostsBytes?: number[],
    budgetBytes?: number,
    levelObjectCounts?: (number | undefined)[],
    levelCellCounts?: number[],
  ) {
    const perCell =
      levelCostsBytes !== undefined &&
      levelCellCounts !== undefined &&
      levelCellCounts.length === levelCostsBytes.length
        ? levelCostsBytes.map((cost, k) =>
            levelCellCounts[k] > 0 ? cost / levelCellCounts[k] : Number.NaN,
          )
        : [];
    this.spatialSkeletonPerCellCostBytes.value = perCell;
    const levels = buildSpatialSkeletonGridLevels(gridSizes, levelObjectCounts);
    const { origin: histogramOrigin, binSize: histogramBinSize } =
      getSpatialSkeletonGridHistogramConfig(levels);
    if (
      this.spatialSkeletonGridRenderScaleHistogram2d.logScaleOrigin !==
        histogramOrigin ||
      this.spatialSkeletonGridRenderScaleHistogram2d.logScaleBinSize !==
        histogramBinSize
    ) {
      this.spatialSkeletonGridRenderScaleHistogram2d.logScaleOrigin =
        histogramOrigin;
      this.spatialSkeletonGridRenderScaleHistogram2d.logScaleBinSize =
        histogramBinSize;
      this.spatialSkeletonGridRenderScaleHistogram2d.changed.dispatch();
    }
    if (
      this.spatialSkeletonGridRenderScaleHistogram3d.logScaleOrigin !==
        histogramOrigin ||
      this.spatialSkeletonGridRenderScaleHistogram3d.logScaleBinSize !==
        histogramBinSize
    ) {
      this.spatialSkeletonGridRenderScaleHistogram3d.logScaleOrigin =
        histogramOrigin;
      this.spatialSkeletonGridRenderScaleHistogram3d.logScaleBinSize =
        histogramBinSize;
      this.spatialSkeletonGridRenderScaleHistogram3d.changed.dispatch();
    }
    this.spatialSkeletonGridLevels.value = levels;
    if (levels.length === 0) return;
    // Kept so the ceiling can be recomputed later without re-activating the
    // datasource -- see `updateSpatialSkeletonBudget`.
    this.spatialSkeletonLevelCostsBytes =
      levelCostsBytes !== undefined && levelCostsBytes.length === levels.length
        ? levelCostsBytes.slice()
        : undefined;
    this.spatialSkeletonBudgetBytes = budgetBytes;
    const budgeted = this.computeSpatialSkeletonBudgetLevel(budgetBytes);
    this.spatialSkeletonBudgetLevel = budgeted;
    // Initial selection SUBSTITUTES the ceiling, where later changes clamp to
    // it (`applySpatialSkeletonResolutionTarget`). That difference is
    // deliberate: the resolution targets still hold their default of 1 at
    // activation, because under auto-LOD the render layer only starts deriving
    // them from the camera once it has drawn a frame. Clamping against a
    // not-yet-meaningful target would pick a level from a placeholder.
    const target3dIndex =
      budgeted ??
      findClosestSpatialSkeletonGridLevelBySpacing(
        levels,
        this.spatialSkeletonGridResolutionTarget3d.value,
      );
    this.setSpatialSkeletonGridLevel("3d", target3dIndex);
    const target2dIndex =
      budgeted ??
      findClosestSpatialSkeletonGridLevelBySpacing(
        levels,
        this.spatialSkeletonGridResolutionTarget2d.value,
      );
    this.setSpatialSkeletonGridLevel("2d", target2dIndex);
    // Under OBJECT focus this immediately overrides both picks above with the
    // level the memory budget actually implies.
    this.refreshSpatialSkeletonAdmission();
  }

  /** Per-level fully-resident cost estimates, coarsest first; see the ctor. */
  private spatialSkeletonLevelCostsBytes: number[] | undefined;
  /**
   * The tract source's own answer to "which whole objects fit in this many
   * bytes", or `undefined` for a store that cannot be budgeted per object.
   * Set when the subsource activates; see `computeObjectAdmission`.
   */
  spatialSkeletonComputeAdmission:
    | ((budgetBytes: number) => ObjectAdmission | undefined)
    | undefined;
  /** Budget the ceiling was last computed against, so a toggle can reuse it. */
  private spatialSkeletonBudgetBytes: number | undefined;

  /**
   * Estimated fully-resident bytes per level, **finest-first** (index == the
   * export level number, 0 = full resolution); `[]` when the store carries no
   * per-level metadata. Reuses the estimate already computed for the streamline
   * budget -- the Export tab uses it to grey out levels a browser export cannot
   * afford.
   */
  roiExportLevelCostsBytes(): number[] {
    const costs = this.spatialSkeletonLevelCostsBytes; // coarsest-first
    return costs === undefined ? [] : costs.slice().reverse();
  }

  private computeSpatialSkeletonBudgetLevel(
    budgetBytes: number | undefined,
  ): number | undefined {
    // `undefined` means "no ceiling", which is exactly what the override wants:
    // `applySpatialSkeletonResolutionTarget` then passes the requested level
    // through unclamped.
    if (this.ignoreSpatialSkeletonMemoryCeiling.value) return undefined;
    const costs = this.spatialSkeletonLevelCostsBytes;
    if (
      costs === undefined ||
      budgetBytes === undefined ||
      !Number.isFinite(budgetBytes)
    ) {
      return undefined;
    }
    return selectSpatialSkeletonGridLevelByBudget(costs, budgetBytes);
  }

  /**
   * Recompute the memory ceiling against a possibly-changed budget.
   *
   * The ceiling used to be decided exactly once, inside the datasource
   * activation path, so raising the GPU memory limit changed nothing until the
   * layer was reloaded -- the limit is user-editable, but the level it gated
   * was not re-derived from it.
   *
   * Re-applies the current resolution targets rather than substituting the new
   * ceiling: by the time this runs the targets are live (camera-derived under
   * auto-LOD, or user-set), so clamping respects them while still refusing
   * anything that does not fit.
   */
  /** Last GPU limit seen, so the auto budget can be recomputed on its own. */
  /**
   * The GPU byte limit the layer sizes against.
   *
   * A watchable, not a plain number: the render layers receive a SPREAD of this
   * display state, which copies plain fields by value — so a number would freeze
   * at its value on activation and stop tracking the user's memory limit, while
   * a watchable is copied by reference and keeps reporting the live one.
   */
  spatialSkeletonGpuBudgetBytes = new WatchableValue<number>(0);

  updateSpatialSkeletonBudget(budgetBytes?: number | undefined) {
    if (this.spatialSkeletonGridLevels.value.length === 0) return;
    if (budgetBytes !== undefined)
      this.spatialSkeletonBudgetBytes = budgetBytes;
    const next = this.computeSpatialSkeletonBudgetLevel(
      this.spatialSkeletonBudgetBytes,
    );
    if (next !== this.spatialSkeletonBudgetLevel) {
      this.spatialSkeletonBudgetLevel = next;
      this.reapplySpatialSkeletonResolutionTargets();
    }
    // Always, even when the whole-level ceiling did not move: raising the GPU
    // limit by less than a whole rung still buys more objects out of the next
    // one, and that is the entire point of budgeting per object.
    this.refreshSpatialSkeletonAdmission();
  }

  /**
   * Re-run both views' resolution targets through the current ceiling.
   *
   * Needed wherever the CEILING moved without the target doing so -- a new
   * budget level, or a detail-focus switch, which changes whether the
   * whole-level ceiling applies at all (see
   * {@link applySpatialSkeletonResolutionTarget}). Without this the level stays
   * where the previous mode left it: on a store that cannot be budgeted per
   * object, switching to OBJECT focus would keep a level chosen for the cells
   * in view and then make its whole volume resident.
   */
  reapplySpatialSkeletonResolutionTargets() {
    this.applySpatialSkeletonResolutionTarget(
      "3d",
      this.spatialSkeletonGridResolutionTarget3d.value,
    );
    this.applySpatialSkeletonResolutionTarget(
      "2d",
      this.spatialSkeletonGridResolutionTarget2d.value,
    );
  }

  private setSpatialSkeletonGridLevel(kind: "2d" | "3d", index: number) {
    const levels = this.spatialSkeletonGridLevels.value;
    if (levels.length === 0) return 0;
    const clampedIndex = Math.min(Math.max(index, 0), levels.length - 1);
    if (kind === "2d") {
      this.spatialSkeletonGridLevel2d.value = clampedIndex;
      const nextLod = levels[clampedIndex].lod;
      if (this.spatialSkeletonLod2d.value !== nextLod) {
        this.spatialSkeletonLod2d.value = nextLod;
      }
      return clampedIndex;
    }
    this.spatialSkeletonGridLevel3d.value = clampedIndex;
    const nextLod = levels[clampedIndex].lod;
    if (this.skeletonLod.value !== nextLod) {
      this.skeletonLod.value = nextLod;
    }
    return clampedIndex;
  }

  /**
   * Re-derive the fill level for both views from the levels they now draw.
   *
   * Offered ONLY to a view sitting exactly on the memory ceiling. Below the
   * ceiling the next level down fits whole, so there is nothing to ration and
   * a partial load would be a worse picture than the one the user (or the
   * camera) asked for; above it there is no ceiling in force at all, because
   * the source published no costs or the user overrode it -- and in that case
   * the finer level is already free to be selected outright.
   */
  /**
   * Re-derive which objects the memory budget buys, and which level to read
   * them from.
   *
   * This REPLACES level selection under OBJECT focus. The camera cannot answer
   * the question: an object-sparsity pyramid's levels differ in how many whole
   * tracts they hold, not in detail per pixel, so a resolution target saturates
   * at "finest" over the entire useful zoom range and the level stops moving.
   * What actually bounds the picture is memory, so memory chooses — the finest
   * level whose objects fit, plus a rationed share of the level below it.
   *
   * Under LOCAL focus this is inert and the camera keeps deciding, per cell.
   */
  private refreshSpatialSkeletonAdmission() {
    const levels = this.spatialSkeletonGridLevels.value;
    const compute = this.spatialSkeletonComputeAdmission;
    const budgetBytes = this.spatialSkeletonBudgetBytes;
    const objectFocus =
      this.spatialSkeletonDetailFocus.value ===
      SpatialSkeletonDetailFocus.OBJECT;
    // The user's override has to reach here too. It is consulted nowhere else
    // than the whole-level ceiling, and OBJECT focus bypasses that path
    // entirely -- so without this the checkbox is inert on exactly the layers it
    // exists for. Under per-object budgeting "ignore the ceiling" means "spend
    // as if memory were unlimited", which admits every object at the finest
    // level: the same escape hatch, expressed in this mode's terms.
    const effectiveBudget = this.ignoreSpatialSkeletonMemoryCeiling.value
      ? Number.POSITIVE_INFINITY
      : budgetBytes;
    const admission =
      objectFocus &&
      compute !== undefined &&
      effectiveBudget !== undefined &&
      levels.length > 0
        ? compute(effectiveBudget)
        : undefined;
    if (admission === undefined) {
      // No per-object budgeting available (or not asked for): draw whole levels
      // and leave the rationing off.
      if (this.spatialSkeletonAdmissionFraction.value !== 1) {
        this.spatialSkeletonAdmissionFraction.value = 1;
      }
      if (objectFocus) {
        warnOnceAdmissionUnavailable(compute !== undefined);
        // ...and that is the whole of it: the mode STAYS SELECTED.
        //
        // Object focus is two behaviours, and only one of them needs the store
        // to carry per-object membership.
        //
        // The half that always works: one level everywhere instead of per-cell
        // arbitration, and that level's WHOLE VOLUME resident rather than the
        // frustum. That is what makes an object load as an object -- uniform
        // detail across it, no cut-off where the finer chunks ran out, no decay
        // into the visible piece as the camera turns -- and it is exactly what
        // a mesh or point-cloud store wants. Level selection simply reverts to
        // the camera under the whole-level ceiling (see
        // {@link applySpatialSkeletonResolutionTarget}), which is the right
        // ceiling here because the whole level really is loaded.
        //
        // The half that does not: drawing the UNION of every level with
        // `gridIndex <= gridLevel` (in the draw list
        // `SpatiallyIndexedSkeletonLayer.forEachVisibleChunkSlot` and in the
        // worker's request set `selectScales`), plus rationing the finest one.
        // That is sound only where the levels PARTITION the objects between
        // them, as `admitObjects` in the zarr-vectors backend guarantees from
        // `coarserMembership`; on a plain resolution pyramid -- every level a
        // decimated copy of every object, `object_sparsity` 1.0 -- it draws the
        // same object once per resident level, superimposed. Overlapping
        // decimations of one surface sum on screen, which reads as additive
        // rendering; it is duplicated geometry, not blending, so no opacity
        // control affects it.
        //
        // So the union is gated at its own sites, on the same store property
        // (`partitionsObjects`, published per source), and this branch no
        // longer has to switch the user's mode off to keep it safe. It used
        // to, which is why object focus could not be selected at all on a mesh
        // or point-cloud layer: the control sprang back to LOCAL.
      }
      return;
    }
    // `loadLevel` counts from the finest level; a grid level counts from the
    // coarsest (see `gridIndex` in the zarr-vectors datasource).
    const gridLevel = levels.length - 1 - admission.loadLevel;
    this.setSpatialSkeletonGridLevel("3d", gridLevel);
    this.setSpatialSkeletonGridLevel("2d", gridLevel);
    if (this.spatialSkeletonAdmissionFraction.value !== admission.fraction) {
      this.spatialSkeletonAdmissionFraction.value = admission.fraction;
    }
  }

  linkedSegmentationGroup: LinkedLayerGroup;
  linkedSegmentationColorGroup: LinkedLayerGroup;
  originalSegmentationGroupState: SegmentationUserLayerGroupState;
  originalSegmentationColorGroupState: SegmentationUserLayerColorGroupState;

  /**
   * Reference to a segmentation (parcellation) layer whose anatomical labels can
   * be toggled include/exclude to dissect the tracts (the streamline Filter tab's
   * "By segmentation label" panel). Persisted so a chosen parcellation survives a
   * reload. Undefined-name = no parcellation linked.
   */
  roiLabelLayer: LayerReference;

  segmentationGroupState: WatchableValueInterface<SegmentationUserLayerGroupState>;
  segmentationColorGroupState: WatchableValueInterface<SegmentationUserLayerColorGroupState>;

  // Indirect properties
  hideSegmentZero: WatchableValueInterface<boolean>;
  segmentColorHash: TrackableValueInterface<number>;
  segmentStatedColors: WatchableValueInterface<Uint64Map>;
  effectiveSegmentStatedColors: WatchableValueInterface<Uint64Map>;
  private derivedSegmentStatedColors: Uint64Map;
  private effectiveStatedColorsUsesLut = false;
  private effectiveSegmentStatedColorsChanged = new Signal();
  tempSegmentStatedColors2d: WatchableValueInterface<Uint64Map>;
  segmentDefaultColor: WatchableValueInterface<vec3 | undefined>;
  tempSegmentDefaultColor2d: WatchableValueInterface<vec3 | vec4 | undefined>;
  highlightColor: WatchableValueInterface<vec4 | undefined>;
  segmentQuery: WatchableValueInterface<string>;
  segmentPropertyMap: WatchableValueInterface<
    PreprocessedSegmentPropertyMap | undefined
  >;
}

interface SegmentationActionContext extends LayerActionContext {
  // Restrict the `select` action not to both toggle on and off segments.  If segment would be
  // toggled on in at least one layer, only toggle segments on.
  segmentationToggleSegmentState?: boolean | undefined;
}

interface SelectedSpatialSkeletonNodeInfo {
  nodeId: number;
  segmentId?: number;
  position?: Float32Array;
  sourceState?: SpatialSkeletonSourceState;
}

function normalizeOptionalPositiveSafeInteger(value: unknown) {
  if (value === undefined) return undefined;
  const normalized = Math.round(Number(value));
  return Number.isSafeInteger(normalized) && normalized > 0
    ? normalized
    : undefined;
}

function copyOptionalSpatialSkeletonPosition(
  value: ArrayLike<number> | undefined,
) {
  if (value === undefined) return undefined;
  return new Float32Array(Array.from(value, Number));
}

const Base = UserLayerWithAnnotationsMixin(UserLayer);
export class SegmentationUserLayer extends Base {
  sliceViewRenderScaleHistogram = new RenderScaleHistogram();
  sliceViewRenderScaleTarget = trackableRenderScaleTarget(1);
  codeVisible = new TrackableBoolean(true);
  readonly spatialSkeletonState = this.registerDisposer(
    new SpatialSkeletonState(),
  );
  readonly selectedSpatialSkeletonNodeId = new WatchableValue<
    number | undefined
  >(undefined);
  readonly selectedSpatialSkeletonNodeInfo = new WatchableValue<
    SelectedSpatialSkeletonNodeInfo | undefined
  >(undefined);
  readonly hoveredSpatialSkeletonNodeId = this.registerDisposer(
    new SpatialSkeletonHoverState(),
  );
  readonly spatialSkeletonVisibleChunksNeeded = new WatchableValue(0);
  readonly spatialSkeletonVisibleChunksAvailable = new WatchableValue(0);
  readonly spatialSkeletonVisibleChunksLoaded = new WatchableValue(false);

  graphConnection = new WatchableValue<
    SegmentationGraphSourceConnection | undefined
  >(undefined);

  bindSegmentListWidth(element: HTMLElement) {
    return bindSegmentListWidth(this.displayState, element);
  }

  segmentQueryFocusTime = new WatchableValue<number>(Number.NEGATIVE_INFINITY);

  selectSegment = (id: bigint, pin: boolean | "toggle" | "force-unpin") => {
    this.manager.root.selectionState.captureSingleLayerState(
      this,
      (state) => {
        state.value = id;
        return true;
      },
      pin,
    );
  };

  private captureSpatialSkeletonSelectionState(
    capture: (state: this["selectionState"]) => boolean,
    pin: boolean | "toggle" | "force-unpin",
    options: { position?: ArrayLike<number> } = {},
  ) {
    const selectionState = this.manager.root.selectionState;
    if (pin !== false || selectionState.pin.value) {
      selectionState.captureSingleLayerState(this, capture, pin, options);
      return;
    }
    const state = {} as UserLayerSelectionState;
    this.initializeSelectionState(state);
    if (!capture(state)) return;
    selectionState.value = {
      layers: [{ layer: this, state }],
      coordinateSpace: selectionState.coordinateSpace.value,
      position:
        options.position === undefined
          ? undefined
          : new Float32Array(options.position),
    };
  }

  private getGlobalSelectionPositionFromModelPosition(
    modelPosition: ArrayLike<number> | undefined,
  ) {
    if (modelPosition === undefined) return undefined;
    const transform =
      this.getSpatiallyIndexedSkeletonLayer()?.displayState.transform.value;
    if (transform === undefined || transform.error !== undefined)
      return undefined;
    const rank = transform.rank;
    const paddedModelPosition = new Float32Array(rank);
    for (let i = 0; i < Math.min(modelPosition.length, rank); ++i) {
      paddedModelPosition[i] = Number(modelPosition[i]);
    }
    const layerPosition = new Float32Array(rank);
    matrix.transformPoint(
      layerPosition,
      transform.modelToRenderLayerTransform,
      rank + 1,
      paddedModelPosition,
      rank,
    );
    const result = this.manager.root.globalPosition.value.slice();
    gatherUpdate(
      result,
      layerPosition,
      transform.globalToRenderLayerDimensions,
    );
    return result;
  }

  moveViewToSpatialSkeletonNodePosition(position: ArrayLike<number>) {
    const transform =
      this.getSpatiallyIndexedSkeletonLayer()?.displayState.transform.value;
    if (transform === undefined || transform.error !== undefined) return;
    const rank = transform.rank;
    const modelPosition = new Float32Array(rank);
    for (let i = 0; i < Math.min(position.length, rank); ++i) {
      modelPosition[i] = Number(position[i]);
    }
    const layerPosition = new Float32Array(rank);
    matrix.transformPoint(
      layerPosition,
      transform.modelToRenderLayerTransform,
      rank + 1,
      modelPosition,
      rank,
    );
    this.setLayerPosition(transform, layerPosition);
  }

  selectSpatialSkeletonNode = (
    nodeId: number,
    pin: boolean | "toggle" = false,
    options: {
      segmentId?: number;
      position?: ArrayLike<number>;
      sourceState?: SpatialSkeletonSourceState;
    } = {},
  ) => {
    const normalizedNodeId = normalizeOptionalPositiveSafeInteger(nodeId);
    if (normalizedNodeId === undefined) {
      return;
    }
    const selectedNodeInfo =
      this.getSpatiallyIndexedSkeletonLayer()?.getNode(normalizedNodeId);
    const requestedSegmentId =
      options.segmentId ?? selectedNodeInfo?.segmentId ?? undefined;
    const segmentId = normalizeOptionalPositiveSafeInteger(requestedSegmentId);
    const selectedNodePosition = options.position ?? selectedNodeInfo?.position;
    const selectedGlobalPosition =
      this.getGlobalSelectionPositionFromModelPosition(selectedNodePosition);
    const sourceState = options.sourceState ?? selectedNodeInfo?.sourceState;
    this.selectedSpatialSkeletonNodeInfo.value = {
      nodeId: normalizedNodeId,
      segmentId,
      position: copyOptionalSpatialSkeletonPosition(selectedNodePosition),
      sourceState,
    };
    this.captureSpatialSkeletonSelectionState(
      (state) => {
        state.nodeId = normalizedNodeId.toString();
        state.value = segmentId === undefined ? undefined : BigInt(segmentId);
        return true;
      },
      pin,
      { position: selectedGlobalPosition },
    );
  };

  selectAndMoveToSpatialSkeletonNode(
    node:
      | Pick<SpatiallyIndexedSkeletonNode, "nodeId" | "segmentId" | "position">
      | undefined,
    pin: boolean | "toggle" = this.manager.root.selectionState.pin.value,
  ) {
    if (node === undefined) {
      this.clearSpatialSkeletonNodeSelection(pin);
      return false;
    }
    this.selectSpatialSkeletonNode(node.nodeId, pin, {
      segmentId: node.segmentId,
      position: node.position,
    });
    this.moveViewToSpatialSkeletonNodePosition(node.position);
    return true;
  }

  inspectSpatialSkeletonSegment = (
    segmentId: number,
    options: { secondary?: boolean } = {},
  ) => {
    void options;
    const normalizedSegmentId = Math.round(Number(segmentId));
    if (
      !Number.isSafeInteger(normalizedSegmentId) ||
      normalizedSegmentId <= 0
    ) {
      return false;
    }
    const visibleSegments = getVisibleSegments(
      this.displayState.segmentationGroupState.value,
    );
    if (visibleSegments.has(BigInt(normalizedSegmentId))) {
      return false;
    }
    addSegmentToVisibleSets(
      this.displayState.segmentationGroupState.value,
      BigInt(normalizedSegmentId),
    );
    return true;
  };

  setSpatialSkeletonMergeAnchor = (nodeId: number | undefined) => {
    return this.spatialSkeletonState.setMergeAnchor(nodeId);
  };

  clearSpatialSkeletonMergeAnchor = () => {
    return this.spatialSkeletonState.clearMergeAnchor();
  };

  ensureSpatialSkeletonInspectionFromSelection = () => {
    const selectedNodeId = this.selectedSpatialSkeletonNodeId.value;
    const selectedNode =
      selectedNodeId === undefined
        ? undefined
        : this.spatialSkeletonState.getCachedNode(selectedNodeId);
    const visibleSegments = getVisibleSegments(
      this.displayState.segmentationGroupState.value,
    );
    if (
      selectedNode !== undefined &&
      visibleSegments.has(BigInt(selectedNode.segmentId))
    ) {
      return selectedNode.segmentId;
    }
    const selectedSegmentValue =
      this.displayState.segmentSelectionState.baseValue ?? undefined;
    const selectedSegmentId =
      selectedSegmentValue === undefined
        ? undefined
        : Number(selectedSegmentValue);
    if (
      selectedSegmentId === undefined ||
      !Number.isSafeInteger(selectedSegmentId) ||
      selectedSegmentId <= 0
    ) {
      return undefined;
    }
    return visibleSegments.has(BigInt(selectedSegmentId))
      ? selectedSegmentId
      : undefined;
  };

  clearSpatialSkeletonNodeSelection = (
    pin: boolean | "toggle" | "force-unpin" = false,
  ) => {
    this.selectedSpatialSkeletonNodeInfo.value = undefined;
    this.captureSpatialSkeletonSelectionState((state) => {
      state.nodeId = undefined;
      state.value = undefined;
      return true;
    }, pin);
  };

  filterBySegmentLabel = (id: bigint) => {
    const augmented = augmentSegmentId(this.displayState, id);
    const { label } = augmented;
    if (!label) return;
    this.filterSegments(label);
  };

  filterSegments = (query: string) => {
    this.displayState.segmentationGroupState.value.segmentQuery.value = query;
    this.segmentQueryFocusTime.value = Date.now();
    this.tabs.value = "segments";
    this.manager.root.selectedLayer.layer = this.managedLayer;
  };

  displayState = new SegmentationUserLayerDisplayState(this);
  readonly spatialSkeletonEditMode = this.spatialSkeletonState.editMode;
  readonly spatialSkeletonMergeMode = this.spatialSkeletonState.mergeMode;
  readonly spatialSkeletonSplitMode = this.spatialSkeletonState.splitMode;
  readonly spatialSkeletonNodeDataVersion =
    this.spatialSkeletonState.nodeDataVersion;

  anchorSegment = new TrackableValue<bigint | undefined>(undefined, (x) =>
    x === undefined ? undefined : parseUint64(x),
  );

  constructor(managedLayer: Borrowed<ManagedUserLayer>) {
    super(managedLayer);
    this.codeVisible.changed.add(this.specificationChanged.dispatch);
    this.registerDisposer(
      registerNestedSync((context, group) => {
        context.registerDisposer(
          group.specificationChanged.add(this.specificationChanged.dispatch),
        );
        this.specificationChanged.dispatch();
      }, this.displayState.segmentationGroupState),
    );
    this.registerDisposer(
      registerNestedSync((context, group) => {
        context.registerDisposer(
          group.specificationChanged.add(this.specificationChanged.dispatch),
        );
        this.specificationChanged.dispatch();
      }, this.displayState.segmentationColorGroupState),
    );
    this.displayState.segmentSelectionState.bindTo(
      this.manager.layerSelectedValues,
      this,
    );
    const syncSelectedSpatialSkeletonNodeIdFromGlobalSelection = () => {
      const nextLayerSelectionState =
        this.manager.root.selectionState.value?.layers.find(
          (entry) => entry.layer === this,
        )?.state;
      const nextSelectedNodeId = getNodeIdFromLayerSelectionState(
        nextLayerSelectionState,
      );
      const nextSelectedSegmentId = getSegmentIdFromLayerSelectionValue(
        nextLayerSelectionState,
      );
      if (this.selectedSpatialSkeletonNodeId.value !== nextSelectedNodeId) {
        this.selectedSpatialSkeletonNodeId.value = nextSelectedNodeId;
      }
      const selectedNodeInfo = this.selectedSpatialSkeletonNodeInfo.value;
      if (
        selectedNodeInfo !== undefined &&
        (selectedNodeInfo.nodeId !== nextSelectedNodeId ||
          selectedNodeInfo.segmentId !== nextSelectedSegmentId)
      ) {
        this.selectedSpatialSkeletonNodeInfo.value = undefined;
      }
    };
    this.registerDisposer(
      this.manager.root.selectionState.changed.add(
        syncSelectedSpatialSkeletonNodeIdFromGlobalSelection,
      ),
    );
    syncSelectedSpatialSkeletonNodeIdFromGlobalSelection();
    this.hoveredSpatialSkeletonNodeId.bindTo(
      this.manager.layerSelectedValues,
      this,
    );
    this.displayState.selectedAlpha.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.saturation.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.notSelectedAlpha.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.objectAlpha.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.hiddenObjectAlpha.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonNodeQuery.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonNodeFilter.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.roiFilter.changed.add(this.specificationChanged.dispatch);
    this.displayState.roiLabelLayer.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridResolutionTarget2d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridResolutionTarget3d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridResolutionBias2d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridResolutionBias3d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.hoverHighlight.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.baseSegmentColoring.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.ignoreNullVisibleSet.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.skeletonRenderingOptions.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.renderScaleTarget.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.silhouetteRendering.changed.add(
      this.specificationChanged.dispatch,
    );
    this.anchorSegment.changed.add(this.specificationChanged.dispatch);
    this.sliceViewRenderScaleTarget.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.originalSegmentationGroupState.localGraph.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.linkedSegmentationGroup.changed.add(() =>
      this.updateDataSubsourceActivations(),
    );
    this.registerDisposer(
      this.layersChanged.add(() => this.updateSpatialSkeletonChunkLoadState()),
    );
    this.registerDisposer(
      this.layersChanged.add(() => this.updateSpatialSkeletonSourceState()),
    );
    this.registerDisposer(
      this.manager.chunkManager.layerChunkStatisticsUpdated.add(() =>
        this.updateSpatialSkeletonChunkLoadState(),
      ),
    );
    this.tabs.add("rendering", {
      label: "Render",
      order: -100,
      getter: () => new DisplayOptionsTab(this),
    });
    this.tabs.add("segments", {
      label: "Seg.",
      order: -50,
      getter: () => new SegmentDisplayTab(this),
    });
    const hideSpatialSkeletonEditTab = this.registerDisposer(
      makeCachedLazyDerivedWatchableValue(
        (layers) =>
          !layers.some(
            (layer) =>
              (layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
                layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer) &&
              getSpatiallyIndexedSkeletonSource(layer.base) !== undefined,
          ),
        { changed: this.layersChanged, value: this.renderLayers },
      ),
    );
    this.tabs.add("skeleton", {
      label: "Skeleton",
      order: -45,
      getter: () => new SpatialSkeletonEditTab(this),
      hidden: hideSpatialSkeletonEditTab,
    });
    // Show the Filter tab whenever a spatially-indexed skeleton render layer is
    // present AND the ROI channel was created for it. The channel is created
    // (setting displayState.roiPassingSegments) only for sources that opt into
    // the filter, so this also excludes other spatially-indexed skeleton
    // sources (e.g. CATMAID) that render tracts but cannot be filtered. Unlike
    // the Skeleton *edit* tab, it does NOT require the skeleton-editing API,
    // which read-only zarr-vectors tracts lack.
    //
    // Every zarr-vectors geometry kind opts in, points and meshes included: a
    // dissection of a point cloud is what its attribute predicates select, and
    // the fold knows from each chunk how to attribute geometry to objects (see
    // `RoiFilterableChunk.perVertexObjects`).
    const hideFilterTab = this.registerDisposer(
      makeCachedLazyDerivedWatchableValue(
        (layers) =>
          this.displayState.roiPassingSegments === undefined ||
          !layers.some(
            (layer) =>
              layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
              layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer,
          ),
        { changed: this.layersChanged, value: this.renderLayers },
      ),
    );
    this.tabs.add("filter", {
      label: "Filter",
      order: -40,
      getter: () => new StreamlineFilterTab(this),
      hidden: hideFilterTab,
    });
    // The Filter tab's condition AND a store the export applies to. Both of the
    // Export tab's formats are streamline-shaped -- TrackVis `.trk` is polylines
    // by definition, and the zarr-vectors exporter reads whole tracts -- so a
    // point-cloud or mesh store is filterable without being exportable, and
    // offering the tab there would only lead to a job that cannot be written.
    const hideExportTab = this.registerDisposer(
      makeCachedLazyDerivedWatchableValue(
        (layers) =>
          this.displayState.roiPassingSegments === undefined ||
          this.displayState.roiSupportsTractExport !== true ||
          !layers.some(
            (layer) =>
              layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
              layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer,
          ),
        { changed: this.layersChanged, value: this.renderLayers },
      ),
    );
    this.tabs.add("export", {
      label: "Export",
      order: -38,
      getter: () => new TractExportTab(this),
      hidden: hideExportTab,
    });
    // Shares the Filter tab's visibility condition: the guide documents that
    // panel, so it should never appear without it.
    this.tabs.add("filterGuide", {
      label: "Guide",
      order: -39,
      getter: () => new StreamlineGuideTab(),
      hidden: hideFilterTab,
    });
    const hideGraphTab = this.registerDisposer(
      makeCachedDerivedWatchableValue(
        (x) => x === undefined,
        [this.displayState.segmentationGroupState.value.graph],
      ),
    );
    this.tabs.add("graph", {
      label: "Graph",
      order: -25,
      getter: () => new SegmentationGraphSourceTab(this),
      hidden: hideGraphTab,
    });
    // The memory ceiling on spatially-indexed skeleton levels is derived from
    // the GPU capacity, which the user can edit at runtime. Without this the
    // ceiling kept whatever value it was given during datasource activation, so
    // raising the limit appeared to do nothing to a tractogram.
    {
      const gpuLimit =
        this.manager.chunkManager.chunkQueueManager.capacities.gpuMemory
          .sizeLimit;
      const reapplyBudgets = () => {
        this.displayState.spatialSkeletonGpuBudgetBytes.value = gpuLimit.value;
        this.displayState.updateSpatialSkeletonBudget(gpuLimit.value);
      };
      this.displayState.spatialSkeletonGpuBudgetBytes.value = gpuLimit.value;
      this.registerDisposer(gpuLimit.changed.add(reapplyBudgets));
      this.registerDisposer(
        this.displayState.ignoreSpatialSkeletonMemoryCeiling.changed.add(() =>
          this.displayState.updateSpatialSkeletonBudget(),
        ),
      );
      // Switching focus moves the same leftover from one consumer to the other,
      // so both have to be re-derived: the spatial fill level and the whole-
      // object set are each cleared or repopulated by this pass.
      this.registerDisposer(
        this.displayState.spatialSkeletonDetailFocus.changed.add(() => {
          this.displayState.noteSpatialSkeletonDetailFocusChanged();
          this.displayState.updateSpatialSkeletonBudget();
          // ...and the level itself, which the budget pass re-derives only when
          // the budget LEVEL moved. Switching focus does not move it; it moves
          // which ceiling applies to it.
          this.displayState.reapplySpatialSkeletonResolutionTargets();
        }),
      );
    }
    this.tabs.default = "rendering";
    this.updateSpatialSkeletonChunkLoadState();
    this.updateSpatialSkeletonSourceState();
  }

  get volumeOptions() {
    return { volumeType: VolumeType.SEGMENTATION };
  }

  readonly has2dLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) => layers.some((x) => x instanceof SegmentationRenderLayer),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly has3dLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layers.some(
          (x) =>
            x instanceof MeshLayer ||
            x instanceof MultiscaleMeshLayer ||
            x instanceof PerspectiveViewSkeletonLayer ||
            x instanceof SliceViewPanelSkeletonLayer ||
            x instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
            x instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer,
        ),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly hasSkeletonsLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layers.some(
          (x) =>
            x instanceof PerspectiveViewSkeletonLayer ||
            x instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer,
        ),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly hasSpatiallyIndexedSkeletonsLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layers.some(
          (x) =>
            x instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
            x instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer,
        ),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  /**
   * Whether any drawn geometry is made of LINE segments.
   *
   * Distinct from {@link hasSkeletonsLayer}, which is true for anything drawn by
   * the skeleton render layers -- including a zarr-vectors point cloud or mesh,
   * which go through the same class but draw circles and triangles. The
   * lines-versus-points preference is meaningless for those, so the control that
   * offers it is gated on this instead.
   */
  readonly hasLineGeometryLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layers.some(
          (x) =>
            x instanceof PerspectiveViewSkeletonLayer ||
            (x instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer &&
              x.base.geometryPrimitive === "lines"),
        ),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly hasMeshLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layers.some(
          (x) => x instanceof MeshLayer || x instanceof MultiscaleMeshLayer,
        ),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly getSkeletonLayer = () => {
    for (const layer of this.renderLayers) {
      if (layer instanceof PerspectiveViewSkeletonLayer) {
        return layer.base;
      }
      if (layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer) {
        return layer.base;
      }
    }
    return undefined;
  };

  readonly getSpatiallyIndexedSkeletonLayer = () => {
    for (const layer of this.renderLayers) {
      if (layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer) {
        return layer.base;
      }
      if (layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer) {
        return layer.base;
      }
    }
    return undefined;
  };

  getSpatialSkeletonChunkStats(kind: "2d" | "3d") {
    // 2D chunks are now handled by the same backend as 3D, so only report
    // under "3d" to avoid double-counting in updateSpatialSkeletonChunkLoadState.
    if (kind === "2d") return { presentCount: 0, totalCount: 0 };
    let needed = 0;
    let available = 0;
    for (const layer of this.renderLayers) {
      if (layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer) {
        needed += layer.layerChunkProgressInfo.numVisibleChunksNeeded;
        available += layer.layerChunkProgressInfo.numVisibleChunksAvailable;
      }
    }
    return { presentCount: available, totalCount: needed };
  }

  private setSpatialSkeletonChunkLoadState(needed: number, available: number) {
    if (this.spatialSkeletonVisibleChunksNeeded.value !== needed) {
      this.spatialSkeletonVisibleChunksNeeded.value = needed;
    }
    if (this.spatialSkeletonVisibleChunksAvailable.value !== available) {
      this.spatialSkeletonVisibleChunksAvailable.value = available;
    }
    const loaded = needed > 0 && available >= needed;
    if (this.spatialSkeletonVisibleChunksLoaded.value !== loaded) {
      this.spatialSkeletonVisibleChunksLoaded.value = loaded;
    }
  }

  private updateSpatialSkeletonChunkLoadState() {
    const stats2d = this.getSpatialSkeletonChunkStats("2d");
    const stats3d = this.getSpatialSkeletonChunkStats("3d");
    this.setSpatialSkeletonChunkLoadState(
      stats2d.totalCount + stats3d.totalCount,
      stats2d.presentCount + stats3d.presentCount,
    );
  }

  private updateSpatialSkeletonSourceState() {
    let hasSpatialSkeletonLayer = false;
    for (const layer of this.renderLayers) {
      if (
        layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
        layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer
      ) {
        hasSpatialSkeletonLayer = true;
        break;
      }
    }
    if (!hasSpatialSkeletonLayer) {
      this.spatialSkeletonState.clearInspectedSkeletonCache();
    }
    this.spatialSkeletonState.updateCommandHistorySource(
      this.getSpatialSkeletonCommandHistorySource(),
    );
  }

  private getSpatialSkeletonCommandHistorySource() {
    for (const layer of this.renderLayers) {
      if (
        layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
        layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer
      ) {
        return layer.base.source;
      }
    }
    return undefined;
  }

  private supportsSpatialSkeletonAction(action: SpatialSkeletonAction) {
    const skeletonLayer = this.getSpatiallyIndexedSkeletonLayer();
    if (skeletonLayer === undefined) {
      return false;
    }
    if (action === SpatialSkeletonActions.inspect) {
      return getSpatiallyIndexedSkeletonSource(skeletonLayer) !== undefined;
    }
    const source = getEditableSpatiallyIndexedSkeletonSource(skeletonLayer);
    if (source === undefined) return false;
    return editableSpatiallyIndexedSkeletonSourceSupportsAction(source, action);
  }

  private getMissingSpatialSkeletonSupportReason(
    requiredActions: SpatialSkeletonAction | readonly SpatialSkeletonAction[],
  ) {
    const skeletonLayer = this.getSpatiallyIndexedSkeletonLayer();
    const requirements = Array.isArray(requiredActions)
      ? requiredActions
      : [requiredActions];
    if (
      skeletonLayer !== undefined &&
      requirements.some(
        (action) => action !== SpatialSkeletonActions.inspect,
      ) &&
      isSpatiallyIndexedSkeletonSourceReadOnly(skeletonLayer)
    ) {
      return "The active spatial skeleton source is read-only.";
    }
    const missingRequirements = requirements.filter(
      (action) => !this.supportsSpatialSkeletonAction(action),
    );
    if (missingRequirements.length === 0) {
      return undefined;
    }
    const names = missingRequirements.map(getSpatialSkeletonActionSupportLabel);
    return `The active spatial skeleton source does not support ${names.join(", ")}.`;
  }

  getSpatialSkeletonActionsDisabledReason(
    requiredActions:
      | SpatialSkeletonAction
      | readonly SpatialSkeletonAction[] = DEFAULT_SPATIAL_SKELETON_EDIT_ACTIONS,
    options: {
      requireVisibleChunks?: boolean;
    } = {},
  ) {
    const { requireVisibleChunks = false } = options;
    const missingSupportReason =
      this.getMissingSpatialSkeletonSupportReason(requiredActions);
    if (missingSupportReason !== undefined) {
      return missingSupportReason;
    }
    if (
      requireVisibleChunks &&
      !this.spatialSkeletonVisibleChunksLoaded.value
    ) {
      const needed = this.spatialSkeletonVisibleChunksNeeded.value;
      const available = this.spatialSkeletonVisibleChunksAvailable.value;
      if (needed === 0) {
        return "Waiting for visible skeleton chunks.";
      }
      return `Wait for visible skeleton chunks to load (${available}/${needed}).`;
    }
    return undefined;
  }

  getCachedSpatialSkeletonSegmentNodesForEdit(segmentId: number) {
    const segmentNodes =
      this.spatialSkeletonState.getCachedSegmentNodes(segmentId);
    if (segmentNodes === undefined) {
      throw new Error(
        `Segment ${segmentId} is not available in the inspected skeleton cache. Load the full skeleton before editing it.`,
      );
    }
    return segmentNodes;
  }

  async getSpatialSkeletonDeleteOperationContext(
    node: SpatiallyIndexedSkeletonNode,
  ) {
    const skeletonLayer = this.getSpatiallyIndexedSkeletonLayer();
    if (skeletonLayer === undefined) {
      throw new Error(
        "No active spatial skeleton layer found for delete action.",
      );
    }
    if (
      getEditableSpatiallyIndexedSkeletonSource(skeletonLayer) === undefined
    ) {
      throw new Error(
        "Unable to resolve editable skeleton source for the active layer.",
      );
    }

    const segmentNodes = this.getCachedSpatialSkeletonSegmentNodesForEdit(
      node.segmentId,
    );
    const currentNode = findSpatiallyIndexedSkeletonNode(
      segmentNodes,
      node.nodeId,
    );
    if (currentNode === undefined) {
      throw new Error(
        `Node ${node.nodeId} is not available in the inspected skeleton cache.`,
      );
    }
    const childNodes = getSpatiallyIndexedSkeletonDirectChildren(
      segmentNodes,
      currentNode.nodeId,
    );
    if (currentNode.parentNodeId === undefined && childNodes.length > 0) {
      throw new Error(
        "Deleting a root node with children is blocked. Reroot the skeleton manually before deleting it.",
      );
    }
    return {
      node: currentNode,
      parentNode: getSpatiallyIndexedSkeletonNodeParent(
        segmentNodes,
        currentNode,
      ),
      childNodes,
    };
  }

  getSpatialSkeletonNodeDisplayDescription(node: SpatiallyIndexedSkeletonNode) {
    return node.description?.length ? node.description : undefined;
  }

  async rerootSpatialSkeletonNode(
    node: Pick<
      SpatiallyIndexedSkeletonNode,
      "nodeId" | "segmentId" | "parentNodeId" | "position"
    >,
  ) {
    if (node.parentNodeId === undefined) {
      throw new Error(`Node ${node.nodeId} is already root.`);
    }
    await executeSpatialSkeletonReroot(this, node);
  }

  markSpatialSkeletonNodeDataChanged(options?: {
    invalidateFullSkeletonCache?: boolean;
  }) {
    this.spatialSkeletonState.markNodeDataChanged(options);
  }

  /**
   * Evaluate `groups` over the tract render layer's currently-resident chunks and
   * return each group's passing object ids (WYSIWYG). Backs the
   * `computeRoiExportIds` display-state callback the Export tab calls; the lookup
   * is done here (not captured at channel-creation) because the render layer is
   * created after the ROI channel. Rejects if no tract render layer exists yet.
   */
  private async computeRoiExportIds(
    groups: readonly RoiGroupConfig[],
  ): Promise<bigint[][]> {
    for (const renderLayer of this.renderLayers) {
      if (
        renderLayer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
        renderLayer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer
      ) {
        return renderLayer.base.computeRoiExportIds(groups);
      }
    }
    throw new Error(
      "This layer's tract geometry is not ready yet — wait for it to load, " +
        "then export.",
    );
  }

  /**
   * Measure the named per-vertex attributes over the geometry render layer's
   * resident chunks. Backs the `computeRoiVertexAttrStats` display-state
   * callback the Filter tab calls before it can draw a control for an
   * attribute; the render-layer lookup is done here, not captured at
   * channel-creation, because the render layer is created after the ROI
   * channel. Rejects while no geometry render layer exists.
   */
  private async computeRoiVertexAttrStats(
    names: readonly string[],
  ): Promise<VertexAttrStats[]> {
    for (const renderLayer of this.renderLayers) {
      if (
        renderLayer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
        renderLayer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer
      ) {
        return renderLayer.base.computeRoiVertexAttrStats(names);
      }
    }
    throw new Error(
      "This layer's geometry is not ready yet — wait for it to load.",
    );
  }

  /**
   * Create the ROI streamline-filter data channel once, on first sight of a
   * zarr-vectors spatially-indexed (tract) source. Idempotent — later calls
   * return immediately.
   *
   * Bridges the persisted {@link RoiFilterState} (URL truth) to the render
   * layer / worker: `roiGroups`, `roiFilterActive`, and `roiGhostAlpha` mirror
   * it into plain watchables (the render layer wraps the first two as shared
   * objects for the worker's passing-set recompute, and the shader reads
   * active/ghostAlpha as uniforms); `roiPassingSegments` is the shared set the
   * worker fills with the ids that survive the filter and the shader ghosts the
   * rest against. All four are disposed with the layer.
   */
  private ensureRoiFilterChannel() {
    const displayState = this.displayState;
    if (displayState.roiPassingSegments !== undefined) return;
    const rpc = this.manager.chunkManager.rpc!;
    const roiFilter = displayState.roiFilter;
    const passingSegments = this.registerDisposer(
      Uint64Set.makeWithCounterpart(rpc),
    );
    const highDetailSegments = this.registerDisposer(
      Uint64Set.makeWithCounterpart(rpc),
    );
    // Effective-active: the shader ghosts non-passing streamlines only when the
    // user has the filter on AND some visible group has an ROI. With no ROIs an
    // empty passing set would otherwise ghost EVERY streamline (nothing is
    // "passing"), whereas "no ROIs" means "no filter". Folding that in here
    // keeps the case correct without the shader needing to know it.
    const effectiveActive = () =>
      roiFilter.active && roiFilter.hasVisibleRois();
    const active = new WatchableValue<boolean>(effectiveActive());
    const ghostAlpha = new WatchableValue<number>(roiFilter.ghostAlpha);
    const groups = new WatchableValue<readonly RoiGroupConfig[]>(
      buildRoiGroupConfigs(roiFilter),
    );
    // Push URL-truth changes into the non-persisted watchables. Each setter
    // only dispatches when its value actually changed, so an unrelated edit
    // (e.g. colour-by-group) does not needlessly re-trigger a worker recompute.
    this.registerDisposer(
      roiFilter.changed.add(() => {
        active.value = effectiveActive();
        ghostAlpha.value = roiFilter.ghostAlpha;
        groups.value = buildRoiGroupConfigs(roiFilter);
      }),
    );
    displayState.roiPassingSegments = passingSegments;
    displayState.roiFilterActive = active;
    displayState.roiGhostAlpha = ghostAlpha;
    displayState.roiGroups = groups;
    displayState.roiHighDetailSegments = highDetailSegments;
    // The Export tab reaches the (later-created) tract render layer through this
    // callback. Lazy: the render layer does not exist yet, so the lookup runs at
    // call time. Set on the real display state (not the per-activation spread the
    // render layers receive), which is the one the tab holds.
    displayState.computeRoiExportIds = (roiGroups) =>
      this.computeRoiExportIds(roiGroups);

    // Colour-by-group: the worker fills `segmentColors` (id -> packed group
    // colour) for passing tracts, which a dedicated ROI colour shader tier reads
    // DIRECTLY to override the streamline's directional RGB. It deliberately
    // does NOT touch the user-facing `segmentStatedColors` map (reusing that
    // clobbers manual segment colours and bakes the materialised colour set into
    // the URL). `colorByGroup` drives the tier's on/off shader uniform.
    const segmentColors = this.registerDisposer(
      Uint64Map.makeWithCounterpart(rpc),
    );
    const colorByGroup = new WatchableValue<boolean>(roiFilter.colorByGroup);
    this.registerDisposer(
      roiFilter.changed.add(() => {
        colorByGroup.value = roiFilter.colorByGroup;
      }),
    );
    displayState.roiSegmentColors = segmentColors;
    displayState.roiColorByGroup = colorByGroup;

    // Per-object numeric attributes (length, …) for the worker's length filter
    // and object-attribute colouring. Rebuilt whenever the segment-property map
    // changes (e.g. a group switch or the store finishing its load).
    const objectAttrColumns = new WatchableValue<
      ReadonlyMap<string, RoiObjectAttrColumn>
    >(buildObjectAttrColumns(displayState.segmentPropertyMap.value));
    this.registerDisposer(
      displayState.segmentPropertyMap.changed.add(() => {
        objectAttrColumns.value = buildObjectAttrColumns(
          displayState.segmentPropertyMap.value,
        );
      }),
    );
    displayState.roiObjectAttrColumns = objectAttrColumns;

    // Dense anatomical label grid, built from the linked parcellation layer
    // ({@link roiLabelLayer}) and shipped to the worker for `labelMask` ROIs.
    // Rebuilt whenever the reference changes or the parcellation finishes
    // loading; an in-flight build is aborted so a rapid re-link cannot land a
    // stale grid. Kept undefined (label ROIs then select nothing) until ready.
    const roiLabelField = new WatchableValue<RoiLabelField | undefined>(
      undefined,
    );
    displayState.roiLabelField = roiLabelField;
    let labelFieldBuild: AbortController | undefined;
    let lastLabelLayerName: string | undefined;
    const rebuildLabelField = (force = false) => {
      const ref = displayState.roiLabelLayer;
      const managed = ref.layer;
      const parcellation = managed?.layer;
      if (
        !(parcellation instanceof SegmentationUserLayer) ||
        parcellation === this ||
        managed === undefined
      ) {
        labelFieldBuild?.abort();
        labelFieldBuild = undefined;
        lastLabelLayerName = undefined;
        if (roiLabelField.value !== undefined) roiLabelField.value = undefined;
        return;
      }
      // The layersChanged signal fires often; only rebuild when the referenced
      // parcellation actually changed, or it just became ready (force), or we
      // have not built for it yet.
      if (!force && managed.name === lastLabelLayerName) return;
      if (!managed.isReady()) return;
      lastLabelLayerName = managed.name;
      labelFieldBuild?.abort();
      const abort = (labelFieldBuild = new AbortController());
      const globalNames = this.manager.root.coordinateSpace.value.names ?? [];
      buildRoiLabelField(parcellation, globalNames, { signal: abort.signal })
        .then((field) => {
          if (!abort.signal.aborted) roiLabelField.value = field;
        })
        .catch((e) => {
          if (!abort.signal.aborted) {
            console.error(
              "ROI label filter: parcellation grid build failed",
              e,
            );
          }
        });
    };
    this.registerDisposer(
      displayState.roiLabelLayer.changed.add(() => {
        // A fresh reference: forget the last-built name so a re-link to a
        // now-ready layer rebuilds even if the name coincides.
        lastLabelLayerName = undefined;
        rebuildLabelField();
      }),
    );
    // Catch the parcellation transitioning to ready after the reference was set
    // (e.g. restored from the URL before its data loaded).
    this.registerDisposer(
      this.manager.rootLayers.layersChanged.add(() => rebuildLabelField(true)),
    );
    this.registerDisposer(() => labelFieldBuild?.abort());
    rebuildLabelField();

    // Background (whole-tractogram) length filter + flat colour-by-attribute: a
    // frontend-only per-object value map (id -> packed normalised values) read
    // directly by the shader, plus the resolved uniforms.
    //
    // ID-space caveat (shared with buildObjectAttrColumns): the keys are the
    // segment-property map's ids (dense object index). For a store with
    // `object_index_convention: "identity"` they equal the streamline segment
    // ids the shader looks up; a `"standard"` store would need re-keying through
    // `object_attributes/segment_id` first, else the tier silently no-ops.
    const roiObjectValues = this.registerDisposer(new Uint64Map());
    const roiBackground = new WatchableValue<RoiBackgroundUniforms | undefined>(
      undefined,
    );
    // Each packed value holds TWO normalised attributes: the length-filter
    // attribute in the low 16 bits and the colour attribute in the high 16, so a
    // length filter on one attribute and colour-by another coexist in one map.
    let lastKey = "";
    let lastPropMap: PreprocessedSegmentPropertyMap | undefined;
    const enc16 = (v: number, min: number, span: number) => {
      const t = span > 0 ? (Number(v) - min) / span : 0;
      return Math.max(0, Math.min(65535, Math.round(t * 65535)));
    };
    const norm01 = (v: number, min: number, span: number) =>
      span > 0 ? Math.max(0, Math.min(1, (v - min) / span)) : 0;
    const refreshBackground = () => {
      const propMap = displayState.segmentPropertyMap.value;
      const bg = roiFilter.backgroundColorBy;
      const lf = roiFilter.backgroundLengthFilter;
      const num = propMap?.numericalProperties ?? [];
      const lengthProp =
        lf !== undefined ? num.find((p) => p.id === lf.name) : undefined;
      const colorProp =
        bg.kind === "objectAttr"
          ? num.find((p) => p.id === bg.name)
          : undefined;
      if (lengthProp === undefined && colorProp === undefined) {
        lastKey = "";
        lastPropMap = undefined;
        if (roiObjectValues.size !== 0) roiObjectValues.clear();
        roiBackground.value = undefined;
        return;
      }
      const lMin = lengthProp !== undefined ? Number(lengthProp.bounds[0]) : 0;
      const lMax = lengthProp !== undefined ? Number(lengthProp.bounds[1]) : 1;
      const cMin = colorProp !== undefined ? Number(colorProp.bounds[0]) : 0;
      const cMax = colorProp !== undefined ? Number(colorProp.bounds[1]) : 1;
      const key = `${lengthProp?.id ?? ""}:${lMin}:${lMax}|${colorProp?.id ?? ""}:${cMin}:${cMax}`;
      // Rebuild the O(objects) map only when an attribute or its bounds change,
      // or the property map object itself was reloaded (a source swap gives a
      // fresh ids array even if name+bounds coincide). A range/mode tweak skips
      // the rebuild and only updates the cheap uniforms below.
      if (key !== lastKey || propMap !== lastPropMap) {
        lastKey = key;
        lastPropMap = propMap;
        roiObjectValues.clear();
        const ids = propMap!.segmentPropertyMap.inlineProperties!.ids;
        const lVals = lengthProp?.values as ArrayLike<number> | undefined;
        const cVals = colorProp?.values as ArrayLike<number> | undefined;
        const lSpan = lMax - lMin;
        const cSpan = cMax - cMin;
        for (let i = 0; i < ids.length; ++i) {
          const lo16 = lVals !== undefined ? enc16(lVals[i], lMin, lSpan) : 0;
          const hi16 = cVals !== undefined ? enc16(cVals[i], cMin, cSpan) : 0;
          // BigInt shifts (not `<<`) — `65535 << 16` overflows JS's 32-bit
          // signed int and would set the value negative.
          roiObjectValues.set_(ids[i], BigInt(lo16) | (BigInt(hi16) << 16n));
        }
        // One coalesced change signal instead of one per object.
        roiObjectValues.changed.dispatch(null, true);
      }
      roiBackground.value = {
        lengthActive: lf !== undefined && lengthProp !== undefined,
        lo: lf !== undefined ? norm01(lf.min, lMin, lMax - lMin) : 0,
        hi: lf !== undefined ? norm01(lf.max, lMin, lMax - lMin) : 1,
        colorMode: bg.kind === "objectAttr" && colorProp !== undefined,
      };
    };
    refreshBackground();
    this.registerDisposer(roiFilter.changed.add(refreshBackground));
    this.registerDisposer(
      displayState.segmentPropertyMap.changed.add(refreshBackground),
    );
    displayState.roiObjectValues = roiObjectValues;
    displayState.roiBackground = roiBackground;
  }

  /**
   * Draw the ROI regions as annotation overlays on the tract subsource: box /
   * plane ROIs as a coloured wireframe box, sphere ROIs as a coloured fill, each
   * in its group's colour. Attached via the annotation mixin, so it renders in
   * both the 2-d and 3-d views. The overlays mirror {@link RoiFilterState} and
   * are read-only (placement/editing is via the Filter tab's sliders).
   */
  private addRoiOverlays(loadedSubsource: LoadedDataSubsource) {
    const refCounted = loadedSubsource.activated;
    if (refCounted === undefined) return;
    const roiFilter = this.displayState.roiFilter;
    const properties = new WatchableValue<AnnotationPropertySpec[]>([
      {
        identifier: "color",
        type: "rgb",
        default: packColor(vec3.fromValues(1, 1, 0)),
        description: undefined,
      },
    ]);
    const source = new LocalAnnotationSource(
      loadedSubsource.loadedDataSource.transform,
      properties,
      [],
    );
    this.addLocalAnnotations(
      loadedSubsource,
      source,
      RenderLayerRole.DEFAULT_ANNOTATION,
    );
    // The overlay colour/hide-2d shader lives on the layer-shared annotation
    // display state; restore whatever was there when the tract source goes away
    // so it does not linger onto any other annotations the layer might gain.
    const previousShader = this.annotationDisplayState.shader.value;
    refCounted.registerDisposer(() => {
      this.annotationDisplayState.shader.value = previousShader;
    });
    const refs: AnnotationReference[] = [];
    const sync = () => {
      this.annotationDisplayState.shader.value = roiFilter.hideOverlays2d
        ? ROI_OVERLAY_SHADER_HIDE_2D
        : ROI_OVERLAY_SHADER;
      rebuildRoiAnnotations(source, roiFilter, refs);
    };
    refCounted.registerDisposer(roiFilter.changed.add(sync));
    sync();
  }

  activateDataSubsources(subsources: Iterable<LoadedDataSubsource>) {
    const updatedSegmentPropertyMaps: SegmentPropertyMap[] = [];
    const isGroupRoot =
      this.displayState.linkedSegmentationGroup.root.value === this;
    let updatedGraph: SegmentationGraphSource | undefined;
    let hasVolume = false;
    let spatialSkeletonGridSizes: SpatialSkeletonGridSize[] | undefined;
    let spatialSkeletonLevelCostsBytes: number[] | undefined;
    let spatialSkeletonLevelObjectCounts: (number | undefined)[] | undefined;
    let spatialSkeletonLevelCellCounts: number[] | undefined;
    let spatialSkeletonBudgetBytes: number | undefined;
    // A datasource-preferred default shader, and whether any subsource would be
    // One entry per skeleton subsource: the shader it nominates as the layer
    // default, or `undefined` for no opinion. Resolved after the loop, once
    // every subsource has voted -- see `resolveSkeletonDefaultShader`.
    const skeletonShaderCandidates: (string | undefined)[] = [];
    for (const loadedSubsource of subsources) {
      if (this.addStaticAnnotations(loadedSubsource)) continue;
      const {
        volume,
        mesh,
        zarrVectors,
        segmentPropertyMap,
        segmentationGraph,
        local,
      } = loadedSubsource.subsourceEntry.subsource;
      // The two slots are distinct in the data model -- a zarr-vectors store is
      // not a `MeshSource` -- but they resolve to the same render layers here,
      // chosen by source class below. Binding them together keeps one activation
      // path rather than two copies that would drift apart.
      const geometry = mesh ?? zarrVectors;
      if (volume instanceof MultiscaleVolumeChunkSource) {
        switch (volume.dataType) {
          case DataType.FLOAT32:
            loadedSubsource.deactivate(
              "Data type not compatible with segmentation layer",
            );
            continue;
        }
        hasVolume = true;
        loadedSubsource.activate(
          () =>
            loadedSubsource.addRenderLayer(
              new SegmentationRenderLayer(volume, {
                ...this.displayState,
                transform: loadedSubsource.getRenderLayerTransform(),
                renderScaleTarget: this.sliceViewRenderScaleTarget,
                renderScaleHistogram: this.sliceViewRenderScaleHistogram,
                localPosition: this.localPosition,
              }),
            ),
          this.displayState.segmentationGroupState.value,
        );
      } else if (geometry !== undefined) {
        if (geometry instanceof MultiscaleSpatiallyIndexedSkeletonSource) {
          // Collect grid metadata outside `activate`, since `activate` is a no-op
          // when guard values are unchanged and may skip the callback.
          // Compose the live render-layer transform (reflects any output
          // CoordinateSpaceTransform the user has applied, e.g. correcting
          // a source's declared voxel size) with the global coordinate
          // space's scales, so level sizes stay unit-consistent with the
          // camera-driven resolution target even after such an edit.
          //
          // Use the plain (non-"Watchable") `getRenderLayerTransform` here,
          // not `loadedSubsource.getRenderLayerTransform()` — that method
          // requires `loadedSubsource.activated`, which this code runs
          // BEFORE (`activate()` below is what sets it), and would throw.
          let liveScale: Float64Array | undefined;
          const { layer, transform } = loadedSubsource.loadedDataSource;
          const transformValue = getRenderLayerTransform(
            layer.manager.root.coordinateSpace.value,
            layer.localPosition.coordinateSpace.value,
            transform.value,
            loadedSubsource,
          );
          if (transformValue.error === undefined) {
            liveScale = computeDiagonalModelToGlobalMetersScale(
              transformValue,
              this.manager.root.coordinateSpace.value.scales,
            );
          }
          spatialSkeletonGridSizes =
            geometry.getSpatialSkeletonGridSizes(liveScale);
          // A source that can estimate what each level costs opts into
          // budget-driven selection; see `setSpatialSkeletonGridSizes`.
          const costs = (
            geometry as {
              getSpatialSkeletonLevelCostsBytes?: () => number[];
            }
          ).getSpatialSkeletonLevelCostsBytes?.();
          if (costs !== undefined) {
            spatialSkeletonLevelCostsBytes = costs;
            // The whole GPU pool. The object-keyed pass draws from the same
            // one, but it is sized from what the level chosen here LEAVES (see
            // `refreshSpatialSkeletonObjectFill`), so the two cannot outbid
            // each other and no share needs reserving up front.
            spatialSkeletonBudgetBytes =
              this.manager.chunkManager.chunkQueueManager.capacities.gpuMemory
                .sizeLimit.value;
          }
          // Objects per level, when the source can say. Sizes the resolution
          // histogram's bars by how many streamlines each level holds, which is
          // what a user means by "how big is this level".
          spatialSkeletonLevelObjectCounts = (
            geometry as {
              getSpatialSkeletonLevelObjectCounts?: () => (
                | number
                | undefined
              )[];
            }
          ).getSpatialSkeletonLevelObjectCounts?.();
          // Cells per level, so a whole-level cost becomes a per-cell one.
          spatialSkeletonLevelCellCounts = (
            geometry as {
              getSpatialSkeletonLevelCellCounts?: () => number[];
            }
          ).getSpatialSkeletonLevelCellCounts?.();
          // How this store answers "which whole objects fit in N bytes". Only a
          // source with per-level object membership can; the rest keep
          // whole-level selection.
          const objectSource = geometry as {
            computeObjectAdmission?: (b: number) => ObjectAdmission | undefined;
            canBudgetPerObject?: boolean;
          };
          // Gate on the source's actual CAPABILITY, not on the method existing:
          // a store missing per-level object membership has the method but
          // always answers `undefined`, and installing the closure anyway
          // suppresses whole-level selection without providing a per-object
          // replacement. `=== false` so a source that does not declare the
          // capability at all keeps the previous behaviour.
          this.displayState.spatialSkeletonComputeAdmission =
            objectSource.computeObjectAdmission === undefined ||
            objectSource.canBudgetPerObject === false
              ? undefined
              : (budgetBytes: number) =>
                  objectSource.computeObjectAdmission!(budgetBytes);
          // ...and where it cannot, object focus stops being the DEFAULT.
          //
          // It remains selectable, and doing so gets the half of it that needs
          // no per-object membership. But defaulting to it is wrong on a
          // resolution pyramid: sizing "one level everywhere, whole volume
          // resident" against the whole-level ceiling assumes the layer owns
          // the GPU budget, and the mesh/point/skeleton stores this applies to
          // are exactly the ones loaded three-at-a-time over one volume. See
          // {@link SegmentationUserLayerDisplayState.spatialSkeletonDetailFocusExplicit}.
          if (
            this.displayState.spatialSkeletonComputeAdmission === undefined &&
            !this.displayState.spatialSkeletonDetailFocusExplicit
          ) {
            this.displayState.applyDefaultSpatialSkeletonDetailFocus(
              SpatialSkeletonDetailFocus.LOCAL,
            );
          }
          skeletonShaderCandidates.push(geometry.defaultFragmentMain);
        } else if (
          geometry !== undefined &&
          !(
            geometry instanceof MeshSource ||
            geometry instanceof MultiscaleMeshSource
          )
        ) {
          // Anything else in the `geometry` slot that is not a geometry is drawn by the
          // plain `SkeletonLayer` and shares this layer's skeleton shader, so
          // it gets a vote. Meshes have their own shader and are not consulted.
          skeletonShaderCandidates.push(
            (geometry as { defaultFragmentMain?: string }).defaultFragmentMain,
          );
        }
        loadedSubsource.activate(() => {
          // A tract source that opts into the ROI streamline filter enables the
          // data channel. Gate on the source capability, NOT the shared
          // spatially-indexed skeleton base class: other datasources (e.g.
          // CATMAID) use the same base but emit no per-vertex segment column, so
          // their passing set could never be populated and the filter would
          // ghost every streamline. Create the channel on the real display state
          // *before* the spread below copies it into the per-activation display
          // state the render layers receive (which is what lights up the shader).
          if (
            (geometry as { supportsRoiStreamlineFilter?: boolean })
              .supportsRoiStreamlineFilter === true
          ) {
            this.ensureRoiFilterChannel();
            this.addRoiOverlays(loadedSubsource);
            // What the Filter tab can offer for THIS store: its loaded
            // per-vertex attribute columns (the only filterable tier a point
            // cloud has), and whether the tract export applies at all.
            const filterable = geometry as {
              vertexAttributeNames?: readonly string[];
              vertexAttributeDtypes?: readonly string[];
              supportsTractExport?: boolean;
              geometryPrimitive?: "points" | "lines" | "triangles";
            };
            this.displayState.roiVertexAttributeNames =
              filterable.vertexAttributeNames;
            this.displayState.roiVertexAttributeDtypes =
              filterable.vertexAttributeDtypes;
            this.displayState.roiSupportsTractExport =
              filterable.supportsTractExport === true;
            this.displayState.roiGeometryPrimitive =
              filterable.geometryPrimitive;
            this.displayState.computeRoiVertexAttrStats = (names) =>
              this.computeRoiVertexAttrStats(names);
          }
          const displayState = {
            ...this.displayState,
            transform: loadedSubsource.getRenderLayerTransform(),
            localPosition: this.localPosition,
          };
          if (geometry instanceof MeshSource) {
            loadedSubsource.addRenderLayer(
              new MeshLayer(this.manager.chunkManager, geometry, displayState),
            );
          } else if (geometry instanceof MultiscaleMeshSource) {
            loadedSubsource.addRenderLayer(
              new MultiscaleMeshLayer(
                this.manager.chunkManager,
                geometry,
                displayState,
              ),
            );
          } else if (
            geometry instanceof MultiscaleSpatiallyIndexedSkeletonSource
          ) {
            const perspectiveSources = geometry.getPerspectiveSources();
            const slicePanelSources = geometry.getSliceViewPanelSources();
            const sharedSpatialSkeletonSources =
              perspectiveSources.length > 0
                ? perspectiveSources
                : slicePanelSources;
            // Honour the source's auto-LOD preference: data sources that
            // emit several pyramid levels and want camera-driven level
            // switching (e.g. zarr-vectors) opt in here.  CATMAID
            // leaves it false, preserving manual-slider UX.
            if (geometry.prefersAutoSpatialSkeletonGridLevel) {
              this.displayState.autoSpatialSkeletonGridLevel3d.value = true;
              this.displayState.autoSpatialSkeletonGridLevel2d.value = true;
            }
            if (sharedSpatialSkeletonSources.length > 0) {
              // Share one mutable skeleton base across 2D/3D projections so
              // local edit state stays consistent across panels.
              const base = new SpatiallyIndexedSkeletonLayer(
                this.manager.chunkManager,
                sharedSpatialSkeletonSources,
                displayState,
                {
                  gridLevel: displayState.spatialSkeletonGridLevel3d,
                  lod: displayState.skeletonLod,
                  gridLevel2d: displayState.spatialSkeletonGridLevel2d,
                  lod2d: displayState.spatialSkeletonLod2d,
                  detailFocus: displayState.spatialSkeletonDetailFocus,
                  admissionFraction:
                    displayState.spatialSkeletonAdmissionFraction,
                  sources2d: slicePanelSources,
                  selectedNodeId: this.selectedSpatialSkeletonNodeId,
                  pendingNodePositionVersion:
                    this.spatialSkeletonState.pendingNodePositionVersion,
                  getPendingNodePosition: (nodeId) =>
                    this.spatialSkeletonState.getPendingNodePosition(nodeId),
                  getCachedNode: (nodeId) =>
                    this.spatialSkeletonState.getCachedNode(nodeId),
                  inspectionState: this.spatialSkeletonState,
                },
              );
              if (perspectiveSources.length > 0) {
                loadedSubsource.addRenderLayer(
                  new PerspectiveViewSpatiallyIndexedSkeletonLayer(
                    base.addRef(),
                  ),
                );
              }
              if (slicePanelSources.length > 0) {
                loadedSubsource.addRenderLayer(
                  new SliceViewPanelSpatiallyIndexedSkeletonLayer(
                    /* transfer ownership */ base,
                  ),
                );
              } else {
                base.dispose();
              }
            }
          } else if (geometry instanceof SpatiallyIndexedSkeletonSource) {
            const base = new SpatiallyIndexedSkeletonLayer(
              this.manager.chunkManager,
              geometry,
              displayState,
              {
                gridLevel: displayState.spatialSkeletonGridLevel3d,
                lod: displayState.skeletonLod,
                gridLevel2d: displayState.spatialSkeletonGridLevel2d,
                lod2d: displayState.spatialSkeletonLod2d,
                selectedNodeId: this.selectedSpatialSkeletonNodeId,
                pendingNodePositionVersion:
                  this.spatialSkeletonState.pendingNodePositionVersion,
                getPendingNodePosition: (nodeId) =>
                  this.spatialSkeletonState.getPendingNodePosition(nodeId),
                getCachedNode: (nodeId) =>
                  this.spatialSkeletonState.getCachedNode(nodeId),
                inspectionState: this.spatialSkeletonState,
              },
            );
            loadedSubsource.addRenderLayer(
              new PerspectiveViewSpatiallyIndexedSkeletonLayer(base.addRef()),
            );
            loadedSubsource.addRenderLayer(
              new SliceViewPanelSpatiallyIndexedSkeletonLayer(
                /* transfer ownership */ base,
              ),
            );
          } else {
            // The zarr-vectors pass-2 source is the ROI filter's full-detail
            // render layer: give it a DEDICATED visible set
            // (roiHighDetailSegments) via a proxy group state, so it draws only
            // the high-detail groups' tracts and never touches the user's
            // selection. Consumers of the group state read fields
            // (the 6 shared visible-segment objects, hideSegmentZero, …), never
            // methods, so a spread proxy is safe; colouring uses the separate
            // segmentationColorGroupState, which is unchanged.
            let skeletonDisplayState = displayState;
            const highDetail = displayState.roiHighDetailSegments;
            if (
              highDetail !== undefined &&
              (geometry as { isRoiHighDetailSource?: boolean })
                .isRoiHighDetailSource === true
            ) {
              const realGroupState =
                this.displayState.segmentationGroupState.value;
              // Structural proxy: all fields of the real group state, but with
              // `visibleSegments` swapped. Cast back to the class type — the
              // consumers only read the (present) fields, never call methods.
              const proxyGroupState = {
                ...realGroupState,
                visibleSegments: highDetail,
              } as unknown as SegmentationUserLayerGroupState;
              skeletonDisplayState = {
                ...displayState,
                segmentationGroupState: new WatchableValue(proxyGroupState),
              };
              // Expose this pass-2 source so the Export tab can read whole tracts'
              // geometry straight from its resident chunks (the fast in-browser
              // TRK path) instead of re-reading the store. Set on the real display
              // state (the one the tab holds), not the spread above.
              this.displayState.roiHighDetailSkeletonSource =
                geometry as SkeletonSource;
            }
            const base = new SkeletonLayer(
              this.manager.chunkManager,
              geometry,
              skeletonDisplayState,
            );
            loadedSubsource.addRenderLayer(
              new PerspectiveViewSkeletonLayer(base.addRef()),
            );
            loadedSubsource.addRenderLayer(
              new SliceViewPanelSkeletonLayer(/* transfer ownership */ base),
            );
          }
        }, this.displayState.segmentationGroupState.value);
      } else if (segmentPropertyMap !== undefined) {
        if (!isGroupRoot) {
          loadedSubsource.deactivate(
            "Not supported on non-root linked segmentation layers",
          );
        } else {
          loadedSubsource.activate(() => {});
          updatedSegmentPropertyMaps.push(segmentPropertyMap);
        }
      } else if (segmentationGraph !== undefined) {
        if (!isGroupRoot) {
          loadedSubsource.deactivate(
            "Not supported on non-root linked segmentation layers",
          );
        } else {
          if (updatedGraph !== undefined) {
            loadedSubsource.deactivate(
              "Only one segmentation graph is supported",
            );
          } else {
            updatedGraph = segmentationGraph;
            loadedSubsource.activate((refCounted) => {
              const graphConnection = segmentationGraph.connect(this);
              refCounted.registerDisposer(() => {
                graphConnection.dispose();
                this.graphConnection.value = undefined;
              });
              const displayState = {
                ...this.displayState,
                transform: loadedSubsource.getRenderLayerTransform(),
              };

              const graphRenderLayers = graphConnection.createRenderLayers(
                this.manager.chunkManager,
                displayState,
                this.localPosition,
              );
              this.graphConnection.value = graphConnection;
              for (const renderLayer of graphRenderLayers) {
                loadedSubsource.addRenderLayer(renderLayer);
              }
            });
          }
        }
      } else if (local === LocalDataSource.equivalences) {
        if (!isGroupRoot) {
          loadedSubsource.deactivate(
            "Not supported on non-root linked segmentation layers",
          );
        } else {
          if (updatedGraph !== undefined) {
            loadedSubsource.deactivate(
              "Only one segmentation graph is supported",
            );
          } else {
            updatedGraph =
              this.displayState.originalSegmentationGroupState.localGraph;
            loadedSubsource.activate((refCounted) => {
              this.graphConnection.value = refCounted.registerDisposer(
                updatedGraph!.connect(this),
              );
              refCounted.registerDisposer(() => {
                this.graphConnection.value = undefined;
              });
            });
          }
        }
      } else {
        loadedSubsource.deactivate("Not compatible with segmentation layer");
      }
    }
    this.displayState.originalSegmentationGroupState.segmentPropertyMap.value =
      getPreprocessedSegmentPropertyMap(
        this.manager.chunkManager,
        updatedSegmentPropertyMaps,
      );
    this.displayState.originalSegmentationGroupState.graph.value = updatedGraph;
    this.applySkeletonDefaultShader(
      resolveSkeletonDefaultShader(skeletonShaderCandidates),
    );
    this.displayState.setSpatialSkeletonGridSizes(
      spatialSkeletonGridSizes ?? [],
      spatialSkeletonLevelCostsBytes,
      spatialSkeletonBudgetBytes,
      spatialSkeletonLevelObjectCounts,
      spatialSkeletonLevelCellCounts,
    );
    this.displayState.hasVolume.value = hasVolume;
    this.updateSpatialSkeletonChunkLoadState();
  }

  /**
   * Adopt a datasource's preferred skeleton shader as the layer's *default*.
   *
   * Applied as the default rather than as a value, for two reasons that both
   * hinge on `TrackableValue.toJSON()` emitting only when
   * `value !== defaultValue`:
   *
   *  - Setting only `value` would make the shader text serialise into every
   *    saved link, and on reload it would come back as an *explicit user
   *    shader* -- permanently pinning the layer to whatever the default
   *    happened to be that day, and defeating any later improvement to it.
   *  - Moving `defaultValue` too keeps `value === defaultValue`, so the state
   *    stays clean, and "Reset" and a shader-less restore both land on the
   *    datasource's shader rather than back on `emitDefault()`.
   *
   * A user's own shader still wins: the layer spec is restored synchronously,
   * while this runs later from `activateDataSubsources` once the datasource has
   * resolved, so `value` has already diverged from `defaultValue` and only the
   * default moves (verified against a link carrying an explicit shader). The
   * same guard makes re-activation non-clobbering.
   *
   * `sourceShader` is undefined when the layer's skeleton subsources have no
   * agreed nomination -- no skeleton subsource, an abstaining one (CATMAID), or
   * two that DISAGREE. In that case revert to the generic segment-coloured
   * default ({@link DEFAULT_FRAGMENT_MAIN}) rather than leaving a previously
   * installed datasource shader in place. The retract is load-bearing: subsources
   * activate incrementally (a source whose `loadState` is still pending is
   * skipped and this re-runs when it resolves), so a tangent-bearing tract source
   * can activate ALONE first and install its `prop_tangent()` default, and then a
   * no-tangent skeleton subsource loads and forces disagreement. Since the whole
   * layer shares one `skeletonRenderingOptions.shader`, a stuck `prop_tangent()`
   * default would fail to compile against the no-tangent subsource (blank tracts)
   * -- so `undefined` must actively pull the default back to the generic one that
   * compiles for every subsource. `defaultValue` is a plain field (no dispatch)
   * and the `value` setter is change-guarded, so the common no-skeleton case
   * (target already generic) is a true no-op.
   */
  private applySkeletonDefaultShader(sourceShader: string | undefined) {
    const target = sourceShader ?? DEFAULT_FRAGMENT_MAIN;
    const { shader } = this.displayState.skeletonRenderingOptions;
    const untouched = shader.value === shader.defaultValue;
    shader.defaultValue = target;
    if (untouched) shader.value = target;
  }

  getLegacyDataSourceSpecifications(
    sourceSpec: any,
    layerSpec: any,
    legacyTransform: CoordinateTransformSpecification | undefined,
    explicitSpecs: DataSourceSpecification[],
  ): DataSourceSpecification[] {
    const specs = super.getLegacyDataSourceSpecifications(
      sourceSpec,
      layerSpec,
      legacyTransform,
      explicitSpecs,
    );
    const meshPath = verifyOptionalObjectProperty(
      layerSpec,
      json_keys.MESH_JSON_KEY,
      (x) => (x === null ? null : verifyString(x)),
    );
    const skeletonsPath = verifyOptionalObjectProperty(
      layerSpec,
      json_keys.SKELETONS_JSON_KEY,
      (x) => (x === null ? null : verifyString(x)),
    );
    if (meshPath !== undefined || skeletonsPath !== undefined) {
      for (const spec of specs) {
        spec.enableDefaultSubsources = false;
        spec.subsources = new Map([
          ["default", { enabled: true }],
          ["bounds", { enabled: true }],
        ]);
      }
    }
    if (meshPath != null) {
      specs.push(
        layerDataSourceSpecificationFromJson(
          this.manager.dataSourceProviderRegistry.convertLegacyUrl({
            url: meshPath,
            type: "mesh",
          }),
        ),
      );
    }
    if (skeletonsPath != null) {
      specs.push(
        layerDataSourceSpecificationFromJson(
          this.manager.dataSourceProviderRegistry.convertLegacyUrl({
            url: skeletonsPath,
            type: "skeletons",
          }),
        ),
      );
    }
    if (
      layerSpec[json_keys.EQUIVALENCES_JSON_KEY] !== undefined &&
      explicitSpecs.find((spec) => spec.url === localEquivalencesUrl) ===
        undefined
    ) {
      specs.push({
        url: localEquivalencesUrl,
        enableDefaultSubsources: true,
        transform: {
          outputSpace: emptyValidCoordinateSpace,
          sourceRank: 0,
          transform: undefined,
          inputSpace: emptyValidCoordinateSpace,
        },
        subsources: new Map(),
      });
    }
    return specs;
  }

  restoreState(specification: any) {
    super.restoreState(specification);
    this.displayState.selectedAlpha.restoreState(
      specification[json_keys.SELECTED_ALPHA_JSON_KEY],
    );
    this.displayState.saturation.restoreState(
      specification[json_keys.SATURATION_JSON_KEY],
    );
    this.displayState.notSelectedAlpha.restoreState(
      specification[json_keys.NOT_SELECTED_ALPHA_JSON_KEY],
    );
    this.displayState.hoverHighlight.restoreState(
      specification[json_keys.HOVER_HIGHLIGHT_JSON_KEY],
    );
    this.displayState.objectAlpha.restoreState(
      specification[json_keys.OBJECT_ALPHA_JSON_KEY],
    );
    this.displayState.hiddenObjectAlpha.restoreState(
      specification[json_keys.HIDDEN_OPACITY_3D_JSON_KEY],
    );
    this.displayState.spatialSkeletonNodeQuery.restoreState(
      specification[json_keys.SPATIAL_SKELETON_NODE_QUERY_JSON_KEY],
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.SPATIAL_SKELETON_NODE_FILTER_JSON_KEY,
      (value) =>
        this.displayState.spatialSkeletonNodeFilter.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.ROI_FILTER_JSON_KEY,
      (value) => this.displayState.roiFilter.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.ROI_LABEL_LAYER_JSON_KEY,
      (value) => this.displayState.roiLabelLayer.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.IGNORE_SKELETON_MEMORY_CEILING_JSON_KEY,
      (value) =>
        this.displayState.ignoreSpatialSkeletonMemoryCeiling.restoreState(
          value,
        ),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.SPATIAL_SKELETON_DETAIL_FOCUS_JSON_KEY,
      (value) => {
        this.displayState.spatialSkeletonDetailFocus.restoreState(value);
        // A focus in the JSON is a choice, and outlives whatever the source
        // turns out to support.
        this.displayState.spatialSkeletonDetailFocusExplicit = true;
      },
    );
    this.displayState.spatialSkeletonGridResolutionTarget2d.restoreState(
      specification[json_keys.SKELETON_CROSS_SECTION_RENDER_SCALE_JSON_KEY],
    );
    this.displayState.spatialSkeletonGridResolutionTarget3d.restoreState(
      specification[json_keys.SKELETON_PERSPECTIVE_RENDER_SCALE_JSON_KEY],
    );
    this.displayState.spatialSkeletonGridResolutionBias2d.restoreState(
      specification[json_keys.SPATIAL_SKELETON_GRID_BIAS_2D_JSON_KEY],
    );
    this.displayState.spatialSkeletonGridResolutionBias3d.restoreState(
      specification[json_keys.SPATIAL_SKELETON_GRID_BIAS_3D_JSON_KEY],
    );
    this.displayState.baseSegmentColoring.restoreState(
      specification[json_keys.BASE_SEGMENT_COLORING_JSON_KEY],
    );
    this.displayState.silhouetteRendering.restoreState(
      specification[json_keys.MESH_SILHOUETTE_RENDERING_JSON_KEY],
    );
    this.displayState.ignoreNullVisibleSet.restoreState(
      specification[json_keys.IGNORE_NULL_VISIBLE_SET_JSON_KEY],
    );

    const { skeletonRenderingOptions } = this.displayState;
    skeletonRenderingOptions.restoreState(
      specification[json_keys.SKELETON_RENDERING_JSON_KEY],
    );
    const skeletonShader = specification[json_keys.SKELETON_SHADER_JSON_KEY];
    if (skeletonShader !== undefined) {
      skeletonRenderingOptions.shader.restoreState(skeletonShader);
    }
    this.codeVisible.restoreState(json_keys.SKELETON_CODE_VISIBLE_KEY);
    this.displayState.renderScaleTarget.restoreState(
      specification[json_keys.MESH_RENDER_SCALE_JSON_KEY],
    );
    this.anchorSegment.restoreState(
      specification[json_keys.ANCHOR_SEGMENT_JSON_KEY],
    );
    this.sliceViewRenderScaleTarget.restoreState(
      specification[json_keys.CROSS_SECTION_RENDER_SCALE_JSON_KEY],
    );
    const linkedSegmentationGroupName = verifyOptionalObjectProperty(
      specification,
      json_keys.LINKED_SEGMENTATION_GROUP_JSON_KEY,
      verifyString,
    );
    if (linkedSegmentationGroupName !== undefined) {
      this.displayState.linkedSegmentationGroup.linkByName(
        linkedSegmentationGroupName,
      );
    }
    const linkedSegmentationColorGroupName = verifyOptionalObjectProperty(
      specification,
      json_keys.LINKED_SEGMENTATION_COLOR_GROUP_JSON_KEY,
      (x) => (x === false ? undefined : verifyString(x)),
      linkedSegmentationGroupName,
    );
    if (linkedSegmentationColorGroupName !== undefined) {
      this.displayState.linkedSegmentationColorGroup.linkByName(
        linkedSegmentationColorGroupName,
      );
    }
    this.displayState.segmentationGroupState.value.restoreState(specification);
    this.displayState.segmentationColorGroupState.value.restoreState(
      specification,
    );
  }

  toJSON() {
    const x = super.toJSON();
    x[json_keys.SELECTED_ALPHA_JSON_KEY] =
      this.displayState.selectedAlpha.toJSON();
    x[json_keys.NOT_SELECTED_ALPHA_JSON_KEY] =
      this.displayState.notSelectedAlpha.toJSON();
    x[json_keys.SATURATION_JSON_KEY] = this.displayState.saturation.toJSON();
    x[json_keys.OBJECT_ALPHA_JSON_KEY] = this.displayState.objectAlpha.toJSON();
    x[json_keys.SPATIAL_SKELETON_NODE_QUERY_JSON_KEY] =
      this.displayState.spatialSkeletonNodeQuery.toJSON();
    x[json_keys.SPATIAL_SKELETON_NODE_FILTER_JSON_KEY] =
      this.displayState.spatialSkeletonNodeFilter.toJSON();
    x[json_keys.ROI_FILTER_JSON_KEY] = this.displayState.roiFilter.toJSON();
    x[json_keys.ROI_LABEL_LAYER_JSON_KEY] =
      this.displayState.roiLabelLayer.toJSON();
    x[json_keys.IGNORE_SKELETON_MEMORY_CEILING_JSON_KEY] =
      this.displayState.ignoreSpatialSkeletonMemoryCeiling.toJSON();
    x[json_keys.SPATIAL_SKELETON_DETAIL_FOCUS_JSON_KEY] =
      this.displayState.spatialSkeletonDetailFocus.toJSON();
    x[json_keys.HIDDEN_OPACITY_3D_JSON_KEY] =
      this.displayState.hiddenObjectAlpha.toJSON();
    x[json_keys.SKELETON_CROSS_SECTION_RENDER_SCALE_JSON_KEY] =
      this.displayState.spatialSkeletonGridResolutionTarget2d.toJSON();
    x[json_keys.SKELETON_PERSPECTIVE_RENDER_SCALE_JSON_KEY] =
      this.displayState.spatialSkeletonGridResolutionTarget3d.toJSON();
    x[json_keys.SPATIAL_SKELETON_GRID_BIAS_2D_JSON_KEY] =
      this.displayState.spatialSkeletonGridResolutionBias2d.toJSON();
    x[json_keys.SPATIAL_SKELETON_GRID_BIAS_3D_JSON_KEY] =
      this.displayState.spatialSkeletonGridResolutionBias3d.toJSON();
    x[json_keys.HOVER_HIGHLIGHT_JSON_KEY] =
      this.displayState.hoverHighlight.toJSON();
    x[json_keys.BASE_SEGMENT_COLORING_JSON_KEY] =
      this.displayState.baseSegmentColoring.toJSON();
    x[json_keys.IGNORE_NULL_VISIBLE_SET_JSON_KEY] =
      this.displayState.ignoreNullVisibleSet.toJSON();
    x[json_keys.MESH_SILHOUETTE_RENDERING_JSON_KEY] =
      this.displayState.silhouetteRendering.toJSON();
    x[json_keys.ANCHOR_SEGMENT_JSON_KEY] = this.anchorSegment
      .toJSON()
      ?.toString();
    x[json_keys.SKELETON_RENDERING_JSON_KEY] =
      this.displayState.skeletonRenderingOptions.toJSON();
    x[json_keys.SKELETON_CODE_VISIBLE_KEY] = this.codeVisible.toJSON();
    x[json_keys.MESH_RENDER_SCALE_JSON_KEY] =
      this.displayState.renderScaleTarget.toJSON();
    x[json_keys.CROSS_SECTION_RENDER_SCALE_JSON_KEY] =
      this.sliceViewRenderScaleTarget.toJSON();

    const { linkedSegmentationGroup, linkedSegmentationColorGroup } =
      this.displayState;
    x[json_keys.LINKED_SEGMENTATION_GROUP_JSON_KEY] =
      linkedSegmentationGroup.toJSON();
    if (
      linkedSegmentationColorGroup.root.value !==
      linkedSegmentationGroup.root.value
    ) {
      x[json_keys.LINKED_SEGMENTATION_COLOR_GROUP_JSON_KEY] =
        linkedSegmentationColorGroup.toJSON() ?? false;
    }
    x[json_keys.EQUIVALENCES_JSON_KEY] =
      this.displayState.originalSegmentationGroupState.localGraph.toJSON();
    if (linkedSegmentationGroup.root.value === this) {
      Object.assign(x, this.displayState.segmentationGroupState.value.toJSON());
    }
    if (linkedSegmentationColorGroup.root.value === this) {
      Object.assign(
        x,
        this.displayState.segmentationColorGroupState.value.toJSON(),
      );
    }
    return x;
  }

  transformPickedValue(value: any) {
    if (value == null) {
      return value;
    }
    return maybeAugmentSegmentId(this.displayState, value);
  }

  handleAction(action: string, context: SegmentationActionContext) {
    switch (action) {
      case "recolor": {
        this.displayState.segmentationColorGroupState.value.segmentColorHash.randomize();
        break;
      }
      case "clear-segments": {
        if (!this.pick.value) break;
        this.displayState.segmentationGroupState.value.visibleSegments.clear();
        break;
      }
      case "select":
      case "star": {
        if (!this.pick.value) break;
        const { segmentSelectionState } = this.displayState;
        if (segmentSelectionState.hasSelectedSegment) {
          const segment = segmentSelectionState.selectedSegment;
          const group = this.displayState.segmentationGroupState.value;
          const segmentSet =
            action === "select"
              ? group.visibleSegments
              : group.selectedSegments;
          const newValue = !segmentSet.has(segment);
          if (
            newValue ||
            context.segmentationToggleSegmentState === undefined
          ) {
            context.segmentationToggleSegmentState = newValue;
          }
          context.defer(() => {
            if (context.segmentationToggleSegmentState === newValue) {
              segmentSet.set(segment, newValue);
            }
          });
        }
        break;
      }
    }
  }
  selectionStateFromJson(state: this["selectionState"], json: any) {
    super.selectionStateFromJson(state, json);
    let parsedValue = state.value;
    if (typeof parsedValue === "number") parsedValue = parsedValue.toString();
    try {
      state.value = parseUint64(parsedValue);
    } catch {
      state.value = undefined;
    }
  }

  captureSelectionState(
    state: this["selectionState"],
    mouseState: MouseSelectionState,
  ) {
    super.captureSelectionState(state, mouseState);
    const pickedSpatialSkeleton = mouseState.pickedSpatialSkeleton;
    if (pickedSpatialSkeleton === undefined) return;
    const pickedRenderLayer = mouseState.pickedRenderLayer;
    if (
      pickedRenderLayer !== null &&
      !this.renderLayers.includes(pickedRenderLayer)
    ) {
      return;
    }
    const nodeId = normalizeOptionalPositiveSafeInteger(
      pickedSpatialSkeleton.nodeId,
    );
    state.nodeId = nodeId === undefined ? undefined : nodeId.toString();
    // Picked fragment's global segment id, surfaced as the selection value
    // exactly like a picked voxel's segment id.  Prefer the full uint64
    // (`segmentIdU64`); fall back to the safe-integer `segmentId` for
    // callers that only set the numeric field.
    const u64 = pickedSpatialSkeleton.segmentIdU64;
    const segNum = pickedSpatialSkeleton.segmentId;
    if (typeof u64 === "bigint" && u64 > 0n) {
      state.value = u64;
    } else if (
      typeof segNum === "number" &&
      Number.isSafeInteger(segNum) &&
      segNum > 0
    ) {
      state.value = BigInt(segNum);
    }
  }

  selectionStateToJson(state: this["selectionState"], forPython: boolean): any {
    const json = super.selectionStateToJson(state, forPython);
    const { value } = state;
    if (value instanceof Uint64MapEntry) {
      if (forPython) {
        json.value = {
          key: value.key.toString(),
          value: value.value ? value.value.toString() : undefined,
          label: value.label,
        };
      } else {
        json.value = (value.value || value.key).toString();
      }
    } else if (typeof value === "bigint") {
      json.value = value.toString();
    }
    return json;
  }

  private displaySegmentationSelection(
    state: this["selectionState"],
    parent: HTMLElement,
    context: DependentViewContext,
  ): boolean {
    const { value } = state;
    let id: bigint;
    if (typeof value === "number" || typeof value === "string") {
      try {
        id = parseUint64(value);
      } catch {
        return false;
      }
    }
    if (typeof value === "bigint") {
      id = value;
    } else if (value instanceof Uint64MapEntry) {
      id = value.key;
    } else {
      return false;
    }
    const { displayState } = this;
    const normalizedId = augmentSegmentId(displayState, id);
    const {
      segmentEquivalences,
      segmentPropertyMap: { value: segmentPropertyMap },
    } = this.displayState.segmentationGroupState.value;
    const mapped = segmentEquivalences.get(id);
    const row = makeSegmentWidget(this.displayState, normalizedId);
    registerCallbackWhenSegmentationDisplayStateChanged(
      displayState,
      context,
      context.redraw,
    );
    context.registerDisposer(bindSegmentListWidth(displayState, row));
    row.classList.add("neuroglancer-selection-details-segment");
    parent.appendChild(row);

    if (segmentPropertyMap !== undefined) {
      const { inlineProperties } = segmentPropertyMap.segmentPropertyMap;
      if (inlineProperties !== undefined) {
        const index = segmentPropertyMap.getSegmentInlineIndex(mapped);
        if (index !== -1) {
          for (const property of inlineProperties.properties) {
            if (property.type === "label") continue;
            if (property.type === "description") {
              const value = property.values[index];
              if (!value) continue;
              const descriptionElement = document.createElement("div");
              descriptionElement.classList.add(
                "neuroglancer-selection-details-segment-description",
              );
              descriptionElement.textContent = value;
              parent.appendChild(descriptionElement);
            } else if (
              property.type === "number" ||
              property.type === "string"
            ) {
              const value = property.values[index];
              if (
                property.type === "number"
                  ? Number.isNaN(value as number)
                  : !value
              )
                continue;
              const propertyElement = document.createElement("div");
              propertyElement.classList.add(
                "neuroglancer-selection-details-segment-property",
              );
              const nameElement = document.createElement("div");
              nameElement.classList.add(
                "neuroglancer-selection-details-segment-property-name",
              );
              nameElement.textContent = property.id;
              if (property.description) {
                nameElement.title = property.description;
              }
              const valueElement = document.createElement("div");
              valueElement.classList.add(
                "neuroglancer-selection-details-segment-property-value",
              );
              valueElement.textContent = value.toString();
              propertyElement.appendChild(nameElement);
              propertyElement.appendChild(valueElement);
              parent.appendChild(propertyElement);
            }
          }
        }
      }
    }
    return true;
  }

  displaySelectionState(
    state: this["selectionState"],
    parent: HTMLElement,
    context: DependentViewContext,
  ): boolean {
    let displayed = this.displaySegmentationSelection(state, parent, context);
    if (displaySpatialSkeletonSelection(this, state, parent, context))
      displayed = true;
    if (super.displaySelectionState(state, parent, context)) displayed = true;
    return displayed;
  }

  moveToSegment(id: bigint) {
    for (const layer of this.renderLayers) {
      if (
        !(layer instanceof MultiscaleMeshLayer || layer instanceof MeshLayer)
      ) {
        continue;
      }
      const transform = layer.displayState.transform.value;
      if (transform.error !== undefined) return undefined;
      const { rank, globalToRenderLayerDimensions } = transform;
      const { globalPosition } = this.manager.root;
      const globalLayerPosition = new Float32Array(rank);
      const renderToGlobalLayerDimensions: number[] = [];
      for (let i = 0; i < rank; i++) {
        renderToGlobalLayerDimensions[globalToRenderLayerDimensions[i]] = i;
      }
      gatherUpdate(
        globalLayerPosition,
        globalPosition.value,
        renderToGlobalLayerDimensions,
      );
      const layerPosition =
        layer instanceof MeshLayer
          ? layer.getObjectPosition(id, globalLayerPosition)
          : layer.getObjectPosition(id);
      if (layerPosition === undefined) continue;
      this.setLayerPosition(transform, layerPosition);
      return;
    }
    StatusMessage.showTemporaryMessage(
      `No position information loaded for segment ${id}`,
    );
  }

  observeLayerColor(callback: () => void) {
    const disposer = super.observeLayerColor(callback);
    const defaultColorDisposer = observeWatchable(
      callback,
      this.displayState.segmentDefaultColor,
    );
    const visibleSegmentDisposer =
      this.displayState.segmentationGroupState.value.visibleSegments.changed.add(
        callback,
      );
    const colorHashChangeDisposer =
      this.displayState.segmentationColorGroupState.value.segmentColorHash.changed.add(
        callback,
      );
    const showAllByDefaultDisposer =
      this.displayState.ignoreNullVisibleSet.changed.add(callback);
    const hasVolumeDisposer = this.displayState.hasVolume.changed.add(callback);
    return () => {
      disposer();
      defaultColorDisposer();
      visibleSegmentDisposer();
      colorHashChangeDisposer();
      showAllByDefaultDisposer();
      hasVolumeDisposer();
    };
  }

  get automaticLayerBarColors() {
    const { displayState } = this;
    const visibleSegmentsSet =
      displayState.segmentationGroupState.value.visibleSegments;
    const fixedColor = displayState.segmentDefaultColor.value;

    const noVisibleSegments = visibleSegmentsSet.size === 0;
    const tooManyVisibleSegments =
      visibleSegmentsSet.size > MAX_LAYER_BAR_UI_INDICATOR_COLORS;
    const hasMappedColors =
      displayState.segmentationColorGroupState.value.segmentStatedColors.size >
      0;
    const isFixedColorOnly = fixedColor !== undefined && !hasMappedColors;
    const showAllByDefault = displayState.ignoreNullVisibleSet.value;
    const hasVolume = displayState.hasVolume.value;

    if (noVisibleSegments) {
      if (!showAllByDefault || !hasVolume) return []; // No segments visible
      if (isFixedColorOnly) return [getCssColor(fixedColor)];
      return undefined; // Rainbow colors
    }
    if (isFixedColorOnly) {
      return [getCssColor(fixedColor)]; // All segments show as one color
    }

    // Because manually mapped colors are not guaranteed to be unique,
    // we need to actually check all the visible segments if
    // manually mapped colors are used
    if (!hasMappedColors && tooManyVisibleSegments) {
      return undefined; // Too many segments to show
    }

    const visibleSegments = [...visibleSegmentsSet];
    const colors = visibleSegments.map((id) => {
      const color = getCssColor(getBaseObjectColor(displayState, id));
      return { color, id };
    });

    // Sort the colors by their segment ID
    // Otherwise, the order is random which is a bit confusing in the UI
    colors.sort((a, b) => {
      const aId = a.id;
      const bId = b.id;
      return aId < bId ? -1 : aId > bId ? 1 : 0;
    });

    const uniqueColors = [...new Set(colors.map((color) => color.color))];
    if (uniqueColors.length > MAX_LAYER_BAR_UI_INDICATOR_COLORS) {
      return undefined; // Too many colors to show
    }
    return uniqueColors;
  }

  static type = "segmentation";
  static typeAbbreviation = "seg";
  static supportsPickOption = true;
  static supportsLayerBarColorSyncOption = true;
}

registerLayerControls(SegmentationUserLayer);

registerLayerType(SegmentationUserLayer);
registerVolumeLayerType(VolumeType.SEGMENTATION, SegmentationUserLayer);
registerLayerTypeDetector((subsource) => {
  if (subsource.mesh !== undefined || subsource.zarrVectors !== undefined) {
    return { layerConstructor: SegmentationUserLayer, priority: 1 };
  }
  return undefined;
});

registerLayerShaderControlsTool(
  SegmentationUserLayer,
  (layer) => ({
    shaderControlState:
      layer.displayState.skeletonRenderingOptions.shaderControlState,
  }),
  json_keys.SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID,
);

registerSpatialSkeletonEditModeTool(SegmentationUserLayer);
registerSegmentSplitMergeTools(SegmentationUserLayer);
registerSegmentSelectTools(SegmentationUserLayer);
