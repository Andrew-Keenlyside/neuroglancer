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

/**
 * The four tabs the fork adds to a segmentation layer -- Skeleton, Filter,
 * Export and Guide -- together with the conditions under which each appears.
 *
 * Not registered through upstream's USER_LAYER_TABS: that registry is consumed
 * in the UserLayer base constructor, so it applies to every layer type, and its
 * UserLayerTab has no `hidden` field. All four of these tabs need one. So they
 * install from here instead, called once from the layer constructor, which
 * keeps the predicates and their reasoning out of the upstream file.
 */

import "#src/layer/segmentation/style.css";
import "#src/layer/segmentation/spatial_skeleton.css";

import { StreamlineFilterTab } from "#src/datasource/zarr-vectors/streamline_filter_tab.js";
import { StreamlineGuideTab } from "#src/datasource/zarr-vectors/streamline_guide_tab.js";
import { TractExportTab } from "#src/datasource/zarr-vectors/tract_export_tab.js";

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
} from "#src/skeleton/spatial_frontend.js";

import { getSpatiallyIndexedSkeletonSource } from "#src/skeleton/spatial_skeleton_manager.js";

import { makeCachedLazyDerivedWatchableValue } from "#src/trackable_value.js";

import { SpatialSkeletonEditTab } from "#src/ui/spatial_skeleton_edit_tab.js";


export function registerSpatialSkeletonTabs(layer: SegmentationUserLayer) {
  const hideSpatialSkeletonEditTab = layer.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        !layers.some(
          (layer) =>
            (layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
              layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer) &&
            getSpatiallyIndexedSkeletonSource(layer.base) !== undefined,
        ),
      { changed: layer.layersChanged, value: layer.renderLayers },
    ),
  );
  layer.tabs.add("skeleton", {
    label: "Skeleton",
    order: -45,
    getter: () => new SpatialSkeletonEditTab(layer),
    hidden: hideSpatialSkeletonEditTab,
  });
  // Show the Filter tab whenever a spatially-indexed skeleton render layer is
  // present AND the ROI channel was created for it. The channel is created
  // (setting displayState.roiPassingSegments) only for sources that opt into
  // the filter, so layer also excludes other spatially-indexed skeleton
  // sources (e.g. CATMAID) that render tracts but cannot be filtered. Unlike
  // the Skeleton *edit* tab, it does NOT require the skeleton-editing API,
  // which read-only zarr-vectors tracts lack.
  //
  // Every zarr-vectors geometry kind opts in, points and meshes included: a
  // dissection of a point cloud is what its attribute predicates select, and
  // the fold knows from each chunk how to attribute geometry to objects (see
  // `RoiFilterableChunk.perVertexObjects`).
  const hideFilterTab = layer.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layer.displayState.roiPassingSegments === undefined ||
        !layers.some(
          (layer) =>
            layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
            layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer,
        ),
      { changed: layer.layersChanged, value: layer.renderLayers },
    ),
  );
  layer.tabs.add("filter", {
    label: "Filter",
    order: -40,
    getter: () => new StreamlineFilterTab(layer),
    hidden: hideFilterTab,
  });
  // The Filter tab's condition AND a store the export applies to. Both of the
  // Export tab's formats are streamline-shaped -- TrackVis `.trk` is polylines
  // by definition, and the zarr-vectors exporter reads whole tracts -- so a
  // point-cloud or mesh store is filterable without being exportable, and
  // offering the tab there would only lead to a job that cannot be written.
  const hideExportTab = layer.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layer.displayState.roiPassingSegments === undefined ||
        layer.displayState.roiSupportsTractExport !== true ||
        !layers.some(
          (layer) =>
            layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
            layer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer,
        ),
      { changed: layer.layersChanged, value: layer.renderLayers },
    ),
  );
  layer.tabs.add("export", {
    label: "Export",
    order: -38,
    getter: () => new TractExportTab(layer),
    hidden: hideExportTab,
  });
  // Shares the Filter tab's visibility condition: the guide documents that
  // panel, so it should never appear without it.
  layer.tabs.add("filterGuide", {
    label: "Guide",
    order: -39,
    getter: () => new StreamlineGuideTab(),
    hidden: hideFilterTab,
  });
}
