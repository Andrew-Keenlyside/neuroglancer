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
 * The zarr-vectors ROI-filter channel: turning the layer's persisted ROI groups
 * into the structured-clone-safe form the worker consumes, and turning the
 * groups back into the annotation overlay that draws them.
 *
 * Free functions over explicit arguments, so the ROI feature stays out of
 * SegmentationUserLayer's body -- the layer only holds the state and calls in.
 */

import "#src/layer/segmentation/style.css";
import "#src/layer/segmentation/spatial_skeleton.css";

import type { Annotation, AnnotationReference } from "#src/annotation/index.js";
import {
  AnnotationType,
  LocalAnnotationSource,
} from "#src/annotation/index.js";

import type {
  RoiGroupConfig,
  RoiObjectAttrColumn,
} from "#src/datasource/zarr-vectors/roi.js";
import { RoiFilterState } from "#src/datasource/zarr-vectors/roi_filter_state.js";

import type { PreprocessedSegmentPropertyMap } from "#src/segmentation_display_state/property_map.js";

import { packColor } from "#src/util/color.js";

import { vec4 } from "#src/util/geom.js";

/**
 * Flatten the persisted ROI groups into the plain, structured-clone-safe form
 * the worker consumes: each group's ROI list, its colour packed to an int, and
 * its visibility. The worker unions the visible groups' passing tracts (ghost
 * shader) and attributes each passing tract the colour of its group.
 */
export function buildRoiGroupConfigs(
  roiFilter: RoiFilterState,
): RoiGroupConfig[] {
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
export function buildObjectAttrColumns(
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
export const ROI_OVERLAY_SHADER =
  "void main() {\n  setColor(prop_color());\n}\n";
// Same, but discard in the 2-d slice views (hide-overlays-in-2d toggle).
export const ROI_OVERLAY_SHADER_HIDE_2D =
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
export function rebuildRoiAnnotations(
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
export function warnOnceAdmissionUnavailable(hasSource: boolean) {
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
