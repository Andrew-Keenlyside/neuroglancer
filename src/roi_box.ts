/**
 * @license
 * Copyright 2024 Google Inc.
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
 * @file Per-layer cursor ROI bounding box.
 *
 * A ROI box is an axis-aligned bounding box, defined in physical units, that is
 * centered on (or anchored relative to) the cursor and used to restrict both
 * data loading and rendering for a single data layer. This module holds the
 * dependency-light state + serialization + geometry helpers so that the backend
 * chunk-priority logic, the frontend render layers, and the UI/display can all
 * share the same definition.
 */

import { TrackableBoolean } from "#src/trackable_boolean.js";
import { TrackableVec3 } from "#src/trackable_vec3.js";
import { RefCounted } from "#src/util/disposable.js";
import { mat4, vec3 } from "#src/util/geom.js";
import {
  verifyFiniteFloat,
  verifyObject,
  verifyOptionalObjectProperty,
} from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";
import type { Trackable } from "#src/util/trackable.js";

/**
 * Plain, serializable snapshot of a ROI box configuration. Physical sizes are in
 * SI meters; `center` is the absolute box center in global ("display") voxel
 * coordinates. This is what gets sent to the backend chunk-priority logic.
 *
 * The box has a single authoritative location: `center`. When `followNavCenter`
 * is true, the frontend keeps `center` synced to the navigation position; when it
 * is turned off, `center` simply stops updating, so the box stays exactly where it
 * was. There is no separate "offset" or "anchor" — the center is the source of
 * truth in both modes.
 */
export interface RoiBoxParameters {
  enabled: boolean;
  /** Full size of the box along each global display axis, in SI meters. */
  physicalSize: Float32Array;
  /** Absolute box center, in global ("display") voxel coordinates. */
  center: Float32Array;
  /** When true, the box center tracks the navigation position. */
  followNavCenter: boolean;
  /** When true, the box stays a constant size on screen (scales with zoom). */
  zoomRelative: boolean;
  /** When true, the three size components are kept equal. */
  uniformSize: boolean;
}

const DEFAULT_PHYSICAL_SIZE = 1e-6; // 1 μm in SI meters

export function makeDefaultRoiBoxParameters(): RoiBoxParameters {
  return {
    enabled: false,
    physicalSize: Float32Array.of(
      DEFAULT_PHYSICAL_SIZE,
      DEFAULT_PHYSICAL_SIZE,
      DEFAULT_PHYSICAL_SIZE,
    ),
    center: Float32Array.of(0, 0, 0),
    followNavCenter: true,
    zoomRelative: false,
    uniformSize: true,
  };
}

export function copyRoiBoxParameters(p: RoiBoxParameters): RoiBoxParameters {
  return {
    enabled: p.enabled,
    physicalSize: Float32Array.from(p.physicalSize),
    center: Float32Array.from(p.center),
    followNavCenter: p.followNavCenter,
    zoomRelative: p.zoomRelative,
    uniformSize: p.uniformSize,
  };
}

export interface RoiGlobalBox {
  /** Lower corner in global ("display") voxel coordinates. */
  lower: Float32Array;
  /** Upper corner in global ("display") voxel coordinates. */
  upper: Float32Array;
}

/**
 * Compute the effective ROI box in global ("display") voxel coordinates.
 *
 * The global display space consumed by `viewProjectionMat` /
 * `chunkLayout.transform` is in **native voxel coordinates per display axis**
 * (see `NavigationState.toMat4`, where the camera translation is set directly
 * from the native `voxelCoordinates`). So both `center` and the per-axis
 * half-extents must be expressed in native voxels of each display axis — using a
 * single canonical scale would be wrong for anisotropic data.
 *
 * @param center Absolute box center in native display-voxel coordinates (i.e.
 *     `globalPosition.value` restricted to the display axes). Length equals the
 *     number of display dimensions (<= 3).
 * @param physicalSize Full box size per display axis, in SI meters.
 * @param displayScales Native physical scale (SI meters per voxel) for each
 *     display axis (`displayDimensionRenderInfo.displayDimensionScales`). Used to
 *     convert SI-meter sizes to per-axis voxel extents, handling anisotropy.
 * @param zoomFactor Current zoom (canonical voxels per screen pixel). When
 *     `zoomRelative` is set the half-extents are scaled by this so the box stays
 *     a constant size on screen, mirroring the depth-range convention.
 * @param zoomRelative When true, scale half-extents by `zoomFactor`.
 */
export function computeRoiGlobalBox(
  center: ArrayLike<number>,
  physicalSize: ArrayLike<number>,
  displayScales: ArrayLike<number>,
  zoomFactor: number,
  zoomRelative: boolean,
): RoiGlobalBox {
  const rank = center.length;
  const lower = new Float32Array(rank);
  const upper = new Float32Array(rank);
  for (let i = 0; i < rank; ++i) {
    const scale = (displayScales[i] as number | undefined) || 1;
    let halfExtentVoxels =
      (physicalSize[i] ?? DEFAULT_PHYSICAL_SIZE) / 2 / scale;
    if (zoomRelative) {
      halfExtentVoxels *= zoomFactor;
    }
    const c = center[i] ?? 0;
    lower[i] = c - halfExtentVoxels;
    upper[i] = c + halfExtentVoxels;
  }
  return { lower, upper };
}

const tempCorner = vec3.create();
const tempTransformed = vec3.create();

/**
 * Convert a ROI box from global ("display") voxel coordinates into a source's
 * chunk-layout local space, returning the axis-aligned bounds (component-wise
 * min/max of the transformed 8 corners). For oblique/rotated transforms this is a
 * conservative superset; for axis-aligned transforms it is exact.
 *
 * @param box ROI box in global voxel coordinates (length-3 lower/upper).
 * @param globalToLocal The source's `chunkLayout.invTransform` (global -> local).
 * @param chunkSize When provided (the source's `chunkLayout.size`), the result is
 *     returned in chunk-grid units (local / chunkSize) snapped outward to integer
 *     chunk boundaries; otherwise the result is in local voxel units.
 */
export function roiGlobalBoxToLocalBounds(
  box: RoiGlobalBox,
  globalToLocal: mat4,
  chunkSize?: ArrayLike<number>,
): { lower: vec3; upper: vec3 } {
  const lower = vec3.fromValues(Infinity, Infinity, Infinity);
  const upper = vec3.fromValues(-Infinity, -Infinity, -Infinity);
  for (let corner = 0; corner < 8; ++corner) {
    for (let i = 0; i < 3; ++i) {
      const useUpper = (corner >> i) & 1;
      tempCorner[i] = useUpper ? (box.upper[i] ?? 0) : (box.lower[i] ?? 0);
    }
    vec3.transformMat4(tempTransformed, tempCorner, globalToLocal);
    for (let i = 0; i < 3; ++i) {
      lower[i] = Math.min(lower[i], tempTransformed[i]);
      upper[i] = Math.max(upper[i], tempTransformed[i]);
    }
  }
  if (chunkSize !== undefined) {
    for (let i = 0; i < 3; ++i) {
      // Snap outward to integer chunk boundaries so the requested chunk set only
      // changes when the box crosses a chunk boundary (reduces chunk churn).
      lower[i] = Math.floor(lower[i] / chunkSize[i]);
      upper[i] = Math.ceil(upper[i] / chunkSize[i]);
    }
  }
  return { lower, upper };
}

/**
 * Per-layer trackable ROI box state (the UI-facing source of truth). Wraps the
 * individual trackables, exposes an aggregate `changed` signal, and enforces the
 * uniform-size lock.
 */
export class TrackableRoiBoxState extends RefCounted implements Trackable {
  changed = new NullarySignal();

  enabled = new TrackableBoolean(false);
  /** Activates box-edit interaction priority (the "mode"). */
  editActive = new TrackableBoolean(false);
  physicalSize = new TrackableVec3(
    vec3.fromValues(
      DEFAULT_PHYSICAL_SIZE,
      DEFAULT_PHYSICAL_SIZE,
      DEFAULT_PHYSICAL_SIZE,
    ),
    vec3.fromValues(
      DEFAULT_PHYSICAL_SIZE,
      DEFAULT_PHYSICAL_SIZE,
      DEFAULT_PHYSICAL_SIZE,
    ),
  );
  /** Absolute box center in global ("display") voxel coordinates. */
  center = new TrackableVec3(vec3.create(), vec3.create());
  followNavCenter = new TrackableBoolean(true);
  zoomRelative = new TrackableBoolean(false);
  uniformSize = new TrackableBoolean(true);

  constructor() {
    super();
    for (const t of [
      this.enabled,
      this.editActive,
      this.physicalSize,
      this.center,
      this.followNavCenter,
      this.zoomRelative,
      this.uniformSize,
    ]) {
      this.registerDisposer(t.changed.add(this.changed.dispatch));
    }
    // Disabling the box also exits edit mode.
    this.registerDisposer(
      this.enabled.changed.add(() => {
        if (!this.enabled.value && this.editActive.value) {
          this.editActive.value = false;
        }
      }),
    );
  }

  /** Set one size component, mirroring to all three when uniformSize is on. */
  setSizeComponent(axis: number, valuePhysical: number) {
    const cur = this.physicalSize.value;
    const next = vec3.clone(cur);
    if (this.uniformSize.value) {
      next[0] = next[1] = next[2] = valuePhysical;
    } else {
      next[axis] = valuePhysical;
    }
    this.physicalSize.value = next;
  }

  /** Build a serializable snapshot for transmission to the backend. */
  toParameters(): RoiBoxParameters {
    return {
      enabled: this.enabled.value,
      physicalSize: Float32Array.from(this.physicalSize.value),
      center: Float32Array.from(this.center.value),
      followNavCenter: this.followNavCenter.value,
      zoomRelative: this.zoomRelative.value,
      uniformSize: this.uniformSize.value,
    };
  }

  toJSON() {
    const result: Record<string, unknown> = {};
    if (this.enabled.value) result.enabled = true;
    if (this.editActive.value) result.edit = true;
    const size = this.physicalSize.toJSON();
    if (size !== undefined) result.size = size;
    // The box center is only meaningful as a persisted location when the box is
    // NOT following the navigation position; while following, it is re-derived
    // from the nav position on load, so omit it to avoid churn.
    if (!this.followNavCenter.value) {
      result.followNav = false;
      const center = this.center.toJSON();
      if (center !== undefined) result.center = center;
    }
    if (this.zoomRelative.value) result.zoomRelative = true;
    if (!this.uniformSize.value) result.uniform = false;
    return Object.keys(result).length === 0 ? undefined : result;
  }

  restoreState(obj: unknown) {
    this.reset();
    if (obj === undefined) return;
    verifyObject(obj);
    verifyOptionalObjectProperty(obj, "enabled", (x) => {
      this.enabled.restoreState(x);
    });
    verifyOptionalObjectProperty(obj, "edit", (x) => {
      this.editActive.restoreState(x);
    });
    verifyOptionalObjectProperty(obj, "size", (x) => {
      this.physicalSize.restoreState(x);
    });
    verifyOptionalObjectProperty(obj, "followNav", (x) => {
      this.followNavCenter.restoreState(x);
    });
    verifyOptionalObjectProperty(obj, "center", (x) => {
      this.center.restoreState(x);
    });
    verifyOptionalObjectProperty(obj, "zoomRelative", (x) => {
      this.zoomRelative.restoreState(x);
    });
    verifyOptionalObjectProperty(obj, "uniform", (x) => {
      this.uniformSize.restoreState(x);
    });
    // Guard against malformed sizes.
    for (const v of this.physicalSize.value) verifyFiniteFloat(v);
  }

  reset() {
    this.enabled.reset();
    this.editActive.reset();
    this.physicalSize.reset();
    this.center.reset();
    this.followNavCenter.reset();
    this.zoomRelative.reset();
    this.uniformSize.reset();
  }
}
