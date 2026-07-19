/**
 * @license
 * Copyright 2026 Google Inc.
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
 * @file Persisted state for the ROI streamline filter.
 *
 * The filter is organised into GROUPS: each group is one named, coloured
 * dissection holding an ordered ROI list (geometry + predicate + operator,
 * evaluated as the include/or/exclude fold). A streamline belongs to a group
 * iff it passes that group's ROIs; every visible group's tracts are shown,
 * coloured by the group's colour, and everything else is ghosted.
 *
 * Self-contained and serialisable: it round-trips to the URL. Per the design,
 * the URL stores the groups + ROIs + config only — never the materialised
 * passing-id set, which is recomputed from the loaded streamlines.
 *
 * The mutators reassign arrays (never mutate in place) so downstream mirrors can
 * detect a change by array identity — treat everything returned here as
 * immutable and edit via the mutators.
 */

import type {
  Roi,
  RoiOperator,
  RoiPredicate,
  RoiShape,
} from "#src/datasource/zarr-vectors/roi.js";
import {
  RoiOperator as Op,
  RoiPredicate as Pred,
} from "#src/datasource/zarr-vectors/roi.js";
import { serializeColor } from "#src/util/color.js";
import { vec3 } from "#src/util/geom.js";
import {
  parseArray,
  verifyFiniteFloat,
  verifyObject,
  verifyObjectProperty,
  verifyOptionalObjectProperty,
  verifyString,
} from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";

const PREDICATE_TO_JSON: Record<RoiPredicate, string> = {
  [Pred.ANY_SEGMENT]: "any_segment",
  [Pred.ANY_VERTEX]: "any_vertex",
  [Pred.EITHER_ENDPOINT]: "either_endpoint",
  [Pred.BOTH_ENDPOINTS]: "both_endpoints",
};
const JSON_TO_PREDICATE = new Map<string, RoiPredicate>(
  Object.entries(PREDICATE_TO_JSON).map(([k, v]) => [
    v,
    Number(k) as RoiPredicate,
  ]),
);

const OPERATOR_TO_JSON: Record<RoiOperator, string> = {
  [Op.AND]: "and",
  [Op.OR]: "or",
  [Op.ANDNOT]: "andnot",
};
const JSON_TO_OPERATOR = new Map<string, RoiOperator>(
  Object.entries(OPERATOR_TO_JSON).map(([k, v]) => [
    v,
    Number(k) as RoiOperator,
  ]),
);

const DEFAULT_GHOST_ALPHA = 0.3;

/** Distinct default group colours, cycled as groups are created. */
const GROUP_PALETTE = [
  "#ff3b30",
  "#34c759",
  "#0a84ff",
  "#ffcc00",
  "#af52de",
  "#ff9500",
  "#5ac8fa",
  "#ff2d55",
];

/**
 * Parse `#rrggbb` to an rgb vec3 in [0,1]. Canvas-free (unlike
 * `parseRGBColorSpecification`, which needs a 2-d context) so this state stays
 * unit-testable; ROI colours are always hex — from the palette, `serializeColor`,
 * or an `<input type=color>`.
 */
function parseHexColor(hex: string): vec3 {
  const m = /^#?([0-9a-fA-F]{6})$/.exec(hex.trim());
  if (m === null) throw new Error(`Invalid hex colour: ${JSON.stringify(hex)}`);
  const n = Number.parseInt(m[1], 16);
  return vec3.fromValues(
    ((n >> 16) & 0xff) / 255,
    ((n >> 8) & 0xff) / 255,
    (n & 0xff) / 255,
  );
}

function paletteColor(index: number): vec3 {
  return parseHexColor(GROUP_PALETTE[index % GROUP_PALETTE.length]);
}

function shapeToJson(shape: RoiShape): any {
  switch (shape.kind) {
    case "ellipsoid":
      return {
        type: "ellipsoid",
        center: Array.from(shape.center),
        radii: Array.from(shape.radii),
      };
    case "box":
      return {
        type: "box",
        lower: Array.from(shape.lower),
        upper: Array.from(shape.upper),
      };
    case "halfspace":
      return {
        type: "halfspace",
        origin: Array.from(shape.origin),
        normal: Array.from(shape.normal),
      };
  }
}

function floatVec(obj: any): Float32Array {
  const arr = parseArray(obj, verifyFiniteFloat);
  return Float32Array.from(arr);
}

function shapeFromJson(obj: any): RoiShape {
  verifyObject(obj);
  const type = verifyObjectProperty(obj, "type", verifyString);
  switch (type) {
    case "ellipsoid":
      return {
        kind: "ellipsoid",
        center: verifyObjectProperty(obj, "center", floatVec),
        radii: verifyObjectProperty(obj, "radii", floatVec),
      };
    case "box":
      return {
        kind: "box",
        lower: verifyObjectProperty(obj, "lower", floatVec),
        upper: verifyObjectProperty(obj, "upper", floatVec),
      };
    case "halfspace":
      return {
        kind: "halfspace",
        origin: verifyObjectProperty(obj, "origin", floatVec),
        normal: verifyObjectProperty(obj, "normal", floatVec),
      };
    default:
      throw new Error(`Unknown ROI shape type: ${JSON.stringify(type)}`);
  }
}

function entryToJson(roi: Roi): any {
  const json: any = {
    shape: shapeToJson(roi.shape),
    predicate: PREDICATE_TO_JSON[roi.predicate],
    operator: OPERATOR_TO_JSON[roi.operator],
  };
  // Omitted when unnamed, matching the omit-defaults style of `groupToJson`:
  // an unnamed ROI must not start carrying the UI's positional placeholder.
  if (roi.name !== undefined) json.name = roi.name;
  return json;
}

function entryFromJson(obj: any): Roi {
  verifyObject(obj);
  const shape = verifyObjectProperty(obj, "shape", shapeFromJson);
  const predicate = verifyObjectProperty(obj, "predicate", (v) => {
    const p = JSON_TO_PREDICATE.get(verifyString(v));
    if (p === undefined)
      throw new Error(`Unknown ROI predicate: ${JSON.stringify(v)}`);
    return p;
  });
  const operator = verifyObjectProperty(obj, "operator", (v) => {
    const o = JSON_TO_OPERATOR.get(verifyString(v));
    if (o === undefined)
      throw new Error(`Unknown ROI operator: ${JSON.stringify(v)}`);
    return o;
  });
  const name = verifyOptionalObjectProperty(obj, "name", verifyString);
  // Spread-free so an unnamed ROI has no `name` key at all, rather than one
  // holding `undefined` -- `toStrictEqual` in the round-trip tests can tell.
  return name === undefined
    ? { shape, predicate, operator }
    : { shape, predicate, operator, name };
}

/** One named, coloured dissection: an ordered ROI list evaluated as a fold. */
export interface RoiGroup {
  /** Stable within a session (not persisted); the tab references groups by it. */
  readonly id: number;
  readonly name: string;
  readonly color: vec3;
  readonly visible: boolean;
  /** Opacity of this group's passing streamlines, in [0, 1]. */
  readonly opacity: number;
  /** Load this group's passing tracts at full detail (object-keyed pass 2). */
  readonly highDetail: boolean;
  readonly rois: readonly Roi[];
}

const DEFAULT_GROUP_OPACITY = 1;

/**
 * Serialises one group.  Also used by the shared ROI group store, so that a
 * saved document and the URL carry byte-identical group JSON.
 */
export function groupToJson(group: RoiGroup): any {
  const json: any = {
    name: group.name,
    color: serializeColor(group.color),
    rois: group.rois.map(entryToJson),
  };
  if (!group.visible) json.visible = false;
  if (group.opacity !== DEFAULT_GROUP_OPACITY) json.opacity = group.opacity;
  if (group.highDetail) json.highDetail = true;
  return json;
}

/** Inverse of {@link groupToJson}; `id` is assigned by the caller. */
export function groupFromJson(obj: any, id: number): RoiGroup {
  verifyObject(obj);
  return {
    id,
    name: verifyObjectProperty(obj, "name", verifyString),
    color: verifyObjectProperty(obj, "color", (v) =>
      parseHexColor(verifyString(v)),
    ),
    visible:
      verifyOptionalObjectProperty(obj, "visible", (v) => v === true) ?? true,
    opacity:
      verifyOptionalObjectProperty(obj, "opacity", verifyFiniteFloat) ??
      DEFAULT_GROUP_OPACITY,
    highDetail:
      verifyOptionalObjectProperty(obj, "highDetail", (v) => v === true) ??
      false,
    rois: verifyObjectProperty(obj, "rois", (v) =>
      parseArray(v, entryFromJson),
    ),
  };
}

/**
 * The ordered group list plus display options, as persisted layer state.
 * Implements the `Trackable` contract (`changed`/`toJSON`/`restoreState`/
 * `reset`) so it plugs into the segmentation layer's state under one JSON key.
 */
export class RoiFilterState {
  changed = new NullarySignal();

  private active_ = false;
  private ghostAlpha_ = DEFAULT_GHOST_ALPHA;
  private colorByGroup_ = true;
  private hideOverlays2d_ = false;
  private groups_: RoiGroup[] = [];
  private nextGroupId_ = 1;

  /** Whether the filter is applied (colouring/ghosting streamlines). */
  get active(): boolean {
    return this.active_;
  }
  set active(value: boolean) {
    if (value === this.active_) return;
    this.active_ = value;
    this.changed.dispatch();
  }

  /** Opacity of ghosted (non-passing) streamlines, in [0, 1]. */
  get ghostAlpha(): number {
    return this.ghostAlpha_;
  }
  set ghostAlpha(value: number) {
    const clamped = Math.min(1, Math.max(0, value));
    if (clamped === this.ghostAlpha_) return;
    this.ghostAlpha_ = clamped;
    this.changed.dispatch();
  }

  /** Whether to recolour passing streamlines by their group's colour. */
  get colorByGroup(): boolean {
    return this.colorByGroup_;
  }
  set colorByGroup(value: boolean) {
    if (value === this.colorByGroup_) return;
    this.colorByGroup_ = value;
    this.changed.dispatch();
  }

  /** Whether to hide the ROI region overlays in the 2-d slice views. */
  get hideOverlays2d(): boolean {
    return this.hideOverlays2d_;
  }
  set hideOverlays2d(value: boolean) {
    if (value === this.hideOverlays2d_) return;
    this.hideOverlays2d_ = value;
    this.changed.dispatch();
  }

  /** The ordered group list. Treat as immutable; edit via the mutators. */
  get groups(): readonly RoiGroup[] {
    return this.groups_;
  }

  /** Whether any visible group has at least one ROI (i.e. the filter can act). */
  hasVisibleRois(): boolean {
    return this.groups_.some((g) => g.visible && g.rois.length > 0);
  }

  private groupIndex(id: number): number {
    return this.groups_.findIndex((g) => g.id === id);
  }

  /** Append a new empty group with a default name + palette colour; returns its id. */
  addGroup(): number {
    const id = this.nextGroupId_++;
    this.groups_ = [
      ...this.groups_,
      {
        id,
        name: `Group ${this.groups_.length + 1}`,
        color: paletteColor(this.groups_.length),
        visible: true,
        opacity: DEFAULT_GROUP_OPACITY,
        highDetail: false,
        rois: [],
      },
    ];
    this.changed.dispatch();
    return id;
  }

  /**
   * Append an existing group's contents as a new group; returns its new id.
   *
   * Used to move a group between layers. `addGroup` + `updateGroup` cannot do
   * this -- `updateGroup` deliberately does not take `rois`, and rebuilding the
   * ROIs one `addRoi` at a time would dispatch `changed` per ROI, firing a
   * filter recompute for each intermediate state. The id is reassigned because
   * ids are only unique within one state.
   */
  insertGroup(group: Omit<RoiGroup, "id">): number {
    const id = this.nextGroupId_++;
    this.groups_ = [...this.groups_, { ...group, id }];
    this.changed.dispatch();
    return id;
  }

  removeGroup(id: number): void {
    const next = this.groups_.filter((g) => g.id !== id);
    if (next.length === this.groups_.length) return;
    this.groups_ = next;
    this.changed.dispatch();
  }

  updateGroup(
    id: number,
    changes: {
      name?: string;
      color?: vec3;
      visible?: boolean;
      opacity?: number;
      highDetail?: boolean;
    },
  ): void {
    const idx = this.groupIndex(id);
    if (idx < 0) return;
    const next = this.groups_.slice();
    next[idx] = { ...next[idx], ...changes };
    this.groups_ = next;
    this.changed.dispatch();
  }

  moveGroup(from: number, to: number): void {
    const n = this.groups_.length;
    if (from < 0 || from >= n || to < 0 || to >= n || from === to) return;
    const next = this.groups_.slice();
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    this.groups_ = next;
    this.changed.dispatch();
  }

  /** Append one ROI to the group; returns its index, or -1 if the group is gone. */
  addRoi(groupId: number, roi: Roi): number {
    const idx = this.groupIndex(groupId);
    if (idx < 0) return -1;
    const group = this.groups_[idx];
    const rois = [...group.rois, roi];
    const next = this.groups_.slice();
    next[idx] = { ...group, rois };
    this.groups_ = next;
    this.changed.dispatch();
    return rois.length - 1;
  }

  updateRoi(groupId: number, roiIndex: number, changes: Partial<Roi>): void {
    const idx = this.groupIndex(groupId);
    if (idx < 0) return;
    const group = this.groups_[idx];
    if (roiIndex < 0 || roiIndex >= group.rois.length) return;
    const rois = group.rois.slice();
    rois[roiIndex] = { ...rois[roiIndex], ...changes };
    const next = this.groups_.slice();
    next[idx] = { ...group, rois };
    this.groups_ = next;
    this.changed.dispatch();
  }

  removeRoi(groupId: number, roiIndex: number): void {
    const idx = this.groupIndex(groupId);
    if (idx < 0) return;
    const group = this.groups_[idx];
    if (roiIndex < 0 || roiIndex >= group.rois.length) return;
    const rois = group.rois.slice();
    rois.splice(roiIndex, 1);
    const next = this.groups_.slice();
    next[idx] = { ...group, rois };
    this.groups_ = next;
    this.changed.dispatch();
  }

  moveRoi(groupId: number, from: number, to: number): void {
    const idx = this.groupIndex(groupId);
    if (idx < 0) return;
    const group = this.groups_[idx];
    const n = group.rois.length;
    if (from < 0 || from >= n || to < 0 || to >= n || from === to) return;
    const rois = group.rois.slice();
    const [item] = rois.splice(from, 1);
    rois.splice(to, 0, item);
    const next = this.groups_.slice();
    next[idx] = { ...group, rois };
    this.groups_ = next;
    this.changed.dispatch();
  }

  toJSON(): any {
    if (!this.active_ && this.groups_.length === 0) {
      return undefined; // stays out of the URL when unused
    }
    const json: any = {};
    if (this.groups_.length > 0) json.groups = this.groups_.map(groupToJson);
    if (this.active_) json.active = true;
    if (this.ghostAlpha_ !== DEFAULT_GHOST_ALPHA)
      json.ghostAlpha = this.ghostAlpha_;
    if (!this.colorByGroup_) json.colorByGroup = false;
    if (this.hideOverlays2d_) json.hideOverlays2d = true;
    return json;
  }

  restoreState(x: unknown): void {
    if (x === undefined) {
      this.reset();
      return;
    }
    verifyObject(x);
    this.nextGroupId_ = 1;
    const groupsJson = verifyOptionalObjectProperty(x, "groups", (v) => v);
    if (groupsJson !== undefined) {
      this.groups_ = parseArray(groupsJson, (obj) =>
        groupFromJson(obj, this.nextGroupId_++),
      );
    } else {
      // Back-compat: an older flat `rois` list restores as one default group.
      const flatRois = verifyOptionalObjectProperty(x, "rois", (v) =>
        parseArray(v, entryFromJson),
      );
      this.groups_ =
        flatRois !== undefined && flatRois.length > 0
          ? [
              {
                id: this.nextGroupId_++,
                name: "Group 1",
                color: paletteColor(0),
                visible: true,
                opacity: DEFAULT_GROUP_OPACITY,
                highDetail: false,
                rois: flatRois,
              },
            ]
          : [];
    }
    this.active_ =
      verifyOptionalObjectProperty(x, "active", (v) => v === true) ?? false;
    this.ghostAlpha_ =
      verifyOptionalObjectProperty(x, "ghostAlpha", verifyFiniteFloat) ??
      DEFAULT_GHOST_ALPHA;
    this.colorByGroup_ =
      verifyOptionalObjectProperty(x, "colorByGroup", (v) => v === true) ??
      true;
    this.hideOverlays2d_ =
      verifyOptionalObjectProperty(x, "hideOverlays2d", (v) => v === true) ??
      false;
    this.changed.dispatch();
  }

  reset(): void {
    const wasDefault =
      !this.active_ &&
      this.colorByGroup_ &&
      !this.hideOverlays2d_ &&
      this.groups_.length === 0 &&
      this.ghostAlpha_ === DEFAULT_GHOST_ALPHA;
    this.active_ = false;
    this.ghostAlpha_ = DEFAULT_GHOST_ALPHA;
    this.colorByGroup_ = true;
    this.hideOverlays2d_ = false;
    this.groups_ = [];
    this.nextGroupId_ = 1;
    if (!wasDefault) this.changed.dispatch();
  }
}
