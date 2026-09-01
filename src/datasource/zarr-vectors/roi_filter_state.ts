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
  RoiAttrFilter,
  RoiColorSpec,
  RoiLengthFilter,
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
  verifyInt,
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
    case "labelMask":
      return { type: "labelMask", labels: Array.from(shape.labels) };
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
    case "labelMask":
      return {
        kind: "labelMask",
        labels: verifyObjectProperty(obj, "labels", (v) =>
          parseArray(v, verifyInt),
        ),
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

/**
 * Colour and length-filter specs live in `roi.js` (the dependency-free module)
 * so the worker-facing `RoiGroupConfig` can reference them without an import
 * cycle. Re-exported here — this module owns their JSON encoding and defaults.
 *
 * `RoiColorSpec` mirrors the opacity model: each group carries its own spec and
 * the background carries one too, settable independently. `group`/`objectAttr`
 * are flat per object and ride the colour-override map (fully independent per
 * group and background); `direction`/`position`/`vertexAttr` vary along a
 * polyline and are realised in the layer shader (a group choosing one of them
 * "inherits" the background per-vertex colour).
 */
export type { RoiAttrFilter, RoiColorSpec, RoiLengthFilter };

export const DEFAULT_GROUP_COLOR_BY: RoiColorSpec = { kind: "group" };
export const DEFAULT_BACKGROUND_COLOR_BY: RoiColorSpec = { kind: "direction" };

export function colorSpecEquals(a: RoiColorSpec, b: RoiColorSpec): boolean {
  if (a.kind !== b.kind) return false;
  return (a as { name?: string }).name === (b as { name?: string }).name;
}

export function lengthFilterEquals(
  a: RoiLengthFilter | undefined,
  b: RoiLengthFilter | undefined,
): boolean {
  if (a === undefined || b === undefined) return a === b;
  return a.name === b.name && a.min === b.min && a.max === b.max;
}

/** Whether two attribute-predicate lists are the same, in the same order. */
export function attrFiltersEqual(
  a: readonly RoiAttrFilter[],
  b: readonly RoiAttrFilter[],
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; ++i) {
    if (
      a[i].name !== b[i].name ||
      a[i].min !== b[i].min ||
      a[i].max !== b[i].max ||
      (a[i].scope ?? "object") !== (b[i].scope ?? "object")
    ) {
      return false;
    }
  }
  return true;
}

/** Compact, URL-friendly encoding (`"direction"`, `"object:<name>"`, …). */
function colorSpecToJson(spec: RoiColorSpec): string {
  switch (spec.kind) {
    case "direction":
    case "group":
    case "position":
      return spec.kind;
    case "objectAttr":
      return `object:${spec.name}`;
    case "vertexAttr":
      return `vertex:${spec.name}`;
  }
}

function colorSpecFromJson(v: unknown): RoiColorSpec {
  const s = verifyString(v);
  if (s === "direction" || s === "group" || s === "position") {
    return { kind: s };
  }
  const sep = s.indexOf(":");
  if (sep > 0) {
    const prefix = s.slice(0, sep);
    const name = s.slice(sep + 1);
    if (name.length > 0 && prefix === "object") {
      return { kind: "objectAttr", name };
    }
    if (name.length > 0 && prefix === "vertex") {
      return { kind: "vertexAttr", name };
    }
  }
  throw new Error(`Invalid colour spec: ${JSON.stringify(s)}`);
}

function lengthFilterToJson(f: RoiLengthFilter): any {
  return { name: f.name, min: f.min, max: f.max };
}

/**
 * An attribute predicate's JSON. `scope` is omitted for the default `"object"`
 * tier, which is also what an old `lengthFilter` key migrates to, so a state
 * saved before per-vertex predicates existed round-trips byte-identically.
 */
function attrFilterToJson(f: RoiAttrFilter): any {
  const json: any = { name: f.name, min: f.min, max: f.max };
  if (f.scope !== undefined && f.scope !== "object") json.scope = f.scope;
  return json;
}

function attrFilterFromJson(obj: any): RoiAttrFilter {
  verifyObject(obj);
  const base = {
    name: verifyObjectProperty(obj, "name", verifyString),
    min: verifyObjectProperty(obj, "min", verifyFiniteFloat),
    max: verifyObjectProperty(obj, "max", verifyFiniteFloat),
  };
  const scope = verifyOptionalObjectProperty(obj, "scope", (v) => {
    const s = verifyString(v);
    if (s !== "object" && s !== "vertex") {
      throw new Error(`Invalid attribute scope: ${JSON.stringify(s)}`);
    }
    return s;
  });
  // Spread-free so an object-scope predicate has no `scope` key at all, which
  // is what makes the legacy round-trip exact (see `attrFilterToJson`).
  return scope === undefined ? base : { ...base, scope };
}

function lengthFilterFromJson(obj: any): RoiLengthFilter {
  verifyObject(obj);
  return {
    name: verifyObjectProperty(obj, "name", verifyString),
    min: verifyObjectProperty(obj, "min", verifyFiniteFloat),
    max: verifyObjectProperty(obj, "max", verifyFiniteFloat),
  };
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
  /** How this group's passing streamlines are coloured. */
  readonly colorBy: RoiColorSpec;
  /**
   * Attribute predicates restricting this group, ANDed with each other and with
   * the ROI fold. Always present (possibly empty) so every reader can treat it
   * as a list rather than as an optional single filter -- which is what the
   * legacy `lengthFilter` key was, and what it migrates into.
   */
  readonly attrFilters: readonly RoiAttrFilter[];
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
  if (!colorSpecEquals(group.colorBy, DEFAULT_GROUP_COLOR_BY))
    json.colorBy = colorSpecToJson(group.colorBy);
  if (group.attrFilters.length !== 0)
    json.attrFilters = group.attrFilters.map(attrFilterToJson);
  return json;
}

/**
 * Inverse of {@link groupToJson}; `id` is assigned by the caller.
 *
 * `defaultColorBy` seeds `colorBy` when the group JSON carries no explicit key.
 * The state restorer passes the migrated legacy `colorByGroup` here so an old
 * URL keeps its colouring; individual store loads use the group default.
 */
export function groupFromJson(
  obj: any,
  id: number,
  defaultColorBy: RoiColorSpec = DEFAULT_GROUP_COLOR_BY,
): RoiGroup {
  verifyObject(obj);
  // `lengthFilter` is the pre-list spelling: one object-scope range. Read it
  // when `attrFilters` is absent so a saved dissection (URL or ROI store) keeps
  // filtering exactly as it did.
  const legacy = verifyOptionalObjectProperty(
    obj,
    "lengthFilter",
    lengthFilterFromJson,
  );
  const attrFilters =
    verifyOptionalObjectProperty(obj, "attrFilters", (v) =>
      parseArray(v, attrFilterFromJson),
    ) ?? (legacy === undefined ? [] : [legacy]);
  const base = {
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
    colorBy:
      verifyOptionalObjectProperty(obj, "colorBy", colorSpecFromJson) ??
      defaultColorBy,
    attrFilters,
    rois: verifyObjectProperty(obj, "rois", (v) =>
      parseArray(v, entryFromJson),
    ),
  };
  return base;
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
  private backgroundColorBy_: RoiColorSpec = DEFAULT_BACKGROUND_COLOR_BY;
  private backgroundLengthFilter_: RoiLengthFilter | undefined = undefined;
  private groups_: RoiGroup[] = [];
  private nextGroupId_ = 1;
  /**
   * The live, uncommitted "label selection" dissection: the group the segmentation-
   * label panel edits as the user toggles parcellation labels include/exclude.
   * It filters (and previews) exactly like a committed group, but is deliberately
   * kept OUT of `groups_` so it neither appears in the group list nor persists to
   * the URL — the user commits it to a real group explicitly (see
   * {@link commitPreviewGroup}). `undefined` when nothing is staged.
   */
  private previewGroup_: RoiGroup | undefined = undefined;

  /** Reserved session id for {@link previewGroup}; real ids start at 1. */
  static readonly PREVIEW_GROUP_ID = 0;

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

  /** How the background (non-passing / ungrouped) streamlines are coloured. */
  get backgroundColorBy(): RoiColorSpec {
    return this.backgroundColorBy_;
  }
  set backgroundColorBy(value: RoiColorSpec) {
    if (colorSpecEquals(value, this.backgroundColorBy_)) return;
    this.backgroundColorBy_ = value;
    this.changed.dispatch();
  }

  /** Overall length range applied to the background tractogram (or `undefined`). */
  get backgroundLengthFilter(): RoiLengthFilter | undefined {
    return this.backgroundLengthFilter_;
  }
  set backgroundLengthFilter(value: RoiLengthFilter | undefined) {
    if (lengthFilterEquals(this.backgroundLengthFilter_, value)) return;
    this.backgroundLengthFilter_ = value;
    this.changed.dispatch();
  }

  /** The ordered group list. Treat as immutable; edit via the mutators. */
  get groups(): readonly RoiGroup[] {
    return this.groups_;
  }

  /** The live label-selection preview dissection, or `undefined` if none staged. */
  get previewGroup(): RoiGroup | undefined {
    return this.previewGroup_;
  }

  /**
   * Replace (or, with `undefined`, clear) the live label-selection preview. The
   * panel calls this on every toggle with a freshly-built group whose ROIs are
   * the currently-selected labels; the reserved id is stamped so the tab and
   * worker can address it stably.
   */
  setPreviewGroup(group: Omit<RoiGroup, "id"> | undefined): void {
    if (group === undefined) {
      if (this.previewGroup_ === undefined) return;
      this.previewGroup_ = undefined;
      this.changed.dispatch();
      return;
    }
    this.previewGroup_ = { ...group, id: RoiFilterState.PREVIEW_GROUP_ID };
    this.changed.dispatch();
  }

  /**
   * Promote the staged selection to a real, persisted group and clear the
   * preview; returns the new group's id, or `undefined` if nothing was staged.
   * A fresh palette colour and (optional) name are assigned so the committed
   * group reads as a first-class dissection.
   *
   * "Nothing staged" means neither ROIs nor attribute predicates: an
   * attribute-only selection (the "By attribute" panel) is a complete
   * dissection on its own, and refusing to commit it would make that panel's
   * one action a no-op.
   */
  commitPreviewGroup(name?: string): number | undefined {
    const preview = this.previewGroup_;
    if (
      preview === undefined ||
      (preview.rois.length === 0 && preview.attrFilters.length === 0)
    ) {
      return undefined;
    }
    const id = this.nextGroupId_++;
    this.groups_ = [
      ...this.groups_,
      {
        ...preview,
        id,
        name: name ?? preview.name,
        color: paletteColor(this.groups_.length),
      },
    ];
    this.previewGroup_ = undefined;
    this.changed.dispatch();
    return id;
  }

  /**
   * Groups the worker should evaluate: the committed groups plus the live
   * label-selection preview (if any), so a staged selection ghosts/colours
   * streamlines before it is committed.
   */
  groupsForWorker(): readonly RoiGroup[] {
    return this.previewGroup_ === undefined
      ? this.groups_
      : [...this.groups_, this.previewGroup_];
  }

  /**
   * Whether any visible group can select anything -- it has an ROI or an
   * attribute predicate. Either makes it a dissection; a group with neither is
   * an empty shell (a just-added group) and the filter stays inert for it.
   */
  hasVisibleRois(): boolean {
    const selects = (g: Pick<RoiGroup, "rois" | "attrFilters">) =>
      g.rois.length > 0 || g.attrFilters.length > 0;
    if (this.previewGroup_ !== undefined && selects(this.previewGroup_)) {
      return true;
    }
    return this.groups_.some((g) => g.visible && selects(g));
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
        colorBy: DEFAULT_GROUP_COLOR_BY,
        attrFilters: [],
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
      colorBy?: RoiColorSpec;
      /** Replaces the whole predicate list; `[]` clears it. */
      attrFilters?: readonly RoiAttrFilter[];
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
    // Background colour/length are meaningful without any group (colour or clip
    // the whole tractogram), so they keep the state in the URL on their own.
    const hasBackground =
      !colorSpecEquals(this.backgroundColorBy_, DEFAULT_BACKGROUND_COLOR_BY) ||
      this.backgroundLengthFilter_ !== undefined;
    if (!this.active_ && this.groups_.length === 0 && !hasBackground) {
      return undefined; // stays out of the URL when unused
    }
    const json: any = {};
    if (this.groups_.length > 0) json.groups = this.groups_.map(groupToJson);
    if (this.active_) json.active = true;
    if (this.ghostAlpha_ !== DEFAULT_GHOST_ALPHA)
      json.ghostAlpha = this.ghostAlpha_;
    if (!this.colorByGroup_) json.colorByGroup = false;
    if (this.hideOverlays2d_) json.hideOverlays2d = true;
    if (!colorSpecEquals(this.backgroundColorBy_, DEFAULT_BACKGROUND_COLOR_BY))
      json.backgroundColorBy = colorSpecToJson(this.backgroundColorBy_);
    if (this.backgroundLengthFilter_ !== undefined)
      json.backgroundLengthFilter = lengthFilterToJson(
        this.backgroundLengthFilter_,
      );
    return json;
  }

  restoreState(x: unknown): void {
    if (x === undefined) {
      this.reset();
      return;
    }
    verifyObject(x);
    this.nextGroupId_ = 1;
    // Legacy: a global `colorByGroup` predates the unified per-group `colorBy`.
    // Read it first so it can seed the default for groups that carry no explicit
    // `colorBy` key (`true` → group colour, `false` → direction). Still kept as a
    // live field while the render path is transitioned off it.
    this.colorByGroup_ =
      verifyOptionalObjectProperty(x, "colorByGroup", (v) => v === true) ??
      true;
    const legacyDefaultColorBy: RoiColorSpec = this.colorByGroup_
      ? DEFAULT_GROUP_COLOR_BY
      : { kind: "direction" };
    const groupsJson = verifyOptionalObjectProperty(x, "groups", (v) => v);
    if (groupsJson !== undefined) {
      this.groups_ = parseArray(groupsJson, (obj) =>
        groupFromJson(obj, this.nextGroupId_++, legacyDefaultColorBy),
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
                attrFilters: [],
                colorBy: legacyDefaultColorBy,
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
    this.hideOverlays2d_ =
      verifyOptionalObjectProperty(x, "hideOverlays2d", (v) => v === true) ??
      false;
    this.backgroundColorBy_ =
      verifyOptionalObjectProperty(x, "backgroundColorBy", colorSpecFromJson) ??
      DEFAULT_BACKGROUND_COLOR_BY;
    this.backgroundLengthFilter_ = verifyOptionalObjectProperty(
      x,
      "backgroundLengthFilter",
      lengthFilterFromJson,
    );
    this.changed.dispatch();
  }

  reset(): void {
    const wasDefault =
      !this.active_ &&
      this.colorByGroup_ &&
      !this.hideOverlays2d_ &&
      this.groups_.length === 0 &&
      this.previewGroup_ === undefined &&
      this.ghostAlpha_ === DEFAULT_GHOST_ALPHA &&
      colorSpecEquals(this.backgroundColorBy_, DEFAULT_BACKGROUND_COLOR_BY) &&
      this.backgroundLengthFilter_ === undefined;
    this.active_ = false;
    this.ghostAlpha_ = DEFAULT_GHOST_ALPHA;
    this.colorByGroup_ = true;
    this.hideOverlays2d_ = false;
    this.backgroundColorBy_ = DEFAULT_BACKGROUND_COLOR_BY;
    this.backgroundLengthFilter_ = undefined;
    this.groups_ = [];
    this.previewGroup_ = undefined;
    this.nextGroupId_ = 1;
    if (!wasDefault) this.changed.dispatch();
  }
}
