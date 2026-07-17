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
 * Self-contained and serialisable: it holds the ordered ROI list (geometry +
 * predicate + operator) and the display options, and round-trips to the URL.
 * Per the design, the URL stores the ROIs and filter config only — never the
 * materialised passing-id set, which is recomputed from the loaded streamlines.
 *
 * `entries` is already a valid `Roi[]` (same field names), so `rois` is a
 * direct view fed to {@link buildRoiList}'s consumers / the per-chunk filter.
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

const DEFAULT_GHOST_ALPHA = 0.1;

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
  return {
    shape: shapeToJson(roi.shape),
    predicate: PREDICATE_TO_JSON[roi.predicate],
    operator: OPERATOR_TO_JSON[roi.operator],
  };
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
  return { shape, predicate, operator };
}

/**
 * The ordered ROI list plus display options, as persisted layer state.
 * Implements the `Trackable` contract (`changed`/`toJSON`/`restoreState`/
 * `reset`) so it plugs into the segmentation layer's state under one JSON key.
 */
export class RoiFilterState {
  changed = new NullarySignal();

  private active_ = false;
  private ghostAlpha_ = DEFAULT_GHOST_ALPHA;
  private colorByGroup_ = false;
  private entries_: Roi[] = [];

  /** Whether the filter is applied (ghosting non-passing streamlines). */
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

  /** Whether to recolour passing streamlines by their matched ROI group. */
  get colorByGroup(): boolean {
    return this.colorByGroup_;
  }
  set colorByGroup(value: boolean) {
    if (value === this.colorByGroup_) return;
    this.colorByGroup_ = value;
    this.changed.dispatch();
  }

  /** The ordered ROI list. Already a valid `Roi[]` for the filter. */
  get rois(): readonly Roi[] {
    return this.entries_;
  }

  /** Replace the ROI list (used by the authoring UI). */
  setRois(rois: readonly Roi[]): void {
    this.entries_ = rois.slice();
    this.changed.dispatch();
  }

  /** Append one ROI; returns its index. */
  addRoi(roi: Roi): number {
    this.entries_ = [...this.entries_, roi];
    this.changed.dispatch();
    return this.entries_.length - 1;
  }

  /** Replace fields of the ROI at `index` (no-op if out of range). */
  updateRoi(index: number, changes: Partial<Roi>): void {
    if (index < 0 || index >= this.entries_.length) return;
    const next = this.entries_.slice();
    next[index] = { ...next[index], ...changes };
    this.entries_ = next;
    this.changed.dispatch();
  }

  /** Remove the ROI at `index` (no-op if out of range). */
  removeRoi(index: number): void {
    if (index < 0 || index >= this.entries_.length) return;
    const next = this.entries_.slice();
    next.splice(index, 1);
    this.entries_ = next;
    this.changed.dispatch();
  }

  /**
   * Move the ROI at `from` to position `to`. Order is the whole of the
   * left-fold syntax, so this is the primary editing operation.
   */
  moveRoi(from: number, to: number): void {
    const n = this.entries_.length;
    if (from < 0 || from >= n || to < 0 || to >= n || from === to) return;
    const next = this.entries_.slice();
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    this.entries_ = next;
    this.changed.dispatch();
  }

  toJSON(): any {
    if (!this.active_ && this.entries_.length === 0) {
      return undefined; // stays out of the URL when unused
    }
    const json: any = { rois: this.entries_.map(entryToJson) };
    if (this.active_) json.active = true;
    if (this.ghostAlpha_ !== DEFAULT_GHOST_ALPHA)
      json.ghostAlpha = this.ghostAlpha_;
    if (this.colorByGroup_) json.colorByGroup = true;
    return json;
  }

  restoreState(x: unknown): void {
    if (x === undefined) {
      this.reset();
      return;
    }
    verifyObject(x);
    this.entries_ =
      verifyOptionalObjectProperty(x, "rois", (v) =>
        parseArray(v, entryFromJson),
      ) ?? [];
    this.active_ =
      verifyOptionalObjectProperty(x, "active", (v) => v === true) ?? false;
    this.ghostAlpha_ =
      verifyOptionalObjectProperty(x, "ghostAlpha", verifyFiniteFloat) ??
      DEFAULT_GHOST_ALPHA;
    this.colorByGroup_ =
      verifyOptionalObjectProperty(x, "colorByGroup", (v) => v === true) ??
      false;
    this.changed.dispatch();
  }

  reset(): void {
    const wasDefault =
      !this.active_ &&
      !this.colorByGroup_ &&
      this.entries_.length === 0 &&
      this.ghostAlpha_ === DEFAULT_GHOST_ALPHA;
    this.active_ = false;
    this.ghostAlpha_ = DEFAULT_GHOST_ALPHA;
    this.colorByGroup_ = false;
    this.entries_ = [];
    if (!wasDefault) this.changed.dispatch();
  }
}
