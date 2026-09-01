/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * @file What a layer can be filtered BY: the attributes an ROI group's
 * predicates ({@link RoiAttrFilter}) may name, and the observed distribution of
 * each one.
 *
 * Two tiers exist and the filter tab has to offer both, because which one a
 * store has is decided by its geometry kind, not by the user:
 *
 * - **per-object** — the store's `object_attributes/`, surfaced as the layer's
 *   segment-property map. Its values are already in frontend memory and cover
 *   every object, on screen or not.
 * - **per-vertex** — the store's `vertex_attributes/`. For a `point_cloud`
 *   (one vertex = one cell) this is the ONLY tier: that kind has no object
 *   model at all. The values live in the worker, so their range comes back over
 *   an RPC, measured on the same resident chunks the filter folds.
 *
 * The range matters as much as the name. ZVF declares no bounds for an
 * attribute, so a slider has nothing to span until the data has been looked at,
 * and the same look is what tells a FLAG (`high_quality_transfer`: two integer
 * values) from a measurement (a gene's expression) -- which is the difference
 * between drawing a checkbox and drawing two sliders.
 *
 * Pure enough to unit-test: the layer only ever reaches this module through the
 * narrow structural types below, never as a `SegmentationUserLayer`.
 */

import type {
  RoiAttrFilter,
  RoiAttrScope,
} from "#src/datasource/zarr-vectors/roi.js";
import type { VertexAttrStats } from "#src/skeleton/base.js";
import { DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";
import { NullarySignal } from "#src/util/signal.js";

/** The observed distribution of one attribute, whichever tier it came from. */
export interface AttrStats extends VertexAttrStats {
  readonly scope: RoiAttrScope;
  /**
   * The column's declared dtype, lower-cased (`"uint8"`, `"float32"`), when the
   * layer knows it.
   *
   * Needed because what was OBSERVED is not the whole truth about a flag: a
   * boolean column can be all-true over the chunks currently loaded, which reads
   * as one distinct value and would otherwise be offered as a (useless) range
   * from 1 to 1. The dtype says it is still a flag.
   */
  readonly dtype?: string;
}

/** One filterable attribute: what the picker lists. */
export interface AttrChoice {
  readonly name: string;
  readonly scope: RoiAttrScope;
}

/** `name` + `scope` as one map key (the two tiers may share a name). */
export function attrKey(name: string, scope: RoiAttrScope): string {
  return `${scope}:${name}`;
}

/** The scope a predicate names, defaulting as {@link RoiAttrFilter} documents. */
export function filterScope(filter: RoiAttrFilter): RoiAttrScope {
  return filter.scope ?? "object";
}

/** How many distinct values to count before stopping; mirrors the backend's. */
const DISTINCT_LIMIT = 64;

/** The subset of a segment-property map this module reads. */
interface NumericalPropertyLike {
  readonly id: string;
  readonly values: ArrayLike<number>;
  readonly bounds: readonly [number | bigint, number | bigint];
  /** `DataType` enum member, if the property map carries one. */
  readonly dataType?: number;
}

/** The subset of a segmentation layer this module reads. */
export interface AttrCatalogLayer {
  readonly displayState: {
    readonly segmentPropertyMap: {
      readonly value?: {
        readonly numericalProperties?: NumericalPropertyLike[];
      };
    };
    readonly roiVertexAttributeNames?: readonly string[];
    /** On-disk dtypes parallel to {@link roiVertexAttributeNames}. */
    readonly roiVertexAttributeDtypes?: readonly string[];
    readonly computeRoiVertexAttrStats?: (
      names: readonly string[],
    ) => Promise<VertexAttrStats[]>;
  };
}

/**
 * Every attribute the layer can be filtered by, per-object tier first.
 *
 * Order is the offer order in the picker, and per-object comes first
 * deliberately: those values are complete (they describe objects the view has
 * not loaded too), so where a store has both tiers the more truthful one is the
 * first thing a user meets.
 */
export function listAttrChoices(layer: AttrCatalogLayer): AttrChoice[] {
  const choices: AttrChoice[] = [];
  for (const property of numericalProperties(layer)) {
    choices.push({ name: property.id, scope: "object" });
  }
  for (const name of layer.displayState.roiVertexAttributeNames ?? []) {
    choices.push({ name, scope: "vertex" });
  }
  return choices;
}

function numericalProperties(
  layer: AttrCatalogLayer,
): readonly NumericalPropertyLike[] {
  return layer.displayState.segmentPropertyMap.value?.numericalProperties ?? [];
}

/**
 * Measure one per-object column from the values already in frontend memory.
 *
 * `bounds` would give the range on its own, but not whether the column is a
 * flag, so the scan happens anyway and the range comes from the same pass --
 * one source of truth rather than two that can disagree.
 */
export function measureObjectAttr(property: NumericalPropertyLike): AttrStats {
  const { values } = property;
  let count = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let integral = true;
  const seen = new Set<number>();
  for (let i = 0; i < values.length; ++i) {
    const value = values[i];
    if (!Number.isFinite(value)) continue;
    ++count;
    if (value < min) min = value;
    if (value > max) max = value;
    if (integral && !Number.isInteger(value)) integral = false;
    if (seen.size < DISTINCT_LIMIT) seen.add(value);
  }
  const dtype =
    property.dataType === undefined
      ? undefined
      : DataType[property.dataType]?.toLowerCase();
  return count === 0
    ? {
        name: property.id,
        scope: "object",
        count: 0,
        min: 0,
        max: 0,
        integral: true,
        distinct: 0,
        dtype,
      }
    : {
        name: property.id,
        scope: "object",
        count,
        min,
        max,
        integral,
        distinct: seen.size,
        dtype,
      };
}

/**
 * Whether a column's dtype is narrow enough to be a flag: one byte, signed or
 * not. Wider integers hold category codes far more often than booleans, and
 * mistaking one for a flag would hide 20 parcellation labels behind a checkbox.
 */
function isNarrowIntDtype(dtype: string | undefined): boolean {
  return dtype === "uint8" || dtype === "int8";
}

/**
 * Whether to offer this attribute as a FLAG (a checkbox) rather than a range.
 *
 * Two cases, and the second is not optional:
 *
 * - Two whole-number values and nothing else. A boolean reaches the reader as
 *   0/1 (every per-vertex attribute decodes to float32), and a `uint8` mask may
 *   use 0/255 instead; both are flags.
 * - ONE whole-number value in [0, 1] from a one-byte column. A boolean that
 *   happens to be all-true over the chunks currently loaded observes a single
 *   value, and offering it as a range from 1 to 1 would be a control that can
 *   express nothing. `high_quality_transfer` in the MERFISH panel is exactly
 *   this.
 *
 * A three-valued category code stays a range either way.
 */
export function isFlagAttr(stats: AttrStats): boolean {
  if (stats.count === 0 || !stats.integral) return false;
  if (stats.distinct === 2) return true;
  return (
    stats.distinct === 1 &&
    isNarrowIntDtype(stats.dtype) &&
    stats.min >= 0 &&
    stats.max <= 1
  );
}

/**
 * The two states' bounds and the value between them.
 *
 * Taken from the observed extremes, so a 0/255 mask splits at 127.5 exactly as
 * a 0/1 boolean splits at 0.5. When only one value was observed those extremes
 * are the same number and cannot express two states, so the canonical boolean
 * encoding stands in -- which is sound precisely because that case is only
 * reached for a one-byte column whose values are already known to be 0 or 1.
 * Both ends stay finite, so the predicate round-trips through the state JSON's
 * `verifyFiniteFloat`.
 */
function flagSplit(stats: AttrStats): { lo: number; mid: number; hi: number } {
  if (stats.min < stats.max) {
    return { lo: stats.min, mid: (stats.min + stats.max) / 2, hi: stats.max };
  }
  return { lo: 0, mid: 0.5, hi: 1 };
}

/** The predicate for "this flag is true" / "is false". */
export function flagFilter(stats: AttrStats, value: boolean): RoiAttrFilter {
  const { lo, mid, hi } = flagSplit(stats);
  const base = value
    ? { name: stats.name, min: mid, max: hi }
    : { name: stats.name, min: lo, max: mid };
  return stats.scope === "vertex" ? { ...base, scope: "vertex" } : base;
}

/** Which state {@link flagFilter} encoded (its `true` branch starts at the midpoint). */
export function flagFilterValue(
  filter: RoiAttrFilter,
  stats: AttrStats,
): boolean {
  return filter.min >= flagSplit(stats).mid;
}

/**
 * The largest integer a float32 represents exactly. Above it, consecutive
 * integers share a representation.
 */
const EXACT_INTEGER_LIMIT = 2 ** 24;

/** 64-bit on-disk dtypes: the ones the reader has to downcast to float32. */
const WIDE_DTYPES = new Set(["int64", "uint64", "float64"]);

/**
 * Whether this column's values have lost precision on the way in, so a
 * predicate's ends are approximate rather than exact.
 *
 * Every per-vertex attribute is decoded to float32, so a 64-bit column is
 * downcast. That is harmless for scores, coordinates and small category codes,
 * and NOT harmless for an id-shaped column (a MICrONS `cell_root_id`), where
 * neighbouring ids collapse onto one value. The UI says so rather than
 * refusing: which columns are ids is the user's knowledge, not ours.
 */
export function hasApproximateValues(stats: AttrStats): boolean {
  return (
    WIDE_DTYPES.has(stats.dtype ?? "") &&
    Math.max(Math.abs(stats.min), Math.abs(stats.max)) > EXACT_INTEGER_LIMIT
  );
}

/** A predicate spanning the attribute's whole observed range: a no-op to start from. */
export function fullRangeFilter(stats: AttrStats): RoiAttrFilter {
  const base = { name: stats.name, min: stats.min, max: stats.max };
  return stats.scope === "vertex" ? { ...base, scope: "vertex" } : base;
}

/**
 * Per-attribute stats, fetched once and kept.
 *
 * Lazy on purpose: a MERFISH panel can have a thousand loaded columns, and
 * measuring all of them to populate a `<select>` would scan every resident
 * vertex a thousand times for names the user will never pick. So the picker
 * lists names, and the measurement happens when one is chosen.
 *
 * A pending request is deduplicated, and `changed` fires when an answer lands
 * so a panel can re-render without polling.
 */
export class AttrStatsCache extends RefCounted {
  readonly changed = new NullarySignal();
  private stats = new Map<string, AttrStats>();
  private pending = new Map<string, Promise<AttrStats | undefined>>();

  constructor(private layer: AttrCatalogLayer) {
    super();
  }

  /** The stats for one attribute if already known; `undefined` while pending. */
  get(name: string, scope: RoiAttrScope): AttrStats | undefined {
    return this.stats.get(attrKey(name, scope));
  }

  /**
   * Measure one attribute, from memory (per-object) or over the RPC
   * (per-vertex). Resolves `undefined` when the attribute is not there to
   * measure -- an unknown name, or a layer whose worker link is not up yet.
   */
  async request(
    name: string,
    scope: RoiAttrScope,
  ): Promise<AttrStats | undefined> {
    const key = attrKey(name, scope);
    const known = this.stats.get(key);
    if (known !== undefined) return known;
    const inFlight = this.pending.get(key);
    if (inFlight !== undefined) return inFlight;
    const promise = this.measure(name, scope)
      .then((stats) => {
        this.pending.delete(key);
        if (stats === undefined || this.wasDisposed) return undefined;
        this.stats.set(key, stats);
        this.changed.dispatch();
        return stats;
      })
      .catch(() => {
        // A failed measurement is not an error the user can act on (the worker
        // link may simply not be up yet); leave it unknown so the next open of
        // the panel asks again.
        this.pending.delete(key);
        return undefined;
      });
    this.pending.set(key, promise);
    return promise;
  }

  /**
   * Drop everything measured so far.
   *
   * Per-vertex ranges are measured over the RESIDENT chunks, so they age as the
   * view moves; the tab re-requests after an invalidation rather than showing a
   * slider spanning a range that is no longer on screen.
   */
  invalidate(): void {
    if (this.stats.size === 0) return;
    this.stats.clear();
    this.changed.dispatch();
  }

  private async measure(
    name: string,
    scope: RoiAttrScope,
  ): Promise<AttrStats | undefined> {
    if (scope === "object") {
      const property = numericalProperties(this.layer).find(
        (p) => p.id === name,
      );
      return property === undefined ? undefined : measureObjectAttr(property);
    }
    const compute = this.layer.displayState.computeRoiVertexAttrStats;
    if (compute === undefined) return undefined;
    const [stats] = await compute([name]);
    if (stats === undefined) return undefined;
    // The worker measured the values; the dtype is the frontend's to add (it
    // comes from the source parameters, which the worker's chunk view drops).
    const names = this.layer.displayState.roiVertexAttributeNames ?? [];
    const dtype =
      this.layer.displayState.roiVertexAttributeDtypes?.[names.indexOf(name)];
    return { ...stats, scope: "vertex", dtype };
  }
}
