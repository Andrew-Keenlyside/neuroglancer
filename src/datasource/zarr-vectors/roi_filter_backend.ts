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
 * @file Per-chunk ROI crossing test, run in the backend worker.
 *
 * A pass-1 chunk holds a slice of many streamlines, one fragment per
 * `(object, chunk)`. This computes, for each object present in the chunk, a
 * boolean vector of which ROIs its geometry *in this chunk* crosses.
 *
 * The verdict is deliberately NOT decided here: an object spans chunks, so its
 * crossing vector must be OR-ed across all its chunks before the operator fold
 * ({@link combineRoiVerdicts}) is applied. The accumulator that owns that merge
 * lives with the backend render layer; this module is the pure per-chunk step,
 * so it unit-tests against a synthetic chunk with no render dependency.
 */

import type { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import type {
  LabelSampler,
  Roi,
  RoiAttrFilter,
  RoiGroupConfig,
  RoiObjectAttrColumn,
  StreamlineRef,
} from "#src/datasource/zarr-vectors/roi.js";
import {
  combineRoiVerdicts,
  RoiPredicate,
  streamlinePassesRoi,
} from "#src/datasource/zarr-vectors/roi.js";
import { packColor } from "#src/util/color.js";
import { vec4 } from "#src/util/geom.js";

/** JS mirror of the GLSL `colormapJet` in `src/webgl/colormaps.ts`, so a tract's
 * CPU-packed colour-by-attribute colour matches the shader's colourmap. */
function colormapJet(x: number): [number, number, number] {
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  const r = x < 0.89 ? (x - 0.35) / 0.31 : 1.0 - ((x - 0.89) / 0.11) * 0.5;
  const g = x < 0.64 ? (x - 0.125) * 4.0 : 1.0 - (x - 0.64) / 0.27;
  const b = x < 0.34 ? 0.5 + (x * 0.5) / 0.11 : 1.0 - (x - 0.34) / 0.31;
  return [clamp(r), clamp(g), clamp(b)];
}

/** One attribute's id→value lookup plus its data bounds (for normalisation). */
interface AttrLookup {
  readonly map: ReadonlyMap<bigint, number>;
  readonly min: number;
  readonly max: number;
}

/**
 * Build id→value lookups for exactly the per-object attributes the visible
 * groups reference (via a length range or an `objectAttr` colour), once per
 * recompute. Attributes not yet loaded are simply absent, and the caller
 * degrades gracefully (no length narrowing / falls back to the group colour).
 */
function buildAttrLookups(
  groups: readonly RoiGroupConfig[],
  attrColumns: ReadonlyMap<string, RoiObjectAttrColumn> | undefined,
): Map<string, AttrLookup> {
  const lookups = new Map<string, AttrLookup>();
  if (attrColumns === undefined) return lookups;
  const need = new Set<string>();
  for (const g of groups) {
    if (!g.visible) continue;
    for (const f of g.attrFilters ?? []) {
      if (f.scope !== "vertex") need.add(f.name);
    }
    if (g.colorBy?.kind === "objectAttr") need.add(g.colorBy.name);
  }
  for (const name of need) {
    const col = attrColumns.get(name);
    if (col === undefined) continue;
    const map = new Map<bigint, number>();
    const n = Math.min(col.ids.length, col.values.length);
    for (let i = 0; i < n; ++i) map.set(col.ids[i], col.values[i]);
    lookups.set(name, { map, min: col.min, max: col.max });
  }
  return lookups;
}

/** Pack `colormapJet(normalise(value))` at the given alpha byte, as the shared
 * `roiSegmentColors` map (id→packed RGBA) expects. */
function packAttrColor(
  value: number,
  min: number,
  max: number,
  alpha255: number,
): number {
  const t = max > min ? (value - min) / (max - min) : 0;
  const [r, g, b] = colormapJet(t);
  return packColor(vec4.fromValues(r, g, b, alpha255 / 255));
}

/** The subset of a decoded `SkeletonChunk` the ROI test reads. */
export interface RoiFilterableChunk {
  readonly rank: number;
  readonly numVertices: number;
  /** Flat `(numVertices, rank)` vertex positions in tract model space. */
  readonly positions: Float32Array;
  /** Per-vertex `uvec2` segment column: `[lo, hi]` per vertex (uint64 split). */
  readonly segmentIds?: Uint32Array;
  readonly fragmentIndex: FragmentIndex;
  /**
   * Whether one VERTEX is one object, rather than one fragment being one
   * object's slice.
   *
   * True for the kinds with no object model (`point_cloud`), where the segment
   * column holds a distinct id per vertex. It is not an optimisation but a
   * correctness switch: a point-cloud fragment is a spatial BIN of unrelated
   * points, so the per-fragment fold below -- which reads the id of a
   * fragment's first vertex and attributes the whole fragment to it -- would
   * put a whole bin in whatever region one of its points happened to land in.
   */
  readonly perVertexObjects?: boolean;
  /**
   * Whether this chunk's vertices bound a SURFACE rather than tracing a curve.
   *
   * A fragment's vertices are then a triangle soup in arbitrary order, so the
   * polyline-shaped predicates do not describe anything real: consecutive
   * vertices are not joined, and there is no "first"/"last" vertex to be an
   * endpoint. Every predicate is evaluated as {@link RoiPredicate.ANY_VERTEX}
   * for such a chunk.
   */
  readonly surfaceVertices?: boolean;
  /**
   * Per-vertex attribute values by on-disk attribute name, each of length
   * {@link numVertices}, retained for vertex-scope attribute predicates (see
   * `RoiAttrFilter`). Present only where there is no per-OBJECT tier to read
   * instead; absent everywhere else, so no store pays the retention for a
   * feature it does not use.
   */
  readonly vertexAttributes?: ReadonlyMap<string, Float32Array>;
}

/** Read the uint64 segment id of vertex `v` from the `uvec2` segment column. */
function segmentIdOf(segmentIds: Uint32Array, v: number): bigint {
  const lo = BigInt(segmentIds[v * 2] >>> 0);
  const hi = BigInt(segmentIds[v * 2 + 1] >>> 0);
  return (hi << 32n) | lo;
}

/**
 * The predicate to actually evaluate for a chunk: a surface's vertices are not
 * a walk, so every predicate collapses to "any vertex inside" there. See
 * {@link RoiFilterableChunk.surfaceVertices}.
 */
function effectivePredicate(
  chunk: RoiFilterableChunk,
  predicate: RoiPredicate,
): RoiPredicate {
  return chunk.surfaceVertices === true ? RoiPredicate.ANY_VERTEX : predicate;
}

/**
 * Per object present in `chunk`, the OR (over that object's fragments in this
 * chunk) of a `crossed[i]` vector: whether the fragment crosses `rois[i]` under
 * its predicate.
 *
 * Returns an empty map when the chunk carries no segment column (no way to
 * attribute geometry to an object) or when `rois` is empty (nothing to test;
 * the caller treats "no ROIs" as "no filter").
 *
 * A range fragment references `chunk.positions` directly; an explicit
 * (non-contiguous) fragment is gathered into a reused scratch buffer first,
 * since {@link StreamlineRef} assumes a contiguous vertex run.
 *
 * A chunk whose objects are VERTICES ({@link RoiFilterableChunk.perVertexObjects})
 * skips the fragment index entirely and tests each vertex on its own -- see that
 * flag for why folding a point-cloud fragment would be wrong.
 */
export function computeChunkCrossings(
  chunk: RoiFilterableChunk,
  rois: readonly Roi[],
  sampleLabel?: LabelSampler,
): Map<bigint, boolean[]> {
  const result = new Map<bigint, boolean[]>();
  const { segmentIds, fragmentIndex, positions, rank } = chunk;
  if (rois.length === 0 || segmentIds === undefined) return result;

  if (chunk.perVertexObjects === true) {
    // One vertex, one object: a `count: 1` run. Every predicate degrades to
    // "is this point inside" on it (`streamlinePassesRoi` falls back to the
    // vertex test for a single-vertex run), so the ROI's own predicate needs no
    // special-casing.
    for (let v = 0; v < chunk.numVertices; ++v) {
      const id = segmentIdOf(segmentIds, v);
      let crossed = result.get(id);
      if (crossed === undefined) {
        crossed = new Array<boolean>(rois.length).fill(false);
        result.set(id, crossed);
      }
      const ref: StreamlineRef = { positions, start: v, count: 1, rank };
      for (let i = 0; i < rois.length; ++i) {
        if (crossed[i]) continue;
        if (
          streamlinePassesRoi(
            ref,
            rois[i].shape,
            rois[i].predicate,
            sampleLabel,
          )
        ) {
          crossed[i] = true;
        }
      }
    }
    return result;
  }

  let scratch = new Float32Array(0);
  const numFragments = fragmentIndex.numFragments;
  for (let f = 0; f < numFragments; ++f) {
    let ref: StreamlineRef;
    let firstVertex: number;
    if (fragmentIndex.isRange(f)) {
      const { start, count } = fragmentIndex.range(f);
      if (count === 0) continue;
      ref = { positions, start, count, rank };
      firstVertex = start;
    } else {
      const indices = fragmentIndex.indices(f);
      if (indices.length === 0) continue;
      const needed = indices.length * rank;
      if (scratch.length < needed) scratch = new Float32Array(needed);
      for (let i = 0; i < indices.length; ++i) {
        const src = indices[i] * rank;
        const dst = i * rank;
        for (let d = 0; d < rank; ++d) scratch[dst + d] = positions[src + d];
      }
      ref = { positions: scratch, start: 0, count: indices.length, rank };
      firstVertex = indices[0];
    }

    // All vertices of a fragment belong to one object; read the id once.
    const id = segmentIdOf(segmentIds, firstVertex);

    let crossed = result.get(id);
    if (crossed === undefined) {
      crossed = new Array<boolean>(rois.length).fill(false);
      result.set(id, crossed);
    }
    for (let i = 0; i < rois.length; ++i) {
      if (crossed[i]) continue; // already crossed by an earlier fragment
      if (
        streamlinePassesRoi(
          ref,
          rois[i].shape,
          effectivePredicate(chunk, rois[i].predicate),
          sampleLabel,
        )
      ) {
        crossed[i] = true;
      }
    }
  }
  return result;
}

/** The set changes to apply after folding in one chunk's crossings. */
export interface PassingSetDiff {
  /** Objects that newly satisfy the filter. */
  readonly added: bigint[];
  /** Objects that no longer satisfy it (e.g. an exclusion chunk arrived). */
  readonly removed: bigint[];
}

/**
 * Accumulates per-chunk crossings across the whole loaded set and maintains the
 * passing-object set, emitting the incremental diff each chunk causes.
 *
 * Deciding "passing" cannot be done per chunk once ROIs compose: an object may
 * cross inclusion region A in one chunk (provisionally passing) and only later,
 * when a different chunk loads, be found to also cross an exclusion region — at
 * which point it must be *removed*. So crossings are OR-merged per object across
 * all chunks and the operator fold ({@link combineRoiVerdicts}) is re-evaluated
 * whenever an object's crossing vector grows. Crossings are monotonic within a
 * fixed ROI set, so an object's verdict only flips in one direction per ROI
 * change; {@link reset} clears everything when the ROI list changes.
 */
export class RoiFilterAccumulator {
  private crossings = new Map<bigint, boolean[]>();
  private passing = new Set<bigint>();

  constructor(private rois: readonly Roi[]) {}

  /** Fold one chunk's crossings in, returning the resulting passing-set diff. */
  addChunk(chunkCrossings: Map<bigint, boolean[]>): PassingSetDiff {
    const added: bigint[] = [];
    const removed: bigint[] = [];
    const n = this.rois.length;
    for (const [id, chunkCrossed] of chunkCrossings) {
      let acc = this.crossings.get(id);
      if (acc === undefined) {
        acc = new Array<boolean>(n).fill(false);
        this.crossings.set(id, acc);
      }
      let grew = false;
      for (let i = 0; i < n; ++i) {
        if (chunkCrossed[i] && !acc[i]) {
          acc[i] = true;
          grew = true;
        }
      }
      // A verdict can only change when the crossing vector grew, or when this
      // object is seen for the first time (grew is true then anyway).
      if (!grew) continue;
      const verdict = combineRoiVerdicts(acc, this.rois);
      const was = this.passing.has(id);
      if (verdict && !was) {
        this.passing.add(id);
        added.push(id);
      } else if (!verdict && was) {
        this.passing.delete(id);
        removed.push(id);
      }
    }
    return { added, removed };
  }

  /** The current passing-object set. */
  get passingSet(): ReadonlySet<bigint> {
    return this.passing;
  }

  /** Drop all accumulated state (call when the ROI list changes). */
  reset(): void {
    this.crossings.clear();
    this.passing.clear();
  }
}

/**
 * Full recompute of the passing-object set from a batch of chunks.
 *
 * Unlike {@link RoiFilterAccumulator}, which folds chunks in incrementally and
 * cannot forget a chunk, this recomputes from scratch over exactly the chunks
 * given. The render layer uses it when it cannot cheaply diff the resident set
 * (an ROI edit, an LOD switch, an eviction): it is simpler and robustly correct
 * to rebuild from the current residents than to track per-chunk removals, and
 * at the interactive (coarse-LOD) operating point the chunk count is small.
 *
 * An empty ROI list yields an empty set (the caller treats "no ROIs" as "no
 * filter" — every streamline passes, so nothing is ghosted, so no id need be
 * listed as passing).
 */
export function computePassingSet(
  chunks: Iterable<RoiFilterableChunk>,
  rois: readonly Roi[],
  sampleLabel?: LabelSampler,
): Set<bigint> {
  const accumulator = new RoiFilterAccumulator(rois);
  if (rois.length !== 0) {
    for (const chunk of chunks) {
      accumulator.addChunk(computeChunkCrossings(chunk, rois, sampleLabel));
    }
  }
  return new Set(accumulator.passingSet);
}

/**
 * The add/remove delta to turn `current` into `target`. Returned as arrays so
 * the caller can apply them to a shared {@link Uint64Set} in two batched RPC
 * mutations rather than one per id. Empty arrays mean the sets already match —
 * the caller then does nothing, which is what makes a redundant recompute (e.g.
 * the second of two render-layer backends sharing one passing set) free.
 */
export function diffPassingSet(
  target: ReadonlySet<bigint>,
  current: ReadonlySet<bigint>,
): { added: bigint[]; removed: bigint[] } {
  const added: bigint[] = [];
  const removed: bigint[] = [];
  for (const id of target) {
    if (!current.has(id)) added.push(id);
  }
  for (const id of current) {
    if (!target.has(id)) removed.push(id);
  }
  return { added, removed };
}

/**
 * The object ids whose resident geometry satisfies every VERTEX-scope filter.
 *
 * An object qualifies when ANY of its resident vertices passes all of them at
 * once -- the same "any part of it is in there" rule the geometric fold uses,
 * and for a point cloud (one vertex, one object) simply the point's own test.
 * Note the "at once": two filters are satisfied by one vertex, not by two
 * different vertices of the same object, so `gene_A > 1 AND gene_B > 1` means
 * co-expression in one cell rather than in one tract.
 *
 * A chunk missing one of the named columns contributes nothing rather than
 * everything: the predicate cannot be evaluated there, and silently passing the
 * whole chunk would turn a filter on an unloaded attribute into "select all".
 * The picker only offers loaded attributes, so this is reachable only from a
 * restored state whose `#attributes=` no longer names the column.
 */
export function computeVertexAttrPassingSet(
  chunks: Iterable<RoiFilterableChunk>,
  filters: readonly RoiAttrFilter[],
): Set<bigint> {
  const passing = new Set<bigint>();
  if (filters.length === 0) return passing;
  for (const chunk of chunks) {
    const { segmentIds, vertexAttributes, numVertices } = chunk;
    if (segmentIds === undefined || vertexAttributes === undefined) continue;
    const columns: Float32Array[] = [];
    let complete = true;
    for (const f of filters) {
      const column = vertexAttributes.get(f.name);
      if (column === undefined) {
        complete = false;
        break;
      }
      columns.push(column);
    }
    if (!complete) continue;
    for (let v = 0; v < numVertices; ++v) {
      let ok = true;
      for (let i = 0; i < filters.length; ++i) {
        const value = columns[i][v];
        if (!(value >= filters[i].min && value <= filters[i].max)) {
          ok = false;
          break;
        }
      }
      if (ok) passing.add(segmentIdOf(segmentIds, v));
    }
  }
  return passing;
}

/**
 * Every object id whose per-OBJECT values satisfy `filters`, enumerated from
 * the columns themselves rather than from resident geometry.
 *
 * This is how an attribute-only group can select objects that are not on screen
 * at all: the column has a value for every object in the store. Returns
 * `undefined` when the leading column has not been loaded yet, which the caller
 * treats as "nothing to seed from" (the group shows empty until it arrives)
 * rather than as "everything".
 */
function objectAttrPassingSet(
  filters: readonly RoiAttrFilter[],
  lookups: ReadonlyMap<string, AttrLookup>,
): Set<bigint> | undefined {
  const seed = lookups.get(filters[0].name);
  if (seed === undefined) return undefined;
  const passing = new Set<bigint>();
  for (const [id, value] of seed.map) {
    if (!(value >= filters[0].min && value <= filters[0].max)) continue;
    passing.add(id);
  }
  narrowByObjectAttrs(passing, filters.slice(1), lookups);
  return passing;
}

/**
 * Drop from `ids` those whose per-object value is out of range, in place.
 *
 * A missing value (attribute not yet loaded, or an object absent from the
 * column) is left IN rather than dropped, so a group never blinks out while its
 * per-object values are still arriving -- the same rule the single length
 * filter this generalises always applied.
 */
function narrowByObjectAttrs(
  ids: Set<bigint>,
  filters: readonly RoiAttrFilter[],
  lookups: ReadonlyMap<string, AttrLookup>,
): void {
  for (const f of filters) {
    const lookup = lookups.get(f.name);
    if (lookup === undefined) continue;
    for (const id of ids) {
      const value = lookup.map.get(id);
      if (value !== undefined && !(value >= f.min && value <= f.max)) {
        ids.delete(id);
      }
    }
  }
}

/**
 * One group's members over `chunks`: the ROI fold narrowed by the group's
 * attribute predicates, or -- with no ROIs -- the predicates alone.
 *
 * The three seeds are ordered by how much they can enumerate: ROIs when the
 * group has any, else the vertex-scope predicates over resident geometry, else
 * the object-scope predicates over the whole column. A group with neither ROIs
 * nor predicates selects nothing (an empty ROI list has always meant "not a
 * dissection", and that is what keeps a freshly added, still-empty group from
 * claiming the entire store).
 */
function groupPassingSet(
  chunks: readonly RoiFilterableChunk[],
  group: RoiGroupConfig,
  lookups: ReadonlyMap<string, AttrLookup>,
  sampleLabel?: LabelSampler,
): Set<bigint> {
  const filters = group.attrFilters ?? [];
  const vertexFilters = filters.filter((f) => f.scope === "vertex");
  const objectFilters = filters.filter((f) => f.scope !== "vertex");
  const hasRois = group.rois.length !== 0;
  if (!hasRois && filters.length === 0) return new Set<bigint>();

  let ids: Set<bigint>;
  if (hasRois) {
    ids = computePassingSet(chunks, group.rois, sampleLabel);
    if (vertexFilters.length !== 0) {
      const byAttr = computeVertexAttrPassingSet(chunks, vertexFilters);
      for (const id of ids) {
        if (!byAttr.has(id)) ids.delete(id);
      }
    }
  } else if (vertexFilters.length !== 0) {
    ids = computeVertexAttrPassingSet(chunks, vertexFilters);
  } else {
    ids = objectAttrPassingSet(objectFilters, lookups) ?? new Set<bigint>();
    return ids; // objectAttrPassingSet already applied every object filter
  }
  narrowByObjectAttrs(ids, objectFilters, lookups);
  return ids;
}

/**
 * Evaluate several ROI groups against one batch of chunks. Each visible group
 * is an independent dissection (its own include/or/exclude fold), so groups are
 * computed separately — flattening them into one fold would wrongly let one
 * group's exclusion drop another group's tracts.
 *
 * Returns the union of all visible groups' passing objects (what the ghost
 * shader keeps visible) and, per passing object, the packed colour of the FIRST
 * (topmost) visible group it belongs to — the group-colour attribution the
 * frontend paints tracts with. Empty/invisible groups contribute nothing.
 */
export function computeGroupedPassingSet(
  chunks: Iterable<RoiFilterableChunk>,
  groups: readonly RoiGroupConfig[],
  attrColumns?: ReadonlyMap<string, RoiObjectAttrColumn>,
  sampleLabel?: LabelSampler,
): {
  passing: Set<bigint>;
  colorById: Map<bigint, number>;
} {
  const passing = new Set<bigint>();
  const colorById = new Map<bigint, number>();
  // The chunk iterable may be single-use (a generator); materialise it once so
  // every group sees the same batch.
  const chunkArray = Array.isArray(chunks) ? chunks : [...chunks];
  const lookups = buildAttrLookups(groups, attrColumns);
  for (const group of groups) {
    if (!group.visible) continue;
    const colorBy = group.colorBy ?? { kind: "group" };
    // Per-vertex colouring (direction/position/vertex attribute) varies along
    // the tract, so the group writes NO colour override and the shader colours
    // it per vertex. Flat kinds (group / object attribute) write an override.
    const perVertex =
      colorBy.kind === "direction" ||
      colorBy.kind === "position" ||
      colorBy.kind === "vertexAttr";
    const colorLookup =
      colorBy.kind === "objectAttr" ? lookups.get(colorBy.name) : undefined;
    // Group opacity rides the packed colour's alpha byte; reuse it for the
    // object-attribute colourmap so the two colouring modes share one opacity.
    const alpha255 = (group.colorPacked >>> 24) & 0xff;
    for (const id of groupPassingSet(chunkArray, group, lookups, sampleLabel)) {
      const firstClaim = !passing.has(id);
      passing.add(id);
      // First group in list order owns the colour (topmost, deterministic).
      if (!firstClaim || perVertex) continue;
      if (colorLookup !== undefined) {
        const v = colorLookup.map.get(id);
        if (v !== undefined) {
          colorById.set(
            id,
            packAttrColor(v, colorLookup.min, colorLookup.max, alpha255),
          );
          continue;
        }
      }
      // "group" colour, or "objectAttr" fallback when the value is missing.
      colorById.set(id, group.colorPacked);
    }
  }
  return { passing, colorById };
}

/**
 * Per group, the set of object ids that pass that group's dissection over
 * `chunks` -- the same include/or/exclude fold and length narrowing
 * {@link computeGroupedPassingSet} applies, but kept SEPARATE per group rather
 * than merged into a union.
 *
 * This is what the tract export needs: the export tab lets the user tick
 * individual groups and reports a per-group count, so it cannot use the shared
 * union passing set (which discards which group each id came from). The result
 * is positional -- `result[i]` corresponds to `groups[i]` -- because group names
 * are not unique (see the note in `tract_export/run.py`).
 *
 * Unlike {@link computeGroupedPassingSet} this does NOT skip invisible groups:
 * the caller passes exactly the groups it wants evaluated (the export selection),
 * and visibility is a rendering concern, not an export one. Groups with neither
 * ROIs nor attribute predicates yield an empty set.
 */
export function computePerGroupPassingSets(
  chunks: Iterable<RoiFilterableChunk>,
  groups: readonly RoiGroupConfig[],
  attrColumns?: ReadonlyMap<string, RoiObjectAttrColumn>,
  sampleLabel?: LabelSampler,
): Set<bigint>[] {
  // Materialise once so every group sees the same batch (the iterable may be a
  // single-use generator).
  const chunkArray = Array.isArray(chunks) ? chunks : [...chunks];
  const lookups = buildAttrLookups(groups, attrColumns);
  return groups.map((group) =>
    groupPassingSet(chunkArray, group, lookups, sampleLabel),
  );
}
