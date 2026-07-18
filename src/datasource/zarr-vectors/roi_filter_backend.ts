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
  Roi,
  RoiGroupConfig,
  StreamlineRef,
} from "#src/datasource/zarr-vectors/roi.js";
import {
  combineRoiVerdicts,
  streamlinePassesRoi,
} from "#src/datasource/zarr-vectors/roi.js";

/** The subset of a decoded `SkeletonChunk` the ROI test reads. */
export interface RoiFilterableChunk {
  readonly rank: number;
  readonly numVertices: number;
  /** Flat `(numVertices, rank)` vertex positions in tract model space. */
  readonly positions: Float32Array;
  /** Per-vertex `uvec2` segment column: `[lo, hi]` per vertex (uint64 split). */
  readonly segmentIds?: Uint32Array;
  readonly fragmentIndex: FragmentIndex;
}

/** Read the uint64 segment id of vertex `v` from the `uvec2` segment column. */
function segmentIdOf(segmentIds: Uint32Array, v: number): bigint {
  const lo = BigInt(segmentIds[v * 2] >>> 0);
  const hi = BigInt(segmentIds[v * 2 + 1] >>> 0);
  return (hi << 32n) | lo;
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
 */
export function computeChunkCrossings(
  chunk: RoiFilterableChunk,
  rois: readonly Roi[],
): Map<bigint, boolean[]> {
  const result = new Map<bigint, boolean[]>();
  const { segmentIds, fragmentIndex, positions, rank } = chunk;
  if (rois.length === 0 || segmentIds === undefined) return result;

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
      if (streamlinePassesRoi(ref, rois[i].shape, rois[i].predicate)) {
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
): Set<bigint> {
  const accumulator = new RoiFilterAccumulator(rois);
  if (rois.length !== 0) {
    for (const chunk of chunks) {
      accumulator.addChunk(computeChunkCrossings(chunk, rois));
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
): { passing: Set<bigint>; colorById: Map<bigint, number> } {
  const passing = new Set<bigint>();
  const colorById = new Map<bigint, number>();
  // The chunk iterable may be single-use (a generator); materialise it once so
  // every group sees the same batch.
  const chunkArray = Array.isArray(chunks) ? chunks : [...chunks];
  for (const group of groups) {
    if (!group.visible || group.rois.length === 0) continue;
    for (const id of computePassingSet(chunkArray, group.rois)) {
      passing.add(id);
      // First group in list order wins the colour (topmost, deterministic).
      if (!colorById.has(id)) colorById.set(id, group.colorPacked);
    }
  }
  return { passing, colorById };
}
