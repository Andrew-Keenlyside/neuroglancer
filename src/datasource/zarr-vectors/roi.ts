/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
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
 * TrackVis-style region-of-interest tests for streamlines.
 *
 * A streamline is a polyline: a run of vertices with an implicit edge between
 * consecutive ones. The tests here answer "does this polyline pass through
 * this region", and compose several regions into one verdict.
 *
 * The geometry is deliberately pure and free of any zarr/render dependency so
 * it can be unit-tested directly, and so the same definitions can be restated
 * in a compiled kernel without the two drifting apart in meaning.
 */

/** Regions a streamline can be tested against. All are axis-aligned. */
export type RoiShape =
  /**
   * Axis-aligned ellipsoid. A sphere is `radii = (r, r, r)`; a disc/slab is one
   * radius much smaller than the others. This is the only bounded primitive
   * needed: sphere, disc and slab are all choices of its three radii, which
   * avoids a family of near-duplicate shapes.
   */
  | {
      readonly kind: "ellipsoid";
      readonly center: Float32Array;
      readonly radii: Float32Array;
    }
  /** Axis-aligned bounding box, `lower <= upper` per axis. */
  | {
      readonly kind: "box";
      readonly lower: Float32Array;
      readonly upper: Float32Array;
    }
  /**
   * Infinite half-space `{x : dot(normal, x - origin) >= 0}`. `normal` need not
   * be unit length. Used as a *filter*; a clipping plane that truncates
   * rendering is a separate display concern.
   */
  | {
      readonly kind: "halfspace";
      readonly origin: Float32Array;
      readonly normal: Float32Array;
    };

/**
 * What it means for a streamline to "pass" a region.
 *
 * `ANY_SEGMENT` is the default and the only one robust to vertex spacing: it
 * tests the polyline itself, not its samples. TrackVis's historical test is
 * `ANY_VERTEX`, which is only safe while the step size is small relative to
 * the region -- a track can otherwise step clean over a small sphere without
 * ever landing a vertex inside it. (DSI Studio calls this the "leap-across
 * artifact"; its documented workaround, enlarging the region, is a symptom of
 * the wrong test rather than a fix.) `ANY_VERTEX` is kept for compatibility
 * with results produced by those tools.
 */
export const enum RoiPredicate {
  /** Any point of any edge lies within the region. Exact for a polyline. */
  ANY_SEGMENT = 0,
  /** Any vertex lies within the region. Sensitive to vertex spacing. */
  ANY_VERTEX = 1,
  /** The first or last vertex lies within: tracts *terminating* in a region. */
  EITHER_ENDPOINT = 2,
  /** Both endpoints lie within. */
  BOTH_ENDPOINTS = 3,
}

/** How a region combines with the verdict accumulated so far. */
export const enum RoiOperator {
  /** Keep streamlines that pass this region too. */
  AND = 0,
  /** Also keep streamlines that pass this region. */
  OR = 1,
  /** Drop streamlines that pass this region (an exclusion region). */
  ANDNOT = 2,
}

export interface Roi {
  readonly shape: RoiShape;
  readonly predicate: RoiPredicate;
  /** Ignored for the first region in a list, which seeds the verdict. */
  readonly operator: RoiOperator;
}

/** A streamline: `count` vertices from `start`, in a flat `rank`-strided array. */
export interface StreamlineRef {
  readonly positions: Float32Array;
  /** Index of the first vertex (in vertices, not floats). */
  readonly start: number;
  readonly count: number;
  readonly rank: number;
}

/** Squared distance from the origin to the segment `a -> b`. */
function pointToSegmentDistanceSq(
  a: Float64Array,
  b: Float64Array,
  rank: number,
): number {
  // Project the origin onto the segment, clamping to its ends.
  let abDotAb = 0;
  let aDotAb = 0;
  for (let i = 0; i < rank; ++i) {
    const d = b[i] - a[i];
    abDotAb += d * d;
    aDotAb += -a[i] * d;
  }
  let t = abDotAb > 0 ? aDotAb / abDotAb : 0;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  let distSq = 0;
  for (let i = 0; i < rank; ++i) {
    const p = a[i] + t * (b[i] - a[i]);
    distSq += p * p;
  }
  return distSq;
}

/** Scratch buffers, reused across calls to keep the hot path allocation-free. */
const scratchA = new Float64Array(4);
const scratchB = new Float64Array(4);

/** Whether a vertex lies inside the shape. */
function vertexInside(
  shape: RoiShape,
  positions: Float32Array,
  offset: number,
  rank: number,
): boolean {
  switch (shape.kind) {
    case "ellipsoid": {
      // Normalise by the radii: the ellipsoid becomes the unit sphere.
      let sum = 0;
      for (let i = 0; i < rank; ++i) {
        const r = shape.radii[i];
        if (r <= 0) return false;
        const d = (positions[offset + i] - shape.center[i]) / r;
        sum += d * d;
      }
      return sum <= 1;
    }
    case "box": {
      for (let i = 0; i < rank; ++i) {
        const v = positions[offset + i];
        if (v < shape.lower[i] || v > shape.upper[i]) return false;
      }
      return true;
    }
    case "halfspace": {
      let dot = 0;
      for (let i = 0; i < rank; ++i) {
        dot += shape.normal[i] * (positions[offset + i] - shape.origin[i]);
      }
      return dot >= 0;
    }
  }
}

/** Whether the segment between two vertices touches the shape. */
function segmentIntersects(
  shape: RoiShape,
  positions: Float32Array,
  offsetA: number,
  offsetB: number,
  rank: number,
): boolean {
  switch (shape.kind) {
    case "ellipsoid": {
      // Map both endpoints into the space where the ellipsoid is the unit
      // sphere centred at the origin, then ask for the distance from the
      // origin to the segment. Exact, ~15 flops, and no tolerance to tune.
      for (let i = 0; i < rank; ++i) {
        const r = shape.radii[i];
        if (r <= 0) return false;
        scratchA[i] = (positions[offsetA + i] - shape.center[i]) / r;
        scratchB[i] = (positions[offsetB + i] - shape.center[i]) / r;
      }
      return pointToSegmentDistanceSq(scratchA, scratchB, rank) <= 1;
    }
    case "box": {
      // Slab method: intersect the segment's parametric range with each axis's
      // slab; the segment misses iff the ranges ever become disjoint.
      let tMin = 0;
      let tMax = 1;
      for (let i = 0; i < rank; ++i) {
        const a = positions[offsetA + i];
        const b = positions[offsetB + i];
        const d = b - a;
        if (Math.abs(d) < 1e-20) {
          // Parallel to this slab: inside it for all t, or for none.
          if (a < shape.lower[i] || a > shape.upper[i]) return false;
          continue;
        }
        let t0 = (shape.lower[i] - a) / d;
        let t1 = (shape.upper[i] - a) / d;
        if (t0 > t1) {
          const tmp = t0;
          t0 = t1;
          t1 = tmp;
        }
        if (t0 > tMin) tMin = t0;
        if (t1 < tMax) tMax = t1;
        if (tMin > tMax) return false;
      }
      return true;
    }
    case "halfspace": {
      // Convex and unbounded: the segment touches it iff either endpoint does.
      return (
        vertexInside(shape, positions, offsetA, rank) ||
        vertexInside(shape, positions, offsetB, rank)
      );
    }
  }
}

/** Whether one streamline passes one region under `predicate`. */
export function streamlinePassesRoi(
  streamline: StreamlineRef,
  shape: RoiShape,
  predicate: RoiPredicate,
): boolean {
  const { positions, start, count, rank } = streamline;
  if (count === 0) return false;
  if (rank > scratchA.length) {
    throw new Error(
      `streamlinePassesRoi: rank ${rank} exceeds supported maximum`,
    );
  }
  const first = start * rank;
  const last = (start + count - 1) * rank;

  switch (predicate) {
    case RoiPredicate.EITHER_ENDPOINT:
      return (
        vertexInside(shape, positions, first, rank) ||
        vertexInside(shape, positions, last, rank)
      );
    case RoiPredicate.BOTH_ENDPOINTS:
      return (
        vertexInside(shape, positions, first, rank) &&
        vertexInside(shape, positions, last, rank)
      );
    case RoiPredicate.ANY_VERTEX:
      for (let v = 0; v < count; ++v) {
        if (vertexInside(shape, positions, (start + v) * rank, rank))
          return true;
      }
      return false;
    case RoiPredicate.ANY_SEGMENT: {
      // A single-vertex streamline has no edge; fall back to the vertex.
      if (count === 1) return vertexInside(shape, positions, first, rank);
      for (let v = 0; v + 1 < count; ++v) {
        const offsetA = (start + v) * rank;
        if (
          segmentIntersects(shape, positions, offsetA, offsetA + rank, rank)
        ) {
          return true;
        }
      }
      return false;
    }
  }
}

/**
 * Whether a streamline survives a list of regions.
 *
 * The list is evaluated as a left fold, in order: the first region seeds the
 * verdict (its operator is ignored) and each subsequent one folds in. This is
 * how TrackVis and DSI Studio express dissections -- neither offers a nested
 * expression tree, and a left fold covers the dissections found in practice --
 * so region order is the whole of the syntax, and reordering the list is the
 * only editing operation a user needs.
 *
 * An empty list passes everything: no regions means no filtering, rather than
 * filtering everything away.
 */
export function streamlinePassesRois(
  streamline: StreamlineRef,
  rois: readonly Roi[],
): boolean {
  if (rois.length === 0) return true;
  let verdict = streamlinePassesRoi(
    streamline,
    rois[0].shape,
    rois[0].predicate,
  );
  for (let i = 1; i < rois.length; ++i) {
    const roi = rois[i];
    // AND/ANDNOT can only ever clear a false verdict, and OR can only ever set
    // a true one, so the test is skippable in those cases.
    if (verdict === false && roi.operator !== RoiOperator.OR) continue;
    if (verdict === true && roi.operator === RoiOperator.OR) continue;
    const passes = streamlinePassesRoi(streamline, roi.shape, roi.predicate);
    switch (roi.operator) {
      case RoiOperator.AND:
        verdict = verdict && passes;
        break;
      case RoiOperator.OR:
        verdict = verdict || passes;
        break;
      case RoiOperator.ANDNOT:
        verdict = verdict && !passes;
        break;
    }
  }
  return verdict;
}
