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

/**
 * How a set of streamlines is coloured. Defined here (the dependency-free
 * module) rather than in `roi_filter_state.ts` so the worker-facing
 * `RoiGroupConfig` can reference it without an import cycle; `roi_filter_state`
 * re-exports it and owns its JSON encoding.
 *
 * - `direction`  — the shader default (colour-by-tangent) `prop_tangent()`.
 * - `group`      — the group's flat colour swatch (the old "colour by group").
 * - `position`   — per-vertex, intrinsic xyz → RGB.
 * - `objectAttr` — one flat colour per streamline from a per-object numeric
 *   attribute (e.g. `length`) through a colourmap.
 * - `vertexAttr` — per-vertex, a declared `vertex_attributes/<name>` colourmap.
 */
export type RoiColorSpec =
  | { readonly kind: "direction" }
  | { readonly kind: "group" }
  | { readonly kind: "position" }
  | { readonly kind: "objectAttr"; readonly name: string }
  | { readonly kind: "vertexAttr"; readonly name: string };

/** A closed length range on a named per-object numeric attribute. */
export interface RoiLengthFilter {
  readonly name: string;
  readonly min: number;
  readonly max: number;
}

/**
 * Where an attribute predicate reads its value.
 *
 * - `"object"`: one value per OBJECT, from the store's `object_attributes/`
 *   (surfaced as the layer's segment-property map and shipped to the worker as
 *   a {@link RoiObjectAttrColumn}). Complete: every object has a value whether
 *   or not its geometry is resident.
 * - `"vertex"`: one value per VERTEX, from `vertex_attributes/`, read off the
 *   resident chunk. This is the only tier a `point_cloud` has -- that kind
 *   carries no object model at all -- and for it "one vertex" IS "one object".
 *   On an object geometry a vertex-scope predicate passes an object when ANY of
 *   its resident vertices satisfies it, matching how a geometric ROI is folded.
 *   WYSIWYG over resident chunks and over the attributes the source actually
 *   loaded (`#attributes=`), for the same reason the geometric fold is.
 */
export type RoiAttrScope = "object" | "vertex";

/**
 * A closed range `[min, max]` on one named attribute, used to define group
 * membership by DATA rather than by geometry.
 *
 * Booleans have no separate spelling: a boolean column reaches the reader as
 * 0/1 (every per-vertex attribute is decoded to float32), so "is true" is the
 * range `[0.5, +inf)` and "is false" is `(-inf, 0.5]`. The UI decides which
 * control to draw from the attribute's dtype and observed values; the predicate
 * itself stays one shape, which is what keeps the fold, the JSON and the Python
 * wire format from growing a second case.
 */
export interface RoiAttrFilter {
  readonly name: string;
  readonly min: number;
  readonly max: number;
  /** Absent means `"object"` -- the tier the legacy `lengthFilter` read. */
  readonly scope?: RoiAttrScope;
}

/**
 * Resolved background (whole-tractogram) shader uniforms for the per-object
 * value tier: the length-filter discard range and the flat colour-by-attribute
 * mode, both already normalised to [0,1] against the active attribute's bounds
 * (the same normalisation baked into the shipped per-object value map).
 */
export interface RoiBackgroundUniforms {
  /** Whether to discard background tracts whose value is outside `[lo, hi]`. */
  readonly lengthActive: boolean;
  /** Normalised length range for the discard test. */
  readonly lo: number;
  readonly hi: number;
  /** Whether to recolour background tracts by the attribute (colourmap). */
  readonly colorMode: boolean;
}

/**
 * A voxelised region defined by anatomical labels in a *linked segmentation
 * layer* (e.g. a parcellation). A streamline "passes" it when a vertex falls in
 * a voxel whose label is one of `labels`. Unlike the geometric primitives this
 * is not a closed-form shape: it is evaluated by sampling a dense label field
 * ({@link LabelSampler}) shipped alongside the ROIs, so the crossing test needs
 * that sampler and the pure geometry helpers below never see it.
 *
 * This is what lets a dissection be authored by *toggling parcellation labels*
 * (include/exclude) rather than by drawing spheres and boxes.
 */
export interface LabelMaskShape {
  readonly kind: "labelMask";
  /** Parcellation label ids this region matches (matches any one of them). */
  readonly labels: readonly number[];
}

/** Regions a streamline can be tested against. All geometric ones are axis-aligned. */
export type RoiShape = GeometricRoiShape | LabelMaskShape;

/** The closed-form (sampler-free) regions. Sampled directly against geometry. */
export type GeometricRoiShape =
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
  /**
   * How this region folds into the verdict accumulated so far. The *first*
   * region in a list seeds the verdict instead of folding into one: `AND`
   * seeds it to "passes this region", `ANDNOT` to "does not pass this region",
   * which is how a dissection made only of exclusions is expressed. `OR` is
   * degenerate in that position and is treated as `AND`. See
   * {@link seedVerdict}.
   */
  readonly operator: RoiOperator;
  /**
   * Optional display name. Absent means unnamed, and the UI shows a positional
   * `ROI ${index + 1}` placeholder instead -- deliberately *not* stored, since
   * persisting it would freeze "ROI 3" onto a region that a later delete moves
   * to index 0.
   *
   * Display-only: it reaches neither the fold nor the Python wire format.
   */
  readonly name?: string;
}

/**
 * One group of ROIs, evaluated together as a single dissection, in the plain
 * serialisable form the worker consumes. A streamline belongs to the group iff
 * it passes the group's `rois` (the {@link streamlinePassesRois} include/or/
 * exclude fold). Every visible group's members are shown; a member is coloured
 * by `colorPacked` (packed RGBA), which is also its ROI overlays' colour.
 */
export interface RoiGroupConfig {
  readonly rois: readonly Roi[];
  /** Packed RGBA: rgb = group colour, a = group opacity (colour-by-group + per-group opacity ride this one value). */
  readonly colorPacked: number;
  readonly visible: boolean;
  /**
   * How this group's passing streamlines are coloured. Absent is treated as the
   * legacy `{kind:"group"}` (flat group colour). `"objectAttr"` recolours each
   * tract from a per-object attribute through a colourmap; the per-vertex kinds
   * (`direction`/`position`/`vertexAttr`) leave the tract out of the colour
   * override so the shader colours it per vertex.
   */
  readonly colorBy?: RoiColorSpec;
  /**
   * Attribute predicates restricting this group, ANDed together (see
   * {@link RoiAttrFilter}).
   *
   * Unlike the ROI fold these are not geometry, so they are the one way to
   * define a group for a store that has no geometry to draw regions around --
   * "the cells expressing this gene", "the objects flagged high-quality".
   *
   * Composition: a group with ROIs *and* predicates keeps the objects passing
   * BOTH. A group with predicates and NO ROIs is a pure attribute group and its
   * members are everything satisfying the predicates (this is what makes an
   * attribute group possible at all -- an empty ROI list used to mean "select
   * nothing"). A group with neither still contributes nothing.
   */
  readonly attrFilters?: readonly RoiAttrFilter[];
}

/**
 * One per-object numeric attribute column, as shipped to the worker for
 * length-filtering and colour-by-object-attribute. `ids[i]` (an object id in the
 * same space as the per-vertex segment column) has value `values[i]`; `min`/`max`
 * are the column's data bounds, used to normalise for the colourmap.
 */
export interface RoiObjectAttrColumn {
  readonly ids: BigUint64Array;
  readonly values: Float32Array;
  readonly min: number;
  readonly max: number;
}

/** A streamline: `count` vertices from `start`, in a flat `rank`-strided array. */
export interface StreamlineRef {
  readonly positions: Float32Array;
  /** Index of the first vertex (in vertices, not floats). */
  readonly start: number;
  readonly count: number;
  readonly rank: number;
}

/**
 * Reads the anatomical label at a point in tract model space, for a
 * {@link LabelMaskShape}. Returns `0` (conventionally "unlabelled/background")
 * for a point outside the label field. The x/y/z arguments are model-space
 * coordinates (the same frame the tract vertices and geometric ROIs live in);
 * an implementation folds in whatever world→voxel transform its field needs.
 */
export type LabelSampler = (x: number, y: number, z: number) => number;

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
  shape: GeometricRoiShape,
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
  shape: GeometricRoiShape,
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

/**
 * Whether any of a streamline's tested vertices lands in a voxel labelled with
 * one of `shape.labels`. A label field is sampled per vertex (there is no
 * closed-form region), so `sampleLabel` must be supplied; without it — the field
 * has not loaded yet — no streamline can be decided to pass, so the group is
 * momentarily empty rather than wrong.
 *
 * `ANY_SEGMENT` degrades to a per-vertex test here: sampling the discrete voxel
 * grid continuously along an edge is not worth its cost, and at parcellation
 * resolution (~1 mm) versus tract vertex spacing the vertex test is faithful.
 */
function streamlinePassesLabelMask(
  streamline: StreamlineRef,
  shape: LabelMaskShape,
  predicate: RoiPredicate,
  sampleLabel: LabelSampler | undefined,
): boolean {
  const { positions, start, count, rank } = streamline;
  if (count === 0 || sampleLabel === undefined || shape.labels.length === 0) {
    return false;
  }
  const labels = shape.labels;
  const hit = (v: number): boolean => {
    const o = v * rank;
    const label = sampleLabel(
      positions[o],
      positions[o + 1],
      rank > 2 ? positions[o + 2] : 0,
    );
    if (label === 0) return false;
    for (let i = 0; i < labels.length; ++i)
      if (labels[i] === label) return true;
    return false;
  };
  const first = start;
  const last = start + count - 1;
  switch (predicate) {
    case RoiPredicate.EITHER_ENDPOINT:
      return hit(first) || hit(last);
    case RoiPredicate.BOTH_ENDPOINTS:
      return hit(first) && hit(last);
    default:
      for (let v = 0; v < count; ++v) if (hit(start + v)) return true;
      return false;
  }
}

/**
 * A dense anatomical label grid plus the transform that places tract model-space
 * points into it. Built once on the frontend from a linked segmentation
 * (parcellation) layer and shipped to the worker, where {@link makeLabelSampler}
 * turns it into the {@link LabelSampler} the label-mask crossing test reads.
 *
 * Structured-clone-safe (a typed array + a small matrix), so it rides the same
 * shared-watchable channel as the ROI groups.
 */
export interface RoiLabelField {
  /**
   * Dense label ids, x fastest: the voxel `(vx, vy, vz)` is
   * `data[vx + dims[0] * (vy + dims[1] * vz)]`. Widened to uint32 regardless of
   * the source dtype so the worker has one representation (labels beyond 2^32
   * are out of scope — parcellation ids are small).
   */
  readonly data: Uint32Array;
  /** Grid extent in voxels along model x, y, z. */
  readonly dims: readonly [number, number, number];
  /**
   * Row-major 4×4 mapping a model-space point `[x, y, z, 1]` to continuous voxel
   * coordinates; the sampler rounds to the nearest voxel and indexes `data`.
   */
  readonly modelToVoxel: Float32Array;
}

/**
 * Build the per-vertex label reader for a field. Nearest-neighbour: rounds the
 * mapped voxel coordinate and returns `0` outside the grid, matching the
 * "0 = background" convention {@link streamlinePassesRoi} relies on.
 */
export function makeLabelSampler(field: RoiLabelField): LabelSampler {
  const { data, modelToVoxel: m } = field;
  const [nx, ny, nz] = field.dims;
  return (x, y, z) => {
    const vx = Math.round(m[0] * x + m[1] * y + m[2] * z + m[3]);
    const vy = Math.round(m[4] * x + m[5] * y + m[6] * z + m[7]);
    const vz = Math.round(m[8] * x + m[9] * y + m[10] * z + m[11]);
    if (vx < 0 || vy < 0 || vz < 0 || vx >= nx || vy >= ny || vz >= nz)
      return 0;
    return data[vx + nx * (vy + ny * vz)];
  };
}

/** Whether one streamline passes one region under `predicate`. */
export function streamlinePassesRoi(
  streamline: StreamlineRef,
  shape: RoiShape,
  predicate: RoiPredicate,
  sampleLabel?: LabelSampler,
): boolean {
  if (shape.kind === "labelMask") {
    return streamlinePassesLabelMask(streamline, shape, predicate, sampleLabel);
  }
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
 * How the first region in a list seeds the fold, given whether the streamline
 * passes it.
 *
 * `AND` seeds the verdict to "passes this region". `ANDNOT` seeds it to the
 * negation, which is what makes a dissection of pure exclusions expressible:
 * "everything except the tracts crossing here" needs no leading
 * include-everything region. `OR` is degenerate in this position -- folding it
 * into an empty verdict is `false || x === x` -- so it behaves as `AND` rather
 * than being a third seeding mode.
 *
 * Both folds below route their seed through this, so they cannot disagree about
 * what a leading `ANDNOT` means; the Python port in
 * `neuroglancer/tractography/roi.py` mirrors it.
 */
function seedVerdict(passes: boolean, operator: RoiOperator): boolean {
  return operator === RoiOperator.ANDNOT ? !passes : passes;
}

/**
 * Whether a streamline survives a list of regions.
 *
 * The list is evaluated as a left fold, in order: the first region seeds the
 * verdict (see {@link seedVerdict}) and each subsequent one folds in. This is
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
  sampleLabel?: LabelSampler,
): boolean {
  if (rois.length === 0) return true;
  let verdict = seedVerdict(
    streamlinePassesRoi(
      streamline,
      rois[0].shape,
      rois[0].predicate,
      sampleLabel,
    ),
    rois[0].operator,
  );
  for (let i = 1; i < rois.length; ++i) {
    const roi = rois[i];
    // AND/ANDNOT can only ever clear a false verdict, and OR can only ever set
    // a true one, so the test is skippable in those cases.
    if (verdict === false && roi.operator !== RoiOperator.OR) continue;
    if (verdict === true && roi.operator === RoiOperator.OR) continue;
    const passes = streamlinePassesRoi(
      streamline,
      roi.shape,
      roi.predicate,
      sampleLabel,
    );
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

/**
 * The same left-fold as {@link streamlinePassesRois}, but over a pre-computed
 * per-region crossing vector rather than the geometry.
 *
 * `crossed[i]` is whether the streamline crosses `rois[i]` (under its
 * predicate). Splitting the crossing test from the fold is what makes the
 * per-chunk filter correct: a streamline that spans several chunks may cross
 * region A in one chunk and region B in another, so `crossed[]` must be OR-ed
 * across all of its fragments *before* the fold is applied. Evaluating the
 * whole `AND` fold against a single fragment would wrongly drop it.
 *
 * `crossed.length` must equal `rois.length`. An empty list passes everything.
 */
export function combineRoiVerdicts(
  crossed: readonly boolean[],
  rois: readonly Roi[],
): boolean {
  if (rois.length === 0) return true;
  if (crossed.length !== rois.length) {
    throw new Error(
      `combineRoiVerdicts: crossed has ${crossed.length} entries, expected ${rois.length}`,
    );
  }
  let verdict = seedVerdict(crossed[0], rois[0].operator);
  for (let i = 1; i < rois.length; ++i) {
    switch (rois[i].operator) {
      case RoiOperator.AND:
        verdict = verdict && crossed[i];
        break;
      case RoiOperator.OR:
        verdict = verdict || crossed[i];
        break;
      case RoiOperator.ANDNOT:
        verdict = verdict && !crossed[i];
        break;
    }
  }
  return verdict;
}

/**
 * Axis-aligned world-space box enclosing every region in `rois`, or `undefined`
 * when any of them is unbounded.
 *
 * Unions over **all** operators, including `ANDNOT`. That differs from the
 * exporter's equivalent, and deliberately: this answers "which geometry must be
 * present to decide the crossing tests", and deciding that a tract crosses an
 * exclusion region needs that region's geometry just as much as an inclusion's.
 * (The exporter's version asks a different question -- which tracts can
 * possibly pass -- for which an exclusion's complement is unbounded.)
 *
 * A halfspace has no finite extent, so any list containing one yields
 * `undefined`: the caller then has no bounded guarantee to offer and falls back
 * to whatever the camera happens to make resident.
 */
export function roiRegionBounds(
  rois: readonly Roi[],
): { lower: Float32Array; upper: Float32Array } | undefined {
  let lower: Float32Array | undefined;
  let upper: Float32Array | undefined;
  for (const { shape } of rois) {
    let lo: Float32Array;
    let hi: Float32Array;
    switch (shape.kind) {
      case "ellipsoid": {
        const { center, radii } = shape;
        lo = new Float32Array(center.length);
        hi = new Float32Array(center.length);
        for (let i = 0; i < center.length; ++i) {
          const r = Math.abs(radii[i]);
          lo[i] = center[i] - r;
          hi[i] = center[i] + r;
        }
        break;
      }
      case "box": {
        lo = new Float32Array(shape.lower.length);
        hi = new Float32Array(shape.upper.length);
        for (let i = 0; i < lo.length; ++i) {
          lo[i] = Math.min(shape.lower[i], shape.upper[i]);
          hi[i] = Math.max(shape.lower[i], shape.upper[i]);
        }
        break;
      }
      case "halfspace":
        return undefined;
      case "labelMask":
        // No closed-form extent: the region is wherever the parcellation places
        // those labels. Give no bounded guarantee, so the caller evaluates the
        // dissection over whatever chunks the camera makes resident (WYSIWYG).
        return undefined;
    }
    if (lower === undefined || upper === undefined) {
      lower = lo;
      upper = hi;
      continue;
    }
    for (let i = 0; i < lower.length && i < lo.length; ++i) {
      lower[i] = Math.min(lower[i], lo[i]);
      upper[i] = Math.max(upper[i], hi[i]);
    }
  }
  return lower === undefined || upper === undefined
    ? undefined
    : { lower, upper };
}
