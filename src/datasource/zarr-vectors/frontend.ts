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

import type { AnnotationGeometryChunkSpecification } from "#src/annotation/base.js";
import {
  AnnotationGeometryChunkSource,
  MultiscaleAnnotationSource,
} from "#src/annotation/frontend_source.js";
import type { AnnotationPropertySpec } from "#src/annotation/index.js";
import {
  AnnotationType,
  parseAnnotationPropertySpecs,
} from "#src/annotation/index.js";
import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import {
  makeCoordinateSpace,
  makeIdentityTransform,
  makeIdentityTransformedBoundingBox,
} from "#src/coordinate_transform.js";
import {
  type DataSource,
  type DataSourceLookupResult,
  type GetKvStoreBasedDataSourceOptions,
  type KvStoreBasedDataSourceProvider,
} from "#src/datasource/index.js";
import type {
  ZarrVectorsAttributeDtype,
  ZarrVectorsPyramidMode,
} from "#src/datasource/zarr-vectors/base.js";
import {
  ZarrVectorsAnnotationSourceParameters,
  ZarrVectorsAnnotationSpatialIndexSourceParameters,
  ZarrVectorsObjectKeyedSkeletonSourceParameters,
  ZarrVectorsSpatiallyIndexedSkeletonSourceParameters,
} from "#src/datasource/zarr-vectors/base.js";
import {
  ZarrVectorsMultiscaleSpatiallyIndexedSkeletonSource,
  ZarrVectorsObjectKeyedSkeletonSource,
} from "#src/datasource/zarr-vectors/skeleton_frontend.js";
import type { AutoDetectRegistry } from "#src/kvstore/auto_detect.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import {
  joinBaseUrlAndPath,
  kvstoreEnsureDirectoryPipelineUrl,
  parseUrlSuffix,
  pipelineUrlJoin,
} from "#src/kvstore/url.js";
import type {
  InlineSegmentNumericalProperty,
  InlineSegmentProperty,
  InlineSegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import {
  normalizeInlineSegmentPropertyMap,
  SegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { makeSliceViewChunkSpecification } from "#src/sliceview/base.js";
import { DataType } from "#src/util/data_type.js";
import * as matrix from "#src/util/matrix.js";
import type { ProgressOptions } from "#src/util/progress_listener.js";
import { ProgressSpan } from "#src/util/progress_listener.js";
import { allSiPrefixes, supportedUnits } from "#src/util/si_units.js";

// ---------------------------------------------------------------
// zstd decompression helper (object_attributes chunks may be zstd-compressed)

const ZSTD_MAGIC = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd]);

/** Keys already warned about, so one diagnostic does not repeat per level. */
const warnedOnceKeys = new Set<string>();
function warnOnceFe(key: string, message: string): void {
  if (warnedOnceKeys.has(key)) return;
  warnedOnceKeys.add(key);
  console.warn(message);
}

function looksLikeZstdFe(bytes: Uint8Array): boolean {
  return (
    bytes.byteLength >= 4 &&
    bytes[0] === ZSTD_MAGIC[0] &&
    bytes[1] === ZSTD_MAGIC[1] &&
    bytes[2] === ZSTD_MAGIC[2] &&
    bytes[3] === ZSTD_MAGIC[3]
  );
}

async function maybeDecompressObjAttr(
  bytes: Uint8Array<ArrayBuffer>,
  signal: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  if (!looksLikeZstdFe(bytes)) return bytes;
  return await requestAsyncComputation(
    decodeZstd,
    signal,
    [bytes.buffer],
    bytes,
  );
}

// ---------------------------------------------------------------
// Chunk source classes
// ---------------------------------------------------------------

class ZarrVectorsAnnotationSpatialIndexSource extends WithParameters(
  WithSharedKvStoreContext(AnnotationGeometryChunkSource),
  ZarrVectorsAnnotationSpatialIndexSourceParameters,
) {}

const MultiscaleAnnotationSourceBase = WithParameters(
  WithSharedKvStoreContext(MultiscaleAnnotationSource),
  ZarrVectorsAnnotationSourceParameters,
);

interface ZarrVectorsAnnotationSourceOptions {
  metadata: AnnotationMetadata;
  parameters: ZarrVectorsAnnotationSourceParameters;
  sharedKvStoreContext: SharedKvStoreContext;
}

export class ZarrVectorsAnnotationSource extends MultiscaleAnnotationSourceBase {
  declare key: unknown;
  metadata: AnnotationMetadata;
  declare OPTIONS: ZarrVectorsAnnotationSourceOptions;
  constructor(
    chunkManager: ChunkManager,
    options: ZarrVectorsAnnotationSourceOptions,
  ) {
    const { parameters } = options;
    super(chunkManager, {
      rank: parameters.rank,
      relationships: [],
      properties: parameters.properties,
      sharedKvStoreContext: options.sharedKvStoreContext,
      parameters,
    } as any);
    this.readonly = true;
    this.metadata = options.metadata;
  }

  getSources() {
    return [
      this.metadata.spatialIndices.map((level) => ({
        chunkSource: this.chunkManager.getChunkSource(
          ZarrVectorsAnnotationSpatialIndexSource,
          {
            sharedKvStoreContext: this.sharedKvStoreContext,
            parent: this,
            spec: level.spec,
            parameters: level.parameters,
          },
        ),
        chunkToMultiscaleTransform: level.spec.chunkToMultiscaleTransform,
      })),
    ];
  }
}

// ---------------------------------------------------------------
// Metadata parsing
// ---------------------------------------------------------------

// NGFF long-form unit strings → base SI letter + decimal exponent.
// Built from neuroglancer's known SI prefix table so any prefix
// understood elsewhere in the codebase round-trips correctly.
const OME_LONG_UNITS = (() => {
  const m = new Map<string, { unit: string; exponent: number }>();
  for (const baseUnit of ["meter", "second"]) {
    for (const p of allSiPrefixes) {
      if (p.longPrefix === undefined) continue;
      m.set(`${p.longPrefix}${baseUnit}`, {
        unit: baseUnit[0],
        exponent: p.exponent,
      });
    }
  }
  // Common irregular forms.
  m.set("micron", { unit: "m", exponent: -6 });
  m.set("microns", { unit: "m", exponent: -6 });
  return m;
})();

/**
 * Translate a (scale, unit) pair from user-facing form to the
 * normalised form neuroglancer's coordinate space expects: unit is one
 * of the base SI letters ("m", "s", "Hz", "rad/s", "") and any SI
 * prefix is folded into scale.  Returns {unit: "", scale} when the
 * unit string isn't recognised.
 */
function normalizeUnitScale(
  rawScale: number,
  rawUnit: unknown,
): { unit: string; scale: number } {
  if (typeof rawUnit !== "string" || rawUnit === "") {
    return { unit: "", scale: rawScale };
  }
  const longForm = OME_LONG_UNITS.get(rawUnit);
  if (longForm !== undefined) {
    return {
      unit: longForm.unit,
      scale: rawScale * 10 ** longForm.exponent,
    };
  }
  const shortForm = supportedUnits.get(rawUnit);
  if (shortForm !== undefined) {
    return {
      unit: shortForm.unit,
      scale: rawScale * 10 ** shortForm.exponent,
    };
  }
  return { unit: "", scale: rawScale };
}

const ATTR_DTYPE_TO_NG_TYPE: Record<string, AnnotationPropertySpec["type"]> = {
  float32: "float32",
  uint8: "uint8",
  uint16: "uint16",
  uint32: "uint32",
  int8: "int8",
  int16: "int16",
  int32: "int32",
};

interface AnnotationSpatialIndexLevelMetadata {
  parameters: ZarrVectorsAnnotationSpatialIndexSourceParameters;
  spec: AnnotationGeometryChunkSpecification;
}

interface AnnotationMetadata {
  rank: number;
  coordinateSpace: ReturnType<typeof makeCoordinateSpace>;
  parameters: ZarrVectorsAnnotationSourceParameters;
  spatialIndices: AnnotationSpatialIndexLevelMetadata[];
}

function buildCoordinateSpaceFromHints(
  hints: any,
  lowerBounds: Float64Array,
  upperBounds: Float64Array,
) {
  const names: string[] = hints.names.map((n: unknown) => String(n));
  const rawScales: number[] = hints.scales.map((s: unknown) => Number(s));
  const units: string[] = new Array(names.length);
  const scales = new Float64Array(names.length);
  for (let i = 0; i < names.length; ++i) {
    const normalized = normalizeUnitScale(rawScales[i], hints.units?.[i]);
    units[i] = normalized.unit;
    scales[i] = normalized.scale;
  }
  return makeCoordinateSpace({
    rank: names.length,
    names,
    units,
    scales,
    boundingBoxes: [
      makeIdentityTransformedBoundingBox({ lowerBounds, upperBounds }),
    ],
  });
}

function buildCoordinateSpaceFromMultiscales(
  multiscales: any,
  lowerBounds: Float64Array,
  upperBounds: Float64Array,
  rank: number,
) {
  const entry = Array.isArray(multiscales) ? multiscales[0] : undefined;
  const axes: any[] = entry?.axes ?? [];
  let names = axes.map((a, i) => (a?.name ? String(a.name) : `d${i}`));
  while (names.length < rank) names.push(`d${names.length}`);
  names = names.slice(0, rank);

  const dataset = entry?.datasets?.[0];
  const scaleXform = dataset?.coordinateTransformations?.find(
    (t: any) => t?.type === "scale",
  );
  const scaleArr: number[] = scaleXform?.scale ?? [];
  const units: string[] = new Array(rank);
  const scales = new Float64Array(rank);
  for (let i = 0; i < rank; ++i) {
    const rawScale = scaleArr[i] !== undefined ? Number(scaleArr[i]) : 1.0;
    const normalized = normalizeUnitScale(rawScale, axes[i]?.unit);
    units[i] = normalized.unit;
    scales[i] = normalized.scale;
  }
  return makeCoordinateSpace({
    rank,
    names,
    units,
    scales,
    boundingBoxes: [
      makeIdentityTransformedBoundingBox({ lowerBounds, upperBounds }),
    ],
  });
}

async function listAttributeNames(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  options: Partial<ProgressOptions>,
): Promise<string[]> {
  const attributesUrl = joinBaseUrlAndPath(levelUrl, "vertex_attributes/");
  let response;
  try {
    response = await sharedKvStoreContext.kvStoreContext.list(attributesUrl, {
      responseKeys: "suffix",
      ...options,
    });
  } catch (e) {
    // A genuinely-absent directory lists as empty (no throw); a throw here means
    // listing failed or is denied (e.g. a bucket without objects.list). Warn
    // rather than silently reporting "no attributes", which would drop
    // attribute-based colouring with no diagnostic.
    warnOnceFe(
      "vertex-attributes-list",
      "zarr-vectors: could not list vertex_attributes/ (object listing may be " +
        "denied on this store); per-vertex attribute-based colouring will be " +
        `unavailable even if attributes exist. ${e instanceof Error ? e.message : String(e)}`,
    );
    return [];
  }
  // Each subdirectory under vertex_attributes/ is one property.  Strip the
  // trailing "/" if present.
  return response.directories
    .map((d) => (d.endsWith("/") ? d.slice(0, -1) : d))
    .filter((d) => d.length > 0)
    .sort();
}

async function getJsonResource(
  sharedKvStoreContext: SharedKvStoreContext,
  url: string,
  description: string,
  options: Partial<ProgressOptions>,
): Promise<any | undefined> {
  return sharedKvStoreContext.chunkManager.memoize.getAsync(
    { type: "zarr-vectors:json", url },
    options,
    async (progressOptions) => {
      using _span = new ProgressSpan(progressOptions.progressListener, {
        message: `Reading ${description} from ${url}`,
      });
      const response = await sharedKvStoreContext.kvStoreContext.read(
        url,
        progressOptions,
      );
      if (response === undefined) return undefined;
      return await response.response.json();
    },
  );
}

/**
 * Read the per-chunk-array grid geometry from a level's `vertices/zarr.json`
 * (v0.9.0 single-array format): `chunk_grid_origin`, whether the array uses the
 * `sharding_indexed` codec, the shard shape, and the chunk-key separator. These
 * drive the backend cell reader (see `shard_cell_reader.ts`). Rejects the
 * pre-0.9.0 "Option G" layout, where the per-array node is a group rather than a
 * single array — those stores are unreadable and must be rewritten from source.
 */
async function readChunkGridParams(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  rank: number,
  options: Partial<ProgressOptions>,
): Promise<{
  chunkGridOrigin: number[];
  sharded: boolean;
  shardChunkShape: number[];
  cellSeparator: string;
}> {
  const json = await getJsonResource(
    sharedKvStoreContext,
    joinBaseUrlAndPath(levelUrl, "vertices/zarr.json"),
    "zarr-vectors vertices array metadata",
    options,
  );
  if (json === undefined || json.node_type !== "array") {
    throw new Error(
      `zarr-vectors: ${levelUrl}vertices is not a single zarr array ` +
        `(node_type=${JSON.stringify(json?.node_type)}). Pre-0.9.0 "Option G" ` +
        `stores are unreadable; rewrite from source with zarr-vectors >= 0.9.0.`,
    );
  }
  const attrs = json.attributes ?? {};
  const originRaw = attrs.chunk_grid_origin;
  const chunkGridOrigin =
    Array.isArray(originRaw) && originRaw.length === rank
      ? originRaw.map((x: any) => Number(x))
      : new Array<number>(rank).fill(0);
  const codecs = json.codecs;
  const sharded =
    Array.isArray(codecs) &&
    codecs.length > 0 &&
    codecs[0]?.name === "sharding_indexed";
  let shardChunkShape: number[] = [];
  if (sharded) {
    const cs = json.chunk_grid?.configuration?.chunk_shape;
    if (!Array.isArray(cs) || cs.length !== rank) {
      throw new Error(
        `zarr-vectors: sharded vertices array at ${levelUrl} is missing a ` +
          `chunk_grid.configuration.chunk_shape of rank ${rank}`,
      );
    }
    shardChunkShape = cs.map((x: any) => Number(x));
  }
  const sep = json.chunk_key_encoding?.configuration?.separator;
  return {
    chunkGridOrigin,
    sharded,
    shardChunkShape,
    cellSeparator: typeof sep === "string" ? sep : "/",
  };
}

async function buildPropertySpecsAndDtypes(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  hints: any,
  options: Partial<ProgressOptions>,
): Promise<{
  properties: AnnotationPropertySpec[];
  attributeNames: string[];
  attributeDtypes: ZarrVectorsAttributeDtype[];
}> {
  const declaredHints: any[] = Array.isArray(hints?.properties)
    ? hints.properties
    : [];
  const declaredByName = new Map<string, any>(
    declaredHints.map((p) => [String(p.identifier), p]),
  );

  const names = await listAttributeNames(
    sharedKvStoreContext,
    levelUrl,
    options,
  );

  // Stable order: declared properties first (in their declared order),
  // then any remaining listed attributes in alphabetical order.
  const orderedNames: string[] = [];
  for (const p of declaredHints) {
    const id = String(p.identifier);
    if (names.includes(id) && !orderedNames.includes(id)) {
      orderedNames.push(id);
    }
  }
  for (const n of names) {
    if (!orderedNames.includes(n)) orderedNames.push(n);
  }

  const attributeNames: string[] = [];
  const attributeDtypes: ZarrVectorsAttributeDtype[] = [];
  const rawPropertyJson: any[] = [];

  for (const name of orderedNames) {
    // `tangent` is reserved for the synthesised per-vertex direction
    // (vec3 float32), exposed as `prop_tangent()` for the default
    // colour-by-direction shader (see skeleton_shader_bridge.ts). A store
    // that ALSO ships its own `tangent` vertex_attribute would redefine
    // the `prop_tangent` macro and — being multi-component (vec3) while the
    // reader packs user attributes as 1-component — make the skeleton
    // shader fail to compile ("illegal vector field selection"), rendering
    // nothing. The synthesised tangent wins; skip the store's copy.
    if (name === "tangent") {
      console.warn(
        'zarr-vectors: ignoring reserved vertex attribute "tangent" — the ' +
          "reader synthesises prop_tangent() for colour-by-direction.",
      );
      continue;
    }
    let dtype: string | undefined;
    let arrayMeta: any | undefined;
    try {
      arrayMeta = await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(levelUrl, `vertex_attributes/${name}/zarr.json`),
        `attribute ${JSON.stringify(name)} metadata`,
        options,
      );
      dtype = arrayMeta?.attributes?.dtype ?? arrayMeta?.data_type ?? undefined;
    } catch {
      dtype = undefined;
    }
    if (dtype === undefined || ATTR_DTYPE_TO_NG_TYPE[dtype] === undefined) {
      // Skip attributes we can't represent.
      continue;
    }
    attributeNames.push(name);
    attributeDtypes.push(dtype as ZarrVectorsAttributeDtype);

    // Dictionary-encoded (categorical/enum) attributes: surface as a
    // numeric annotation property with enum_values + enum_labels so
    // shaders see category names, not raw codes.  The on-disk
    // convention is documented in zarr-vectors:
    // ``encoding: "dictionary"`` + ``categories: [...]`` lives in the
    // attribute array's metadata block.
    let enumValues: number[] | undefined;
    let enumLabels: string[] | undefined;
    if (arrayMeta?.attributes?.encoding === "dictionary") {
      const categories = arrayMeta.attributes.categories;
      if (Array.isArray(categories)) {
        enumLabels = categories.map((c: unknown) => String(c));
        enumValues = enumLabels.map((_, i) => i);
      }
    }

    const hint = declaredByName.get(name);
    if (hint !== undefined) {
      rawPropertyJson.push({
        ...hint,
        type: hint.type ?? dtype,
        // Hints win for enum metadata; only fill in from the on-disk
        // dictionary when the user didn't already specify it.
        enum_values: hint.enum_values ?? enumValues,
        enum_labels: hint.enum_labels ?? enumLabels,
      });
    } else {
      rawPropertyJson.push({
        id: name,
        type: ATTR_DTYPE_TO_NG_TYPE[dtype],
        enum_values: enumValues,
        enum_labels: enumLabels,
      });
    }
  }

  // parseAnnotationPropertySpecs expects "id" — normalise from
  // "identifier" if hints used that key.
  const normalized = rawPropertyJson.map((p) => {
    if (p.id !== undefined) return p;
    if (p.identifier !== undefined) {
      const { identifier, ...rest } = p;
      return { id: identifier, ...rest };
    }
    return p;
  });

  let properties: AnnotationPropertySpec[];
  try {
    properties = parseAnnotationPropertySpecs(normalized);
  } catch (e) {
    throw new Error(
      `Failed to parse annotation property specs from zarr-vectors hints: ${(e as Error).message}`,
    );
  }
  return { properties, attributeNames, attributeDtypes };
}

/**
 * Map a zarr-vectors object-attribute dtype string to a neuroglancer
 * `DataType` plus the typed-array constructor that reinterprets the
 * raw bytes.  Subset that matches what `SegmentPropertyMap`'s numerical
 * properties can carry — `uint64` is excluded (the segment-properties
 * UI rejects it; the existing precomputed parser does the same).
 */
const OBJECT_ATTR_DTYPE_TABLE: Record<
  string,
  {
    dataType: DataType;
    elementSize: number;
    ctor: new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ) =>
      | Float32Array
      | Uint8Array
      | Uint16Array
      | Uint32Array
      | Int8Array
      | Int16Array
      | Int32Array;
  }
> = {
  float32: { dataType: DataType.FLOAT32, elementSize: 4, ctor: Float32Array },
  uint8: { dataType: DataType.UINT8, elementSize: 1, ctor: Uint8Array },
  uint16: { dataType: DataType.UINT16, elementSize: 2, ctor: Uint16Array },
  uint32: { dataType: DataType.UINT32, elementSize: 4, ctor: Uint32Array },
  int8: { dataType: DataType.INT8, elementSize: 1, ctor: Int8Array },
  int16: { dataType: DataType.INT16, elementSize: 2, ctor: Int16Array },
  int32: { dataType: DataType.INT32, elementSize: 4, ctor: Int32Array },
};

/**
 * List per-object attribute names by enumerating subdirectories under
 * the level's `object_attributes/`.  Returns the empty list when the
 * directory is absent (older stores that don't carry per-object
 * attributes).
 */
async function listObjectAttributeNames(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  options: Partial<ProgressOptions>,
): Promise<string[]> {
  const dirUrl = joinBaseUrlAndPath(levelUrl, "object_attributes/");
  let response;
  try {
    response = await sharedKvStoreContext.kvStoreContext.list(dirUrl, {
      responseKeys: "suffix",
      ...options,
    });
  } catch (e) {
    // See listAttributeNames: a throw is a listing failure/denial, not an
    // absent directory, so warn rather than silently dropping the
    // segment-properties panel.
    warnOnceFe(
      "object-attributes-list",
      "zarr-vectors: could not list object_attributes/ (object listing may be " +
        "denied on this store); the segment-properties panel will be unavailable " +
        `even if object attributes exist. ${e instanceof Error ? e.message : String(e)}`,
    );
    return [];
  }
  return response.directories
    .map((d) => (d.endsWith("/") ? d.slice(0, -1) : d))
    .filter((d) => d.length > 0)
    .sort();
}

/**
 * Reinterpret a raw byte blob into a typed array of `dataType`, copying
 * when the source offset isn't aligned to the element size.  Mirrors
 * the chunk-decoder's `reinterpretBytes` so per-object attribute reads
 * follow the same alignment conventions as per-vertex reads.
 */
function reinterpretObjectAttributeBytes(
  bytes: Uint8Array,
  ctor: (typeof OBJECT_ATTR_DTYPE_TABLE)[string]["ctor"],
  elementSize: number,
  expectedElements: number,
):
  | Float32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array {
  const expectedBytes = expectedElements * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors object_attributes: expected ${expectedBytes} bytes ` +
        `(${expectedElements} elements), got ${bytes.byteLength}`,
    );
  }
  if (bytes.byteOffset % elementSize === 0) {
    return new (ctor as any)(bytes.buffer, bytes.byteOffset, expectedElements);
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new (ctor as any)(copy.buffer, 0, expectedElements);
}

/**
 * Reinterpret an 8-byte-per-element blob (float64 / int64 / uint64) as
 * `Float32Array`, downcasting values. The segment-properties columns hold
 * float32 (or the narrow int types), so a float64 tortuosity or int64 vertex
 * count would otherwise be dropped as "exotic"; downcasting keeps them usable
 * for filtering and colouring (precision loss is immaterial for these).
 */
function reinterpretWideToFloat32(
  bytes: Uint8Array,
  kind: "float64" | "int64" | "uint64",
  expectedElements: number,
): Float32Array {
  const elementSize = 8;
  const expectedBytes = expectedElements * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors object_attributes: expected ${expectedBytes} bytes ` +
        `(${expectedElements} elements), got ${bytes.byteLength}`,
    );
  }
  let buffer: ArrayBufferLike = bytes.buffer;
  let offset = bytes.byteOffset;
  if (offset % elementSize !== 0) {
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    buffer = copy.buffer;
    offset = 0;
  }
  const out = new Float32Array(expectedElements);
  if (kind === "float64") {
    const wide = new Float64Array(buffer, offset, expectedElements);
    for (let i = 0; i < expectedElements; ++i) out[i] = wide[i];
  } else {
    const wide =
      kind === "int64"
        ? new BigInt64Array(buffer, offset, expectedElements)
        : new BigUint64Array(buffer, offset, expectedElements);
    for (let i = 0; i < expectedElements; ++i) out[i] = Number(wide[i]);
  }
  return out;
}

/**
 * Build a `SegmentPropertyMap` from level-0 `object_attributes/`.  Each
 * scalar (num_channels=1) attribute becomes one numerical column in
 * neuroglancer's segment-properties UI; the row order maps directly to
 * object_ids `[0, 1, ..., O-1]` (dense layout per spec §6).  Returns
 * `undefined` when the store has no object attributes — the caller
 * skips the subsource entry in that case.
 *
 * Multi-channel (num_channels > 1) attributes are silently dropped:
 * neuroglancer's segment-properties UI is scalar-per-column, and
 * splitting an O×C attribute into C named columns would require a
 * naming convention writers don't currently follow.  (Colouring
 * streamlines by a 3-vector attribute as RGB is a separate path.)
 * Wide scalar dtypes (float64 / int64 / uint64) ARE accepted, downcast
 * to float32 so common computed attributes (tortuosity, counts) are not
 * lost; only genuinely unrepresentable dtypes are skipped.
 *
 * `present_mask` arrays are honoured: rows with mask=0 are dropped
 * from the segment-properties output so absent objects don't appear
 * with zero-padded values.
 */
async function buildObjectAttributePropertyMap(
  sharedKvStoreContext: SharedKvStoreContext,
  level0Url: string,
  options: Partial<ProgressOptions>,
): Promise<SegmentPropertyMap | undefined> {
  const names = await listObjectAttributeNames(
    sharedKvStoreContext,
    level0Url,
    options,
  );
  if (names.length === 0) return undefined;

  // Read each attribute's metadata in parallel.  Skip attributes the
  // segment-properties UI can't represent rather than failing the whole
  // map — keeps the store openable even when one column is exotic.
  type AttrSpec = {
    name: string;
    dataType: DataType;
    values: ReturnType<typeof reinterpretObjectAttributeBytes>;
    presentMask?: Uint8Array;
    numObjects: number;
  };
  const specs = await Promise.all(
    names.map(async (name): Promise<AttrSpec | undefined> => {
      const arrayUrl = joinBaseUrlAndPath(
        level0Url,
        `object_attributes/${name}/`,
      );
      const meta = await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(arrayUrl, "zarr.json"),
        `object_attribute ${JSON.stringify(name)} metadata`,
        options,
      );
      if (meta === undefined) return undefined;
      const attrs = meta?.attributes ?? meta;
      // Semantic dtype rides the group attributes; fall back to the array's own
      // `data_type` for robustness.
      const dtype = String(attrs?.dtype ?? meta?.data_type ?? "");
      // Shape is `[O]` for a scalar column or `[O, C]` for a C-vector; the
      // trailing dim is the channel count. (There is no `num_channels` field.)
      const shape = Array.isArray(attrs?.shape)
        ? attrs.shape
        : Array.isArray(meta?.shape)
          ? meta.shape
          : undefined;
      if (shape === undefined || shape.length === 0) return undefined;
      const numObjects = Number(shape[0]);
      const numChannels =
        shape.length >= 2 ? Number(shape[shape.length - 1]) : 1;
      if (numObjects <= 0) return undefined;
      const entry = OBJECT_ATTR_DTYPE_TABLE[dtype];
      // Wide scalar dtypes are downcast to float32 (see reinterpretWideToFloat32)
      // so a float64 tortuosity or int64 count is still discovered.
      const wideKind =
        dtype === "float64"
          ? "float64"
          : dtype === "int64"
            ? "int64"
            : dtype === "uint64"
              ? "uint64"
              : undefined;
      if (
        numChannels !== 1 ||
        (entry === undefined && wideKind === undefined)
      ) {
        // Multi-channel attributes (e.g. a 3-vector orientation) are not scalar
        // columns; truly exotic dtypes can't be represented — skip both.
        return undefined;
      }
      const dataType = entry?.dataType ?? DataType.FLOAT32;
      // On-disk layout: `object_attributes/<name>/` is itself a single-chunk
      // zarr v3 array (not a group with a nested `data/` array). Its one chunk
      // lives at `c/<0…>/` under the default chunk-key encoding ("/" separator):
      // `c/0` for a scalar `[O]` array, `c/0/0` for a `[O, C]` vector. Read that
      // blob directly; the semantic dtype is carried in the array attributes.
      const chunkKey = `c/${shape.map(() => 0).join("/")}`;
      const dataResponse = await sharedKvStoreContext.kvStoreContext.read(
        joinBaseUrlAndPath(arrayUrl, chunkKey),
        options,
      );
      if (dataResponse === undefined) return undefined;
      const rawBytes = new Uint8Array(
        (await dataResponse.response.arrayBuffer()) as ArrayBuffer,
      );
      // Decompress if the writer applied zstd (the zarr codec pipeline writes
      // zstd by default; we bypass zarr's own chunk decoder here and handle
      // it manually so we can read the semantic dtype from the outer group).
      const bytes = await maybeDecompressObjAttr(
        rawBytes,
        options.signal ?? new AbortController().signal,
      );
      const values =
        entry !== undefined
          ? reinterpretObjectAttributeBytes(
              bytes,
              entry.ctor,
              entry.elementSize,
              numObjects,
            )
          : reinterpretWideToFloat32(bytes, wideKind!, numObjects);
      let presentMask: Uint8Array | undefined;
      if (attrs?.has_present_mask === true) {
        const maskResponse = await sharedKvStoreContext.kvStoreContext.read(
          joinBaseUrlAndPath(arrayUrl, "present_mask/c/0"),
          options,
        );
        if (maskResponse !== undefined) {
          const rawMaskBytes = new Uint8Array(
            (await maskResponse.response.arrayBuffer()) as ArrayBuffer,
          );
          presentMask = await maybeDecompressObjAttr(
            rawMaskBytes,
            options.signal ?? new AbortController().signal,
          );
        }
      }
      return {
        name,
        dataType,
        values,
        presentMask,
        numObjects,
      };
    }),
  );

  // Reconcile object counts across attributes — they MUST agree because
  // every row is keyed by the same global object_id space.  Use the
  // first present-mask-aware count as the reference and union of
  // present masks for the id list.
  const present = specs.filter((s): s is AttrSpec => s !== undefined);
  if (present.length === 0) return undefined;
  const numObjects = present[0].numObjects;
  for (const s of present) {
    if (s.numObjects !== numObjects) {
      throw new Error(
        `zarr-vectors object_attributes: row count mismatch ` +
          `(${s.name}=${s.numObjects} vs ${numObjects})`,
      );
    }
  }
  // Effective present-mask: AND of all attribute masks (a row is
  // emitted only if every attribute considers it real).  In the common
  // case where no attribute carries a mask, every row is kept.
  let effectiveMask: Uint8Array | undefined;
  for (const s of present) {
    if (s.presentMask === undefined) continue;
    if (effectiveMask === undefined) {
      effectiveMask = new Uint8Array(s.presentMask);
    } else {
      for (let i = 0; i < numObjects; ++i) {
        effectiveMask[i] = effectiveMask[i] && s.presentMask[i] ? 1 : 0;
      }
    }
  }
  const keepIndices: number[] = [];
  for (let i = 0; i < numObjects; ++i) {
    if (effectiveMask === undefined || effectiveMask[i] === 1) {
      keepIndices.push(i);
    }
  }
  if (keepIndices.length === 0) return undefined;

  const ids = new BigUint64Array(keepIndices.length);
  for (let i = 0; i < keepIndices.length; ++i) {
    ids[i] = BigInt(keepIndices[i]);
  }

  const properties: InlineSegmentProperty[] = [];
  for (const s of present) {
    const compactValues = new (s.values.constructor as new (
      n: number,
    ) => typeof s.values)(keepIndices.length);
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < keepIndices.length; ++i) {
      const v = s.values[keepIndices[i]];
      (compactValues as unknown as { [k: number]: number })[i] = v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      min = 0;
      max = 0;
    }
    const numerical: InlineSegmentNumericalProperty = {
      id: s.name,
      type: "number",
      dataType: s.dataType,
      description: undefined,
      values:
        compactValues as unknown as InlineSegmentNumericalProperty["values"],
      bounds: [min, max],
    };
    properties.push(numerical);
  }

  const inline: InlineSegmentPropertyMap = normalizeInlineSegmentPropertyMap({
    ids,
    properties,
  });
  return new SegmentPropertyMap({ inlineProperties: inline });
}

const FALLBACK_LEVEL_LIMIT = 1_000_000;

function parsePyramidMode(raw: unknown): ZarrVectorsPyramidMode {
  if (raw === "additive" || raw === "replace") return raw;
  // Default: "replace" — see plan §"Context (v2)".  Safe for
  // metanode-style pyramids (no double-count) and matches the
  // image-style mental model of resolution alternatives.  A single
  // level falls through both branches identically.
  return "replace";
}

function enumerateLevelPaths(multiscales: any): string[] {
  const entry = Array.isArray(multiscales) ? multiscales[0] : undefined;
  const datasets = entry?.datasets;
  if (Array.isArray(datasets) && datasets.length > 0) {
    const paths: string[] = [];
    for (const d of datasets) {
      if (typeof d?.path === "string" && d.path.length > 0) {
        paths.push(d.path);
      }
    }
    if (paths.length > 0) return paths;
  }
  // No multiscales metadata → assume a single level "0".  This
  // preserves v1 behavior for stores written before pyramid support.
  return ["0"];
}

/**
 * Live object count per level, finest-first, or `undefined` per level where it
 * cannot be determined.
 *
 * This is the detail axis of an object-sparsity pyramid: every level covers
 * the whole volume with the same `chunk_shape`, and coarser levels simply hold
 * fewer *complete* objects. It cannot be read off `object_index/manifests`,
 * whose length is the object-id space rather than the live count -- with
 * ``preserves_object_ids`` a dropped object keeps its slot and stores an empty
 * manifest, so every level reports the same shape.
 *
 * So it is derived instead: ``inherited_num_objects`` scaled by the product of
 * ``object_sparsity`` down the chain (each level's sparsity is relative to its
 * parent). Falls back to ``vertex_count``, which tracks object count closely
 * while vertices-per-object stays roughly constant across levels.
 */
function computePerLevelObjectCount(
  perLevelMeta: ReadonlyArray<{
    vertexCount: number | undefined;
    objectSparsity: number | undefined;
    numObjects: number | undefined;
  }>,
): (number | undefined)[] {
  const n = perLevelMeta.length;
  const base = perLevelMeta[0]?.numObjects;
  if (base !== undefined) {
    const out: (number | undefined)[] = [];
    let cumulative = 1;
    let ok = true;
    for (let k = 0; k < n; ++k) {
      const s = perLevelMeta[k].objectSparsity;
      if (s === undefined) {
        ok = false;
        break;
      }
      // Level 0's sparsity is 1.0; each later level's is relative to its parent.
      cumulative *= s;
      out.push(Math.max(1, base * cumulative));
    }
    if (ok) return out;
  }
  return perLevelMeta.map((m) => m.vertexCount);
}

function computeLevelLimit(
  levelAttrs: any,
  numChunks: number,
  level0Limit: number,
  rank: number,
): number {
  const vertexCount = Number(levelAttrs?.vertex_count);
  if (Number.isFinite(vertexCount) && vertexCount > 0 && numChunks > 0) {
    return Math.max(1, Math.ceil(vertexCount / numChunks));
  }
  const binRatio = levelAttrs?.bin_ratio;
  if (Array.isArray(binRatio) && binRatio.length === rank) {
    let prod = 1;
    for (const v of binRatio) prod *= Math.max(1, Number(v) || 1);
    if (prod > 1) return Math.max(1, Math.round(level0Limit / prod));
  }
  return level0Limit;
}

/**
 * The on-disk format version this reader targets. ZVF 0.9.0 was a breaking
 * change (connectivity moved to the `links/<delta>/<offsets>/` family and the
 * single-array chunk layout landed); the spec states pre-0.9 stores are not
 * readable by 0.9+ readers.
 */
const SUPPORTED_ZV_MAJOR = 0;
const MIN_SUPPORTED_ZV_MINOR = 9;

/**
 * Gate on `zarr_vectors.zv_version` so a version-driven layout change surfaces a
 * diagnostic instead of failing silently downstream (as the 0.9 links move did
 * before this reader was migrated). Throws for a store older than this reader
 * supports; warns for a missing or newer-than-target version and proceeds.
 */
function checkZvVersion(zv: any): void {
  const raw = zv?.zv_version;
  if (raw === undefined) {
    console.warn(
      "zarr-vectors: store does not declare zv_version; assuming a ZVF " +
        `${SUPPORTED_ZV_MAJOR}.${MIN_SUPPORTED_ZV_MINOR}-compatible layout.`,
    );
    return;
  }
  const match = /^(\d+)\.(\d+)(?:\.\d+)?$/.exec(String(raw));
  if (match === null) {
    console.warn(
      `zarr-vectors: unrecognised zv_version ${JSON.stringify(raw)}; ` +
        "proceeding as if it were the supported version.",
    );
    return;
  }
  const major = Number(match[1]);
  const minor = Number(match[2]);
  if (
    major < SUPPORTED_ZV_MAJOR ||
    (major === SUPPORTED_ZV_MAJOR && minor < MIN_SUPPORTED_ZV_MINOR)
  ) {
    throw new Error(
      `zarr-vectors: store zv_version ${raw} predates the ${SUPPORTED_ZV_MAJOR}.` +
        `${MIN_SUPPORTED_ZV_MINOR}.0 layout this reader requires (connectivity and ` +
        "chunk layout changed in 0.9.0). Rewrite the store from source with a current " +
        "zarr-vectors writer.",
    );
  }
  if (major > SUPPORTED_ZV_MAJOR || minor > MIN_SUPPORTED_ZV_MINOR) {
    console.warn(
      `zarr-vectors: store zv_version ${raw} is newer than this reader's target ` +
        `${SUPPORTED_ZV_MAJOR}.${MIN_SUPPORTED_ZV_MINOR}; newer features may not render.`,
    );
  }
}

async function buildAnnotationMetadata(
  sharedKvStoreContext: SharedKvStoreContext,
  storeUrl: string,
  rootAttrs: any,
  options: Partial<ProgressOptions>,
  /**
   * When false (default), require `geometry_types` to declare
   * `"point_cloud"` — protects callers that came through the main
   * `zarr-vectors:` dispatcher.  When true, skip the check so the
   * `zarr-vectors-pointcloud:` alias can re-interpret any zarr-vectors
   * store (graph, skeleton, polyline, streamline) as a point cloud by
   * reading only its vertex / vertex_attribute arrays.
   */
  allowAnyGeometry: boolean = false,
): Promise<AnnotationMetadata> {
  const zv = rootAttrs?.zarr_vectors;
  if (zv === undefined) {
    throw new Error(
      "Not a zarr-vectors store: root attributes lack a 'zarr_vectors' block",
    );
  }
  checkZvVersion(zv);
  // Geometry-type validation lives at the dispatch layer
  // (`ZarrVectorsDataSource.get`) — by the time `buildAnnotationMetadata`
  // is called, the dispatcher has already verified that `geometry_types`
  // is `["point_cloud"]`.  Re-validate defensively here so a direct
  // caller (e.g. tests) gets a clear error.  The pointcloud-alias
  // provider passes `allowAnyGeometry=true` to skip this re-validation.
  const geometryTypes: string[] = Array.isArray(zv.geometry_types)
    ? zv.geometry_types
    : [];
  if (!allowAnyGeometry && !geometryTypes.includes("point_cloud")) {
    throw new Error(
      `buildAnnotationMetadata: called for a non-point_cloud store ` +
        `(geometry_types=${JSON.stringify(geometryTypes)})`,
    );
  }
  const bounds = zv.bounds;
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 2 ||
    !Array.isArray(bounds[0]) ||
    !Array.isArray(bounds[1])
  ) {
    throw new Error(
      "zarr-vectors store: 'bounds' must be [[lower...], [upper...]]",
    );
  }
  const rank = bounds[0].length;
  if (bounds[1].length !== rank) {
    throw new Error(
      "zarr-vectors store: bounds[0] and bounds[1] have different rank",
    );
  }
  const chunkShape = zv.chunk_shape;
  if (!Array.isArray(chunkShape) || chunkShape.length !== rank) {
    throw new Error(`zarr-vectors store: 'chunk_shape' must have rank ${rank}`);
  }

  const lowerBounds = Float64Array.from(bounds[0], Number);
  const upperBounds = Float64Array.from(bounds[1], Number);

  // Build coordinate space: prefer neuroglancer hints, fall back to
  // NGFF multiscales.
  const ngHints = rootAttrs.neuroglancer ?? {};
  let coordinateSpace: ReturnType<typeof makeCoordinateSpace>;
  if (
    ngHints.coordinate_space &&
    Array.isArray(ngHints.coordinate_space.names) &&
    ngHints.coordinate_space.names.length === rank
  ) {
    coordinateSpace = buildCoordinateSpaceFromHints(
      ngHints.coordinate_space,
      lowerBounds,
      upperBounds,
    );
  } else {
    coordinateSpace = buildCoordinateSpaceFromMultiscales(
      rootAttrs.multiscales,
      lowerBounds,
      upperBounds,
      rank,
    );
  }

  const pyramidMode = parsePyramidMode(ngHints.pyramid_mode);
  const levelPaths = enumerateLevelPaths(rootAttrs.multiscales);

  // Property discovery runs once against level 0 — per the
  // zarr-vectors spec, attribute dtypes don't vary across levels.
  const level0Url = kvstoreEnsureDirectoryPipelineUrl(
    pipelineUrlJoin(storeUrl, levelPaths[0]),
  );
  const { properties, attributeNames, attributeDtypes } =
    await buildPropertySpecsAndDtypes(
      sharedKvStoreContext,
      level0Url,
      ngHints,
      options,
    );

  // Shared per-level geometry: all levels share the same physical
  // chunk grid on disk in zarr-vectors.
  const chunkShapeF32 = new Float32Array(rank);
  let numChunks = 1;
  for (let i = 0; i < rank; ++i) {
    const cs = Number(chunkShape[i]);
    chunkShapeF32[i] = cs;
    const extent = upperBounds[i] - lowerBounds[i];
    const g = Math.max(1, Math.ceil(extent / cs));
    numChunks *= g;
  }
  // zarr-vectors chunks are indexed around world origin (chunk
  // `(i, j, k)` covers world `[i*chunkShape, (i+1)*chunkShape]`) — see
  // `assign_chunks` in zarr-vectors-py: it computes the chunk index
  // as `floor(world / chunk_shape)` with no `lowerBounds` offset.  So
  // the chunk-to-multiscale transform must be identity (NOT translated
  // by `lowerBounds` like an image-style anchor-at-lower-corner store)
  // for the URL the reader fetches to match the URL the writer wrote.
  // The data-extent gating still works because
  // `makeSliceViewChunkSpecification` derives `lower/upperChunkBound`
  // from `lower/upperVoxelBound`.
  //
  // Matches `getSliceViewSources` in `skeleton_frontend.ts` (also
  // identity) — the two paths agree about the chunk-coord convention.
  const chunkToMultiscaleTransform = matrix.createIdentity(
    Float32Array,
    rank + 1,
  );

  // Build one spatial-index level per zarr-vectors level.  Order:
  // finest first (level 0 first), which is the order
  // forEachVisibleAnnotationChunk expects (it iterates length-1 → 0).
  const spatialIndices: AnnotationSpatialIndexLevelMetadata[] = [];
  let level0Limit = FALLBACK_LEVEL_LIMIT;
  for (let k = 0; k < levelPaths.length; ++k) {
    const levelPath = levelPaths[k];
    const levelUrl = kvstoreEnsureDirectoryPipelineUrl(
      pipelineUrlJoin(storeUrl, levelPath),
    );
    let levelAttrs: any;
    try {
      const levelJson = await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(levelUrl, "zarr.json"),
        `zarr-vectors level ${JSON.stringify(levelPath)} metadata`,
        options,
      );
      levelAttrs = levelJson?.attributes?.zarr_vectors_level;
    } catch {
      levelAttrs = undefined;
    }
    const limit = computeLevelLimit(levelAttrs, numChunks, level0Limit, rank);
    if (k === 0) level0Limit = limit;

    // Anchor the chunk grid at the world origin, NOT at `lowerBounds`,
    // to match the writer's `floor(world / chunkShape)` convention.
    // Pass the actual `[lowerBounds, upperBounds]` extent so
    // `makeSliceViewChunkSpecification` derives a chunk-index range
    // that can include negative indices when `lowerBounds < 0` (the
    // typical case for stores whose data straddles the world origin).
    const spec: AnnotationGeometryChunkSpecification = {
      limit,
      chunkToMultiscaleTransform,
      pyramidMode,
      ...makeSliceViewChunkSpecification({
        rank,
        chunkDataSize: chunkShapeF32,
        lowerVoxelBound: Float32Array.from(lowerBounds),
        upperVoxelBound: Float32Array.from(upperBounds),
      }),
    };

    const spatialParams =
      new ZarrVectorsAnnotationSpatialIndexSourceParameters();
    spatialParams.baseUrl = levelUrl;
    spatialParams.rank = rank;
    spatialParams.attributeNames = attributeNames;
    spatialParams.attributeDtypes = attributeDtypes;

    spatialIndices.push({ parameters: spatialParams, spec });
  }

  const parameters = new ZarrVectorsAnnotationSourceParameters();
  parameters.rank = rank;
  parameters.type = AnnotationType.POINT;
  parameters.properties = properties;
  parameters.pyramidMode = pyramidMode;

  const meta: AnnotationMetadata = {
    rank,
    coordinateSpace,
    parameters,
    spatialIndices,
  };
  return meta;
}

function getAnnotationDataSource(
  sharedKvStoreContext: SharedKvStoreContext,
  metadata: AnnotationMetadata,
): DataSource {
  return {
    modelTransform: makeIdentityTransform(metadata.coordinateSpace),
    subsources: [
      {
        id: "default",
        default: true,
        subsource: {
          annotation: sharedKvStoreContext.chunkManager.getChunkSource(
            ZarrVectorsAnnotationSource,
            {
              sharedKvStoreContext,
              metadata,
              parameters: metadata.parameters,
            },
          ),
        },
      },
    ],
  };
}

// ---------------------------------------------------------------
// Skeleton / polyline / streamline path
// ---------------------------------------------------------------

interface SkeletonMetadata {
  rank: number;
  coordinateSpace: ReturnType<typeof makeCoordinateSpace>;
  /** Parameters for the per-segment (pass-2) chunk source. */
  pass2Params: ZarrVectorsObjectKeyedSkeletonSourceParameters;
  /**
   * Per-object attributes assembled into a neuroglancer
   * `SegmentPropertyMap`.  `undefined` when the store has no
   * `object_attributes/` at level 0.  Exposed by
   * `getSkeletonDataSource` as the opt-in `"properties"` subsource.
   */
  segmentPropertyMap?: SegmentPropertyMap;
  /**
   * Per-level parameter blobs for the spatially-indexed (pass-1) chunk
   * sources, finest-first.  Together with `spatialGrid` they let the
   * multiscale source build the per-level chunk specs.
   *
   * `undefined` when the store's rank is not 3 (neuroglancer's
   * spatially-indexed skeleton machinery hardcodes vec3 position
   * texture format) — the dispatch falls back to pass-2 only in that
   * case.
   */
  pass1Levels?: ReadonlyArray<{
    parameters: ZarrVectorsSpatiallyIndexedSkeletonSourceParameters;
  }>;
  /** Grid info shared across all pass-1 levels.  Co-defined with `pass1Levels`. */
  spatialGrid?: {
    /**
     * Per-level chunk shape in world units.  Length == pass1Levels.length.
     * Each entry comes from the level's ``zarr_vectors_level.chunk_shape``
     * override if present, otherwise from root chunk_shape.  Writers
     * that want the spatial grid-resolution picker to expose multiple
     * LOD levels should set ``chunk_scale_factors`` so each level's
     * chunk_shape is monotonically distinct — that's the same pattern
     * CATMAID's per-level ``chunkSize`` follows
     * (`src/datasource/catmaid/frontend.ts:386-390`).  Sparsity-only
     * pyramids without per-level chunk-shape changes still load, but
     * adjacent levels with identical chunk_shape will collapse into a
     * single picker entry.
     */
    perLevelChunkShape: Float32Array[];
    /**
     * Live object count per level, parallel to `perLevelChunkShape`;
     * `undefined` where it cannot be determined.  See
     * `computePerLevelObjectCount`.
     *
     * This is what makes an object-sparsity pyramid pickable: such a pyramid
     * keeps the same `chunk_shape` at every level and drops whole objects
     * instead, so chunk size — the framework's usual proxy for detail —
     * cannot tell the levels apart. Object count can. See
     * `getSpatialSkeletonGridSizes` in `skeleton_frontend.ts`.
     */
    perLevelObjectCount: (number | undefined)[];
    /**
     * ``zarr_vectors_level.vertex_count`` per level -- the *cost* axis, as
     * distinct from `perLevelObjectCount`'s *detail* axis.  See
     * `getSpatialSkeletonLevelCostsBytes`.
     */
    perLevelVertexCount: (number | undefined)[];
    /**
     * World-space lower bound of the data.  Can be negative — zarr-vectors
     * chunks are indexed around world origin `(0,0,0)`, NOT around
     * `lowerBounds`.  `makeSliceViewChunkSpecification` consumes this as
     * `lowerVoxelBound` and computes negative chunk indices accordingly.
     */
    lowerBounds: Float32Array;
    /** World-space upper bound of the data. */
    upperBounds: Float32Array;
  };
}

const SKELETON_LIKE_GEOM = new Set<string>([
  "skeleton",
  "polyline",
  "streamline",
  "graph",
]);

/**
 * Read store metadata for a skeleton / polyline / streamline store and
 * assemble the parameter blob needed to construct chunk sources.
 *
 * Layout assumptions (per the zarr-vectors spec):
 *
 * - `zarr_vectors.geometry_types` contains exactly one of
 *   `"skeleton"`, `"polyline"`, `"streamline"`.
 * - `zarr_vectors.object_index_convention === "standard"` (the only
 *   value that maps to the segmentation-layer pathway).
 * - Level-0 metadata lives under `multiscales[0].datasets[0].path`.
 * - `links/0/.zattrs.dtype` (or `data_type` fallback) declares the
 *   on-disk link dtype; absent for `implicit_sequential` stores.
 */
async function buildSkeletonMetadata(
  sharedKvStoreContext: SharedKvStoreContext,
  storeUrl: string,
  rootAttrs: any,
  options: Partial<ProgressOptions>,
): Promise<SkeletonMetadata> {
  const zv = rootAttrs?.zarr_vectors;
  if (zv === undefined) {
    throw new Error(
      "Not a zarr-vectors store: root attributes lack a 'zarr_vectors' block",
    );
  }
  checkZvVersion(zv);
  const geometryTypes: string[] = Array.isArray(zv.geometry_types)
    ? zv.geometry_types
    : [];
  const skeletonKindsPresent = geometryTypes.filter((g) =>
    SKELETON_LIKE_GEOM.has(g),
  );
  if (skeletonKindsPresent.length !== 1) {
    throw new Error(
      `buildSkeletonMetadata: expected exactly one skeleton-like ` +
        `geometry type (got ${JSON.stringify(skeletonKindsPresent)})`,
    );
  }
  const geometryKind = skeletonKindsPresent[0] as
    | "skeleton"
    | "polyline"
    | "streamline"
    | "graph";

  // Bounds + rank — identical idiom to the annotation path.
  const bounds = zv.bounds;
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 2 ||
    !Array.isArray(bounds[0]) ||
    !Array.isArray(bounds[1])
  ) {
    throw new Error(
      "zarr-vectors store: 'bounds' must be [[lower...], [upper...]]",
    );
  }
  const rank = bounds[0].length;
  if (bounds[1].length !== rank) {
    throw new Error(
      "zarr-vectors store: bounds[0] and bounds[1] have different rank",
    );
  }
  const lowerBounds = Float64Array.from(bounds[0], Number);
  const upperBounds = Float64Array.from(bounds[1], Number);

  // Coordinate space — prefer NGFF multiscales axes / scales.
  const ngHints = rootAttrs.neuroglancer ?? {};
  const coordinateSpace =
    ngHints.coordinate_space &&
    Array.isArray(ngHints.coordinate_space.names) &&
    ngHints.coordinate_space.names.length === rank
      ? buildCoordinateSpaceFromHints(
          ngHints.coordinate_space,
          lowerBounds,
          upperBounds,
        )
      : buildCoordinateSpaceFromMultiscales(
          rootAttrs.multiscales,
          lowerBounds,
          upperBounds,
          rank,
        );

  // Resolve level 0 — the per-segment manifest lookup operates at one
  // fixed level (the v1 dispatch always uses level 0).  Multi-level
  // pass-1 spatial rendering will use `levelPaths` more broadly in
  // slice 4c-step2.
  const levelPaths = enumerateLevelPaths(rootAttrs.multiscales);
  const level0Url = kvstoreEnsureDirectoryPipelineUrl(
    pipelineUrlJoin(storeUrl, levelPaths[0]),
  );

  // Whether the store carries a per-fragment `segment_id` attribute. Probed by
  // a direct GET of its array metadata rather than trusted from each level's
  // `arrays_present`, which the 0.9 writer leaves incomplete at coarse levels
  // (it can omit arrays — links, vertex_fragments — that physically exist). The
  // fragment-attribute schema is store-wide, so level 0's answer holds for every
  // level. `arrays_present` is kept only as a fallback for pre-0.9 stores that
  // do not materialise this array.
  const segmentIdMeta = await getJsonResource(
    sharedKvStoreContext,
    joinBaseUrlAndPath(level0Url, "fragment_attributes/segment_id/zarr.json"),
    "fragment_attributes/segment_id metadata",
    options,
  );
  const hasFragmentSegmentIds = segmentIdMeta !== undefined;

  // Per-vertex attribute discovery — reuse the annotation-path machinery
  // verbatim; the resulting (attributeNames, attributeDtypes) feed the
  // skeleton render layer's `prop_<name>()` shader bridge.
  const { attributeNames, attributeDtypes } = await buildPropertySpecsAndDtypes(
    sharedKvStoreContext,
    level0Url,
    ngHints,
    options,
  );

  // Links convention — drives whether explicit `links/0/<chunk>` edges
  // are read in addition to the implicit sequential ones synthesised
  // from fragment ranges.
  const linksConventionRaw = zv.links_convention;
  let linksConvention:
    | "implicit_sequential"
    | "implicit_sequential_with_branches"
    | "explicit";
  if (linksConventionRaw === undefined) {
    // Spec default per geometry: streamline / polyline → implicit_sequential,
    // skeleton → implicit_sequential_with_branches, graph → explicit.
    if (geometryKind === "skeleton") {
      linksConvention = "implicit_sequential_with_branches";
    } else if (geometryKind === "graph") {
      linksConvention = "explicit";
    } else {
      linksConvention = "implicit_sequential";
    }
  } else if (
    linksConventionRaw === "implicit_sequential" ||
    linksConventionRaw === "implicit_sequential_with_branches" ||
    linksConventionRaw === "explicit"
  ) {
    linksConvention = linksConventionRaw;
  } else {
    throw new Error(
      `zarr-vectors links_convention=${JSON.stringify(linksConventionRaw)}: ` +
        `expected 'implicit_sequential', 'implicit_sequential_with_branches', or 'explicit'`,
    );
  }

  // Link dtype: read `links/0/zarr.json`'s declared dtype if the
  // convention has explicit edges.  Default to int64 (universally safe)
  // when no links array exists.
  let linkDtype:
    | "uint8"
    | "uint16"
    | "uint32"
    | "int8"
    | "int16"
    | "int32"
    | "int64";
  if (linksConvention === "implicit_sequential") {
    linkDtype = "int64";
  } else {
    let linksZarrJson: any | undefined;
    try {
      linksZarrJson = await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(level0Url, "links/0/zarr.json"),
        "zarr-vectors links/0 metadata",
        options,
      );
    } catch {
      linksZarrJson = undefined;
    }
    const raw =
      linksZarrJson?.attributes?.dtype ?? linksZarrJson?.data_type ?? "int64";
    if (
      raw !== "uint8" &&
      raw !== "uint16" &&
      raw !== "uint32" &&
      raw !== "int8" &&
      raw !== "int16" &&
      raw !== "int32" &&
      raw !== "int64"
    ) {
      throw new Error(
        `zarr-vectors links/0 dtype=${JSON.stringify(raw)}: expected an integer dtype`,
      );
    }
    linkDtype = raw;
  }

  const pass2Params = new ZarrVectorsObjectKeyedSkeletonSourceParameters();
  pass2Params.baseUrl = level0Url;
  pass2Params.rank = rank;
  pass2Params.attributeNames = attributeNames;
  pass2Params.attributeDtypes = attributeDtypes;
  pass2Params.linksConvention = linksConvention;
  pass2Params.geometryKind = geometryKind;
  pass2Params.linkDtype = linkDtype;
  pass2Params.hasFragmentSegmentIds = hasFragmentSegmentIds;
  // Per-chunk-array grid geometry (origin + sharding) from level 0's
  // vertices/zarr.json. Needed by pass 2 regardless of rank (it reads the same
  // per-chunk arrays). Cached, so re-read per level below with no extra traffic.
  const level0Grid = await readChunkGridParams(
    sharedKvStoreContext,
    level0Url,
    rank,
    options,
  );
  pass2Params.chunkGridOrigin = level0Grid.chunkGridOrigin;
  pass2Params.sharded = level0Grid.sharded;
  pass2Params.shardChunkShape = level0Grid.shardChunkShape;
  pass2Params.cellSeparator = level0Grid.cellSeparator;

  // Pass-1 (spatially-indexed) wiring — only when the store is 3-D.
  // Neuroglancer's spatially-indexed skeleton machinery hardcodes a
  // vec3 position texture format (`skeleton/frontend.ts:1706-1709`); 2-D
  // or higher-rank stores fall back to pass-2 only.
  let pass1Levels:
    | ReadonlyArray<{
        parameters: ZarrVectorsSpatiallyIndexedSkeletonSourceParameters;
      }>
    | undefined;
  let spatialGrid:
    | {
        perLevelChunkShape: Float32Array[];
        perLevelObjectCount: (number | undefined)[];
        perLevelVertexCount: (number | undefined)[];
        lowerBounds: Float32Array;
        upperBounds: Float32Array;
      }
    | undefined;
  if (rank === 3) {
    const chunkShape = zv.chunk_shape;
    if (!Array.isArray(chunkShape) || chunkShape.length !== rank) {
      throw new Error(
        `zarr-vectors store: 'chunk_shape' must have rank ${rank}`,
      );
    }
    // Root chunk_shape is the default per-level chunk size.  When a
    // level stamps its own ``zarr_vectors_level.chunk_shape`` on disk
    // (writers using ``chunk_scale_factors`` to grow chunks at coarser
    // levels), that overrides the root for that level.
    const rootChunkShapeF32 = new Float32Array(rank);
    for (let i = 0; i < rank; ++i) {
      rootChunkShapeF32[i] = Number(chunkShape[i]);
    }
    const lowerBoundsF32 = Float32Array.from(lowerBounds);
    const upperBoundsF32 = Float32Array.from(upperBounds);

    // Fetch each level's zarr.json in parallel to read its optional
    // per-level chunk_shape override and arrays_present list.  Reuses
    // kvstore caching: the same files are read again below by the
    // chunk-source download path with zero net traffic.
    const perLevelMeta: {
      chunkShape: Float32Array;
      hasFragmentSegmentIds: boolean;
      vertexCount: number | undefined;
      objectSparsity: number | undefined;
      numObjects: number | undefined;
      grid: {
        chunkGridOrigin: number[];
        sharded: boolean;
        shardChunkShape: number[];
        cellSeparator: string;
      };
    }[] = await Promise.all(
      levelPaths.map(async (levelPath) => {
        const levelUrl = kvstoreEnsureDirectoryPipelineUrl(
          pipelineUrlJoin(storeUrl, levelPath),
        );
        // Required: per-level per-chunk-array grid geometry (throws on the
        // unreadable pre-0.9.0 "Option G" layout).
        const grid = await readChunkGridParams(
          sharedKvStoreContext,
          levelUrl,
          rank,
          options,
        );
        try {
          const levelJson = await getJsonResource(
            sharedKvStoreContext,
            joinBaseUrlAndPath(levelUrl, "zarr.json"),
            `zarr-vectors level ${JSON.stringify(levelPath)} metadata`,
            options,
          );
          const lvlAttrs = levelJson?.attributes?.zarr_vectors_level ?? {};
          const override = lvlAttrs?.chunk_shape;
          const chunkShape =
            Array.isArray(override) && override.length === rank
              ? (() => {
                  const arr = new Float32Array(rank);
                  for (let i = 0; i < rank; ++i) arr[i] = Number(override[i]);
                  return arr;
                })()
              : new Float32Array(rootChunkShapeF32);
          // `hasFragmentSegmentIds` comes from the store-wide direct probe
          // above, not this level's `arrays_present` (unreliable at coarse
          // levels); the fragment-attribute schema does not vary by level.
          const vc = Number(lvlAttrs?.vertex_count);
          const vertexCount = Number.isFinite(vc) && vc > 0 ? vc : undefined;
          const sp = Number(lvlAttrs?.object_sparsity);
          const objectSparsity = Number.isFinite(sp) && sp > 0 ? sp : undefined;
          const no = Number(lvlAttrs?.inherited_num_objects);
          const numObjects = Number.isFinite(no) && no > 0 ? no : undefined;
          return {
            chunkShape,
            hasFragmentSegmentIds,
            vertexCount,
            objectSparsity,
            numObjects,
            grid,
          };
        } catch {
          // fall through to defaults
        }
        return {
          chunkShape: new Float32Array(rootChunkShapeF32),
          hasFragmentSegmentIds: true,
          vertexCount: undefined,
          objectSparsity: undefined,
          numObjects: undefined,
          grid,
        };
      }),
    );
    const perLevelChunkShape = perLevelMeta.map((m) => m.chunkShape);
    const perLevelObjectCount = computePerLevelObjectCount(perLevelMeta);
    const perLevelVertexCount = perLevelMeta.map((m) => m.vertexCount);

    // Per-level parameter blobs.  Each level gets its own chunkShape
    // (may differ when the writer used ``chunk_scale_factors``).
    const levels: {
      parameters: ZarrVectorsSpatiallyIndexedSkeletonSourceParameters;
    }[] = [];
    for (let k = 0; k < levelPaths.length; ++k) {
      const levelUrl = kvstoreEnsureDirectoryPipelineUrl(
        pipelineUrlJoin(storeUrl, levelPaths[k]),
      );
      const params = new ZarrVectorsSpatiallyIndexedSkeletonSourceParameters();
      params.baseUrl = levelUrl;
      params.rank = rank;
      params.attributeNames = attributeNames;
      params.attributeDtypes = attributeDtypes;
      params.linksConvention = linksConvention;
      params.geometryKind = geometryKind;
      params.linkDtype = linkDtype;
      // gridIndex must match the framework's `spatialSkeletonGridLevels`
      // ordering, which is sorted DESCENDING by spacing (largest first).
      // Our `levelPaths[0]` is the FINEST pyramid level (smallest spacing),
      // so it should land at the END of the sorted list:
      //   levelPaths[0]  (finest)   → gridIndex = numLevels - 1
      //   levelPaths[N-1] (coarsest) → gridIndex = 0
      // See `findClosestSpatialSkeletonGridLevelBySpacing` in
      // `src/layer/segmentation/index.ts:588-603` for the picker that
      // then looks up sources by `gridIndex`.
      params.gridIndex = levelPaths.length - 1 - k;
      params.hasFragmentSegmentIds = perLevelMeta[k].hasFragmentSegmentIds;
      params.chunkGridOrigin = perLevelMeta[k].grid.chunkGridOrigin;
      params.sharded = perLevelMeta[k].grid.sharded;
      params.shardChunkShape = perLevelMeta[k].grid.shardChunkShape;
      params.cellSeparator = perLevelMeta[k].grid.cellSeparator;
      levels.push({ parameters: params });
    }

    pass1Levels = levels;
    spatialGrid = {
      perLevelChunkShape,
      perLevelObjectCount,
      perLevelVertexCount,
      lowerBounds: lowerBoundsF32,
      upperBounds: upperBoundsF32,
    };
    // Pass-2 always operates at level 0 — use its arrays_present to gate
    // the fragment_attributes/segment_id fetch.  Overrides the default-true
    // placeholder set above.
    pass2Params.hasFragmentSegmentIds = perLevelMeta[0].hasFragmentSegmentIds;
  }

  // Discover per-object attributes at level 0 and build the
  // segment-properties map.  Pinned to level 0 because object_ids are
  // global across the pyramid; coarser levels reuse the level-0 row
  // assignments via `present_mask` (handled inside the builder).
  const segmentPropertyMap = await buildObjectAttributePropertyMap(
    sharedKvStoreContext,
    level0Url,
    options,
  );

  // The model transform is identity: zarr-vectors writers store vertices in
  // exact physical coordinates.  The NGFF per-level `translation` on
  // `datasets[0]` is a bin-center convention (`chunk_shape / 2`, written by the
  // resolution-level writer to round-trip `bin_shape`) that describes the chunk
  // *grid*, not the vertex positions — this store carries [15.5, 16, 17].
  // Applying it shifts every vertex half a chunk off its true coordinate
  // (verified by decoding `0/vertices/c/0/0/0`: chunk (0,0,0)'s vertices already
  // lie within its absolute [0,31)×[0,32)×[0,34) range), which breaks
  // coordinate-based navigation and alignment with any reference volume.  All
  // spatial-index calculations already run in raw coordinates.
  return {
    rank,
    coordinateSpace,
    pass2Params,
    pass1Levels,
    spatialGrid,
    segmentPropertyMap,
  };
}

/**
 * Construct a segmentation-shaped `DataSource` from skeleton metadata.
 *
 * The data source exposes up to **two** chunk sources under
 * `subsource.mesh`, each in its own subsource entry:
 *
 *   - `"skeleton-spatial"` — the multiscale spatially-indexed source
 *     that drives **pass 1** (camera-relative chunk loading).  Only
 *     emitted when the store is 3-D (the spatially-indexed skeleton
 *     render layer in neuroglancer assumes vec3 positions).
 *   - `"skeleton"` — the per-segment source that drives **pass 2**
 *     (user-typed object IDs in the segments-list UI).
 *
 * Neuroglancer's `SegmentationUserLayer.renderLayers` dispatches by
 * `instanceof` on `subsource.mesh`:
 *
 *   - `MultiscaleSpatiallyIndexedSkeletonSource` → mounts the
 *     spatially-indexed render layer.
 *   - `SkeletonSource` → mounts the per-segment render layer.
 *
 * Both render layers can coexist on the same segmentation layer, which
 * is how the two passes compose visually.
 */
function getSkeletonDataSource(
  sharedKvStoreContext: SharedKvStoreContext,
  metadata: SkeletonMetadata,
): DataSource {
  const subsources: DataSource["subsources"] = [];

  if (
    metadata.pass1Levels !== undefined &&
    metadata.spatialGrid !== undefined
  ) {
    subsources.push({
      id: "skeleton-spatial",
      default: true,
      subsource: {
        mesh: new ZarrVectorsMultiscaleSpatiallyIndexedSkeletonSource(
          sharedKvStoreContext.chunkManager,
          sharedKvStoreContext,
          {
            levels: metadata.pass1Levels,
            perLevelChunkShape: metadata.spatialGrid.perLevelChunkShape,
            perLevelObjectCount: metadata.spatialGrid.perLevelObjectCount,
            perLevelVertexCount: metadata.spatialGrid.perLevelVertexCount,
            metersPerUnit: Float64Array.from(metadata.coordinateSpace.scales),
            lowerBounds: metadata.spatialGrid.lowerBounds,
            upperBounds: metadata.spatialGrid.upperBounds,
          },
        ),
      },
    });
  }

  // Pass-2 (per-segment, full-detail) source. When there is a pass-1 bulk to
  // draw it on top of, it defaults ON so the ROI filter's per-group "high
  // detail" toggle just works (the segmentation layer repurposes it as the
  // high-detail render layer, driven by roiHighDetailSegments; it renders
  // nothing until a group is marked high-detail). Without pass-1 (e.g. 2-d or
  // higher-rank stores) it stays the opt-in per-segment source.
  subsources.push({
    id: "skeleton",
    default: metadata.pass1Levels !== undefined,
    subsource: {
      mesh: sharedKvStoreContext.chunkManager.getChunkSource(
        ZarrVectorsObjectKeyedSkeletonSource,
        {
          sharedKvStoreContext,
          parameters: metadata.pass2Params,
        },
      ),
    },
  });

  // Segment-properties subsource.  Beyond the sortable/filterable columns in
  // the segments-list panel, this map now BACKS the streamline filter's per-
  // group + background length filter and colour-by-object-attribute: those UIs
  // read `displayState.segmentPropertyMap.numericalProperties`, which is only
  // populated when this subsource is active.  So it defaults ON (the feature
  // must work without the user hunting for a source toggle); it carries no extra
  // geometry.  Only emitted when the store carries `object_attributes/` at level
  // 0; otherwise the subsource doesn't exist to toggle.
  if (metadata.segmentPropertyMap !== undefined) {
    subsources.push({
      id: "properties",
      default: true,
      subsource: { segmentPropertyMap: metadata.segmentPropertyMap },
    });
  }

  return {
    modelTransform: makeIdentityTransform(metadata.coordinateSpace),
    subsources,
  };
}

// ---------------------------------------------------------------
// Provider
// ---------------------------------------------------------------

function resolveUrl(options: GetKvStoreBasedDataSourceOptions) {
  const { authorityAndPath, query, fragment } = parseUrlSuffix(
    options.url.suffix,
  );
  if (query) {
    throw new Error(
      `Invalid URL ${JSON.stringify(options.url.url)}: query parameters not supported`,
    );
  }
  if (fragment) {
    throw new Error(
      `Invalid URL ${JSON.stringify(options.url.url)}: fragment not supported`,
    );
  }
  return {
    kvStoreUrl: kvstoreEnsureDirectoryPipelineUrl(options.kvStoreUrl),
    additionalPath: authorityAndPath ?? "",
  };
}

export class ZarrVectorsDataSource implements KvStoreBasedDataSourceProvider {
  get scheme() {
    return "zarr-vectors";
  }
  get expectsDirectory() {
    return true;
  }
  get description() {
    return "Zarr Vectors (experimental) data source";
  }

  async get(
    options: GetKvStoreBasedDataSourceOptions,
  ): Promise<DataSourceLookupResult> {
    let { kvStoreUrl, additionalPath } = resolveUrl(options);
    kvStoreUrl = kvstoreEnsureDirectoryPipelineUrl(
      pipelineUrlJoin(kvStoreUrl, additionalPath),
    );
    return options.registry.chunkManager.memoize.getAsync(
      { type: "zarr-vectors:get", url: kvStoreUrl },
      options,
      async (progressOptions) => {
        const { sharedKvStoreContext } = options.registry;
        const rootJson = await getJsonResource(
          sharedKvStoreContext,
          joinBaseUrlAndPath(kvStoreUrl, "zarr.json"),
          "zarr-vectors root metadata",
          progressOptions,
        );
        if (rootJson === undefined) {
          throw new Error(
            `No zarr.json found at ${kvStoreUrl} — is this a zarr v3 store?`,
          );
        }
        if (rootJson.node_type && rootJson.node_type !== "group") {
          throw new Error(
            `zarr-vectors expected a zarr v3 group, got node_type=${JSON.stringify(rootJson.node_type)}`,
          );
        }
        const attrs = rootJson.attributes ?? {};
        // Dispatch by geometry_types: point_cloud → annotation layer
        // (existing path); skeleton / polyline / streamline → segmentation
        // layer (slice 4c).  Validation (unknown types, mixed-geometry,
        // wrong object_index_convention) happens here so the failure
        // surface is consistent across geometries.
        const zv = attrs.zarr_vectors;
        if (zv === undefined) {
          throw new Error(
            "Not a zarr-vectors store: root attributes lack a 'zarr_vectors' block",
          );
        }
        const geometryTypes: string[] = Array.isArray(zv.geometry_types)
          ? zv.geometry_types
          : [];
        const supportedGeom = new Set<string>([
          "point_cloud",
          "skeleton",
          "polyline",
          "streamline",
          "graph",
        ]);
        const unsupported = geometryTypes.filter((g) => !supportedGeom.has(g));
        if (unsupported.length > 0) {
          throw new Error(
            `zarr-vectors datasource: unsupported geometry types ` +
              `${JSON.stringify(unsupported)}.  Supported: ` +
              `${JSON.stringify(Array.from(supportedGeom))}`,
          );
        }
        const hasSkeletonLike = geometryTypes.some((g) =>
          SKELETON_LIKE_GEOM.has(g),
        );
        const hasPointCloud = geometryTypes.includes("point_cloud");
        if (hasSkeletonLike && hasPointCloud) {
          throw new Error(
            `zarr-vectors datasource: stores with both point_cloud and ` +
              `skeleton/polyline/streamline geometry are not yet supported ` +
              `(found: ${JSON.stringify(geometryTypes)})`,
          );
        }
        if (!hasSkeletonLike && !hasPointCloud) {
          throw new Error(
            `zarr-vectors datasource: no recognised geometry type in ` +
              `${JSON.stringify(geometryTypes)}; expected 'point_cloud' or ` +
              `one of ${JSON.stringify(Array.from(SKELETON_LIKE_GEOM))}`,
          );
        }

        let dataSource: DataSource;
        if (hasSkeletonLike) {
          // The spec defines two object-index conventions. "standard" maps a
          // selected segment id to a dense object index via
          // `object_attributes/segment_id`; "identity" means the selected id IS
          // the dense index. The backend already degrades to identity when that
          // attribute is absent (see `resolveObjectIndex`), so both conventions
          // are supported here.
          const objectIndexConvention = zv.object_index_convention;
          if (
            objectIndexConvention !== "standard" &&
            objectIndexConvention !== "identity" &&
            objectIndexConvention !== undefined
          ) {
            throw new Error(
              `zarr-vectors datasource: skeleton/polyline/streamline geometry ` +
                `requires object_index_convention 'standard' or 'identity' (got ` +
                `${JSON.stringify(objectIndexConvention)})`,
            );
          }
          const skelMeta = await buildSkeletonMetadata(
            sharedKvStoreContext,
            kvStoreUrl,
            attrs,
            progressOptions,
          );
          dataSource = getSkeletonDataSource(sharedKvStoreContext, skelMeta);
        } else {
          const meta = await buildAnnotationMetadata(
            sharedKvStoreContext,
            kvStoreUrl,
            attrs,
            progressOptions,
          );
          dataSource = getAnnotationDataSource(sharedKvStoreContext, meta);
        }
        dataSource.canonicalUrl = `${kvStoreUrl}|${options.url.scheme}:`;
        return dataSource;
      },
    );
  }
}

/**
 * Companion data source that re-interprets ANY zarr-vectors store
 * (graph, skeleton, polyline, streamline, or point_cloud) as a point
 * cloud.  Routes through the existing annotation-layer path: reads
 * `vertices/<level>/<chunk>` + `vertex_attributes/<name>/<level>/<chunk>`
 * (which exist for every geometry kind) and ignores everything edge-
 * or segment-related (links, manifests, cross_chunk_links,
 * object_index, object_attributes).
 *
 * Useful for spot-checking a graph store's vertex positions without
 * waiting for the segmentation-layer machinery, or for stripping a
 * heavyweight skeleton store down to a point cloud when the connectivity
 * isn't relevant.
 */
export class ZarrVectorsPointCloudDataSource
  implements KvStoreBasedDataSourceProvider
{
  get scheme() {
    return "zarr-vectors-pointcloud";
  }
  get expectsDirectory() {
    return true;
  }
  get description() {
    return "Zarr Vectors as a point cloud (ignores edges)";
  }

  async get(
    options: GetKvStoreBasedDataSourceOptions,
  ): Promise<DataSourceLookupResult> {
    let { kvStoreUrl, additionalPath } = resolveUrl(options);
    kvStoreUrl = kvstoreEnsureDirectoryPipelineUrl(
      pipelineUrlJoin(kvStoreUrl, additionalPath),
    );
    return options.registry.chunkManager.memoize.getAsync(
      { type: "zarr-vectors-pointcloud:get", url: kvStoreUrl },
      options,
      async (progressOptions) => {
        const { sharedKvStoreContext } = options.registry;
        const rootJson = await getJsonResource(
          sharedKvStoreContext,
          joinBaseUrlAndPath(kvStoreUrl, "zarr.json"),
          "zarr-vectors root metadata",
          progressOptions,
        );
        if (rootJson === undefined) {
          throw new Error(
            `No zarr.json found at ${kvStoreUrl} — is this a zarr v3 store?`,
          );
        }
        if (rootJson.node_type && rootJson.node_type !== "group") {
          throw new Error(
            `zarr-vectors expected a zarr v3 group, got node_type=${JSON.stringify(rootJson.node_type)}`,
          );
        }
        const attrs = rootJson.attributes ?? {};
        const meta = await buildAnnotationMetadata(
          sharedKvStoreContext,
          kvStoreUrl,
          attrs,
          progressOptions,
          /*allowAnyGeometry=*/ true,
        );
        const dataSource = getAnnotationDataSource(sharedKvStoreContext, meta);
        dataSource.canonicalUrl = `${kvStoreUrl}|${options.url.scheme}:`;
        return dataSource;
      },
    );
  }
}

export function registerAutoDetect(_registry: AutoDetectRegistry) {
  // Auto-detect is intentionally omitted in v1: a zarr-vectors store
  // is also a valid zarr v3 group, and we don't want to shadow the
  // existing zarr datasource by default.  Users opt in explicitly via
  // the "zarr-vectors://" scheme.
}
