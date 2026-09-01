/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Decoding of `object_attributes/<name>/` chunk payloads into typed arrays.
 *
 * These sit apart from the datasource so they can be exercised without pulling
 * in the frontend's WebGL/DOM dependencies: the padding rules below are pure
 * byte arithmetic, and getting them wrong fails a whole store at load time.
 */

import { DataType } from "#src/util/data_type.js";

/**
 * Map a zarr-vectors object-attribute dtype string to a neuroglancer
 * `DataType` plus the typed-array constructor that reinterprets the
 * raw bytes.  Subset that matches what `SegmentPropertyMap`'s numerical
 * properties can carry — `uint64` is excluded (the segment-properties
 * UI rejects it; the existing precomputed parser does the same).
 */
export const OBJECT_ATTR_DTYPE_TABLE: Record<
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
 * Reinterpret a raw byte blob into a typed array of `dataType`, copying
 * when the source offset isn't aligned to the element size.  Mirrors
 * the chunk-decoder's `reinterpretBytes` so per-object attribute reads
 * follow the same alignment conventions as per-vertex reads.
 */
export function reinterpretObjectAttributeBytes(
  bytes: Uint8Array,
  ctor: (typeof OBJECT_ATTR_DTYPE_TABLE)[string]["ctor"],
  elementSize: number,
  expectedElements: number,
  chunkElements: number = expectedElements,
):
  | Float32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array {
  const expectedBytes = chunkElements * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors object_attributes: expected ${expectedBytes} bytes ` +
        `(${chunkElements} elements), got ${bytes.byteLength}`,
    );
  }
  // A zarr chunk is always full size, so the tail beyond `expectedElements` is
  // fill-value padding; trim it.
  if (bytes.byteOffset % elementSize === 0) {
    return new (ctor as any)(
      bytes.buffer,
      bytes.byteOffset,
      chunkElements,
    ).subarray(0, expectedElements);
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new (ctor as any)(copy.buffer, 0, chunkElements).subarray(
    0,
    expectedElements,
  );
}

/**
 * Reinterpret an 8-byte-per-element blob (float64 / int64 / uint64) as
 * `Float32Array`, downcasting values. The segment-properties columns hold
 * float32 (or the narrow int types), so a float64 tortuosity or int64 vertex
 * count would otherwise be dropped as "exotic"; downcasting keeps them usable
 * for filtering and colouring (precision loss is immaterial for these).
 */
export function reinterpretWideToFloat32(
  bytes: Uint8Array,
  kind: "float64" | "int64" | "uint64",
  expectedElements: number,
  chunkElements: number = expectedElements,
): Float32Array {
  const elementSize = 8;
  const expectedBytes = chunkElements * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors object_attributes: expected ${expectedBytes} bytes ` +
        `(${chunkElements} elements), got ${bytes.byteLength}`,
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
    const wide = new Float64Array(buffer, offset, chunkElements);
    for (let i = 0; i < expectedElements; ++i) out[i] = wide[i];
  } else {
    const wide =
      kind === "int64"
        ? new BigInt64Array(buffer, offset, chunkElements)
        : new BigUint64Array(buffer, offset, chunkElements);
    for (let i = 0; i < expectedElements; ++i) out[i] = Number(wide[i]);
  }
  return out;
}

/**
 * Reinterpret a 64-bit integer object-attribute chunk WITHOUT the lossy
 * float32 downcast {@link reinterpretWideToFloat32} applies.
 *
 * Needed for the column that supplies the segment-property map's id space.
 * Those ids are connectomics root ids (MICrONS ids run to ~8.6e17, far past
 * 2^53), so routing them through `Number` — as the float32 path does — does
 * not merely lose precision, it changes the id: 864691136021592568 comes back
 * as 864691136021592576 and matches nothing the renderer ever asks about.
 *
 * Values are returned unsigned, and `int64` needs no separate case: both
 * spellings are the same 8-byte little-endian payload and differ only in how
 * the top bit reads, so reinterpreting as unsigned preserves the bits — which
 * is what connectomics writers mean when they store a root id in a signed
 * column.
 */
export function reinterpretWideToBigUint64(
  bytes: Uint8Array,
  expectedElements: number,
  chunkElements: number = expectedElements,
): BigUint64Array {
  const elementSize = 8;
  const expectedBytes = chunkElements * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors object_attributes: expected ${expectedBytes} bytes ` +
        `(${chunkElements} elements), got ${bytes.byteLength}`,
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
  const wide = new BigUint64Array(buffer, offset, chunkElements);
  const out = new BigUint64Array(expectedElements);
  for (let i = 0; i < expectedElements; ++i) out[i] = wide[i];
  return out;
}
