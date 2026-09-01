/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  OBJECT_ATTR_DTYPE_TABLE,
  reinterpretObjectAttributeBytes,
  reinterpretWideToBigUint64,
  reinterpretWideToFloat32,
} from "#src/datasource/zarr-vectors/object_attribute_bytes.js";

/** Bytes for `values` followed by `pad` trailing fill elements. */
function paddedChunk(
  ctor: new (n: number) => { set(v: number[]): void; buffer: ArrayBuffer },
  values: number[],
  pad: number,
): Uint8Array {
  const arr = new ctor(values.length + pad);
  arr.set(values);
  return new Uint8Array(arr.buffer);
}

describe("reinterpretObjectAttributeBytes", () => {
  const f32 = OBJECT_ATTR_DTYPE_TABLE.float32;

  it("reads a chunk whose size matches the column exactly", () => {
    const bytes = paddedChunk(Float32Array, [1, 2, 3], 0);
    const v = reinterpretObjectAttributeBytes(bytes, f32.ctor, 4, 3);
    expect(Array.from(v)).toEqual([1, 2, 3]);
  });

  it("trims the fill padding of an over-sized chunk", () => {
    // A writer with a fixed chunk_shape stores shape[0] real values followed by
    // fill values; the column is 3 long but the chunk holds 8.
    const bytes = paddedChunk(Float32Array, [1, 2, 3], 5);
    const v = reinterpretObjectAttributeBytes(bytes, f32.ctor, 4, 3, 8);
    expect(Array.from(v)).toEqual([1, 2, 3]);
  });

  it("trims padding when the blob is not element-aligned", () => {
    const padded = paddedChunk(Float32Array, [4, 5, 6], 5);
    const shifted = new Uint8Array(padded.byteLength + 1);
    shifted.set(padded, 1);
    const unaligned = shifted.subarray(1);
    expect(unaligned.byteOffset % 4).not.toBe(0);
    const v = reinterpretObjectAttributeBytes(unaligned, f32.ctor, 4, 3, 8);
    expect(Array.from(v)).toEqual([4, 5, 6]);
  });

  it("throws when the blob is not a whole chunk", () => {
    const bytes = paddedChunk(Float32Array, [1, 2, 3], 0);
    expect(() =>
      reinterpretObjectAttributeBytes(bytes, f32.ctor, 4, 3, 8),
    ).toThrow(/expected 32 bytes/);
  });

  it("handles the narrow integer dtypes", () => {
    const i16 = OBJECT_ATTR_DTYPE_TABLE.int16;
    const bytes = paddedChunk(Int16Array, [7, -1, 9], 5);
    const v = reinterpretObjectAttributeBytes(bytes, i16.ctor, 2, 3, 8);
    expect(Array.from(v)).toEqual([7, -1, 9]);
  });
});

describe("reinterpretWideToFloat32", () => {
  it("downcasts float64 and trims padding", () => {
    const bytes = paddedChunk(Float64Array, [1.5, 2.5, 3.5], 5);
    const v = reinterpretWideToFloat32(bytes, "float64", 3, 8);
    expect(Array.from(v)).toEqual([1.5, 2.5, 3.5]);
  });

  it("downcasts int64 and trims padding", () => {
    const arr = new BigInt64Array(8);
    arr.set([10n, 20n, 30n]);
    const v = reinterpretWideToFloat32(
      new Uint8Array(arr.buffer),
      "int64",
      3,
      8,
    );
    expect(Array.from(v)).toEqual([10, 20, 30]);
  });

  it("throws when the blob is not a whole chunk", () => {
    const bytes = paddedChunk(Float64Array, [1, 2, 3], 0);
    expect(() => reinterpretWideToFloat32(bytes, "float64", 3, 8)).toThrow(
      /expected 64 bytes/,
    );
  });
});

describe("reinterpretWideToBigUint64", () => {
  /** Real MICrONS root ids: ~8.6e17, far above Number.MAX_SAFE_INTEGER. */
  const ROOT_IDS = [
    864691136021592568n,
    864691135504615478n,
    864691135777755269n,
  ];

  function chunkOf(ids: bigint[], pad: number): Uint8Array {
    const arr = new BigUint64Array(ids.length + pad);
    arr.set(ids);
    return new Uint8Array(arr.buffer);
  }

  it("preserves 64-bit root ids exactly, where the float32 path corrupts them", () => {
    const bytes = chunkOf(ROOT_IDS, 0);
    const exact = reinterpretWideToBigUint64(bytes, ROOT_IDS.length);
    expect([...exact]).toEqual(ROOT_IDS);

    // The contrast that motivates this helper: routing the same bytes through
    // the float32 downcast changes the id rather than merely blurring it, so a
    // property map keyed that way matches nothing.
    const lossy = reinterpretWideToFloat32(bytes, "uint64", ROOT_IDS.length);
    expect(BigInt(lossy[0])).not.toEqual(ROOT_IDS[0]);
  });

  it("reads an int64-spelled column bit-for-bit as unsigned", () => {
    // Writers commonly store root ids in a signed column; the payload is
    // identical, so the values must survive unchanged.
    const signed = new BigInt64Array(ROOT_IDS.map((v) => BigInt.asIntN(64, v)));
    const bytes = new Uint8Array(signed.buffer);
    expect([...reinterpretWideToBigUint64(bytes, ROOT_IDS.length)]).toEqual(
      ROOT_IDS,
    );
  });

  it("trims writer fill padding to the real object count", () => {
    const bytes = chunkOf(ROOT_IDS, 5);
    const out = reinterpretWideToBigUint64(bytes, ROOT_IDS.length, 8);
    expect(out.length).toBe(3);
    expect([...out]).toEqual(ROOT_IDS);
  });

  it("rejects a chunk whose length disagrees with the declared element count", () => {
    expect(() =>
      reinterpretWideToBigUint64(chunkOf(ROOT_IDS, 0), 3, 8),
    ).toThrow(/expected 64 bytes/);
  });
});
