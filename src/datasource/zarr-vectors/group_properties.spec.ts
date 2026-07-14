/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  decodeFixedLengthUtf32Strings,
  invertGroupMembershipsToTags,
  sanitizeTagName,
} from "#src/datasource/zarr-vectors/group_properties.js";

/** Encode `s` as a fixed-width, NUL-padded UTF-32LE slot `lengthBytes` wide. */
function encodeFixedLengthUtf32(s: string, lengthBytes: number): Uint8Array {
  const codepoints = Array.from(s).map((ch) => ch.codePointAt(0)!);
  const out = new Uint8Array(lengthBytes);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < codepoints.length; ++i) {
    dv.setUint32(i * 4, codepoints[i], /* littleEndian= */ true);
  }
  // Remaining slots stay zero (NUL) — `new Uint8Array` is zero-initialized.
  return out;
}

describe("decodeFixedLengthUtf32Strings", () => {
  it("decodes NUL-padded fixed-width slots back to their original strings", () => {
    const lengthBytes = 16; // 4 codepoints/slot
    const short = ["abc", "d", ""];
    const bytes = new Uint8Array(lengthBytes * short.length);
    short.forEach((s, i) =>
      bytes.set(encodeFixedLengthUtf32(s, lengthBytes), i * lengthBytes),
    );
    expect(
      decodeFixedLengthUtf32Strings(bytes, lengthBytes, short.length),
    ).toEqual(short);
  });

  it("stops at the first NUL codepoint even mid-slot", () => {
    const lengthBytes = 12; // 3 codepoints/slot
    const bytes = encodeFixedLengthUtf32("Hi", lengthBytes); // 2 chars, 1 NUL slot unused
    expect(decodeFixedLengthUtf32Strings(bytes, lengthBytes, 1)).toEqual(["Hi"]);
  });

  it("throws when the byte length doesn't match count * lengthBytes", () => {
    const bytes = new Uint8Array(10);
    expect(() => decodeFixedLengthUtf32Strings(bytes, 8, 2)).toThrow(/expected/);
  });

  it("handles an unaligned byteOffset view (copy path) identically to an aligned one", () => {
    const lengthBytes = 8;
    const encoded = encodeFixedLengthUtf32("Ri", lengthBytes);
    // Build a buffer with a 1-byte odd prefix so the view's byteOffset is unaligned.
    const padded = new Uint8Array(1 + encoded.byteLength);
    padded.set(encoded, 1);
    const unaligned = padded.subarray(1);
    expect(unaligned.byteOffset).toBe(1);
    expect(decodeFixedLengthUtf32Strings(unaligned, lengthBytes, 1)).toEqual([
      "Ri",
    ]);
  });
});

describe("invertGroupMembershipsToTags", () => {
  it("gives every streamline in no group an empty tag string", () => {
    const { tagValues } = invertGroupMembershipsToTags(3, 2, [[0], []]);
    expect(tagValues).toEqual([String.fromCharCode(0), "", ""]);
  });

  it("encodes a single group membership as one character code", () => {
    const { tagValues } = invertGroupMembershipsToTags(2, 1, [[0, 1]]);
    expect(tagValues).toEqual([
      String.fromCharCode(0),
      String.fromCharCode(0),
    ]);
  });

  it("sorts and dedupes multi-group membership per the tags contract (distinct, ascending)", () => {
    // oid 0 is in groups 2 and 0 (input order deliberately unsorted).
    const { tagValues } = invertGroupMembershipsToTags(1, 3, [[0], [], [0]]);
    expect(tagValues[0]).toBe(String.fromCharCode(0, 2));
    expect(tagValues[0].charCodeAt(0)).toBeLessThan(tagValues[0].charCodeAt(1));
  });

  it("silently skips out-of-range member oids instead of throwing", () => {
    const { tagValues } = invertGroupMembershipsToTags(2, 1, [[-1, 0, 5]]);
    expect(tagValues).toEqual([String.fromCharCode(0), ""]);
  });

  it("returns groupsByOid usable for numeric-attribute projection (sorted, lowest group id first)", () => {
    const { groupsByOid } = invertGroupMembershipsToTags(1, 3, [[0], [0], []]);
    expect(groupsByOid[0]).toEqual([0, 1]);
  });
});

describe("sanitizeTagName", () => {
  it("replaces internal spaces with underscores", () => {
    expect(sanitizeTagName("Optic Radiation")).toBe("Optic_Radiation");
  });

  it("collapses runs of whitespace into a single underscore", () => {
    expect(sanitizeTagName("Corpus  Callosum\tBody")).toBe(
      "Corpus_Callosum_Body",
    );
  });

  it("trims leading and trailing whitespace instead of leaving stray underscores", () => {
    expect(sanitizeTagName("  AF_L  ")).toBe("AF_L");
  });

  it("leaves names with no whitespace unchanged", () => {
    expect(sanitizeTagName("AF_L")).toBe("AF_L");
  });
});
