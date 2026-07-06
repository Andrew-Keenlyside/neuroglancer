/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import { MANIFEST_MODE_SINGLE } from "#src/datasource/zarr-vectors/object_manifest.js";
import {
  probeObjectAcrossLevels,
  type LevelManifestProbeOptions,
} from "#src/datasource/zarr-vectors/multiscale_manifest.js";

/**
 * Build one per-object manifest blob holding one or more mode-0 blocks.
 * Mirrors the encoder layout from /Users/forrestc/ConnectomeStack/
 * zarr-vectors-py/zarr_vectors/encoding/fragments.py (see also
 * `object_manifest_reader.spec.ts`'s single-block variant of this helper).
 */
function buildManifestBlob(
  blocks: ReadonlyArray<{ chunkCoords: number[]; fragmentIndex: number }>,
): Uint8Array {
  const sidNdim = blocks.length > 0 ? blocks[0].chunkCoords.length : 0;
  const blockSize = 8 * sidNdim + 1 + 8;
  const blob = new Uint8Array(4 + blockSize * blocks.length);
  const view = new DataView(blob.buffer);
  view.setUint32(0, blocks.length, true);
  let off = 4;
  for (const { chunkCoords, fragmentIndex } of blocks) {
    for (const c of chunkCoords) {
      view.setBigInt64(off, BigInt(c), true);
      off += 8;
    }
    view.setUint8(off, MANIFEST_MODE_SINGLE);
    off += 1;
    view.setBigInt64(off, BigInt(fragmentIndex), true);
    off += 8;
  }
  return blob;
}

/** Empty (zero-block) manifest blob — "object sparsified out at this level." */
function buildEmptyManifestBlob(): Uint8Array {
  return new Uint8Array(4); // uint32 numBlocks = 0
}

/**
 * Build a vlen-bytes chunk holding the supplied element blobs.
 * Format: `uint32 N + (uint32 len + bytes) per element`.
 */
function buildVlenBytesChunk(elements: ReadonlyArray<Uint8Array>): Uint8Array {
  let total = 4;
  for (const e of elements) total += 4 + e.byteLength;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, elements.length, true);
  let off = 4;
  for (const e of elements) {
    view.setUint32(off, e.byteLength, true);
    off += 4;
    out.set(e, off);
    off += e.byteLength;
  }
  return out;
}

/** One level's kvStore + reader options, wrapping a single manifest chunk-0. */
function makeLevel(
  manifestBlob: Uint8Array | undefined,
  numObjects = 1,
  chunkSize = 16384,
): LevelManifestProbeOptions {
  return {
    numObjects,
    chunkSize,
    kvStoreRead: async (path: string) =>
      path === "object_index/manifests/c/0" && manifestBlob !== undefined
        ? buildVlenBytesChunk([manifestBlob])
        : undefined,
  };
}

describe("probeObjectAcrossLevels", () => {
  const sidNdim = 3;

  it("marks a level present when its manifest is non-empty", async () => {
    const level0 = makeLevel(
      buildManifestBlob([{ chunkCoords: [1, 0, 0], fragmentIndex: 0 }]),
    );
    const result = await probeObjectAcrossLevels(
      0,
      [level0],
      sidNdim,
      new AbortController().signal,
    );
    expect(result.presentLevels).toEqual([true]);
  });

  it("distinguishes an empty manifest (sparsified out) from an absent one (out of range)", async () => {
    const presentLevel = makeLevel(
      buildManifestBlob([{ chunkCoords: [0, 0, 0], fragmentIndex: 0 }]),
    );
    const emptyLevel = makeLevel(buildEmptyManifestBlob());
    const absentLevel = makeLevel(undefined);
    const result = await probeObjectAcrossLevels(
      0,
      [presentLevel, emptyLevel, absentLevel],
      sidNdim,
      new AbortController().signal,
    );
    expect(result.presentLevels).toEqual([true, false, false]);
  });

  it("marks every level present when the object survives the whole pyramid", async () => {
    const level0 = makeLevel(
      buildManifestBlob([{ chunkCoords: [0, 0, 0], fragmentIndex: 0 }]),
    );
    const level1 = makeLevel(
      buildManifestBlob([
        { chunkCoords: [0, 0, 0], fragmentIndex: 0 },
        { chunkCoords: [2, 1, 0], fragmentIndex: 1 },
      ]),
    );
    const result = await probeObjectAcrossLevels(
      0,
      [level0, level1],
      sidNdim,
      new AbortController().signal,
    );
    expect(result.presentLevels).toEqual([true, true]);
  });

  it("marks every level absent when the object is absent everywhere", async () => {
    const absentLevel = makeLevel(undefined);
    const emptyLevel = makeLevel(buildEmptyManifestBlob());
    const result = await probeObjectAcrossLevels(
      0,
      [absentLevel, emptyLevel],
      sidNdim,
      new AbortController().signal,
    );
    expect(result.presentLevels).toEqual([false, false]);
  });

  it("treats a level with no object_index array (undefined level options) as not present, without affecting other levels", async () => {
    // Level 0 genuinely present; level 1 has no object_index array at
    // all (the caller passes `undefined` for it, e.g. because the
    // per-level `object_index/manifests/zarr.json` fetch failed) --
    // must not prevent level 0 from resolving correctly.
    const level0 = makeLevel(
      buildManifestBlob([{ chunkCoords: [0, 0, 0], fragmentIndex: 0 }]),
    );
    const result = await probeObjectAcrossLevels(
      0,
      [level0, undefined],
      sidNdim,
      new AbortController().signal,
    );
    expect(result.presentLevels).toEqual([true, false]);
  });

  it("probes an out-of-range OID as absent at every level without throwing", async () => {
    const level0 = makeLevel(
      buildManifestBlob([{ chunkCoords: [0, 0, 0], fragmentIndex: 0 }]),
      /* numObjects */ 1,
    );
    const result = await probeObjectAcrossLevels(
      5,
      [level0],
      sidNdim,
      new AbortController().signal,
    );
    expect(result.presentLevels).toEqual([false]);
  });
});
