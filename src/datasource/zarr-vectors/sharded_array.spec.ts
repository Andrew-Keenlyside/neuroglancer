/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  MISSING_SHARD_ENTRY,
  chunkToShardLocation,
  parseShardedArrayMeta,
  readShardedArrayChunk,
  readShardedArrayChunkScoped,
  resolveShardedArrayCellRange,
  shardIndexRegionLength,
  type ShardedArrayReadContext,
} from "#src/datasource/zarr-vectors/sharded_array.js";

/**
 * Real `zarr.json` shape produced by `zarr_vectors.core.group.Group
 * .create_sharded_chunk_array` (captured from an actual 0.8.1 ingest):
 * the shard shape lives on the ARRAY's own `chunk_grid.configuration
 * .chunk_shape`, while the `sharding_indexed` codec's own `configuration
 * .chunk_shape` is the inner per-cell sub-chunk shape, always all-1s.
 * A reader that reads the shard shape from the wrong one of these two
 * fields will misresolve every chunk-to-shard mapping.
 */
function realVerticesZarrJson(opts: {
  gridShape: number[];
  shardShape: number[];
  cellCodec: "raw" | "zstd";
}): Uint8Array {
  const json = {
    shape: opts.gridShape,
    data_type: "variable_length_bytes",
    chunk_grid: {
      name: "regular",
      configuration: { chunk_shape: opts.shardShape },
    },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    fill_value: "",
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: opts.gridShape.map(() => 1),
          codecs:
            opts.cellCodec === "raw"
              ? [{ name: "vlen-bytes", configuration: {} }]
              : [{ name: "vlen-bytes", configuration: {} }, { name: "zstd" }],
          index_codecs: [
            { name: "bytes", configuration: { endian: "little" } },
            { name: "crc32c" },
          ],
          index_location: "end",
        },
      },
    ],
    attributes: {
      cell_codec: opts.cellCodec,
      zv_array: "vertices",
      dtype: "float32",
    },
    zarr_format: 3,
    node_type: "array",
    storage_transformers: [],
  };
  return new TextEncoder().encode(JSON.stringify(json));
}

/** vlen-bytes single-item frame: [uint32 count=1][uint32 length][bytes]. */
function vlenBytesSingleItem(payload: Uint8Array): Uint8Array {
  const out = new Uint8Array(8 + payload.byteLength);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 1, true);
  dv.setUint32(4, payload.byteLength, true);
  out.set(payload, 8);
  return out;
}

/**
 * Build one raw-cell shard file exactly as the writer does: cell
 * payloads (vlen-bytes-framed) concatenated in `subChunk` iteration
 * order, followed by the fixed `prod(shardShape)`-entry C-order
 * `(offset,length)` uint64 LE index, followed by a 4-byte crc32c
 * (content irrelevant — this reader never verifies it).
 */
function buildRawShard(
  shardShape: readonly number[],
  cells: Map<string, Uint8Array>,
): Uint8Array {
  const numCells = shardShape.reduce((a, b) => a * b, 1);
  const framedPayloads: (Uint8Array | undefined)[] = new Array(numCells);
  for (const [key, payload] of cells) {
    const subChunk = key.split(".").map(Number);
    let flat = 0;
    for (let d = 0; d < shardShape.length; ++d) {
      flat = flat * shardShape[d] + subChunk[d];
    }
    framedPayloads[flat] = vlenBytesSingleItem(payload);
  }
  const bodyParts: Uint8Array[] = [];
  const offsets: (bigint | undefined)[] = new Array(numCells);
  const lengths: (bigint | undefined)[] = new Array(numCells);
  let cursor = 0n;
  for (let i = 0; i < numCells; ++i) {
    const framed = framedPayloads[i];
    if (framed === undefined) continue;
    offsets[i] = cursor;
    lengths[i] = BigInt(framed.byteLength);
    bodyParts.push(framed);
    cursor += BigInt(framed.byteLength);
  }
  const bodyLen = Number(cursor);
  const indexLen = shardIndexRegionLength(shardShape);
  const out = new Uint8Array(bodyLen + indexLen);
  let pos = 0;
  for (const part of bodyParts) {
    out.set(part, pos);
    pos += part.byteLength;
  }
  const dv = new DataView(out.buffer, bodyLen, indexLen - 4);
  for (let i = 0; i < numCells; ++i) {
    const off = offsets[i];
    const len = lengths[i];
    if (off === undefined || len === undefined) {
      dv.setBigUint64(i * 16, MISSING_SHARD_ENTRY, true);
      dv.setBigUint64(i * 16 + 8, MISSING_SHARD_ENTRY, true);
    } else {
      dv.setBigUint64(i * 16, off, true);
      dv.setBigUint64(i * 16 + 8, len, true);
    }
  }
  return out;
}

/** Serve byte-range reads from a fixed set of in-memory shard files (path → whole-file bytes). */
function makeShardReader(shards: Map<string, Uint8Array>) {
  const calls: { path: string; byteRange: any }[] = [];
  const kvStoreReadRange = async (
    path: string,
    byteRange: { offset: number; length: number } | { suffixLength: number },
  ) => {
    calls.push({ path, byteRange });
    const bytes = shards.get(path);
    if (bytes === undefined) return undefined;
    if ("suffixLength" in byteRange) {
      return bytes.subarray(bytes.byteLength - byteRange.suffixLength);
    }
    return bytes.subarray(byteRange.offset, byteRange.offset + byteRange.length);
  };
  return { kvStoreReadRange, calls };
}

describe("parseShardedArrayMeta", () => {
  it("reads the shard shape from chunk_grid, not from the sharding codec's inner chunk_shape", () => {
    const bytes = realVerticesZarrJson({
      gridShape: [4, 9, 6],
      shardShape: [8, 8, 8],
      cellCodec: "raw",
    });
    const meta = parseShardedArrayMeta(bytes, "0/vertices");
    expect(meta.shardShape).toEqual([8, 8, 8]);
    expect(meta.gridShape).toEqual([4, 9, 6]);
    expect(meta.cellsAreRaw).toBe(true);
    expect(meta.indexLocation).toBe("end");
  });

  it("rejects a sharding codec whose inner chunk_shape isn't all-1s", () => {
    const bytes = realVerticesZarrJson({
      gridShape: [4, 9, 6],
      shardShape: [8, 8, 8],
      cellCodec: "raw",
    });
    const json = JSON.parse(new TextDecoder().decode(bytes));
    json.codecs[0].configuration.chunk_shape = [2, 1, 1];
    const bad = new TextEncoder().encode(JSON.stringify(json));
    expect(() => parseShardedArrayMeta(bad, "0/vertices")).toThrow(/all-1s/);
  });
});

describe("chunkToShardLocation", () => {
  it("maps a global chunk coordinate to its shard + within-shard sub-chunk", () => {
    expect(chunkToShardLocation([0, 0, 0], [8, 8, 8])).toEqual({
      shardCoord: [0, 0, 0],
      subChunk: [0, 0, 0],
    });
    expect(chunkToShardLocation([3, 8, 5], [8, 8, 8])).toEqual({
      shardCoord: [0, 1, 0],
      subChunk: [3, 0, 5],
    });
  });
});

describe("resolveShardedArrayCellRange / readShardedArrayChunk (real on-disk format)", () => {
  // A grid of (2,2,1) with ONE shard covering the whole grid — mirrors
  // the real writer's single-shard case, small enough to hand-verify.
  const shardShape = [2, 2, 1];
  const cellBytes = {
    "0.0.0": new Uint8Array([1, 2, 3, 4]),
    "0.1.0": new Uint8Array([9, 9]),
    "1.1.0": new Uint8Array([7, 7, 7, 7, 7, 7]),
    // (1,0,0) intentionally absent — empty cell.
  };
  const shard = buildRawShard(
    shardShape,
    new Map(Object.entries(cellBytes)),
  );

  function makeCtx(reader: ReturnType<typeof makeShardReader>) {
    return {
      kvStoreReadRange: reader.kvStoreReadRange,
      arrayPath: "0/vertices",
      meta: {
        gridShape: [2, 2, 1],
        shardShape,
        indexLocation: "end" as const,
        cellsAreRaw: true,
      },
    } satisfies ShardedArrayReadContext;
  }

  it("fetches only the shard's index region, not the whole shard file", async () => {
    const reader = makeShardReader(new Map([["0/vertices/c/0/0/0", shard]]));
    const ctx = makeCtx(reader);
    const range = await resolveShardedArrayCellRange(
      ctx,
      [0, 0, 0],
      new AbortController().signal,
    );
    expect(range).toEqual({ shardPath: "0/vertices/c/0/0/0", offset: 0, length: 12 });
    expect(reader.calls).toEqual([
      {
        path: "0/vertices/c/0/0/0",
        byteRange: { suffixLength: shardIndexRegionLength(shardShape) },
      },
    ]);
  });

  it("decodes each populated cell to its original bytes via readShardedArrayChunk", async () => {
    const reader = makeShardReader(new Map([["0/vertices/c/0/0/0", shard]]));
    const ctx = makeCtx(reader);
    const signal = new AbortController().signal;
    await expect(
      readShardedArrayChunk(ctx, [0, 0, 0], signal),
    ).resolves.toEqual(cellBytes["0.0.0"]);
    await expect(
      readShardedArrayChunk(ctx, [0, 1, 0], signal),
    ).resolves.toEqual(cellBytes["0.1.0"]);
    await expect(
      readShardedArrayChunk(ctx, [1, 1, 0], signal),
    ).resolves.toEqual(cellBytes["1.1.0"]);
  });

  it("returns undefined for an empty cell without extra requests beyond the index", async () => {
    const reader = makeShardReader(new Map([["0/vertices/c/0/0/0", shard]]));
    const ctx = makeCtx(reader);
    const result = await readShardedArrayChunk(
      ctx,
      [1, 0, 0],
      new AbortController().signal,
    );
    expect(result).toBeUndefined();
    expect(reader.calls).toHaveLength(1);
  });

  it("reuses one cached index fetch across multiple chunks in the same shard", async () => {
    const reader = makeShardReader(new Map([["0/vertices/c/0/0/0", shard]]));
    const ctx = makeCtx(reader);
    const cache = new Map();
    const signal = new AbortController().signal;
    await readShardedArrayChunk(ctx, [0, 0, 0], signal, cache);
    await readShardedArrayChunk(ctx, [0, 1, 0], signal, cache);
    await readShardedArrayChunk(ctx, [1, 1, 0], signal, cache);
    expect(reader.calls.filter((c) => "suffixLength" in c.byteRange)).toHaveLength(1);
  });

  it("does not let one caller's abort poison a concurrent caller sharing the same cached shard-index fetch", async () => {
    // Regression test for a race where the shard-index cache stored a
    // bare Promise tied to whichever caller happened to start the
    // fetch first: if THAT caller's chunk got cancelled (e.g. panned
    // out of view) before the fetch resolved, every other concurrent
    // caller still awaiting the same cached promise saw an unrelated
    // AbortError and lost its own, still-wanted data. The fix routes
    // the shared fetch through `asyncMemoize`'s `SharedAbortController`
    // (aborts only once EVERY registered caller's signal has fired)
    // instead of any single caller's raw signal.
    // Only the shard-INDEX read (a `suffixLength` request) is cached/
    // shared and thus part of the race under test; defer its
    // resolution under manual control. Cell-DATA reads (an
    // `offset`+`length` request) are never cached — serve those
    // immediately, sliced from the real shard bytes, same as
    // `makeShardReader` does for the other tests in this file.
    const { promise: indexFetchPromise, resolve: resolveIndexFetch } =
      Promise.withResolvers<Uint8Array | undefined>();
    let indexFetchCallCount = 0;
    const ctx: ShardedArrayReadContext = {
      kvStoreReadRange: async (
        _path: string,
        byteRange: { offset: number; length: number } | { suffixLength: number },
      ) => {
        if ("suffixLength" in byteRange) {
          ++indexFetchCallCount;
          return indexFetchPromise;
        }
        return shard.subarray(byteRange.offset, byteRange.offset + byteRange.length);
      },
      arrayPath: "0/vertices",
      meta: {
        gridShape: [2, 2, 1],
        shardShape,
        indexLocation: "end",
        cellsAreRaw: true,
      },
    };
    const cache = new Map();
    const controllerA = new AbortController();
    const controllerB = new AbortController();

    // Two concurrent chunk loads, each wanting a DIFFERENT cell of the
    // SAME shard, so both share the one in-flight index fetch.
    const resultA = readShardedArrayChunk(ctx, [0, 0, 0], controllerA.signal, cache);
    const resultB = readShardedArrayChunk(ctx, [1, 1, 0], controllerB.signal, cache);

    // Let both callers register as consumers of the shared fetch, then
    // cancel ONLY caller A. Caller B's own signal never fires.
    await Promise.resolve();
    await Promise.resolve();
    controllerA.abort();

    await expect(resultA).rejects.toThrow();

    // The underlying fetch resolves AFTER A's abort. Before the fix,
    // A's abort would have rejected the shared cached promise outright,
    // and B would see A's AbortError instead of real data.
    resolveIndexFetch(shard.subarray(shard.byteLength - shardIndexRegionLength(shardShape)));
    await expect(resultB).resolves.toEqual(cellBytes["1.1.0"]);
    expect(indexFetchCallCount).toBe(1); // one shared fetch, not one per caller
  });

  it("scoped read fetches an intra-cell byte range, skipping the vlen-bytes header", async () => {
    const reader = makeShardReader(new Map([["0/vertices/c/0/0/0", shard]]));
    const ctx = makeCtx(reader);
    const result = await readShardedArrayChunkScoped(
      ctx,
      [1, 1, 0],
      { offset: 2, length: 2 },
      new AbortController().signal,
    );
    expect(result).toEqual(new Uint8Array([7, 7]));
    const lastCall = reader.calls[reader.calls.length - 1];
    expect("offset" in lastCall.byteRange).toBe(true);
    if ("offset" in lastCall.byteRange) {
      // cell payload starts after the shard-relative offset for (1,1,0)
      // plus the 8-byte vlen-bytes header, plus the requested inner offset.
      const cellOffset = 4 + 2; // (0.0.0) payload=4B data + (0.1.0)=2B data, each +8B header handled below
      expect(lastCall.byteRange.offset).toBeGreaterThan(cellOffset);
      expect(lastCall.byteRange.length).toBe(2);
    }
  });
});
