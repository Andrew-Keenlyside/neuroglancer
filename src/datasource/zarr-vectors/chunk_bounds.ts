/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Chunk-index bounds for a zarr-vectors resolution level.
 *
 * Split out of `geometry_frontend.ts` so it can be unit-tested under Node
 * without the WebGL-coupled render-layer imports.
 */

/**
 * Convert a level's world-space extent into the half-open chunk-index range
 * `[lower, upper)` the sliceview frustum walk enumerates.
 *
 * A zarr-vectors `chunk_shape` is a physical extent, not a voxel count, so it
 * is routinely fractional (a 0.5 mm MERFISH grid) and the arithmetic has to
 * stay in floats -- truncating the shape to an integer yields 0 and a bound of
 * +/-Infinity, and the frustum walk then binary-splits a box it can never
 * reduce to one chunk until it exhausts the stack.
 *
 * Chunks are indexed around the world origin, so negative indices are normal
 * and floor/ceil handle the sign. A degenerate axis (lower === upper) still
 * yields one chunk: a zero-volume range would make the walk terminate before
 * drawing anything.
 */
export function computeChunkIndexBounds(
  lowerBounds: ArrayLike<number>,
  upperBounds: ArrayLike<number>,
  chunkShape: ArrayLike<number>,
  rank = 3,
): { lowerChunkBound: Float32Array; upperChunkBound: Float32Array } {
  const lowerChunkBound = new Float32Array(rank);
  const upperChunkBound = new Float32Array(rank);
  for (let i = 0; i < rank; ++i) {
    const size = chunkShape[i];
    if (!(size > 0)) {
      throw new Error(
        `zarr-vectors: chunk_shape[${i}] = ${size} is not a positive extent`,
      );
    }
    lowerChunkBound[i] = Math.floor(lowerBounds[i] / size);
    upperChunkBound[i] = Math.max(
      Math.ceil(upperBounds[i] / size),
      lowerChunkBound[i] + 1,
    );
  }
  return { lowerChunkBound, upperChunkBound };
}
