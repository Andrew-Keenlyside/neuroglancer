/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Probes a single object across every pyramid level of a zarr-vectors
 * store, for the object-keyed multi-resolution skeleton subsource
 * (`ZarrVectorsMultiscaleObjectKeyedSkeletonSourceBackend`,
 * `skeleton_backend.ts`). Each level's `object_index/manifests` array is
 * already written independently by the pyramid coarsener (see
 * `zarr-vectors-tools/zarr_vectors_tools/multiresolution/coarsen.py`), so
 * "is this object present at level N" can be answered by re-reading
 * `readObjectManifest` once per level — no writer-side changes are needed
 * for this feature.
 */

import {
  readObjectManifest,
  type ObjectManifestReaderOptions,
} from "#src/datasource/zarr-vectors/object_manifest_reader.js";

export type LevelManifestProbeOptions = Omit<
  ObjectManifestReaderOptions,
  "sidNdim"
>;

export interface ObjectPyramidProbeResult {
  /** Index 0 = level 0 (finest). `true` iff the object's manifest at that level is non-empty. */
  readonly presentLevels: boolean[];
}

/**
 * Fans out one `readObjectManifest` call per level in parallel and builds
 * the per-level presence bitmap.
 *
 * A level counts as "present" iff its manifest is defined *and non-empty*
 * — the pyramid coarsener writes an empty (not absent) manifest for
 * objects dropped by a coarse level's sparsity threshold, so `undefined`
 * (out-of-range OID / missing manifest chunk file) and `[]` (genuinely
 * sparsified out) must be distinguished from "present."
 *
 * A `levels` entry may itself be `undefined` — the caller passes this for
 * a level whose `object_index/manifests` array doesn't exist at all (e.g.
 * a coarse level a writer chose not to build an object index for at all,
 * as opposed to building one where this particular object's manifest
 * happens to be empty). Such levels are treated as "not present" without
 * ever calling `readObjectManifest` for them — critically, one level
 * lacking an object index must not fail the probe for every *other*
 * level, the way letting a per-level fetch exception propagate out of
 * `Promise.all` would.
 */
export async function probeObjectAcrossLevels(
  resolvedOid: number,
  levels: readonly (LevelManifestProbeOptions | undefined)[],
  sidNdim: number,
  signal: AbortSignal,
): Promise<ObjectPyramidProbeResult> {
  const manifestsPerLevel = await Promise.all(
    levels.map((levelOptions) =>
      levelOptions === undefined
        ? undefined
        : readObjectManifest(resolvedOid, { ...levelOptions, sidNdim }, signal),
    ),
  );
  const presentLevels: boolean[] = manifestsPerLevel.map(
    (manifest) => manifest !== undefined && manifest.length > 0,
  );
  return { presentLevels };
}
