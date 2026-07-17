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

/**
 * Resolve the skeleton shader a layer may adopt as its default, given one
 * candidate per skeleton subsource (`undefined` = "no opinion").
 *
 * A segmentation layer has a single `skeletonRenderingOptions.shader` shared by
 * every skeleton subsource it draws, so a datasource-nominated shader is only
 * safe when they all agree. The streamline default calls `prop_tangent()`,
 * which exists only for geometry that carries a synthesised tangent; applying
 * it to a layer that also holds a conventional (e.g. precomputed) skeleton
 * would fail to compile.
 *
 * Agreement is the test, deliberately, rather than "is this the
 * spatially-indexed source?". Source class is the wrong proxy in both
 * directions: it rejects the pass-2 object-keyed source, which is the same
 * geometry nominating the same shader (rejecting it silently reverted
 * streamlines to hash colours), and it would accept any future source that
 * happened to share a base class while needing different attributes.
 *
 * Returns `undefined` -- meaning "keep the generic default" -- when no
 * candidate has an opinion, when any candidate abstains, or when two disagree.
 * An abstention is not consent: a source that nominates nothing says nothing
 * about whether another source's shader compiles against its geometry.
 *
 * Mesh subsources must not be passed here; they carry their own shader and get
 * no vote.
 */
export function resolveSkeletonDefaultShader(
  candidates: readonly (string | undefined)[],
): string | undefined {
  let resolved: string | undefined;
  for (const candidate of candidates) {
    if (candidate === undefined) return undefined;
    if (resolved === undefined) {
      resolved = candidate;
    } else if (resolved !== candidate) {
      return undefined;
    }
  }
  return resolved;
}
