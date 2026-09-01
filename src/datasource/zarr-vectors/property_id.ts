/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Neuroglancer annotation property identifiers must match
 * `/^[a-z][a-zA-Z0-9_]*$/`: they become `prop_<id>()` accessors in generated
 * GLSL, so anything else would not compile.  Store attribute names carry no
 * such restriction — MERFISH gene panels ship names like `gene_H2-Q2` — so map
 * each to the nearest legal identifier (and disambiguate collisions) rather
 * than failing the whole datasource.  Callers keep the original name as the
 * property description, and `attributeNames[i]` still holds the on-disk name.
 */
export function toAnnotationPropertyId(
  name: string,
  used: Set<string>,
): string {
  let base = name.replace(/[^a-zA-Z0-9_]/g, "_");
  if (!/^[a-z]/.test(base)) base = `p_${base}`;
  let id = base;
  for (let i = 2; used.has(id); ++i) id = `${base}_${i}`;
  used.add(id);
  return id;
}
