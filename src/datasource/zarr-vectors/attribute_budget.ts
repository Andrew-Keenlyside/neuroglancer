/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Which of a store's per-vertex attributes to hand the render layer.
 *
 * Lives apart from `frontend.ts` so it can be unit-tested under Node without
 * dragging in the WebGL-coupled datasource module.
 */

/**
 * How many per-vertex attributes to expose when the user has not chosen.
 *
 * This is a DEFAULT, not a ceiling: an explicit `#attributes=` selection is
 * honoured however long it is. Attributes no longer cost a GPU texture unit
 * each -- they share one packed texture -- so the only thing a wide store
 * still costs is fetch and memory: one blob per attribute per chunk, per
 * level, every time a chunk scrolls into view. A MERFISH panel has one column
 * per gene (1136 in Zhuang-ABCA-1), and defaulting to all of them would turn
 * every chunk into a thousand-request stampede for data no shader reads.
 *
 * So: pick a workable default, say what was left out, and let the user name
 * what they actually want.
 */
export const DEFAULT_VERTEX_ATTRIBUTES = 32;

/**
 * Choose which per-vertex attributes to expose, honouring an explicit
 * `#attributes=` selection when given and otherwise truncating to the default.
 *
 * An explicit selection is taken literally, including its order and length, so
 * the user gets exactly the columns they asked for; names the store does not
 * have are reported rather than silently dropped.  Without one, the caller's
 * order -- store-declared properties first, then the rest alphabetically --
 * decides who survives, and the truncation is announced with the escape hatch,
 * because a silently-dropped gene reads as missing data.
 *
 * The caller filters by dtype BEFORE calling: a column the reader cannot
 * decode must not occupy a slot it will then vacate, which is how a store with
 * six 64-bit obs columns ended up showing four genes out of a ten-attribute
 * budget.
 */
export function applyAttributeBudget(
  orderedNames: readonly string[],
  availableNames: readonly string[],
  selectedAttributes: readonly string[] | undefined,
  limit: number = DEFAULT_VERTEX_ATTRIBUTES,
): string[] {
  if (selectedAttributes !== undefined) {
    const available = new Set(availableNames);
    const missing = selectedAttributes.filter((n) => !available.has(n));
    if (missing.length > 0) {
      throw new Error(
        `zarr-vectors: #attributes names ${JSON.stringify(missing)}, which ` +
          `this store has no readable vertex_attributes/ entry for.`,
      );
    }
    return [...selectedAttributes];
  }
  if (orderedNames.length <= limit) return [...orderedNames];
  const kept = orderedNames.slice(0, limit);
  warnOnceBudget(
    `attribute-budget-${orderedNames.length}`,
    `zarr-vectors: this store declares ${orderedNames.length} per-vertex ` +
      `attributes; exposing the first ${limit}. Append ` +
      "`#attributes=name1,name2` to the source URL to choose others (any " +
      "number of them).",
  );
  return kept;
}

/** Keys already warned about, so one diagnostic does not repeat per level. */
const warnedBudgetKeys = new Set<string>();
function warnOnceBudget(key: string, message: string): void {
  if (warnedBudgetKeys.has(key)) return;
  warnedBudgetKeys.add(key);
  console.warn(message);
}

/**
 * Resolve the attributes to expose, consulting dtypes so that a column the
 * reader cannot decode never occupies a budget slot.
 *
 * The ordering matters and used to be the other way round: the budget was
 * spent first and the dtype check ran second, so a store whose first
 * alphabetical columns were undecodable spent slots on them and vacated them
 * silently. `Zhuang-ABCA-1` lost two of ten that way, and the eight survivors
 * included only four genes.
 *
 * Dtypes are read in pages rather than all at once: one metadata request per
 * attribute is fine for the ~32 that will be kept and ruinous for the 1136 a
 * MERFISH store declares. Each page tops the selection back up to the limit,
 * so the cost tracks what is kept, not what exists.
 *
 * An explicit selection is a different contract: every name in it is read, and
 * an undecodable one is an error rather than a quiet omission -- the user
 * named that column, so silence about it is the failure mode this whole
 * function exists to remove.
 */
export async function resolveAttributeSelection(options: {
  /** Candidate names, most-wanted first (declared properties, then sorted). */
  orderedNames: readonly string[];
  /** Every name the store lists, for validating an explicit selection. */
  availableNames: readonly string[];
  /** The `#attributes=` selection, if the URL carried one. */
  selectedAttributes: readonly string[] | undefined;
  /** Read the declared dtype of each named attribute; `undefined` = unreadable. */
  readDtypes: (
    names: readonly string[],
  ) => Promise<Map<string, string | undefined>>;
  /** Whether a declared dtype is one the reader can decode. */
  isSupported: (dtype: string) => boolean;
  limit?: number;
}): Promise<{ names: string[]; dtypes: Map<string, string> }> {
  const {
    orderedNames,
    availableNames,
    selectedAttributes,
    readDtypes,
    isSupported,
    limit = DEFAULT_VERTEX_ATTRIBUTES,
  } = options;

  if (selectedAttributes !== undefined) {
    const names = applyAttributeBudget(
      orderedNames,
      availableNames,
      selectedAttributes,
      limit,
    );
    const dtypes = await readDtypes(names);
    const rejected: string[] = [];
    const unreadable: string[] = [];
    const kept: string[] = [];
    const keptDtypes = new Map<string, string>();
    for (const name of names) {
      const dtype = dtypes.get(name);
      if (dtype === undefined) {
        unreadable.push(name);
        continue;
      }
      if (!isSupported(dtype)) {
        rejected.push(`${name} (${dtype})`);
        continue;
      }
      kept.push(name);
      keptDtypes.set(name, dtype);
    }
    if (rejected.length > 0) {
      throw new Error(
        `zarr-vectors: #attributes names ${JSON.stringify(rejected)}, whose ` +
          "dtype this reader cannot decode into a per-vertex value.",
      );
    }
    if (unreadable.length > 0) {
      warnOnceBudget(
        `attribute-unreadable-${unreadable.join(",")}`,
        `zarr-vectors: could not read array metadata for ` +
          `${JSON.stringify(unreadable)}; those attributes are unavailable.`,
      );
    }
    return { names: kept, dtypes: keptDtypes };
  }

  const kept: string[] = [];
  const keptDtypes = new Map<string, string>();
  const skipped: string[] = [];
  let next = 0;
  while (kept.length < limit && next < orderedNames.length) {
    const page = orderedNames.slice(next, next + (limit - kept.length));
    next += page.length;
    const dtypes = await readDtypes(page);
    for (const name of page) {
      const dtype = dtypes.get(name);
      if (dtype === undefined || !isSupported(dtype)) {
        skipped.push(dtype === undefined ? name : `${name} (${dtype})`);
        continue;
      }
      kept.push(name);
      keptDtypes.set(name, dtype);
    }
  }
  if (skipped.length > 0) {
    warnOnceBudget(
      `attribute-skipped-${skipped.join(",")}`,
      `zarr-vectors: skipped ${JSON.stringify(skipped)} — no readable dtype ` +
        "this reader can decode into a per-vertex value.",
    );
  }
  if (next < orderedNames.length) {
    warnOnceBudget(
      `attribute-budget-${orderedNames.length}`,
      `zarr-vectors: this store declares ${orderedNames.length} per-vertex ` +
        `attributes; exposing ${kept.length}. Append ` +
        "`#attributes=name1,name2` to the source URL to choose others (any " +
        "number of them).",
    );
  }
  return { names: kept, dtypes: keptDtypes };
}
