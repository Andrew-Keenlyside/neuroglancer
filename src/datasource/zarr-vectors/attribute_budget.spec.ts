/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  applyAttributeBudget,
  resolveAttributeSelection,
} from "#src/datasource/zarr-vectors/attribute_budget.js";

const NAMES = ["fa", "md", "radius", "curvature", "tortuosity"];

describe("applyAttributeBudget", () => {
  it("keeps everything when the store fits in the budget", () => {
    expect(applyAttributeBudget(NAMES, NAMES, undefined, 10)).toEqual(NAMES);
  });

  it("truncates in the caller's order when the store is too wide", () => {
    // The caller orders store-declared properties first, then the rest
    // alphabetically, so truncation keeps the declared ones.
    expect(applyAttributeBudget(NAMES, NAMES, undefined, 2)).toEqual([
      "fa",
      "md",
    ]);
  });

  it("takes an explicit selection literally, order included", () => {
    expect(
      applyAttributeBudget(NAMES, NAMES, ["tortuosity", "fa"], 10),
    ).toEqual(["tortuosity", "fa"]);
  });

  it("reports a selected name the store does not have", () => {
    expect(() => applyAttributeBudget(NAMES, NAMES, ["gene_Sst"], 10)).toThrow(
      /no readable vertex_attributes\/ entry/,
    );
  });

  it("honours a selection longer than the default", () => {
    // The default is a default, not a ceiling: attributes share one packed
    // texture, so asking for more than the default costs fetch time, not
    // correctness.
    expect(applyAttributeBudget(NAMES, NAMES, NAMES, 3)).toEqual(NAMES);
  });

  it("lets an explicit selection name an attribute the ordering dropped", () => {
    // The store is wider than the budget, so `tortuosity` would be truncated
    // away -- naming it is the whole point of the selector.
    expect(applyAttributeBudget(NAMES, NAMES, ["tortuosity"], 2)).toEqual([
      "tortuosity",
    ]);
  });
});

/** The shape of the store that motivated dtype-before-budget ordering. */
const ZHUANG_LIKE = new Map<string, string>([
  ["brain_section_label", "int32"],
  ["cluster_alias", "int64"],
  ["cluster_confidence_score", "float64"],
  ["donor_genotype", "int32"],
  ["donor_label", "int32"],
  ["donor_sex", "int32"],
  ["gene_a", "float32"],
  ["gene_b", "float32"],
  ["gene_c", "float32"],
  ["gene_d", "float32"],
  ["gene_e", "float32"],
  ["parcellation_label", "variable_length_bytes"],
]);

const DECODABLE = new Set([
  "float32",
  "int32",
  "uint8",
  "float64",
  "int64",
  "uint64",
]);

function readerFor(dtypes: Map<string, string>) {
  const pages: string[][] = [];
  const readDtypes = async (names: readonly string[]) => {
    pages.push([...names]);
    return new Map(names.map((n) => [n, dtypes.get(n)]));
  };
  return { readDtypes, pages };
}

describe("resolveAttributeSelection", () => {
  const orderedNames = [...ZHUANG_LIKE.keys()];

  it("fills the budget with decodable attributes, topping up past the rest", async () => {
    const { readDtypes } = readerFor(ZHUANG_LIKE);
    const { names } = await resolveAttributeSelection({
      orderedNames,
      availableNames: orderedNames,
      selectedAttributes: undefined,
      readDtypes,
      isSupported: (d) => DECODABLE.has(d),
      limit: 6,
    });
    // A budget spent before the dtype check would have stopped at the first
    // six names and then vacated the undecodable one, leaving five.
    expect(names).toHaveLength(6);
    expect(names).not.toContain("parcellation_label");
  });

  it("reads only the pages it needs, not every attribute in the store", async () => {
    const wide = new Map(ZHUANG_LIKE);
    for (let i = 0; i < 500; ++i) wide.set(`gene_x${i}`, "float32");
    const { readDtypes, pages } = readerFor(wide);
    await resolveAttributeSelection({
      orderedNames: [...wide.keys()],
      availableNames: [...wide.keys()],
      selectedAttributes: undefined,
      readDtypes,
      isSupported: (d) => DECODABLE.has(d),
      limit: 4,
    });
    const read = pages.flat().length;
    expect(read).toBeLessThan(10);
    expect(read).toBeGreaterThanOrEqual(4);
  });

  it("reports the dtype of every attribute it keeps", async () => {
    const { readDtypes } = readerFor(ZHUANG_LIKE);
    const { dtypes } = await resolveAttributeSelection({
      orderedNames,
      availableNames: orderedNames,
      selectedAttributes: ["cluster_alias", "gene_a"],
      readDtypes,
      isSupported: (d) => DECODABLE.has(d),
    });
    expect(dtypes.get("cluster_alias")).toBe("int64");
    expect(dtypes.get("gene_a")).toBe("float32");
  });

  it("errors on an explicitly selected attribute it cannot decode", async () => {
    // Silence is what made these columns look like missing data; a name the
    // user typed deserves a reason.
    const { readDtypes } = readerFor(ZHUANG_LIKE);
    await expect(
      resolveAttributeSelection({
        orderedNames,
        availableNames: orderedNames,
        selectedAttributes: ["parcellation_label"],
        readDtypes,
        isSupported: (d) => DECODABLE.has(d),
      }),
    ).rejects.toThrow(/parcellation_label \(variable_length_bytes\)/);
  });

  it("honours an explicit selection longer than the default", async () => {
    const { readDtypes } = readerFor(ZHUANG_LIKE);
    const selection = ["gene_a", "gene_b", "gene_c", "gene_d", "gene_e"];
    const { names } = await resolveAttributeSelection({
      orderedNames,
      availableNames: orderedNames,
      selectedAttributes: selection,
      readDtypes,
      isSupported: (d) => DECODABLE.has(d),
      limit: 2,
    });
    expect(names).toEqual(selection);
  });

  it("skips an attribute whose metadata is unreadable without failing the store", async () => {
    const partial = new Map(ZHUANG_LIKE);
    partial.delete("gene_a");
    const { readDtypes } = readerFor(partial);
    const { names } = await resolveAttributeSelection({
      orderedNames,
      availableNames: orderedNames,
      selectedAttributes: ["gene_a", "gene_b"],
      readDtypes,
      isSupported: (d) => DECODABLE.has(d),
    });
    expect(names).toEqual(["gene_b"]);
  });
});
