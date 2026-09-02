/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it, vi } from "vitest";
import type {
  AttrCatalogLayer,
  AttrStats,
} from "#src/datasource/zarr-vectors/attribute_catalog.js";
import {
  attrKey,
  AttrStatsCache,
  filterScope,
  flagFilter,
  flagFilterValue,
  fullRangeFilter,
  hasApproximateValues,
  isFlagAttr,
  listAttrChoices,
  measureObjectAttr,
} from "#src/datasource/zarr-vectors/attribute_catalog.js";
import type { VertexAttrStats } from "#src/skeleton/spatial_base.js";

const stats = (partial: Partial<AttrStats>): AttrStats => ({
  name: "attr",
  scope: "vertex",
  count: 10,
  min: 0,
  max: 1,
  integral: true,
  distinct: 2,
  ...partial,
});

/** A layer with the given per-object properties and per-vertex attribute names. */
function fakeLayer(options: {
  objectProperties?: { id: string; values: number[]; dataType?: number }[];
  vertexNames?: string[];
  vertexDtypes?: string[];
  vertexStats?: (names: readonly string[]) => VertexAttrStats[];
}): AttrCatalogLayer {
  const properties = (options.objectProperties ?? []).map((p) => ({
    id: p.id,
    values: p.values,
    bounds: [Math.min(...p.values), Math.max(...p.values)] as [number, number],
    dataType: p.dataType,
  }));
  return {
    displayState: {
      segmentPropertyMap: {
        value: { numericalProperties: properties },
      },
      roiVertexAttributeNames: options.vertexNames,
      roiVertexAttributeDtypes: options.vertexDtypes,
      computeRoiVertexAttrStats:
        options.vertexStats === undefined
          ? undefined
          : async (names) => options.vertexStats!(names),
    },
  };
}

describe("listAttrChoices", () => {
  it("offers per-object attributes before per-vertex ones", () => {
    const layer = fakeLayer({
      objectProperties: [{ id: "length", values: [1, 2] }],
      vertexNames: ["gene_a", "flag"],
    });
    expect(listAttrChoices(layer)).toEqual([
      { name: "length", scope: "object" },
      { name: "gene_a", scope: "vertex" },
      { name: "flag", scope: "vertex" },
    ]);
  });

  it("is empty for a layer with neither tier", () => {
    expect(listAttrChoices(fakeLayer({}))).toEqual([]);
  });

  it("keys the two tiers apart, so a shared name is two choices", () => {
    expect(attrKey("x", "object")).not.toBe(attrKey("x", "vertex"));
  });
});

describe("measureObjectAttr", () => {
  it("reports the range, integrality and distinct count", () => {
    const measured = measureObjectAttr({
      id: "length",
      values: [10, 20, 20, 30],
      bounds: [10, 30],
    });
    expect(measured).toEqual({
      name: "length",
      scope: "object",
      count: 4,
      min: 10,
      max: 30,
      integral: true,
      distinct: 3,
    });
  });

  it("ignores non-finite values rather than letting them define the range", () => {
    const measured = measureObjectAttr({
      id: "fa",
      values: [Number.NaN, 0.25, 0.75],
      bounds: [0, 1],
    });
    expect(measured.count).toBe(2);
    expect(measured.min).toBe(0.25);
    expect(measured.max).toBe(0.75);
    expect(measured.integral).toBe(false);
  });

  it("reports an empty column as measurable-but-empty, not as a [0,0] range", () => {
    const measured = measureObjectAttr({ id: "x", values: [], bounds: [0, 0] });
    expect(measured.count).toBe(0);
  });
});

describe("flag detection", () => {
  it("treats a two-valued integer column as a flag", () => {
    expect(isFlagAttr(stats({ min: 0, max: 1, distinct: 2 }))).toBe(true);
    // A uint8 mask spelled 0/255 is the same thing.
    expect(isFlagAttr(stats({ min: 0, max: 255, distinct: 2 }))).toBe(true);
  });

  it("does not treat a measurement or a category set as a flag", () => {
    expect(isFlagAttr(stats({ integral: false, distinct: 2 }))).toBe(false);
    expect(isFlagAttr(stats({ distinct: 3 }))).toBe(false);
    expect(isFlagAttr(stats({ distinct: 1 }))).toBe(false);
    expect(isFlagAttr(stats({ count: 0, distinct: 0 }))).toBe(false);
    // A wide integer column with one observed value is a category code that
    // happens to be constant here, not a flag.
    expect(
      isFlagAttr(stats({ distinct: 1, min: 5, max: 5, dtype: "int32" })),
    ).toBe(false);
  });

  it("still offers a checkbox for a boolean column that is all-true in view", () => {
    // `high_quality_transfer` in the MERFISH panel: uint8, every loaded cell 1.
    const s = stats({
      name: "high_quality_transfer",
      min: 1,
      max: 1,
      distinct: 1,
      dtype: "uint8",
    });
    expect(isFlagAttr(s)).toBe(true);
    const on = flagFilter(s, true);
    const off = flagFilter(s, false);
    // The two states must actually differ, which the observed extremes (1..1)
    // cannot express -- the canonical 0/1 split does.
    expect(on).not.toEqual(off);
    expect(flagFilterValue(on, s)).toBe(true);
    expect(flagFilterValue(off, s)).toBe(false);
    // "true" selects the value that is there; "false" selects none of it.
    expect(on.min <= 1 && on.max >= 1).toBe(true);
    expect(off.max < 1).toBe(true);
  });

  it("round-trips true/false through the predicate, for 0/1 and for 0/255", () => {
    for (const max of [1, 255]) {
      const s = stats({ name: "flag", min: 0, max, distinct: 2 });
      const on = flagFilter(s, true);
      const off = flagFilter(s, false);
      expect(flagFilterValue(on, s)).toBe(true);
      expect(flagFilterValue(off, s)).toBe(false);
      // Both ends are finite, so the state survives its JSON round trip.
      expect(Number.isFinite(on.min) && Number.isFinite(on.max)).toBe(true);
      // The ranges select the two states and nothing else.
      expect(on.min <= max && on.max >= max).toBe(true);
      expect(off.min <= 0 && off.max >= 0).toBe(true);
      expect(on.min > 0).toBe(true);
      expect(off.max < max).toBe(true);
    }
  });

  it("carries the scope into the predicate only for the per-vertex tier", () => {
    const vertex = flagFilter(stats({ scope: "vertex" }), true);
    const object = flagFilter(stats({ scope: "object" }), true);
    expect(filterScope(vertex)).toBe("vertex");
    expect(object.scope).toBeUndefined();
    expect(filterScope(object)).toBe("object");
  });

  it("flags a downcast 64-bit column whose values pass float32's exact range", () => {
    // An id column: neighbouring ids collapse onto one float32 value.
    expect(
      hasApproximateValues(
        stats({ dtype: "uint64", min: 0, max: 864691135000000000 }),
      ),
    ).toBe(true);
    // The same dtype stays exact while the values are small (a category code).
    expect(
      hasApproximateValues(stats({ dtype: "int64", min: 0, max: 5000 })),
    ).toBe(false);
    // A 32-bit column was never downcast.
    expect(
      hasApproximateValues(stats({ dtype: "float32", min: 0, max: 1e9 })),
    ).toBe(false);
    // An unknown dtype claims nothing.
    expect(hasApproximateValues(stats({ min: 0, max: 1e9 }))).toBe(false);
  });

  it("seeds a range predicate spanning everything observed", () => {
    const s = stats({ name: "gene_a", min: 0.5, max: 12, integral: false });
    expect(fullRangeFilter(s)).toEqual({
      name: "gene_a",
      min: 0.5,
      max: 12,
      scope: "vertex",
    });
  });
});

describe("AttrStatsCache", () => {
  it("carries the per-vertex dtype the worker cannot know", async () => {
    const cache = new AttrStatsCache(
      fakeLayer({
        vertexNames: ["flag", "gene_a"],
        vertexDtypes: ["uint8", "float32"],
        vertexStats: (names) =>
          names.map((name) => ({
            name,
            count: 5,
            min: 1,
            max: 1,
            integral: true,
            distinct: 1,
          })),
      }),
    );
    const measured = await cache.request("flag", "vertex");
    expect(measured?.dtype).toBe("uint8");
    expect(isFlagAttr(measured!)).toBe(true);
    const gene = await cache.request("gene_a", "vertex");
    expect(gene?.dtype).toBe("float32");
    expect(isFlagAttr(gene!)).toBe(false);
  });

  it("measures a per-object attribute without asking the worker", async () => {
    const compute = vi.fn();
    const layer = fakeLayer({
      objectProperties: [{ id: "length", values: [1, 5] }],
      vertexStats: compute as never,
    });
    const cache = new AttrStatsCache(layer);
    const measured = await cache.request("length", "object");
    expect(measured?.max).toBe(5);
    expect(compute).not.toHaveBeenCalled();
    expect(cache.get("length", "object")?.max).toBe(5);
  });

  it("asks the worker once per per-vertex attribute and caches the answer", async () => {
    let calls = 0;
    const layer = fakeLayer({
      vertexNames: ["gene_a"],
      vertexStats: (names) => {
        ++calls;
        return names.map((name) => ({
          name,
          count: 3,
          min: 0,
          max: 7,
          integral: false,
          distinct: 3,
        }));
      },
    });
    const cache = new AttrStatsCache(layer);
    const [a, b] = await Promise.all([
      cache.request("gene_a", "vertex"),
      cache.request("gene_a", "vertex"),
    ]);
    expect(a?.max).toBe(7);
    expect(b?.max).toBe(7);
    expect(a?.scope).toBe("vertex");
    expect(calls).toBe(1); // the concurrent request was deduplicated
    await cache.request("gene_a", "vertex");
    expect(calls).toBe(1); // and the answer was kept
    cache.invalidate();
    await cache.request("gene_a", "vertex");
    expect(calls).toBe(2); // …until the resident geometry it described moved
  });

  it("answers undefined, not an exception, when there is nothing to measure", async () => {
    const cache = new AttrStatsCache(fakeLayer({ vertexNames: ["gene_a"] }));
    // No worker link: the layer's geometry has not loaded yet.
    expect(await cache.request("gene_a", "vertex")).toBeUndefined();
    // An unknown per-object name.
    expect(await cache.request("nope", "object")).toBeUndefined();
  });

  it("swallows a failed measurement so the next open retries it", async () => {
    let calls = 0;
    const cache = new AttrStatsCache(
      fakeLayer({
        vertexNames: ["gene_a"],
        vertexStats: () => {
          ++calls;
          throw new Error("worker not ready");
        },
      }),
    );
    expect(await cache.request("gene_a", "vertex")).toBeUndefined();
    expect(await cache.request("gene_a", "vertex")).toBeUndefined();
    expect(calls).toBe(2);
  });
});
