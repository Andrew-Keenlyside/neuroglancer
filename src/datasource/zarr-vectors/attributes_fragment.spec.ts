/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  formatAttributesFragment,
  parseAttributesFragment,
} from "#src/datasource/zarr-vectors/attributes_fragment.js";

describe("the #attributes= fragment", () => {
  it("is absent when the URL has no fragment", () => {
    expect(parseAttributesFragment(undefined)).toBeUndefined();
    expect(parseAttributesFragment("")).toBeUndefined();
    expect(formatAttributesFragment(undefined)).toBe("");
  });

  it("parses a plain list, trimming whitespace", () => {
    expect(parseAttributesFragment("attributes=fa, md ,radius")).toEqual([
      "fa",
      "md",
      "radius",
    ]);
  });

  it("round-trips names that need percent-encoding", () => {
    // The formatted fragment is saved into the layer's JSON, so parse(format(x))
    // must be x -- otherwise reopening a saved link fails or selects the wrong
    // columns.
    for (const names of [
      ["gene_Sst"],
      ["odd name"],
      ["pct%"],
      ["has,comma", "plain"],
      ["100%", "a+b"],
    ]) {
      const fragment = formatAttributesFragment(names);
      expect(parseAttributesFragment(fragment.slice(1))).toEqual(names);
    }
  });

  it("keeps a percent-encoded comma as one name", () => {
    expect(parseAttributesFragment("attributes=has%2Ccomma,plain")).toEqual([
      "has,comma",
      "plain",
    ]);
  });

  it("takes a stray percent literally rather than failing the load", () => {
    // A hand-typed `pct%` is not valid percent-encoding; refusing to open the
    // layer over it would be worse than reading it as written.
    expect(parseAttributesFragment("attributes=pct%")).toEqual(["pct%"]);
  });

  it("rejects a fragment that is not an attribute list", () => {
    expect(() => parseAttributesFragment("something=else")).toThrow(
      /only supported fragment/,
    );
  });
});
