/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Pins the contract that makes the fork's Skeleton tab appear for a
 * zarr-vectors layer.
 *
 * The layer does not ask a source what it is; it duck-types it
 * (`isSpatiallyIndexedSkeletonSource`) and hides the tab when the shape does
 * not match. That check is easy to break by renaming a method, and nothing else
 * in the build would notice -- the tab would just silently stop appearing. So
 * assert the shape directly, against the real prototype.
 */

import { describe, expect, it } from "vitest";

import { ZarrVectorsSpatialGeometrySource } from "#src/datasource/zarr-vectors/geometry_frontend.js";
import {
  isEditableSpatiallyIndexedSkeletonSource,
  isSpatiallyIndexedSkeletonSource,
} from "#src/skeleton/spatial_skeleton_manager.js";

/**
 * A stand-in carrying the class's real prototype plus the parameter blob the
 * editable-source getters read. Constructing the source for real needs a chunk
 * manager and a worker; the duck-type only reads the prototype.
 */
function sourceShape(parameters: object = {}): object {
  const source = Object.create(ZarrVectorsSpatialGeometrySource.prototype);
  Object.defineProperty(source, "parameters", { value: parameters });
  return source;
}

/** The parameters a layer opened with `#edit=<url>` carries. */
const EDITABLE = {
  editServiceUrl: "http://127.0.0.1:9099",
  editStore: "single_axon.zv",
};

describe("zarr-vectors spatial skeleton source contract", () => {
  it("satisfies the read-side duck-type the Skeleton tab is gated on", () => {
    expect(isSpatiallyIndexedSkeletonSource(sourceShape())).toBe(true);
  });

  it("is read-only without an edit service, so edit actions stay disabled", () => {
    const source = sourceShape() as { readonly: boolean };
    expect(source.readonly).toBe(true);
    expect(isEditableSpatiallyIndexedSkeletonSource(sourceShape())).toBe(false);
  });

  it("becomes editable when the layer carries an edit service", () => {
    // This is the whole mechanism behind `#edit=<url>`: the same class reports
    // itself editable only when it has somewhere to send the edit. If this
    // fails, the Skeleton tools go grey with "the active spatial skeleton
    // source is read-only" and nothing else in the build notices.
    const source = sourceShape(EDITABLE) as { readonly: boolean };
    expect(source.readonly).toBe(false);
    expect(
      isEditableSpatiallyIndexedSkeletonSource(sourceShape(EDITABLE)),
    ).toBe(true);
  });

  it("exposes all five required command factories when editable", () => {
    // `isEditableSpatiallyIndexedSkeletonSource` demands every required slot;
    // omitting one makes the source non-editable outright, so the four that
    // this prototype does not implement are present and fail loudly instead.
    const source = sourceShape(EDITABLE) as Record<string, { action?: string }>;
    for (const name of [
      "splitSkeletonsCommand",
      "mergeSkeletonsCommand",
      "addNodesCommand",
      "moveNodesCommand",
      "deleteNodesCommand",
    ]) {
      expect(source[name], name).toBeDefined();
    }
    expect(source.splitSkeletonsCommand!.action).toBe("splitSkeletons");
  });

  it("exposes getSkeleton, the only one of the four the UI calls", () => {
    const source = sourceShape() as Record<string, unknown>;
    for (const name of [
      "getSkeleton",
      "listSkeletons",
      "fetchNodes",
      "getSpatialIndexMetadata",
    ]) {
      expect(typeof source[name]).toBe("function");
    }
  });
});
