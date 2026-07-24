/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import type { RoiLabelField } from "#src/datasource/zarr-vectors/roi.js";
import { makeLabelSampler } from "#src/datasource/zarr-vectors/roi.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";

// `label_field.js` (and the segment-property map) pull in the WebGL render
// stack at import time. In the node/jsdom test environment `WebGL2RenderingContext`
// is undefined, so stub it before dynamically importing those modules — the same
// approach used by `src/layer/segmentation/index.spec.ts`.
if (!("WebGL2RenderingContext" in globalThis)) {
  Object.defineProperty(globalThis, "WebGL2RenderingContext", {
    value: new Proxy(class WebGL2RenderingContext {} as any, {
      get(target, property, receiver) {
        if (Reflect.has(target, property)) {
          return Reflect.get(target, property, receiver);
        }
        return 0;
      },
    }),
    configurable: true,
  });
}

const { buildModelToVoxelAffine, readParcellationLabels } = await import(
  "#src/datasource/zarr-vectors/label_field.js"
);
const { PreprocessedSegmentPropertyMap, SegmentPropertyMap } = await import(
  "#src/segmentation_display_state/property_map.js"
);

describe("makeLabelSampler round-trips a hand-built field", () => {
  // A 2x2x2 grid, identity model->voxel transform (so model (x,y,z) rounds
  // straight to voxel (x,y,z)). Two voxels are labelled.
  const dims: [number, number, number] = [2, 2, 2];
  const data = new Uint32Array(dims[0] * dims[1] * dims[2]);
  const at = (vx: number, vy: number, vz: number) =>
    vx + dims[0] * (vy + dims[1] * vz);
  data[at(0, 0, 0)] = 11;
  data[at(1, 1, 1)] = 42;
  const field: RoiLabelField = {
    data,
    dims,
    // Identity 4x4 (row-major).
    modelToVoxel: Float32Array.of(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
  };
  const sample = makeLabelSampler(field);

  it("returns the label at a labelled voxel centre", () => {
    expect(sample(0, 0, 0)).toBe(11);
    expect(sample(1, 1, 1)).toBe(42);
  });

  it("rounds to the nearest voxel", () => {
    // (0.4, 0.4, 0.4) rounds to voxel (0,0,0); (0.6, 0.6, 0.6) -> (1,1,1).
    expect(sample(0.4, 0.4, 0.4)).toBe(11);
    expect(sample(0.6, 0.6, 0.6)).toBe(42);
  });

  it("returns 0 for an unlabelled in-bounds voxel", () => {
    expect(sample(1, 0, 0)).toBe(0);
  });

  it("returns 0 for out-of-bounds points", () => {
    expect(sample(-1, 0, 0)).toBe(0);
    expect(sample(0, 0, 2)).toBe(0);
    expect(sample(5, 5, 5)).toBe(0);
  });
});

describe("buildModelToVoxelAffine", () => {
  it("recovers a diagonal affine and folds in the lower bound", () => {
    // A pure per-axis affine: model (x,y,z) -> voxel (2x+10, 3y+20, z+30).
    const globalToVoxel = (
      x: number,
      y: number,
      z: number,
    ): readonly [number, number, number] => [2 * x + 10, 3 * y + 20, z + 30];
    const lo: [number, number, number] = [1, 2, 3];
    const m = buildModelToVoxelAffine(globalToVoxel, lo);

    // Row 0: [2, 0, 0, 10-1]; row 1: [0, 3, 0, 20-2]; row 2: [0, 0, 1, 30-3].
    expect(Array.from(m)).toEqual([
      2, 0, 0, 9, //
      0, 3, 0, 18, //
      0, 0, 1, 27, //
      0, 0, 0, 1,
    ]);

    // The affine must map a model point to (rawVoxel - lo).
    const apply = (x: number, y: number, z: number) => [
      m[0] * x + m[1] * y + m[2] * z + m[3],
      m[4] * x + m[5] * y + m[6] * z + m[7],
      m[8] * x + m[9] * y + m[10] * z + m[11],
    ];
    const raw = globalToVoxel(4, 5, 6);
    expect(apply(4, 5, 6)).toEqual([
      raw[0] - lo[0],
      raw[1] - lo[1],
      raw[2] - lo[2],
    ]);
  });

  it("composes with makeLabelSampler to index the grid directly", () => {
    // Grid is offset by lo=(1,1,0): model (1,1,0) is stored at data voxel 0.
    const dims: [number, number, number] = [2, 2, 1];
    const data = new Uint32Array(dims[0] * dims[1] * dims[2]);
    const at = (vx: number, vy: number, vz: number) =>
      vx + dims[0] * (vy + dims[1] * vz);
    data[at(0, 0, 0)] = 7; // model (1,1,0)
    data[at(1, 0, 0)] = 9; // model (2,1,0)

    // Identity global->voxel; the lower bound is the only offset.
    const globalToVoxel = (
      x: number,
      y: number,
      z: number,
    ): readonly [number, number, number] => [x, y, z];
    const lo: [number, number, number] = [1, 1, 0];
    const field: RoiLabelField = {
      data,
      dims,
      modelToVoxel: buildModelToVoxelAffine(globalToVoxel, lo),
    };
    const sample = makeLabelSampler(field);

    expect(sample(1, 1, 0)).toBe(7);
    expect(sample(2, 1, 0)).toBe(9);
    // Below the lower bound -> out of the stored grid -> background.
    expect(sample(0, 0, 0)).toBe(0);
  });
});

describe("readParcellationLabels", () => {
  function makeLayer(
    pre:
      | InstanceType<typeof PreprocessedSegmentPropertyMap>
      | undefined,
  ): SegmentationUserLayer {
    return {
      displayState: { segmentPropertyMap: { value: pre } },
    } as unknown as SegmentationUserLayer;
  }

  it("returns [] when there is no property map", () => {
    expect(readParcellationLabels(makeLayer(undefined))).toEqual([]);
  });

  it("returns [] when the map has no inline properties", () => {
    const pre = new PreprocessedSegmentPropertyMap(
      new SegmentPropertyMap({ inlineProperties: undefined }),
    );
    expect(readParcellationLabels(makeLayer(pre))).toEqual([]);
  });

  it("reads id, name and colour, index-aligned across columns", () => {
    const ids = BigUint64Array.of(1n, 5n, 7n);
    const pre = new PreprocessedSegmentPropertyMap(
      new SegmentPropertyMap({
        inlineProperties: {
          ids,
          properties: [
            { id: "label", type: "label", values: ["A", "B", "C"] },
            // Middle id has no colour (-1 sentinel).
            {
              id: "color",
              type: "rgb",
              values: Int32Array.of(0x0000ff, -1, 0x00ff00),
            },
          ],
        },
      }),
    );
    expect(readParcellationLabels(makeLayer(pre))).toEqual([
      { id: 1, name: "A", colorPacked: 0x0000ff },
      { id: 5, name: "B" },
      { id: 7, name: "C", colorPacked: 0x00ff00 },
    ]);
  });

  it("falls back to an empty name when there is no label column", () => {
    const pre = new PreprocessedSegmentPropertyMap(
      new SegmentPropertyMap({
        inlineProperties: { ids: BigUint64Array.of(3n), properties: [] },
      }),
    );
    expect(readParcellationLabels(makeLayer(pre))).toEqual([{ id: 3, name: "" }]);
  });
});
