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

import { describe, expect, it } from "vitest";

import {
  pickFinestPresentLevelAtOrBelow,
  pickReadyLevelToDraw,
  pickTargetLevelByRealWorldSpacing,
  pickTargetLevelByScreenSize,
} from "#src/skeleton/multiscale_object_selection.js";

describe("pickTargetLevelByScreenSize", () => {
  const levelSpacings = [1, 4, 16, 64, 256];

  it("picks the level whose spacing is closest to the desired spacing", () => {
    expect(pickTargetLevelByScreenSize(levelSpacings, 1)).toBe(0);
    expect(pickTargetLevelByScreenSize(levelSpacings, 4)).toBe(1);
    expect(pickTargetLevelByScreenSize(levelSpacings, 16)).toBe(2);
    expect(pickTargetLevelByScreenSize(levelSpacings, 260)).toBe(4);
  });

  it("picks the nearer level for an in-between desired spacing", () => {
    // 5 is much closer to 4 (level 1) than to 16 (level 2).
    expect(pickTargetLevelByScreenSize(levelSpacings, 5)).toBe(1);
    // 50 is closer to 64 (level 3) than to 16 (level 2).
    expect(pickTargetLevelByScreenSize(levelSpacings, 50)).toBe(3);
  });

  it("breaks exact ties toward the finer (lower-index) level", () => {
    // 10 is exactly equidistant between 4 (level 1) and 16 (level 2).
    expect(pickTargetLevelByScreenSize(levelSpacings, 10)).toBe(1);
  });

  it("quantizes the desired spacing so nearby values don't thrash", () => {
    // Values within the same quarter-octave bucket as 4 should all
    // resolve to the same level as exactly 4 would.
    const near = pickTargetLevelByScreenSize(levelSpacings, 4.02);
    const exact = pickTargetLevelByScreenSize(levelSpacings, 4);
    expect(near).toBe(exact);
  });

  it("returns 0 for a single-level list regardless of desired spacing", () => {
    expect(pickTargetLevelByScreenSize([7], 1000)).toBe(0);
    expect(pickTargetLevelByScreenSize([7], 0.001)).toBe(0);
  });
});

describe("pickFinestPresentLevelAtOrBelow", () => {
  it("returns the target level itself when present", () => {
    expect(
      pickFinestPresentLevelAtOrBelow([true, true, true, true], 2),
    ).toBe(2);
  });

  it("walks toward level 0 when the target level is absent", () => {
    expect(
      pickFinestPresentLevelAtOrBelow([true, true, false, false], 3),
    ).toBe(1);
  });

  it("never walks toward coarser levels", () => {
    // Level 2 absent, level 3 present — must not "fall back" to 3.
    expect(
      pickFinestPresentLevelAtOrBelow([true, true, false, true], 2),
    ).toBe(1);
  });

  it("terminates at level 0 when it is the only present level", () => {
    expect(
      pickFinestPresentLevelAtOrBelow([true, false, false, false], 3),
    ).toBe(0);
  });

  it("clamps an out-of-range target down to the coarsest known level", () => {
    expect(pickFinestPresentLevelAtOrBelow([true, true], 10)).toBe(1);
  });

  it("returns undefined when the object is absent at every level", () => {
    expect(
      pickFinestPresentLevelAtOrBelow([false, false, false], 2),
    ).toBeUndefined();
  });
});

describe("pickReadyLevelToDraw", () => {
  it("prefers the exact target level when it is ready", () => {
    const ready = new Set([1]);
    expect(
      pickReadyLevelToDraw([true, true, true], 1, (l) => ready.has(l)),
    ).toBe(1);
  });

  it("falls back to the nearest ready coarser level", () => {
    const ready = new Set([3]);
    expect(
      pickReadyLevelToDraw([true, true, true, true], 1, (l) => ready.has(l)),
    ).toBe(3);
  });

  it("skips coarser levels the object isn't even present at", () => {
    const ready = new Set([2, 3]);
    // Level 2 absent for this object -- must skip to 3, not stop at 2.
    expect(
      pickReadyLevelToDraw([true, true, false, true], 1, (l) => ready.has(l)),
    ).toBe(3);
  });

  it("never returns a level finer than the target", () => {
    const ready = new Set([0]);
    expect(
      pickReadyLevelToDraw([true, true, true], 1, (l) => ready.has(l)),
    ).toBeUndefined();
  });

  it("returns undefined when nothing at or coarser than target is ready", () => {
    expect(pickReadyLevelToDraw([true, true, true], 1, () => false)).toBe(
      undefined,
    );
  });
});

describe("pickTargetLevelByRealWorldSpacing", () => {
  // Raw chunk spacings in "coordinate units"; metersPerUnit converts them
  // to the same real-world unit `targetSpacingMeters` is expressed in.
  const levelSpacings = [1, 4, 16];

  it("picks the level whose real-world spacing matches the target directly", () => {
    // metersPerUnit=1 -> levelSpacingsMeters == levelSpacings.
    expect(pickTargetLevelByRealWorldSpacing(levelSpacings, 1, 4)).toBe(1);
    expect(pickTargetLevelByRealWorldSpacing(levelSpacings, 1, 16)).toBe(2);
  });

  it("converts through metersPerUnit before comparing", () => {
    // metersPerUnit=2 -> levelSpacingsMeters == [2, 8, 32]; target 8 -> level 1.
    expect(pickTargetLevelByRealWorldSpacing(levelSpacings, 2, 8)).toBe(1);
  });

  it("has no dependency on camera/projection state — same result regardless of caller context", () => {
    // Calling it twice with identical inputs must be perfectly stable;
    // there is no hidden state (unlike the old centroid/matrix approach).
    const a = pickTargetLevelByRealWorldSpacing(levelSpacings, 1, 4);
    const b = pickTargetLevelByRealWorldSpacing(levelSpacings, 1, 4);
    expect(a).toBe(b);
  });
});
