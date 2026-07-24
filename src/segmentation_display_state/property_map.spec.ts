/**
 * @license
 * Copyright 2021 Google Inc.
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

import { describe, test, expect } from "vitest";
import {
  mergeSegmentPropertyMaps,
  parseSegmentQuery,
  PreprocessedSegmentPropertyMap,
  SegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { DataType } from "#src/util/data_type.js";

describe("PreprocessedSegmentPropertyMap", () => {
  test("handles lookups correctly", () => {
    const map = new PreprocessedSegmentPropertyMap({
      inlineProperties: {
        ids: BigUint64Array.of(5n, 15n, 0x500000014n),
        properties: [],
      },
    });
    expect(map.getSegmentInlineIndex(5n)).toEqual(0);
    expect(map.getSegmentInlineIndex(15n)).toEqual(1);
    expect(map.getSegmentInlineIndex(0x500000014n)).toEqual(2);
    expect(map.getSegmentInlineIndex(0x50000001en)).toEqual(-1);
    expect(map.getSegmentInlineIndex(0n)).toEqual(-1);
  });

  test("returns rgb property colors, packed as R|G<<8|B<<16", () => {
    const map = new PreprocessedSegmentPropertyMap({
      inlineProperties: {
        ids: BigUint64Array.of(5n, 15n),
        properties: [
          // #010203 => R=1, G=2, B=3 => 1 | 2<<8 | 3<<16.
          {
            type: "rgb",
            id: "color",
            values: Int32Array.of(1 | (2 << 8) | (3 << 16), -1),
          },
        ],
      },
    });
    expect(map.getSegmentColor(5n)).toEqual(1 | (2 << 8) | (3 << 16));
    // -1 sentinel => no color for this id.
    expect(map.getSegmentColor(15n)).toBeUndefined();
    // id not in the map.
    expect(map.getSegmentColor(99n)).toBeUndefined();
  });

  test("returns undefined color when there is no color property", () => {
    const map = new PreprocessedSegmentPropertyMap({
      inlineProperties: {
        ids: BigUint64Array.of(5n),
        properties: [{ type: "label", id: "label", values: ["a"] }],
      },
    });
    expect(map.getSegmentColor(5n)).toBeUndefined();
  });
});

describe("mergeSegmentPropertyMaps", () => {
  test("works correctly for 2 maps", () => {
    const a = new SegmentPropertyMap({
      inlineProperties: {
        ids: BigUint64Array.of(5n, 6n, 8n),
        properties: [{ type: "string", id: "prop1", values: ["x", "y", "z"] }],
      },
    });
    const b = new SegmentPropertyMap({
      inlineProperties: {
        ids: BigUint64Array.of(5n, 7n),
        properties: [{ type: "string", id: "prop2", values: ["a", "b"] }],
      },
    });
    expect(mergeSegmentPropertyMaps([])).toBe(undefined);
    expect(mergeSegmentPropertyMaps([a])).toBe(a);
    expect(mergeSegmentPropertyMaps([b])).toBe(b);
    const c = mergeSegmentPropertyMaps([a, b]);
    expect(c?.inlineProperties).toEqual({
      ids: BigUint64Array.of(5n, 6n, 7n, 8n),
      properties: [
        { type: "string", id: "prop1", values: ["x", "y", "", "z"] },
        { type: "string", id: "prop2", values: ["a", "", "b", ""] },
      ],
    });
  });

  test("fills missing ids in a merged rgb property with the -1 sentinel", () => {
    const a = new SegmentPropertyMap({
      inlineProperties: {
        ids: BigUint64Array.of(5n, 6n, 8n),
        properties: [
          { type: "rgb", id: "color", values: Int32Array.of(10, 20, 30) },
        ],
      },
    });
    const b = new SegmentPropertyMap({
      inlineProperties: {
        ids: BigUint64Array.of(5n, 7n),
        properties: [{ type: "string", id: "prop2", values: ["a", "b"] }],
      },
    });
    const c = mergeSegmentPropertyMaps([a, b]);
    expect(c?.inlineProperties).toEqual({
      ids: BigUint64Array.of(5n, 6n, 7n, 8n),
      properties: [
        // ids 5/6/8 keep their colors; id 7 (only in `b`) gets -1 = no color.
        { type: "rgb", id: "color", values: Int32Array.of(10, 20, -1, 30) },
        { type: "string", id: "prop2", values: ["a", "", "b", ""] },
      ],
    });
    const preprocessed = new PreprocessedSegmentPropertyMap(c!);
    expect(preprocessed.getSegmentColor(6n)).toEqual(20);
    expect(preprocessed.getSegmentColor(7n)).toBeUndefined();
  });
});

describe("parseSegmentQuery", () => {
  const map = new PreprocessedSegmentPropertyMap({
    inlineProperties: {
      ids: BigUint64Array.of(),
      properties: [
        { type: "label", id: "label", values: [] },
        {
          type: "number",
          dataType: DataType.INT32,
          description: undefined,
          id: "prop1",
          values: Int32Array.of(),
          bounds: [-10, 100],
        },
        {
          id: "tags",
          type: "tags",
          tags: ["abc", "def"],
          tagDescriptions: ["foo", "bar"],
          values: [],
        },
      ],
    },
  });

  test("handles empty query", () => {
    expect(parseSegmentQuery(undefined, "")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "id",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles single number", () => {
    expect(parseSegmentQuery(undefined, "123")).toMatchInlineSnapshot(`
      {
        "ids": [
          123n,
        ],
      }
    `);
  });

  test("handles multiple numbers", () => {
    expect(parseSegmentQuery(undefined, "123 456")).toMatchInlineSnapshot(`
      {
        "ids": [
          123n,
          456n,
        ],
      }
    `);
  });

  test("handles regular expression", () => {
    expect(parseSegmentQuery(map, "/xyz")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [],
        "prefix": undefined,
        "regexp": /xyz/,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles prefix", () => {
    expect(parseSegmentQuery(map, "xyz")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [],
        "prefix": "xyz",
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles prefix", () => {
    expect(parseSegmentQuery(map, "xyz")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [],
        "prefix": "xyz",
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles numeric > comparison", () => {
    expect(parseSegmentQuery(map, "prop1>5")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [
          {
            "bounds": [
              6,
              100,
            ],
            "fieldId": "prop1",
          },
        ],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles numeric >= comparison", () => {
    expect(parseSegmentQuery(map, "prop1>=5")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [
          {
            "bounds": [
              5,
              100,
            ],
            "fieldId": "prop1",
          },
        ],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles numeric = comparison", () => {
    expect(parseSegmentQuery(map, "prop1=5")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [
          {
            "bounds": [
              5,
              5,
            ],
            "fieldId": "prop1",
          },
        ],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles numeric >= comparison", () => {
    expect(parseSegmentQuery(map, "prop1>=5")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [
          {
            "bounds": [
              5,
              100,
            ],
            "fieldId": "prop1",
          },
        ],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles numeric > comparison", () => {
    expect(parseSegmentQuery(map, "prop1>5")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [
          {
            "bounds": [
              6,
              100,
            ],
            "fieldId": "prop1",
          },
        ],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles numeric > comparison negative", () => {
    expect(parseSegmentQuery(map, "prop1>-5")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [
          {
            "bounds": [
              -4,
              100,
            ],
            "fieldId": "prop1",
          },
        ],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles sort field", () => {
    expect(parseSegmentQuery(map, ">prop1")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [],
        "includeTags": [],
        "numericalConstraints": [],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "prop1",
            "order": ">",
          },
        ],
      }
    `);
  });

  test("handles column inclusions", () => {
    expect(parseSegmentQuery(map, "|prop1")).toMatchInlineSnapshot(`
      {
        "excludeTags": [],
        "includeColumns": [
          "prop1",
        ],
        "includeTags": [],
        "numericalConstraints": [],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });

  test("handles tags", () => {
    expect(parseSegmentQuery(map, "#abc -#def")).toMatchInlineSnapshot(`
      {
        "excludeTags": [
          "def",
        ],
        "includeColumns": [],
        "includeTags": [
          "abc",
        ],
        "numericalConstraints": [],
        "prefix": undefined,
        "regexp": undefined,
        "sortBy": [
          {
            "fieldId": "label",
            "order": "<",
          },
        ],
      }
    `);
  });
});
