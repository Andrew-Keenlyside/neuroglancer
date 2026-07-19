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
 * @file Document format for a saved ROI group.
 *
 * One document is one named dissection, stored at `groups/<id>.json`.  The
 * group itself is the verbatim output of `groupToJson` from the zarr-vectors
 * filter state, so a saved document and the URL hash carry identical group
 * JSON and there is only one serialisation to keep correct.
 *
 * Two things the URL form does not need, and this one does:
 *
 *  - A STABLE ID.  `RoiGroup.id` is session-scoped and deliberately dropped by
 *    `groupToJson`; documents need an identity that survives a reload so they
 *    can be updated and deleted.
 *  - A SOURCE REFERENCE.  ROI shapes are expressed in tract model space, so a
 *    group only means anything against the store it was drawn on.  In the URL
 *    that is implicit — the group travels inside a state that also names the
 *    data source — but a standalone document has to say so itself.
 */

import {
  verifyObject,
  verifyObjectProperty,
  verifyOptionalObjectProperty,
  verifyString,
} from "#src/util/json.js";
import { getRandomHexString } from "#src/util/random.js";

export const ROI_GROUP_SCHEMA_VERSION = 1;

/** Object name prefix under which documents live in the bucket. */
export const ROI_GROUP_PREFIX = "groups/";

export interface RoiGroupSource {
  /** Data source URL the ROIs were drawn against, e.g. `zarr-vectors://...`. */
  url: string;
  /** Coordinate space of the layer, for diagnosing a mismatched load. */
  coordinateSpace?: any;
}

/**
 * Where the dissection was drawn.
 *
 * Recorded as a viewer URL rather than an embedded state object: the URL hash
 * IS neuroglancer's serialisation of a scene, so this stays in step with the
 * viewer's own format for free and is directly openable by a person.
 */
export interface RoiGroupScene {
  /** Full viewer URL, including the `#!` state fragment. */
  url?: string;
  /** Layer the group belonged to, to reattach it to the right one on load. */
  layerName?: string;
}

export interface RoiGroupDocument {
  schemaVersion: number;
  /** Stable, unguessable document id; also the object name stem. */
  id: string;
  /** Verbatim `groupToJson` output: name, color, rois, visible?, opacity?, highDetail?. */
  group: any;
  source: RoiGroupSource;
  /**
   * The view the dissection was drawn in, so a saved group can be reopened in
   * context rather than as bare geometry.  Optional: a group saved without it
   * is still loadable.
   */
  scene?: RoiGroupScene;
  /** Email from the id token; provenance only, never an access decision. */
  createdBy?: string;
  createdAt: string;
  updatedAt: string;
}

export function roiGroupObjectName(id: string): string {
  return `${ROI_GROUP_PREFIX}${id}.json`;
}

/** Extracts the document id from an object name, or undefined if it is not one. */
export function roiGroupIdFromObjectName(name: string): string | undefined {
  if (!name.startsWith(ROI_GROUP_PREFIX) || !name.endsWith(".json")) {
    return undefined;
  }
  const id = name.slice(ROI_GROUP_PREFIX.length, -".json".length);
  return id.length > 0 && !id.includes("/") ? id : undefined;
}

export function makeRoiGroupDocument(options: {
  group: any;
  source: RoiGroupSource;
  scene?: RoiGroupScene;
  createdBy?: string;
  /** Supplied when updating an existing document; a new one is minted otherwise. */
  id?: string;
  createdAt?: string;
}): RoiGroupDocument {
  const now = new Date().toISOString();
  return {
    schemaVersion: ROI_GROUP_SCHEMA_VERSION,
    id: options.id ?? getRandomHexString(128),
    group: options.group,
    source: options.source,
    ...(options.scene === undefined ? {} : { scene: options.scene }),
    ...(options.createdBy === undefined
      ? {}
      : { createdBy: options.createdBy }),
    createdAt: options.createdAt ?? now,
    updatedAt: now,
  };
}

export function parseRoiGroupDocument(json: unknown): RoiGroupDocument {
  verifyObject(json);
  const schemaVersion = verifyObjectProperty(json, "schemaVersion", (v) => {
    if (typeof v !== "number" || !Number.isInteger(v)) {
      throw new Error(`Expected integer, got ${JSON.stringify(v)}`);
    }
    // Reject newer documents rather than silently dropping fields this build
    // does not understand.
    if (v > ROI_GROUP_SCHEMA_VERSION) {
      throw new Error(
        `Document schema version ${v} is newer than this viewer supports ` +
          `(${ROI_GROUP_SCHEMA_VERSION})`,
      );
    }
    return v;
  });
  return {
    schemaVersion,
    id: verifyObjectProperty(json, "id", verifyString),
    // Left as opaque JSON: `groupFromJson` is what validates its shape, at the
    // point where it is turned back into a RoiGroup.
    group: verifyObjectProperty(json, "group", (v) => verifyObject(v)),
    source: verifyObjectProperty(json, "source", (v) => {
      verifyObject(v);
      return {
        url: verifyObjectProperty(v, "url", verifyString),
        coordinateSpace: verifyOptionalObjectProperty(
          v,
          "coordinateSpace",
          (c) => c,
        ),
      };
    }),
    scene: verifyOptionalObjectProperty(json, "scene", (v) => {
      verifyObject(v);
      return {
        url: verifyOptionalObjectProperty(v, "url", verifyString),
        layerName: verifyOptionalObjectProperty(v, "layerName", verifyString),
      };
    }),
    createdBy: verifyOptionalObjectProperty(json, "createdBy", verifyString),
    createdAt: verifyObjectProperty(json, "createdAt", verifyString),
    updatedAt: verifyObjectProperty(json, "updatedAt", verifyString),
  };
}

/**
 * Summary shown in the browse list.
 *
 * Built from GCS object custom metadata so listing the bucket renders the list
 * in one request, instead of fetching every document to read its name.
 */
export interface RoiGroupSummary {
  id: string;
  name: string;
  createdBy?: string;
  sourceUrl?: string;
  updated?: string;
}

/**
 * Fields mirrored into GCS object custom metadata.  Values must be strings;
 * anything absent is simply omitted rather than written as "undefined".
 */
export function roiGroupCustomMetadata(
  doc: RoiGroupDocument,
): Record<string, string> {
  const metadata: Record<string, string> = {};
  const name = doc.group?.name;
  if (typeof name === "string") metadata.roiGroupName = name;
  if (doc.createdBy !== undefined) metadata.createdBy = doc.createdBy;
  if (doc.source?.url !== undefined) metadata.sourceUrl = doc.source.url;
  return metadata;
}
