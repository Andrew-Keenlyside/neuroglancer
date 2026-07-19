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
 * @file In-process stand-in for the slice of the GCS JSON API that the ROI
 * group store uses.
 *
 * A thin wrapper over `build_tools/roi_store_dev_server.ts` — the same server
 * developers run locally — so what these tests pin is exactly what the feature
 * is developed against. `fake_gcs_server.ts` wraps a real emulator, but that
 * needs a Go toolchain to build; this needs nothing, so the round-trip stays
 * runnable everywhere.
 *
 * What it proves is this client's behaviour — object naming, the multipart
 * upload carrying custom metadata, list pagination, the 401 retry — against a
 * documented reading of the API. That the real service agrees is confirmed
 * separately against a live bucket.
 */

import {
  startRoiStoreServer,
  type RoiStoreServer,
  type StoredObject,
} from "../../build_tools/roi_store_dev_server.js";

export type { StoredObject };
export type FakeGcsSubset = RoiStoreServer;

export async function startFakeGcsSubset(
  bucket = "roi-groups-test",
  pageSize = 1000,
): Promise<FakeGcsSubset> {
  // Memory-only and on an ephemeral port: tests must not touch the developer's
  // dev-server data directory or collide with a running one.
  return startRoiStoreServer({ bucket, pageSize, port: 0 });
}
