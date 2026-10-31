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
 * @file How Neuroglancer starts its workers, and how an embedding
 * application can start them itself.
 *
 * A data source defined outside Neuroglancer (in its own package) has code
 * that must run in the chunk worker, and possibly in the async-computation
 * workers.  The application then supplies its own worker scripts, each
 * importing Neuroglancer's worker contents and the extra modules:
 *
 *     // my_chunk_worker.js
 *     import "my-datasource/backend";
 *     import "neuroglancer/worker/chunk";
 *
 *     // main thread, before creating a viewer
 *     setChunkWorkerFactory(
 *       () => new Worker(new URL("./my_chunk_worker.js", import.meta.url),
 *                        { type: "module" }));
 *
 * Neuroglancer's worker contents must come last: they report the worker
 * ready.  Because the application's bundler builds the script, the extra
 * modules share Neuroglancer's module instances (its RPC and shared-object
 * registries) rather than getting copies.
 *
 * The chunk worker starts async-computation workers itself, so a factory
 * for those must be set in the chunk worker script as well as on the main
 * thread.
 */

export type WorkerFactory = () => Worker;

let chunkWorkerFactory: WorkerFactory | undefined;
let asyncComputationWorkerFactory: WorkerFactory | undefined;

/** Sets how the chunk worker is started; `undefined` restores the default. */
export function setChunkWorkerFactory(factory: WorkerFactory | undefined) {
  chunkWorkerFactory = factory;
}

/** Sets how async-computation workers are started; `undefined` restores the default. */
export function setAsyncComputationWorkerFactory(
  factory: WorkerFactory | undefined,
) {
  asyncComputationWorkerFactory = factory;
}

export function getChunkWorkerFactory(): WorkerFactory | undefined {
  return chunkWorkerFactory;
}

export function getAsyncComputationWorkerFactory(): WorkerFactory | undefined {
  return asyncComputationWorkerFactory;
}
