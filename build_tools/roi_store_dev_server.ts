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
 * Local stand-in for the slice of the Google Cloud Storage JSON API that the
 * shared ROI group store uses, so the save / browse / load flow can be
 * exercised without a real bucket.
 *
 * NOT a GCS emulator and NOT a conformance test. It implements exactly four
 * operations and 404s loudly on anything else, so a client change that reaches
 * for a new endpoint fails here rather than only in production.
 *
 * The same module backs `tests/fixtures/fake_gcs_subset.ts`, so what the tests
 * pin and what you develop against cannot drift apart.
 *
 * Usage:
 *   node build_tools/roi_store_dev_server.ts [--port 9000] [--bucket dev-roi-groups]
 *                                            [--dir .roi-store-dev]
 *
 * Then build the viewer pointed at it:
 *   npm run build -- --define 'ROI_STORE={"bucket":"dev-roi-groups",
 *     "endpoint":"http://localhost:9000","clientId":"...","scopes":["openid","email"]}'
 *
 * Tokens are NOT verified: any Authorization header is accepted. That is the
 * point — it lets the client's auth path run end to end before a real OAuth
 * client exists. Never expose this server beyond localhost.
 */

import fs from "node:fs";
import http from "node:http";
import type { AddressInfo } from "node:net";
import path from "node:path";

export interface StoredObject {
  content: string;
  metadata: Record<string, string>;
  updated: string;
}

export interface RoiStoreServer {
  url: string;
  bucket: string;
  objects: Map<string, StoredObject>;
  /** Seeds an object directly, bypassing the upload path. */
  putRaw(
    name: string,
    content: string,
    metadata?: Record<string, string>,
  ): void;
  /** When set, writes without this bearer token are rejected. */
  expectToken(token: string | undefined): void;
  /** Rejects the next N authenticated requests with this status. */
  failNextAuth(count: number, status?: number): void;
  /**
   * Reject listings that carry no bearer token, reproducing a bucket that
   * serves objects publicly but withholds `storage.objects.list`.
   */
  rejectAnonymousList(reject: boolean): void;
  /** Writes seen, for asserting on retry behaviour. */
  authRequests: { method: string; token: string | undefined }[];
  close(): Promise<void>;
}

export interface RoiStoreServerOptions {
  bucket?: string;
  /** 0 picks a free port, which is what the tests want. */
  port?: number;
  /** Page size for list responses; small values exercise pagination. */
  pageSize?: number;
  /** Directory to persist objects in. Memory-only when omitted. */
  dataDir?: string;
  /** Log each request. */
  verbose?: boolean;
}

function parseMultipart(body: string, contentType: string) {
  const match = /boundary=(.*)$/.exec(contentType);
  if (match === null) throw new Error("missing multipart boundary");
  const boundary = `--${match[1].trim()}`;
  const parts = body
    .split(boundary)
    .map((p) => p.trim())
    .filter((p) => p.length > 0 && p !== "--");
  if (parts.length !== 2) {
    throw new Error(`expected 2 multipart parts, got ${parts.length}`);
  }
  // Each part is headers, a blank line, then the payload.
  const payload = (part: string) => {
    const idx = part.indexOf("\r\n\r\n");
    if (idx < 0) throw new Error("malformed multipart part");
    return part.slice(idx + 4);
  };
  return {
    metadata: JSON.parse(payload(parts[0])),
    content: payload(parts[1]),
  };
}

/** Object names contain a slash, so they are stored flat with it escaped. */
function diskName(objectName: string): string {
  return `${encodeURIComponent(objectName)}.store.json`;
}

export async function startRoiStoreServer(
  options: RoiStoreServerOptions = {},
): Promise<RoiStoreServer> {
  const {
    bucket = "dev-roi-groups",
    port = 0,
    pageSize = 1000,
    dataDir,
    verbose = false,
  } = options;

  const objects = new Map<string, StoredObject>();
  const authRequests: { method: string; token: string | undefined }[] = [];
  let expectedToken: string | undefined;
  let failAuthCount = 0;
  let failAuthStatus = 401;
  let rejectAnonymousListing = false;

  if (dataDir !== undefined) {
    fs.mkdirSync(dataDir, { recursive: true });
    for (const file of fs.readdirSync(dataDir)) {
      if (!file.endsWith(".store.json")) continue;
      try {
        const record = JSON.parse(
          fs.readFileSync(path.join(dataDir, file), "utf8"),
        );
        objects.set(record.name, {
          content: record.content,
          metadata: record.metadata ?? {},
          updated: record.updated,
        });
      } catch {
        // A corrupt file should not stop the server starting.
      }
    }
  }

  function persist(name: string, obj: StoredObject | undefined) {
    if (dataDir === undefined) return;
    const file = path.join(dataDir, diskName(name));
    if (obj === undefined) {
      fs.rmSync(file, { force: true });
    } else {
      fs.writeFileSync(file, JSON.stringify({ name, ...obj }, null, 2));
    }
  }

  const objectPrefix = `/storage/v1/b/${encodeURIComponent(bucket)}/o`;
  const uploadPrefix = `/upload/storage/v1/b/${encodeURIComponent(bucket)}/o`;

  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8");
      const url = new URL(req.url ?? "/", "http://localhost");
      if (verbose) console.log(`  ${req.method} ${url.pathname}${url.search}`);

      // The viewer is served from a different origin, so every request is
      // cross-origin and the writes are preflighted.
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
      res.setHeader(
        "Access-Control-Allow-Headers",
        "Authorization,Content-Type",
      );
      res.setHeader("Access-Control-Max-Age", "3600");
      if (req.method === "OPTIONS") {
        res.writeHead(204);
        return res.end();
      }

      const send = (status: number, payload?: unknown) => {
        res.writeHead(status, { "Content-Type": "application/json" });
        res.end(payload === undefined ? "" : JSON.stringify(payload));
      };

      const authHeader = req.headers.authorization;
      const token = authHeader?.startsWith("Bearer ")
        ? authHeader.slice("Bearer ".length)
        : undefined;
      const isWrite = req.method === "POST" || req.method === "DELETE";
      if (isWrite) {
        authRequests.push({ method: req.method!, token });
        if (failAuthCount > 0) {
          --failAuthCount;
          return send(failAuthStatus, { error: { message: "forced failure" } });
        }
        if (expectedToken !== undefined && token !== expectedToken) {
          return send(401, { error: { message: "invalid token" } });
        }
      }

      // Upload: POST /upload/storage/v1/b/<bucket>/o?uploadType=multipart
      if (req.method === "POST" && url.pathname === uploadPrefix) {
        if (url.searchParams.get("uploadType") !== "multipart") {
          return send(400, { error: { message: "expected multipart upload" } });
        }
        try {
          const { metadata, content } = parseMultipart(
            body,
            req.headers["content-type"] ?? "",
          );
          if (typeof metadata?.name !== "string") {
            return send(400, { error: { message: "missing object name" } });
          }
          const stored: StoredObject = {
            content,
            metadata: metadata.metadata ?? {},
            updated: new Date().toISOString(),
          };
          objects.set(metadata.name, stored);
          persist(metadata.name, stored);
          return send(200, { name: metadata.name });
        } catch (e) {
          return send(400, { error: { message: (e as Error).message } });
        }
      }

      // List: GET /storage/v1/b/<bucket>/o?prefix=...
      if (req.method === "GET" && url.pathname === objectPrefix) {
        if (rejectAnonymousListing && token === undefined) {
          return send(401, {
            error: {
              message:
                "Anonymous caller does not have storage.objects.list access",
            },
          });
        }
        const prefix = url.searchParams.get("prefix") ?? "";
        const all = [...objects.entries()]
          .filter(([name]) => name.startsWith(prefix))
          .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
        const start = Number(url.searchParams.get("pageToken") ?? "0");
        const page = all.slice(start, start + pageSize);
        const nextIndex = start + pageSize;
        return send(200, {
          items: page.map(([name, obj]) => ({
            name,
            updated: obj.updated,
            metadata: obj.metadata,
          })),
          ...(nextIndex < all.length
            ? { nextPageToken: String(nextIndex) }
            : {}),
        });
      }

      // Read / delete: /storage/v1/b/<bucket>/o/<urlencoded name>
      if (url.pathname.startsWith(`${objectPrefix}/`)) {
        const name = decodeURIComponent(
          url.pathname.slice(objectPrefix.length + 1),
        );
        const obj = objects.get(name);
        if (req.method === "GET") {
          if (url.searchParams.get("alt") !== "media") {
            return send(400, { error: { message: "expected alt=media" } });
          }
          if (obj === undefined) {
            return send(404, { error: { message: "no such object" } });
          }
          res.writeHead(200, { "Content-Type": "application/json" });
          return res.end(obj.content);
        }
        if (req.method === "DELETE") {
          if (obj === undefined) {
            return send(404, { error: { message: "no such object" } });
          }
          objects.delete(name);
          persist(name, undefined);
          return send(204);
        }
      }

      send(404, {
        error: { message: `unsupported: ${req.method} ${url.pathname}` },
      });
    });
  });

  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );
  const actualPort = (server.address() as AddressInfo).port;

  return {
    url: `http://127.0.0.1:${actualPort}`,
    bucket,
    objects,
    authRequests,
    putRaw(name, content, metadata = {}) {
      const stored = {
        content,
        metadata,
        updated: new Date().toISOString(),
      };
      objects.set(name, stored);
      persist(name, stored);
    },
    expectToken(t) {
      expectedToken = t;
    },
    failNextAuth(count, status = 401) {
      failAuthCount = count;
      failAuthStatus = status;
    },
    rejectAnonymousList(reject) {
      rejectAnonymousListing = reject;
    },
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

function parseArg(args: string[], name: string): string | undefined {
  const i = args.indexOf(`--${name}`);
  if (i >= 0) {
    const value = args[i + 1];
    if (value === undefined) throw new Error(`--${name} requires a value`);
    return value;
  }
  const inline = args.find((a) => a.startsWith(`--${name}=`));
  return inline?.substring(name.length + 3);
}

async function main() {
  const args = process.argv.slice(2);
  const bucket = parseArg(args, "bucket") ?? "dev-roi-groups";
  const port = Number(parseArg(args, "port") ?? "9000");
  const dataDir = parseArg(args, "dir") ?? ".roi-store-dev";

  const server = await startRoiStoreServer({
    bucket,
    port,
    dataDir,
    verbose: true,
  });
  console.log(`ROI group store stand-in listening on ${server.url}`);
  console.log(`  bucket : ${bucket}`);
  console.log(`  data   : ${path.resolve(dataDir)}`);
  console.log(`  objects: ${server.objects.size} loaded`);
  console.log("");
  console.log("Build the viewer against it with:");
  console.log(
    `  npm run build -- --define 'ROI_STORE={"bucket":"${bucket}",` +
      `"endpoint":"${server.url}","clientId":"<your-oauth-client-id>",` +
      `"scopes":["openid","email"]}'`,
  );
  console.log("");
  console.log("Tokens are NOT verified. Do not expose this beyond localhost.");
}

// Only run the CLI when invoked directly, not when imported by the tests.
if (process.argv[1]?.includes("roi_store_dev_server")) {
  await main();
}
