/**
 * @license
 * Copyright 2024 Google Inc.
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
 * Build script for the Pyodide (browser-only, serverless) neuroglancer deployment.
 *
 * Produces dist/pyodide/ with:
 *   - main_pyodide.[hash].js       - Main neuroglancer + bootstrap bundle
 *   - pyodide_sw.js                - Service Worker (no hash for stable URL)
 *   - pyodide_worker.[hash].js     - Pyodide Web Worker
 *   - neuroglancer_pyodide.zip     - Bundled Python package
 *   - user_script.py               - Default user script (override with ?script=)
 *   - index.html                   - Standalone HTML page
 *   - [chunk files]                - Code-split chunks
 *
 * Usage:
 *   node build_tools/build_pyodide.ts [--watch] [--mode=development|production]
 */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import type { Configuration, Stats } from "@rspack/core";
import { rspack, HtmlRspackPlugin } from "@rspack/core";
import packageJson from "../package.json" with { type: "json" };

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const outDir = path.resolve(repoRoot, "dist", "pyodide");
const args = process.argv.slice(2);
const watchMode = args.includes("--watch");
const mode = args.includes("--mode=development") ? "development" : "production";

// ---------------------------------------------------------------------------
// Shared rspack module rules
// ---------------------------------------------------------------------------

const moduleRules: Configuration["module"] = {
  rules: [
    {
      test: /\.tsx?$/,
      loader: "builtin:swc-loader",
      options: {
        jsc: {
          parser: { syntax: "typescript", decorators: true },
        },
        env: { targets: (packageJson as any).browserslist },
      },
      type: "javascript/auto",
    },
    {
      test: /\.wasm$/,
      generator: { filename: "[name].[contenthash][ext]" },
    },
    {
      resourceQuery: /raw/,
      type: "asset/source",
    },
    {
      test: /(bossauth|google_oauth2_redirect)\.html$/,
      type: "asset/resource",
      generator: { filename: "[name][ext]" },
    },
  ],
};

// ---------------------------------------------------------------------------
// Config 1: Main web bundle (main_pyodide.ts → web target)
// ---------------------------------------------------------------------------

const mainConfig: Configuration = {
  mode,
  context: repoRoot,
  entry: {
    main_pyodide: "./src/main_pyodide.ts",
  },
  target: ["web", "browserslist"],
  module: moduleRules,
  output: {
    path: outDir,
    filename: "[name].[chunkhash].js",
    chunkFilename: "[name].[contenthash].js",
    // Absolute public path so assets work after history.replaceState('/v/pyodide/')
    publicPath: "/",
    asyncChunks: true,
    clean: true,
  },
  optimization: {
    splitChunks: { chunks: "all" },
  },
  resolve: {
    // Enable the python datasource (same condition as build-python)
    conditionNames: ["...", "neuroglancer/python"],
  },
  devtool: mode === "development" ? "source-map" : false,
  performance: {
    maxAssetSize: 10 * 1024 * 1024,
    maxEntrypointSize: 10 * 1024 * 1024,
  },
  experiments: { css: true },
  plugins: [
    new HtmlRspackPlugin({
      template: path.resolve(repoRoot, "python", "examples", "pyodide", "index.html"),
      filename: "index.html",
      chunks: ["main_pyodide"],
      scriptLoading: "module",
    }),
  ],
};

// ---------------------------------------------------------------------------
// Config 2: Web Workers (Service Worker + Pyodide Worker → webworker target)
// ---------------------------------------------------------------------------

const workersConfig: Configuration = {
  mode,
  context: repoRoot,
  entry: {
    // Stable filename for Service Worker registration URL
    pyodide_sw: "./src/python_integration/pyodide_service_worker.ts",
    // Content-hashed filename for the Pyodide Worker
    pyodide_worker: "./src/python_integration/pyodide_worker.ts",
  },
  target: "webworker",
  module: moduleRules,
  output: {
    path: outDir,
    filename: (pathData) => {
      if (pathData.chunk?.name === "pyodide_sw") {
        return "pyodide_sw.js";
      }
      return "[name].[chunkhash].js";
    },
    // No code splitting for workers
  },
  resolve: {},
  devtool: mode === "development" ? "source-map" : false,
};

// ---------------------------------------------------------------------------
// Build helpers
// ---------------------------------------------------------------------------

function runCompiler(configs: Configuration[]): Promise<Stats[]> {
  return new Promise((resolve, reject) => {
    const compiler = rspack(configs as any);
    const handler = (err: Error | null, stats: any) => {
      if (err) {
        reject(err);
        return;
      }
      // An array of configs yields a MultiCompiler, whose callback receives a
      // MultiStats -- an object with a `.stats` array, not an array itself.
      const multiStats = (stats as { stats?: Stats[] } | null)?.stats;
      const statsList: Stats[] = Array.isArray(multiStats) ? multiStats : [stats];
      let hasErrors = false;
      for (const s of statsList) {
        const info = s.toJson();
        if (s.hasErrors()) {
          for (const e of info.errors ?? []) {
            console.error((e as any).message ?? e);
          }
          hasErrors = true;
        }
        if (s.hasWarnings()) {
          for (const w of info.warnings ?? []) {
            console.warn((w as any).message ?? w);
          }
        }
      }
      if (hasErrors) {
        reject(new Error("Build failed with errors"));
      } else {
        resolve(statsList);
      }
    };
    if (watchMode) {
      compiler.watch({}, handler);
    } else {
      compiler.run(handler);
    }
  });
}

// Paths and the zip name are passed as argv rather than interpolated into the
// script, so Windows separators are never parsed as Python escape sequences.
const ZIP_PYTHON_SCRIPT = `
import pathlib, sys, zipfile

pkg = pathlib.Path(sys.argv[1]) / "neuroglancer"
with zipfile.ZipFile(sys.argv[2], "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(pkg.rglob("*.py")):
        arcname = f.relative_to(pkg).as_posix()
        zf.write(f, "neuroglancer/" + arcname)
`;

function resolvePythonInterpreter(): string {
  for (const candidate of ["python3", "python"]) {
    try {
      execFileSync(candidate, ["--version"], { stdio: "ignore" });
      return candidate;
    } catch {
      // Try the next candidate.
    }
  }
  throw new Error(
    "No Python interpreter found (tried python3, python). Python is required " +
      "to package the neuroglancer module for pyodide.",
  );
}

function createPythonZip() {
  console.log("Creating neuroglancer_pyodide.zip…");
  const pythonDir = path.resolve(repoRoot, "python");
  const zipPath = path.resolve(outDir, "neuroglancer_pyodide.zip");

  // Remove existing zip if present
  if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);

  execFileSync(resolvePythonInterpreter(), ["-c", ZIP_PYTHON_SCRIPT, pythonDir, zipPath], {
    stdio: "inherit",
  });
  console.log("Created", zipPath);
}

// Default script fetched by main_pyodide.ts when ?script= is not supplied.
// Keep this name in sync with DEFAULT_USER_SCRIPT_PATH there.
const USER_SCRIPT_NAME = "user_script.py";

function copyExampleScript() {
  const src = path.resolve(repoRoot, "python", "examples", "pyodide", USER_SCRIPT_NAME);
  const dest = path.resolve(outDir, USER_SCRIPT_NAME);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, dest);
    console.log("Copied", USER_SCRIPT_NAME);
  } else {
    console.warn(`Warning: ${USER_SCRIPT_NAME} not found at`, src);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log(`Building Pyodide bundle (mode=${mode}, watch=${watchMode})…`);
fs.mkdirSync(outDir, { recursive: true });

// Run main config first (it has clean:true, which wipes outDir).
// Workers config must run after so its output isn't deleted.
await runCompiler([mainConfig]);
await runCompiler([workersConfig]);

if (!watchMode) {
  createPythonZip();
  copyExampleScript();
  console.log(`\nBuild complete! Output: ${outDir}`);
  console.log("\nTo test locally:");
  console.log("  python python/examples/pyodide/dev_server.py");
  console.log("  Open http://localhost:8080/");
}
