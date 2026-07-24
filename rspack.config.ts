import path from "node:path";
import { defineConfig } from "@rspack/cli";
import { HtmlRspackPlugin, ProgressPlugin } from "@rspack/core";
import { normalizeConfigurationWithDefine } from "./build_tools/rspack/configuration_with_define.js";
import packageJson from "./package.json" with { type: "json" };

export default defineConfig((env, args) => {
  const mode = args.mode === "production" ? "production" : "development";
  const config = {
    mode,
    context: import.meta.dirname,
    entry: {
      main: "./src/main.bundle.js",
    },
    performance: {
      // Avoid unhelpful warnings due to large bundles.
      maxAssetSize: 3 * 1024 * 1024,
      maxEntrypointSize: 3 * 1024 * 1024,
    },
    optimization: {
      splitChunks: {
        chunks: "all",
      },
    },
    devtool: "source-map",
    module: {
      rules: [
        // Needed to support Neuroglancer TypeScript sources.
        {
          test: /\.tsx?$/,
          loader: "builtin:swc-loader",
          options: {
            jsc: {
              parser: {
                syntax: "typescript",
                decorators: true,
              },
            },
            env: {
              targets: packageJson.browserslist,
            },
          },
          type: "javascript/auto",
        },
        {
          test: /\.wasm$/,
          generator: {
            filename: "[name].[contenthash][ext]",
          },
        },
        // Needed for .svg?raw imports used for embedding icons.
        {
          resourceQuery: /raw/,
          type: "asset/source",
        },
        // Needed for .html assets used for auth redirect pages for the
        // brainmaps and bossDB data sources.
        {
          test: /(bossauth|google_oauth2_redirect)\.html$/,
          type: "asset/resource",
          generator: {
            // Filename must be preserved since exact redirect URLs must be allowlisted.
            filename: "[name][ext]",
          },
        },
      ],
    },
    devServer: {
      client: {
        overlay: {
          // Prevent intrusive notification spam.
          runtimeErrors: false,
        },
      },
      hot: false,
    },
    plugins: [
      new ProgressPlugin(),
      new HtmlRspackPlugin({
        title: "neuroglancer",
      }),
    ],
    output: {
      path: path.resolve(import.meta.dirname, "dist", "client"),
      filename: "[name].[chunkhash].js",
      chunkFilename: "[name].[contenthash].js",
      asyncChunks: true,
      clean: true,
    },
    target: ["web", "browserslist"],
    experiments: {
      css: true,
    },
    // Additional defines, to be added via `DefinePlugin`.  This is not a
    // standard webpack configuration property, but is handled specially by
    // `normalizeConfigurationWithDefine`.
    define: {
      // This is the default client ID used for the hosted neuroglancer.
      // In addition to the hosted neuroglancer origin, it is valid for
      // the origins:
      //
      //   localhost:8000
      //   127.0.0.1:8000
      //   localhost:8080
      //   127.0.0.1:8080
      //
      // To deploy to a different origin, you will need to generate your
      // own client ID from on the Google Developer Console and substitute
      // it in.
      NEUROGLANCER_BRAINMAPS_CLIENT_ID: JSON.stringify(
        "639403125587-4k5hgdfumtrvur8v48e3pr7oo91d765k.apps.googleusercontent.com",
      ),

      // Shared store for saved zarr-vectors ROI groups.  When this is not
      // defined the save/browse UI and the sign-in chip are omitted entirely.
      //
      // `bucket` is a public-read GCS bucket holding `groups/*.json`.  It now
      // grants anonymous object READ and LIST (allUsers roles/storage.objectViewer
      // + CORS, enabled by the bucket admin), so BROWSE and LOAD enumerate groups
      // with no sign-in -- matching the `gs://` read sources in
      // `python/examples/pyodide/user_script.py`, which rely on that same list
      // permission for complete cross-chunk link discovery.
      //
      // Uses the `google` provider (the default): a Google OAuth2 token with the
      // `devstorage.read_write` scope, written straight to the GCS JSON API. This
      // is the "coarse-select-then-save-to-my-store" flow -- read is anonymous,
      // but the sign-in chip is always offered, and clicking "Save to store" (or
      // "Save to GCS" in the Export tab) prompts Google sign-in and then WRITES
      // straight to `bucket`. The write succeeds iff the signed-in account holds
      // an object-write role on `bucket` (otherwise GCS returns 403, surfaced as
      // "Could not save: ..."). No middleauth-fronted endpoint is needed, and it
      // works under COOP too -- the redirect page broadcasts the token past
      // cross-origin isolation, so the pyodide build can save as well.
      //
      // The `middleauth` provider was tried here (commit 03a1cad8) but its CAVE
      // bearer token is rejected by raw GCS with 401, so it cannot write to this
      // public bucket without a middleauth-fronted upload proxy -- which is why
      // saving disappeared. The `google` provider restores it.
      //
      // TWO deployment prerequisites, neither expressible in this file:
      //   1. `clientId` must be an OAuth2 client authorised for THIS deploy's
      //      origin AND the `devstorage.read_write` scope (a sensitive scope
      //      needing Google verification). The brainmaps client ID above is
      //      neither, so it cannot be reused. Create one in the Google Cloud
      //      Console and set it below, or inject it at build time via
      //      NEUROGLANCER_ROI_STORE_CLIENT_ID (client IDs are public, not secret).
      //   2. Grant each editor's Google account an object-write role on the
      //      bucket (e.g. roles/storage.objectAdmin); read stays anonymous.
      // With the REPLACE_ME placeholder below the store is still configured, so
      // anonymous BROWSE/LOAD work and the sign-in chip shows; only SAVE fails
      // (the fake OAuth client cannot complete sign-in) until a real clientId is
      // supplied (see src/roi_store/config.ts).
      ROI_STORE: (() => {
        const clientId =
          process.env.NEUROGLANCER_ROI_STORE_CLIENT_ID ??
          "REPLACE_ME.apps.googleusercontent.com";
        return JSON.stringify({
          bucket: "hip_ct_zarr_vector_03987646472fethdsvdvdfg",
          clientId,
        });
      })(),

      // NEUROGLANCER_CREDIT_LINK: JSON.stringify({url: '...', text: '...'}),
      // NEUROGLANCER_DEFAULT_STATE_FRAGMENT: JSON.stringify('gs://bucket/state.json'),
      // NEUROGLANCER_SHOW_LAYER_BAR_EXTRA_BUTTONS: true,
      // NEUROGLANCER_SHOW_OBJECT_SELECTION_TOOLTIP: true

      // NEUROGLANCER_GOOGLE_TAG_MANAGER: JSON.stringify('GTM-XXXXXX'),
    },
    watchOptions: {
      ignored: /node_modules/,
    },
  };
  return env.NEUROGLANCER_CLI
    ? config
    : normalizeConfigurationWithDefine(config);
});
