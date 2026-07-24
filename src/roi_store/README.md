# Shared ROI group store

Saves zarr-vectors ROI groups (named tract dissections) to a Google Cloud
Storage bucket so they can be reloaded later or opened by someone else.

The bucket is **public-read**: browsing and loading are anonymous and need no
sign-in, in either deployment. A Google token is required only to **write**.

Groups continue to live in the URL hash exactly as before — this is an
additional, named artifact, not a replacement.

## Enabling it

The feature is off unless the `ROI_STORE` define is set, following the
`STATE_SERVERS` precedent in `src/datasource/state_share.ts`. With it unset,
the sign-in chip, the per-group **Save to store** action, the **From store**
checklist and **Browse saved…** are all omitted from the UI.

```jsonc
ROI_STORE = {
  "bucket": "my-roi-groups",                     // public-read GCS bucket
  "clientId": "....apps.googleusercontent.com",  // OAuth2 client (google provider)
  "endpoint": "http://localhost:9000",           // optional: local stand-in
  "scopes": ["openid", "email"],                 // optional: LOCAL DEV ONLY
  "eager": false                                 // optional: sign in at startup
}
```

## Which sign-in (`provider`)

`google` (the default) obtains a Google OAuth2 token with a storage scope and
writes straight to the GCS JSON API. It is the only provider that works in the
cross-origin isolated pyodide build, because our own redirect page can broadcast
the response past COOP.

`middleauth` reuses neuroglancer's existing CAVE/middleauth login — the same one
`state_share` uses — so no separate OAuth client is needed:

```jsonc
ROI_STORE = {
  "bucket": "my-roi-groups",
  "provider": "middleauth",
  "authServer": "https://global.daf-apis.com",
  "endpoint": "https://my-middleauth-fronted-store"
}
```

One constraint applies to `middleauth`:

- The token is a **CAVE bearer token**, not a Google one, so `endpoint` must be
  a server that accepts it — not raw `storage.googleapis.com`.

The login popup gets its response from the CAVE server's own page via
`window.opener`, which **COOP:same-origin severs**. `ng-pyodide` therefore
deliberately does **not** set COOP/COEP (the runtime is single-threaded, so
cross-origin isolation buys nothing — see `firebase.json`), and middleauth
completes there just as it does in `dist/client`.

`eager: true` signs in at viewer start rather than lazily on first save — the
"authenticate on the wasm before adding the url" path — so a bucket that denies
anonymous listing populates the picker without a later prompt.

`dist/client` — set it in the `define:` block of `rspack.config.ts`, or pass it
on the command line:

```sh
npm run build -- --define 'ROI_STORE={"bucket":"my-roi-groups","clientId":"...apps.googleusercontent.com"}'
```

`dist/pyodide` — the same flag, which `build_tools/build_pyodide.ts` now
accepts:

```sh
node ./build_tools/build_pyodide.ts --define 'ROI_STORE={"bucket":"my-roi-groups","clientId":"...apps.googleusercontent.com"}'
```

> The pyodide build previously had no `DefinePlugin` at all, so _no_ build-time
> define could reach it and every optionally-configured feature was silently
> absent from that deployment however it was configured elsewhere.

**Windows note:** `python/examples/pyodide/dev_server.py` defaults `--dir` to
`dist/pyodide` and `chdir`s into it, which locks the directory. A dev server
left running makes the pyodide build fail with `EBUSY … rmdir dist/pyodide`.
Stop it, or start it with an explicit `--dir`.

## Running it locally, without a bucket

The whole flow — sign in, save, browse, load, delete — can be exercised before
any Google Cloud setup exists. Start the stand-in store:

```sh
node ./build_tools/roi_store_dev_server.ts --port 9000 --bucket dev-roi-groups
```

It implements only the four operations this feature uses, serves CORS, persists
to `.roi-store-dev/` so saved groups survive a restart, and **does not verify
tokens**. Never expose it beyond localhost.

Then build the viewer against it:

```sh
npm run build -- --define 'ROI_STORE={"bucket":"dev-roi-groups",
  "endpoint":"http://localhost:9000","clientId":"<client-id>",
  "scopes":["openid","email"]}'
```

Two things make this work before the real setup exists:

- **`endpoint`** points the client at the stand-in instead of Google.
- **`scopes`** drops `devstorage.read_write`. That scope is sensitive and
  cannot be granted until Google has verified the OAuth client, but the
  stand-in never checks the token — so `openid email` is enough to drive the
  real sign-in path end to end. Any client ID authorised for your origin will
  do, including the in-tree `NEUROGLANCER_BRAINMAPS_CLIENT_ID`, which is valid
  for `localhost:8000` and `localhost:8080`.

A build pointed at real Google Cloud Storage must leave `scopes` unset —
without `devstorage.read_write`, every save gets a 403.

`tests/fixtures/fake_gcs_subset.ts` wraps this same server, so what the tests
pin is exactly what you develop against.

## Google Cloud setup

### OAuth client

Create a **Web application** OAuth client. The existing
`NEUROGLANCER_BRAINMAPS_CLIENT_ID` cannot be reused: it is restricted to other
origins and is not authorised for the storage scope.

- Authorised JavaScript origins: each deployment origin, plus
  `http://localhost:8080` for local testing.
- Authorised redirect URIs: `<origin>/google_oauth2_redirect.html` for each.
  That file is emitted unhashed by both builds precisely so these URLs stay
  stable.

Scopes requested: `openid`, `email`,
`https://www.googleapis.com/auth/devstorage.read_write`.

> `devstorage.read_write` is a **sensitive** scope, so the client will likely
> need Google verification before accounts outside the project can grant it.
> Owners and consent-screen test users work immediately, so this does not block
> development — but start it early if others need write access.
>
> The scope is also account-wide: it authorises writes to every bucket the
> signed-in account can reach, not just this one. There is no narrower storage
> scope. If that breadth is unacceptable, the alternative is a token-broker
> service issuing downscoped credentials — `ngauth_server/` is the in-tree
> precedent.

### Bucket

Public read, which also grants the object listing the browse dialog needs:

```sh
gcloud storage buckets add-iam-policy-binding gs://my-roi-groups \
  --member=allUsers --role=roles/storage.objectViewer
```

Write access, granted to a Google Group so collaborators are managed by group
membership rather than individual IAM bindings:

```sh
gcloud storage buckets add-iam-policy-binding gs://my-roi-groups \
  --member=group:my-lab@googlegroups.com --role=roles/storage.objectAdmin
```

`objectAdmin` rather than `objectCreator` because the browse dialog offers
delete. Use `objectCreator` for append-only.

CORS — easy to forget, and every write fails without it:

```sh
cat > cors.json <<'EOF'
[{
  "origin": ["https://my-site.web.app", "http://localhost:8080"],
  "method": ["GET", "POST", "DELETE"],
  "responseHeader": ["Content-Type", "Authorization"],
  "maxAgeSeconds": 3600
}]
EOF
gcloud storage buckets update gs://my-roi-groups --cors-file=cors.json
```

## Deploying

`.firebaserc` already maps the `ng-pyodide` hosting target to a site named
`ng-pyodide` under project `em-270621`. That site does not exist yet — create it
once, then deploy:

```sh
# One-time: create the hosting site the target maps to.
firebase hosting:sites:create ng-pyodide --project em-270621

# Stop any dev_server holding dist/pyodide as its CWD first (it makes the
# build's clean step fail with EBUSY on Windows).

# Build the pyodide bundle with the ROI store define and the zarr-vectors
# packages bundled in, then publish.
NEUROGLANCER_PYODIDE_PACKAGES="/path/to/zarr-vectors-py/zarr_vectors,/path/to/zarr-vectors-tools/zarr_vectors_tools" \
  node build_tools/build_pyodide.ts \
    --define 'ROI_STORE={"bucket":"hip_ct_zarr_vector_03987646472fethdsvdvdfg","provider":"middleauth","authServer":"https://global.daf-apis.com"}'
firebase deploy --only hosting:ng-pyodide --project em-270621
```

The `dist/client` build (target `ng-zarr-vectors`) already carries the ROI store
config from `rspack.config.ts`, so it deploys with `npm run build && firebase
deploy --only hosting:ng-zarr-vectors --project em-270621`.

Note the middleauth constraints above: browse/load work anonymously on both
targets; save works only in `dist/client` and only once `endpoint` points at a
middleauth-fronted store.

`firebase.json` gives `ng-pyodide` a `** → /index.html` rewrite so that
share/continue links — which live under the `/v/pyodide/` path the app assigns
itself at boot — load the app shell instead of 404ing. Those links carry the
full viewer state including the ROI filter groups (positions, not the passing-ID
set) in the `#!{…}` hash; see
`python/examples/pyodide/README.md` → "Sharing a filter session via URL".

## Layout

| File                | Role                                                 |
| ------------------- | ---------------------------------------------------- |
| `config.ts`         | The `ROI_STORE` define and `roiStoreEnabled`         |
| `credentials.ts`    | Google sign-in, token caching, expiry                |
| `gcs_client.ts`     | List / read / save / delete against the GCS JSON API |
| `schema.ts`         | The `groups/<id>.json` document format               |
| `browse_dialog.ts`  | The shared-library browser (all datasets)            |
| `sign_in_widget.ts` | Top-row identity chip                                |

Two pieces live with the zarr-vectors datasource instead, because they depend
on `RoiFilterState`:

| File                                               | Role                                        |
| -------------------------------------------------- | ------------------------------------------- |
| `../datasource/zarr-vectors/store_group_picker.ts` | "From store" checklist in the Filter tab    |
| `../datasource/zarr-vectors/store_provenance.ts`   | Which document each live group is backed by |

## In the Filter tab

- **Save to store**, per group. Prompts for sign-in the first time. Re-saving
  updates the same document rather than adding a copy — the mapping lives in
  `store_provenance.ts`, keyed on the layer's `RoiFilterState` so it survives
  the side panel being closed and reopened.
- **From store**, a checklist of the dissections saved _for this dataset_.
  Ticking loads and shows it; unticking hides it. It refreshes on the
  `roiGroupStoreChanged` signal, so a save anywhere updates every open picker,
  and reads "none found" until the first group is saved. Unticking never
  deletes: hiding must not discard edits made since loading.
- **Browse saved…**, the full-library dialog, which also shows entries from
  other datasets with a mismatch warning.

## Security notes

- The access token is cached in `localStorage`; anything that can run script on
  the origin can read it. This matches the existing `middleauth` provider.
- The OAuth response is delivered over a `BroadcastChannel` whose name embeds
  the attempt's random `state`, not a well-known name. Any same-origin context
  can open a channel it can _name_, and there is no enumeration API — so
  scoping by a secret keeps the token away from unrelated same-origin code,
  which matters in the pyodide build where user-supplied Python runs in the
  same origin.
- `createdBy` comes from an unverified `id_token` payload and is **provenance
  only**. It is displayed, never used for an access decision — the bucket's IAM
  policy is the authority on who may write.
