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
  "clientId": "....apps.googleusercontent.com",  // OAuth2 client
  "endpoint": "http://localhost:9000",           // optional: local stand-in
  "scopes": ["openid", "email"]                  // optional: LOCAL DEV ONLY
}
```

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

## Firebase hosting

`firebase.json` defines an `ng-pyodide` target, but `.firebaserc` has no
mapping for it, so `firebase deploy --only hosting:ng-pyodide` fails. Create the
site and map it once:

```sh
firebase target:apply hosting ng-pyodide <site-name>
```

then commit the resulting `.firebaserc` change.

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
