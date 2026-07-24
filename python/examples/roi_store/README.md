# Local ROI-group store CLI

Push ROI groups you drew in Neuroglancer into the shared GCS store from your own
machine, using your existing `gcloud` login — for when you can upload/delete
objects but **cannot change bucket CORS or config**.

## Why not just "Save to store" in the browser?

The in-browser save is a cross-origin `POST` to the GCS JSON API. Browsers only
allow that if the bucket carries a write-CORS rule for the app origin. If you
can't edit bucket CORS, that save is blocked in the browser — but **CORS is a
browser rule and does not apply to `gcloud`**, so a command-line upload works
with nothing more than object-write permission.

Reads are unaffected: the viewer's **Browse saved…** lists and loads groups
anonymously, so a group you push here shows up in the browser with no sign-in.

## One-time

```bash
gcloud auth login          # if you aren't already logged in
```

No `application-default login`, no service-account key, no `pip install` — the
tool shells out to `gcloud storage`.

## Workflow

1. In Neuroglancer, run your coarse ROI selection to build one or more groups in
   the **Filter** tab.
2. Open the viewer's JSON state (the `{}` button, top toolbar), copy it, and save
   it as `state.json`.
3. Push the groups:

   ```bash
   python roi_store_cli.py push --state state.json
   ```

   Every ROI group in every layer is uploaded to `gs://<bucket>/groups/<id>.json`
   in the exact document format the browser writes, tagged with your `gcloud`
   account as author. Restrict to one layer with `--layer <name>`.
4. Back in Neuroglancer: **Browse saved…** → your group is listed → load it.

## Other commands

```bash
python roi_store_cli.py list                      # group ids in the store
python roi_store_cli.py pull --id <id> --out g.json
python roi_store_cli.py rm   --id <id>
```

`--bucket` defaults to the demo bucket; pass `--bucket <name>` for your own.

## What it mirrors

Kept in step with `src/roi_store/schema.ts` and `src/roi_store/gcs_client.ts`:
`groups/<id>.json` naming, 128-bit hex ids, and the `roiGroupName` / `createdBy`
/ `sourceUrl` custom metadata the browse list reads. A group's JSON in the layer
state is byte-identical to a document's `group` field, which is why this needs no
changes to Neuroglancer itself.
