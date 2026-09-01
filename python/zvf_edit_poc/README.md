# ZVF skeleton split — proof of concept

Takes one skeleton object out of the MICrONS minnie65 store, writes it as its
own dataset, and cuts it into two independent objects by breaking a single edge.

```
STORE=/hdd/ZV_MICRONS_Conversion/minnie65_v1822/skeletons.zv
PY=/home/andrew/scripts/zarr-vectors-py/.venv/bin/python

$PY split_skeleton.py --store $STORE --object 281 --out /tmp/zv_poc
```

Writes `/tmp/zv_poc/single_axon.zv` (one object) and `/tmp/zv_poc/split_axon.zv`
(two), plus `report.json`. Options: `--edge-index N`, `--near X,Y,Z`, or the
default, which picks the most balanced cut; `--absolute` adds the store's
`coordinate_offset` so the output lands in the same frame as the sibling
synapse/mesh stores.

Measured on object 281 (dense index; segment id 864691135305383591):

| | vertices | edges | objects |
|---|---|---|---|
| extracted | 1075 | 1074 | 1 |
| after cut | 1075 | 1073 | 2 (509 + 566) |

The cut edge was #77, 8833 nm long, midpoint `[503884, 109416, 346660]`.

## Viewing it

```
python serve_store.py /tmp/zv_poc 9077      # CORS-enabled static server
npm run dev-server                          # in the repo root, serves :8081
node screenshot.mjs 9077 split_axon.zv 1,2 480388,150172,333700 300000 out.png
```

Or open the viewer and add a segmentation layer with source
`http://127.0.0.1:9077/split_axon.zv/|zarr-vectors:`, then select segments 1
and 2 — they render as two independently coloured objects.

## Four things that cost real time, recorded so they don't again

1. **Object ids must be 1-based.** Neuroglancer hides segment 0 by default
   (`hideSegmentZero`), so an object written as id 0 loads, reports correct
   bounds, and renders nothing.
2. **The store must declare its axis unit.** `write_graph` leaves NGFF axes
   without a `unit`, which makes the store dimensionless; the viewer then places
   the geometry at ~1e14 nm and the layer is silently a billion times too far
   away. `stamp_axis_units()` sets `nanometer` to match the source store.
3. **`write_graph` is the wrong writer for this convention.** It records nearly
   every edge as an explicit link while still declaring
   `implicit_sequential_with_branches`, so a conforming reader adds the implicit
   edges on top: 1074 edges in, 1992 back, 105 spurious, the longest 53 um
   against a true median of 1.8 um — a visible hairball. The skeleton-native
   builder (`init_skeleton_store` / `write_skeleton_chunk`, which runs
   `decompose_tree_to_paths`) round-trips exactly: 1074 in, 1074 out, zero
   spurious.
4. **`read_graph` cannot be used to extract one object.** It refuses an
   `object_ids` filter outright, and its edge list chains across object
   boundaries — 209,416 edges over 10 um, longest 868 um. Extraction goes
   through the object manifest instead.

## Splitting interactively, in the viewer

The Skeleton tab is no longer inert: with an edit service attached, the source
reports itself editable and the Split tool acts on the ZVF store.

```bash
# 1. a store with one object, carrying per-vertex node ids
$PY split_skeleton.py --object 281 --out /tmp/zv_poc

# 2. serve it, and run the edit service over the same directory
python3 serve_store.py /tmp/zv_poc 9077
$PY edit_service.py --root /tmp/zv_poc --port 9099 --token ""

# 3. the viewer
npm run dev-server
```

Add a segmentation layer whose source carries the `#edit=` fragment:

```
http://127.0.0.1:9077/single_axon.zv/|zarr-vectors:#edit=http%3A%2F%2F127.0.0.1%3A9099
```

Then: make object `1` visible, open the **Skeleton** tab, click the **Split**
button's binding widget and press a key to bind it (the widget's tooltip says
"click → bind key, dbclick → unbind"), press that key to activate the tool, and
**Ctrl + right-click** a node. The object becomes two, both appear in the
segment list, and the viewer re-reads the rewritten store without a reload.

Nodes are only pickable when the layer draws points as well as lines; the edit
tool sets that itself (`setSpatialSkeletonModesToLinesAndPoints`).

### How it fits together

| piece | where |
|---|---|
| per-vertex `node_id` column + `vertex_id_attribute` | `split_skeleton.py` |
| decode that column into `chunk.nodeIds` for every geometry kind | `src/datasource/zarr-vectors/geometry_chunk_download.ts` |
| carry it through the chunk transforms, dropping it when alignment cannot be vouched for | `geometry_chunk.ts`, `geometry_backend.ts` |
| `#edit=<url>` → source parameters | `frontend.ts`, `base.ts` |
| `readonly:false` + the five command factories | `geometry_frontend.ts`, `spatial_skeleton_edit.ts` |
| the split itself | `edit_service.py` |

### What is and is not verified

Verified end to end, headless, against the real store: the source reports
`readonly:false`; `getSpatialSkeletonActionsDisabledReason("splitSkeletons")`
returns nothing, which is the same gate the tool checks before it will act;
hovering resolves a node (`nodeId 928` of object 1); running the split command
rewrites the store; and the layer re-reads it live — object 1 (1075) becomes
objects 1 (1072) and 2 (3), the 3-vertex piece being the subtree below the
picked node.

NOT verified: the key-bind-then-Ctrl+right-click gesture under headless
automation. Synthetic key events did not activate the bound tool in the test
harness, so the last step was driven by calling the same command the tool calls.
Worth confirming by hand.

### Limitations

* Only `splitSkeletons` is implemented. The other four required factories exist
  because the duck type demands all five, and they throw a clear message.
* No undo: undoing a split is a merge, which the service does not implement, so
  `undo()` rejects rather than silently doing nothing.
* Each split rewrites the whole store. Fine for one axon (well under a second),
  not a design for a large one.
* The service writes to local disk on request; it binds to 127.0.0.1 and takes a
  token, which `--token ""` disables for local use.

## The in-place variant

`inplace_split.py` performs the same cut as a real mutation of an existing
store, via `EditSession.remove_link(ref, update_objects=True)`:

```
$PY inplace_split.py --store /tmp/zv_poc/single_axon.zv --object 1 \
    --out /tmp/zv_poc/inplace_split.zv
```

The object bookkeeping is correct — object 1 (1075 vertices) becomes objects 2
(140) and 3 (935), summing exactly, with one link row removed (35 -> 34).

**It does not yet render correctly, and the rewrite path is the one to use.**
The edit auto-materialises the store to `links_convention="explicit"`, but the
links family still holds only the 34 branch links, so every within-fragment edge
stops being implied and the viewer draws 34 edges instead of 1073. The original
object 1 also survives with its pre-edit geometry (the edit is copy-on-write),
so the edited store carries both the old and the new objects.

Other constraints measured on that path: only an explicit (branch) link can be
removed, since implicit edges have no row to delete; `atomic=True` is required,
as the in-place variant empties the whole link fragment and destroys other
objects' edges; the second output object's id is never reported and must be
recovered by diffing the present-object set; and `remove_link(update_objects=True)`
refuses a cross-chunk link.
