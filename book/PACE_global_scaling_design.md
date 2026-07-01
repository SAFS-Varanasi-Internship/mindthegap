# Scaling gap-fill to global PACE OCI — design notes

Working doc for the eScience meeting. Goal: move the U-Net gap-filler from a single
region (Arab Sea, CMEMS `IO.zarr`) to a **much larger / global** area using **PACE OCI
Level-3 `chlor_a`**, batching across **time _and_ space**.

## TL;DR of what changes

| | Arab Sea `IO.zarr` (current) | Global PACE OCI (target) |
|---|---|---|
| Extent | 1 region, 104×152 | Global 1800×3600 (0p1deg) or ~4320×8640 (4km) |
| Batching | full domain per day, stream over **time** | must tile over **time × lat × lon** |
| Ground truth | CMEMS **L4 gapfree** exists → supervised | **No L4.** Only gappy L3 `chlor_a` → self-supervised |
| Gaps | clouds (blobby) | **orbital swath bands** + clouds (banded missingness) |
| Predictors | sst, so, winds, air_temp, prev/next CHL, flags | `chlor_a` only (unless we join SST/RRS) |
| Location | fixed region, no positional input | global → **latitude/longitude must be inputs** |
| Access | public GCS zarr, anywhere | **Icechunk, in-region us-west-2 + earthaccess auth** |
| Chunking | `time=100, lat/lon=full` (clean) | `time=1`, `lat=16 or 512`, `lon=1024` — **two schemes to reconcile** |

The three hard problems below are what we should spend the meeting on.

---

## Problem 1 — Spatial + temporal tiling (and chunk alignment)

The model is a *local* gap-filler; we can't and shouldn't push a global 1800×3600 frame
through the U-Net. We tile space into patches and stream over `(time × lat-tile × lon-tile)`.

xbatcher already supports this — it's the same `BatchGenerator`, just with spatial dims in
`input_dims`:

```python
input_dims    = {"time": 1, "lat": 128, "lon": 128}
input_overlap = {"lat": 32, "lon": 32}   # halo for context + seamless re-stitching
bgen = BatchGenerator(ds_std, input_dims=input_dims, input_overlap=input_overlap)
```

**Our pass-through `make_tf_gen` already generalizes to this** — each xbatcher block is now a
`(1, 128, 128)` tile instead of a full frame, `batch.sizes["time"] == 1`, and it yields one
`(128, 128, C)` sample. Two additions needed (below): tile filtering and a masked target.

**Chunk alignment (straight from our earlier lesson — don't hardcode, read the real chunks):**
- PACE daily `chlor_a` is `time=1` chunked (one day per chunk) — good, per-day reads are clean.
- BUT lat is chunked `16` (post-2026-02 subgroup) or `512` (pre) and lon `1024`. A 128×128
  tile crosses **8 lat-chunks** in the `16` scheme and lon-chunk boundaries → read amplification.
- The two subgroups (`chunks_16`, `chunks_512`) must be `concat`+`sortby('time')`, which leaves
  **inconsistent chunking along time**. For ML tiling we almost certainly want to **rechunk to a
  uniform tile-friendly scheme once** (e.g. `{time:1, lat:128, lon:128}`), possibly persisting a
  training-subset zarr. → *Open question for eScience: best practice here.*

## Problem 2 — No L4 truth → self-supervised masked reconstruction

This is the big one. The Arab Sea pipeline "cheated": CMEMS gives a gapfree **L4** product we
trained toward. **PACE L3M has no gapfree product** — only the gappy `chlor_a`. So we can't
supervise against a complete field.

The standard fix is **self-supervised inpainting**:
1. Take the observed (gappy) `chlor_a` for a day.
2. Add **extra synthetic gaps** on top (our existing +N-day mask-shift trick — and because PACE
   gaps are *banded*, shifting a real day's gap mask naturally produces realistic banded fake
   gaps, so this transfers well).
3. Input = observed **with** the synthetic gaps punched out; target = the observed field.
4. **Loss is computed only where the target is actually observed** (and ideally only on the
   held-out synthetic-gap pixels). Everything NaN/land is ignored.

That requires a **masked loss** — we can't use plain `mse` because most of a global frame is
NaN/land/gap:

```python
def masked_mse(y_true, y_pred):
    # y_true packs [value, mask] in the last axis; mask=1 where we want to score
    val, mask = y_true[..., 0:1], y_true[..., 1:2]
    se = tf.square(val - y_pred) * mask
    return tf.reduce_sum(se) / (tf.reduce_sum(mask) + 1e-6)
```

and the generator yields a 2-channel `y = stack([value, mask])`. This is a genuine change from
the current notebook and worth aligning on with the data scientists (they may have a preferred
inpainting objective / partial-conv approach).

## Problem 3 — Banded gaps, empty tiles, and global position

- **Tile filtering.** Globally, a huge fraction of `(day, tile)` pairs are all-land or fully
  inside a swath gap. Training on them is wasteful/harmful. Skip any tile whose valid-ocean
  fraction is below a threshold (the tutorial did the day-level analog: drop days with >5% NaN
  response). Do it in the generator (`continue`) or pre-compute a valid tile-index list.
- **Land/ocean mask.** PACE L3M CHL has no land flag variable — derive one (`chlor_a` all-NaN
  across time ≈ land, like the tutorial's `invalid_ocean`), or bring an external land mask.
- **Global position matters.** A single region model didn't need location; a global model does
  (Chl dynamics differ by latitude/biome). Add **normalized lat/lon** (or `sin/cos` of lat) as
  input channels so the network knows where each tile sits.
- **Predictors are thin.** The CHL store has only `chlor_a`. Options: (a) start CHL-only
  (masked `chlor_a` + prev/next-day + time + lat/lon + gap flags); (b) join `PACE_OCI_L3M_RRS`
  bands as predictors; (c) join an SST product. Recommend starting with (a) to get the tiling +
  masked-loss pipeline working, then add predictors.

---

## Proposed first milestone (something runnable in-region)

Small, end-to-end, on a **subset** — not the whole globe on day one:

1. `create_ds("PACE_OCI_L3M_CHL", "daily/0p1deg/chunks_512")` + `chunks_16`, concat, `sortby`.
2. Subset to a manageable band (e.g. a 30°×60° box, ~90 days) and **rechunk** to `{time:1, lat:128, lon:128}`.
3. `build_pace_lazy(...)` — a PACE variant of `build_standardized_lazy`: `log10(chlor_a>0)`,
   prev/next-day, sin/cos time, lat/lon channels, gap/valid flags, +N-day fake-gap mask.
   Standardize predictors on a train-time window (label left in log space — our new default).
4. `BatchGenerator` with spatial `input_dims` + overlap; `make_tf_gen` variant that **skips empty
   tiles** and yields `y = [value, mask]`.
5. Same U-Net, `loss=masked_mse`, batch 1, BatchNorm (per our finding).
6. Reassemble tiles (average the overlaps) to view a full-region gap-filled map.

### Skeleton (adapt in us-west-2; not runnable from here)

```python
# --- data ---
ds512 = create_ds("PACE_OCI_L3M_CHL", "daily/0p1deg/chunks_512")
ds16  = create_ds("PACE_OCI_L3M_CHL", "daily/0p1deg/chunks_16")
ds = xr.concat([ds512, ds16], dim="time", coords="minimal",
               compat="override", combine_attrs="override").sortby("time")

ds = ds.sel(lat=slice(30, 0), lon=slice(50, 80))          # subset for the prototype
ds = ds.chunk({"time": 1, "lat": 128, "lon": 128})        # uniform, tile-friendly

# --- lazy feature build (PACE variant of build_standardized_lazy) ---
# log10(chlor_a>0); prev/next-day; sin/cos time; lat/lon channels; valid + fake-gap flags;
# standardize predictors on train window; leave label in log space (standardize_chl=False).

# --- tiled streaming ---
input_dims    = {"time": 1, "lat": 128, "lon": 128}
input_overlap = {"lat": 32, "lon": 32}
bgen = BatchGenerator(ds_std, input_dims=input_dims, input_overlap=input_overlap)

def make_tf_gen_masked(batcher, x_vars, min_valid=0.05):
    def gen():
        for b in batcher:
            b = b.load()
            for t in range(b.sizes["time"]):
                val  = b["CHL"].isel(time=t).values
                mask = (~np.isnan(val)).astype("float32")     # score only observed pixels
                if mask.mean() < min_valid:                    # skip empty/land/gap tiles
                    continue
                x = np.stack([np.nan_to_num(b[v].isel(time=t).values, 0.0) for v in x_vars], -1).astype("float32")
                y = np.stack([np.nan_to_num(val, 0.0), mask], -1).astype("float32")
                yield x, y
    return gen

# model.compile(optimizer="adam", loss=masked_mse, metrics=[...])
```

---

## Open questions for the eScience data scientists

1. **Access at training scale.** Best pattern for high-throughput reads from the Icechunk store
   during training — dask cluster / gateway on CryoCloud? Local caching? Expected read throughput
   in-region for tiled random access?
2. **Rechunking.** Should we persist a rechunked, ML-ready training zarr (uniform `{1,128,128}`),
   or tile directly off the virtual store? How to handle the `chunks_16` vs `chunks_512` seam?
3. **Grid.** 0p1deg vs 4km for a first global model — recommendation given storage/throughput/GPU?
4. **Objective.** Do they have a preferred self-supervised inpainting setup for banded L3 gaps
   (masked MSE vs partial convolutions vs something else)? Any leakage concerns with the
   +N-day fake-gap trick given orbital repeat cycles?
5. **Swath/quality metadata.** Is there a per-day coverage or swath-geometry mask we should use to
   distinguish "real gap" from "cloud" from "land"? (PACE L3M CHL as read has no land flag.)
6. **Co-gridded predictors.** Recommended companion variables (SST, RRS bands from the RRS store)
   already on the same grid, and how to join them efficiently in-region.
7. **Compute.** GPU availability in us-west-2 for training, and whether they'd run this on a
   cluster vs single node.

## Notes / carryover from the Arab Sea work

- `mtg.build_standardized_lazy(..., standardize_chl=False)` and the pass-through `mtg.make_tf_gen`
  are the reusable primitives; the PACE versions are variants of these (spatial tiling + masked y).
- Keep **BatchNorm** (LayerNorm/GroupNorm hit this image's GPU kernels; BatchNorm at batch 1 works).
- Don't hardcode chunk sizes — read `ds.chunksizes` and align tiles to them.
