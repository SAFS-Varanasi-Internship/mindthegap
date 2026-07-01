# Scaling gap-fill to global PACE OCI, discussion notes

Working doc for the eScience meeting. These are proposals and open questions, not settled
decisions. Goal: explore moving the U-Net gap-filler from a single region (Arab Sea, CMEMS
`IO.zarr`) to a much larger or global area using PACE OCI Level-3 `chlor_a`, batching across
time and space.

Most of the "facts" below are read off the PACE material shared with us; citations point back
to the source so we can double check them with the data scientists. See the Sources list at the
bottom.

## Update: concrete details from Eli's PACE notebook (meeting 2026-07-02)

Eli built a notebook that creates `zarr_ds` from PACE for the Indian Ocean, daily and 8-day [P3].
Key specifics and guidance that sharpen the plan:

- **Use `chunks_512` only for now.** Eli found the `chunks_512` + `chunks_16` concat to be a memory
  hog, so the current approach skips it and reads `chunks_512` alone [P3]. That mostly retires our
  earlier open question about reconciling the two subgroups.
- **`create_ds` opens with `chunks={}`** [P3], which matches the chunk-aware reading we planned.
- **Arabian Sea daily (chunks_512), cropped to lat 5 to 31, lon 42 to 80:** dims
  `time: 683, lat: 260, lon: 380`, `chlor_a` only, chunk `(1, 260, 380)` [P3]. Two things follow:
  - Resolution is **260 x 380** at 0.1 deg, different from the CMEMS 104 x 152. The U-Net input
    shape changes (cropping to a multiple of 8 gives 256 x 376). Eli flagged this explicitly.
  - For a region this size the whole crop is **one spatial chunk per day**, so spatial tiling is
    not needed for the Arabian Sea itself. Tiling (Area 1 below) only matters if we later go to a
    much larger or global extent.
- **8-Day product (8Day/0p1deg/chunks_512):** dims `time: 87, lat: 260, lon: 380` [P3]. Far fewer
  time steps, but per Eli a lot less missing data than daily.
- **Daily vs 8-Day is now an open decision.** Daily has 683 steps but heavy banded missingness;
  8-day has only 87 steps but much fuller coverage. Eli's suggestion is either to focus on filling
  the **8-day** product, or do **daily but add 3 days before and 3 days after** as temporal context
  (an extension of the prev/next-day channels to a plus/minus 3 day window).
- **We do need a test set** (Eli reversed an earlier "no test data" note).
- **Train/val/test split is genuinely open** given only about 678 days and strong temporal
  autocorrelation. A naive random or contiguous split risks leakage because nearby days are
  correlated. Worth asking the data scientists how they would split (for example contiguous blocks
  with a buffer gap between them, or holding out whole seasons) to keep leakage low while retaining
  enough data.

Still unchanged and central: PACE (this store) has no L4 gapfree product, so training stays
self-supervised masked reconstruction (Area 2), and the gaps are banded (Area 3).

## What seems to change (for discussion)

| Aspect | Arab Sea `IO.zarr` (current) | Global PACE OCI (target) |
|---|---|---|
| Extent | 1 region, 104x152 | Global 1800x3600 at 0p1deg [S1] |
| Batching | full domain per day, stream over time | probably need to tile over time, lat, lon |
| Ground truth | CMEMS L4 gapfree exists, so supervised | no L4 gapfree in this store, only gappy L3 `chlor_a` [S2] |
| Gaps | clouds (blobby) | banded (orbital swaths) plus clouds [S3] |
| Predictors | sst, so, winds, air_temp, prev/next CHL, flags | `chlor_a` only in this store [S2] |
| Location | fixed region, no positional input | global, so lat/lon may need to be inputs |
| Access | public GCS zarr, anywhere | Icechunk, appears to need in-region us-west-2 plus earthaccess auth [S4] |
| Chunking | `time=100, lat/lon=full` (from our diagnostic) [S5] | `time=1`, `lat=16 or 512`, `lon=1024`, two schemes [S1][S6] |

The three areas below are where we think the meeting time is best spent. We are not confident
on any of them yet; each ends with what we would want to ask.

## Area 1: spatial plus temporal tiling (and chunk alignment)

The model is a local gap-filler, so pushing a full global 1800x3600 frame through the U-Net is
probably not the right approach. A natural option is to tile space into patches and stream over
`(time, lat-tile, lon-tile)`. xbatcher can already do this by adding spatial dims to `input_dims`:

```python
input_dims    = {"time": 1, "lat": 128, "lon": 128}
input_overlap = {"lat": 32, "lon": 32}   # halo for context and re-stitching
bgen = BatchGenerator(ds_std, input_dims=input_dims, input_overlap=input_overlap)
```

We think our pass-through `make_tf_gen` mostly generalizes to this: each block becomes a
`(1, 128, 128)` tile, so `batch.sizes["time"] == 1` and it yields one `(128, 128, C)` sample. The
tile size of 128 is a guess and should be tuned.

Chunk alignment looks like it matters more here than it did for `IO.zarr`:
- PACE daily `chlor_a` appears to be `time=1` chunked (one day per chunk), which is convenient for
  per-day reads [S1][S6].
- lat is chunked at 16 (post 2026-02 subgroup) or 512 (pre), and lon at 1024 [S1]. If that is
  right, a 128x128 tile would cross several lat chunks in the `16` scheme and cross lon-chunk
  boundaries, which could amplify reads.
- The two subgroups (`chunks_16`, `chunks_512`) have to be concatenated and sorted by time [S6],
  which would leave inconsistent chunking along time. We suspect we want to rechunk to a uniform,
  tile-friendly scheme once (for example `{time:1, lat:128, lon:128}`), and possibly persist a
  training-subset zarr, but this is exactly the kind of thing to confirm with them.

Questions: is rechunking to a uniform ML grid the right move, or is there a better pattern for
tiled reads directly off the virtual store? How should we handle the `chunks_16` vs `chunks_512`
seam? What tile size and overlap do they usually use?

## Area 2: no L4 truth, so likely self-supervised masked reconstruction

This may be the most important thing to get their read on. The Arab Sea pipeline had it easy:
CMEMS provides a gapfree L4 product we trained toward. As far as we can tell, PACE L3M CHL has no
gapfree product in this store, only the gappy `chlor_a` [S2]. If that is correct, we cannot
supervise against a complete field.

One common approach is self-supervised inpainting:
1. Take the observed (gappy) `chlor_a` for a day.
2. Add extra synthetic gaps on top (our existing `+N-day` mask-shift trick). Because PACE gaps are
   banded [S3], shifting a real day's gap mask should produce realistic banded fake gaps, so this
   part may transfer well.
3. Input is the observed field with the synthetic gaps punched out; target is the observed field.
4. Loss is computed only where the target is actually observed (ideally only on the held-out
   synthetic-gap pixels). NaN, land, and real gaps are ignored.

That points to a masked loss rather than plain `mse`, since most of a global frame is NaN, land,
or gap:

```python
def masked_mse(y_true, y_pred):
    # y_true packs [value, mask] in the last axis; mask=1 where we want to score
    val, mask = y_true[..., 0:1], y_true[..., 1:2]
    se = tf.square(val - y_pred) * mask
    return tf.reduce_sum(se) / (tf.reduce_sum(mask) + 1e-6)
```

with the generator yielding a 2-channel `y = stack([value, mask])`. This is a real change from the
current notebook, so it is worth checking whether they have a preferred objective.

Questions: is masked MSE the right objective, or do they prefer partial convolutions or another
inpainting setup? Does the `+N-day` fake-gap trick risk leakage given PACE's orbital repeat cycle?
Is there a coverage or quality flag that distinguishes real gap from cloud from land?

## Area 3: banded gaps, empty tiles, and global position

- Tile filtering. Globally, we expect many `(day, tile)` pairs to be all-land or fully inside a
  swath gap. Training on those seems wasteful, so we would probably skip tiles whose valid-ocean
  fraction is below a threshold. The OHW Part I tutorial did the day-level analog, dropping days
  with more than 5 percent NaN in the response [S7]. Threshold TBD.
- Land or ocean mask. This store looks like it has only `chlor_a` and `palette`, with no land flag
  [S2], so we may need to derive an ocean mask (for example, pixels that are always NaN across time
  approximate land, as in the tutorial's `invalid_ocean` step [S7]), or bring an external mask.
- Global position. A single-region model did not need location, but a global model probably does,
  since Chl behavior varies by latitude and biome. Adding normalized lat/lon (or sin/cos of lat) as
  input channels is one option to discuss.
- Predictors are thin. This store has only `chlor_a` [S2]. Options: (a) start CHL-only (masked
  `chlor_a`, prev/next-day, time, lat/lon, gap flags); (b) join `PACE_OCI_L3M_RRS` bands; (c) join
  an SST product. We lean toward (a) first just to get the pipeline working, but this is open.

Questions: which co-gridded predictors do they recommend, and how to join them efficiently
in-region? Is deriving the land mask from all-time-NaN acceptable, or is there a canonical mask?

## A possible first milestone (to sanity check with them)

Small and end-to-end on a subset, not the whole globe on day one. This is a strawman, not a plan:

1. `create_ds("PACE_OCI_L3M_CHL", "daily/0p1deg/chunks_512")` plus `chunks_16`, concat, sort by
   time [S6].
2. Subset to a manageable box (for example 30 by 60 degrees, roughly 90 days) and rechunk to
   `{time:1, lat:128, lon:128}`.
3. `build_pace_lazy(...)`, a PACE variant of `build_standardized_lazy`: `log10(chlor_a>0)` [S8],
   prev/next-day, sin/cos time, lat/lon channels, gap and valid flags, plus the `+N-day` fake-gap
   mask. Standardize predictors on a train-time window; leave the label in log space (our new
   default).
4. `BatchGenerator` with spatial `input_dims` plus overlap; a `make_tf_gen` variant that skips
   empty tiles and yields `y = [value, mask]`.
5. Same U-Net, `loss=masked_mse`, batch 1, BatchNorm (per our earlier finding on this image).
6. Reassemble tiles (average the overlaps) to view a full-region gap-filled map.

### Skeleton (would run in us-west-2; we cannot run it from here)

```python
# --- data ---
ds512 = create_ds("PACE_OCI_L3M_CHL", "daily/0p1deg/chunks_512")
ds16  = create_ds("PACE_OCI_L3M_CHL", "daily/0p1deg/chunks_16")
ds = xr.concat([ds512, ds16], dim="time", coords="minimal",
               compat="override", combine_attrs="override").sortby("time")   # [S6]

ds = ds.sel(lat=slice(30, 0), lon=slice(50, 80))          # subset for the prototype
ds = ds.chunk({"time": 1, "lat": 128, "lon": 128})        # uniform, tile-friendly (proposed)

# --- lazy feature build (PACE variant of build_standardized_lazy) ---
# log10(chlor_a>0); prev/next-day; sin/cos time; lat/lon channels; valid and fake-gap flags;
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

## Consolidated questions for the eScience data scientists

1. Access at training scale: best pattern for high-throughput reads from the Icechunk store during
   training (dask cluster or gateway on CryoCloud, local caching), and expected in-region throughput
   for tiled random access.
2. Rechunking: persist a rechunked ML-ready training zarr (uniform `{1,128,128}`), or tile directly
   off the virtual store? How to handle the `chunks_16` vs `chunks_512` seam [S6]?
3. Grid: 0p1deg vs 4km for a first global model, given storage, throughput, and GPU.
4. Objective: preferred self-supervised inpainting setup for banded L3 gaps (masked MSE, partial
   convolutions, or other), and any leakage concerns with the `+N-day` fake-gap trick.
5. Swath and quality metadata: is there a per-day coverage or swath-geometry mask to separate real
   gap from cloud from land? (This store as read appears to have no land flag [S2].)
6. Co-gridded predictors: recommended companion variables (SST, RRS bands from the RRS store) on
   the same grid, and how to join them efficiently in-region.
7. Compute: GPU availability in us-west-2, and cluster vs single node for training.

## Carryover from the Arab Sea work

- `mtg.build_standardized_lazy(..., standardize_chl=False)` and the pass-through `mtg.make_tf_gen`
  are the reusable primitives; the PACE versions would be variants (spatial tiling plus masked y).
- Keep BatchNorm for now (LayerNorm and GroupNorm hit this image's GPU kernels; BatchNorm at
  batch 1 works). This was specific to our environment and should be re-checked in theirs.
- Do not hardcode chunk sizes; read `ds.chunksizes` and align tiles to whatever is there.

## Sources

Primary PACE references (the store and its example notebook):
- [P1] fish-pace/pace-icechunks repo: https://github.com/fish-pace/pace-icechunks
- [P2] pace-icechunk-examples.ipynb:
  https://github.com/fish-pace/pace-icechunks/blob/main/pace-icechunk-examples.ipynb
- [P3] Eli's Indian Ocean PACE batches notebook (daily + 8-day zarr_ds creation):
  https://github.com/SAFS-Varanasi-Internship/mindthegap/blob/eli-branch/contributor_folders/eli/PACE_CHL_batches.ipynb
  (source for the 260x380 Arabian Sea dims, `time: 683` daily / `time: 87` 8-day, `chunks_512`-only
  approach, `chunks={}` open, and Eli's guidance on test data, +/-3 day context, and the split.)

- [S1] PACE `ds` repr in [P2]: `chlor_a (time, lat, lon) float32` with
  `chunksize=(1, 16, 1024)`, dims `time: 710, lat: 1800, lon: 3600`, time span 2024-03-05 to
  2026-02-28.
- [S2] PACE CHL `ds` repr in [P2]: data variables are `chlor_a` and `palette` only (no gapfree, no
  land flag). Store description in [P1] lists `PACE_OCI_L3M_CHL` and `PACE_OCI_L3M_RRS`.
- [S3] User note from the meeting: PACE is "very banded data, gaps missing in bands," consistent
  with orbital swath coverage.
- [S4] [P1] / [P2]: "requires you are in AWS us-west-2," and auth via `earthaccess.login()` plus
  `get_s3_credentials(daac="OBDAAC")`.
- [S5] This session's chunk diagnostic on `IO.zarr` cropped: `time` chunks all 100 (last 71),
  `lat (104,)`, `lon (152,)`.
- [S6] [P1] / [P2]: daily CHL split into `daily/0p1deg/chunks_512` and `chunks_16` subgroups by
  date (before vs after 2026-02); "these subgroups need to be merged after reading," via
  `xr.concat(...).sortby("time")`.
- [S7] OHW "Preparing a Zarr dataset for our CNN training" tutorial: drops days with more than
  5 percent NaN in the response and builds an ocean mask from always-NaN SST (`invalid_ocean`).
  (No URL captured; from the notebook text shared this session.)
- [S8] PACE plot example in [P2]: `np.log10(da_small.where(da_small > 0))`.
