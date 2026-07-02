# Scaling gap-fill to global PACE OCI, discussion notes

Working doc for the eScience meeting. These are proposals and open questions, not settled
decisions. Goal: explore moving the U-Net gap-filler from a single region (Arab Sea, CMEMS
`IO.zarr`) to a much larger or global area using PACE OCI Level-3 `chlor_a`, batching across
time and space.

Most of the facts below are read off the PACE material; citations point back
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
    not needed for the Arabian Sea itself. Tiling only matters if we later scale to a much larger
    or global extent.
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

Still unchanged and central: PACE (this store) has no L4 gapfree product [S2], and the gaps are
banded [S3].

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

Spatial tiling for a global extent is a later concern, not a near-term question: the current
streaming code already reads chunk-aware, and per Eli the Arabian Sea crop is one chunk per day,
so no tiling is needed yet. The open questions are collected at the end; the two Eli flagged,
daily-vs-8-day and a leakage-safe split, are the ones we most want their read on.

## Practical questions: land mask and predictors

**Land / ocean mask.** The model should ignore land pixels: they hold no chlorophyll to fill, and
we do not want them counted in the loss. In the Arab Sea `IO.zarr` a `land_flag` variable was
provided for exactly this. The PACE CHL store does not include one, it has only `chlor_a` (the
chlorophyll values) and `palette` (a color lookup table for plotting, not usable as data) [S2]. So
we would have to build the mask ourselves. The simplest way is to treat any pixel that is NaN on
every day as land, since land is never measured (the OHW tutorial calls this `invalid_ocean` [S7]),
but that is only approximate because it also catches ocean pixels that happen to be missing on
every day. **Question:** is deriving land from all-time-NaN good enough, or is there an official
land or coastline mask they would point us to?

**Predictors.** Predictors are the extra input channels the model gets alongside the gappy
chlorophyll to help it fill the gaps. The Arab Sea model had several (sea-surface temperature,
salinity, winds, air temperature, plus previous and next-day CHL and the flags). The PACE CHL store
has only `chlor_a` [S2], so the only inputs we can build from it are ones derived from CHL itself:
the masked CHL, previous and next-day CHL, a day-of-year time encoding, and the gap flags. To get
richer inputs we could pull reflectance bands from the sibling PACE RRS store (`PACE_OCI_L3M_RRS`)
or bring in an outside SST product. The catch is that an extra variable only helps if it sits on the
**same lat/lon grid and the same days** as the CHL, so each pixel lines up cell-for-cell; if it is
on a different grid it has to be regridded first, which is extra work and a source of error.
**Question:** which extra variables would they recommend, and are any of them already on the PACE
grid so we can join them directly instead of regridding?

Two more things come back only if we later scale past a single region, and neither is needed for
the Arabian Sea run:
- **Tile filtering.** A large or global area gets cut into many small patches, and a lot of those
  patches will be entirely land or entirely inside a data gap. Training on them wastes effort and
  can hurt the model, so we would skip any patch whose ocean-with-data fraction is too low.
- **Position channels.** A global model sees many regions at once, and chlorophyll behaves
  differently by latitude and ecosystem, so we would feed each patch its lat/lon to tell the model
  where it sits. A single-region model does not need this because the location never changes.

## Consolidated questions for the eScience data scientists

1. Daily vs 8-Day: fill the fuller 8-day product (87 steps), or daily (683 steps, heavy banded
   missingness) with a plus/minus 3 day temporal window? [P3]
2. Train/val/test split: with about 678 autocorrelated days, how would you split without leakage
   (buffered contiguous blocks, whole-season holdout, other) while keeping enough data? [P3]
3. Metadata: is there a per-day coverage or quality flag that separates real gap from cloud from
   land? (This store as read has no land flag [S2].)
4. Predictors: which extra input variables would they recommend (for example SST, or RRS bands
   from the sibling RRS store), and are any already on the same lat/lon grid and days as the CHL so
   we can join them without regridding?
5. Access and compute: high-throughput reads from the Icechunk store during training (dask on
   CryoCloud, caching), and GPU availability in us-west-2.

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
