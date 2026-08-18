# The tiling artifact in the gap-fill U-Net: cause, threshold, and scaling limits

*mind the gap. Where the grid-like smudging in tiled training comes from, why it happens, when it stops
mattering, how to avoid it, and the region-size limits that decide when you have to tile in the first place.*

## Summary

Training the gap-fill U-Net on small spatial tiles produces a grid or seam texture in the filled field that
the whole-frame model does not have. The cause is the self-supervised training itself: the synthetic clouds
that hide pixels get cut at tile boundaries, so the model learns to fill each cut piece without seeing across
the seam. The effect scales with the size of the synthetic clouds relative to the tile, so it is a ratio
between two pixel sizes, not a fixed pixel count. It disappears once the tile is comfortably larger than the
clouds, roughly ten times the cloud sigma, which is about 128 px for the blob-12 clouds we train with. We
confirmed the same pixel threshold on two datasets six times apart in resolution (IO.zarr at 25 km and
Copernicus GlobColour at 4 km), so the rule transfers across datasets. Whether you ever hit the artifact
depends on whether you have to tile at all: a region that fits a whole-frame pass does not tile and cannot
show it, and the region-size ceilings that force tiling are the GPU whole-frame limit (about 1000 px per side)
and the in-memory cube limit (about a 3.5 GB cube of days by lat by lon by channels), the latter of which
streaming from disk removes.

## What the artifact looks like

The panels below are one moderate-gap day and one clearer day off the Oman coast, each filled by a model
trained at a different tile size, with the same synthetic clouds. The observed column is the input, gray is
land, and white is cloud. Reading left to right, the whole-frame and 128 px fills are smooth, and a fine
stippled grid texture appears at 112 px and becomes pervasive at 88 px.

![Copernicus 4 km fills by training tile size, Oman upwelling. Whole-frame and 128 px are smooth; 112 px and 88 px show a stippled grid texture.](copoman_fills.png)

The texture is aligned to the training tile grid, not to any physical feature, and it is strongest over the
lower-structure open water where there is nothing real to anchor the fill.

## Where it comes from and why it happens

The model is trained self-supervised. Synthetic clouds hide some observed pixels, and the loss asks the model
to reconstruct those hidden pixels from what remains. When training happens on tiles, each tile is a separate
training example with no information from its neighbors. A synthetic cloud that is larger than a tile, or that
straddles a tile boundary, is therefore split: the model only ever sees the part of the cloud that falls
inside the current tile and learns to fill it from that tile alone. At inference on the whole frame, the fills
that were learned independently on either side of a boundary meet there and do not agree, which is the seam.
Because the clouds are placed at random each epoch, the seams accumulate along the whole tile grid and read as
a texture rather than a single line.

We ruled out the other candidate causes in turn:

- Undertraining. The artifact is still present with 40 epochs and early stopping, so it is not a matter of
  training longer.
- The geo channels. It appears in both the geo and no-geo models and reproduces across two seeds, and a
  no-geo U-Net is translation invariant, so the texture is not locked to absolute position.
- The transposed-convolution checkerboard (Odena et al., 2016). The whole-frame model uses the same
  architecture, including the `Conv2DTranspose` upsampling, and is clean, so the upsampler is not the dominant
  cause here.
- The loss location. Cropping the loss to each tile's interior, so boundary pixels contribute no gradient, did
  not remove it.

The single most telling piece of evidence is that the whole-frame-trained model, which never cuts a cloud, is
clean, and every tiled model artifacts even when it is then run on the whole frame. So the cause is the
training-time tiling cutting the synthetic clouds, not the architecture, the inputs, or the inference.

## Cloud size drives it

Holding the tile fixed and varying the synthetic cloud size makes the mechanism explicit. At a fixed 88 px
tile, blob_sigma 4 gives essentially no artifact but also a weaker, smoother model because the gaps are too
easy to force any structure learning, blob_sigma 12 gives real structure with some artifact, and blob_sigma 30
gives a severe artifact. The artifact grows as the cloud grows relative to the tile, which is what makes it a
ratio rather than an absolute tile size.

## When it disappears with tile size

Sweeping the tile size at a fixed cloud size locates the threshold. On the IO.zarr 176 by 240 domain at
blob 12, the artifact is present at 112 px and basically gone by 128 px, and the held-out reconstruction error
is flat across tile sizes, so the sizes are equally accurate on ordinary pixels and the artifact is the only
thing that separates them. We then repeated the sweep on Copernicus GlobColour at 4 km, on a 176 by 240 box
over the Oman upwelling, holding the pixel tile sizes and the cloud size identical. The result is the same
progression shown above: present at 88 and 112, gone by 128 and whole. Since 4 km is six times finer than the
25 km IO grid and the threshold landed at the same pixel sizes, the artifact is a property of the tile size in
pixels relative to the cloud size in pixels, and it transfers across datasets rather than being specific to one
resolution.

## How to fix it

The rule that follows from the threshold is to make the training tile comfortably larger than the synthetic
clouds. From the sweep, 128 px over blob_sigma 12 is clean and 112 px over 12 is not, so the safe ratio is
about ten to eleven times blob_sigma, with the artifact starting to appear below about nine. The ratio is
dimensionless, so it reads the same in pixels, degrees, or kilometers, and the only thing that changes between
datasets is how many pixels and degrees that ratio lands on.

| blob_sigma | safe tile (about 10.7x) | vs the 1000 px GPU cap | cloud at 4 km | tile at 4 km |
|---|---|---|---|---|
| 12 px | 128 px | 8x under | about 55 km | about 590 km |
| 20 px | 214 px | about 5x under | about 90 km | about 985 km |
| 30 px | 320 px | about 3x under | about 140 km | about 1470 km |
| 50 px | 535 px | about 2x under | about 230 km | about 2460 km |
| 90 px | 960 px | at the cap | about 415 km | about 4400 km |

The same ratio on the two datasets, to show how resolution moves it:

| dataset | resolution | cloud (blob 12) | safe tile (128 px) |
|---|---|---|---|
| Copernicus GlobColour | 4 km, 24 px/deg | 0.5 deg, about 55 km | 5.3 deg, about 590 km |
| IO.zarr | 25 km, 4 px/deg | 3.0 deg, about 330 km | 32 deg, about 3550 km |

On the coarse IO grid the safe tile is most of a basin, which is why on that data you simply train the whole
frame and never tile. The best fix, whenever the region allows it, is exactly that: a region that fits a
whole-frame pass has no training tiles to cut, so it cannot show the artifact at all. When a region is too
large for a whole-frame pass, tile it with patches that clear the clouds, and blend overlapping tiles at
inference with a Hann window so the prediction-side seams are feathered out as well (following the
overlapping-tile guidance in Reina et al., 2020).

## Scaling limits

Whether you have to tile, and how large a region you can hold, is set by two separate ceilings measured on the
Hub (a T4 GPU, batch 8, about 12 channels). They govern different things.

The GPU caps the largest single image it processes in one pass, a tile or a whole frame, at about 1000 px per
side (a whole-frame pass fit at 960 by 960 and ran out of memory at 1200 by 1200). The safe 128 px tile is
about eight times under this, so for any normal cloud size the safe tile fits with room to spare, as the third
column of the ratio table shows.

The RAM caps the whole cube you build in memory at once, which is days by lat by lon by channels by 4 bytes,
at about a 3.5 GB final cube (a build peak of roughly three times that, near 10 GB, is the real requirement).
This does not depend on the tile size, since you hold the whole cube regardless of how you later chop it. More
days trades against a smaller box:

| days | max region side | at 4 km (24 px/deg) | final cube |
|---|---|---|---|
| 30 | about 1560 px | about 65 deg | 3.5 GB |
| 45 | about 1270 px | about 53 deg | 3.5 GB |
| 90 | about 900 px | about 37 deg | 3.5 GB |
| 180 | about 640 px | about 27 deg | 3.5 GB |
| 365 | about 450 px | about 19 deg | 3.5 GB |

Reusable form: final cube GB is days times side in pixels squared times channels times 4, divided by 1e9, kept
under about 3.5. For a whole-frame pass the region also cannot exceed the GPU's 1000 px, so below about 70 days
the GPU cap binds first and above it the RAM cap does; they cross near 73 days at 1000 px.

Streaming removes the RAM ceiling. Instead of building the whole cube, the raw field is written to disk once
and opened as a memmap, and a generator reads one tile at a time, builds that tile's channels on the fly with
shared global statistics, and feeds it to the model through a prefetching `tf.data` pipeline. At any instant
memory holds only one tile's channels plus a prefetch buffer, about 0.3 GB for a 128 px tile over 365 days,
and that number does not grow with the region. A basin or the whole globe then streams with the same small
working set, and the binding limit becomes disk space for the memmap and read speed from it, not memory. This
is the path for anything larger than the table above.

All of these numbers translate to another dataset through its pixels per degree. The pixel ceilings (1000 px
per side, the cube in pixels) are hardware properties and hold as they are, and converting them to a region in
degrees just divides by that dataset's pixels per degree.

## Practical recommendations

- If the region fits a whole-frame pass, roughly 1000 px per side and within the cube limit, train it whole.
  There is no tiling and therefore no artifact.
- If the region is larger, tile it with patches at least about ten times blob_sigma (128 px for the clouds we
  use), stream the tiles from a memmap so RAM stops being the limit, and blend overlapping tiles with a Hann
  window at inference.
- Watch the cloud size, since it sets the minimum safe tile. On fine data a given physical cloud is many
  pixels, so the minimum safe tile in pixels grows, and if the clouds you fill are 400 km-scale the safe tile
  reaches the GPU cap and coarsening becomes the better option.

## Figures

- `copoman_fills.png` (embedded above): Copernicus 4 km fills by training tile size over the Oman upwelling,
  two days, showing the artifact at 88 and 112 px and its absence at 128 px and whole frame. Generated by the
  Oman analysis cell (Cell C) in `11-Tile_Size_Sweep.ipynb`.
- Optional, IO reference days: the same comparison on IO.zarr for 2022-11-30 and 2021-04-26, the two
  high-gap reference frames. Generated by the IO artifact analysis cell in the same notebook.
- Optional, cloud-size effect: the fixed-tile, varying-cloud comparison (blob 4, 12, 30) that shows the
  artifact growing with cloud size. If you want it in the writeup, tell me and I will give a cell that
  regenerates it from the saved blob-sweep models.
