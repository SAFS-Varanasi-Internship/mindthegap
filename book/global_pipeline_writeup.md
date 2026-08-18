# Global gap-fill pipeline: native 4 km, a first working version

*mind the gap. Standing up a whole-globe run of the gap-fill U-Net at native 4 km on a few recent days, the
design choices it required, and the two training and inference changes, both from Eli's streamlined pipeline,
that were needed before it filled anything sensible.*

## Summary

The pipeline reads native 4 km Copernicus GlobColour chlorophyll for a few recent days, tiles the whole globe,
trains one fully convolutional U-Net, and gap-fills every ocean pixel by blending overlapping tile predictions
with a Hann window. It runs end to end in `book/12-Global_Pipeline.ipynb`. Getting it to produce a real field
rather than a uniform wash came down to two things that the streamlined pipeline already does and ours did
not: training on the full target with plain mean squared error, and relabeling the estimate flag at inference
so the model treats real gaps like the synthetic clouds it was trained to fill. With those in place the fills
land in the right chlorophyll range. The remaining artifact, a fine checkerboard, is undertraining on only
three days, and is being addressed by adding days and training to convergence.

## What it does

One variable (`CHL`), native 4 km, a handful of recent daily frames from Copernicus Marine (the dense
multi-sensor era, well past 2010), clipped to plus or minus 80 degrees latitude. The raw log-chlorophyll cube
for those days is held in memory (a few hundred MB per day), land comes from the product flag, and only the
selected training tiles are ever materialized as model input, so the full global channel cube (which would be
tens of GB) is never built. Prediction tiles the whole frame, feathers the overlaps with a Hann window, and
composites the result: real observations are kept where they exist and the model is shown only in the gaps.

## Design decisions

- **Land-heavy tiles** are dropped by a low ocean-fraction threshold. Land is masked out of the loss, so this
  is an efficiency filter, and the threshold is low so coastal tiles, the most dynamic ones, survive.
- **Ocean-but-no-data tiles** (polar night, persistent cloud) are dropped by a minimum observed-data fraction.
  This also removes the empty high-latitude tiles, so the poles need no special handling beyond a latitude
  clip.
- **Poles** use the native lat/lon grid with no reprojection. The geo x, y, z channels give the model its
  position on the sphere, and the plus or minus 80 degree clip removes the worst distortion and the mostly
  empty rows. Equal-area reprojection is a later refinement.
- **Tile size follows the artifact ratio.** Synthetic clouds are blob_sigma 20 px (about 90 km) and the tiles
  are 240 px, about twelve times the cloud, above the roughly ten-times rule and well under the roughly
  1000 px GPU whole-frame cap.
- **Rectangular tiles that divide the grid.** Following Eli's point that square tiles force overlap, the tile
  height and width are each chosen as the divisor of the grid dimension closest to the target, so the tiles
  partition the globe with no forced overlap in training. On the 3840 by 8640 grid both land on 240, which
  divides both, so it is square here for the right reason (an exact fit) rather than by assumption.
- **The date line wraps.** Longitude tiles wrap around the antimeridian by indexing columns modulo the width,
  and prediction pads longitude with a wrapped margin before the Hann tiler and crops it back, so there is no
  seam at 180 degrees. The geo channels are continuous across that line because they are x, y, z on the sphere.

## The two changes that made it fill

Both come from the streamlined pipeline's `prepare_model_data` and its training setup, and both were necessary.

**Train on the full target with plain MSE.** The gap-fill model fills a pixel because an estimate flag is set
there, which it learns from the synthetic clouds during training. Our earlier objective scored the loss only
at those synthetic-cloud pixels (a held-out masked loss). That taught the model to produce a good value only
for a small synthetic cloud sitting in observed context, and left the rest of the field, including the
observed pixels it uses as context, unconstrained. When asked to fill real gaps it emitted a constant
out-of-range value everywhere (about +585 in standardized units, independent of distance from data). Training
instead on the full target with plain mean squared error, exactly as the streamlined pipeline does with its
`full_target` label, teaches the model to reproduce the whole field and to condition on the estimate flag, and
the fills then land in the normal log-chlorophyll range.

**Relabel the estimate flag at inference.** In training the synthetic clouds carry the estimate flag
(`fake_cloud_flag`) and real clouds carry a separate unavailable flag (`real_cloud_flag`). At inference the two
must be swapped, the way `prepare_model_data(mode="gapfill")` does it: the real gaps become the estimate flag
and the unavailable flag is cleared, so the model treats real clouds like the synthetic clouds it was trained
to fill. Building the inference channels with zero synthetic-cloud coverage alone leaves the estimate flag
empty everywhere, the model sees nothing marked to estimate, and it fills nothing. This one relabel is the
difference between a filled map and a uniform wash.

## Scaling: why it fits, and when to stream

The region-size ceilings measured earlier (a T4 GPU) still frame this: the GPU caps one image, a tile or a
whole frame, at about 1000 px per side, and RAM caps an in-memory channel cube at about 3.5 GB. This pipeline
stays under both by holding only the raw one-channel cube and materializing just the selected tiles, so memory
scales with the number of tiles kept rather than with the globe. Going to many days or the full record is
where the raw cube itself no longer fits, and the next step there is the streaming path: write the raw field
to a disk memmap and build each tile's channels on the fly, which removes the memory ceiling and leaves read
speed as the limit.

## Status and next steps

- The pipeline now produces a coherent global field. The composite is smooth and physically plausible, the
  fills are in the right chlorophyll range, the date line and orientation are correct, and the earlier
  checkerboard is gone. Two changes cleared the checkerboard: a resize-then-conv upsampler (bilinear
  UpSampling2D plus a convolution) in place of the transposed convolution, and training on ten days to
  convergence rather than three. What remains reads as data-limited rather than broken.
- To move from plausible to measurably accurate, in order of value: a held-out score against persistence and a
  climatology, so quality is a number rather than an impression; more days, the cleanest lever on the
  data-limited fill; and matching the synthetic cloud size to the real gap size (a larger blob, and by the
  ratio a larger tile) so the biggest holes are filled rather than leaned across.
- Later: an equal-area treatment of the poles, and the memmap streaming path for many days or the
  full-resolution record.
- Resilience is baked into the notebook: the read cube and the built tiles are both cached to disk, and a
  checkpoint saves the best model as it trains, so a kernel crash costs a quick reload rather than a full
  rebuild.
