# PACE gap-fill: streaming to basin scale

*mind the gap. Picking up at the end of nb7.*

**Starting point.** At the end of nb7 we made the synthetic clouds temporally autocorrelated (`time_sigma`,
a 3-D gaussian-noise cube), so a hidden pixel is usually also hidden on the neighboring days, the way a real
cloud is. This removed the shortcut where the model reconstructed a hidden pixel by copying its value from
the previous or next day. The benefit of the neighbor-day channels dropped from about 21 percent under
independent clouds to about 4 percent under correlated clouds, and the U-Net's margin over persistence
dropped with it. On realistically persistent gaps the U-Net was about even with persistence, because most of
its earlier margin came from copying neighbor days rather than from spatial reconstruction.

Everything after that point tests two things: whether the model adds real spatial value over persistence now
that the temporal shortcut is gone, and whether the approach scales toward a global map. This note covers
what changed after that.

## How the tiling scales from the Arabian Sea to a basin

The Arabian Sea runs trained on small 40×56 chunks cut from the region. Because the U-Net is fully
convolutional, a model trained on those chunks predicts the whole region by sliding the same filters across
it: the chunk is the training unit, and the region is only a question of how many chunks cover it.

Going from the Arabian Sea to the Indian Ocean, several times its area, changes neither the chunk nor the
model. We cut proportionally more chunks so the per-area sampling density stays the same as the Arabian Sea
had, about 16 chunks over an Arab-Sea-sized area. A model trained on chunks drawn from across the larger
domain then predicts the whole domain one tile at a time, the same way it predicted the Arabian Sea. The
basin is therefore the same operation as the single sea, run with more chunks and a larger stitch, not a new
method.

![Two regions drawn to scale, with training chunks at the same per-area density. The chunk stays 40×56 pixels; the larger domain simply takes proportionally more of them. The sampler keeps chunks that are mostly ocean.](regions_chunks.png)

## Streaming

A basin at daily resolution is too large to hold as a single array. The channel cube for a 600×696 region
across all days would need roughly 7 to 8 GB, more than the kernel's RAM, so the pipeline never builds it
whole. It streams the chunks instead, keeping only one in memory at a time:

- Disk-backed memmap cache. The region is written once to a `(T,H,W)` float32 memmap on disk, filled in
  time-blocks so RAM holds only one block, and reused across kernel restarts. This avoids both the RAM limit
  and the `to_zarr` zarr v2 versus v3 codec crash.
- A `tf.data.from_generator` pipeline reads one ocean-covering 40×56 chunk at a time from the memmap, builds
  its channels with `build_pace_channels`, shuffles, and trains.
- Two backward-compatible arguments were added to `build_pace_channels`: `stats=`, shared global statistics
  computed once from a coarse strided read so every chunk normalizes the same way and predictions invert
  with a single CHL mean and standard deviation; and `land=`, described in the bugs below.

Training on a 600×696 daily all-time region runs at about 6 seconds per epoch, with validation loss falling
from 0.30 to 0.057. A model that predicted the per-pixel mean would sit near 1.0 in these standardized units,
so 0.057 shows the model is reconstructing real structure.

![Training and validation loss per epoch. Best validation loss 0.057 at epoch 10.](loss_curve.png)

## Reconstruction on a held-out day

![Streamed gap-fill. From left: input with fake clouds removed, U-Net reconstruction, truth, and absolute error at the hidden pixels. The first three panels share one color scale.](gapfill_day.png)

The reconstruction is a complete, coherent field: high chlorophyll along the Indian, Somali, and Indonesian
coasts and in the Bay of Bengal, low chlorophyll in the open subtropics. The error panel is mostly dark,
with the largest errors in the productive coastal and upwelling zones.

## Results relative to the starting point

![Held-out fake-cloud MAE, U-Net versus standard persistence, in two settings.](mae_bars.png)

- Daily basin (600×696): U-Net 0.20 versus standard persistence 0.175. Persistence is about 12 percent lower.
- Composites over a compact region: U-Net 0.168 versus 0.213. The U-Net is lower here, but 3-day
  compositing weakens persistence, since it then predicts across roughly 3-day gaps rather than 1 day, so
  part of that margin is an easier baseline.

With the temporal shortcut removed, the model has to beat persistence on spatial reconstruction alone. On a
compact, single-regime region it does. On the full basin it does not, because one model with no location
information cannot represent both nutrient-poor subtropical gyres and equatorial upwelling at once. Adding
positional channels (latitude and longitude) is the next step, and is required for a global model in any case.

## Two problems worth recording

- Keras 3 compiles the training step with XLA by default, and XLA plus a dynamic `(None,None)` U-Net input
  raised `CUDNN_STATUS_EXECUTION_FAILED` at the first fit step, with 13 GB free, so not a memory issue.
  Setting `model.compile(..., jit_compile=False)` fixes it. This is separate from the earlier batch-1 and
  80×112 shape failures.
- Land-window bug, which cost a debugging cycle (error 0.64 falling to 0.20 once fixed). `build_pace_channels`
  identifies land as pixels that are NaN across all frames. At inference we built the input channels from a
  3-frame window, so any pixel that happened to be cloudy on those 3 days was labeled as land, which pushed
  the flag and masked-value channels far from the training distribution and drove the prediction toward the
  mean. The model itself was fine (validation loss 0.057). Passing the global land mask (`land=~ocean`) at
  inference restored the result. Any channel build from a short time window must be given a precomputed land
  mask.

## Next

- Positional channels (latitude and longitude, or sin and cos of latitude) so one model can represent
  several chlorophyll regimes. Required for a global model.
- Grid versus random chunks. The streaming sampler currently draws random ocean chunks. Earlier
  spatial-chunk work found explicit non-overlapping grid tiling beat random in a region that tiles cleanly
  (about 0.27 versus 0.31), so testing fixed boxes that cover the whole region is worth doing, since that
  result may transfer.
- Globe sampling. The sampler is uniform in pixel space, so it oversamples high latitudes where a grid cell
  covers less ocean. Weight chunk positions by area (cos of latitude) once the domain reaches beyond the
  tropics.
- Daily resolution on a single homogeneous sea, as a clean test of whether spatial reconstruction beats the
  daily persistence bar without any location information.
- Calibrate the synthetic-cloud persistence (`time_sigma`) against real cloud statistics before reporting
  final numbers, and widen the region toward the full basin (memory stays flat under streaming).
