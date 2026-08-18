# Update: loss on held-out pixels, and the persistence check

*Reply to Eli. What the loss change did, and a few questions on where to take it.*

## What I changed

The masked loss now scores only the held-out fake-cloud pixels (`fake` and not real-cloud and not land),
instead of every observed ocean pixel. So the gradient is only on the pixels the model actually has to
predict, not on the visible pixels it can copy from the input.

## Results

Data: daily GlobColour ocean color from the source.coop store, Arabian Sea box (`COMPOSITE_DAYS = 1`). The
store now covers 1997-09-04 to 2010-07-31 (4660 daily frames), so these results span both the early
single-sensor swath years and the denser multi-sensor era after 2002. Held-out fake-cloud MAE in log Chl-a,
scored per pixel (pooled over all validation frames). Persistence here is the previous day's real observed
value.

The loss change, on validation:

| | U-Net | persistence |
|---|---|---|
| before (loss on all observed ocean) | 0.311 | 0.223 |
| after (loss on held-out pixels only) | 0.205 | 0.196 |

After the change, train versus validation (your check):

| | U-Net | persistence |
|---|---|---|
| train | 0.205 | 0.193 |
| val | 0.205 | 0.196 |

So the model went from about 40 percent worse than persistence to roughly tied. It also fills the roughly 16
percent of held-out pixels where the previous composite is missing, which persistence cannot predict at all,
at about 0.20, which is about the same as its error everywhere else. So the loss change did what you expected:
the model now leans on the yesterday feature about as heavily as it can, and it still covers the pixels where
yesterday does not exist.

I also ran a cloud-size check, training and testing at several synthetic cloud sizes (`blob_sigma`, in pixels
at 4 km) and comparing to persistence and to a trivial nearest-neighbor fill, where each hidden pixel copies
its closest observed pixel. Held-out MAE in log Chl-a, pooled:

| cloud size | U-Net | persistence | nearest-neighbor |
|---|---|---|---|
| 4 | 0.12 | 0.195 | 0.115 |
| 8 | 0.25 | 0.195 | 0.16 |
| 12 | 0.25 | 0.195 | 0.195 |
| 20 | 0.32 | 0.194 | 0.25 |

Persistence is roughly flat across cloud size, which fits since it is a temporal predictor and does not care
how large the spatial gap is. The U-Net and nearest-neighbor both fall off as the gaps grow. The U-Net comes
out ahead of persistence only at the smallest size, 4, and even there the nearest-neighbor fill matches or
beats it, so a trivial spatial fill is doing at least as well as the model everywhere.

This also explains why the main comparison above shows a near-tie rather than a win: that comparison is run at
cloud size 12, which is the size-12 row here, where the U-Net is already tied to slightly behind persistence.
The apparent win only shows up at size 4. (The per-size models are quick separate trainings, so the
larger-gap U-Net numbers are noisier than the standalone run above, which reached 0.205 at size 12.)

## Questions

1. On the training-data check: persistence still edges the model slightly even on the training frames (0.193
   versus 0.205). You said that should not happen, but you also said a small gap is fine as long as it is not
   way worse, because the model also has to fill the days where yesterday is missing. We are now tied rather
   than way worse. Do you consider this resolved, or do you still think something is off and want me to dig
   into why the model does not fully beat persistence on the training data?

2. One thing I noticed that might be the cause, if you do think something is still off: the previous-day and
   next-day feature channels are built from the fake-clouded field, so at the held-out pixels the model's
   yesterday is usually also masked out. The model does not have yesterday's value at exactly the pixels where
   it would help, while the persistence baseline uses the real, unmasked yesterday. Is that expected, or
   should the neighbor channels use the real previous day? I have it as a toggle and can test it either way.

3. These results are daily over the full 1997 to 2010 record, which mixes the sparse early swath years with
   the denser years after 2002. If you would rather see a cleaner controlled slice, or the most recent data,
   I have a copernicusmarine loader wired up to pull a specific window, for example 2023 to 2024. Do you want
   me to run that next, and is a one-year or two-year window the right amount to start with?

## Status and a blocker

The two runs I want to do next, the realistic large-cloud test and a clean run on the recent dense data, are
currently blocked by the Hub environment rather than by the method. The home directory reports about 128 GB
free, but every write fails with a no-space error, so the cached data cube cannot be saved to disk. Each
kernel restart therefore reloads the full multi-year daily cube, which is about 10 GB, and that reload plus
training repeatedly runs the kernel out of memory. If the disk quota can be raised or reset, I can pull a
recent one to two year window, cache it once, and run the large-cloud test cleanly. Until then, the results
above are where things stand.

For the large-cloud test specifically, the measurement above showed real gaps are about five times larger
than the synthetic clouds used so far, and the ablation shows the model getting worse as gaps grow while
persistence stays flat, so the expected result is that the model trails persistence on realistic gaps. Its
value is on the gap interiors where neither persistence nor a nearby-pixel fill can help. I will confirm that
with a proper run once the environment allows it.
