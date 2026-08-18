# Copernicus GlobColour plankton gap-fill: what runs, and where it beats persistence

*mind the gap. Running the streaming gap-fill U-Net on Copernicus ocean color, and a careful read of where
it beats standard persistence and where it does not.*

## Summary

We put the Copernicus GlobColour L3 multi 4km daily product through the existing streaming gap-fill U-Net and
worked through the three questions posed for this dataset. The pipeline runs and produces gap-filled
chlorophyll, the land and cloud masks come directly from the product flag, and a whole-globe pass is feasible
and was demonstrated at a coarsened resolution. The main qualification is about evaluation rather than
plumbing: on this dataset the U-Net does not beat standard persistence when the two are scored per pixel, and
the reason is specific to chlorophyll and to the kind of gaps the self-supervised test hides. Standard
persistence here means the previous real observation at a pixel (the previous day, or the previous composite),
not any masked variant.

## Dataset

The store is `cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D` on the fish-pace source.coop repository: a
4km global grid (4320 x 8640) with total chlorophyll `CHL` plus nine phytoplankton functional type variables
(`DIATO`, `GREEN`, `DINO`, `HAPTO`, `NANO`, `PROKAR`, `PICO`, `MICRO`, `PROCHLO`) and a `flags` variable. The
time span currently loaded is 1997-09-04 to 2002-12-30 (1890 daily frames). That period is effectively the
single-sensor SeaWiFS era, so the raw daily fields are dominated by orbital swaths and have large systematic
gaps. The fuller multi-sensor record (post-2002, and especially post-2014) is not yet loaded into a store we
can reach, which matters for the plankton-group work below. Work was done on an Arabian Sea box (about
624 x 912) for comparison with the earlier PACE runs, and globally for the feasibility test.

## Question 1: does it run through the U-Net and gap-fill?

Yes. Notebook 9 opens the store, streams `CHL` through the same U-Net, and produces gap-filled fields scored
by held-out synthetic clouds. One qualification is essential in this era: at daily resolution the ocean is
only about 19 percent observed per frame, which is too sparse for a spatial model to have context, so the
fills default to a smooth mean and the reconstruction is poor. Compositing fixes this. A 9-day composite
raises coverage to about 85 percent, because the swaths rotate and their union fills most gaps, and the target
is smoother. Compositing is the standard way to use the sparse early record, so this is a reasonable step
rather than a shortcut, but it does change the task: a composite has fewer gaps and its persistence baseline
is measured across a 9-day gap rather than a single day.

## Question 2: land and cloud from the flag

Yes, and it is cleaner than the derivation used for PACE. Land is `flags == 1`, taken once because it is
static, and everything that is NaN after removing land is a real cloud or no-data gap, which defines the cloud
mask exactly as specified. This explicit mask is passed into the channel builder everywhere (statistics,
training tiles, and inference), which was also a necessary correctness fix: the PACE pipeline inferred land
from pixels that are NaN across the whole window, and in the swath era a pixel can sit in a gap for an entire
short window, so that heuristic would mislabel persistently cloudy ocean as land. Using the flag removes that
failure mode.

## Question 3: is the whole globe possible?

Yes, and it was demonstrated rather than only argued. The U-Net is fully convolutional and the geo channels
(position as x, y, z on the unit sphere) let one model distinguish regimes, so a single model can gap-fill the
whole domain in one forward pass. A coarsened global run (about 64km, block-mean of the 4km grid, which also
raises coverage) trains one model on full global frames and produces a whole-globe gap-filled field. The limit
at full 4km resolution is storage and read speed, not the model: a global 4km cube is on the order of 100 GB
per variable and the store reads slowly, so a full-resolution global run is a data-transfer problem. The
architecture scales; the data pull is the cost. Coarsening, or the faster multi-sensor data, is the practical
path to a full global product.

## The evaluation caveat: per-frame versus per-pixel

This is the result that most changes the story, so it is worth stating plainly. The same single-variable CHL
model scores differently depending on how the held-out error is aggregated:

| aggregation | U-Net | persistence | reading |
|---|---|---|---|
| mean over frames | 0.227 | 0.309 | model appears to win |
| pooled over pixels | 0.311 | 0.223 | persistence wins |

All numbers are held-out fake-cloud MAE in log Chl-a on 9-day composites. Averaging per frame gives equal
weight to gappy frames, where the smoothing model does relatively well. Pooling per pixel is dominated by the
abundant well-observed open-ocean pixels, where chlorophyll barely changes over 9 days, so persistence copies
a nearly correct real value and the MSE-trained model smooths away real texture. The per-pixel view is the
more defensible one, and under it the U-Net does not beat persistence. The earlier per-frame result that
suggested the model wins was largely an artifact of the averaging.

The underlying reason is structural. The self-supervised test hides an observed pixel and asks the model to
beat a copy of that pixel's previous real value. Chlorophyll is strongly autocorrelated in time, so that copy
is already close, and the comparison is only made where the previous observation exists, which is exactly
where persistence is strongest. Denser data therefore helps persistence, not the model. This matches the
gap-filling literature: temporal methods win for scattered gaps in a persistent field, and spatial methods
earn their value on large contiguous gaps where there is no recent value to copy. The model even receives the
previous composite as an input channel and still loses, which suggests its spatial convolutions smooth away
the temporal signal it is given.

The practical consequence is that the model's genuine value is completing the large holes persistence leaves
empty, not beating it pixel for pixel on easy water. Scored where persistence has no value to offer, the
model fills the swath and cloud interiors (about 0.20 log Chl-a in the daily runs). The fair comparison for
those pixels is climatology, the seasonal and positional mean, rather than persistence, and adding that
comparison is the clearest next step for demonstrating the model's value honestly.

The global run shows the same effect more strongly (U-Net 0.41 versus persistence 0.17 at 64km), because a
coarse, block-averaged, open-ocean-dominated field is even easier for persistence and the single global model
was trained on only a few frames.

## Multivariate plankton groups

The joint model that gap-fills several correlated plankton groups at once is built and correct, but it cannot
be exercised on this store because the functional-type variables are too sparse in the swath era. In the
Arabian Sea box, `CHL` is dense (about 68 percent), but `DIATO` is only about 5.7 percent observed, and its
observations rarely fall in two consecutive composites, so there are zero held-out pixels where both the
current and previous composite are observed. Against persistence, `DIATO` cannot be scored at all here. The
functional-type retrievals are higher-quality-gated and therefore much sparser than total chlorophyll,
especially with a single sensor. The joint design is the right one, since it lets a sparse group borrow from
dense CHL at the same pixel, but proving it needs data with real functional-type coverage, which points to the
fuller multi-sensor or post-2014 record.

An early joint run also filled CHL noticeably worse than the single-variable model, partly because capacity is
shared across outputs and partly because that run undertrained; this is secondary to the coverage problem.

## Limitations and next steps

- The loaded store is 1997-2002 only, which is the swath era. The multi-sensor and post-2014 record is what
  the plankton-group product and a strong global result actually want. The immediate action is to ask the data
  team to ingest the later years, or to pull them from Copernicus Marine directly; the pipeline changes only at
  the loader.
- Add a climatology comparison on the pixels persistence cannot fill, so the model's value on large gaps is
  measured against something rather than reported only where it loses.
- The MSE loss produces the conditional mean and therefore smooths. Beating persistence on well-observed
  pixels would need either a more temporal-aware model or a loss that does not average away structure.
- At full 4km resolution the global run is limited by read speed from the store, not by the model. A faster
  data path is needed before a full-resolution global product is practical.
