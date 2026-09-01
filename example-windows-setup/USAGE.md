# Using the PACE gap-fill SkyPilot flow

Practical steps to run a training job, resume one that died, and collect the
outputs. For how the pipeline was built and its failure modes, see
[README.md](README.md). All commands run from the repo root inside WSL.

## One-time setup

Covered in the README. In short: `pixi` installed, `.env` filled in from
`.env.example`, and the SkyPilot client logged in to the eScience controller.
Sanity check with:

```
pixi run sky check
```

## Run a training job

```
pixi run remote-train -y
```

This launches a managed GPU job that logs in to Earthdata, loads PACE
chlorophyll for the configured region and dates, trains the U-Net, and syncs the
checkpoint, the model bundle, and full-region gap-fill graphs to your HF bucket
under `outputs/`. The worker VM tears down on completion (or after 30 idle
minutes). Drop the `-y` to see the resource and cost estimate first.

## Configure the run

Edit the knobs at the top of `recipes/train.py`:

| knob              | what it controls                                             |
|-------------------|-------------------------------------------------------------|
| `REGION`          | named region, e.g. `"arabian sea"`                          |
| `TIME_SLICE`      | date range, e.g. `slice("2024-03-01", "2025-03-01")`        |
| `N_TEMPORAL_LAGS` | days of temporal context before and after each frame        |
| `EPOCHS`          | override the epoch count (`None` uses the package default)  |
| `RESUME`          | resume from a bucket checkpoint (see below); default `False`|

GPU type and spot vs on-demand live in `recipes/train.yaml`
(`accelerators: T4:1`, `use_spot: true`). For a larger or longer run use a
bigger GPU (`A10G:1`, `L40S:1`) and `use_spot: false`; see the README's
spot-capacity note for why on-demand is safer there.

## Monitor a run

```
pixi run sky jobs queue -a                              # status + duration, all jobs
pixi run sky jobs logs <job-id>                         # stream or replay a job log
pixi run -e runtime hf buckets ls <user>/<bucket>/outputs   # what has landed
```

Every epoch syncs to the bucket, so you can watch progress from the bucket even
if the API connection drops. If a launch's log stream dies (the eScience API
blips), the job keeps running on the controller; check `sky jobs queue -a`
rather than trusting the dead stream.

## Resume a run that died

Managed jobs auto-recover from a single spot preemption on their own. For
anything that leaves the job actually dead (repeated preemptions, a crash, a
timeout, a torn-down VM), resume from the last checkpoint in the bucket:

1. Set `RESUME = True` in `recipes/train.py`.
2. Relaunch: `pixi run remote-train -y`.

On startup it pulls `pace_checkpoint.keras` and the epoch marker from the bucket
with `hf buckets cp`, loads the model with its optimizer state, and continues
from the epoch it reached. The log shows
`Resuming from checkpoint ... at epoch N` and Keras picks up at `Epoch N+1`. If
the bucket has no checkpoint, it says so and starts fresh.

Set `RESUME` back to `False` for normal runs, so you never resume a model you did
not mean to.

Note: the checkpoint is the best epoch so far (`save_best_only`), so a resume
continues from the best weights at the current epoch count, not a byte-exact
snapshot of the instant it died. That is a correct continue for training. If you
need exact preemption recovery, switch the checkpoint to `save_best_only=False`
in the recipe.

## Collect the outputs

Everything a run produces is under `outputs/` in your bucket:

- `pace_checkpoint.keras`: the trained model (best epoch)
- `pace-unet-bundle/`: the mindthegap model bundle, for `load_model_bundle`
- `gapfill_<date>.png`: full-region observed-vs-gap-filled graphs

Pull one down with:

```
pixi run -e runtime hf buckets cp hf://buckets/<user>/<bucket>/outputs/pace_checkpoint.keras ./pace_checkpoint.keras
```
