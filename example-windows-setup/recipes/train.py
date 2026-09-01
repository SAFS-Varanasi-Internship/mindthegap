#!/usr/bin/env python3
"""PACE gap-fill training on a SkyPilot VM (mirrors the 2-Train notebook).

Loads PACE chlorophyll via NASA Earthdata, trains the mindthegap U-Net on a GPU,
checkpoints the best model, saves the model bundle, and saves the built-in
gap-fill graphs for a few held-out days -- all under outputs/, which the
sync-outputs task pushes to the HF bucket.

Current config: Arabian Sea, one year, with 5 prev + 5 next days of temporal
context (n_temporal_lags=5), to check whether the tiling/checkerboard artifacts
persist under the resize-conv model with more temporal context. Judge from the
gapfill_*.png graphs in the bucket.
"""
from pathlib import Path
import os          # EDIT (item 2): env + checkpoint path handling for resume
import subprocess  # EDIT (item 2): drive `hf buckets` for checkpoint sync/pull

import matplotlib
matplotlib.use("Agg")  # headless VM: no display, just save figures
import matplotlib.pyplot as plt
import pandas as pd
import keras       # EDIT (item 2): base class for the checkpoint-sync callback

import earthaccess
import mindthegap as mtg
from mindthegap import viz

SMOKE = False                                    # full training run to convergence
REGION = "arabian sea"                           # small coastal/dynamic region for the artifact test
TIME_SLICE = slice("2024-03-01", "2025-03-01")   # one year; PACE is 2024+
N_TEMPORAL_LAGS = 5                              # 5 days before + 5 after as input channels
RESUME = False   # EDIT (item 2): True resumes from a checkpoint in the bucket; off by default
EPOCHS = None    # EDIT (item 2): override options.fit.epochs (None = package default); handy for testing

BUCKET = os.environ.get("HF_BUCKET", "")  # EDIT (item 2)


# EDIT (item 2): after each epoch, record the last-completed epoch and push outputs/
# to the bucket. `hf buckets sync` only uploads changed files, so the big checkpoint
# uploads only when it improved. This lets a preempted run resume (with RESUME=True).
class BucketCheckpointSync(keras.callbacks.Callback):
    def __init__(self, epoch_marker, bucket):
        super().__init__()
        self.epoch_marker = Path(epoch_marker)
        self.bucket = bucket

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_marker.write_text(str(epoch + 1))
        if self.bucket:
            subprocess.run(
                ["hf", "buckets", "sync", "outputs", f"{self.bucket}/outputs"],
                check=False,
            )


# Earthdata login from env vars (SkyPilot injects EARTHDATA_USERNAME/PASSWORD).
earthaccess.login(strategy="environment")

options = mtg.set_up(smoke_test=SMOKE)
ds, target, missing_flag, land_flag = mtg.demo_data(
    "pace", region=REGION, time_slice=TIME_SLICE,
)
options.set_up_data_options(
    ds,
    target=target,
    missing_flag=missing_flag,
    land_flag=land_flag,
    log_target=True,                  # chlorophyll is 0-bounded, so log it
    add_geo=True,                     # spherical x/y/z position channels
    n_temporal_lags=N_TEMPORAL_LAGS,  # 5 prev + 5 next days of context
)
mtg.set_up_train_split_options(ds, options, split_mode="random")
# apply=True applies the recommendation without the [y/N] prompt (headless VM).
mtg.set_up_gridder_options(ds, options, apply=True)
if EPOCHS is not None:  # EDIT (item 2): override epoch count (testing / short runs)
    options.fit.epochs = EPOCHS

outdir = Path("outputs")
outdir.mkdir(parents=True, exist_ok=True)
ckpt_path = outdir / "pace_checkpoint.keras"
epoch_marker = outdir / "pace_checkpoint.epoch"  # EDIT (item 2): last-completed-epoch marker

# EDIT (item 2): resume support. Off unless RESUME=True, so we never accidentally
# resume a model we did not mean to. When on, pull the checkpoint + epoch marker
# from the bucket; if present, continue from where the last run stopped.
resume_from = None
initial_epoch = 0
if RESUME and BUCKET:
    subprocess.run(
        ["hf", "buckets", "cp", f"{BUCKET}/outputs/pace_checkpoint.keras", str(ckpt_path)],
        check=False,
    )
    subprocess.run(
        ["hf", "buckets", "cp", f"{BUCKET}/outputs/pace_checkpoint.epoch", str(epoch_marker)],
        check=False,
    )
    if ckpt_path.exists():
        resume_from = str(ckpt_path)
        initial_epoch = (
            int(epoch_marker.read_text().strip()) if epoch_marker.exists() else 0
        )
        print(f"RESUME: found checkpoint, continuing from epoch {initial_epoch}")
    else:
        print("RESUME: no checkpoint in the bucket; starting fresh")

result = mtg.train_model(
    ds,
    options,
    load_data="auto",
    checkpoint_path=str(ckpt_path),
    resume_from=resume_from,       # EDIT (item 2)
    initial_epoch=initial_epoch,   # EDIT (item 2)
    extra_callbacks=[BucketCheckpointSync(epoch_marker, BUCKET)],  # EDIT (item 2): periodic bucket sync
)
model = result.model
print(result.summary())

mtg.save_model_bundle(
    result,
    outdir / "pace-unet-bundle",
    limitations="Arabian Sea, one year, n_temporal_lags=5; research artifact test.",
    overwrite=True,
)
print("saved checkpoint + bundle under outputs/")

# FULL gap-fill graphs: fill the ENTIRE region and save observed vs gap-filled.
# mode="gapfill" relabels the real gaps as the thing to estimate, so the model
# fills the whole field (not just synthetic test clouds, which is what mode="test"
# would show). A spread of days across the year so artifacts (if any) are easy to
# spot. This mirrors the 3-Gapfill notebook (prepare_model_data mode="gapfill" ->
# gapfill_std -> observed-vs-gapfilled plot).
all_dates = [str(pd.to_datetime(t).date()) for t in ds.time.values]
mid = all_dates[5:-5]  # skip the lag-edge days (n_temporal_lags=5 has no full lags there)
step = max(1, len(mid) // 6)
check_dates = mid[::step][:6]  # ~6 days across the year

ds_gap = mtg.prepare_model_data(ds, options, mode="gapfill")

for d in check_dates:
    try:
        gapfilled = mtg.gapfill_std(ds_gap, model, options, time=d)
        pred = gapfilled["gapfilled_target"].isel(time=0)
        obs = ds_gap["observed_target"].sel(time=d)
        land = ds_gap["land_flag"].sel(time=d)
        obs = obs.where(land == 0)
        pred = pred.where(land == 0)
        lo = float(min(float(obs.min()), float(pred.min())))
        hi = float(max(float(obs.max()), float(pred.max())))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        obs.plot(ax=axes[0], vmin=lo, vmax=hi)
        axes[0].set_title(f"Observed (standardized): {d}")
        pred.plot(ax=axes[1], vmin=lo, vmax=hi)
        axes[1].set_title(f"Gap-filled entire region: {d}")
        fig.savefig(outdir / f"gapfill_{d}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("saved full gap-fill graph:", d)
    except Exception as exc:
        print(f"skipped gap-fill graph for {d}: {exc}")

print("done: bundle + full gap-fill graphs under outputs/ (synced to the bucket)")
