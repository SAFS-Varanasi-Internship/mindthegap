# Running PACE gap-fill on SkyPilot (eScience CloudBank): setup runbook

How the PACE gap-fill training pipeline was put on SkyPilot,
including every caveat and hiccup and how each was worked around. Written so the
setup is reproducible and so the failure modes are on record.

Environment on the driving side: Windows laptop, running the SkyPilot client
inside WSL (Ubuntu, linux-64). The repo lives at
`/home/truss/projects/skypilot-accelerator-mindthegap`. The API server is the
shared eScience CloudBank controller at
`cloudbank-skypilot.westus2.cloudapp.azure.com`. Compute VMs run in AWS
us-west-2.

--------------------------------------------------------------------------------

## 0. Starting point (already working before this)

The fork of `uw-escience-cloudbank/skypilot-accelerator-mindthegap` had a working
CPU example: `recipes/example.py` reads a public GlobColour Zarr over http and
saves a PNG; `recipes/example.yaml` runs it on a CPU VM; the output syncs to a
Hugging Face bucket. `pixi run local-workflow` and `pixi run remote-workflow` had
both been validated end to end. Login, bucket, and a full cloud round trip
already worked. Everything below is a variation on that example, which is how the
repo is meant to be extended.

### The two-environment pixi design (important, do not merge them)

`pixi.toml` defines two environments on purpose:
- `default`: just the SkyPilot client. Used locally to submit jobs.
- `runtime`: the science stack plus the `hf` CLI. Installed on the VM.

They are kept apart because SkyPilot pins `click <8.2` while `huggingface_hub`
wants `click >=8.4`; a single environment forces a solve where the `hf` CLI
crashes at exit. `runtime` is declared `no-default-feature = true` so it does not
drag the SkyPilot client (and its click cap) back in.

### Credentials and `.env`

`load-env.sh` runs on pixi activation. Locally it sources `.env`; on a SkyPilot
VM it self-disables (guarded on `SKYPILOT_NUM_NODES`), because SkyPilot injects
the env vars itself from the YAML `envs:` block plus `--env-file .env` at launch.
`.env` holds:
- `HF_TOKEN`, `HF_BUCKET` (the HF bucket, e.g. `hf://buckets/TroyRusso/varanasi`, no trailing slash)
- `SKYPILOT_API_*` (endpoint + credentials for the controller)
- `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` (added later, for PACE; see phase 3)

### The workflow pattern

Each recipe is a pair plus a pixi task:
- `setup:` in the YAML installs pixi, then `pixi install -e runtime`, then (for
  training) pip-installs mindthegap.
- `run:` calls `pixi run -e runtime <name>-workflow`, a pixi task that chains the
  compute task and then `sync-outputs`.
- `sync-outputs` = `hf buckets sync outputs $HF_BUCKET/outputs`, so anything the
  script writes under `outputs/` lands in the bucket.
- `sky jobs launch` (managed jobs) auto-recovers from spot preemption and tears
  the worker VM down on completion.

--------------------------------------------------------------------------------

## 1. Prediction-only smoke test (cheapest first real workload)

Goal: prove the whole chain with a real model, cheaply, before spending on
training. Suggested by Valentina.

Steps:
1. Uploaded a pretrained model to a private HF model repo:
   `hf upload TroyRusso/mindthegap-arabsea-demo <local>/UNet_DoubleConv_mse.keras UNet_DoubleConv_mse.keras --repo-type model --private`
   (the model file is visible from WSL at `/mnt/c/vscode/mindthegap/models/...`).
2. Wrote `recipes/predict.py`: `hf_hub_download` the model, `keras.models.load_model(compile=False)`,
   predict on a deterministic synthetic `(1,H,W,C)` input, save a PNG to `outputs/`.
3. Wrote `recipes/predict.yaml` (CPU only) and added `tensorflow` to the runtime
   deps plus `predict` / `predict-workflow` / `remote-predict` tasks.
4. Validated locally (`pixi run -e runtime predict-workflow`), then remotely
   (`pixi run remote-predict`).

Result: the remote CPU job SUCCEEDED and the cloud prediction values were
identical to the local run (deterministic input), which confirmed environment
parity. The bucket received the PNG from the VM.

--------------------------------------------------------------------------------

## 2. Training recipe and dependency integration

Checked the upstream repo first: it is the same as the fork (only the CPU
example, no training recipe), so the training recipe is ours to build.

Files added:
- `recipes/train.py`: `earthaccess.login(strategy="environment")`, `mtg.demo_data("pace", ...)`,
  `set_up_data_options`, `set_up_train_split_options`, `set_up_gridder_options(apply=True)`,
  `train_model(load_data="auto", checkpoint_path=...)`, `save_model_bundle`, then the graphs.
- `recipes/train.yaml`: GPU, `envs:` including the `EARTHDATA_*` and `HF_*`,
  `setup:` installs the conda stack then pip-installs mindthegap.
- `pixi.toml`: added conda deps and `train` / `train-workflow` / `remote-train` tasks.

### Dependency caveat (this bit is fiddly)

mindthegap's own declared dependencies are tiny (`PyYAML`, `scipy`). Its heavy
imports (tensorflow, xbatcher, earthaccess, cartopy) are expected from the
environment. Per the repo's own guidance, conda-available packages go in the
runtime deps table and pip/git packages go in the YAML `setup:` block. So:
- conda runtime deps added: `tensorflow`, `xbatcher`, `earthaccess`, `scipy`, `pip`, `cartopy`.
- `setup:` runs `pixi run -e runtime python -m pip install "git+https://github.com/SAFS-Varanasi-Internship/mindthegap.git"`
  (main branch, no pinned commit, so each fresh VM pulls main HEAD).

Hiccup: the first local import of mindthegap failed with
`ModuleNotFoundError: No module named 'cartopy'`, because `mindthegap.viz` imports
cartopy at package import time. Fix: add `cartopy` to the conda runtime deps. A
local `pixi install -e runtime` + pip install + `import` then reported
`STACK OK 0.2.x`.

--------------------------------------------------------------------------------

## 3. Data access: the us-west-2 constraint

A local run of `train.py` failed at the data read, not in any of our code:

```
icechunk.StorageError: AccessDenied ... assumed-role/s3-same-region-access-role/troyrusso
is not authorized to perform s3:GetObject on ob-cumulus-prod-public/PACE_OCI....nc
```

The NASA temporary S3 role is literally `s3-same-region-access-role`: PACE data
is readable only from the same AWS region as the bucket (us-west-2). A laptop is
denied by policy. This is not a bug; it is why PACE work must run in us-west-2.
Everything upstream of the read (Earthdata login, dataset metadata, channel
build, gridder) ran fine locally, which was enough to validate the code. The VM
(us-west-2) reads it without issue.

Earthdata credentials reach the VM the same way `HF_TOKEN` does: `EARTHDATA_USERNAME`
and `EARTHDATA_PASSWORD` in `.env`, declared in the YAML `envs:` block, read by
`earthaccess.login(strategy="environment")`.

--------------------------------------------------------------------------------

## 4. GPU tensorflow: the CUDA-build fix (the big one)

First cloud training run succeeded but the log showed
`Compute device: CPU / GPUs available: 0`: TensorFlow ran on the CPU even on a T4,
because conda-forge `tensorflow` resolves to a CPU-only build by default. The GPU
was paid for and unused.

Fix attempt 1 (failed): added `[feature.runtime.system-requirements] cuda = "12"`
and pinned `tensorflow = { build = "cuda*" }`. The local solve failed:

```
failed to solve environment 'runtime' for platform 'osx-arm64'
No candidates were found for tensorflow >=2.16,<3 cuda*.
```

Two problems: the workspace also targeted `osx-arm64`, where a CUDA tensorflow
build cannot exist; and `[system-requirements]` is deprecated. Both surfaced
locally and cost nothing (the solve fails before any VM launch).

Fix that worked: target linux-64 only and declare CUDA on the platform.
- In `[workspace]`: `platforms = [{ platform = "linux-64", cuda = "12" }]`
  (dropped osx-arm64, since the client is WSL/linux-64 and the VMs are linux-64).
- Keep `tensorflow = { version = ">=2.16,<3", build = "cuda*" }`.
- Remove the `[feature.runtime.system-requirements]` table.

Verified on a T4 smoke run: `Compute device: GPU / GPUs available: 1`, and steps
went from ~10 s/step (CPU) to ~460 ms/step (GPU), about 20x.

--------------------------------------------------------------------------------

## 5. Spot capacity: the on-demand fallback

Scaling to the whole Indian Ocean used an A10G with `memory: 60+` (to fit the
year in RAM). The A10G spot job sat `PENDING` for about four hours with zero
instances acquired: spot capacity for that instance in us-west-2 was exhausted,
made worse by the large-RAM instance the memory request forced (g5.4xlarge). It
costs nothing while pending, but it never starts.

Fix: `use_spot: false` (on-demand). It provisioned immediately, and as a bonus
on-demand removes the preemption-restart risk on a long run. T4 spot, by
contrast, was always available in seconds, so for small runs T4 spot is fine.

The whole-IO run then SUCCEEDED (696x1000 cropped, gridder tiled into 3 x
232x1000 tiles, 12 channels, converged in 36 epochs). Note: its low `val_loss`
is the full-target reproduction MSE, not a gap-fill-skill number versus
persistence.

--------------------------------------------------------------------------------

## Recurring gotchas (driving SkyPilot from Windows/WSL)

- Running WSL commands from Windows/PowerShell is quoting-hostile. What works:
  `wsl -u truss bash -lc '<command>'` with the whole command in PowerShell single
  quotes and no single quotes inside. Call pixi by full path
  `$HOME/.pixi/bin/pixi`; do NOT `export PATH=...:$PATH`, because the inherited
  Windows PATH contains `Program Files (x86)` and the unquoted parenthesis breaks
  bash parsing. For anything with inner quotes or control flow (loops, ifs), write
  a `.sh` file and run it, rather than fighting inline quoting.
- Use `wsl -u truss` explicitly; the default invocation did not resolve `~` to the
  right home.
- `hf buckets ls` is slow to start (tens of seconds), sometimes past a 2-minute
  tool timeout. It is fine; just let it run.
- The job log's "Useful Commands" block prints a dashboard URL with the SkyPilot
  API password in plaintext. Do not share those logs, and rotate that password,
  since it has appeared in logs.
- `cache_locally` (the opt-in local read cache) on PACE needs a rechunk first:
  the icechunk read comes back with uneven lon chunks that Zarr rejects; do
  `ds = ds.chunk({"time": 1, "lat": -1, "lon": -1})` before caching.

--------------------------------------------------------------------------------

## Current file layout (what these changes added)

- `recipes/predict.py`, `recipes/predict.yaml`: CPU prediction smoke test.
- `recipes/train.py`, `recipes/train.yaml`: PACE training (currently Arabian Sea,
  one year, `n_temporal_lags=5`, full-region gap-fill graphs) on a T4.
- `pixi.toml`: two-env layout plus the added conda deps (tensorflow, xbatcher,
  earthaccess, scipy, pip, cartopy), the linux-64 + cuda platform, and the
  predict/train tasks.
- `watch_and_launch.sh` (in the scratchpad, not the repo): the health-poll loop.

## How to run it

```
# from the repo, in WSL:
pixi run -e runtime predict-workflow   # local prediction check
pixi run remote-predict                # prediction on a CPU VM
pixi run -e runtime train-workflow     # local train check (fails at the PACE read off-region)
pixi run remote-train                  # training on a GPU VM (add -y to skip the confirm)
pixi run sky jobs queue                # job status
pixi run -e runtime hf buckets ls TroyRusso/varanasi/outputs   # outputs
```