# PACE gap-fill smoke test: train, then evaluate

A start-to-finish smoke test for a new machine: set up the tools and accounts,
run a short training job on a GPU in the cloud, and check that it learned by
looking at the built-in gap-fill evaluation it writes to your bucket. It is
deliberately small (about 10 to 15 minutes and a few cents), so the point is to
prove the whole chain works, not to train a good model.

Instructions are given for both macOS and Windows. Where a step differs, the
macOS line comes first and the Windows line second. Every `pixi run ...` command
is identical on both; only the shell differs (macOS runs them in Terminal,
Windows runs them inside a WSL Ubuntu terminal).

For how the pipeline was built and its failure modes, see [README.md](README.md).
For day-to-day operation (resume, monitoring, all the knobs), see
[USAGE.md](USAGE.md).

--------------------------------------------------------------------------------

## 1. Prerequisites 

### Accounts

- **Hugging Face**: create an account at https://huggingface.co, then make an
  access token with write permission (Settings, Access Tokens). You will also
  need a bucket to hold outputs; you can create one after the tools are installed
  with `pixi run -e runtime hf buckets create <name>`, which gives you a bucket
  path like `hf://buckets/<user>/<name>`.
- **NASA Earthdata**: create an account at https://urs.earthdata.nasa.gov. This
  is what lets the training VM read PACE data.
- **SkyPilot controller (eScience CloudBank)**: Scott sent a login to the
  shared controller. That login is what gives you cloud compute; you do not need
  your own AWS account, and the controller already has the AWS quota. You will
  get an endpoint, a username, and a password.

### Tools

- **Windows only, first**: install WSL, then do everything below inside the WSL
  Ubuntu terminal (not PowerShell).
  - Windows: open PowerShell as admin, run `wsl --install`, reboot, and finish the
    Ubuntu first-run setup. From then on, open "Ubuntu" and work there.
  - macOS: nothing extra; use Terminal.
- **pixi** (both platforms), then restart the shell:
  - macOS: `curl -fsSL https://pixi.sh/install.sh | bash`
  - Windows (in WSL): `curl -fsSL https://pixi.sh/install.sh | bash`
- **The repo**:
  - macOS: `git clone https://github.com/SAFS-Varanasi-Internship/mindthegap.git`
  - Windows (in WSL): `git clone https://github.com/SAFS-Varanasi-Internship/mindthegap.git`
  - Then, both: `cd mindthegap/example-windows-setup`

### Credentials file

Copy the template and fill it in with the values from the accounts above:

```
cp .env.example .env
```

Edit `.env` and set `HF_TOKEN`, `HF_BUCKET` (e.g. `hf://buckets/<user>/<name>`,
no trailing slash), `SKYPILOT_API_ENDPOINT`, `SKYPILOT_API_USER`,
`SKYPILOT_API_PASSWORD`, `EARTHDATA_USERNAME`, and `EARTHDATA_PASSWORD`. `.env` is
gitignored; never commit it.

--------------------------------------------------------------------------------

## 2. Verify the setup

Log the SkyPilot client into the controller, then confirm it can see cloud
compute. Same commands on both platforms (macOS in Terminal, Windows in WSL):

```
pixi run login
pixi run check
```

The first `pixi run` will install the client environment (a minute or two). Then
launch a trivial managed job to prove your access and quota end to end:

```
pixi run test
```

This submits a tiny CPU job that just echoes "hello" and tears itself down. When
it reports SUCCEEDED, your access works. If it sits in `PENDING`, see the
spot-capacity note in the README.

Note: pixi prints a one-line deprecation warning about `[system-requirements]`
when it reads the manifest. It is harmless; the environment still builds
correctly.

--------------------------------------------------------------------------------

## 3. Run the training smoke test

Open `recipes/train.py` and set a small, fast configuration near the top:

```python
REGION = "arabian sea"                          # small region
TIME_SLICE = slice("2024-06-01", "2024-07-01")  # one month, quick
EPOCHS = 3                                       # a few epochs, enough to see it learn
```

Then launch it (same on both platforms):

```
pixi run remote-train -y
```

This starts a managed GPU job that logs in to Earthdata, loads that month of
PACE chlorophyll, trains the U-Net for a few epochs, and after each epoch syncs
the checkpoint, the model bundle, and full-region gap-fill graphs to your bucket
under `outputs/`. It tears the VM down when done. Expect roughly 10 to 15
minutes, most of which is environment setup on the VM.

If it sits in `PENDING` waiting for a spot GPU, open `recipes/train.yaml` and set
`use_spot: false` (on-demand), then relaunch; the README explains why.

--------------------------------------------------------------------------------

## 4. Evaluate the result

The training run evaluates itself, so there is no separate evaluation job. Two
things to look at.

**The held-out metric, in the job log.** Training reports `val_fakecloud_mse` and
`val_fakecloud_mae` each epoch. These are measured only under synthetic clouds
(pixels hidden from the model at training time), so they are a real gap-fill
skill number, not just reconstruction of what the model already saw. They should
fall over the few epochs. Stream or replay the log with:

```
pixi run sky jobs queue -a          # find the job id and status
pixi run sky jobs logs <job-id>     # read the epoch-by-epoch metrics
```

**The gap-fill graphs, in your bucket.** The run writes `gapfill_<date>.png`
files that put the observed field next to the model's full-region gap-filled
field. For a smoke test, "the two panels broadly resemble each other and the
fill is smooth" is a pass; do not expect a polished model from three epochs. List
and pull them (same on both platforms):

```
pixi run -e runtime hf buckets ls <user>/<name>/outputs
pixi run -e runtime hf buckets cp hf://buckets/<user>/<name>/outputs/gapfill_2024-06-15.png ./gapfill.png
```

Open the downloaded PNG (macOS: `open gapfill.png`; Windows in WSL:
`explorer.exe gapfill.png`).

If the metric fell and the graphs look sensible, the whole chain works:
accounts, client, controller, GPU VM, PACE read, training, evaluation, and the
bucket sync.
