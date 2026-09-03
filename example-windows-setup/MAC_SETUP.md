# macOS setup for the PACE gap-fill smoke test

**Validation note.** This exact command flow (install, login, `sky check`,
`sky jobs queue`, `hf buckets ls` and `cp`) was tested on Linux, where `sky` and
`hf` are the same CLIs as on macOS, and it worked. The macOS-specific pieces
(installing Python, `open` to view a file) are standard, but a run on an actual
Mac has not been done yet, so treat the first one as a shakedown and say so if
anything differs.

--------------------------------------------------------------------------------

## 1. Prerequisites (do these once)

### Accounts

- **Hugging Face**: create an account at https://huggingface.co, then make an
  access token with write permission (Settings, Access Tokens). You also need a
  bucket to hold outputs. You can create one in the browser, or later from the
  `hf` venv in step 5 with `hf buckets create <name>`. Either way you end up with
  a bucket path like `hf://buckets/<user>/<name>`.
- **NASA Earthdata**: create an account at https://urs.earthdata.nasa.gov. This
  is what lets the training VM read PACE data.
- **SkyPilot controller (eScience CloudBank)**: Scott sent an email with a login to the
  shared controller. That login is what gives you cloud compute; you do not need
  your own AWS account, and the controller already has the AWS quota. You will
  get an endpoint, a username, and a password.

### Tools

- **Python 3**: macOS ships one; check with `python3 --version`. If it is
  missing, `brew install python`.
- **git**: `git --version` will offer to install the command line tools if it is
  not already there.

### Clone the repo and fill in credentials

```
git clone https://github.com/SAFS-Varanasi-Internship/mindthegap.git
cd mindthegap/example-windows-setup
cp .env.example .env
```

Edit `.env` and set (do not put quotations around any of these values):
- `SKYPILOT_API_ENDPOINT`: cloudbank-skypilot.westus2.cloudapp.azure.com. no https:
- `HF_TOKEN`: your Hugging Face write token.
- `HF_BUCKET`: your bucket path, e.g. `hf://buckets/<user>/<name>`, no trailing slash.
- `SKYPILOT_API_ENDPOINT`, `SKYPILOT_API_USER`, `SKYPILOT_API_PASSWORD`: from Scott.
- `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`: your Earthdata login.

`.env` is gitignored; never commit it.

--------------------------------------------------------------------------------

## 2. Two small virtualenvs (and why two)

SkyPilot pins `click <8.2` while the Hugging Face CLI wants `click >=8.4`. Put
both in one environment and the `hf` command crashes on exit. So use one venv for
`sky` and a separate one for `hf`. This mirrors the two pixi environments the
Linux and Windows setup uses.

SkyPilot client venv:

```
python3 -m venv ~/sky-venv
source ~/sky-venv/bin/activate
pip install "skypilot[aws]>=0.13,<0.14"
deactivate
```

Hugging Face CLI venv:

```
python3 -m venv ~/hf-venv
source ~/hf-venv/bin/activate
pip install "huggingface_hub>=1.16"
deactivate
```

--------------------------------------------------------------------------------

## 3. Log in and verify (sky venv)

Load your credentials into the shell, then log the client into the controller.
`set -a; source .env; set +a` exports every value in `.env` for the commands that
follow, so keep it in the same terminal session.

```
source ~/sky-venv/bin/activate
set -a; source .env; set +a
sky api login -e "https://$SKYPILOT_API_USER:$SKYPILOT_API_PASSWORD@$SKYPILOT_API_ENDPOINT"
sky check
```

Prove access and quota with a trivial job:

```
sky jobs launch -n demo-test --cpus 1+ -y -- echo hello
```

When it reports SUCCEEDED, your access works.

--------------------------------------------------------------------------------

## 4. Run the training smoke test (sky venv)

Edit `recipes/train.py` and set a small, fast config near the top:

```python
REGION = "arabian sea"                          # small region
TIME_SLICE = slice("2024-06-01", "2024-07-01")  # one month, quick
EPOCHS = 3                                       # a few epochs, enough to see it learn
```

Launch it:

```
sky jobs launch recipes/train.yaml --env-file .env -y
```

This starts a managed GPU job that logs in to Earthdata, loads that month of PACE
chlorophyll, trains the U-Net for a few epochs, and after each epoch syncs the
checkpoint, the model bundle, and full-region gap-fill graphs to your bucket
under `outputs/`. It tears the VM down when done. Expect roughly 10 to 15
minutes, most of which is environment setup on the VM.

Monitor it:

```
sky jobs queue -a          # find the job id and status
sky jobs logs <job-id>     # read the epoch-by-epoch metrics
```

If the job sits in `PENDING` waiting for a spot GPU, open `recipes/train.yaml`,
set `use_spot: false` (on-demand), and relaunch.

Watch jobs here: https://cloudbank-skypilot.westus2.cloudapp.azure.com/dashboard/jobs

--------------------------------------------------------------------------------

## 5. Evaluate the result

The training run evaluates itself, so there is no separate evaluation job. Two
things to look at.

**The held-out metric, in the job log.** Training reports `val_fakecloud_mse` and
`val_fakecloud_mae` each epoch. These are measured only under synthetic clouds
(pixels hidden from the model at training time), so they are a real gap-fill
skill number, not just reconstruction of what the model already saw. They should
fall over the few epochs. You already saw them with `sky jobs logs <job-id>` in
step 4.

**The gap-fill graphs, in your bucket.** The run writes `gapfill_<date>.png` files
that put the observed field next to the model's full-region gap-filled field. For
a smoke test, "the two panels broadly resemble each other and the fill is smooth"
is a pass; do not expect a polished model from three epochs.

Get them with the `hf` venv (the `sky` venv cannot run `hf` cleanly, so switch):

```
deactivate
source ~/hf-venv/bin/activate
set -a; source .env; set +a
hf buckets ls <user>/<name>/outputs
hf buckets cp hf://buckets/<user>/<name>/outputs/gapfill_2024-06-15.png ./gapfill.png
open gapfill.png
```

You can also browse the bucket and its `gapfill_<date>.png` graphs in a web
browser on https://huggingface.co under your account.

If the metric fell and the graphs look sensible, the whole chain works: accounts,
client, controller, GPU VM, PACE read, training, evaluation, and the bucket sync.
Everything is self-cleaning: managed jobs tear the VM down on completion (or after
30 idle minutes), so nothing keeps running after the test.

--------------------------------------------------------------------------------

## Command-to-task reference

Each raw command above is the body of a pixi task in `pixi.toml`, so the macOS
path and the Linux/Windows path run the same underlying commands:

| pixi task (Linux/Windows)            | raw command (macOS)                                    |
|--------------------------------------|--------------------------------------------------------|
| `pixi run login`                     | `sky api login -e "https://.../"`                      |
| `pixi run check`                     | `sky check`                                            |
| `pixi run test`                      | `sky jobs launch -n demo-test --cpus 1+ -- echo hello` |
| `pixi run remote-train`              | `sky jobs launch recipes/train.yaml --env-file .env`   |
| `pixi run sky jobs queue -a`         | `sky jobs queue -a`                                    |
| `pixi run -e runtime hf buckets ...` | `hf buckets ...` (in the hf venv)                      |
