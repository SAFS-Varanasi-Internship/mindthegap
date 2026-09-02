# macOS setup for the PACE gap-fill smoke test

The pixi client environment in this repo is `linux-64` only, so on a Mac you set
up the SkyPilot client and the Hugging Face CLI directly with pip instead of
pixi. Nothing about the training changes: it still runs on a Linux GPU VM in the
cloud, which builds its own environment. You are only replacing the local
`pixi run ...` wrappers with the raw `sky ...` and `hf ...` commands they call.

Read [SMOKE_TEST.md](SMOKE_TEST.md) first for the accounts, the smoke-test
config, and what the evaluation means. This file only covers the macOS-specific
client setup; the meaning of each step is the same.

**Validation note.** This exact command flow (install, login, `sky check`,
`sky jobs queue`, `hf buckets ls` and `cp`) was tested on Linux, where `sky` and
`hf` are the same CLIs as on macOS, and it worked. The macOS-specific pieces
(installing Python, `open` to view a file) are standard. A run on an actual Mac
has not been done yet, so treat the first one as a shakedown and tell Troy if
anything differs.

--------------------------------------------------------------------------------

## 1. Prerequisites

Same accounts as SMOKE_TEST.md: Hugging Face (an access token plus a bucket),
NASA Earthdata, and a SkyPilot controller login from Scott. On the Mac you also
need:

- **Python 3**: macOS ships one; check with `python3 --version`. If missing,
  `brew install python`.
- **git**: `git --version` will offer to install the command line tools if it is
  not already there.

Clone the repo and enter the example folder:

```
git clone https://github.com/SAFS-Varanasi-Internship/mindthegap.git
cd mindthegap/example-windows-setup
```

Copy the credentials template and fill it in (same fields as SMOKE_TEST.md:
`HF_TOKEN`, `HF_BUCKET`, `SKYPILOT_API_*`, `EARTHDATA_*`):

```
cp .env.example .env
```

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

Prove access and quota with a trivial job (this is what `pixi run test` does):

```
sky jobs launch -n demo-test --cpus 1+ -y -- echo hello
```

When it reports SUCCEEDED, your access works.

--------------------------------------------------------------------------------

## 4. Run the training smoke test (sky venv)

Edit `recipes/train.py` and set a small, fast config near the top:

```python
REGION = "arabian sea"
TIME_SLICE = slice("2024-06-01", "2024-07-01")
EPOCHS = 3
```

Launch it (this is what `pixi run remote-train -y` does):

```
sky jobs launch recipes/train.yaml --env-file .env -y
```

Monitor it:

```
sky jobs queue -a
sky jobs logs <job-id>
```

Watch `val_fakecloud_mse` fall in the log; see SMOKE_TEST.md section 4 for what
that number means. If the job sits in `PENDING`, set `use_spot: false` in
`recipes/train.yaml` and relaunch.

--------------------------------------------------------------------------------

## 5. Get the outputs (hf venv)

The `sky` venv cannot run `hf` cleanly, so switch venvs:

```
deactivate
source ~/hf-venv/bin/activate
set -a; source .env; set +a
hf buckets ls <user>/<name>/outputs
hf buckets cp hf://buckets/<user>/<name>/outputs/gapfill_2024-06-15.png ./gapfill.png
open gapfill.png
```

If you would rather not use the CLI, you can also browse the bucket and its
`gapfill_<date>.png` graphs in a web browser on https://huggingface.co under your
account.

--------------------------------------------------------------------------------

## Command-to-task reference

Each raw command above is the body of a pixi task in `pixi.toml`, so the two
paths run the same underlying commands:

| pixi task (Linux/Windows)         | raw command (macOS)                                      |
|-----------------------------------|----------------------------------------------------------|
| `pixi run login`                  | `sky api login -e "https://.../"`                        |
| `pixi run check`                  | `sky check`                                              |
| `pixi run test`                   | `sky jobs launch -n demo-test --cpus 1+ -- echo hello`   |
| `pixi run remote-train`           | `sky jobs launch recipes/train.yaml --env-file .env`     |
| `pixi run sky jobs queue -a`      | `sky jobs queue -a`                                      |
| `pixi run -e runtime hf buckets ...` | `hf buckets ...` (in the hf venv)                     |
