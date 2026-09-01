#!/usr/bin/env python3
"""Prediction-only smoke test for the SkyPilot pipeline.

Downloads a pretrained mindthegap U-Net from a private Hugging Face model repo,
runs ONE forward pass on a deterministic synthetic input, and saves a PNG of the
prediction to outputs/ (which the sync-outputs task pushes to the HF bucket).

This is a PLUMBING test: the input is synthetic, so the picture proves the chain
runs end to end (HF model download, TensorFlow loading a keras-3 model on the VM,
CPU inference, output sync), NOT gap-fill quality. Swap in a real prepared frame
to get a meaningful gap-fill (step 2).
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless VM: no display
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# keras 3 needs a backend; tensorflow is the one we install in the runtime env.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

REPO_ID = "TroyRusso/mindthegap-arabsea-demo"
FILENAME = "UNet_DoubleConv_mse.keras"

# 1. Pull the model from HF. The repo is private, so pass the token from the env
#    (SkyPilot injects HF_TOKEN on the VM; load-env.sh supplies it locally).
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="model",
    token=os.environ.get("HF_TOKEN"),
)
print("downloaded model to", model_path)

# 2. Load it. compile=False: we only predict, so skip optimizer/loss restoration.
try:
    import keras
except ImportError:  # older stacks expose keras via tensorflow
    from tensorflow import keras
model = keras.models.load_model(model_path, compile=False)

_, H, W, C = tuple(model.inputs[0].shape)  # (None, 104, 152, 10)
H, W, C = (H or 104), (W or 152), (C or 10)
print("model input shape:", (H, W, C))

# 3. Deterministic synthetic input matching the model's expected shape. Seeded so
#    the run is reproducible; this is a plumbing input, not a real ocean frame.
rng = np.random.default_rng(0)
x = rng.standard_normal((1, H, W, C)).astype("float32")

# 4. One forward pass on CPU.
y = model.predict(x, verbose=0)
print(
    "prediction shape:", y.shape,
    "| min/mean/max: %.4f / %.4f / %.4f" % (float(y.min()), float(y.mean()), float(y.max())),
)

# 5. Save a PNG (synthetic input vs prediction) into outputs/ for sync-outputs.
outdir = Path("outputs")
outdir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(x[0, :, :, 0]); ax[0].set_title("synthetic input (channel 0)")
ax[1].imshow(y[0, :, :, 0]); ax[1].set_title("model prediction")
for a in ax:
    a.axis("off")
fig.suptitle("SkyPilot prediction smoke test (synthetic input)")
out = outdir / "prediction_smoke.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved figure to", out)
