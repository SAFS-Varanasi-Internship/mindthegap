# Saving and publishing Mind the Gap models

This directory is for released model bundles. A bundle is one directory with
exactly the files needed to understand and reload a trained model:

```text
my-model-bundle/
├── model.keras
├── model_metadata.yaml
└── README.md
```

- `model.keras` contains the complete Keras architecture and trained weights.
- `model_metadata.yaml` records the dataset, ordered input channels, target,
  preprocessing, standardization, geographic and temporal scope, and source
  code version.
- `README.md` is a model card. Hugging Face displays it on the model's page.

Do not add training datasets, checkpoints, logs, or credentials to a bundle.

## Required Python libraries

From the root of this repository, install Mind the Gap and the libraries used
to train and load Keras models:

```bash
python -m pip install -e .
python -m pip install "keras>=3" "tensorflow>=2.19"
```

`mindthegap` installs PyYAML, which reads and writes bundle metadata.

To upload or download models from Hugging Face, also install:

```bash
python -m pip install -U huggingface_hub
```

This provides both the `huggingface_hub` Python package and the `hf`
command-line program.

## Save a model bundle

Creating a bundle is intentionally a two-step process. First, create and
review the metadata. Only after confirming that it is correct should you save
the trained model.

Input names must be listed in the exact order used to construct the model
input.

```python
import mindthegap as mtg

bundle_path = "models/arabian-sea-chlorophyll-unet"

metadata_path = mtg.create_model_bundle_metadata(
    bundle_path,
    model_name="Arabian Sea chlorophyll U-Net",
    dataset_name="Copernicus GlobColour",
    product_id="cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D",
    region={
        "lat": [5.0, 31.0],
        "lon": [42.0, 80.0],
    },
    training_period="2018-01-01 to 2020-12-31",
    input_names=[
        "masked_target",
        "masked_target_m1",
        "masked_target_p1",
        "day_sin",
        "day_cos",
        "synthetic_missing_flag",
        "true_missing_flag",
        "valid_masked_target_flag",
        "land_flag",
    ],
    target_name="CHL",
    target_units="mg m-3",
    expected_input_shape=[None, None, None, 9],
    transforms={
        "target": "natural logarithm",
        "temporal_lags": 1,
    },
    standardization={
        "full_target": {
            "mean": 0.52,
            "std": 0.19,
            "applied": True,
        },
    },
    missing_value_handling=(
        "Replace missing predictor values with zero after standardization; "
        "use the mask channels to identify land and missing observations."
    ),
    limitations=(
        "Use only with the documented product, region, channel order, and "
        "preprocessing."
    ),
)
print(metadata_path)
```

Open `models/arabian-sea-chlorophyll-unet/model_metadata.yaml` and check every
field, especially the dataset, region, dates, input order, transforms, and
standardization values. Correct the code that creates the metadata and run it
again if anything is wrong. Do not proceed until this file is accurate.

After reviewing the metadata, save the model and generate its model card:

```python
mtg.save_model_bundle(
    model,
    bundle_path,
)
print(bundle_path)
```

`create_model_bundle_metadata()` records the current Git commit in
`model_metadata.yaml`; it does not create a Git commit or upload anything.

Confirm the local bundle loads before publishing it:

```python
import numpy as np
import mindthegap as mtg

loaded_model, loaded_metadata = mtg.load_model_bundle(
    "models/arabian-sea-chlorophyll-unet"
)

sample = np.zeros((1, 64, 64, 9), dtype="float32")
prediction = loaded_model.predict(sample)

print(prediction.shape)
print([item["name"] for item in loaded_metadata["inputs"]])
```

## Create a Hugging Face account

1. Open <https://huggingface.co/join> and create an account.
2. Ask an administrator to add your account to the
   [`fish-pace`](https://huggingface.co/fish-pace) organization if the model
   should be owned by that organization.
3. Install `huggingface_hub` as shown above.
4. Log in from a terminal:

   ```bash
   hf auth login
   ```

   The command normally displays a URL and code. Open the URL in a browser,
   enter the code, and approve access.
5. Confirm the login and organization membership:

   ```bash
   hf auth whoami
   ```

Never put a Hugging Face access token in a notebook, source file, bundle,
Git commit, or shell command saved in your history.

## Upload a bundle to Hugging Face

Choose a lowercase repository name such as
`fish-pace/arabian-sea-chlorophyll-unet`. You need write access to the
`fish-pace` organization.

The following Python code creates the model repository and uploads the bundle
contents to its root:

```python
from huggingface_hub import HfApi

repo_id = "fish-pace/arabian-sea-chlorophyll-unet"
bundle_path = "models/arabian-sea-chlorophyll-unet"

api = HfApi()
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=bundle_path,
)

print(f"https://huggingface.co/{repo_id}")
```

Alternatively, after creating the repository, upload from a terminal:

```bash
hf upload fish-pace/arabian-sea-chlorophyll-unet \
  models/arabian-sea-chlorophyll-unet \
  .
```

The final Hugging Face repository root should contain `model.keras`,
`model_metadata.yaml`, and `README.md`. Open the printed URL and check that
the model card is readable and all three files are present.

Re-running `upload_folder()` or `hf upload` updates the existing repository.
Review changes carefully before replacing a published model; prefer a new
model version or repository when scientific behavior changes.

## Load a model from Hugging Face

`load_model_bundle()` loads a local directory. First download the Hugging Face
repository with `snapshot_download()`, then pass the downloaded directory to
Mind the Gap:

```python
from huggingface_hub import snapshot_download
import mindthegap as mtg

local_bundle = snapshot_download(
    repo_id="fish-pace/arabian-sea-chlorophyll-unet",
    repo_type="model",
)

model, metadata = mtg.load_model_bundle(local_bundle)
print(model.input_shape)
print(metadata["target"])
print([item["name"] for item in metadata["inputs"]])
```

Public repositories download without authentication. For a private or gated
repository, run `hf auth login` first and make sure your account has access.
`snapshot_download()` caches files locally, so later loads do not download
unchanged files again.

Before prediction, reconstruct inputs in the exact order recorded under
`metadata["inputs"]` and apply every operation under
`metadata["preprocessing"]`. Loading the Keras model alone does not perform
dataset-specific preprocessing.
