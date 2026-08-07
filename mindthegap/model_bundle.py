"""Portable Keras model bundles for local and hosted inference."""

from pathlib import Path
import subprocess

import numpy as np
import yaml


MODEL_FILENAME = "model.keras"
METADATA_FILENAME = "model_metadata.yaml"
README_FILENAME = "README.md"
REQUIRED_METADATA_SECTIONS = (
    "dataset",
    "inputs",
    "target",
    "preprocessing",
)


def _native(value):
    """Convert NumPy and path values into YAML-safe Python values."""
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _git_value(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _source_metadata():
    repository = _git_value("remote", "get-url", "origin")
    if repository.startswith("git@github.com:"):
        repository = "https://github.com/" + repository.removeprefix(
            "git@github.com:"
        )
    if repository.endswith(".git"):
        repository = repository[:-4]
    return {
        "repository": repository,
        "git_commit": _git_value("rev-parse", "HEAD"),
    }


def _validate_metadata(metadata):
    missing = [
        section
        for section in REQUIRED_METADATA_SECTIONS
        if section not in metadata
    ]
    if missing:
        raise ValueError(
            "metadata is missing required section(s): " + ", ".join(missing)
        )
    if not isinstance(metadata["inputs"], list) or not metadata["inputs"]:
        raise ValueError("metadata['inputs'] must be a non-empty ordered list")
    for channel, item in enumerate(metadata["inputs"]):
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("each metadata input must contain a name")
        if item.get("channel", channel) != channel:
            raise ValueError("metadata input channels must be ordered from zero")


def _model_card(metadata):
    model = metadata["model"]
    dataset = metadata["dataset"]
    target = metadata["target"]
    inputs = "\n".join(
        f"- Channel {item['channel']}: `{item['name']}`"
        for item in metadata["inputs"]
    )
    preprocessing = metadata["preprocessing"]
    region = dataset.get("region", "the documented domain")
    if isinstance(region, dict):
        region = ", ".join(
            f"{name} {bounds[0]} to {bounds[1]}"
            for name, bounds in region.items()
        )
    limitations = metadata.get(
        "limitations",
        "Use only with data matching the documented variables, domain, and "
        "preprocessing.",
    )
    return f"""---
library_name: keras
tags:
  - gap-filling
  - ocean-color
---

# {model['name']}

Keras model predicting **{target['name']}** for **{dataset['name']}**.

## Intended use

This model is intended for {region}
during {dataset.get('training_period', 'the documented training period')}.

## Inputs

{inputs}

## Preprocessing

Expected input shape: `{preprocessing['expected_input_shape']}`

Transforms and standardization parameters are recorded in
`model_metadata.yaml`.

## Limitations

{limitations}

## Source

Repository: {metadata['source']['repository']}

Git commit: `{metadata['source']['git_commit']}`
"""


def create_model_bundle_metadata(
    path,
    *,
    model_name,
    dataset_name,
    product_id,
    region,
    training_period,
    input_names,
    target_name,
    target_units,
    expected_input_shape,
    transforms,
    standardization,
    missing_value_handling,
    model_version="1.0",
    limitations=None,
    overwrite=False,
):
    """Create and save reviewable inference metadata for a model bundle."""
    bundle_path = Path(path)
    metadata_path = bundle_path / METADATA_FILENAME
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata already exists at {metadata_path}; use overwrite=True"
        )
    bundle_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "bundle_version": "1.0",
        "model": {
            "name": model_name,
            "version": model_version,
            "framework": "keras",
        },
        "dataset": {
            "name": dataset_name,
            "product_id": product_id,
            "region": region,
            "training_period": training_period,
        },
        "inputs": [
            {"name": name, "channel": channel}
            for channel, name in enumerate(input_names)
        ],
        "target": {
            "name": target_name,
            "units": target_units,
        },
        "preprocessing": {
            "expected_input_shape": expected_input_shape,
            "transforms": transforms,
            "standardization": standardization,
            "missing_value_handling": missing_value_handling,
        },
        "limitations": limitations
        or (
            "Use only with the documented product, region, channel order, "
            "and preprocessing."
        ),
        "source": _source_metadata(),
    }
    prepared = _native(metadata)
    _validate_metadata(prepared)
    with metadata_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(prepared, file, sort_keys=False)
    return metadata_path


def save_model_bundle(model, path, overwrite=False):
    """Save a Keras model and model card beside reviewed bundle metadata."""
    bundle_path = Path(path)
    metadata_path = bundle_path / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Create and review {metadata_path} before saving the model"
        )
    model_path = bundle_path / MODEL_FILENAME
    readme_path = bundle_path / README_FILENAME
    existing = [file.name for file in (model_path, readme_path) if file.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Bundle files already exist at {bundle_path}: "
            + ", ".join(existing)
            + "; use overwrite=True"
        )

    with metadata_path.open(encoding="utf-8") as file:
        metadata = yaml.safe_load(file)
    _validate_metadata(metadata)
    model.save(bundle_path / MODEL_FILENAME)
    readme_path.write_text(
        _model_card(metadata),
        encoding="utf-8",
    )
    return bundle_path


def load_model_bundle(path, compile=False):
    """Load a local model bundle and return ``(model, metadata)``."""
    bundle_path = Path(path)
    required = (
        bundle_path / MODEL_FILENAME,
        bundle_path / METADATA_FILENAME,
        bundle_path / README_FILENAME,
    )
    missing = [file.name for file in required if not file.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete model bundle at {bundle_path}; missing: "
            + ", ".join(missing)
        )

    import keras

    model = keras.models.load_model(
        bundle_path / MODEL_FILENAME,
        compile=compile,
    )
    # Keras defaults uncompiled GPU models to XLA JIT, which is not reliable
    # with this U-Net/cuDNN combination.
    if not compile:
        model.jit_compile = False
    with (bundle_path / METADATA_FILENAME).open(
        encoding="utf-8"
    ) as file:
        metadata = yaml.safe_load(file)
    _validate_metadata(metadata)
    return model, metadata
