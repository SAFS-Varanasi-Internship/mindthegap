from pathlib import Path

import keras
import numpy as np
import pytest
import yaml

from mindthegap import load_model_bundle, save_model_bundle


def _model():
    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Dense(2)(inputs)
    model = keras.Model(inputs, outputs, name="test-gap-model")
    model.compile(optimizer="adam", loss="mse")
    return model


def _metadata():
    return {
        "dataset": {
            "name": "test product",
            "product_id": "test-id",
            "region": "test region",
            "training_period": "2020-01-01 to 2020-01-31",
        },
        "inputs": [
            {"name": "channel_a", "channel": 0},
            {"name": "channel_b", "channel": 1},
            {"name": "channel_c", "channel": 2},
            {"name": "channel_d", "channel": 3},
        ],
        "target": {"name": "chlor_a", "units": "mg m-3"},
        "preprocessing": {
            "expected_input_shape": [None, 4],
            "transforms": ["log target"],
            "standardization": {
                "chlor_a": np.array([0.5, 1.5], dtype=np.float32)
            },
            "missing_value_handling": "Replace NaN inputs with zero.",
        },
    }


def test_model_bundle_round_trip(tmp_path):
    model = _model()
    sample = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = model.predict(sample, verbose=0)
    bundle_path = tmp_path / "bundle"

    result = save_model_bundle(model, bundle_path, _metadata())
    loaded_model, metadata = load_model_bundle(bundle_path)
    actual = loaded_model.predict(sample, verbose=0)

    assert result == bundle_path
    assert {file.name for file in bundle_path.iterdir()} == {
        "model.keras",
        "model_metadata.yaml",
        "README.md",
    }
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert metadata["inputs"][0] == {"name": "channel_a", "channel": 0}
    assert metadata["preprocessing"]["standardization"]["chlor_a"] == [
        0.5,
        1.5,
    ]
    assert metadata["source"]["git_commit"] != ""
    assert "library_name: keras" in (bundle_path / "README.md").read_text()
    assert yaml.safe_load(
        (bundle_path / "model_metadata.yaml").read_text()
    ) == metadata


def test_bundle_requires_inference_metadata(tmp_path):
    with pytest.raises(ValueError, match="missing required section"):
        save_model_bundle(_model(), tmp_path / "bundle", {"dataset": {}})


def test_bundle_rejects_unordered_channels(tmp_path):
    metadata = _metadata()
    metadata["inputs"][1]["channel"] = 3

    with pytest.raises(ValueError, match="ordered from zero"):
        save_model_bundle(_model(), tmp_path / "bundle", metadata)


def test_bundle_overwrite_is_explicit(tmp_path):
    bundle_path = tmp_path / "bundle"
    save_model_bundle(_model(), bundle_path, _metadata())

    with pytest.raises(FileExistsError):
        save_model_bundle(_model(), bundle_path, _metadata())

    save_model_bundle(_model(), bundle_path, _metadata(), overwrite=True)


def test_load_rejects_incomplete_bundle(tmp_path):
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    (bundle_path / "model.keras").touch()

    with pytest.raises(FileNotFoundError, match="model_metadata.yaml"):
        load_model_bundle(bundle_path)
