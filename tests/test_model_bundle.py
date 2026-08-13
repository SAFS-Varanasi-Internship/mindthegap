from pathlib import Path

import keras
import numpy as np
import pytest
import yaml

from mindthegap import (
    create_model_bundle_metadata,
    load_model_bundle,
    save_model_bundle,
)


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


def _create_metadata(bundle_path, metadata=None, overwrite=False):
    metadata = metadata or _metadata()
    return create_model_bundle_metadata(
        bundle_path,
        model_name="test-gap-model",
        dataset_name=metadata["dataset"]["name"],
        product_id=metadata["dataset"]["product_id"],
        region=metadata["dataset"]["region"],
        training_period=metadata["dataset"]["training_period"],
        input_names=[item["name"] for item in metadata["inputs"]],
        target_name=metadata["target"]["name"],
        target_units=metadata["target"]["units"],
        expected_input_shape=metadata["preprocessing"]["expected_input_shape"],
        transforms=metadata["preprocessing"]["transforms"],
        standardization=metadata["preprocessing"]["standardization"],
        missing_value_handling=metadata["preprocessing"][
            "missing_value_handling"
        ],
        overwrite=overwrite,
    )


def test_model_bundle_round_trip(tmp_path):
    model = _model()
    sample = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = model.predict(sample, verbose=0)
    bundle_path = tmp_path / "bundle"

    metadata_path = _create_metadata(bundle_path)
    result = save_model_bundle(model, bundle_path)
    loaded_model, metadata = load_model_bundle(bundle_path)
    actual = loaded_model.predict(sample, verbose=0)

    assert result == bundle_path
    assert loaded_model.jit_compile is False
    assert metadata_path == bundle_path / "model_metadata.yaml"
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


def test_bundle_requires_metadata_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Create and review"):
        save_model_bundle(_model(), tmp_path / "bundle")


def test_metadata_overwrite_is_explicit(tmp_path):
    bundle_path = tmp_path / "bundle"
    _create_metadata(bundle_path)

    with pytest.raises(FileExistsError, match="Metadata already exists"):
        _create_metadata(bundle_path)

    _create_metadata(bundle_path, overwrite=True)


def test_metadata_preserves_dataset_loader_details(tmp_path):
    bundle_path = tmp_path / "bundle"
    create_model_bundle_metadata(
        bundle_path,
        model_name="test-gap-model",
        dataset_metadata={
            "name": "Synthetic",
            "product_id": "mindthegap-synthetic",
            "region": {"lat": [5.0, 31.0], "lon": [42.0, 80.0]},
            "available_period": "2020-01-01 to 2020-04-29",
        },
        training_period="2020-01-01 to 2020-02-29",
        input_names=["channel_a"],
        target_name="chlor_a",
        target_units="mg m-3",
        expected_input_shape=[None, 1],
        transforms=[],
        standardization={},
        missing_value_handling="Replace NaN inputs with zero.",
    )

    metadata = yaml.safe_load(
        (bundle_path / "model_metadata.yaml").read_text()
    )
    assert (
        metadata["dataset"]["available_period"]
        == "2020-01-01 to 2020-04-29"
    )
    assert metadata["dataset"]["training_period"] == (
        "2020-01-01 to 2020-02-29"
    )


def test_metadata_records_data_source_from_argument(tmp_path):
    bundle_path = tmp_path / "bundle"
    _create_metadata_with_data_source(
        bundle_path,
        data_source="demo_data(dataset='pace')",
    )

    metadata = yaml.safe_load(
        (bundle_path / "model_metadata.yaml").read_text()
    )
    assert (
        metadata["dataset"]["data_source"]
        == "demo_data(dataset='pace')"
    )


def test_metadata_defaults_data_source_to_user_manual(tmp_path):
    bundle_path = tmp_path / "bundle"
    _create_metadata(bundle_path)

    metadata = yaml.safe_load(
        (bundle_path / "model_metadata.yaml").read_text()
    )
    assert metadata["dataset"]["data_source"] == "user manual"


def test_load_model_bundle_reports_data_source(tmp_path, capsys):
    bundle_path = tmp_path / "bundle"
    _create_metadata_with_data_source(
        bundle_path,
        data_source="demo_data(dataset='pace')",
    )
    save_model_bundle(_model(), bundle_path)

    capsys.readouterr()
    load_model_bundle(bundle_path)
    out = capsys.readouterr().out
    assert "demo_data(dataset='pace')" in out


def _create_metadata_with_data_source(bundle_path, *, data_source):
    metadata = _metadata()
    return create_model_bundle_metadata(
        bundle_path,
        model_name="test-gap-model",
        dataset_name=metadata["dataset"]["name"],
        product_id=metadata["dataset"]["product_id"],
        region=metadata["dataset"]["region"],
        training_period=metadata["dataset"]["training_period"],
        input_names=[item["name"] for item in metadata["inputs"]],
        target_name=metadata["target"]["name"],
        target_units=metadata["target"]["units"],
        expected_input_shape=metadata["preprocessing"]["expected_input_shape"],
        transforms=metadata["preprocessing"]["transforms"],
        standardization=metadata["preprocessing"]["standardization"],
        missing_value_handling=metadata["preprocessing"][
            "missing_value_handling"
        ],
        data_source=data_source,
    )


def test_bundle_overwrite_is_explicit(tmp_path):
    bundle_path = tmp_path / "bundle"
    _create_metadata(bundle_path)
    save_model_bundle(_model(), bundle_path)

    with pytest.raises(FileExistsError):
        save_model_bundle(_model(), bundle_path)

    save_model_bundle(_model(), bundle_path, overwrite=True)


def test_load_rejects_incomplete_bundle(tmp_path):
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    (bundle_path / "model.keras").touch()

    with pytest.raises(FileNotFoundError, match="model_metadata.yaml"):
        load_model_bundle(bundle_path)


def test_metadata_includes_resolved_options(tmp_path):
    from mindthegap import Options

    options = Options.default()
    options.data.target = "full_target"
    options.data.input_names = ["observed_target"]
    bundle_path = tmp_path / "bundle"

    _create_metadata(bundle_path)
    metadata_path = create_model_bundle_metadata(
        bundle_path,
        model_name="test-gap-model",
        dataset_name="test product",
        product_id="test-id",
        region="test region",
        training_period="2020-01-01 to 2020-01-31",
        input_names=["channel_a"],
        target_name="chlor_a",
        target_units="mg m-3",
        expected_input_shape=[None, 1],
        transforms=[],
        standardization={},
        missing_value_handling="Replace NaN inputs with zero.",
        options=options,
        overwrite=True,
    )

    saved = yaml.safe_load(metadata_path.read_text())
    assert saved["options"]["data"]["target"] == "full_target"
    assert saved["options"]["fit"]["epochs"] == 50
    assert Options.from_dict(saved["options"]) == options

