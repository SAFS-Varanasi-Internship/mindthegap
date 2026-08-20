import json
from pathlib import Path

import keras
import numpy as np
import pytest

from mindthegap import (
    Options,
    TrainingResult,
    load_model_bundle,
    load_bundle_metrics,
    save_model_bundle,
)


def _model():
    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Dense(2)(inputs)
    model = keras.Model(inputs, outputs, name="test-gap-model")
    model.compile(optimizer="adam", loss="mse")
    return model


def _resolved_options(**overrides):
    """Return an Options with a populated (resolved) data section."""
    options = Options.default(seed=1)
    data = options.data
    data.source = "Test product"
    data.product_id = "test-id"
    data.target = "full_target"
    data.target_variable = "chlor_a"
    data.target_name = "chlor_a"
    data.target_units = "mg m-3"
    data.input_names = ["observed_target", "day_sin", "day_cos", "land_flag"]
    data.lat_bounds = (5.0, 31.0)
    data.lon_bounds = (42.0, 80.0)
    data.data_source = "user manual"
    data.standardization = {
        "full_target": {"mean": 0.5, "std": 1.5, "applied": True},
        "observed_target": {"mean": 0.5, "std": 1.5, "applied": True},
    }
    data.target_mean = 0.5
    data.target_std = 1.5
    data.transforms = {"target": "natural logarithm", "temporal_lags": 1}
    data.missing_value_handling = "Replace NaN inputs with zero."
    # A saved bundle is the product of a completed training run, so its options
    # must also carry a resolved split and gridder. Populate a minimal manual
    # split and an explicit tile so the fixture is a fully valid training
    # configuration.
    options.split.method = "manual"
    options.split.train_dates = ["2010-01-01", "2010-01-02", "2010-01-03"]
    options.split.val_dates = ["2010-01-04"]
    options.gridder.tile_size = (64, 64)
    for key, value in overrides.items():
        setattr(data, key, value)
    return options


def _result(model=None, options=None):
    """Return a TrainingResult wrapping a model + resolved options."""
    return TrainingResult(
        model=model if model is not None else _model(),
        options=options if options is not None else _resolved_options(),
        metadata={"mindthegap_version": "9.9.9", "n_channels": 4},
        metrics={"val_loss": 0.31, "val_mae": 0.22},
        history={"loss": [1.0, 0.5], "val_loss": [1.1, 0.31]},
    )


def test_model_bundle_round_trip(tmp_path):
    model = _model()
    sample = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = model.predict(sample, verbose=0)
    bundle_path = tmp_path / "bundle"
    options = _resolved_options()

    saved_path = save_model_bundle(_result(model, options), bundle_path)
    loaded_model, loaded_options = load_model_bundle(bundle_path)
    actual = loaded_model.predict(sample, verbose=0)

    assert saved_path == bundle_path
    assert loaded_model.jit_compile is False
    assert {file.name for file in bundle_path.iterdir()} == {
        "model.keras",
        "options.json",
        "README.md",
        "metrics.json",
    }
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert loaded_options == options
    assert loaded_options.data.input_names == options.data.input_names
    assert "library_name: keras" in (bundle_path / "README.md").read_text()


def test_options_json_is_the_full_resolved_options(tmp_path):
    bundle_path = tmp_path / "bundle"
    options = _resolved_options()

    save_model_bundle(_result(options=options), bundle_path)

    saved = json.loads((bundle_path / "options.json").read_text())
    assert saved["data"]["target"] == "full_target"
    assert saved["data"]["input_names"] == options.data.input_names
    assert saved["fit"]["epochs"] == options.fit.epochs
    assert Options.from_dict(saved) == options


def test_readme_records_data_source(tmp_path):
    bundle_path = tmp_path / "bundle"
    options = _resolved_options(data_source="demo_data(dataset='pace')")

    save_model_bundle(_result(options=options), bundle_path)

    readme = (bundle_path / "README.md").read_text()
    assert "demo_data(dataset='pace')" in readme


def test_load_model_bundle_reports_data_source(tmp_path, capsys):
    bundle_path = tmp_path / "bundle"
    options = _resolved_options(data_source="demo_data(dataset='pace')")
    save_model_bundle(_result(options=options), bundle_path)

    capsys.readouterr()
    load_model_bundle(bundle_path)
    out = capsys.readouterr().out
    assert "demo_data(dataset='pace')" in out


def test_save_requires_resolved_options(tmp_path):
    # A result whose options.data is unresolved is rejected.
    result = _result(options=Options.default(seed=1))
    with pytest.raises(ValueError, match="prepare_model_data"):
        save_model_bundle(result, tmp_path / "bundle")


def test_save_requires_resolved_split(tmp_path):
    # A fully resolved data section is not enough: the split must also be set,
    # because a saved bundle is the product of a completed training run.
    options = _resolved_options()
    options.split.train_dates = []
    options.split.val_dates = []
    with pytest.raises(ValueError, match="options.split"):
        save_model_bundle(_result(options=options), tmp_path / "bundle")


def test_save_requires_valid_gridder(tmp_path):
    # An unsupported gridder method is rejected on save.
    options = _resolved_options()
    options.gridder.method = "grid"  # bypass construction validation
    with pytest.raises(ValueError, match="xbatcher"):
        save_model_bundle(_result(options=options), tmp_path / "bundle")


def test_save_rejects_non_training_result(tmp_path):
    # A bare Keras model is no longer accepted; a TrainingResult is required.
    with pytest.raises(TypeError, match="TrainingResult"):
        save_model_bundle(_model(), tmp_path / "bundle")


def test_bundle_overwrite_is_explicit(tmp_path):
    bundle_path = tmp_path / "bundle"
    options = _resolved_options()
    save_model_bundle(_result(options=options), bundle_path)

    with pytest.raises(FileExistsError):
        save_model_bundle(_result(options=options), bundle_path)

    save_model_bundle(_result(options=options), bundle_path, overwrite=True)


def test_load_rejects_incomplete_bundle(tmp_path):
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    (bundle_path / "model.keras").touch()

    with pytest.raises(FileNotFoundError, match="options.json"):
        load_model_bundle(bundle_path)


def test_load_rejects_bundle_missing_metrics(tmp_path):
    # metrics.json is a required bundle file.
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    for name in ("model.keras", "options.json", "README.md"):
        (bundle_path / name).touch()

    with pytest.raises(FileNotFoundError, match="metrics.json"):
        load_model_bundle(bundle_path)


def test_dataset_script_written_from_string(tmp_path):
    bundle_path = tmp_path / "bundle"
    options = _resolved_options()
    script = "import mindthegap as mtg\nds = mtg.demo_data('pace', options)\n"

    save_model_bundle(
        _result(options=options), bundle_path, dataset_script=script
    )

    written = (bundle_path / "make_dataset.py").read_text()
    assert written == script
    readme = (bundle_path / "README.md").read_text()
    assert "make_dataset.py" in readme


def test_dataset_script_copied_from_file(tmp_path):
    bundle_path = tmp_path / "bundle"
    options = _resolved_options()
    script_file = tmp_path / "build.py"
    script_file.write_text("# builds the dataset\nds = load()\n")

    save_model_bundle(
        _result(options=options), bundle_path, dataset_script=script_file
    )

    assert (bundle_path / "make_dataset.py").read_text() == (
        "# builds the dataset\nds = load()\n"
    )


def test_metrics_json_persists_run(tmp_path):
    bundle_path = tmp_path / "bundle"

    save_model_bundle(_result(), bundle_path)

    saved = json.loads((bundle_path / "metrics.json").read_text())
    assert saved["metrics"] == {"val_loss": 0.31, "val_mae": 0.22}
    assert saved["metadata"]["mindthegap_version"] == "9.9.9"
    assert saved["history"]["val_loss"] == [1.1, 0.31]


def test_load_bundle_metrics_returns_run(tmp_path):
    bundle_path = tmp_path / "bundle"
    save_model_bundle(_result(), bundle_path)

    run = load_bundle_metrics(bundle_path)
    assert run["metrics"] == {"val_loss": 0.31, "val_mae": 0.22}
    assert run["metadata"]["n_channels"] == 4
    assert run["history"]["loss"] == [1.0, 0.5]


def test_load_bundle_metrics_missing_file(tmp_path):
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    with pytest.raises(FileNotFoundError, match="metrics.json"):
        load_bundle_metrics(bundle_path)
