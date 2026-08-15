"""Tests for the high-level train_model entry point."""

import pytest

import mindthegap as mtg
from mindthegap import Options, TrainingResult, OptionsValidationError
from conftest import make_demo_ds


def _configured_options(ds, metadata, **data_kwargs):
    options = Options.default(
        data=ds, metadata=metadata, smoke_test=True, seed=3
    )
    options.verbose = False
    options.set_up_data_options(
        ds,
        target="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        **data_kwargs,
    )
    return options


@pytest.fixture(scope="module")
def trained():
    ds, metadata = make_demo_ds(days=30, lat_size=16, lon_size=16, seed=3)
    options = _configured_options(ds, metadata)
    result = mtg.train_model(ds, options)
    return ds, options, result


def test_train_model_returns_training_result(trained):
    _, options, result = trained
    assert isinstance(result, TrainingResult)
    assert result.model is not None
    # The same options object is threaded through and now fully resolved.
    assert result.options is options
    assert options.data.is_resolved()
    assert options.split.is_resolved()


def test_train_model_metrics_come_from_evaluation(trained):
    _, _, result = trained
    # Explicit post-fit evaluation yields val_loss and val_mae (not from Options).
    assert "val_loss" in result.metrics
    assert "val_mae" in result.metrics
    assert all(isinstance(v, float) for v in result.metrics.values())


def test_train_model_history_has_training_curves(trained):
    _, _, result = trained
    for key in ("loss", "val_loss", "mae", "val_mae"):
        assert key in result.history
        assert isinstance(result.history[key], list)
    # Two epochs under smoke_test.
    assert len(result.history["loss"]) == result.metadata["epochs_run"]


def test_train_model_metadata_records_provenance(trained):
    _, options, result = trained
    md = result.metadata
    assert md["mindthegap_version"]  # non-empty
    assert md["n_channels"] == len(options.data.input_names)
    assert md["field_shape"] == (16, 16)
    assert md["device"] in ("GPU", "CPU", "unknown")
    assert "trained_at" in md
    # metadata is separate from Options config.
    assert "epochs" not in md


def test_train_model_keeps_lazy_ds_std():
    # With load_data=False the standardized dataset stays lazy (dask-backed).
    ds, metadata = make_demo_ds(days=30, lat_size=16, lon_size=16, seed=3)
    options = _configured_options(ds, metadata)
    result = mtg.train_model(ds, options, load_data=False)
    assert result.ds_std is not None
    assert result.ds_std["observed_target"].chunks is not None


def test_train_model_load_data_materializes_ds_std():
    # load_data=True eagerly loads the standardized dataset into memory.
    ds, metadata = make_demo_ds(days=30, lat_size=16, lon_size=16, seed=3)
    options = _configured_options(ds, metadata)
    result = mtg.train_model(ds, options, load_data=True)
    assert result.ds_std["observed_target"].chunks is None


def test_train_model_rejects_bad_load_data():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    options = _configured_options(ds, metadata)
    with pytest.raises(ValueError, match="load_data"):
        mtg.train_model(ds, options, load_data="sometimes")


def test_train_model_requires_configured_data():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    # No dataset/metadata passed, so options.data is unconfigured and
    # set_up_data_options is not called.
    options = Options.default(smoke_test=True, seed=3)
    options.verbose = False
    with pytest.raises(OptionsValidationError, match="target_variable"):
        mtg.train_model(ds, options)


def test_train_model_summary_is_readable(trained):
    _, _, result = trained
    text = result.summary()
    assert text.startswith("TrainingResult")
    assert "metrics" in text
