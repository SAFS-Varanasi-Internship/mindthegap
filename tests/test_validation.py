"""Tests for the options validator (validate_options)."""

import pytest

import mindthegap as mtg
from mindthegap import Options, validate_options, OptionsValidationError
from conftest import make_demo_ds


def _prepared_options(ds, metadata):
    options = Options.default(data=ds, metadata=metadata, seed=1)
    options.verbose = False
    options.set_up_data_options(
        ds,
        target="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
    )
    return options


def test_validate_requires_options_object():
    with pytest.raises(TypeError, match="mindthegap.Options object"):
        validate_options({"data": {}}, requires=["data"])


def test_validate_rejects_unknown_requirement():
    options = Options.default(seed=1)
    with pytest.raises(ValueError, match="Unknown validation requirement"):
        validate_options(options, requires=["bogus"])


def test_validate_data_missing_reports_all_three_variables():
    options = Options.default(seed=1)  # data section unconfigured
    with pytest.raises(OptionsValidationError) as excinfo:
        validate_options(options, requires=["data"])
    message = str(excinfo.value)
    # Every missing variable is reported together with a how-to-fix example.
    assert "options.data.target_variable is not set" in message
    assert "options.data.missing_flag is not set" in message
    assert "options.data.land_flag is not set" in message
    assert "set_up_data_options" in message


def test_validate_data_passes_after_set_up_data_options():
    ds, metadata = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=1)
    options = _prepared_options(ds, metadata)
    # Returns options unchanged when the requirement is satisfied.
    assert validate_options(options, requires=["data"]) is options


def test_validate_data_prepared_requires_channels_and_stats():
    ds, metadata = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=1)
    options = _prepared_options(ds, metadata)
    # set_up_data_options alone does not resolve channels/standardization.
    with pytest.raises(OptionsValidationError) as excinfo:
        validate_options(options, requires=["data_prepared"])
    message = str(excinfo.value)
    assert "options.data.input_names is empty" in message
    assert "options.data.standardization is empty" in message
    assert "prepare_model_data" in message


def test_validate_split_requires_dates():
    options = Options.default(seed=1)
    with pytest.raises(OptionsValidationError) as excinfo:
        validate_options(options, requires=["split"])
    assert "set_up_train_split" in str(excinfo.value)


def test_validate_split_reports_partial_split():
    options = Options.default(seed=1)
    # train_dates set but val_dates empty: partially set and invalid.
    options.split.train_dates = ["2000-01-01", "2000-01-02"]
    with pytest.raises(OptionsValidationError) as excinfo:
        validate_options(options, requires=["split"])
    message = str(excinfo.value)
    assert "partially set" in message
    assert "val_dates" in message


def test_validate_gridder_rejects_unsupported_method():
    options = Options.default(seed=1)
    options.gridder.method = "grid"  # bypass construction validation
    with pytest.raises(OptionsValidationError) as excinfo:
        validate_options(options, requires=["gridder"])
    assert "only 'xbatcher'" in str(excinfo.value)


def test_validate_collects_problems_across_requirements():
    options = Options.default(seed=1)
    with pytest.raises(OptionsValidationError) as excinfo:
        validate_options(options, requires=["data", "split"])
    message = str(excinfo.value)
    # Problems from both requested sections appear in one error.
    assert "options.data.target_variable is not set" in message
    assert "options.split has no train/validation dates" in message


def test_validate_accepts_single_string_requirement():
    options = Options.default(seed=1)
    with pytest.raises(OptionsValidationError, match="target_variable"):
        validate_options(options, requires="data")


def test_validation_error_is_value_error():
    # OptionsValidationError subclasses ValueError so existing handlers work.
    assert issubclass(OptionsValidationError, ValueError)
    options = Options.default(seed=1)
    with pytest.raises(ValueError):
        validate_options(options, requires=["data"])
