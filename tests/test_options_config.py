import pandas as pd
import pytest

import mindthegap as mtg
from mindthegap import (
    FitOptions,
    GridderOptions,
    Options,
    SplitOptions,
    train_validation_dates,
)
from conftest import make_demo_ds


def test_gridder_defaults_are_used_as_is():
    gridder = GridderOptions()

    assert gridder.tile_size == (64, 64)
    assert gridder.time_chunk == 100


def test_gridder_full_survives_serialization_round_trip():
    from mindthegap.options import _to_plain

    plain = _to_plain(GridderOptions(tile_size="full"))
    assert plain["tile_size"] == "full"

    restored = GridderOptions.from_dict(plain)
    assert restored.tile_size == "full"


def test_split_defaults():
    split = SplitOptions()

    assert split.method == "random"
    assert split.n_days is None
    assert split.train_fraction == 0.8
    assert split.val_fraction == 0.2
    assert split.train_dates == []
    assert split.val_dates == []
    assert split.is_resolved() is False
    assert split.training_period() is None


def test_train_validation_dates_random_populates_split():
    ds, metadata = make_demo_ds(days=100, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata)

    result = train_validation_dates(ds.time, options, method="random", seed=7)

    split = options.split
    assert result is options
    assert split.method == "random"
    assert split.is_resolved()
    assert len(split.train_dates) == 80
    assert len(split.val_dates) == 20
    assert not (set(split.train_dates) & set(split.val_dates))


def test_train_validation_dates_random_limits_total_days():
    ds, metadata = make_demo_ds(days=100, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata)

    train_validation_dates(ds.time, options, n_days=50, seed=7)

    assert options.split.n_days == 50
    assert len(options.split.train_dates) == 40
    assert len(options.split.val_dates) == 10


def test_train_validation_dates_random_reads_n_days_from_options():
    ds, metadata = make_demo_ds(days=100, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata)
    options.split.n_days = 25

    train_validation_dates(ds.time, options, seed=7)

    assert len(options.split.train_dates) == 20
    assert len(options.split.val_dates) == 5


def test_train_validation_dates_caps_n_days_at_available_dates():
    ds, metadata = make_demo_ds(days=20, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata)

    train_validation_dates(ds.time, options, n_days=100, seed=7)

    assert len(options.split.train_dates) == 16
    assert len(options.split.val_dates) == 4


def test_train_validation_dates_random_is_deterministic():
    ds, metadata = make_demo_ds(days=60, lat_size=8, lon_size=8, seed=3)

    a = train_validation_dates(
        ds.time, Options.default(data=ds, metadata=metadata), method="random", seed=5
    )
    b = train_validation_dates(
        ds.time, Options.default(data=ds, metadata=metadata), method="random", seed=5
    )

    assert a.split.train_dates == b.split.train_dates
    assert a.split.val_dates == b.split.val_dates


def test_train_validation_dates_manual_selects_windows():
    ds, metadata = make_demo_ds(days=120, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata)

    train_validation_dates(
        ds.time,
        options,
        method="manual",
        train_slice=slice("2020-01-01", "2020-02-01"),
        val_slice=slice("2020-02-02", "2020-02-20"),
    )

    split = options.split
    assert split.method == "manual"
    assert split.train_dates[0] == "2020-01-01"
    assert split.is_resolved()


def test_train_validation_dates_manual_errors_on_empty_slice():
    ds, metadata = make_demo_ds(days=60, lat_size=8, lon_size=8, seed=2)

    with pytest.raises(ValueError, match="selects no dates"):
        train_validation_dates(
            ds.time,
            Options.default(data=ds, metadata=metadata),
            method="manual",
            train_slice=slice("1990-01-01", "1990-02-01"),
            val_slice=slice("2020-01-05", "2020-01-20"),
        )


def test_train_validation_dates_rejects_unknown_method():
    ds, metadata = make_demo_ds(days=30, lat_size=8, lon_size=8, seed=2)

    with pytest.raises(ValueError, match="Unknown method"):
        train_validation_dates(
            ds.time, Options.default(data=ds, metadata=metadata), method="bogus"
        )


def test_set_up_train_split_options_uses_all_days_by_default():
    ds, metadata = make_demo_ds(days=60, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata, seed=7)

    result = mtg.set_up_train_split_options(ds, options, verbose=False)

    split = options.split
    assert result is options
    assert split.method == "random"
    assert split.is_resolved()
    # n_days defaults to every date in ds.time and is recorded on the split.
    assert split.n_days == ds.sizes["time"]
    assert len(split.train_dates) == 48
    assert len(split.val_dates) == 12
    assert not (set(split.train_dates) & set(split.val_dates))


def test_set_up_train_split_options_respects_explicit_n_days():
    ds, metadata = make_demo_ds(days=60, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata, seed=7)

    mtg.set_up_train_split_options(
        ds, options, split_options={"n_days": 30}, verbose=False
    )

    assert options.split.n_days == 30
    assert len(options.split.train_dates) == 24
    assert len(options.split.val_dates) == 6


def test_set_up_train_split_options_accepts_manual_mode():
    ds, metadata = make_demo_ds(days=120, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata, seed=7)
    times = pd.to_datetime(ds.time.values)
    train_slice = slice(str(times[0].date()), str(times[59].date()))
    val_slice = slice(str(times[60].date()), str(times[-1].date()))

    mtg.set_up_train_split_options(
        ds,
        options,
        split_mode="manual",
        split_options={"train_slice": train_slice, "val_slice": val_slice},
        verbose=False,
    )

    assert options.split.method == "manual"
    # n_days is not forced for a manual split.
    assert options.split.n_days is None
    assert options.split.is_resolved()


def test_set_up_train_split_options_rejects_invalid_keys():
    ds, metadata = make_demo_ds(days=60, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata, seed=7)

    # train_slice does not apply to the random mode.
    with pytest.raises(ValueError, match="do not apply to split_mode='random'"):
        mtg.set_up_train_split_options(
            ds,
            options,
            split_options={"train_slice": slice("2010-01-01", "2010-02-01")},
            verbose=False,
        )


def test_set_up_train_split_options_rejects_unknown_mode():
    ds, metadata = make_demo_ds(days=60, lat_size=8, lon_size=8, seed=2)
    options = Options.default(data=ds, metadata=metadata, seed=7)

    with pytest.raises(ValueError, match="split_mode must be one of"):
        mtg.set_up_train_split_options(ds, options, split_mode="bogus")


def test_prepare_model_data_auto_runs_set_up_train_split_options():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(
        data=ds, metadata=metadata, smoke_test=True, seed=1
    )
    options.verbose = False

    from mindthegap import prepare_model_data

    prepare_model_data(ds, options, mode="train")

    assert options.split.is_resolved()
    assert options.split.method == "random"


def test_prepare_model_data_errors_on_partial_split():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(
        data=ds, metadata=metadata, smoke_test=True, seed=1
    )
    options.verbose = False
    # Only train_dates set, no val_dates: partially set and invalid.
    options.split.train_dates = [
        str(d.date()) for d in pd.to_datetime(ds.time.values[:5])
    ]

    from mindthegap import prepare_model_data, OptionsValidationError

    with pytest.raises(OptionsValidationError, match="partially set"):
        prepare_model_data(ds, options, mode="train")


@pytest.mark.parametrize("n_days", [0, -1, 1.5, True])
def test_split_rejects_invalid_n_days(n_days):
    with pytest.raises(ValueError, match="n_days must be a positive integer"):
        SplitOptions(n_days=n_days)


def test_set_data_config_leaves_gridder_default_and_split_unresolved():
    ds, _ = make_demo_ds(days=120, lat_size=16, lon_size=16, seed=42)
    options = Options.default()

    assert options.gridder.tile_size == (64, 64)
    assert options.split.is_resolved() is False


def test_set_data_config_populates_data_from_metadata():
    ds, metadata = make_demo_ds(days=120, lat_size=16, lon_size=16, seed=42)
    options = Options.default()

    result = options.set_data_config(data=ds, metadata=metadata)

    assert result is options
    assert options.data.source == metadata["dataset"]["name"]
    assert options.data.target_variable == "chlor_a"
    assert options.data.target_name == "chlor_a"
    assert options.data.missing_flag == "cloud_flag"
    assert options.data.land_flag == "land_flag"
    assert options.data.lat_bounds is not None
    # set_data_config never touches the gridder; it stays at its default.
    assert options.gridder.tile_size == (64, 64)
    assert options.split.is_resolved() is False


def test_default_with_data_leaves_gridder_at_default():
    ds, metadata = make_demo_ds(days=120, lat_size=16, lon_size=16, seed=42)

    options = Options.default(data=ds, metadata=metadata)

    assert options.data.target_variable == "chlor_a"
    # The gridder is used exactly as configured -- no dataset-derived tiling.
    assert options.gridder.tile_size == (64, 64)


def test_default_smoke_test_uses_small_gridder():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=42)

    options = Options.default(data=ds, metadata=metadata, smoke_test=True)

    assert options.gridder.tile_size == (16, 16)
    assert options.gridder.time_chunk == 10
    assert options.fit.epochs == 2


def test_data_options_log_target_defaults_false():
    assert Options.default().data.log_target is False


def test_set_up_data_options_cloud_mode_defaults_to_synthetic_bank():
    ds, _ = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=1)
    options = Options.default()

    options.set_up_data_options(
        ds, target="chlor_a", missing_flag="cloud_flag", land_flag="land_flag"
    )

    # cloud_mode is not a required argument; default is "synthetic_bank".
    assert options.data.cloud_mode == "synthetic_bank"
    assert options.data.cloud_coverage == 0.4


def test_set_up_data_options_applies_cloud_options_dict():
    ds, _ = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=1)
    options = Options.default()

    options.set_up_data_options(
        ds,
        target="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        cloud_options={
            "cloud_coverage": 0.6,
            "cloud_blob_sigma": 8.0,
            "cloud_seed": 7,
        },
    )

    assert options.data.cloud_coverage == 0.6
    assert options.data.cloud_blob_sigma == 8.0
    assert options.data.cloud_seed == 7


def test_set_up_data_options_shift_mode_options():
    ds, _ = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=1)
    options = Options.default()

    options.set_up_data_options(
        ds,
        target="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        cloud_mode="shift",
        cloud_options={"missing_flag_shift": 5},
    )

    assert options.data.cloud_mode == "shift"
    assert options.data.missing_flag_shift == 5


def test_set_up_data_options_rejects_option_not_valid_for_mode():
    ds, _ = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=1)
    options = Options.default()

    with pytest.raises(ValueError, match="do not apply to cloud_mode='shift'"):
        options.set_up_data_options(
            ds,
            target="chlor_a",
            missing_flag="cloud_flag",
            land_flag="land_flag",
            cloud_mode="shift",
            cloud_options={"cloud_coverage": 0.5},
        )


def test_set_up_data_options_rejects_unknown_cloud_mode():
    ds, _ = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=1)
    options = Options.default()

    with pytest.raises(ValueError, match="cloud_mode must be one of"):
        options.set_up_data_options(
            ds,
            target="chlor_a",
            missing_flag="cloud_flag",
            land_flag="land_flag",
            cloud_mode="bogus",
        )


def test_set_up_data_options_validates_cloud_option_values():
    ds, _ = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=1)
    options = Options.default()

    with pytest.raises(ValueError, match="cloud_coverage must be between"):
        options.set_up_data_options(
            ds,
            target="chlor_a",
            missing_flag="cloud_flag",
            land_flag="land_flag",
            cloud_options={"cloud_coverage": 1.5},
        )


def test_cloud_options_for_lists_valid_keys():
    from mindthegap import cloud_options_for

    assert cloud_options_for("synthetic_bank") == (
        "cloud_coverage",
        "cloud_blob_sigma",
        "cloud_time_sigma",
        "cloud_seed",
    )
    assert cloud_options_for("shift") == ("missing_flag_shift", "cloud_seed")
    with pytest.raises(ValueError, match="cloud_mode must be one of"):
        cloud_options_for("bogus")


def test_prepare_model_data_reads_config_from_options():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata, smoke_test=True, seed=1)
    options.data.log_target = False
    options.data.n_temporal_lags = 2
    options.set_data_config(data=ds, metadata=metadata)
    train_validation_dates(ds.time, options, seed=1, verbose=False)

    from mindthegap import prepare_model_data

    prepare_model_data(ds, options, mode="train")

    assert options.data.is_resolved()
    assert options.data.transforms["temporal_lags"] == 2
    assert options.data.transforms["target"] == "none"
    std_map = options.data.standardization
    assert options.data.target_mean == float(std_map["full_target"]["mean"])
    assert options.data.target_std == float(std_map["full_target"]["std"])


def test_prepare_model_data_defaults_output_chunks_from_gridder():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata, smoke_test=True)
    train_validation_dates(ds.time, options, seed=1, verbose=False)

    from mindthegap import prepare_model_data

    output = prepare_model_data(ds, options, mode="train")

    time_chunk = options.gridder.time_chunk
    assert output.chunks["time"][0] == time_chunk


def test_smoke_test_caps_epochs():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)

    options = Options.default(data=ds, metadata=metadata, smoke_test=True)

    assert options.smoke_test is True
    assert options.fit.epochs == 2


def test_options_verbose_default_true_and_roundtrips():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)

    assert options.verbose is True
    options.verbose = False
    assert Options.from_dict(options.to_dict()).verbose is False


def test_prepare_model_data_accepts_full_options():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata, smoke_test=True)
    options.verbose = False
    train_validation_dates(ds.time, options, seed=1, verbose=False)

    from mindthegap import prepare_model_data

    output = prepare_model_data(ds, options, mode="train")

    # train_dates and chunks come from options.split / options.gridder
    assert output.chunks["time"][0] == options.gridder.time_chunk
    assert options.data.is_resolved()


def test_prepare_model_data_resolves_split_when_unset():
    # mode="train" auto-chooses the split from options.split when the caller
    # has not already run train_validation_dates.
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata, smoke_test=True, seed=3)

    from mindthegap import prepare_model_data

    assert options.split.is_resolved() is False
    prepare_model_data(ds, options, mode="train")
    assert options.split.is_resolved() is True
    assert len(options.split.train_dates) > 0
    assert len(options.split.val_dates) > 0


def test_prepare_model_data_errors_on_inconsistent_split():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)
    options.split.train_dates = ["1990-01-01"]
    options.split.val_dates = ["1990-01-02"]

    from mindthegap import prepare_model_data

    with pytest.raises(ValueError, match="inconsistent with ds"):
        prepare_model_data(ds, options, mode="train")


def test_prepare_model_data_reads_add_geo_from_options():
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)
    options.data.add_geo = True
    options.verbose = False
    train_validation_dates(ds.time, options, seed=1, verbose=False)

    from mindthegap import prepare_model_data

    prepare_model_data(ds, options, mode="train")

    assert options.data.add_geo is True
    assert options.data.transforms["add_geo"] is True
    assert "x_geo" in options.data.input_names


def test_config_roundtrips_through_dict():
    ds, metadata = make_demo_ds(days=120, lat_size=16, lon_size=16, seed=42)
    options = Options.default(data=ds, metadata=metadata)
    train_validation_dates(ds.time, options, method="random", seed=1)

    assert Options.from_dict(options.to_dict()) == options


def test_train_validation_dates_accepts_full_options():
    ds, metadata = make_demo_ds(days=60, lat_size=16, lon_size=16, seed=2)
    options = Options.default(data=ds, metadata=metadata, seed=42)

    result = train_validation_dates(ds.time, options, method="random")

    # The full Options is returned, and the split records the resolved seed.
    assert result is options
    assert options.split.is_resolved()
    assert options.split.seed == options.resolved_split_seed() == 42


def test_default_seed_is_random_int_and_stored():
    a = Options.default()
    b = Options.default()

    assert isinstance(a.seed, int)
    assert isinstance(b.seed, int)
    # Materialized once; per-stage seeds inherit it by default.
    assert a.resolved_split_seed() == a.seed
    assert a.resolved_shuffle_seed() == a.seed
    # Two independent constructions draw different random seeds.
    assert a.seed != b.seed


def test_explicit_seed_pins_global():
    options = Options.default(seed=42)

    assert options.seed == 42
    assert options.resolved_split_seed() == 42
    assert options.resolved_shuffle_seed() == 42


def test_per_stage_seeds_override_global():
    options = Options.default(seed=42)
    options.split.seed = 7
    options.fit.shuffle_seed = 9

    assert options.resolved_split_seed() == 7
    assert options.resolved_shuffle_seed() == 9
    assert options.resolved_seed() == 42


def test_tf_seed_not_set_by_default():
    options = Options.default(seed=42)

    assert options.fit.tf_seed is None


def test_seed_tensorflow_requires_seed_and_records():
    options = Options.default(seed=42)

    with pytest.raises(ValueError, match="requires an explicit"):
        options.seed_tensorflow(None)

    options.seed_tensorflow(123)
    assert options.fit.tf_seed == 123


def test_seed_roundtrips_through_dict():
    options = Options.default(seed=42)
    options.split.seed = 7
    options.fit.shuffle_seed = 9
    options.seed_tensorflow(123)

    restored = Options.from_dict(options.to_dict())
    assert restored.seed == 42
    assert restored.split.seed == 7
    assert restored.fit.shuffle_seed == 9
    assert restored.fit.tf_seed == 123
