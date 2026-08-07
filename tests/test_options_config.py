import pytest

from mindthegap import (
    FitOptions,
    GridderOptions,
    Options,
    SplitOptions,
    demo_data,
)


def test_gridder_resolve_infers_tile_and_time_chunk():
    ds, _ = demo_data(days=60, lat_size=40, lon_size=40, seed=1)

    gridder = GridderOptions().resolve_for(ds)

    assert gridder.tile_size == (40, 40)
    assert gridder.time_chunk == 10


def test_gridder_resolve_caps_and_aligns_tile():
    ds, _ = demo_data(days=30, lat_size=100, lon_size=100, seed=1)

    gridder = GridderOptions(tile_upper_limit=70, tile_multiple=8).resolve_for(ds)

    assert gridder.tile_size == (64, 64)


def test_fit_resolve_scales_batch_and_short_run_schedule():
    fit = FitOptions(epochs=50, patience=10).resolve_for(
        (128, 128), short_run=False
    )
    assert fit.batch_size == max(1, min(16, fit.pixel_budget // (128 * 128)))
    assert fit.epochs == 50
    assert fit.patience == 10

    short = FitOptions().resolve_for((16, 16), short_run=True)
    assert short.epochs == short.short_run_epochs
    assert short.patience == short.short_run_patience


def test_split_resolve_short_run_uses_fractional_split():
    ds, _ = demo_data(days=100, lat_size=8, lon_size=8, seed=2)

    split = SplitOptions().resolve_for(ds)

    assert split.short_run is True
    assert split.train_start == "2020-01-01"
    assert split.train_slice() == slice("2020-01-01", split.train_end)
    assert split.val_slice() == slice(split.train_end, split.val_end)
    assert split.training_period() == f"{split.train_start} to {split.train_end}"


def test_split_resolve_long_run_uses_year_windows():
    ds, _ = demo_data(days=200, lat_size=8, lon_size=8, seed=2)

    split = SplitOptions(train_years=1, val_years=1).resolve_for(ds)

    assert split.short_run is False


def test_set_config_resolves_all_sections():
    ds, _ = demo_data(days=120, lat_size=16, lon_size=16, seed=42)
    options = Options.default()

    result = options.set_config(ds)

    assert result is options
    assert options.gridder.tile_size == (16, 16)
    assert options.split.short_run is True
    assert options.fit.epochs == options.fit.short_run_epochs
    assert options.data.training_period == options.split.training_period()


def test_set_data_config_populates_data_from_metadata():
    ds, metadata = demo_data(days=120, lat_size=16, lon_size=16, seed=42)
    options = Options.default()

    result = options.set_data_config(data=ds, metadata=metadata)

    assert result is options
    assert options.data.source == metadata["dataset"]["name"]
    assert options.data.target_variable == "chlor_a"
    assert options.data.target_name == "chlor_a"
    assert options.data.missing_flag == "cloud_flag"
    assert options.data.land_flag == "land_flag"
    assert options.data.lat_bounds is not None
    # non-data sections are resolved too
    assert options.gridder.tile_size == (16, 16)
    assert options.split.short_run is True


def test_default_with_data_resolves_configuration():
    ds, metadata = demo_data(days=120, lat_size=16, lon_size=16, seed=42)

    options = Options.default(data=ds, metadata=metadata)

    assert options.data.target_variable == "chlor_a"
    assert options.gridder.tile_size == (16, 16)


def test_build_standardized_lazy_reads_config_from_options():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default()
    options.data.log_target = False
    options.data.n_temporal_lags = 2
    options.set_data_config(data=ds, metadata=metadata)

    from mindthegap import build_standardized_lazy

    _, stats = build_standardized_lazy(
        ds,
        std_vars=options.data.features,
        options=options.data,
    )

    assert options.data.is_resolved()
    assert options.data.transforms["temporal_lags"] == 2
    assert options.data.transforms["target"] == "none"
    assert options.data.target_mean == float(stats["full_target"][0])
    assert options.data.target_std == float(stats["full_target"][1])


def test_set_config_roundtrips_through_dict():
    ds, _ = demo_data(days=120, lat_size=16, lon_size=16, seed=42)
    options = Options.default().set_config(ds)

    assert Options.from_dict(options.to_dict()) == options
