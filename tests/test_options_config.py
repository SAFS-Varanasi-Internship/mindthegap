import pytest

from mindthegap import (
    FitOptions,
    GridderOptions,
    Options,
    SplitOptions,
    demo_data,
    train_validation_dates,
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


def test_fit_resolve_scales_batch_size():
    fit = FitOptions(epochs=50, patience=10).resolve_for((128, 128))
    assert fit.batch_size == max(1, min(16, fit.pixel_budget // (128 * 128)))
    assert fit.epochs == 50
    assert fit.patience == 10


def test_split_defaults():
    split = SplitOptions()

    assert split.method == "random"
    assert split.train_fraction == 0.8
    assert split.val_fraction == 0.2
    assert split.train_dates == []
    assert split.val_dates == []
    assert split.is_resolved() is False
    assert split.training_period() is None


def test_train_validation_dates_random_populates_split():
    ds, _ = demo_data(days=100, lat_size=8, lon_size=8, seed=2)
    split = SplitOptions()

    result = train_validation_dates(ds.time, split, method="random", seed=7)

    assert result is split
    assert split.method == "random"
    assert split.is_resolved()
    assert len(split.train_dates) == 80
    assert len(split.val_dates) == 20
    assert not (set(split.train_dates) & set(split.val_dates))


def test_train_validation_dates_random_is_deterministic():
    ds, _ = demo_data(days=60, lat_size=8, lon_size=8, seed=3)

    a = train_validation_dates(ds.time, SplitOptions(), method="random", seed=5)
    b = train_validation_dates(ds.time, SplitOptions(), method="random", seed=5)

    assert a.train_dates == b.train_dates
    assert a.val_dates == b.val_dates


def test_train_validation_dates_manual_selects_windows():
    ds, _ = demo_data(days=120, lat_size=8, lon_size=8, seed=2)
    split = SplitOptions()

    train_validation_dates(
        ds.time,
        split,
        method="manual",
        train_slice=slice("2020-01-01", "2020-02-01"),
        val_slice=slice("2020-02-02", "2020-02-20"),
    )

    assert split.method == "manual"
    assert split.train_dates[0] == "2020-01-01"
    assert split.is_resolved()


def test_train_validation_dates_manual_errors_on_empty_slice():
    ds, _ = demo_data(days=60, lat_size=8, lon_size=8, seed=2)

    with pytest.raises(ValueError, match="selects no dates"):
        train_validation_dates(
            ds.time,
            SplitOptions(),
            method="manual",
            train_slice=slice("1990-01-01", "1990-02-01"),
            val_slice=slice("2020-01-05", "2020-01-20"),
        )


def test_train_validation_dates_rejects_unknown_method():
    ds, _ = demo_data(days=30, lat_size=8, lon_size=8, seed=2)

    with pytest.raises(ValueError, match="Unknown method"):
        train_validation_dates(ds.time, SplitOptions(), method="bogus")


def test_set_config_resolves_gridder_and_fit_not_split():
    ds, _ = demo_data(days=120, lat_size=16, lon_size=16, seed=42)
    options = Options.default()

    result = options.set_config(ds)

    assert result is options
    assert options.gridder.tile_size == (16, 16)
    assert options.split.is_resolved() is False


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
    assert options.gridder.tile_size == (16, 16)
    assert options.split.is_resolved() is False


def test_default_with_data_resolves_configuration():
    ds, metadata = demo_data(days=120, lat_size=16, lon_size=16, seed=42)

    options = Options.default(data=ds, metadata=metadata)

    assert options.data.target_variable == "chlor_a"
    assert options.gridder.tile_size == (16, 16)


def test_data_options_log_target_defaults_false():
    assert Options.default().data.log_target is False


def test_build_standardized_lazy_reads_config_from_options():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default()
    options.data.log_target = False
    options.data.n_temporal_lags = 2
    options.set_data_config(data=ds, metadata=metadata)

    from mindthegap import build_standardized_lazy

    _, stats = build_standardized_lazy(ds, options=options.data)

    assert options.data.is_resolved()
    assert options.data.transforms["temporal_lags"] == 2
    assert options.data.transforms["target"] == "none"
    assert options.data.target_mean == float(stats["full_target"][0])
    assert options.data.target_std == float(stats["full_target"][1])


def test_build_standardized_lazy_defaults_output_chunks_from_gridder():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)

    from mindthegap import build_standardized_lazy

    output, _ = build_standardized_lazy(
        ds, options=options.data, gridder=options.gridder
    )

    time_chunk = options.gridder.time_chunk
    assert output.chunks["time"][0] == time_chunk


def test_smoke_test_caps_epochs():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)

    options = Options.default(data=ds, metadata=metadata, smoke_test=True)

    assert options.smoke_test is True
    assert options.fit.epochs == 2


def test_options_verbose_default_true_and_roundtrips():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)

    assert options.verbose is True
    options.verbose = False
    assert Options.from_dict(options.to_dict()).verbose is False


def test_build_standardized_lazy_accepts_full_options():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)
    options.verbose = False
    train_validation_dates(ds.time, options.split, seed=1, verbose=False)

    from mindthegap import build_standardized_lazy

    output, _ = build_standardized_lazy(ds, options)

    # train_dates and chunks come from options.split / options.gridder
    assert output.chunks["time"][0] == options.gridder.time_chunk
    assert options.data.is_resolved()


def test_build_standardized_lazy_errors_when_split_unset():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)

    from mindthegap import build_standardized_lazy

    with pytest.raises(ValueError, match="options.split has no dates"):
        build_standardized_lazy(ds, options)


def test_build_standardized_lazy_errors_on_inconsistent_split():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)
    options.split.train_dates = ["1990-01-01"]
    options.split.val_dates = ["1990-01-02"]

    from mindthegap import build_standardized_lazy

    with pytest.raises(ValueError, match="inconsistent with ds"):
        build_standardized_lazy(ds, options)


def test_build_standardized_lazy_reads_add_geo_from_options():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=3)
    options = Options.default(data=ds, metadata=metadata)
    options.data.add_geo = True
    train_validation_dates(ds.time, options.split, seed=1, verbose=False)

    from mindthegap import build_standardized_lazy

    build_standardized_lazy(ds, options, verbose=False)

    assert options.data.add_geo is True
    assert options.data.transforms["add_geo"] is True
    assert "x_geo" in options.data.input_names


def test_config_roundtrips_through_dict():
    ds, metadata = demo_data(days=120, lat_size=16, lon_size=16, seed=42)
    options = Options.default(data=ds, metadata=metadata)
    train_validation_dates(ds.time, options.split, method="random", seed=1)

    assert Options.from_dict(options.to_dict()) == options
