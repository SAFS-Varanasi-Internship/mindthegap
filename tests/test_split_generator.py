import numpy as np
import pandas as pd
import pytest

from mindthegap import (
    Options,
    SplitOptions,
    build_standardized_lazy,
    demo_data,
    make_generator,
    train_validation_dates,
)


def _prepared(days=60, seed=3):
    ds, metadata = demo_data(days=days, lat_size=16, lon_size=16, seed=seed)
    options = Options.default(data=ds, metadata=metadata)
    ds_std, _ = build_standardized_lazy(
        ds, options=options.data, gridder=options.gridder
    )
    return ds, ds_std, options


def test_split_starts_unresolved_after_set_config():
    ds, metadata = demo_data(days=60, lat_size=16, lon_size=16, seed=1)
    options = Options.default(data=ds, metadata=metadata)

    assert options.split.is_resolved() is False
    assert options.split.method == "random"


def test_train_validation_dates_random_spacing():
    ds, _, options = _prepared()

    train_validation_dates(
        ds.time,
        options.split,
        method="random",
        n_train=10,
        n_val=5,
        min_day_difference=2,
        seed=1,
    )

    assert options.split.method == "random"
    assert len(options.split.train_dates) == 10
    assert len(options.split.val_dates) == 5
    all_dates = pd.to_datetime(
        options.split.train_dates + options.split.val_dates
    ).sort_values()
    gaps = np.diff(all_dates.values).astype("timedelta64[D]").astype(int)
    assert (gaps >= 2).all()


def test_train_validation_dates_random_infeasible_spacing_errors():
    ds, _, options = _prepared(days=10)

    with pytest.raises(ValueError, match="at most"):
        train_validation_dates(
            ds.time,
            options.split,
            method="random",
            n_train=8,
            n_val=8,
            min_day_difference=2,
        )


def test_train_validation_dates_manual_selects_windows():
    ds, _, options = _prepared(days=60)

    train_validation_dates(
        ds.time,
        options.split,
        method="manual",
        train_slice=slice("2020-01-01", "2020-01-20"),
        val_slice=slice("2020-01-21", "2020-01-31"),
    )

    assert options.split.method == "manual"
    assert all("2020-01" in d for d in options.split.train_dates)
    assert options.split.is_resolved()


def test_train_validation_dates_manual_empty_slice_errors():
    ds, _, options = _prepared(days=60)

    with pytest.raises(ValueError, match="selects no dates"):
        train_validation_dates(
            ds.time,
            options.split,
            method="manual",
            train_slice=slice("2019-01-01", "2019-02-01"),
            val_slice=slice("2020-01-21", "2020-01-31"),
        )


def test_make_generator_requires_resolved_split():
    _, ds_std, options = _prepared()

    with pytest.raises(ValueError, match="train_validation_dates"):
        make_generator(ds_std, options=options)


def test_make_generator_returns_datasets_and_steps():
    ds, ds_std, options = _prepared(days=60)
    train_validation_dates(
        ds.time,
        options.split,
        method="random",
        n_train=12,
        n_val=6,
        seed=1,
    )

    train_ds, val_ds, train_steps, val_steps = make_generator(
        ds_std, options=options
    )

    assert train_steps >= 1
    assert val_steps >= 1
    batch = next(iter(train_ds.take(1)))
    x, y = batch
    assert x.shape[-1] == len(options.data.input_names)
    assert y.shape[-1] == 1


def test_build_standardized_lazy_defaults_chunks_from_gridder():
    ds, metadata = demo_data(days=40, lat_size=16, lon_size=16, seed=2)
    options = Options.default(data=ds, metadata=metadata)

    ds_std, _ = build_standardized_lazy(
        ds, options=options.data, gridder=options.gridder
    )

    assert ds_std.chunksizes["lat"][0] == options.gridder.tile_size[0]


def test_split_roundtrips_through_dict():
    ds, _, options = _prepared()
    train_validation_dates(
        ds.time, options.split, method="random", n_train=8, n_val=4, seed=1
    )

    assert Options.from_dict(options.to_dict()) == options
