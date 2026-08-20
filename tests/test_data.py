import numpy as np
import pytest
import xarray as xr

from mindthegap import Options, train_validation_dates
from mindthegap.data import prepare_model_data, _resolve_region
from mindthegap.options import DataOptions
from conftest import make_demo_ds


def _full_options(ds, metadata, *, resolve_split=True, seed=1, **cloud_kwargs):
    """Full Options resolved for the demo dataset.

    ``prepare_model_data`` reads every setting from ``options``; cloud
    configuration is set on ``options.data`` (never a call argument).
    """
    options = Options.default(data=ds, metadata=metadata, smoke_test=True, seed=seed)
    options.verbose = False
    for key, value in cloud_kwargs.items():
        setattr(options.data, key, value)
    if resolve_split:
        train_validation_dates(ds.time, options, seed=seed, verbose=False)
    return options


def test_make_demo_ds_has_requested_shape_and_flags():
    ds, metadata = make_demo_ds(days=10, lat_size=8, lon_size=12, seed=7)

    assert ds.sizes == {"time": 10, "lat": 8, "lon": 12}
    assert {"chlor_a", "cloud_flag", "land_flag"} == set(ds.data_vars)
    assert ds.cloud_flag.dtype == np.dtype("int8")
    assert ds.land_flag.dtype == np.dtype("int8")
    assert np.isnan(ds.chlor_a.where(ds.land_flag == 1)).all()
    assert metadata["target"] == {"name": "chlor_a", "units": "mg m-3"}
    assert metadata["variables"]["missing_flag"] == "cloud_flag"


def test_make_demo_ds_is_deterministic():
    first, first_metadata = make_demo_ds(days=5, seed=3)
    second, second_metadata = make_demo_ds(days=5, seed=3)

    assert first.identical(second)
    assert first_metadata == second_metadata


@pytest.mark.parametrize(
    "region, error, message",
    [
        ("unknown", ValueError, "Unknown region"),
        ([10, 20, 50], ValueError, "must contain"),
        ([10, 20, 50, "west"], TypeError, "numeric"),
        ([-1000, 20, 50, 70], ValueError, "latitude bounds"),
        ([10, 20, -1000, 70], ValueError, "longitude bounds"),
        ([20, 10, 50, 70], ValueError, "latitude bounds"),
        ([10, 20, 70, 50], ValueError, "longitude bounds"),
    ],
)
def test_resolve_region_rejects_invalid_region(region, error, message):
    with pytest.raises(error, match=message):
        _resolve_region(region)


def test_resolve_region_named_region_normalizes_name():
    bounds, name = _resolve_region("Arabian_Sea")
    assert name == "arabian sea"
    assert len(bounds) == 4


def test_resolve_region_vector_has_no_name():
    bounds, name = _resolve_region(np.array([10, 20, 50, 70]))
    assert name is None
    assert bounds == (10.0, 20.0, 50.0, 70.0)



def test_prepare_model_data_broadcasts_static_land_mask():
    ds, metadata = make_demo_ds(days=12, lat_size=8, lon_size=8)
    ds["land_flag"] = ds["land_flag"].isel(time=0, drop=True)
    options = _full_options(ds, metadata)

    standardized = prepare_model_data(ds, options, mode="train")

    assert standardized["land_flag"].dims == ("time", "lat", "lon")
    xr.testing.assert_equal(
        standardized["land_flag"].isel(time=0, drop=True),
        standardized["land_flag"].isel(time=-1, drop=True),
    )


def test_prepare_model_data_crops_to_unet_multiple():
    from mindthegap.model import unet_spatial_multiple

    multiple = unet_spatial_multiple()
    # Sizes deliberately not multiples of the U-Net factor.
    ds, metadata = make_demo_ds(
        days=12, lat_size=multiple + 3, lon_size=multiple + 5
    )
    options = _full_options(ds, metadata)

    standardized = prepare_model_data(ds, options, mode="train")

    assert standardized.sizes["lat"] % multiple == 0
    assert standardized.sizes["lon"] % multiple == 0
    assert standardized.sizes["lat"] == multiple
    assert standardized.sizes["lon"] == multiple


def test_synthetic_cloud_cube_shape_and_coverage():
    from mindthegap.data import synthetic_cloud_cube

    rng = np.random.default_rng(0)
    cube = synthetic_cloud_cube(
        20, 32, 32, coverage=0.4, blob_sigma=4.0, time_sigma=2.0, rng=rng
    )
    assert cube.shape == (20, 32, 32)
    assert cube.dtype == np.dtype("bool")
    # Coverage should be close to the requested fraction.
    assert abs(cube.mean() - 0.4) < 0.05
    # Zero coverage yields an all-clear cube.
    empty = synthetic_cloud_cube(4, 8, 8, coverage=0.0)
    assert not empty.any()


def test_prepare_model_data_synthetic_clouds_hide_observed_ocean():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata, cloud_mode="synthetic", cloud_seed=1)

    standardized = prepare_model_data(ds, options, mode="train")

    estimate = standardized["estimate_flag"].values == 1
    land = standardized["land_flag"].values == 1
    unavailable = standardized["unavailable_flag"].values == 1

    assert estimate.any()
    # Synthetic clouds only hide real ocean observations.
    assert not (estimate & land).any()
    assert not (estimate & unavailable).any()
    # The true value is hidden from observed_target but retained in full_target.
    assert standardized["observed_target"].where(
        standardized["estimate_flag"] == 1
    ).isnull().all()


def test_prepare_model_data_synthetic_clouds_are_reproducible():
    ds, metadata = make_demo_ds(days=16, lat_size=16, lon_size=16, seed=5)

    def run(cloud_seed):
        options = _full_options(
            ds, metadata, cloud_mode="synthetic", cloud_seed=cloud_seed
        )
        out = prepare_model_data(ds, options, mode="train")
        return out

    a = run(42)
    b = run(42)
    c = run(99)

    assert np.array_equal(
        a["estimate_flag"].values, b["estimate_flag"].values
    )
    assert not np.array_equal(
        a["estimate_flag"].values, c["estimate_flag"].values
    )


def test_prepare_model_data_shift_cloud_mode_uses_future_clouds():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    options = _full_options(
        ds, metadata, cloud_mode="shift", missing_flag_shift=5
    )

    standardized = prepare_model_data(ds, options, mode="train")

    estimate = standardized["estimate_flag"].values == 1
    land = standardized["land_flag"].values == 1
    assert estimate.any()
    assert not (estimate & land).any()


@pytest.mark.parametrize(
    "cloud_kwargs",
    [
        {"cloud_mode": "synthetic", "cloud_coverage": 0.5, "cloud_seed": 1},
        {"cloud_mode": "shift", "missing_flag_shift": 5},
    ],
)
def test_prepare_model_data_flags_are_mutually_exclusive(cloud_kwargs):
    # A pixel with both a real cloud and a synthetic cloud must count as real
    # only: estimate_flag never overlaps the other cloud/state flags. (land and
    # unavailable can co-occur in the raw demo data -- a real cloud reported
    # over land -- which is unrelated to synthetic-cloud creation.)
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata, **cloud_kwargs)

    standardized = prepare_model_data(ds, options, mode="train")

    estimate = standardized["estimate_flag"].values == 1
    unavailable = standardized["unavailable_flag"].values == 1
    land = standardized["land_flag"].values == 1
    observed = standardized["observed_flag"].values == 1

    assert estimate.any()
    assert not (estimate & unavailable).any()
    assert not (estimate & land).any()
    assert not (estimate & observed).any()


def test_prepare_model_data_test_mode_reuses_stats_and_adds_clouds():
    # mode="test" adds synthetic clouds + flags like train, but standardizes
    # with the statistics recorded during a prior train run (no recompute) and
    # does not modify options.
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata, cloud_mode="synthetic", cloud_seed=1)

    prepare_model_data(ds, options, mode="train")
    recorded = dict(options.data.standardization)
    target_mean_before = options.data.target_mean

    test_out = prepare_model_data(ds, options, mode="test")

    # Synthetic clouds + flags are present, over the whole record.
    assert (test_out["estimate_flag"].values == 1).any()
    assert "unavailable_flag" in test_out
    assert test_out.sizes["time"] == ds.sizes["time"]
    # options is not mutated by a test run.
    assert options.data.standardization == recorded
    assert options.data.target_mean == target_mean_before


def test_prepare_model_data_test_and_gapfill_require_prior_stats():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata)
    # Fresh options: no standardization recorded yet.
    options.data.standardization = {}

    for mode in ("test", "gapfill"):
        with pytest.raises(ValueError, match="standardization is empty"):
            prepare_model_data(ds, options, mode=mode)

def test_prepare_model_data_rejects_bad_mode():
    ds, metadata = make_demo_ds(days=12, lat_size=8, lon_size=8, seed=3)
    options = _full_options(ds, metadata)
    with pytest.raises(ValueError, match="mode must be"):
        prepare_model_data(ds, options, mode="bogus")


def test_prepare_model_data_dry_run_matches_full_shape_and_is_pure():
    # dry_run is a fast probe: it returns the lazy skeleton with the exact
    # channel set/order and cropped shape a full train run produces, but does
    # not compute statistics, generate synthetic clouds, resolve the split, or
    # mutate options.
    ds, metadata = make_demo_ds(days=30, lat_size=16, lon_size=16, seed=3)
    options = _full_options(
        ds, metadata, resolve_split=False, cloud_mode="synthetic", cloud_seed=1
    )

    skeleton = prepare_model_data(ds, options, mode="train", dry_run=True)

    # Pure query: options untouched, split unresolved, no recorded stats.
    assert options.data.input_names == []
    assert not options.split.is_resolved()
    assert not options.data.standardization

    # A full run produces the same channel set and cropped spatial shape.
    full = prepare_model_data(ds, options, mode="train")
    assert list(skeleton.data_vars) == list(full.data_vars)
    assert skeleton.sizes["lat"] == full.sizes["lat"]
    assert skeleton.sizes["lon"] == full.sizes["lon"]


def test_prepare_model_data_dry_run_returns_lazy_without_computing():
    # The skeleton must stay lazy (dask-backed): probing shape/channels should
    # never trigger the expensive standardization/cloud computation.
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    ds = ds.chunk({"time": 5})
    options = _full_options(ds, metadata, resolve_split=False)

    skeleton = prepare_model_data(ds, options, mode="train", dry_run=True)

    assert skeleton["observed_target"].chunks is not None
    # No statistics were computed, so the standardization is the identity.
    assert not options.data.standardization


def test_prepare_model_data_std_target_false_leaves_target_raw():
    # Default: the target is not standardized, so target_mean/std are the
    # identity transform and full_target stays in raw (or log) units.
    ds, metadata = make_demo_ds(days=30, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata, cloud_mode="shift", std_target=False)

    prepare_model_data(ds, options, mode="train")

    assert options.data.target_mean == 0.0
    assert options.data.target_std == 1.0
    assert options.data.standardization["full_target"]["applied"] is False
    assert options.data.standardization["observed_target"]["applied"] is False


def test_prepare_model_data_std_target_shares_one_scale():
    # std_target=True computes a single (mean, std) from the masked
    # observed_target over the training dates and applies it to observed_target,
    # its temporal lags, and full_target so inputs and label share one scale.
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = _full_options(
        ds, metadata, cloud_mode="shift", std_target=True, n_temporal_lags=1
    )

    prepare_model_data(ds, options, mode="train")

    std_map = options.data.standardization
    assert std_map["full_target"]["applied"]
    assert std_map["observed_target"]["applied"]
    assert std_map["observed_target_m1"]["applied"]
    assert std_map["observed_target_p1"]["applied"]

    # full_target and observed_target (and lags) share the SAME mean/std.
    target_group = [
        "full_target",
        "observed_target",
        "observed_target_m1",
        "observed_target_p1",
    ]
    means = {std_map[name]["mean"] for name in target_group}
    stds = {std_map[name]["std"] for name in target_group}
    assert len(means) == 1
    assert len(stds) == 1

    # The recorded target stats match, and std should be non-trivial.
    assert options.data.target_mean == std_map["full_target"]["mean"]
    assert options.data.target_std == std_map["full_target"]["std"]
    assert options.data.target_std != 1.0


def test_prepare_model_data_std_target_stats_from_train_dates_only():
    # The target statistics must be computed from the masked observed_target
    # restricted to the training dates. Verify by recomputing directly and
    # comparing to the recorded target stats.
    import numpy as np
    import pandas as pd

    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=7)
    options = _full_options(ds, metadata, cloud_mode="shift", std_target=True)

    prepared = prepare_model_data(ds, options, mode="train")

    # observed_target over the training dates should have ~zero mean / unit std
    # after standardization, because the target stats were fit on exactly that
    # slice. Select the training dates from the returned (train+val) dataset.
    train_dates = pd.to_datetime(options.split.train_selection())
    obs_train = prepared["observed_target"].sel(time=train_dates)
    mean = float(obs_train.mean().compute())
    std = float(obs_train.std().compute())
    assert abs(mean) < 1e-4
    assert abs(std - 1.0) < 1e-4


def test_prepare_model_data_std_features_standardizes_only_listed():
    # Only the features named in std_features are standardized; others are left
    # raw. Each standardized feature uses its own statistics.
    ds, metadata = make_demo_ds(days=30, lat_size=16, lon_size=16, seed=3)
    ds = ds.assign(sst=ds["chlor_a"] * 2.0 + 5.0, wind=ds["chlor_a"] * 0.1)
    options = _full_options(ds, metadata, cloud_mode="shift")
    options.data.features = ["sst", "wind"]
    options.data.std_features = ["sst"]
    options.data.std_target = False

    prepare_model_data(ds, options, mode="train")

    assert options.data.standardization["sst"]["applied"] is True
    assert options.data.standardization["sst"]["std"] != 1.0
    assert options.data.standardization["wind"]["applied"] is False
    assert options.data.standardization["wind"]["std"] == 1.0


def test_prepare_model_data_rejects_target_in_features():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata, cloud_mode="shift")
    options.data.features = ["chlor_a"]

    with pytest.raises(ValueError, match="never the target"):
        prepare_model_data(ds, options, mode="train")


def test_prepare_model_data_rejects_observed_target_in_features():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata, cloud_mode="shift")
    options.data.features = ["observed_target"]

    with pytest.raises(ValueError, match="never the target"):
        prepare_model_data(ds, options, mode="train")


def test_prepare_model_data_rejects_std_features_not_in_features():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata, cloud_mode="shift")
    options.data.std_features = ["not_a_feature"]

    with pytest.raises(ValueError, match="must be a subset"):
        prepare_model_data(ds, options, mode="train")


def test_prepare_model_data_std_target_reused_in_test_mode():
    # test mode reuses the recorded target standardization (no recompute) and
    # applies it to full_target so the label matches training.
    ds, metadata = make_demo_ds(days=40, lat_size=16, lon_size=16, seed=3)
    options = _full_options(ds, metadata, cloud_mode="shift", std_target=True)

    prepare_model_data(ds, options, mode="train")
    recorded_std = options.data.standardization["full_target"]["std"]

    test_out = prepare_model_data(ds, options, mode="test")

    # The standardized full_target in test mode has ~unit spread (it was scaled
    # by the recorded std), confirming the recorded stats were applied.
    applied_std = float(test_out["full_target"].std().compute())
    assert 0.5 < applied_std < 1.5
    # options unchanged by the test run.
    assert options.data.standardization["full_target"]["std"] == recorded_std


def test_data_options_rejects_bad_cloud_mode():
    with pytest.raises(ValueError, match="cloud_mode"):
        DataOptions(
            target_variable="chlor_a",
            missing_flag="cloud_flag",
            land_flag="land_flag",
            cloud_mode="nonsense",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"days": 0},
        {"lat_size": 0},
        {"lon_size": 0},
        {"cloud_fraction": -0.1},
        {"cloud_fraction": 1.1},
    ],
)
def test_make_demo_ds_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        make_demo_ds(**kwargs)


def test_demo_data_rejects_unknown_dataset():
    from mindthegap.data import demo_data

    with pytest.raises(ValueError, match="Unknown dataset"):
        demo_data(dataset="unknown")
