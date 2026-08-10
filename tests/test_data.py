import numpy as np
import pytest
import xarray as xr

from mindthegap.data import prepare_model_data, demo_data


def test_demo_data_has_requested_shape_and_flags():
    ds, metadata = demo_data(days=10, lat_size=8, lon_size=12, seed=7)

    assert ds.sizes == {"time": 10, "lat": 8, "lon": 12}
    assert {"chlor_a", "cloud_flag", "land_flag"} == set(ds.data_vars)
    assert ds.cloud_flag.dtype == np.dtype("int8")
    assert ds.land_flag.dtype == np.dtype("int8")
    assert np.isnan(ds.chlor_a.where(ds.land_flag == 1)).all()
    assert metadata["dataset"]["name"] == "Synthetic"
    assert metadata["dataset"]["product_id"] == "mindthegap-synthetic"
    assert metadata["dataset"]["loader"] == "synthetic"
    assert (
        metadata["dataset"]["data_source"]
        == "demo_data(dataset='synthetic', days=10, lat_size=8, "
        "lon_size=12, seed=7)"
    )
    assert metadata["target"] == {"name": "chlor_a", "units": "mg m-3"}
    assert metadata["variables"]["missing_flag"] == "cloud_flag"


def test_demo_data_is_deterministic():
    first, first_metadata = demo_data(days=5, seed=3)
    second, second_metadata = demo_data(days=5, seed=3)

    assert first.identical(second)
    assert first_metadata == second_metadata


def test_demo_data_smoke_test_caps_synthetic_size():
    ds, _ = demo_data(
        days=300,
        lat_size=200,
        lon_size=200,
        smoke_test=True,
        smoke_days=30,
        smoke_size=32,
    )

    assert ds.sizes == {"time": 30, "lat": 32, "lon": 32}


def test_demo_data_smoke_test_applies_after_region_selection():
    ds, _ = demo_data(
        region=[10, 25, 50, 70],
        days=60,
        lat_size=64,
        lon_size=64,
        smoke_test=True,
        smoke_days=20,
        smoke_size=8,
    )

    assert ds.sizes["time"] == 20
    assert ds.sizes["lat"] <= 8
    assert ds.sizes["lon"] <= 8
    assert ds.sizes["lat"] > 0 and ds.sizes["lon"] > 0



def test_demo_data_applies_region_and_time_slice():
    ds, metadata = demo_data(
        region=[10, 25, 50, 70],
        time_slice=slice("2020-01-10", "2020-01-20"),
        days=30,
        lat_size=27,
        lon_size=39,
    )

    assert ds.sizes == {"time": 11, "lat": 16, "lon": 21}
    assert metadata["dataset"]["region"] == {
        "lat": [10.0, 25.0],
        "lon": [50.0, 70.0],
    }
    assert (
        metadata["dataset"]["available_period"]
        == "2020-01-10 to 2020-01-20"
    )


def test_demo_data_accepts_named_region():
    ds, metadata = demo_data(
        region="Arabian_Sea",
        days=5,
        lat_size=27,
        lon_size=39,
    )

    assert ds.sizes == {"time": 5, "lat": 27, "lon": 39}
    assert metadata["dataset"]["region_name"] == "arabian sea"


def test_demo_data_accepts_region_vector():
    _, metadata = demo_data(
        region=np.array([10, 20, 50, 70]),
        lat_size=27,
        lon_size=39,
    )

    assert metadata["dataset"]["region"] == {
        "lat": [10.0, 20.0],
        "lon": [50.0, 70.0],
    }
    assert "region_name" not in metadata["dataset"]


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
def test_demo_data_rejects_invalid_region(region, error, message):
    with pytest.raises(error, match=message):
        demo_data(region=region)


def test_prepare_model_data_broadcasts_static_land_mask():
    ds, _ = demo_data(days=12, lat_size=8, lon_size=8)
    ds["land_flag"] = ds["land_flag"].isel(time=0, drop=True)

    standardized, _ = prepare_model_data(
        ds,
        target_variable="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        std_vars=[],
        output_chunks={"time": 4, "lat": 8, "lon": 8},
    )

    assert standardized["land_flag"].dims == ("time", "lat", "lon")
    xr.testing.assert_equal(
        standardized["land_flag"].isel(time=0, drop=True),
        standardized["land_flag"].isel(time=-1, drop=True),
    )


def test_prepare_model_data_crops_to_unet_multiple():
    from mindthegap.model import unet_spatial_multiple

    multiple = unet_spatial_multiple()
    # Sizes deliberately not multiples of the U-Net factor.
    ds, _ = demo_data(days=12, lat_size=multiple + 3, lon_size=multiple + 5)

    standardized, _ = prepare_model_data(
        ds,
        target_variable="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        std_vars=[],
    )

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
    ds, _ = demo_data(days=20, lat_size=16, lon_size=16, seed=3)

    standardized, _ = prepare_model_data(
        ds,
        target_variable="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        std_vars=[],
        cloud_mode="synthetic",
        cloud_seed=1,
    )

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
    ds, _ = demo_data(days=16, lat_size=16, lon_size=16, seed=5)
    common = dict(
        target_variable="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        std_vars=[],
        cloud_mode="synthetic",
    )
    a, _ = prepare_model_data(ds, cloud_seed=42, **common)
    b, _ = prepare_model_data(ds, cloud_seed=42, **common)
    c, _ = prepare_model_data(ds, cloud_seed=99, **common)

    assert np.array_equal(
        a["estimate_flag"].values, b["estimate_flag"].values
    )
    assert not np.array_equal(
        a["estimate_flag"].values, c["estimate_flag"].values
    )


def test_prepare_model_data_shift_cloud_mode_uses_future_clouds():
    ds, _ = demo_data(days=20, lat_size=16, lon_size=16, seed=3)

    standardized, _ = prepare_model_data(
        ds,
        target_variable="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        std_vars=[],
        cloud_mode="shift",
        missing_flag_shift=5,
    )

    estimate = standardized["estimate_flag"].values == 1
    land = standardized["land_flag"].values == 1
    assert estimate.any()
    assert not (estimate & land).any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cloud_mode": "synthetic", "cloud_seed": 1, "coverage": 0.5},
        {"cloud_mode": "shift", "missing_flag_shift": 5},
    ],
)
def test_prepare_model_data_flags_are_mutually_exclusive(kwargs):
    # A pixel with both a real cloud and a synthetic cloud must count as real
    # only: estimate_flag never overlaps the other cloud/state flags. (land and
    # unavailable can co-occur in the raw demo data -- a real cloud reported
    # over land -- which is unrelated to synthetic-cloud creation.)
    ds, _ = demo_data(days=20, lat_size=16, lon_size=16, seed=3)

    standardized, _ = prepare_model_data(
        ds,
        target_variable="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        std_vars=[],
        **kwargs,
    )

    estimate = standardized["estimate_flag"].values == 1
    unavailable = standardized["unavailable_flag"].values == 1
    land = standardized["land_flag"].values == 1
    observed = standardized["observed_flag"].values == 1

    assert estimate.any()
    assert not (estimate & unavailable).any()
    assert not (estimate & land).any()
    assert not (estimate & observed).any()


def test_prepare_model_data_rejects_bad_cloud_mode():
    ds, _ = demo_data(days=12, lat_size=8, lon_size=8)
    with pytest.raises(ValueError, match="cloud_mode"):
        prepare_model_data(
            ds,
            target_variable="chlor_a",
            missing_flag="cloud_flag",
            land_flag="land_flag",
            std_vars=[],
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
def test_demo_data_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        demo_data(**kwargs)


def test_demo_data_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        demo_data(dataset="unknown")
