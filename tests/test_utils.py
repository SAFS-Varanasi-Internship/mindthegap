import numpy as np
import pytest
import xarray as xr

from mindthegap import build_standardized_lazy, demo_data


def test_demo_data_has_requested_shape_and_flags():
    ds = demo_data(days=10, lat_size=8, lon_size=12, seed=7)

    assert ds.sizes == {"time": 10, "lat": 8, "lon": 12}
    assert {"chlor_a", "cloud_flag", "land_flag"} == set(ds.data_vars)
    assert ds.cloud_flag.dtype == np.dtype("int8")
    assert ds.land_flag.dtype == np.dtype("int8")
    assert np.isnan(ds.chlor_a.where(ds.land_flag == 1)).all()


def test_demo_data_is_deterministic():
    first = demo_data(days=5, seed=3)
    second = demo_data(days=5, seed=3)

    assert first.identical(second)


def test_build_standardized_lazy_broadcasts_static_land_mask():
    ds = demo_data(days=12, lat_size=8, lon_size=8)
    ds["land_flag"] = ds["land_flag"].isel(time=0, drop=True)

    standardized, _ = build_standardized_lazy(
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
