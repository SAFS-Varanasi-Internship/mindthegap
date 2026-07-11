import numpy as np
import xarray as xr

from mindthegap.create_zarr import data_preprocessing_new
from mindthegap.utils import build_standardized_lazy_new


def make_test_dataset():
    time = np.array(
        ["2001-01-01", "2001-01-02", "2001-01-03", "2001-01-04"],
        dtype="datetime64[D]",
    )
    lat = np.array([10.0])
    lon = np.array([20.0])

    target = np.exp(np.arange(4, dtype=np.float32)).reshape(4, 1, 1)
    sst = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32).reshape(4, 1, 1)
    u_wind = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(4, 1, 1)
    cloud = np.full((4, 1, 1), 2, dtype=np.int8)
    land = np.zeros((4, 1, 1), dtype=np.int8)

    return xr.Dataset(
        data_vars={
            "CHL_cmes-level3": (("time", "lat", "lon"), target),
            "sst": (("time", "lat", "lon"), sst),
            "u_wind": (("time", "lat", "lon"), u_wind),
            "CHL_cmes-cloud": (("time", "lat", "lon"), cloud),
            "CHL_cmes-land": (("time", "lat", "lon"), land),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_data_preprocessing_new_treats_none_features_as_empty():
    ds = make_test_dataset()

    result_none = data_preprocessing_new(
        ds,
        features=None,
        train_dates=None,
        n_temporal_lags=0,
    )
    result_empty = data_preprocessing_new(
        ds,
        features=[],
        train_dates=None,
        n_temporal_lags=0,
    )

    xr.testing.assert_identical(result_none, result_empty)


def test_data_preprocessing_new_uses_all_dates_when_train_dates_is_none():
    ds = make_test_dataset()

    result = data_preprocessing_new(
        ds,
        features=["sst"],
        train_dates=None,
        std_vars=["sst", "full_target", "masked_target"],
        n_temporal_lags=0,
    )

    expected_sst = np.array([-1.3416408, -0.4472136, 0.4472136, 1.3416408], dtype=np.float32)
    expected_target = np.array([-1.3416407, -0.4472136, 0.4472136, 1.3416407], dtype=np.float32)

    np.testing.assert_allclose(result["sst"].values[:, 0, 0], expected_sst, rtol=1e-6)
    np.testing.assert_allclose(result["full_target"].values[:, 0, 0], expected_target, rtol=1e-6)
    np.testing.assert_allclose(result["masked_target"].values[:, 0, 0], expected_target, rtol=1e-6)
    np.testing.assert_allclose(result["sst_train_mean"].item(), 25.0)
    np.testing.assert_allclose(result["sst_train_std"].item(), np.std([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_allclose(result["full_target_train_mean"].item(), 1.5)
    np.testing.assert_allclose(result["full_target_train_std"].item(), np.std([0.0, 1.0, 2.0, 3.0]))


def test_data_preprocessing_new_sets_identity_stats_for_unstandardized_vars():
    ds = make_test_dataset()

    result = data_preprocessing_new(
        ds,
        features=["sst", "u_wind"],
        std_vars=["sst"],
        train_dates=None,
        n_temporal_lags=0,
    )

    expected_sst = np.array([-1.3416408, -0.4472136, 0.4472136, 1.3416408], dtype=np.float32)

    np.testing.assert_allclose(result["sst"].values[:, 0, 0], expected_sst, rtol=1e-6)
    np.testing.assert_allclose(result["u_wind"].values[:, 0, 0], [1.0, 2.0, 3.0, 4.0], rtol=1e-6)
    np.testing.assert_allclose(result["full_target"].values[:, 0, 0], [0.0, 1.0, 2.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(result["u_wind_train_mean"].item(), 0.0)
    np.testing.assert_allclose(result["u_wind_train_std"].item(), 1.0)
    np.testing.assert_allclose(result["full_target_train_mean"].item(), 0.0)
    np.testing.assert_allclose(result["full_target_train_std"].item(), 1.0)


def test_build_standardized_lazy_new_matches_eager_preprocessing():
    ds = make_test_dataset().chunk({"time": 2, "lat": 1, "lon": 1})

    ds_lazy, stats = build_standardized_lazy_new(
        ds,
        features=["sst", "u_wind"],
        train_dates=slice("2001-01-01", "2001-01-03"),
        std_vars=["sst", "masked_target", "full_target", "masked_target_m1"],
        n_temporal_lags=1,
        output_chunks={"time": 2, "lat": 1, "lon": 1},
    )
    ds_eager = data_preprocessing_new(
        make_test_dataset(),
        features=["sst", "u_wind"],
        train_dates=slice("2001-01-01", "2001-01-03"),
        std_vars=["sst", "masked_target", "full_target", "masked_target_m1"],
        n_temporal_lags=1,
    )

    assert ds_lazy["sst"].chunks == ((2, 2), (1,), (1,))

    compare_vars = [
        "sst",
        "u_wind",
        "full_target",
        "masked_target",
        "masked_target_m1",
        "masked_target_p1",
        "synthetic_missing_flag",
        "true_missing_flag",
        "land_flag",
    ]
    for var in compare_vars:
        xr.testing.assert_allclose(ds_lazy[var].compute(), ds_eager[var])

    np.testing.assert_allclose(
        ds_lazy["day_sin"].isel(lat=0, lon=0).compute().values,
        ds_eager["day_sin"].values,
    )
    np.testing.assert_allclose(
        ds_lazy["day_cos"].isel(lat=0, lon=0).compute().values,
        ds_eager["day_cos"].values,
    )

    x_vars = [v for v in ds_lazy.data_vars if v != "full_target"]
    sample = np.stack(
        [np.nan_to_num(ds_lazy[v].isel(time=0).compute().values, nan=0.0) for v in x_vars],
        axis=-1,
    )
    assert sample.shape == (1, 1, len(x_vars))

    np.testing.assert_allclose(stats["full_target"], [1.0, np.std([0.0, 1.0, 2.0])], rtol=1e-6)
    np.testing.assert_allclose(stats["masked_target"], [1.0, np.std([0.0, 1.0, 2.0])], rtol=1e-6)
    np.testing.assert_allclose(stats["CHL"], stats["full_target"], rtol=1e-6)
    np.testing.assert_allclose(stats["masked_CHL"], stats["masked_target"], rtol=1e-6)
    np.testing.assert_allclose(stats["feat_stats"]["u_wind"], [0.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(stats["feat_stats"]["masked_target_m1"], [0.5, 0.5], rtol=1e-6)
