import keras
import numpy as np
import xarray as xr

from mindthegap import gapfill_std


CHANNELS = ["observed_target", "estimate_flag", "land_flag"]


def _passthrough_model():
    """A model that returns the first input channel unchanged as the target."""
    inputs = keras.Input(shape=(None, None, len(CHANNELS)))
    outputs = keras.layers.Lambda(lambda x: x[..., :1])(inputs)
    model = keras.Model(inputs, outputs, name="passthrough")
    return model


def _ds_std(times=3, lat=8, lon=8, seed=0):
    rng = np.random.default_rng(seed)
    coords = {
        "time": np.arange(
            np.datetime64("2001-01-01"),
            np.datetime64("2001-01-01") + np.timedelta64(times, "D"),
            dtype="datetime64[D]",
        ),
        "lat": np.linspace(-2, 2, lat),
        "lon": np.linspace(10, 14, lon),
    }
    data = {
        name: (("time", "lat", "lon"),
               rng.standard_normal((times, lat, lon)).astype("float32"))
        for name in CHANNELS
    }
    return xr.Dataset(data, coords=coords)


def _metadata():
    return {
        "inputs": [{"name": name, "channel": i}
                   for i, name in enumerate(CHANNELS)],
        "target": {"name": "full_target", "units": "mg m-3"},
        "preprocessing": {
            "standardization": {
                "full_target": {"mean": 0.0, "std": 1.0, "applied": False}
            },
        },
    }


def test_gapfill_std_returns_time_lat_lon_field():
    ds = _ds_std()
    model = _passthrough_model()
    out = gapfill_std(ds, model, _metadata())

    field = out["gapfilled_target"]
    assert field.dims == ("time", "lat", "lon")
    assert field.shape == (3, 8, 8)
    assert "gapfilled_target" in out.data_vars
    xr.testing.assert_equal(out["time"], ds["time"])


def test_gapfill_std_returns_raw_model_output():
    ds = _ds_std()
    model = _passthrough_model()
    # A passthrough model returns the first input channel unchanged. gapfill_std
    # must return exactly that, with no unstandardizing and no de-logging.
    out = gapfill_std(ds, model, _metadata())

    np.testing.assert_allclose(
        out["gapfilled_target"].values,
        ds["observed_target"].values,
        rtol=1e-5,
    )
    assert out["gapfilled_target"].attrs["standardized"] is True


def test_gapfill_std_single_time_keeps_time_dim():
    ds = _ds_std()
    model = _passthrough_model()
    stamp = ds["time"].values[1]
    out = gapfill_std(ds, model, _metadata(), time=stamp)

    field = out["gapfilled_target"]
    assert field.dims == ("time", "lat", "lon")
    assert field.sizes["time"] == 1
    expected = ds["observed_target"].sel(time=stamp).values
    np.testing.assert_allclose(
        field.isel(time=0).values, expected, rtol=1e-5
    )


def test_gapfill_std_time_subset():
    ds = _ds_std()
    model = _passthrough_model()
    subset = ds["time"].values[:2]
    out = gapfill_std(ds, model, _metadata(), time=list(subset))

    assert out.sizes["time"] == 2
    np.testing.assert_array_equal(out["time"].values, subset)


def test_gapfill_std_fills_nan_inputs_with_zero():
    ds = _ds_std()
    ds["observed_target"].values[0, 0, 0] = np.nan
    model = _passthrough_model()
    out = gapfill_std(ds, model, _metadata())

    field = out["gapfilled_target"]
    assert np.isfinite(field.values).all()
    assert field.isel(time=0, lat=0, lon=0).item() == 0.0


def test_gapfill_std_requires_all_input_channels():
    ds = _ds_std().drop_vars("land_flag")
    model = _passthrough_model()
    try:
        gapfill_std(ds, model, _metadata())
    except KeyError as error:
        assert "land_flag" in str(error)
    else:
        raise AssertionError("expected KeyError for missing input channel")
