from typing import Union

import numpy as np
import pandas as pd
import xarray as xr


def demo_data(
    days=120,
    lat_size=16,
    lon_size=16,
    start="2020-01-01",
    seed=42,
    cloud_fraction=0.12,
):
    """Create a deterministic chlorophyll dataset for examples and tests."""
    if days <= 0 or lat_size <= 0 or lon_size <= 0:
        raise ValueError("days, lat_size, and lon_size must be positive")
    if not 0 <= cloud_fraction <= 1:
        raise ValueError("cloud_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=days, freq="D")
    lat = np.linspace(31, 5, lat_size)
    lon = np.linspace(42, 80, lon_size)
    day, latitude, longitude = np.meshgrid(
        np.arange(days),
        lat,
        lon,
        indexing="ij",
    )
    chlorophyll = np.exp(
        0.5
        + 0.25 * np.sin(2 * np.pi * day / 30)
        + 0.15 * np.cos(np.deg2rad(latitude * 3))
        + 0.1 * np.sin(np.deg2rad(longitude * 4))
        + rng.normal(0, 0.05, day.shape)
    ).astype("float32")
    land = np.broadcast_to(lat[:, None] > 28, chlorophyll.shape)
    cloud = rng.random(chlorophyll.shape) < cloud_fraction
    chlorophyll[land | cloud] = np.nan

    return xr.Dataset(
        data_vars={
            "chlor_a": (("time", "lat", "lon"), chlorophyll),
            "cloud_flag": (
                ("time", "lat", "lon"),
                cloud.astype("int8"),
            ),
            "land_flag": (
                ("time", "lat", "lon"),
                land.astype("int8"),
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def crop_to_multiple(
    ds: Union[xr.Dataset, xr.DataArray],
    lat: str = "lat",
    lon: str = "lon",
    multiple: int = 8,
    center: bool = False,
) -> Union[xr.Dataset, xr.DataArray]:
    """Crop spatial dimensions so their lengths are divisible by ``multiple``."""
    if lat not in ds.dims or lon not in ds.dims:
        missing = [dim for dim in (lat, lon) if dim not in ds.dims]
        raise KeyError(
            f"Missing required dimension(s): {missing}. Present: {list(ds.dims)}"
        )
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer")

    slices = {}
    for dim in (lat, lon):
        size = ds.sizes[dim]
        remainder = size % multiple
        if not remainder:
            slices[dim] = slice(None)
        elif center:
            start = remainder // 2
            slices[dim] = slice(start, size - (remainder - start))
        else:
            slices[dim] = slice(0, size - remainder)
    return ds.isel(slices)


def unstdize(stdized_image, mean, stdev):
    """Convert standardized values back to their original scale."""
    return stdized_image * stdev + mean


def compute_mae(y_true, y_pred):
    """Compute mean absolute error over pairs where neither value is NaN."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def compute_mse(y_true, y_pred):
    """Compute mean squared error over pairs where neither value is NaN."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean((y_true[mask] - y_pred[mask]) ** 2)


def make_tf_gen(batcher, x_vars, label="full_target"):
    """Create a TensorFlow generator from standardized xbatcher blocks."""

    def gen():
        for batch in batcher:
            batch = batch.load()
            for time_index in range(batch.sizes["time"]):
                x = np.stack(
                    [
                        np.nan_to_num(
                            batch[var].isel(time=time_index).values,
                            nan=0.0,
                        )
                        for var in x_vars
                    ],
                    axis=-1,
                ).astype(np.float32)
                y = np.nan_to_num(
                    batch[label].isel(time=time_index).values,
                    nan=0.0,
                ).astype(np.float32)[..., np.newaxis]
                yield x, y

    return gen


def _add_spherical_coords(ds, lat="lat", lon="lon"):
    """Add unit-sphere coordinates to an xarray dataset."""
    lat_2d, lon_2d = xr.broadcast(ds[lat], ds[lon])
    latitude = xr.apply_ufunc(np.deg2rad, lat_2d, dask="parallelized")
    longitude = xr.apply_ufunc(np.deg2rad, lon_2d, dask="parallelized")
    cos_latitude = xr.apply_ufunc(np.cos, latitude, dask="parallelized")

    return ds.assign(
        x_geo=(
            cos_latitude
            * xr.apply_ufunc(np.cos, longitude, dask="parallelized")
        ).astype("float32"),
        y_geo=(
            cos_latitude
            * xr.apply_ufunc(np.sin, longitude, dask="parallelized")
        ).astype("float32"),
        z_geo=xr.apply_ufunc(
            np.sin,
            latitude,
            dask="parallelized",
        ).astype("float32"),
    )


def build_standardized_lazy(
    ds,
    target_variable="CHL_cmes-level3",
    missing_flag="CHL_cmes-cloud",
    land_flag="CHL_cmes-land",
    features=None,
    train_dates=None,
    std_vars=None,
    log_target=True,
    missing_flag_shift=10,
    n_temporal_lags=1,
    output_chunks=None,
    add_geo=False,
):
    """Build lazy model inputs, targets, and standardization statistics.

    The returned dataset contains the transformed target, a synthetically masked
    target, temporal target lags, seasonal channels, missingness flags, optional
    feature variables, and optional spherical coordinates.
    """
    features = list(features or [])
    required = [target_variable, *features, missing_flag, land_flag]
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"Dataset is missing required variables: {missing}")
    for coord in ("time", "lat", "lon"):
        if coord not in ds.coords:
            raise ValueError(f"Required coordinate '{coord}' not found in ds")
    if n_temporal_lags < 0:
        raise ValueError("n_temporal_lags must be non-negative")

    processed = ds[required].rename({target_variable: "full_target"})
    for flag in (missing_flag, land_flag):
        processed[flag] = processed[flag].broadcast_like(
            processed["full_target"]
        )
    if log_target:
        processed["full_target"] = np.log(
            processed["full_target"].where(processed["full_target"] > 0)
        )

    shifted_missing = processed[missing_flag].roll(
        time=-missing_flag_shift,
        roll_coords=False,
    )
    processed["masked_target"] = processed["full_target"].where(
        shifted_missing != 0
    )
    processed["true_missing_flag"] = (processed[missing_flag] == 1).astype(
        "int8"
    )
    processed["land_flag"] = (processed[land_flag] == 1).astype("int8")
    processed["valid_masked_target_flag"] = processed[
        "masked_target"
    ].notnull().astype("int8")
    processed["synthetic_missing_flag"] = (
        (shifted_missing == 0)
        & (processed["land_flag"] == 0)
        & (processed["true_missing_flag"] == 0)
        & processed["full_target"].notnull()
    ).astype("int8")

    day_of_year = processed.time.dt.dayofyear
    spatial_template = xr.ones_like(
        processed["full_target"].isel(time=0, drop=True),
        dtype=np.float32,
    )
    processed["day_sin"] = (
        np.sin(2 * np.pi * day_of_year / 365.25).astype("float32")
        * spatial_template
    ).transpose("time", "lat", "lon")
    processed["day_cos"] = (
        np.cos(2 * np.pi * day_of_year / 365.25).astype("float32")
        * spatial_template
    ).transpose("time", "lat", "lon")

    if add_geo:
        with_geo = _add_spherical_coords(processed)
        for name in ("x_geo", "y_geo", "z_geo"):
            processed[name] = (
                xr.ones_like(day_of_year).astype("float32") * with_geo[name]
            ).transpose("time", "lat", "lon")

    standardizable_vars = features + ["full_target", "masked_target"]
    for lag in range(1, n_temporal_lags + 1):
        previous = f"masked_target_m{lag}"
        following = f"masked_target_p{lag}"
        processed[previous] = processed["masked_target"].shift(time=lag)
        processed[following] = processed["masked_target"].shift(time=-lag)
        standardizable_vars.extend([previous, following])

    std_vars = list(features if std_vars is None else std_vars)
    unknown = sorted(set(std_vars) - set(standardizable_vars))
    if unknown:
        raise ValueError(
            "std_vars contains unknown variables: " + ", ".join(unknown)
        )

    means = {name: 0.0 for name in standardizable_vars}
    standard_deviations = {name: 1.0 for name in standardizable_vars}
    if std_vars:
        stats_source = processed[std_vars]
        if train_dates is not None:
            stats_source = stats_source.sel(time=train_dates)
        computed_means = stats_source.mean(
            dim=("time", "lat", "lon"),
            skipna=True,
        ).compute()
        computed_stds = stats_source.std(
            dim=("time", "lat", "lon"),
            skipna=True,
        ).compute()
        for name in std_vars:
            means[name] = float(computed_means[name].item())
            standard_deviations[name] = float(computed_stds[name].item())

    standardized = processed.copy()
    for name in std_vars:
        standardized[name] = (
            processed[name] - means[name]
        ) / standard_deviations[name]

    output_vars = standardizable_vars + ["day_sin", "day_cos"]
    if add_geo:
        output_vars.extend(["x_geo", "y_geo", "z_geo"])
    output_vars.extend(
        [
            "synthetic_missing_flag",
            "true_missing_flag",
            "valid_masked_target_flag",
            "land_flag",
        ]
    )
    chunks = output_chunks or {"time": 100, "lat": -1, "lon": -1}
    output = standardized[output_vars].chunk(chunks)
    stats = {
        name: np.array(
            [means[name], standard_deviations[name]],
            dtype=np.float32,
        )
        for name in standardizable_vars
    }
    return output, stats


def make_xbatcher(ds, patch_dims, overlap=None, preload_batch=False):
    """Create an xbatcher generator over time and spatial windows."""
    import xbatcher as xb

    kwargs = {
        "ds": ds,
        "input_dims": dict(patch_dims),
        "preload_batch": preload_batch,
    }
    if overlap is not None:
        kwargs["input_overlap"] = dict(overlap)
    return xb.BatchGenerator(**kwargs)


def UNet(input_shape):
    """Build the fully convolutional U-Net used by the fitting notebook."""
    import tensorflow as tf
    from tensorflow.keras import Input, layers

    inputs = Input(shape=input_shape)
    x = inputs
    filters = [64, 128, 256]
    encoder_images = []

    for number_filters in filters:
        encoder_images.append(x)
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNormalization()(x)

    for number_filters, encoder_image in zip(
        filters[:-1][::-1],
        encoder_images[::-1][:-1],
    ):
        x = layers.Conv2DTranspose(
            number_filters,
            3,
            2,
            padding="same",
        )(x)
        x = layers.concatenate([x, encoder_image])
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(
        number_filters,
        3,
        2,
        padding="same",
    )(x)
    x = layers.concatenate([x, encoder_images[0]])
    x = layers.Conv2D(
        number_filters,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)
    outputs = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="linear",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="U-net")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
