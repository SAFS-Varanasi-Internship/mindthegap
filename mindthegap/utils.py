from typing import Union
from numbers import Real

import numpy as np
import pandas as pd
import xarray as xr


DEMO_REGIONS = {
    "indian ocean": (-40.0, 30.0, 20.0, 120.0),
    "arabian sea": (3.0, 33.0, 40.0, 82.0),
    "ne atlantic": (0.0, 70.0, -45.0, 20.0),
    "nw pacific": (0.0, 65.0, 100.0, 180.0),
}


def demo_data(
    dataset="synthetic",
    region=None,
    time_slice=None,
    *,
    smoke_test=False,
    smoke_days=120,
    smoke_size=128,
    days=120,
    lat_size=16,
    lon_size=16,
    start="2020-01-01",
    seed=42,
    cloud_fraction=0.12,
):
    """Load a chlorophyll dataset and return ``(dataset, metadata)``.

    ``dataset`` may be ``"pace"``, ``"globcolour"``,
    ``"indian-ocean"``, or ``"synthetic"``. ``region`` may be a supported
    name or ``[lat_min, lat_max, lon_min, lon_max]``. ``None`` selects the
    full spatial extent.

    When ``smoke_test`` is true, the loaded dataset is subset to at most
    ``smoke_days`` time steps and ``smoke_size`` cells in each spatial
    dimension so remote validation runs stay small. The synthetic loader is
    generated at that reduced size directly. Sizing of the synthetic dataset is
    controlled by ``smoke_days``/``smoke_size`` (or the defaults) rather than by
    callers threading ``days``/``lat_size``/``lon_size`` from a notebook.
    """
    loaders = {
        "pace": _load_pace,
        "globcolour": _load_globcolour,
        "indian-ocean": _load_indian_ocean,
        "synthetic": _load_synthetic,
    }
    if dataset not in loaders:
        choices = ", ".join(loaders)
        raise ValueError(f"Unknown dataset {dataset!r}; choose from: {choices}")

    if dataset == "synthetic":
        if smoke_test:
            days = min(days, smoke_days)
            lat_size = min(lat_size, smoke_size)
            lon_size = min(lon_size, smoke_size)
        ds = loaders[dataset](
            days=days,
            lat_size=lat_size,
            lon_size=lon_size,
            start=start,
            seed=seed,
            cloud_fraction=cloud_fraction,
        )
    else:
        ds = loaders[dataset]()

    region_bounds, region_name = _resolve_region(region)
    ds = _select_demo_subset(
        ds,
        region_bounds=region_bounds,
        time_slice=time_slice,
    )
    if smoke_test:
        ds = ds.isel(
            time=slice(0, min(smoke_days, ds.sizes["time"])),
            lat=slice(0, min(smoke_size, ds.sizes["lat"])),
            lon=slice(0, min(smoke_size, ds.sizes["lon"])),
        )
    ds = _prepare_demo_dataset(dataset, ds)
    config = _dataset_config(dataset)
    metadata = {
        "dataset": {
            "name": config["name"],
            "product_id": config["product_id"],
            "loader": dataset,
            "region": {
                "lat": [
                    float(ds["lat"].min()),
                    float(ds["lat"].max()),
                ],
                "lon": [
                    float(ds["lon"].min()),
                    float(ds["lon"].max()),
                ],
            },
            "available_period": (
                f"{pd.to_datetime(ds.time.values[0]).date()} to "
                f"{pd.to_datetime(ds.time.values[-1]).date()}"
            ),
        },
        "target": {
            "name": config["target"],
            "units": ds[config["target"]].attrs.get("units", "unknown"),
        },
        "variables": {
            "target": config["target"],
            "features": [],
            "missing_flag": config["missing_flag"],
            "land_flag": config["land_flag"],
        },
    }
    if region_name is not None:
        metadata["dataset"]["region_name"] = region_name
    return ds, metadata


def _dataset_config(dataset):
    return {
        "pace": {
            "name": "PACE",
            "product_id": "PACE_OCI_L3M_CHL",
            "target": "chlor_a",
            "missing_flag": "cloud_flag",
            "land_flag": "land_flag",
        },
        "globcolour": {
            "name": "GlobColour",
            "product_id": (
                "cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D"
            ),
            "target": "CHL",
            "missing_flag": "cloud_flag",
            "land_flag": "land_flag",
        },
        "indian-ocean": {
            "name": "Indian Ocean",
            "product_id": "mind_the_chl_gap/IO_rechunked.zarr",
            "target": "CHL_cmes-level3",
            "missing_flag": "CHL_cmes-cloud",
            "land_flag": "CHL_cmes-land",
        },
        "synthetic": {
            "name": "Synthetic",
            "product_id": "mindthegap-synthetic",
            "target": "chlor_a",
            "missing_flag": "cloud_flag",
            "land_flag": "land_flag",
        },
    }[dataset]


def _validate_region_bounds(bounds):
    if len(bounds) != 4:
        raise ValueError(
            "region bounds must contain "
            "[lat_min, lat_max, lon_min, lon_max]"
        )
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in bounds):
        raise TypeError("region bounds must be four numeric values")

    lat_min, lat_max, lon_min, lon_max = map(float, bounds)
    if not all(np.isfinite(value) for value in bounds):
        raise ValueError("region bounds must be finite")
    if not -90 <= lat_min < lat_max <= 90:
        raise ValueError(
            "latitude bounds must satisfy -90 <= lat_min < lat_max <= 90"
        )
    if not -180 <= lon_min < lon_max <= 180:
        raise ValueError(
            "longitude bounds must satisfy -180 <= lon_min < lon_max <= 180"
        )
    return lat_min, lat_max, lon_min, lon_max


def _resolve_region(region):
    if region is None:
        return None, None
    if isinstance(region, str):
        region_name = " ".join(
            region.lower().replace("_", " ").replace("-", " ").split()
        )
        if region_name not in DEMO_REGIONS:
            choices = ", ".join(sorted(DEMO_REGIONS))
            raise ValueError(
                f"Unknown region {region!r}; choose from: {choices}"
            )
        return _validate_region_bounds(DEMO_REGIONS[region_name]), region_name
    if not isinstance(region, (list, tuple, np.ndarray)):
        raise TypeError(
            "region must be a name or "
            "[lat_min, lat_max, lon_min, lon_max]"
        )
    return _validate_region_bounds(region), None


def _coordinate_slice(ds, dim, lower, upper):
    if lower is None and upper is None:
        return None
    index = ds.indexes[dim]
    if lower is None:
        lower = index.min()
    if upper is None:
        upper = index.max()
    if lower >= upper:
        raise ValueError(f"{dim}_min must be less than {dim}_max")

    if index.is_monotonic_increasing:
        return slice(lower, upper)
    if index.is_monotonic_decreasing:
        return slice(upper, lower)
    raise ValueError(f"{dim} coordinates must be monotonic")


def _select_demo_subset(
    ds,
    *,
    region_bounds,
    time_slice,
):
    selectors = {}
    if region_bounds is not None:
        lat_min, lat_max, lon_min, lon_max = region_bounds
        selectors["lat"] = _coordinate_slice(
            ds,
            "lat",
            lat_min,
            lat_max,
        )
        selectors["lon"] = _coordinate_slice(
            ds,
            "lon",
            lon_min,
            lon_max,
        )
    if time_slice is not None:
        selectors["time"] = time_slice
    if selectors:
        ds = ds.sel(selectors)
    if any(ds.sizes[dim] == 0 for dim in ("time", "lat", "lon")):
        raise ValueError("The requested dataset slices contain no data")
    return ds


def _prepare_demo_dataset(dataset, ds):
    if dataset == "pace":
        gap = ds["chlor_a"].isnull()
        land = gap.all(dim="time")
        return ds.assign(
            land_flag=land.astype("int8"),
            cloud_flag=(gap & ~land).astype("int8"),
        )
    if dataset == "globcolour":
        land = ds["flags"] == 1
        return ds.assign(
            land_flag=land.astype("int8"),
            cloud_flag=(ds["CHL"].isnull() & ~land).astype("int8"),
        ).drop_vars("flags")
    if dataset == "indian-ocean":
        land = ds["CHL_cmes-cloud"].isel(time=0, drop=True) == 2
        return ds.assign(
            **{
                "CHL_cmes-land": land.astype("int8").broadcast_like(
                    ds["CHL_cmes-level3"]
                )
            }
        )
    return ds


def _load_synthetic(
    days,
    lat_size,
    lon_size,
    start,
    seed,
    cloud_fraction,
):
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

    ds = xr.Dataset(
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
    ds["chlor_a"].attrs["units"] = "mg m-3"
    return ds


def _load_pace():
    try:
        import earthaccess
        import icechunk as ic
    except ImportError as error:
        raise ImportError(
            "PACE requires earthaccess>=0.15 and icechunk>=2"
        ) from error

    auth = earthaccess.login()
    creds = auth.get_s3_credentials(daac="OBDAAC")
    storage = ic.http_storage(
        "https://data.source.coop/fish-pace/pace-oci/inregion/"
        "PACE_OCI_L3M_CHL"
    )
    credentials = ic.credentials.containers_credentials(
        {
            "s3://ob-cumulus-prod-public/": ic.credentials.s3_credentials(
                access_key_id=creds["accessKeyId"],
                secret_access_key=creds["secretAccessKey"],
                session_token=creds["sessionToken"],
            )
        }
    )
    store = ic.Repository.open(
        storage,
        authorize_virtual_chunk_access=credentials,
    ).readonly_session("main").store
    ds = xr.open_zarr(
        store,
        consolidated=False,
        group="daily/0p1deg/chunks_512",
        chunks={},
    )
    return ds


def _load_globcolour():
    try:
        import icechunk as ic
    except ImportError as error:
        raise ImportError("GlobColour requires icechunk>=2") from error

    storage = ic.http_storage(
        "https://data.source.coop/fish-pace/globcolour/"
        "cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D"
    )
    repository = ic.Repository.open(storage)
    credentials = {
        prefix: ic.credentials.HttpAccess
        for prefix in repository.config.virtual_chunk_containers or []
    }
    store = repository.reopen(
        authorize_virtual_chunk_access=credentials
    ).readonly_session("main").store
    return xr.open_zarr(
        store,
        consolidated=False,
        chunks={},
    )[["CHL", "flags"]]


def _load_indian_ocean():
    ds = xr.open_dataset(
        "gcs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/"
        "IO_rechunked.zarr",
        engine="zarr",
        backend_kwargs={"storage_options": {"token": "anon"}},
        consolidated=True,
    )
    return ds


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
    target_variable=None,
    missing_flag=None,
    land_flag=None,
    features=None,
    train_dates=None,
    std_vars=None,
    log_target=None,
    missing_flag_shift=10,
    n_temporal_lags=None,
    output_chunks=None,
    add_geo=False,
    options=None,
):
    """Build lazy model inputs, targets, and standardization statistics.

    The returned dataset contains the transformed target, a synthetically masked
    target, temporal target lags, seasonal channels, missingness flags, optional
    feature variables, and optional spherical coordinates.

    Returns ``(output, stats)``. When ``options`` (a :class:`DataOptions`) is
    provided, variable names, ``features``, ``log_target``, and
    ``n_temporal_lags`` default to the values already resolved on it, so callers
    do not repeat them. ``options`` is then populated in place with the
    canonical resolved data configuration (bounds, input channel names/order,
    transforms, standardization statistics, and target mean/std) so downstream
    functions and the saved model bundle can reproduce these inputs.
    """
    if options is not None:
        target_variable = (
            target_variable
            if target_variable is not None
            else options.target_variable
        )
        missing_flag = (
            missing_flag if missing_flag is not None else options.missing_flag
        )
        land_flag = land_flag if land_flag is not None else options.land_flag
        if features is None:
            features = options.features
        if log_target is None:
            log_target = options.log_target
        if n_temporal_lags is None:
            n_temporal_lags = options.n_temporal_lags

    if target_variable is None or missing_flag is None or land_flag is None:
        raise ValueError(
            "target_variable, missing_flag, and land_flag must be provided "
            "either directly or via options"
        )
    log_target = True if log_target is None else log_target
    n_temporal_lags = 1 if n_temporal_lags is None else n_temporal_lags

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

    if options is not None:
        input_names = [name for name in output_vars if name != "full_target"]
        options.target = "full_target"
        options.lat_bounds = (
            float(output["lat"].min()),
            float(output["lat"].max()),
        )
        options.lon_bounds = (
            float(output["lon"].min()),
            float(output["lon"].max()),
        )
        options.input_names = input_names
        options.log_target = bool(log_target)
        options.n_temporal_lags = n_temporal_lags
        options.transforms = {
            "target": "natural logarithm" if log_target else "none",
            "temporal_lags": n_temporal_lags,
            "add_geo": bool(add_geo),
        }
        options.standardization = {
            name: {
                "mean": float(means[name]),
                "std": float(standard_deviations[name]),
                "applied": name in std_vars,
            }
            for name in standardizable_vars
        }
        options.target_mean = float(means["full_target"])
        options.target_std = float(standard_deviations["full_target"])
        options.missing_value_handling = (
            "Missing predictor values are replaced with zero after "
            "standardization; mask channels identify land and missing data."
        )

    return output, stats


def make_xbatcher(
    ds,
    patch_dims=None,
    overlap=None,
    preload_batch=False,
    options=None,
):
    """Create an xbatcher generator over time and spatial windows.

    Either pass ``patch_dims``/``overlap`` explicitly or pass a
    :class:`GridderOptions` as ``options`` to derive them.
    """
    import xbatcher as xb

    if options is not None:
        patch_dims = options.patch_dims()
        overlap = options.overlap_dims()
        preload_batch = options.preload_batch
    if patch_dims is None:
        raise ValueError("provide either patch_dims or options")

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


def fit_model(
    model,
    train_data,
    options,
    *,
    validation_data=None,
    steps_per_epoch=None,
    validation_steps=None,
    callbacks=None,
    verbose=1,
):
    """Fit a model using a :class:`FitOptions` configuration section.

    ``options`` supplies epochs, batch size, learning rate, patience, loss, and
    optimizer so these choices are not threaded individually through the
    pipeline. Returns the Keras ``History`` object.
    """
    import tensorflow as tf

    optimizers = {
        "adam": tf.keras.optimizers.Adam,
    }
    if options.optimizer not in optimizers:
        raise ValueError(
            f"Unsupported optimizer {options.optimizer!r}; "
            f"choose from: {', '.join(optimizers)}"
        )
    model.compile(
        optimizer=optimizers[options.optimizer](
            learning_rate=options.learning_rate
        ),
        loss=options.loss,
        jit_compile=False,
    )

    if callbacks is None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=options.patience,
                restore_best_weights=True,
                verbose=verbose,
            )
        ]

    return model.fit(
        train_data,
        epochs=options.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )
