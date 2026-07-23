from typing import Union
import xarray as xr
import numpy as np
import dask.array as da

def crop_to_multiple(
    ds: Union[xr.Dataset, xr.DataArray],
    lat: str = "lat",
    lon: str = "lon",
    multiple: int = 8,
    center: bool = False,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Crop an xarray Dataset or DataArray along latitude/longitude so that the
    spatial shape is divisible by `multiple` (useful for U-Net down/upsampling).

    This is a *view*-like operation (no data copy) that trims rows/columns from
    the edges only. It does not pad. If `center=True`, the function crops
    symmetrically; otherwise it drops only from the end.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input object with spatial dims (`lat`, `lon` by default).
    lat : str, default "lat"
        Name of the latitude dimension to crop.
    lon : str, default "lon"
        Name of the longitude dimension to crop.
    multiple : int, default 8
        Target multiple for both spatial dimensions. For a U-Net with `D`
        pooling levels, use `multiple = 2**D` (e.g., D=3 → 8).
    center : bool, default False
        If True, crop symmetrically (half from the start, half from the end).
        If False, drop only from the end (keeps the origin intact).

    Returns
    -------
    xr.Dataset or xr.DataArray
        Cropped object (same type as input) whose spatial shape is divisible by `multiple`.

    Raises
    ------
    KeyError
        If `lat` or `lon` dims are not present in `ds`.

    Notes
    -----
    - If a dimension is already divisible by `multiple`, it is left unchanged.
    - If `ds.sizes[lat] < multiple` (or same for `lon`), this will crop to zero
      for that dimension; consider padding instead in that case (e.g., `xr.pad`).
    - Coordinates and attributes are preserved by `isel`.

    Examples
    --------
    Basic use with a Dataset:
    >>> ds_aligned = crop_to_multiple(zarr_ds, multiple=8)
    >>> ds_aligned.sizes["lat"], ds_aligned.sizes["lon"]
    (104, 152)  # for an original 105x153

    Symmetric crop (centered):
    >>> ds_centered = crop_to_multiple(zarr_ds, multiple=8, center=True)

    With a DataArray:
    >>> chl = zarr_ds["CHL_cmes-level3"]
    >>> chl_aligned = crop_to_multiple(chl, multiple=8)

    Using U-Net depth to choose the multiple:
    >>> depth = 3
    >>> m = 2 ** depth
    >>> ds_aligned = crop_to_multiple(zarr_ds, multiple=m)
    """
    # Validate required dims
    if lat not in ds.dims or lon not in ds.dims:
        missing = [d for d in (lat, lon) if d not in ds.dims]
        raise KeyError(f"Missing required dimension(s): {missing}. Present: {list(ds.dims)}")

    nlat = ds.sizes[lat]
    nlon = ds.sizes[lon]
    rlat = nlat % multiple
    rlon = nlon % multiple

    # Already aligned → return as-is
    if rlat == 0 and rlon == 0:
        return ds

    if not center:
        # Drop only from the end so indices/geo origin are preserved
        sl_lat = slice(0, nlat - rlat) if rlat else slice(0, nlat)
        sl_lon = slice(0, nlon - rlon) if rlon else slice(0, nlon)
    else:
        # Symmetric crop: split the remainder on both sides
        lat_left = rlat // 2
        lat_right = rlat - lat_left
        lon_left = rlon // 2
        lon_right = rlon - lon_left
        sl_lat = slice(lat_left, nlat - lat_right)
        sl_lon = slice(lon_left, nlon - lon_right)

    # isel preserves coords/attrs and is lazy for dask-backed arrays
    return ds.isel({lat: sl_lat, lon: sl_lon})

import numpy as np

# Helper functions in mindthegap
# - `unstdize`: unstandardize model outputs back to the original scale
# - `compute_mae`: mean absolute error, ignoring NaNs
# - `compute_mse`: mean squared error, ignoring NaNs


def unstdize(stdized_image, mean, stdev):
    """
    Unstandardize an array from standardized units back to the original scale.

    Given values standardized as (x - mean) / stdev, this function inverts the
    transform to recover x.

    Parameters
    ----------
    stdized_image : array-like
        Standardized values (e.g., model outputs). Can be a NumPy array or
        any array-like object broadcastable with `mean` and `stdev`.
    mean : float or array-like
        Mean used during standardization. May be a scalar or an array
        broadcastable to `stdized_image`.
    stdev : float or array-like
        Standard deviation used during standardization. May be a scalar or an array
        broadcastable to `stdized_image`.

    Returns
    -------
    array-like
        Unstandardized values on the original scale.

    Examples
    --------
    >>> y_std = np.array([0.0, 1.0, -1.0])
    >>> unstdize(y_std, mean=10.0, stdev=2.0)
    array([10., 12.,  8.])

    >>> y_std = np.array([[0., 1.], [np.nan, -0.5]])
    >>> unstdize(y_std, mean=5.0, stdev=2.0)
    array([[5. , 7. ],
           [nan, 4. ]])
    """
    return stdized_image * stdev + mean


def compute_mae(y_true, y_pred):
    """
    Compute mean absolute error (MAE) while ignoring NaN pairs.

    Elements where either `y_true` or `y_pred` is NaN are excluded from the average.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values. Must be the same shape as `y_true`.

    Returns
    -------
    float
        Mean absolute error over valid (non-NaN) pairs.

    Examples
    --------
    >>> yt = np.array([1.0, 2.0, np.nan, 4.0])
    >>> yp = np.array([0.5, 2.5, 3.0, np.nan])
    >>> compute_mae(yt, yp)
    0.5

    Notes
    -----
    - If all pairs are NaN, `np.mean([])` will return `nan`.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def compute_mse(y_true, y_pred):
    """
    Compute mean squared error (MSE) while ignoring NaN pairs.

    Elements where either `y_true` or `y_pred` is NaN are excluded from the average.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values. Must be the same shape as `y_true`.

    Returns
    -------
    float
        Mean squared error over valid (non-NaN) pairs.

    Examples
    --------
    >>> yt = np.array([1.0, 2.0, np.nan, 4.0])
    >>> yp = np.array([0.5, 2.5, 3.0, np.nan])
    >>> compute_mse(yt, yp)
    0.25

    See Also
    --------
    compute_mae : Mean absolute error with the same NaN handling.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean((y_true[mask] - y_pred[mask]) ** 2)

import numpy as np

def make_tf_gen(batcher, x_vars, label="CHL"):
    """
    Build a generator for ``tf.data.Dataset.from_generator`` that streams an xbatcher
    dataset one time step at a time.

    This is a *pass-through* generator: it assumes the batcher yields data whose channels
    are already engineered and standardized (e.g. the output of `build_standardized_lazy`).
    For each time step it stacks ``x_vars`` into an ``(lat, lon, len(x_vars))`` input tensor
    and returns ``label`` as the ``(lat, lon, 1)`` target. NaNs are replaced with 0.0.

    (The previous version of this function engineered the fake clouds / masked CHL / time
    features on the fly; that logic now lives in `build_standardized_lazy`, so the generator
    only has to stack and yield.)

    Parameters
    ----------
    batcher : xbatcher.BatchGenerator
        Iterable of chunked xarray blocks (e.g. 100-day blocks).
    x_vars : sequence of str
        Channel names to stack, in the desired channel order.
    label : str, default "CHL"
        Name of the target variable in each block.

    Returns
    -------
    callable
        A zero-arg generator ``gen()`` yielding ``(x, y)`` float32 tuples, where ``x`` has
        shape ``(lat, lon, len(x_vars))`` and ``y`` has shape ``(lat, lon, 1)``.
    """
    def gen():
        for batch in batcher:
            batch = batch.load()  # materialize one block once (bounded RAM, avoids per-step recompute)
            time_len = batch.sizes["time"]
            for t in range(time_len):
                x = np.stack(
                    [np.nan_to_num(batch[v].isel(time=t).values, nan=0.0) for v in x_vars],
                    axis=-1,
                ).astype(np.float32)
                y = np.nan_to_num(batch[label].isel(time=t).values, nan=0.0).astype(np.float32)[..., np.newaxis]
                yield x, y

    return gen


# Precomputed standardization stats for the IO.zarr Arabian Sea streaming setup
# (features=['u_wind', 'v_wind', 'sst', 'air_temp'], train_year=2015, train_range=3,
# region lat 5..31 / lon 42..80 cropped to a multiple of 8). Used for training *stability*,
# not statistical significance, so the exact values are not critical. Capture them once by
# calling build_standardized_lazy(..., use_hardcoded_stats=False) and reading
# stats['feat_stats'] and stats['CHL'], then paste them below. They are specific to the
# config above; a different region / feature list / train window needs its own values.
#

IO_ZARR_STATS = {
    'feat_stats': {'u_wind': [0.776536762714386, 3.6197967529296875], 'v_wind': [0.018182145431637764, 3.4297311305999756], 
                   'sst': [301.1195373535156, 2.034740924835205], 'air_temp': [299.70379638671875, 5.489468574523926], 
                   'sin_time': [0.00042300199856981635, 0.7069226503372192], 'cos_time': [0.0008084691362455487, 0.7072902321815491], 
                   'masked_CHL': [-1.266060709953308, 0.9658912420272827], 'prev_day_CHL': [-1.1477137759656528, 0.9668979047828038], 
                   'next_day-CHL': [-1.1472761139312693, 0.9667995380360648]} ,
    'CHL': [np.float32(-1.0840185), np.float32(0.9534597)] ,
}
#IO_ZARR_STATS = None


def build_standardized_lazy_new(
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
):
    """
    Lazy, on-the-fly equivalent of ``create_zarr.data_preprocessing_new``.

    This mirrors the newer preprocessing path: it builds ``full_target``,
    ``masked_target``, true/synthetic missing flags, day-of-year sine/cosine
    features, and optional masked-target lags directly from a raw dataset,
    while returning a dask-backed standardized dataset ready for xbatcher /
    TensorFlow use. Unlike the eager version, the seasonal channels are
    broadcast to ``(time, lat, lon)`` so they can be stacked as spatial
    model inputs without an intermediate Zarr write.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with ``time``, ``lat``, and ``lon`` coordinates.
    target_variable : str, default "CHL_cmes-level3"
        Name of the variable to use as the prediction target.
    missing_flag : str, default "CHL_cmes-cloud"
        Name of the variable whose value ``1`` marks true missing/cloud pixels.
        Its time-shifted pattern is also used to build the synthetic mask.
    land_flag : str, default "CHL_cmes-land"
        Name of the variable whose value ``1`` marks land pixels.
    features : sequence of str or None, default None
        Additional predictor variables to carry through preprocessing. ``None``
        is treated the same as an empty list.
    train_dates : indexer or None, default None
        Time selection used when computing standardization statistics. If
        ``None``, all dates in ``ds`` are used.
    std_vars : sequence of str or None, default None
        Variables to standardize. These must be drawn from ``features`` plus
        the derived target-like variables created by this function. If ``None``,
        defaults to ``features``. Standardizable variables not listed here are
        left unchanged and get identity stats (mean 0, std 1).
    log_target : bool, default True
        If True, apply ``log`` to positive target values before any masking or
        standardization.
    missing_flag_shift : int, default 10
        Number of time steps used to shift ``missing_flag`` when creating the
        synthetic mask pattern.
    n_temporal_lags : int, default 1
        Number of previous and next masked-target time steps to add as
        ``masked_target_m{i}`` and ``masked_target_p{i}``.
    output_chunks : dict or None, default None
        Dask chunking for the returned dataset. ``None`` keeps the default
        ``{"time": 100, "lat": -1, "lon": -1}``.

    Returns
    -------
    ds_out : xr.Dataset
        Lazy standardized dataset containing the engineered channels and
        ``full_target`` label, chunked per ``output_chunks``.
    stats : dict
        Standardization statistics for downstream unstandardization and bundle
        saving. Includes both the new ``full_target`` / ``masked_target`` keys
        and compatibility aliases ``CHL`` / ``masked_CHL``.
    """
    features = list(features or [])

    keep_vars = [target_variable] + features + [missing_flag, land_flag]
    ds = ds[keep_vars]

    for coord in ("time", "lat", "lon"):
        if coord not in ds.coords:
            raise ValueError(
                f"Required coordinate '{coord}' not found in ds. Rename coords if needed."
            )

    ds_processed = ds.rename({target_variable: "full_target"})

    if log_target:
        ds_processed["full_target"] = np.log(
            ds_processed["full_target"].where(ds_processed["full_target"] > 0)
        )

    shifted_missing_flag = ds_processed[missing_flag].roll(
        time=-missing_flag_shift,
        roll_coords=False,
    )

    ds_processed["masked_target"] = ds_processed["full_target"].where(
        shifted_missing_flag != 0
    )
    ds_processed["true_missing_flag"] = (
        ds_processed[missing_flag] == 1
    ).astype("int8")
    ds_processed["land_flag"] = (
        ds_processed[land_flag] == 1
    ).astype("int8")
    ds_processed["valid_masked_target_flag"] = (
        ds_processed["masked_target"].notnull()
    ).astype("int8")
    ds_processed["synthetic_missing_flag"] = (
        (shifted_missing_flag == 0)
        & (ds_processed["land_flag"] == 0)
        & (ds_processed["true_missing_flag"] == 0)
        & ds_processed["full_target"].notnull()
    ).astype("int8")

    day_of_year = ds_processed.time.dt.dayofyear
    spatial_template = xr.ones_like(
        ds_processed["full_target"].isel(time=0, drop=True),
        dtype=np.float32,
    )
    ds_processed["day_sin"] = (
        np.sin(2 * np.pi * day_of_year / 365.25).astype("float32")
        * spatial_template
    ).transpose("time", "lat", "lon")
    ds_processed["day_cos"] = (
        np.cos(2 * np.pi * day_of_year / 365.25).astype("float32")
        * spatial_template
    ).transpose("time", "lat", "lon")

    # Spherical coordinates (broadcast (lat, lon) to (time, lat, lon))
    ds_with_spherical = add_spherical_coords(ds_processed, lat="lat", lon="lon")
    for geo_var in ["x_geo", "y_geo", "z_geo"]:
        # Multiply (lat, lon) geo var by (time,) temporal multiplier to get (time, lat, lon)
        ds_processed[geo_var] = (
            xr.ones_like(day_of_year).astype("float32")
            * ds_with_spherical[geo_var]
        ).transpose("time", "lat", "lon")

    standardizable_vars = features + ["full_target", "masked_target"]

    for i in range(1, n_temporal_lags + 1):
        prev_name = f"masked_target_m{i}"
        next_name = f"masked_target_p{i}"
        ds_processed[prev_name] = ds_processed["masked_target"].shift(time=i)
        ds_processed[next_name] = ds_processed["masked_target"].shift(time=-i)
        standardizable_vars.extend([prev_name, next_name])

    if std_vars is None:
        std_vars = list(features)
    else:
        std_vars = list(std_vars)

    unknown_std_vars = sorted(set(std_vars) - set(standardizable_vars))
    if unknown_std_vars:
        raise ValueError(
            "std_vars contains unknown variables: "
            + ", ".join(unknown_std_vars)
        )

    mean_values = {var: 0.0 for var in standardizable_vars}
    std_values = {var: 1.0 for var in standardizable_vars}

    if std_vars:
        stats_source = ds_processed[std_vars]
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

        for var in std_vars:
            mean_values[var] = float(computed_means[var].item())
            std_values[var] = float(computed_stds[var].item())

    ds_standardized = ds_processed.copy()
    for var in std_vars:
        ds_standardized[var] = (
            ds_processed[var] - mean_values[var]
        ) / std_values[var]

    output_vars = standardizable_vars + [
        "day_sin",
        "day_cos",
        "x_geo",
        "y_geo",
        "z_geo",
        "synthetic_missing_flag",
        "true_missing_flag",
        "valid_masked_target_flag",
        "land_flag",
    ]
    if output_chunks is None:
        output_chunks = {"time": 100, "lat": -1, "lon": -1}
    ds_out = ds_standardized[output_vars].chunk(output_chunks)

    full_target_stats = np.array(
        [mean_values["full_target"], std_values["full_target"]],
        dtype=np.float32,
    )
    masked_target_stats = np.array(
        [mean_values["masked_target"], std_values["masked_target"]],
        dtype=np.float32,
    )
    feat_stats = {
        name: [float(mean_values[name]), float(std_values[name])]
        for name in standardizable_vars
        if name != "full_target"
    }
    stats = {
        "full_target": full_target_stats,
        "masked_target": masked_target_stats,
        "CHL": full_target_stats.copy(),
        "masked_CHL": masked_target_stats.copy(),
        "feat_stats": feat_stats,
    }
    return ds_out, stats


def build_standardized_lazy(zarr_ds, features, train_year, train_range, standardize_chl=False, use_hardcoded_stats=False, output_chunks=None, stats=None):
    """
    Lazy, on-the-fly equivalent of `create_zarr.data_preprocessing` that returns a
    dask-backed standardized ``xr.Dataset`` instead of writing a Zarr store.

    It builds exactly the same variables as `data_preprocessing` -- log CHL label,
    ``sin_time``/``cos_time``, the 10-day-shift ``masked_CHL``, ``prev_day_CHL``/
    ``next_day-CHL``, and the ``land``/``real_cloud``/``valid_CHL``/``fake_cloud`` flags --
    standardizing the numeric predictors with training-window statistics. Because the
    result stays lazy, it can be streamed block-by-block with xbatcher directly from raw
    ``IO.zarr`` (no intermediate Zarr on disk).

    NOTE: keep this in sync with `create_zarr.data_preprocessing` -- the two intentionally
    mirror each other. `data_preprocessing` is the write-to-Zarr path (used by the Fit
    notebook); this is the stream-from-raw path (used by the streaming notebook).

    Parameters
    ----------
    zarr_ds : xr.Dataset
        Raw (already region-sliced / cropped) dataset containing at least
        ``CHL_cmes-level3``, ``CHL_cmes-cloud`` and every name in ``features``.
    features : sequence of str
        Raw numeric predictor names to include (e.g. ``['u_wind', 'v_wind', 'sst', 'air_temp']``).
    train_year : int
        First calendar year of the training window (used for standardization stats).
    train_range : int
        Number of years in the training window.
    standardize_chl : bool, default False
        If True, standardize the CHL *label* to zero mean / unit std using its all-time
        statistics (matches `data_preprocessing`). If False (default), the label is left as
        log CHL: standardizing the label has no measurable effect on model quality, and
        leaving it unscaled means predictions come out directly in log space (no
        unstandardization needed). When False, ``stats['CHL']`` is ``[0.0, 1.0]`` so any
        downstream ``pred * std + mean`` is a no-op.
    use_hardcoded_stats : bool, default False
        If True, use the precomputed ``IO_ZARR_STATS`` constants instead of computing the
        train-window mean/std from the data (no eager read). Only valid for the IO.zarr
        Arabian Sea config the constants were captured for. Raises if ``IO_ZARR_STATS`` is
        not populated.
    output_chunks : dict or None, default None
        Dask chunking for the returned dataset. ``None`` keeps the default
        ``{"time": 100, "lat": -1, "lon": -1}``. Pass e.g. ``{"time": 100, "lat": 40, "lon": 56}``
        to align the output chunks with spatial patches so xbatcher reads line up on disk.
    stats : dict or None, default None
        If given, use these stats instead of computing or hardcoding them (no eager read).
        Must be shaped like the returned ``stats`` (``{'feat_stats': {name: [mean, std], ...},
        'CHL': [mean, std]}``). Lets several runs share one computed stats object so they
        standardize identically. Takes precedence over ``use_hardcoded_stats``.

    Returns
    -------
    ds_out : xr.Dataset
        Lazy standardized dataset: predictor channels + ``CHL`` label, chunked
        ``{"time": 100, "lat": -1, "lon": -1}``.
    stats : dict
        ``{'CHL': array([mean, std]), 'masked_CHL': array([mean, std]), 'feat_stats': {...}}``
        (train-window stats; ``CHL`` is ``[0.0, 1.0]`` when ``standardize_chl=False``).
        ``feat_stats`` maps each numeric channel name to ``[mean, std]`` and can be pasted
        into ``IO_ZARR_STATS`` to enable ``use_hardcoded_stats``.
    """
    numer_features = []
    cat_features = []

    # raw numerical predictors
    for feature in features:
        numer_features.append(zarr_ds[feature].data)

    # label: log(level3)  (NOTE: prep uses level3, not gapfree)
    CHL_data = np.log(zarr_ds['CHL_cmes-level3'].copy())

    # sin/cos seasonal encoding (days since 1900), standardized below
    time_data = da.array(zarr_ds.time)
    day_rad = (time_data - np.datetime64("1900-01-01")) / np.timedelta64(1, "D") / 365 * 2 * np.pi
    day_rad = day_rad.astype(np.float32)
    day_sin = np.sin(day_rad)
    day_cos = np.cos(day_rad)
    day_sin = np.tile(day_sin[:, np.newaxis, np.newaxis], (1,) + CHL_data[0].shape)
    day_sin = da.rechunk(day_sin, (100, *day_sin.shape[1:]))
    numer_features.append(day_sin)
    day_cos = np.tile(day_cos[:, np.newaxis, np.newaxis], (1,) + CHL_data[0].shape)
    day_cos = da.rechunk(day_cos, (100, *day_cos.shape[1:]))
    numer_features.append(day_cos)

    # artificially masked CHL (10-day shift)
    day_shift_flag = np.vstack((zarr_ds['CHL_cmes-cloud'].data[10:], zarr_ds['CHL_cmes-cloud'].data[:10]))
    assert CHL_data.shape == day_shift_flag.shape
    masked_CHL = da.where(day_shift_flag == 0, np.nan, CHL_data)
    numer_features.append(masked_CHL)

    #bad; need to use masked_CHL. Already dask array so no .data
    #prev_day = np.vstack((np.zeros((1,) + CHL_data[0].shape), CHL_data.data[:-1]))
    prev_day = np.vstack((np.zeros((1,) + masked_CHL[0].shape), masked_CHL[:-1]))
    numer_features.append(prev_day)
    #bad; need to use masked_CHL. Already dask array so no .data
    #next_day = np.vstack((CHL_data.data[1:], np.zeros((1,) + CHL_data[0].shape)))
    next_day = np.vstack((masked_CHL[1:], np.zeros((1,) + masked_CHL[0].shape)))
    numer_features.append(next_day)

    # categorical flags (NOT standardized)
    land_flag = da.zeros(CHL_data.shape)
    land_flag = da.where(zarr_ds['CHL_cmes-cloud'][0] == 2, 1, land_flag)
    cat_features.append(land_flag)

    real_cloud_flag = da.zeros(CHL_data.shape)
    real_cloud_flag = da.where(zarr_ds['CHL_cmes-cloud'] == 1, 1, real_cloud_flag)
    cat_features.append(real_cloud_flag)

    valid_CHL_flag = da.zeros(CHL_data.shape)
    valid_CHL_flag = da.where(~da.isnan(masked_CHL), 1, valid_CHL_flag)
    cat_features.append(valid_CHL_flag)

    fake_cloud_flag = da.zeros(CHL_data.shape)
    fake_cloud_flag = da.where((land_flag + real_cloud_flag + valid_CHL_flag) == 0, 1, fake_cloud_flag)
    cat_features.append(fake_cloud_flag)

    # Spherical coordinates (2D lat/lon grid broadcast to 3D time series)
    lat2d, lon2d = np.meshgrid(zarr_ds.lat.values, zarr_ds.lon.values, indexing='ij')
    psi = np.deg2rad(lat2d)
    lam = np.deg2rad(lon2d)
    
    x_geo = np.cos(psi) * np.cos(lam)
    y_geo = np.cos(psi) * np.sin(lam)
    z_geo = np.sin(psi)
    
    # Broadcast (lat, lon) -> (time, lat, lon) and convert to dask
    x_geo_3d = da.from_array(np.tile(x_geo[np.newaxis, :, :], (len(zarr_ds.time), 1, 1)), chunks=(100, *x_geo.shape))
    y_geo_3d = da.from_array(np.tile(y_geo[np.newaxis, :, :], (len(zarr_ds.time), 1, 1)), chunks=(100, *y_geo.shape))
    z_geo_3d = da.from_array(np.tile(z_geo[np.newaxis, :, :], (len(zarr_ds.time), 1, 1)), chunks=(100, *z_geo.shape))
    
    numer_features.append(x_geo_3d)
    numer_features.append(y_geo_3d)
    numer_features.append(z_geo_3d)

    numer_var_names = list(features) + ['sin_time', 'cos_time', 'masked_CHL', 'prev_day_CHL', 'next_day-CHL', 'x_geo', 'y_geo', 'z_geo']
    cat_var_names = ['land_flag', 'real_cloud_flag', 'valid_CHL_flag', 'fake_cloud_flag']

    # Numerical-predictor mean/std: a passed-in stats dict (shared across runs, no data read),
    # the precomputed IO.zarr constants, or the train-window stats computed once from the data.
    if stats is not None:
        feat_stats = stats['feat_stats']
        feat_mean = [feat_stats[name][0] for name in numer_var_names]
        feat_stdev = [feat_stats[name][1] for name in numer_var_names]
    elif use_hardcoded_stats:
        if IO_ZARR_STATS is None:
            raise ValueError(
                "IO_ZARR_STATS is not populated. Call build_standardized_lazy(..., "
                "use_hardcoded_stats=False) once, then paste stats['feat_stats'] and "
                "stats['CHL'] into IO_ZARR_STATS at the top of utils.py."
            )
        feat_stats = IO_ZARR_STATS['feat_stats']
        feat_mean = [feat_stats[name][0] for name in numer_var_names]
        feat_stdev = [feat_stats[name][1] for name in numer_var_names]
    else:
        # train-window mean/std for numerical predictors
        train_start_ind = np.where(zarr_ds.time.values == np.datetime64(f'{train_year}-01-01'))[0][0]
        train_end_ind = np.where(zarr_ds.time.values == np.datetime64(f'{train_year + train_range}-01-01'))[0][0]
        feat_mean, feat_stdev = [], []
        for feature in numer_features:
            feature_train = feature[train_start_ind:train_end_ind]
            feat_mean.append(da.nanmean(feature_train).compute())
            feat_stdev.append(da.nanstd(feature_train).compute())
        feat_stats = {name: [float(m), float(s)] for name, m, s in zip(numer_var_names, feat_mean, feat_stdev)}

    numer_features_stdized = [
        (feature - mean) / stdev
        for feature, mean, stdev in zip(numer_features, feat_mean, feat_stdev)
    ]

    # CHL label: standardize only if asked; passed-in, hardcoded, or all-time stats, else leave as log CHL
    if standardize_chl:
        if stats is not None:
            CHL_mean, CHL_stdev = stats['CHL']
        elif use_hardcoded_stats:
            CHL_mean, CHL_stdev = IO_ZARR_STATS['CHL']
        else:
            CHL_mean = da.nanmean(CHL_data).compute()
            CHL_stdev = da.nanstd(CHL_data).compute()
        CHL_out = (CHL_data - CHL_mean) / CHL_stdev
    else:
        CHL_mean, CHL_stdev = 0.0, 1.0
        CHL_out = CHL_data

    data_vars = {}
    for name, arr in zip(numer_var_names, numer_features_stdized):
        data_vars[name] = (("time", "lat", "lon"), arr)
    for name, arr in zip(cat_var_names, cat_features):
        data_vars[name] = (("time", "lat", "lon"), arr)
    data_vars["CHL"] = (("time", "lat", "lon"), CHL_out.data)

    coords = {c: zarr_ds.coords[c] for c in ("time", "lat", "lon")}
    if output_chunks is None:
        output_chunks = {"time": 100, "lat": -1, "lon": -1}
    ds_out = xr.Dataset(data_vars=data_vars, coords=coords).chunk(output_chunks)

    stats = {
        'CHL': np.array([CHL_mean, CHL_stdev]),
        'masked_CHL': np.array([feat_mean[-3], feat_stdev[-3]]),
        'feat_stats': feat_stats,
    }
    return ds_out, stats


def make_xbatcher(ds, patch_dims, overlap=None, preload_batch=False):
    """
    Create an xbatcher ``BatchGenerator`` over time/lat/lon windows (Eli's helper).

    Line up ``patch_dims`` for lat/lon with the dataset's on-disk chunking so xbatcher reads
    align (e.g. ``build_standardized_lazy(..., output_chunks={"time":100,"lat":40,"lon":56})``).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to tile.
    patch_dims : mapping
        Window size per dim, e.g. ``{"time": 100, "lat": 40, "lon": 56}``.
    overlap : mapping or None, default None
        Overlap per dim, e.g. ``{"time": 0, "lat": 16, "lon": 16}``. None = non-overlapping.
    preload_batch : bool, default False
        Passed through to ``BatchGenerator`` (eagerly ``.load()`` each batch when True).

    Returns
    -------
    xbatcher.BatchGenerator
    """
    import xbatcher as xb  # lazy: keep `import mindthegap` free of the xbatcher dependency
    kwargs = dict(ds=ds, input_dims=dict(patch_dims), preload_batch=preload_batch)
    if overlap is not None:
        kwargs["input_overlap"] = dict(overlap)
    return xb.BatchGenerator(**kwargs)


def UNet(input_shape):
    """
    Build and compile the gap-fill U-Net (three encoder/decoder levels, MSE loss, Adam).

    Fully convolutional: pass ``input_shape=(None, None, n_channels)`` to train on patches and
    still predict on the whole domain. Spatial dims must be multiples of 8 (three 2x pools).

    Parameters
    ----------
    input_shape : tuple
        ``(height, width, n_channels)``; height/width may be ``None`` (fully convolutional).

    Returns
    -------
    tf.keras.Model
        Compiled model.
    """
    import tensorflow as tf  # lazy: keep `import mindthegap` free of the heavy TF import
    from tensorflow.keras import Input, layers

    inputs = Input(shape=input_shape)
    x = inputs
    filters = [64, 128, 256]
    ec_images = []

    for f in filters:
        ec_images.append(x)
        x = layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNormalization()(x)

    for f, ec in zip(filters[:-1][::-1], ec_images[::-1][:-1]):
        x = layers.Conv2DTranspose(f, 3, 2, padding='same')(x)
        x = layers.concatenate([x, ec])
        x = layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(f, 3, 2, padding='same')(x)
    x = layers.concatenate([x, ec_images[0]])
    x = layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, (3, 3), padding='same', activation='linear')(x)

    model = tf.keras.Model(inputs, outputs, name='U-net')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def add_spherical_coords(obj, lat="lat", lon="lon"):
    """
    Add 3D unit-sphere coordinates (x_geo, y_geo, z_geo) computed from lat/lon.

    - If `obj` is an xarray.Dataset:
        * Assumes `lat` and `lon` are 1D coordinates.
        * Broadcasts to 2D over (lat, lon) and keeps Dask laziness.
        * Returns an xarray.Dataset with x_geo, y_geo, z_geo variables.

    - If `obj` is a pandas.DataFrame:
        * Assumes `lat` and `lon` are columns.
        * Computes per-row x_geo, y_geo, z_geo columns.
        * Returns a new DataFrame (original is not modified in place).

    Parameters
    ----------
    obj : xarray.Dataset or pandas.DataFrame
    lat : str
        Name of latitude coordinate/column in degrees.
    lon : str
        Name of longitude coordinate/column in degrees.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
    """
    # ---- xarray path --------------------------------------------------------
    if isinstance(obj, xr.Dataset):
        ds = obj

        # 2D lat/lon (lazy if dask)
        lat2d, lon2d = xr.broadcast(ds[lat], ds[lon])   # (lat, lon), (lat, lon)

        # radians (lazy)
        psi = xr.apply_ufunc(np.deg2rad, lat2d, dask="parallelized")
        lam = xr.apply_ufunc(np.deg2rad, lon2d, dask="parallelized")

        x_geo = xr.apply_ufunc(np.cos, psi, dask="parallelized") * xr.apply_ufunc(np.cos, lam, dask="parallelized")
        y_geo = xr.apply_ufunc(np.cos, psi, dask="parallelized") * xr.apply_ufunc(np.sin, lam, dask="parallelized")
        z_geo = xr.apply_ufunc(np.sin, psi, dask="parallelized")

        return ds.assign(
            x_geo=x_geo.astype("float32"),
            y_geo=y_geo.astype("float32"),
            z_geo=z_geo.astype("float32"),
        )

    # ---- pandas path --------------------------------------------------------
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        # radians (vectorized numpy on Series)
        psi = np.deg2rad(df[lat].to_numpy())
        lam = np.deg2rad(df[lon].to_numpy())

        x_geo = np.cos(psi) * np.cos(lam)
        y_geo = np.cos(psi) * np.sin(lam)
        z_geo = np.sin(psi)

        df["x_geo"] = x_geo.astype("float32")
        df["y_geo"] = y_geo.astype("float32")
        df["z_geo"] = z_geo.astype("float32")

        return df

    # ---- unsupported type ---------------------------------------------------
    raise TypeError(
        f"add_spherical_coords expected xarray.Dataset or pandas.DataFrame, "
        f"got {type(obj)}"
    )
