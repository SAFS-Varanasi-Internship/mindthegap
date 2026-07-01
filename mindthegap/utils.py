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


def build_standardized_lazy(zarr_ds, features, train_year, train_range, standardize_chl=False):
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

    Returns
    -------
    ds_out : xr.Dataset
        Lazy standardized dataset: predictor channels + ``CHL`` label, chunked
        ``{"time": 100, "lat": -1, "lon": -1}``.
    stats : dict
        ``{'CHL': array([mean, std]), 'masked_CHL': array([mean, std])}`` (train-window
        stats; ``CHL`` is ``[0.0, 1.0]`` when ``standardize_chl=False``).
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

    prev_day = np.vstack((np.zeros((1,) + CHL_data[0].shape), CHL_data.data[:-1]))
    numer_features.append(prev_day)
    next_day = np.vstack((CHL_data.data[1:], np.zeros((1,) + CHL_data[0].shape)))
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

    # train-window mean/std for numerical predictors
    train_start_ind = np.where(zarr_ds.time.values == np.datetime64(f'{train_year}-01-01'))[0][0]
    train_end_ind = np.where(zarr_ds.time.values == np.datetime64(f'{train_year + train_range}-01-01'))[0][0]

    feat_mean, feat_stdev = [], []
    for feature in numer_features:
        feature_train = feature[train_start_ind:train_end_ind]
        feat_mean.append(da.nanmean(feature_train).compute())
        feat_stdev.append(da.nanstd(feature_train).compute())

    numer_features_stdized = [
        (feature - mean) / stdev
        for feature, mean, stdev in zip(numer_features, feat_mean, feat_stdev)
    ]

    # CHL label: standardize (all-time stats) only if asked; otherwise leave as log CHL
    if standardize_chl:
        CHL_mean = da.nanmean(CHL_data).compute()
        CHL_stdev = da.nanstd(CHL_data).compute()
        CHL_out = (CHL_data - CHL_mean) / CHL_stdev
    else:
        CHL_mean, CHL_stdev = 0.0, 1.0
        CHL_out = CHL_data

    numer_var_names = list(features) + ['sin_time', 'cos_time', 'masked_CHL', 'prev_day_CHL', 'next_day-CHL']
    cat_var_names = ['land_flag', 'real_cloud_flag', 'valid_CHL_flag', 'fake_cloud_flag']

    data_vars = {}
    for name, arr in zip(numer_var_names, numer_features_stdized):
        data_vars[name] = (("time", "lat", "lon"), arr)
    for name, arr in zip(cat_var_names, cat_features):
        data_vars[name] = (("time", "lat", "lon"), arr)
    data_vars["CHL"] = (("time", "lat", "lon"), CHL_out.data)

    coords = {c: zarr_ds.coords[c] for c in ("time", "lat", "lon")}
    ds_out = xr.Dataset(data_vars=data_vars, coords=coords).chunk({"time": 100, "lat": -1, "lon": -1})

    stats = {
        'CHL': np.array([CHL_mean, CHL_stdev]),
        'masked_CHL': np.array([feat_mean[-3], feat_stdev[-3]]),
    }
    return ds_out, stats
