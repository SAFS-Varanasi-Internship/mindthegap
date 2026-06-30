from typing import Union
import xarray as xr

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

def make_tf_gen(batcher, x_vars, y_mean, y_std, feature_stats):
    """
    Creates a generator function for TensorFlow `tf.data.Dataset.from_generator`.
    
    This factory function iterates through an xbatcher dataset, applies synthetic 
    cloud masking, standardizes variables (log-transforming Chlorophyll), and 
    constructs the input features (X) and target labels (y) for a neural network.
    
    Args:
        batcher (xbatcher.BatchGenerator): An iterable xbatcher object containing 
            chunked xarray datasets.
        x_vars (list of str): List of feature names to extract and stack into 
            the final input tensor. Accepts dataset variables and special keywords:
            'masked_CHL', 'fake_cloud_flag', 'sin_time', and 'cos_time'.
        y_mean (float): The mean of the log-transformed target variable (for standardization).
        y_std (float): The standard deviation of the log-transformed target variable.
        feature_stats (dict): Dictionary containing the mean and standard deviation 
            for standardizing other input variables. Format: {'var_name': (mean, std)}.
            
    Returns:
        callable: A generator function `gen()` that yields tuples of (x, y) 
        where `x` is the input tensor and `y` is the target tensor.
    """
    def gen():
        for batch in batcher:
            time_len = batch.sizes["time"]
            for t in range(time_len):
                
                # --- 1. Synthetic Cloud Masking ---
                # Shift the cloud mask by 10 days to create "fake" clouds that 
                # do not perfectly correlate with the true missing data.
                fake_t = (t + 10) % time_len 
                fake_cloud_flag = batch['CHL_cmes-cloud'].isel(time=fake_t).values
                # Replace any NaN flags with 0.0 (no cloud)
                fake_cloud_flag = np.where(np.isnan(fake_cloud_flag), 0.0, fake_cloud_flag)

                # --- 2. Target Variable (True CHL) Preparation ---
                raw_true_chl = batch['CHL_cmes-gapfree'].isel(time=t).values
                
                # Suppress warnings for log(0) or log(NaN) temporarily
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_true_chl = np.log(raw_true_chl)
                    
                # Standardize the log-transformed true CHL using injected params
                true_chl_std = (log_true_chl - y_mean) / y_std
                true_chl_std = np.where(np.isnan(true_chl_std), 0.0, true_chl_std)
                true_chl_std = np.where(np.isinf(true_chl_std), 0.0, true_chl_std)
                
                # Apply the synthetic cloud mask to create the input CHL feature
                masked_chl = np.where(fake_cloud_flag == 1, 0.0, true_chl_std)

                # --- 3. Cyclical Time Encoding ---
                # Convert the day of the year into continuous circular features 
                # so the model understands that Dec 31st and Jan 1st are adjacent.
                day_of_year = batch['time'].isel(time=t).dt.dayofyear.item()
                sin_grid = np.full(raw_true_chl.shape, np.sin(2 * np.pi * day_of_year / 365), dtype=np.float32)
                cos_grid = np.full(raw_true_chl.shape, np.cos(2 * np.pi * day_of_year / 365), dtype=np.float32)

                # --- 4. Input Tensor Assembly ---
                x_slice = []
                for var in x_vars:
                    # Handle computed/special variables
                    if var == 'masked_CHL':
                        x_slice.append(masked_chl)
                    elif var == 'fake_cloud_flag':
                        x_slice.append(fake_cloud_flag)
                    elif var == 'sin_time':
                        x_slice.append(sin_grid)
                    elif var == 'cos_time':
                        x_slice.append(cos_grid)
                    
                    # Handle standard dataset variables
                    else:
                        # Extract raw values and fill NaNs with 0.0
                        raw_val = np.where(np.isnan(batch[var].isel(time=t).values), 0.0, batch[var].isel(time=t).values)
                        
                        # Standardize using the injected feature_stats if available
                        if var in feature_stats:
                            final_val = (raw_val - feature_stats[var][0]) / feature_stats[var][1]
                        else:
                            final_val = raw_val
                            
                        x_slice.append(final_val)

                # --- 5. Yielding Batch ---
                # Stack all feature arrays along the last dimension (channels)
                x = np.stack(x_slice, axis=-1).astype(np.float32)
                
                # Expand dimensions of y to match expected shape (..., 1)
                y = true_chl_std.astype(np.float32)[..., np.newaxis]
                
                yield x, y
                
    return gen
