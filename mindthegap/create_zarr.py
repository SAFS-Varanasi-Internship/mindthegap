from pathlib import Path
import xarray as xr

def create_zarr(
    ds: xr.Dataset,
    filename: str | Path,
    chunks: tuple[int, int, int] = (1, 40, 56),
    *,
    mode: str = "w",
    consolidated: bool = True,
) -> None:
    """Rechunk an xarray Dataset and write it as a Zarr v2 store."""

    chunk_map = {
        "time": chunks[0],
        "lat": chunks[1],
        "lon": chunks[2],
    }

    dataset_chunks = {
        dim: size
        for dim, size in chunk_map.items()
        if dim in ds.dims
    }

    ds_chunked = ds.chunk(dataset_chunks)

    for name in ds_chunked.variables:
        ds_chunked[name].encoding.clear()

    encoding = {}

    for name, da in ds_chunked.data_vars.items():
        if da.dims:
            encoding[name] = {
                "chunks": tuple(
                    chunk_map.get(dim, da.sizes[dim])
                    for dim in da.dims
                )
            }

    ds_chunked.to_zarr(
        filename,
        mode=mode,
        encoding=encoding,
        zarr_format=2,
        consolidated=consolidated,
    )

import numpy as np
import dask.array as da
import xarray as xr
import zarr
from os import path

def data_preprocessing_new(
    ds,
    target_variable="CHL_cmes-level3",
    missing_flag="CHL_cmes-cloud", # 1 is missing but could be observed; careful that land is not 1
    land_flag="CHL_cmes-land", # 1 is land
    features=None,
    train_dates=None,
    std_vars=None,
    log_target=True,
    missing_flag_shift=10,
    n_temporal_lags=1,  # Number of prev/next days to add
):
    """Prepare target, mask, seasonality, and lag features for model training.

    The function renames the target to ``full_target``, optionally log-transforms
    it, creates a synthetic masked target by shifting the observed missing-value
    pattern in time, adds true-missing and land flags, adds sine/cosine
    day-of-year features, optionally adds temporal lags of the masked target,
    and standardizes the requested variables using statistics computed over the
    selected training dates.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with ``time``, ``lat``, and ``lon`` coordinates.
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

    Returns
    -------
    xarray.Dataset
        Processed dataset containing the standardizable variables, seasonality
        features, missing/land flags, and per-variable training means and
        standard deviations.
    """
    features = list(features or [])

    # Subset to needed vars
    keep_vars = [target_variable] + features + [missing_flag, land_flag]
    ds = ds[keep_vars]

    # Check for required coords
    for c in ("time", "lat", "lon"):
        if not c in ds.coords:
            raise ValueError(f"Required coordinate '{c}' not found in ds. Rename coords if needed.")

    # spec the target
    ds = ds.rename({target_variable: "full_target"})
    print(f'target data created from {target_variable}')
    
    if log_target:
        ds["full_target"] = np.log(ds["full_target"].where(ds["full_target"] > 0))
        print('target data logged')

    # Shift the existing missing/cloud pattern to create the synthetic mask pattern
    shifted_missing_flag = ds[missing_flag].roll(
        time=-missing_flag_shift,
        roll_coords=False,
    )

    # Apply the synthetic mask to the target
    # shifted_missing_flag == 0 means hide the target value
    ds["masked_target"] = ds["full_target"].where(
        shifted_missing_flag != 0
    )
    print('target masked with synthetic missing added')

    # Create true missing mask
    ds["true_missing_flag"] = (
        ds[missing_flag] == 1
    ).astype("int8")
    print(f'true missing flag added from {missing_flag}')

    # Create land mask
    ds["land_flag"] = (
        ds[land_flag] == 1
    ).astype("int8")
    print(f'land flag added from {land_flag}')

    # Locations deliberately hidden for training:
    # shifted pattern says mask, but the location is not land,
    # not truly missing/cloudy, and the original target exists
    ds["synthetic_missing_flag"] = (
        (shifted_missing_flag == 0)
        & (ds["land_flag"] == 0)
        & (ds["true_missing_flag"] == 0)
        & ds["full_target"].notnull()
    ).astype("int8")
    print(f'synthetic missing flag created from {missing_flag} with {missing_flag_shift} day shift')
        
    # Add seasonality variables
    day_of_year = ds.time.dt.dayofyear
    ds["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25).astype("float32")    
    ds["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25).astype("float32")
    print('sin cos day added (not standardized)')
 
    # Variables that can optionally be standardized
    standardizable_vars = features + ["full_target", "masked_target"]

    # Add prev and next days with masking
    for i in range(1, n_temporal_lags + 1):
        prev_name = f"masked_target_m{i}"
        next_name = f"masked_target_p{i}"    
        ds[prev_name] = ds["masked_target"].shift(time=i)
        ds[next_name] = ds["masked_target"].shift(time=-i)    
        standardizable_vars.extend([prev_name, next_name])
        print(f'{prev_name} and {next_name} added')

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

    stats_source = ds[std_vars]
    if train_dates is not None:
        stats_source = stats_source.sel(time=train_dates)

    means_dict = {var: xr.DataArray(0.0) for var in standardizable_vars}
    stds_dict = {var: xr.DataArray(1.0) for var in standardizable_vars}

    if std_vars:
        computed_means = stats_source.mean(dim=("time", "lat", "lon"))
        computed_stds = stats_source.std(dim=("time", "lat", "lon"))
        means_dict.update({var: computed_means[var] for var in std_vars})
        stds_dict.update({var: computed_stds[var] for var in std_vars})

    means = xr.Dataset(means_dict)
    stds = xr.Dataset(stds_dict)

    # Standardize every time step for the requested variables
    ds_standardized = ds.copy()
    if std_vars:
        ds_standardized.update((ds[std_vars] - means[std_vars]) / stds[std_vars])

    # Final clean
    keep_vars = standardizable_vars + ["day_sin", "day_cos", "synthetic_missing_flag", "true_missing_flag", "land_flag"]
    ds_standardized = ds_standardized[keep_vars]

    # Add the standardization variables to the processed data
    for var in standardizable_vars:
        ds_standardized[f"{var}_train_mean"] = means[var]
        ds_standardized[f"{var}_train_std"] = stds[var]

    return ds_standardized
    

def data_preprocessing(
    zarr_ds, 
    target_variable="CHL_cmes-level3",
    cloud_variable="CHL_cmes-cloud",
    features=None, 
    train_year=None,
    train_range=None, 
    zarr_tag=None,
    datadir=None,
    log_target=True,
    fake_cloud_day_shift=10,
):
    numer_features = []  # numerical features
    cat_features = []  # categorical features
    zarr_label = f'{train_year}_{train_range}'  # later passed to create_zarr as zarr file name
    zarr_label = f'{zarr_label}_{zarr_tag}'

    print('label created')

    if path.exists(f'{datadir}/{zarr_label}.zarr'):
        print('Zarr file exists')
        return zarr_label
    
    # add raw data features
    for feature in features:
        feat_arr = zarr_ds[feature].data
        numer_features.append(feat_arr)
    print('raw data features added')

    # get target that is being predicted
    target_data = zarr_ds[target_variable]
    if log_target:
        target_data = np.log(target_data.copy())
        print('target data logged')
    numer_features.append(target)

    print(f'target variable {target_variable} added')
    
    # artifically masked target (n day shift)
    day_shift_flag = np.vstack((zarr_ds[cloud_variable].data[fake_cloud_day_shift:], zarr_ds[cloud_variable].data[:fake_cloud_day_shift]))
    assert target_data.shape == day_shift_flag.shape
    
    masked_target = da.where(day_shift_flag == 0, np.nan, target_data)
    numer_features.append(masked_target)

    print('masked target added (fake cloud applied)')

    prev_day = np.vstack((np.zeros((1, ) + target_data[0].shape), target_data.data[:-1]))
    numer_features.append(prev_day)
    print('prev day target added')
    next_day = np.vstack((target_data.data[1:], np.zeros((1, ) + target_data[0].shape)))
    numer_features.append(next_day)
    print('next day target added')

    # land one-hot encoding
    land_flag = da.zeros(target_data.shape)
    land_flag = da.where(zarr_ds[cloud_variable][0] == 2, 1, land_flag)
    cat_features.append(land_flag)
    
    print('land flag added')

    # real cloud one-hot encoding
    real_cloud_flag = da.zeros(target_data.shape)
    real_cloud_flag = da.where(zarr_ds[cloud_variable] == 1, 1, real_cloud_flag)
    cat_features.append(real_cloud_flag)

    print('real cloud flag added')

    # unmasked target one-hot encoding
    valid_target_flag = da.zeros(target_data.shape)
    valid_target_flag = da.where(~da.isnan(masked_target), 1, valid_target_flag)
    cat_features.append(valid_target_flag)

    print('valid (unmasked) target flag added')

    # fake cloud one-hot encoding
    fake_cloud_flag = da.zeros(target_data.shape)
    fake_cloud_flag = da.where((land_flag + real_cloud_flag + valid_target_flag) == 0, 1, fake_cloud_flag)
    cat_features.append(fake_cloud_flag)

    print('fake cloud flag added')

    # find train data start and end indices
    train_start_ind = np.where(zarr_ds.time.values == np.datetime64(f'{train_year}-01-01'))[0][0]
    train_end_ind = np.where(zarr_ds.time.values == np.datetime64(f'{train_year + train_range}-01-01'))[0][0]
    
    # get mean and stdev for numerical features
    feat_mean = []
    feat_stdev = []

    # compute standardization metrics from the traning data
    for feature in numer_features:
        feature_train = feature[train_start_ind: train_end_ind]
        feat_mean.append(da.nanmean(feature_train).compute())
        feat_stdev.append(da.nanstd(feature_train).compute())
        print(f'calculating mean and stdev of {feature}')

    # calculate standardized features for all the data; not just training
    numer_features_stdized = []
    feature_shape = numer_features[0].shape
    for feature, mean, stdev in zip(numer_features, feat_mean, feat_stdev):
        numer_features_stdized.append((feature - da.full(feature_shape, mean)) / da.full(feature_shape, stdev))
        print(f'standardizing {feature} in full dataset')
    
    # Add sin and cos time features AFTER standardization (they should not be standardized)
    # sin and cos of day for seasonal features - already in [-1, 1] range
    time_data = da.array(zarr_ds.time)
    day_rad = (time_data - np.datetime64("1900-01-01")) / np.timedelta64(1, "D") / 365 * 2 * np.pi
    day_rad = day_rad.astype(np.float32)
    day_sin = np.sin(day_rad)
    day_cos = np.cos(day_rad)
    print('sin and cos time calculated')
    day_sin = np.tile(day_sin[:, np.newaxis, np.newaxis], (1,) + target_data[0].shape)
    day_sin = da.rechunk(day_sin, (100, *day_sin.shape[1:]))
    numer_features_stdized.append(day_sin)
    print('sin time added (not standardized)')
    day_cos = np.tile(day_cos[:, np.newaxis, np.newaxis], (1,) + target_data[0].shape)
    day_cos = da.rechunk(day_cos, (100, *day_cos.shape[1:]))
    numer_features_stdized.append(day_cos)
    print('cos time added (not standardized)')

    # Save all standardization statistics
    # Build dictionary with stats for all standardized features
    stats_dict = {}
    
    # Add stats for each standardized target and feature (excluding sin_time and cos_time which weren't standardized)
    standardized_var_names = features + ['target', 'masked_target', 'prev_day_target', 'next_day_target']
    for var_name, mean, stdev in zip(standardized_var_names, feat_mean, feat_stdev):
        stats_dict[var_name] = np.array([mean, stdev])
    
    # Add identity stats for sin_time and cos_time (mean=0, std=1 since they weren't standardized)
    stats_dict['sin_time'] = np.array([0.0, 1.0])
    stats_dict['cos_time'] = np.array([0.0, 1.0])
    
    np.save(f'{datadir}/{zarr_label}.npy', stats_dict)
    print(f'Saved standardization stats for {len(stats_dict)} variables')

    # calculate standardized target
    target_data_stdized = (target_data - da.full(feature_shape, target_mean)) / da.full(feature_shape, target_stdev)

    print('all standardized')

    numer_var_names = features + ['sin_time', 'cos_time', 'target', 'masked_target', 'prev_day_target', 'next_day-target']
    cat_var_names = ['land_flag', 'real_cloud_flag', 'valid_target_flag', 'fake_cloud_flag']

    print('creating zarr')
    create_zarr(zarr_ds, numer_features_stdized, numer_var_names, cat_features, cat_var_names, target_data_stdized.data, zarr_label, datadir=datadir)

    del time_data, day_rad, day_sin, day_cos
    del feature, feat_arr
    del numer_features, numer_features_stdized, numer_var_names, cat_features, cat_var_names, target_data, target_data_stdized
    del feat_mean, feat_stdev

    return zarr_label