from pathlib import Path
from typing import Sequence, Union
import numpy as np
import xarray as xr


def create_zarr(
    zarr_ds: xr.Dataset,
    numer_features: Sequence[np.ndarray],
    numer_var_names: Sequence[str],
    cat_features: Sequence[np.ndarray],
    cat_var_names: Sequence[str],
    CHL_data: np.ndarray,
    zarr_label: str,
    datadir: Union[str, Path] = "data",
):
    """
    Create a feature/target dataset and write it to a Zarr store.

    This function packages the provided numerical and categorical feature arrays
    (e.g., standardized predictors) together with the target variable
    (standardized/log CHL) into a single ``xarray.Dataset`` with coordinates
    derived from ``zarr_ds`` and writes it to a Zarr store on disk.

    Parameters
    ----------
    zarr_ds : xarray.Dataset
        A reference dataset used to supply coordinates and dimension ordering.
        Must contain coordinates named ``time``, ``lat``, and ``lon`` that match
        the shapes of the provided feature and target arrays.
    numer_features : Sequence[np.ndarray]
        Iterable of 3D arrays for numerical predictors, each shaped
        ``(time, lat, lon)`` and aligned to ``zarr_ds``.
    numer_var_names : Sequence[str]
        Variable names corresponding 1:1 to ``numer_features`` (same length).
    cat_features : Sequence[np.ndarray]
        Iterable of 3D arrays for categorical/flag predictors, each shaped
        ``(time, lat, lon)`` and aligned to ``zarr_ds``.
    cat_var_names : Sequence[str]
        Variable names corresponding 1:1 to ``cat_features`` (same length).
    CHL_data : np.ndarray
        Target variable array shaped ``(time, lat, lon)`` (e.g., standardized or
        log-transformed CHL), aligned to ``zarr_ds``.
    zarr_label : str
        Label used to name the Zarr store (e.g., ``"<label>.zarr"``).
    datadir : str or path-like, optional
        Directory where the Zarr store will be created. Defaults to ``"data"``.
        (This is the only addition versus the original function.)

    Returns
    -------
    str
        The filesystem path to the written Zarr store.

    Notes
    -----
    - Assumes all arrays share the same dimensions and ordering:
      ``(time, lat, lon)``.
    - Variable names should be valid xarray variable identifiers. If any name
      contains characters that are awkward to index (e.g., hyphens), prefer
      selection via ``.data_vars["name-with-hyphen"]``.
    - This version preserves the original computational logic; only the output
      directory is now configurable via ``datadir``.

    Side Effects
    ------------
    - Writes a Zarr store to ``<datadir>/<zarr_label>.zarr`` (overwrites if
      it already exists).
    """
    # --- Build the dataset exactly as in the original, only changing the output path ---
    # Use coords from the reference dataset to guarantee consistency
    coords = {}
    for c in ("time", "lat", "lon"):
        if c in zarr_ds.coords:
            coords[c] = zarr_ds.coords[c]
        elif c in zarr_ds:
            coords[c] = zarr_ds[c]
        else:
            raise ValueError(f"Required coordinate '{c}' not found in zarr_ds.")

    # Map features to names
    if len(numer_features) != len(numer_var_names):
        raise ValueError("Length mismatch: numer_features vs numer_var_names.")
    if len(cat_features) != len(cat_var_names):
        raise ValueError("Length mismatch: cat_features vs cat_var_names.")

    data_vars = {}

    # Numerical predictors
    for name, arr in zip(numer_var_names, numer_features):
        data_vars[name] = (("time", "lat", "lon"), arr)

    # Categorical/flag predictors
    for name, arr in zip(cat_var_names, cat_features):
        data_vars[name] = (("time", "lat", "lon"), arr)

    # Target variable
    data_vars["CHL"] = (("time", "lat", "lon"), CHL_data)

    ds_out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Ensure output directory exists
    datadir = Path(datadir)
    datadir.mkdir(parents=True, exist_ok=True)

    # make chunking uniform across variables
    ds_out = ds_out.chunk({"time": 100, "lat": -1, "lon": -1})

    # Write Zarr (same behavior, only path is configurable now)
    store_path = datadir / f"{zarr_label}.zarr"
    ds_out.to_zarr(store_path.as_posix(), mode="w", consolidated=True, zarr_format=2)

    return store_path.as_posix()

import numpy as np
import dask.array as da
import xarray as xr
import zarr
from os import path

def data_preprocessing(zarr_ds, features, train_year, train_range, zarr_tag, datadir):
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

    # get label
    CHL_data = zarr_ds['CHL_cmes-level3']
    CHL_data = np.log(CHL_data.copy())
    print('CHL logged')
    
    # additional features
    # sin and cos of day for seasonal features
    time_data = da.array(zarr_ds.time)
    day_rad = (time_data - np.datetime64("1900-01-01")) / np.timedelta64(1, "D") / 365 * 2 * np.pi
    day_rad = day_rad.astype(np.float32)
    day_sin = np.sin(day_rad)
    day_cos = np.cos(day_rad)
    print('sin and cos time calculated')
    day_sin = np.tile(day_sin[:, np.newaxis, np.newaxis], (1,) + CHL_data[0].shape)
    day_sin = da.rechunk(day_sin, (100, *day_sin.shape[1:]))
    numer_features.append(day_sin)
    print('sin time added')
    day_cos = np.tile(day_cos[:, np.newaxis, np.newaxis], (1,) + CHL_data[0].shape)
    day_cos = da.rechunk(day_cos, (100, *day_cos.shape[1:]))
    numer_features.append(day_cos)
    print('cos time added')

    
    # artifically masked CHL (10 day shift)
    day_shift_flag = np.vstack((zarr_ds['CHL_cmes-cloud'].data[10:], zarr_ds['CHL_cmes-cloud'].data[:10]))
    assert CHL_data.shape == day_shift_flag.shape
    
    masked_CHL = da.where(day_shift_flag == 0, np.nan, CHL_data)
    numer_features.append(masked_CHL)

    print('masked CHL added')

    prev_day = np.vstack((np.zeros((1, ) + CHL_data[0].shape), CHL_data.data[:-1]))
    numer_features.append(prev_day)
    print('prev day CHL added')
    next_day = np.vstack((CHL_data.data[1:], np.zeros((1, ) + CHL_data[0].shape)))
    numer_features.append(next_day)
    print('next day CHL added')

    # land one-hot encoding
    land_flag = da.zeros(CHL_data.shape)
    land_flag = da.where(zarr_ds['CHL_cmes-cloud'][0] == 2, 1, land_flag)
    cat_features.append(land_flag)
    
    print('land flag added')

    # real cloud one-hot encoding
    real_cloud_flag = da.zeros(CHL_data.shape)
    real_cloud_flag = da.where(zarr_ds['CHL_cmes-cloud'] == 1, 1, real_cloud_flag)
    cat_features.append(real_cloud_flag)

    print('real cloud flag added')

    # valid CHL one-hot encoding
    valid_CHL_flag = da.zeros(CHL_data.shape)
    valid_CHL_flag = da.where(~da.isnan(masked_CHL), 1, valid_CHL_flag)
    cat_features.append(valid_CHL_flag)

    print('valid CHL flag added')

    # fake cloud one-hot encoding
    fake_cloud_flag = da.zeros(CHL_data.shape)
    fake_cloud_flag = da.where((land_flag + real_cloud_flag + valid_CHL_flag) == 0, 1, fake_cloud_flag)
    cat_features.append(fake_cloud_flag)

    print('fake cloud flag added')

    # find train data start and end indices
    train_start_ind = np.where(zarr_ds.time.values == np.datetime64(f'{train_year}-01-01'))[0][0]
    train_end_ind = np.where(zarr_ds.time.values == np.datetime64(f'{train_year + train_range}-01-01'))[0][0]
    
    # get mean and stdev for numerical features
    feat_mean = []
    feat_stdev = []

    for feature in numer_features:
        feature_train = feature[train_start_ind: train_end_ind]
        feat_mean.append(da.nanmean(feature_train).compute())
        feat_stdev.append(da.nanstd(feature_train).compute())
        print('calculating mean and stdev...')

    # calculate standardized features
    numer_features_stdized = []
    feature_shape = numer_features[0].shape
    for feature, mean, stdev in zip(numer_features, feat_mean, feat_stdev):
        numer_features_stdized.append((feature - da.full(feature_shape, mean)) / da.full(feature_shape, stdev))
        print('standardizing...')

    # get mean and stdev for CHL
    CHL_mean = da.nanmean(CHL_data).compute()
    CHL_stdev = da.nanstd(CHL_data).compute()
    np.save(f'{datadir}/{zarr_label}.npy', {'CHL': np.array([CHL_mean, CHL_stdev]), 'masked_CHL': np.array([feat_mean[-3], feat_stdev[-3]])})

    # calculate standardized CHL
    CHL_data_stdized = (CHL_data - da.full(feature_shape, CHL_mean)) / da.full(feature_shape, CHL_stdev)

    print('all standardized')

    numer_var_names = features + ['sin_time', 'cos_time', 'masked_CHL', 'prev_day_CHL', 'next_day-CHL']
    cat_var_names = ['land_flag', 'real_cloud_flag', 'valid_CHL_flag', 'fake_cloud_flag']

    print('creating zarr')
    create_zarr(zarr_ds, numer_features_stdized, numer_var_names, cat_features, cat_var_names, CHL_data_stdized.data, zarr_label, datadir=datadir)

    del time_data, day_rad, day_sin, day_cos
    del feature, feat_arr
    del numer_features, numer_features_stdized, numer_var_names, cat_features, cat_var_names, CHL_data, CHL_data_stdized
    del feat_mean, feat_stdev

    return zarr_label