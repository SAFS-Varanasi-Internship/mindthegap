from .utils import unstdize, compute_mae, compute_mse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Sequence, Union


def _map_extent(zarr_like):
    lon = np.asarray(zarr_like.lon.to_numpy() if hasattr(zarr_like.lon, "to_numpy") else zarr_like.lon, dtype=float)
    lat = np.asarray(zarr_like.lat.to_numpy() if hasattr(zarr_like.lat, "to_numpy") else zarr_like.lat, dtype=float)

    def _axis_bounds(coord):
        if coord.size < 2:
            half_step = 0.5
            return coord[0] - half_step, coord[0] + half_step
        return (float(np.min(coord)), float(np.max(coord)))

    lon_min, lon_max = _axis_bounds(lon)
    lat_min, lat_max = _axis_bounds(lat)
    return [lon_min, lon_max, lat_min, lat_max]


def plot_prediction_observed(
    zarr_stdized,
    zarr_ds,
    zarr_label, 
    model, 
    date_to_predict,
    datadir: Union[str, Path] = "data",
):
    """
    Plot observed vs. predicted log(Chl-a) for a single date, along with flags and differences.

    This function:
      1) loads mean/std used during standardization from ``{datadir}/{zarr_label}.npy``,
      2) builds a predictor tensor X for the requested date from all variables in ``zarr_stdized``
         except ``"CHL"``, filling NaNs with 0.0,
      3) gets the observed Level-3 log(Chl-a) for the same date from the global ``zarr_ds``,
      4) runs the model to produce standardized predictions, then unstandardizes back to log-scale
         using ``utils.unstdize``,
      5) masks predictions where the observation is NaN, and
      6) produces a 2×2 panel plot: observed, flags, predicted, and (observed − predicted).

    Parameters
    ----------
    zarr_stdized : xarray.Dataset
        Dataset containing standardized predictors (and flags) for the model input. Must include
        the variables used by the model as well as ``'CHL'`` (which is removed from X).
    zarr_ds : xarray.Dataset
        The dataset from which zarr_stdized was created.
    zarr_label : str
        Label used to locate the standardization sidecar file ``{datadir}/{zarr_label}.npy``
        containing ``{'CHL': array([mean, std]), 'masked_CHL': ...}``.
    model : tf.keras.Model
        Trained U-Net (or compatible) model expecting input shaped (H, W, C) and returning
        a single-channel prediction of standardized log(Chl-a).
    date_to_predict : str or numpy.datetime64 or pandas.Timestamp
        Date to visualize; must match the ``time`` coordinate in ``zarr_stdized`` and ``zarr_ds``.

    Notes
    -----
    - This function expects the following globals to exist in the module scope:
        * ``datadir``: directory where ``{zarr_label}.npy`` lives,
        * ``zarr_ds``: the source dataset containing the observed variable
          ``'CHL_cmes-level3'`` and flag layers (``land_flag``, ``real_cloud_flag``,
          ``valid_CHL_flag``, ``fake_cloud_flag``).
    - Predictions are unstandardized with :func:`utils.unstdize`.
    - The hard-coded map extent corresponds to the Arab Sea region
      ``[lon_min, lon_max, lat_min, lat_max] = [42, 101.75, -11.75, 32]``.

    See Also
    --------
    utils.unstdize : Convert standardized values back to original (log-scale) units.
    utils.compute_mae : Mean absolute error (NaN-aware).
    utils.compute_mse : Mean squared error (NaN-aware).

    Example
    -------
    >>> # in your package code, with datadir and zarr_ds defined at module scope:
    >>> plot_prediction_observed(zarr_stdized, zarr_label="2015_3_ArabSea_full_2days",
    ...                          model=unet_model, date_to_predict="2015-03-15")
    """
    mean_std = np.load(f'{datadir}/{zarr_label}.npy', allow_pickle='TRUE').item()
    mean, std = mean_std['CHL'][0], mean_std['CHL'][1]

    zarr_date = zarr_stdized.sel(time=date_to_predict)

    # Build model input X from all standardized variables except "CHL"
    X = []
    X_vars = list(zarr_stdized.keys())
    X_vars.remove('CHL')
    for var in X_vars:
        var = zarr_date[var].to_numpy()
        X.append(np.where(np.isnan(var), 0.0, var))
    X = np.array(X)
    X = np.moveaxis(X, 0, -1)  # (C,H,W) -> (H,W,C)

    # Observed log(Chl-a)
    true_CHL = np.log(zarr_ds.sel(time=date_to_predict)['CHL_cmes-level3'].to_numpy())

    # Apply fake-cloud mask to observation for display
    fake_cloud_flag = zarr_date.fake_cloud_flag.to_numpy()
    masked_CHL = np.where(fake_cloud_flag == 1, np.nan, true_CHL)

    # Predict (standardized), then unstandardize to log-scale using utils.unstdize
    predicted_CHL = model.predict(X[np.newaxis, ...], verbose=0)[0]
    predicted_CHL = predicted_CHL[:, :, 0]
    predicted_CHL = unstdize(predicted_CHL, mean, std)

    # Keep NaN wherever observation is NaN (for fair visual diff)
    predicted_CHL = np.where(np.isnan(true_CHL), np.nan, predicted_CHL)
    diff = true_CHL - predicted_CHL

    # Flags panel data (0=land/real cloud, 1=fake cloud, 2=observed valid)
    flag = np.zeros(true_CHL.shape)
    flag = np.where(zarr_date['land_flag'] == 1, 0, flag)
    flag = np.where(zarr_date['valid_CHL_flag'] == 1, 2, flag)
    flag = np.where(zarr_date['real_cloud_flag'] == 1, 0, flag)
    flag = np.where(zarr_date['fake_cloud_flag'] == 1, 1, flag)

    # Color limits matched between observed and predicted
    vmax = np.nanmax((true_CHL, predicted_CHL))
    vmin = np.nanmin((true_CHL, predicted_CHL))

    extent = _map_extent(zarr_stdized)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    im0 = axes[0, 0].imshow(true_CHL, vmin=vmin, vmax=vmax, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(), interpolation='nearest')
    axes[0, 0].add_feature(cfeature.COASTLINE)
    axes[0, 0].set_xlabel('longitude'); axes[0, 0].set_ylabel('latitude')
    axes[0, 0].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[0, 0].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[0, 0].set_title('Observed Level-3 log Chl-a', size=14)

    im1 = axes[0, 1].imshow(flag, extent=extent, origin='upper',
                            transform=ccrs.PlateCarree())
    axes[0, 1].add_feature(cfeature.COASTLINE, color='white')
    axes[0, 1].set_xlabel('longitude'); axes[0, 1].set_ylabel('latitude')
    axes[0, 1].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[0, 1].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[0, 1].set_title('Land, Cloud, and Observed Flags After Applying Fake Cloud', size=13)

    im2 = axes[1, 0].imshow(predicted_CHL, vmin=vmin, vmax=vmax, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(), interpolation='nearest')
    axes[1, 0].add_feature(cfeature.COASTLINE, color='white')
    axes[1, 0].imshow(np.where(flag == 1, np.nan, flag), vmax=2, vmin=0,
                      extent=extent, origin='upper', interpolation='nearest', alpha=1)
    axes[1, 0].set_xlabel('longitude'); axes[1, 0].set_ylabel('latitude')
    axes[1, 0].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[1, 0].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[1, 0].set_title('Predicted log Chl-a from U-Net', size=14)

    vmin2, vmax2 = -1, 1
    im3 = axes[1, 1].imshow(diff, vmin=vmin2, vmax=vmax2, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(),
                            cmap=plt.cm.RdBu, interpolation='nearest')
    axes[1, 1].add_feature(cfeature.COASTLINE)
    axes[1, 1].set_xlabel('longitude'); axes[1, 1].set_ylabel('latitude')
    axes[1, 1].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[1, 1].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    # (Optional) show quick metrics using utils helpers
    mae = compute_mae(true_CHL, predicted_CHL)
    mse = compute_mse(true_CHL, predicted_CHL)
    axes[1, 1].set_title(f'Difference (obs − pred)\nMAE={mae:.3f}, MSE={mse:.3f}', size=13)

    fig.subplots_adjust(right=0.76)
    cbar1_ax = fig.add_axes([0.79, 0.14, 0.025, 0.72]); fig.colorbar(im0, cax=cbar1_ax).ax.set_ylabel(
        'log Chl-a (mg/m-3)', rotation=270, size=14, labelpad=16)
    cbar2_ax = fig.add_axes([0.86, 0.14, 0.025, 0.72]); fig.colorbar(im1, cax=cbar2_ax).ax.set_ylabel(
        'land & real cloud = 0, fake cloud = 1, observed after masking = 2', rotation=270, size=14, labelpad=20)
    cbar3_ax = fig.add_axes([0.94, 0.14, 0.025, 0.72]); fig.colorbar(im3, cax=cbar3_ax).ax.set_ylabel(
        'difference in log Chl-a', rotation=270, size=14, labelpad=16)

    plt.show()

import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .utils import unstdize

def plot_prediction_gapfill(
    zarr_stdized, 
    zarr_ds, 
    zarr_label, model, 
    date_to_predict,
    datadir: Union[str, Path] = "data",
):
    """
    Plot gap-free (Level-4) log(Chl-a) vs. U-Net gap-filled prediction for a single date.

    This function:
      1) Loads the CHL standardization stats from ``{datadir}/{zarr_label}.npy``.
      2) Builds the model input tensor X for the requested date from all variables in
         ``zarr_stdized`` (after swapping a few names so the masked CHL slot is populated
         from Level-3 observations).
      3) Uses ``zarr_ds`` to fetch observed Level-4 gapfree log(Chl-a) (target for comparison)
         and Level-3 log(Chl-a) (to fill the masked-CHL input).
      4) Runs the model, unstandardizes predictions to log scale, and computes both log-space
         and absolute-space differences.
      5) Produces a 2×2 panel: (gapfree obs), (U-Net prediction), (log diff), (absolute diff).

    Parameters
    ----------
    zarr_stdized : xarray.Dataset
        Standardized predictor dataset used as model inputs (includes flags). Must contain all
        variables the model expects, plus ``'CHL'`` (which is replaced in the masked-CHL slot).
    zarr_ds : xarray.Dataset
        Source dataset containing observational variables used here:
        - ``'CHL_cmes-level3'`` (Level-3, used to populate masked CHL input),
        - ``'CHL_cmes-gapfree'`` (Level-4 gapfree, compared against predictions).
    zarr_label : str
        Label used to locate the standardization sidecar file
        ``{datadir}/{zarr_label}.npy`` with keys like ``{'CHL': [mean, std], 'masked_CHL': [...]}``.
    model : tf.keras.Model
        Trained U-Net (or compatible) that outputs a single-channel standardized log(Chl-a).
    date_to_predict : str | numpy.datetime64 | pandas.Timestamp
        Date to visualize; must exist in the ``time`` coordinate of both datasets.

    Notes
    -----
    - This function expects a module-level variable ``datadir`` to be defined, pointing to the
      directory where ``{zarr_label}.npy`` lives.
    - Unstandardization is performed via ``utils.unstdize``.
    - The map extent is currently fixed to ``[42, 101.75, -11.75, 32]`` (Arab Sea).

    Example
    -------
    >>> plot_prediction_gapfill(
    ...     zarr_stdized=stdized_ds,
    ...     zarr_ds=raw_ds,
    ...     zarr_label="2015_3_ArabSea_full_2days",
    ...     model=unet_model,
    ...     date_to_predict="2015-03-15",
    ... )
    """
    mean_std = np.load(f'{datadir}/{zarr_label}.npy', allow_pickle='TRUE').item()
    mean, std = mean_std['CHL'][0], mean_std['CHL'][1]
    zarr_date = zarr_stdized.sel(time=date_to_predict)

    X = []
    X_vars = list(zarr_stdized.keys())
    X_vars.remove('CHL')
    X_vars[X_vars.index('masked_CHL')] = 'CHL'
    X_vars[X_vars.index('real_cloud_flag')] = 'a'
    X_vars[X_vars.index('fake_cloud_flag')] = 'real_cloud_flag'
    X_vars[X_vars.index('a')] = 'fake_cloud_flag'
    
    for var in X_vars:
        var = zarr_date[var].to_numpy()
        X.append(np.where(np.isnan(var), 0.0, var))

    valid_CHL_ind = X_vars.index('valid_CHL_flag')
    X[valid_CHL_ind] = da.where(X[X_vars.index('fake_cloud_flag')] == 1, 1, X[valid_CHL_ind])

    X[X_vars.index('fake_cloud_flag')] = np.zeros(X[0].shape)

    X_masked_CHL = np.log(zarr_ds.sel(time=date_to_predict)['CHL_cmes-level3'].to_numpy())
    X_masked_CHL = (X_masked_CHL - da.full(X_masked_CHL.shape, mean_std['masked_CHL'][0])) / da.full(X_masked_CHL.shape, mean_std['masked_CHL'][1])
    X_vars[X_vars.index('CHL')] = X_masked_CHL

    X = np.array(X)
    X = np.moveaxis(X, 0, -1)

    true_CHL = np.log(zarr_ds.sel(time=date_to_predict)['CHL_cmes-gapfree'].to_numpy())
    masked_CHL = np.log(zarr_ds.sel(time=date_to_predict)['CHL_cmes-level3'].to_numpy())

    predicted_CHL = model.predict(X[np.newaxis, ...], verbose=0)[0]
    predicted_CHL = predicted_CHL[:, :, 0]
    predicted_CHL = unstdize(predicted_CHL, mean, std)
    predicted_CHL = np.where(np.isnan(true_CHL), np.nan, predicted_CHL)

    log_diff = true_CHL - predicted_CHL
    diff = np.exp(true_CHL) - np.exp(predicted_CHL)

    vmax = np.nanmax((true_CHL, predicted_CHL))
    vmin = np.nanmin((true_CHL, predicted_CHL))

    extent = _map_extent(zarr_stdized)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    im0 = axes[0, 0].imshow(true_CHL, vmin=vmin, vmax=vmax, extent=extent, origin='upper', transform=ccrs.PlateCarree())
    axes[0, 0].set_xlabel('longitude')
    axes[0, 0].set_ylabel('latitude')
    axes[0, 0].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[0, 0].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[0, 0].set_title('Log Chl-a from the Gapfree \nLevel-4 GlobColour Copernicus Product', size=14)
    
    im1 = axes[0, 1].imshow(predicted_CHL, extent=extent, origin='upper', transform=ccrs.PlateCarree())
    axes[0, 1].set_xlabel('longitude')
    axes[0, 1].set_ylabel('latitude')
    axes[0, 1].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[0, 1].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[0, 1].set_title('Gapfree log Chl-a from U-Net', size=14)
    
    vmax2 = 1
    vmin2 = -1
    im2 = axes[1, 0].imshow(log_diff, vmin=vmin2, vmax=vmax2, extent=extent, origin='upper', transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu)
    axes[1, 0].set_xlabel('longitude')
    axes[1, 0].set_ylabel('latitude')
    axes[1, 0].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[1, 0].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[1, 0].set_title('Difference Between log Copernicus Product\nand log U-Net Prediction (log Copernicus - log U-Net)', size=13)

    im3 = axes[1, 1].imshow(diff, vmin=vmin2, vmax=vmax2, extent=extent, origin='upper', transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu)
    axes[1, 1].set_xlabel('longitude')
    axes[1, 1].set_ylabel('latitude')
    axes[1, 1].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[1, 1].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[1, 1].set_title('Absolute Difference Between Copernicus Product\nand U-Net Predictions (Copernicus - U-Net)', size=13)

    fig.subplots_adjust(right=0.85)
    cbar1_ax = fig.add_axes([0.87, 0.14, 0.025, 0.72])
    cbar1 = fig.colorbar(im0, cax=cbar1_ax)
    cbar1.ax.set_ylabel('log Chl-a (mg/m-3)', rotation=270, size=14, labelpad=16)

    cbar2_ax = fig.add_axes([0.94, 0.14, 0.025, 0.72])
    cbar2 = fig.colorbar(im2, cax=cbar2_ax)
    cbar2.ax.set_ylabel('difference in Chl-a in log or absolute scales', rotation=270, size=14, labelpad=16)

    plt.subplots_adjust(top=0.96)
    plt.show()
