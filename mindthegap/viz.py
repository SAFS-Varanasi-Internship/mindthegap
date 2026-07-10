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
        start = coord[0] - (coord[1] - coord[0]) / 2
        end = coord[-1] + (coord[-1] - coord[-2]) / 2
        return (min(start, end), max(start, end))

    lon_min, lon_max = _axis_bounds(lon)
    lat_min, lat_max = _axis_bounds(lat)
    return [lon_min, lon_max, lat_min, lat_max]


def _map_ticks(extent, step=5):
    lon_min, lon_max, lat_min, lat_max = extent

    def _aligned_ticks(start, end):
        tick_start = np.ceil(start / step) * step
        tick_end = np.floor(end / step) * step
        ticks = np.arange(tick_start, tick_end + step, step)
        return ticks[(ticks >= start) & (ticks <= end)]

    lon_ticks = _aligned_ticks(lon_min, lon_max)
    lat_ticks = _aligned_ticks(lat_min, lat_max)
    return lon_ticks, lat_ticks


def plot_prediction_observed(
    zarr_stdized,
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
      4) runs the model to produce standardized predictions, then unstandardizes back to log-scale
         using ``utils.unstdize``,
      5) masks predictions where the observation is NaN, and
      6) produces a 2×2 panel plot: observed, flags, predicted, and (observed − predicted).

    Parameters
    ----------
    zarr_stdized : xarray.Dataset
        Dataset containing standardized predictors (and flags) for the model input. Must include
        the variables used by the model as well as ``'CHL'`` (which is removed from X).
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
    - Predictions are unstandardized with :func:`utils.unstdize`.
    - The map extent and tick marks are derived from the ``zarr_stdized`` coordinates so the
      raster fills the panel without distortion.

    See Also
    --------
    utils.unstdize : Convert standardized values back to original (log-scale) units.
    utils.compute_mae : Mean absolute error (NaN-aware).
    utils.compute_mse : Mean squared error (NaN-aware).

    Example
    -------
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
    true_CHL = unstdize(zarr_stdized.sel(time=date_to_predict)['CHL'], mean, std).to_numpy()

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
    flag = np.where(zarr_date['real_cloud_flag'] == 1, 3, flag)
    flag = np.where(zarr_date['fake_cloud_flag'] == 1, 1, flag)

    # Color limits matched between observed and predicted
    vmax = np.nanmax((true_CHL, predicted_CHL))
    vmin = np.nanmin((true_CHL, predicted_CHL))

    extent = _map_extent(zarr_stdized)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Panel 1 true CHL
    im0 = axes[0, 0].imshow(true_CHL, vmin=vmin, vmax=vmax, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(), interpolation='nearest')
    axes[0, 0].add_feature(cfeature.COASTLINE)
    axes[0, 0].set_extent(extent, crs=ccrs.PlateCarree())
    axes[0, 0].set_box_aspect((extent[3] - extent[2]) / (extent[1] - extent[0]))
    axes[0, 0].set_xlabel('longitude'); axes[0, 0].set_ylabel('latitude')
    lon_ticks, lat_ticks = _map_ticks(extent)
    axes[0, 0].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    axes[0, 0].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    axes[0, 0].set_title('Observed Level-3 log Chl-a', size=14)

    # Panel 2 flags
    from matplotlib.colors import ListedColormap
    im1 = axes[0, 1].imshow(flag, extent=extent, origin="upper",
                            transform=ccrs.PlateCarree(),
                            cmap=ListedColormap(["white", "teal", "yellow", "darkblue"]),
                            vmin=0, vmax=3, interpolation="nearest",)
    axes[0, 1].add_feature(cfeature.COASTLINE)
    axes[0, 1].set_extent(extent, crs=ccrs.PlateCarree())
    axes[0, 1].set_box_aspect((extent[3] - extent[2]) / (extent[1] - extent[0]))
    axes[0, 1].set_xlabel('longitude'); axes[0, 1].set_ylabel('latitude')
    axes[0, 1].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    axes[0, 1].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    axes[0, 1].set_title('Land (0), Cloud (3), Observed (2), Masked (1)', size=13)

    im2 = axes[1, 0].imshow(predicted_CHL, vmin=vmin, vmax=vmax, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(), interpolation='nearest')
    axes[1, 0].add_feature(cfeature.COASTLINE)
    axes[1, 0].set_extent(extent, crs=ccrs.PlateCarree())
    axes[1, 0].set_box_aspect((extent[3] - extent[2]) / (extent[1] - extent[0]))
    axes[1, 0].imshow(np.where(flag == 1, np.nan, flag), vmax=2, vmin=0,
                      extent=extent, origin='upper', interpolation='nearest', alpha=1)
    axes[1, 0].set_xlabel('longitude'); axes[1, 0].set_ylabel('latitude')
    axes[1, 0].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    axes[1, 0].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    axes[1, 0].set_title('Predicted log Chl-a from U-Net', size=14)

    vmin2, vmax2 = -1, 1
    im3 = axes[1, 1].imshow(diff, vmin=vmin2, vmax=vmax2, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(),
                            cmap=plt.cm.RdBu, interpolation='nearest')
    axes[1, 1].add_feature(cfeature.COASTLINE)
    axes[1, 1].set_extent(extent, crs=ccrs.PlateCarree())
    axes[1, 1].set_box_aspect((extent[3] - extent[2]) / (extent[1] - extent[0]))
    axes[1, 1].set_xlabel('longitude'); axes[1, 1].set_ylabel('latitude')
    axes[1, 1].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    axes[1, 1].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
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
    - The map extent and tick marks are derived from the ``zarr_stdized`` coordinates.

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
    axes[0, 0].set_extent(extent, crs=ccrs.PlateCarree())
    axes[0, 0].set_xlabel('longitude')
    axes[0, 0].set_ylabel('latitude')
    lon_ticks, lat_ticks = _map_ticks(extent)
    axes[0, 0].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    axes[0, 0].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    axes[0, 0].set_title('Log Chl-a from the Gapfree \nLevel-4 GlobColour Copernicus Product', size=14)
    
    im1 = axes[0, 1].imshow(predicted_CHL, extent=extent, origin='upper', transform=ccrs.PlateCarree())
    axes[0, 1].set_extent(extent, crs=ccrs.PlateCarree())
    axes[0, 1].set_xlabel('longitude')
    axes[0, 1].set_ylabel('latitude')
    axes[0, 1].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    axes[0, 1].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    axes[0, 1].set_title('Gapfree log Chl-a from U-Net', size=14)
    
    vmax2 = 1
    vmin2 = -1
    im2 = axes[1, 0].imshow(log_diff, vmin=vmin2, vmax=vmax2, extent=extent, origin='upper', transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu)
    axes[1, 0].set_extent(extent, crs=ccrs.PlateCarree())
    axes[1, 0].set_xlabel('longitude')
    axes[1, 0].set_ylabel('latitude')
    axes[1, 0].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    axes[1, 0].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    axes[1, 0].set_title('Difference Between log Copernicus Product\nand log U-Net Prediction (log Copernicus - log U-Net)', size=13)

    im3 = axes[1, 1].imshow(diff, vmin=vmin2, vmax=vmax2, extent=extent, origin='upper', transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu)
    axes[1, 1].set_extent(extent, crs=ccrs.PlateCarree())
    axes[1, 1].set_xlabel('longitude')
    axes[1, 1].set_ylabel('latitude')
    axes[1, 1].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    axes[1, 1].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
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


def _fake_cloud_mae_per_day(ds_std, model, mean, std, target="CHL",
                            mask_flag="fake_cloud_flag", block=100):
    """Per-time-step gap-fill error: mean ``|prediction - truth|`` at the fake-cloud pixels.

    Predicts the whole field for each time step (the model is fully convolutional) and averages
    the absolute error over that step's masked pixels. Loads ``ds_std`` in blocks of ``block`` time
    steps so a multi-year lazy dataset does not have to fit in memory. Makes no assumption about the
    time layout.

    Returns
    -------
    (times, mae) : (numpy.ndarray[datetime64], numpy.ndarray[float])
        One value per time step; NaN for steps with no masked pixels.
    """
    xvars = [v for v in ds_std.data_vars if v != target]
    times = np.asarray(ds_std.time.values)
    out = np.full(len(times), np.nan)
    for i in range(0, len(times), block):
        sub = ds_std.isel(time=slice(i, i + block))
        sub = sub.load() if hasattr(sub, "load") else sub
        X = np.stack([np.nan_to_num(sub[v].values, nan=0.0) for v in xvars], -1).astype("float32")
        pred = model.predict(X, batch_size=4, verbose=0)[..., 0] * std + mean
        truth = sub[target].values * std + mean
        m = (sub[mask_flag].values == 1) & np.isfinite(truth) & np.isfinite(pred)
        diff = np.where(m, np.abs(pred - truth), np.nan)
        with np.errstate(invalid="ignore"):
            out[i:i + diff.shape[0]] = np.nanmean(diff.reshape(diff.shape[0], -1), axis=1)
    return times, out


def overall_perf(ds_std, model, mean, std, train_times=None, freq="D", kind="line",
                 target="CHL", mask_flag="fake_cloud_flag", block=100, ax=None, label=None):
    """Time series of gap-fill error (fake-cloud MAE) for one model over ``ds_std``.

    For each time step it predicts the whole field, takes ``|prediction - truth|`` at the
    fake-cloud pixels, and averages. Optionally aggregates to month/year and marks the training
    days in red.

    Deliberately assumption-free about the data:
      * ``train_times`` is any collection of timestamps, matched by day membership, so the training
        days may be non-contiguous / not a single block (or omitted entirely, if unknown).
      * ``kind="scatter"`` draws dots instead of a connected line, for a sparse/random set of days
        where a line would imply continuity that is not there.

    Parameters
    ----------
    ds_std : xr.Dataset
        Standardized model inputs, including ``target`` and ``mask_flag``. May be lazy.
    model : keras model
        Fully-convolutional model returning one standardized channel.
    mean, std : float
        Standardization stats for ``target``; the error is reported in the original (log) units.
    train_times : sequence of datetime-like, or None
        Training days, marked red. None marks nothing (no assumption about the split).
    freq : {"D", "M", "Y"}
        "D" keeps per-day; "M"/"Y" average within each month/year.
    kind : {"line", "scatter"}
        Line for regularly-spaced days; scatter for a sparse/random subset.
    target, mask_flag : str
        Names of the truth variable and the fake-cloud flag in ``ds_std``.
    ax : matplotlib axis or None
        Draw onto an existing axis (to overlay several models); None makes its own figure.
    label : str or None
        Legend label for this series.

    Returns
    -------
    pandas.Series
        The (aggregated) per-time MAE, indexed by time. ``series.mean()`` is the overall number.
    """
    import pandas as pd
    times, mae = _fake_cloud_mae_per_day(ds_std, model, mean, std, target, mask_flag, block)
    s = pd.Series(mae, index=pd.to_datetime(times)).dropna()

    train_norm = set()
    if train_times is not None:
        train_norm = set(pd.to_datetime(list(train_times)).normalize())
    is_train = pd.Series(s.index.normalize().isin(train_norm), index=s.index)

    if freq and str(freq).upper() in ("M", "Y"):
        per = s.index.to_period(str(freq).upper())
        s = s.groupby(per).mean()
        is_train = is_train.groupby(per).mean() > 0.5
        s.index = s.index.to_timestamp()
        is_train.index = s.index

    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(11, 4))
    if kind == "scatter":
        ax.scatter(s.index, s.values, s=18,
                   c=np.where(is_train.values, "red", "black"), zorder=2, label=label)
    else:
        ax.plot(s.index, s.values, color="black", lw=1.3, zorder=1, label=label)
        if is_train.any():
            ax.plot(s.index[is_train.values], s.values[is_train.values], "o",
                    color="red", ms=4, zorder=2)
    ax.set_xlabel("time")
    ax.set_ylabel("fake-cloud MAE")
    if own:
        ax.grid(alpha=0.3)
        if label is not None:
            ax.legend()
        plt.show()

    overall = float(s.mean())
    if train_times is not None and is_train.any() and (~is_train).any():
        print(f"{label or 'model'}: overall {overall:.4f}   train {s[is_train.values].mean():.4f}"
              f"   other {s[~is_train.values].mean():.4f}")
    else:
        print(f"{label or 'model'}: overall MAE {overall:.4f}")
    return s


def spatial_perf(ds_std, model, mean, std, groupby=None, target="CHL",
                 mask_flag="fake_cloud_flag", block=100, plot=True, vmax=None, cmap="viridis"):
    """Per-pixel gap-fill error map: mean ``|prediction - truth|`` at fake-cloud pixels over time.

    Shows *where* the model reconstructs cloud gaps well or badly. ``groupby="month"`` returns a
    seasonal stack (one map per calendar month) instead of a single all-time map.

    Parameters
    ----------
    groupby : None or "month"
        None -> one (lat, lon) map. "month" -> a (month, lat, lon) stack.
    ds_std, model, mean, std, target, mask_flag, block :
        As in :func:`overall_perf`.
    plot : bool
        Draw the map(s) as well as returning them.
    vmax : float or None
        Upper color limit, shared across all panels. None uses the 99th percentile of the data so a
        few very-high-error pixels do not wash out the rest (lower it for more contrast). vmin is 0.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    xr.DataArray
        (lat, lon) if ``groupby`` is None, else (month, lat, lon), of mean absolute error.
    """
    import pandas as pd
    xvars = [v for v in ds_std.data_vars if v != target]
    lat, lon = ds_std["lat"], ds_std["lon"]
    H, W = ds_std.sizes["lat"], ds_std.sizes["lon"]
    months = pd.to_datetime(ds_std.time.values).month.to_numpy()
    G = 12 if groupby == "month" else 1

    ssum = np.zeros((G, H, W)); cnt = np.zeros((G, H, W))
    for i in range(0, ds_std.sizes["time"], block):
        sub = ds_std.isel(time=slice(i, i + block))
        sub = sub.load() if hasattr(sub, "load") else sub
        X = np.stack([np.nan_to_num(sub[v].values, nan=0.0) for v in xvars], -1).astype("float32")
        pred = model.predict(X, batch_size=4, verbose=0)[..., 0] * std + mean
        truth = sub[target].values * std + mean
        m = (sub[mask_flag].values == 1) & np.isfinite(truth) & np.isfinite(pred)
        ad = np.where(m, np.abs(pred - truth), 0.0)
        if groupby == "month":
            mo = months[i:i + pred.shape[0]]
            for k in range(pred.shape[0]):
                g = int(mo[k]) - 1
                ssum[g] += ad[k]; cnt[g] += m[k]
        else:
            ssum[0] += ad.sum(0); cnt[0] += m.sum(0)

    with np.errstate(invalid="ignore"):
        mae = ssum / np.where(cnt == 0, np.nan, cnt)

    if groupby == "month":
        da = xr.DataArray(mae, dims=("month", "lat", "lon"),
                          coords={"month": np.arange(1, 13), "lat": lat, "lon": lon})
    else:
        da = xr.DataArray(mae[0], dims=("lat", "lon"), coords={"lat": lat, "lon": lon})

    if plot:
        extent = _map_extent(ds_std)
        hi = vmax
        if hi is None:                                  # robust default: cap at the 99th percentile
            finite = da.values[np.isfinite(da.values)]
            hi = float(np.percentile(finite, 99)) if finite.size else None
        if groupby == "month":
            fig, axes = plt.subplots(3, 4, figsize=(14, 8),
                                     subplot_kw={"projection": ccrs.PlateCarree()})
            im = None
            for mo, ax in zip(range(1, 13), axes.ravel()):
                im = ax.imshow(da.sel(month=mo).values, vmin=0, vmax=hi, cmap=cmap,
                               extent=extent, origin="upper",
                               transform=ccrs.PlateCarree(), interpolation="nearest")
                ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
                ax.set_title(f"month {mo}", size=9)
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, label="MAE")
        else:
            fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": ccrs.PlateCarree()})
            im = ax.imshow(da.values, vmin=0, vmax=hi, cmap=cmap, extent=extent, origin="upper",
                           transform=ccrs.PlateCarree(), interpolation="nearest")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
            fig.colorbar(im, ax=ax, label="MAE")
        plt.show()
    return da
