"""Visualization helpers for gap-filling model bundles."""

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr


def _axis_bounds(coord):
    coord = np.asarray(coord, dtype=float)
    if coord.ndim != 1 or not coord.size:
        raise ValueError("map coordinates must be non-empty and one-dimensional")
    if coord.size == 1:
        return coord[0] - 0.5, coord[0] + 0.5
    start = coord[0] - (coord[1] - coord[0]) / 2
    end = coord[-1] + (coord[-1] - coord[-2]) / 2
    return min(start, end), max(start, end)


def _map_extent(data):
    """Return longitude/latitude pixel-edge bounds for an xarray object."""
    lon_min, lon_max = _axis_bounds(data["lon"].values)
    lat_min, lat_max = _axis_bounds(data["lat"].values)
    return [lon_min, lon_max, lat_min, lat_max]


def _map_ticks(extent, step=5):
    def aligned(start, end):
        first = np.ceil(start / step) * step
        last = np.floor(end / step) * step
        return np.arange(first, last + step, step)

    return aligned(*extent[:2]), aligned(*extent[2:])


def _frame(dataset, variable, date):
    if variable not in dataset:
        raise KeyError(f"Dataset does not contain '{variable}'")
    data = dataset[variable]
    if "time" in data.dims:
        data = data.sel(time=date)
    if data.dims != ("lat", "lon"):
        data = data.transpose("lat", "lon")
    return data


def _standardization(metadata, variable):
    values = metadata.get("preprocessing", {}).get(
        "standardization", {}
    ).get(variable, {})
    return float(values.get("mean", 0.0)), float(values.get("std", 1.0))


def observed_frame(
    dataset,
    metadata,
    date,
    target="full_target",
):
    """Return one observed target frame in the model's output units."""
    observed = _frame(dataset, target, date)
    mean, std = _standardization(metadata, target)
    return observed * std + mean


def predict_frame(
    dataset,
    model,
    metadata,
    date,
    target="full_target",
):
    """Predict one frame using the bundle's recorded input channel order."""
    inputs = metadata.get("inputs")
    if not inputs:
        raise ValueError("Bundle metadata does not define input channels")
    names = [item["name"] for item in inputs]
    frame = dataset.sel(time=date)
    missing = [name for name in names if name not in frame]
    if missing:
        raise KeyError(f"Dataset is missing model inputs: {missing}")

    values = np.stack(
        [
            np.nan_to_num(frame[name].values, nan=0.0)
            for name in names
        ],
        axis=-1,
    ).astype("float32")
    prediction = np.asarray(
        model.predict(values[np.newaxis, ...], verbose=0)
    )
    if prediction.ndim != 4 or prediction.shape[0] != 1:
        raise ValueError(
            "Model prediction must have shape (1, lat, lon, channels)"
        )
    mean, std = _standardization(metadata, target)
    return xr.DataArray(
        prediction[0, ..., 0] * std + mean,
        dims=("lat", "lon"),
        coords={"lat": frame["lat"], "lon": frame["lon"]},
        name="prediction",
    )


def flag_frame(
    dataset,
    date,
    land_flag="land_flag",
    missing_flag="unavailable_flag",
    estimate_flag="estimate_flag",
):
    """Build categorical map values for land, held-out, observed, and missing."""
    land = _frame(dataset, land_flag, date).values == 1
    missing = _frame(dataset, missing_flag, date).values == 1
    estimate = _frame(dataset, estimate_flag, date).values == 1

    flags = np.full(land.shape, 2, dtype="int8")
    flags[estimate] = 1
    flags[missing] = 3
    flags[land] = 0
    return xr.DataArray(
        flags,
        dims=("lat", "lon"),
        coords={
            "lat": dataset["lat"],
            "lon": dataset["lon"],
        },
        name="flags",
    )


def _new_map_axis(figsize=(7, 5)):
    figure, axis = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    return figure, axis


LAND_COLOR = "gray"


def _land_mask(dataset, date, land_flag="land_flag"):
    """Return a boolean land mask (lat, lon) for one date, or None.

    Returns ``None`` when the variable is absent or no pixels are land, so no
    empty land overlay is drawn.
    """
    if dataset is None or land_flag not in dataset:
        return None
    mask = _frame(dataset, land_flag, date).values == 1
    if not mask.any():
        return None
    return mask


def _plot_map_panel(
    data,
    *,
    ax=None,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    colorbar=True,
    colorbar_label=None,
    colorbar_ticks=None,
    colorbar_ticklabels=None,
    land_mask=None,
):
    own_axis = ax is None
    if own_axis:
        _, ax = _new_map_axis()
    extent = _map_extent(data)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()
    else:
        cmap = cmap.copy()

    image = ax.imshow(
        np.asarray(data.values, dtype=float),
        extent=extent,
        origin="upper",
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if land_mask is not None:
        # Paint land gray as a separate overlay so genuine gaps/clouds (NaN in
        # the data) stay the axes background (white) rather than being colored.
        land_overlay = np.where(land_mask, 1.0, np.nan)
        ax.imshow(
            land_overlay,
            extent=extent,
            origin="upper",
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
            cmap=ListedColormap([LAND_COLOR]),
            vmin=0,
            vmax=1,
        )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_box_aspect(
        (extent[3] - extent[2]) / (extent[1] - extent[0])
    )
    lon_ticks, lat_ticks = _map_ticks(extent)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    if colorbar:
        colorbar_object = ax.figure.colorbar(
            image,
            ax=ax,
            label=colorbar_label,
            fraction=0.046,
            pad=0.04,
            ticks=colorbar_ticks,
        )
        if colorbar_ticklabels is not None:
            colorbar_object.ax.set_yticklabels(colorbar_ticklabels)
    if own_axis:
        plt.show()
    return image


def plot_observed(
    dataset,
    metadata,
    date,
    *,
    target="full_target",
    observed=None,
    ax=None,
    vmin=None,
    vmax=None,
    colorbar=True,
):
    """Plot the observed target for one date (land shown in gray)."""
    observed = (
        observed
        if observed is not None
        else observed_frame(dataset, metadata, date, target)
    )
    return _plot_map_panel(
        observed,
        ax=ax,
        title=f"Observed: {date}",
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        colorbar_label="target",
        land_mask=_land_mask(dataset, date),
    )


def plot_prediction(
    dataset,
    model,
    metadata,
    date,
    *,
    target="full_target",
    prediction=None,
    ax=None,
    vmin=None,
    vmax=None,
    colorbar=True,
):
    """Predict and plot a gap-filled target for one date (land shown in gray)."""
    prediction = (
        prediction
        if prediction is not None
        else predict_frame(dataset, model, metadata, date, target)
    )
    return _plot_map_panel(
        prediction,
        ax=ax,
        title=f"U-Net prediction: {date}",
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        colorbar_label="target",
        land_mask=_land_mask(dataset, date),
    )


def plot_flags(
    dataset,
    date,
    *,
    flags=None,
    ax=None,
    colorbar=True,
):
    """Plot land, synthetic gaps, observed water, and real missing data."""
    flags = flags if flags is not None else flag_frame(dataset, date)
    return _plot_map_panel(
        flags,
        ax=ax,
        title=f"Data flags: {date}",
        cmap=ListedColormap([LAND_COLOR, "teal", "yellow", "darkblue"]),
        vmin=-0.5,
        vmax=3.5,
        colorbar=colorbar,
        colorbar_ticks=[0, 1, 2, 3],
        colorbar_ticklabels=["land", "held out", "observed", "missing"],
    )


def plot_difference(
    dataset,
    model,
    metadata,
    date,
    *,
    target="full_target",
    observed=None,
    prediction=None,
    ax=None,
    limit=1.0,
    colorbar=True,
):
    """Plot observed minus predicted target values for one date."""
    observed = (
        observed
        if observed is not None
        else observed_frame(dataset, metadata, date, target)
    )
    prediction = (
        prediction
        if prediction is not None
        else predict_frame(dataset, model, metadata, date, target)
    )
    difference = observed - prediction
    return _plot_map_panel(
        difference,
        ax=ax,
        title=f"Observed - prediction: {date}",
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        colorbar=colorbar,
        colorbar_label="difference",
        land_mask=_land_mask(dataset, date),
    )


def plot_prediction_observed(
    dataset,
    model,
    metadata,
    date,
    *,
    target="full_target",
    difference_limit=1.0,
):
    """Compose observed, flags, prediction, and difference panels."""
    observed = observed_frame(dataset, metadata, date, target)
    prediction = predict_frame(dataset, model, metadata, date, target)
    flags = flag_frame(dataset, date)
    finite = np.concatenate(
        [
            observed.values[np.isfinite(observed.values)],
            prediction.values[np.isfinite(prediction.values)],
        ]
    )
    vmin = float(finite.min()) if finite.size else None
    vmax = float(finite.max()) if finite.size else None

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    observed_image = plot_observed(
        dataset,
        metadata,
        date,
        target=target,
        observed=observed,
        ax=axes[0, 0],
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
    )
    plot_flags(
        dataset,
        date,
        flags=flags,
        ax=axes[0, 1],
        colorbar=True,
    )
    plot_prediction(
        dataset,
        model,
        metadata,
        date,
        target=target,
        prediction=prediction,
        ax=axes[1, 0],
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
    )
    difference_image = plot_difference(
        dataset,
        model,
        metadata,
        date,
        target=target,
        observed=observed,
        prediction=prediction,
        ax=axes[1, 1],
        limit=difference_limit,
        colorbar=False,
    )
    figure.colorbar(
        observed_image,
        ax=[axes[0, 0], axes[1, 0]],
        label="target",
        fraction=0.046,
        pad=0.04,
    )
    figure.colorbar(
        difference_image,
        ax=axes[1, 1],
        label="observed - prediction",
        fraction=0.046,
        pad=0.04,
    )
    return figure, axes
