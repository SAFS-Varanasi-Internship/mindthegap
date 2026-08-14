"""Dataset loading, selection, feature engineering, and preparation."""

import warnings
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


def _demo_data_call(
    *,
    dataset,
    region,
    time_slice,
    smoke_test,
    smoke_days,
    smoke_size,
):
    """Reconstruct the ``demo_data(...)`` call that produced a dataset.

    Returns a string such as ``demo_data(dataset='pace')`` that records exactly
    how the demo dataset was loaded, so it can be stored in the
    options/model-bundle metadata and printed by
    :func:`mindthegap.load_model_bundle`.
    """

    def _fmt(value):
        if isinstance(value, slice):
            parts = ", ".join(repr(p) for p in (value.start, value.stop))
            return f"slice({parts})"
        return repr(value)

    parts = [f"dataset={dataset!r}"]
    if region is not None:
        parts.append(f"region={_fmt(region)}")
    if time_slice is not None:
        parts.append(f"time_slice={_fmt(time_slice)}")
    if smoke_test:
        parts.append("smoke_test=True")
        if smoke_days != 120:
            parts.append(f"smoke_days={smoke_days}")
        if smoke_size != 128:
            parts.append(f"smoke_size={smoke_size}")
    return "demo_data(" + ", ".join(parts) + ")"


def demo_data(
    dataset,
    region=None,
    time_slice=None,
    *,
    smoke_test=False,
    smoke_days=120,
    smoke_size=128,
    verbose=True,
):
    """Load a real ocean-color dataset and return it with its variable names.

    ``dataset`` may be ``"pace"``, ``"globcolour"``, ``"indian-ocean"``, or
    ``"io-shared-public"``. ``region`` may be a supported name or
    ``[lat_min, lat_max, lon_min, lon_max]``. ``None`` selects the full spatial
    extent.

    This loader does one thing: it loads a dataset and returns it. It does
    **not** touch ``options`` -- configuring ``options`` is the job of
    :meth:`Options.set_up_data_options`. The dataset name and source URL/path
    are stored on ``ds.attrs["dataset_name"]`` / ``ds.attrs["dataset_source"]``
    (when not already present), and the exact loading call is recorded on
    ``ds.attrs["data_source"]`` so it can be replayed and saved with a bundle.

    Returns ``(ds, target, missing_flag, land_flag)``: the loaded dataset and
    the names of its target, missing-data (cloud) flag, and land flag variables,
    which the caller passes to :meth:`Options.set_up_data_options`.

    When ``smoke_test`` is true, the loaded dataset is subset to at most
    ``smoke_days`` time steps and ``smoke_size`` cells in each spatial
    dimension so remote validation runs stay small.
    """
    loaders = {
        "pace": _load_pace,
        "globcolour": _load_globcolour,
        "indian-ocean": _load_indian_ocean,
        "io-shared-public": _load_io_shared_public,
    }
    if dataset not in loaders:
        choices = ", ".join(loaders)
        raise ValueError(f"Unknown dataset {dataset!r}; choose from: {choices}")

    data_source = _demo_data_call(
        dataset=dataset,
        region=region,
        time_slice=time_slice,
        smoke_test=smoke_test,
        smoke_days=smoke_days,
        smoke_size=smoke_size,
    )

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

    if "dataset_name" not in ds.attrs:
        ds.attrs["dataset_name"] = config["name"]
    if "dataset_source" not in ds.attrs:
        ds.attrs["dataset_source"] = config["dataset_source"]
    if "product_id" not in ds.attrs:
        ds.attrs["product_id"] = config["product_id"]
    if region_name is not None and "region_name" not in ds.attrs:
        ds.attrs["region_name"] = region_name
    ds.attrs["data_source"] = data_source

    if verbose:
        print(f"Dataset: {ds.attrs['dataset_name']}")
        print(f"  target:       {config['target']}")
        print(f"  missing_flag: {config['missing_flag']}")
        print(f"  land_flag:    {config['land_flag']}")
        print(f"  dimensions:   {dict(ds.sizes)}")

    return ds, config["target"], config["missing_flag"], config["land_flag"]


def _dataset_config(dataset):
    return {
        "pace": {
            "name": "PACE",
            "product_id": "PACE_OCI_L3M_CHL",
            "dataset_source": (
                "https://data.source.coop/fish-pace/pace-oci/inregion/"
                "PACE_OCI_L3M_CHL"
            ),
            "target": "chlor_a",
            "missing_flag": "cloud_flag",
            "land_flag": "land_flag",
        },
        "globcolour": {
            "name": "GlobColour",
            "product_id": (
                "cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D"
            ),
            "dataset_source": (
                "https://data.source.coop/fish-pace/globcolour/"
                "cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D"
            ),
            "target": "CHL",
            "missing_flag": "cloud_flag",
            "land_flag": "land_flag",
        },
        "indian-ocean": {
            "name": "Indian Ocean",
            "product_id": "mind_the_chl_gap/IO_rechunked.zarr",
            "dataset_source": (
                "gcs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO_rechunked.zarr"
            ),
            "target": "CHL_cmes-level3",
            "missing_flag": "CHL_cmes-cloud",
            "land_flag": "CHL_cmes-land",
        },
        "io-shared-public": {
            "name": "IO rechunkded in shared-public",
            "product_id": "shared-public/IO_rechunked.zarr",
            "dataset_source": (
                "/home/jovyan/shared-public/mindthegap/data/"
                "IO_rechunked.zarr"
            ),
            "target": "CHL_cmes-level3",
            "missing_flag": "CHL_cmes-cloud",
            "land_flag": "CHL_cmes-land",
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
        chunks={"time": 100},
    )
    return ds

def _load_io_shared_public():
    ds = xr.open_dataset(
        "/home/jovyan/shared-public/mindthegap/data/"
        "IO_rechunked.zarr",
        engine="zarr",
        consolidated=True,
        chunks={"time": 100},
    )
    # Keep the land mask lazy and chunked like the rest of the store so the
    # dataset has consistent chunks (a static 2D mask broadcast to the full
    # cube must match the time chunking of the other variables).
    land_flag_2d = ds.sst.isel(time=0).isnull()
    land_flag = (
        land_flag_2d.broadcast_like(ds.sst)
        .astype("int8")
        .chunk(ds.sst.chunksizes)
    )
    ds["CHL_cmes-land"] = land_flag
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


def synthetic_cloud_cube(
    n_time,
    n_lat,
    n_lon,
    coverage,
    blob_sigma=6.0,
    time_sigma=0.0,
    rng=None,
):
    """Return a ``(time, lat, lon)`` boolean synthetic-cloud cube.

    Smooth gaussian noise is thresholded so that roughly ``coverage`` of the
    pixels are flagged as cloud. This produces spatially-blobby clouds (rather
    than salt-and-pepper noise) that are optionally temporally autocorrelated.

    Parameters
    ----------
    n_time, n_lat, n_lon : int
        Cube dimensions.
    coverage : float
        Target fraction of pixels covered by synthetic cloud (0-1).
    blob_sigma : float
        Gaussian smoothing width in the spatial dimensions; larger values make
        larger clouds.
    time_sigma : float
        Gaussian smoothing width along the time axis. ``0`` gives independent
        clouds per time step; larger values make clouds that persist and evolve
        across days (with cores that persist and edges that flicker). Temporal
        correlation keeps the previous/next-day lag channels honest so the model
        cannot trivially copy the hidden pixel from a neighbouring day.
    rng : numpy.random.Generator, optional
        Random generator. A fresh default generator is used when ``None``.

    Notes
    -----
    This generator is adapted from Troy's self-supervised cloud experiments; it
    is intended for hiding observed ocean pixels during self-supervised
    training so the model has a known target at the hidden pixels.
    """
    from scipy.ndimage import gaussian_filter

    if not 0 <= coverage <= 1:
        raise ValueError("coverage must be between 0 and 1")
    if rng is None:
        rng = np.random.default_rng()
    field = gaussian_filter(
        rng.standard_normal((n_time, n_lat, n_lon)),
        sigma=(time_sigma, blob_sigma, blob_sigma),
    )
    if coverage <= 0:
        return np.zeros((n_time, n_lat, n_lon), dtype=bool)
    return field > np.quantile(field, 1.0 - coverage)


def _bank_to_cube(
    bank,
    n_time,
    n_lat,
    n_lon,
    time_chunk=100,
    rng=None,
):
    """Map a precomputed cloud ``bank`` onto a dataset grid, lazily.

    ``bank`` is a boolean ``(bank_time, bank_lat, bank_lon)`` array (typically
    the dask-backed :func:`mindthegap.cloud_bank.open_bank` result). The dataset
    grid ``(n_time, n_lat, n_lon)`` is usually much larger than the bank -- e.g.
    a multi-year global 4km field against a 2-year 640x640 bank -- so the bank is
    tiled and wrapped to cover it:

    - Time: dataset day ``t`` reads bank day ``(t0 + t) % bank_time`` for a random
      start offset ``t0``. Because the bank itself is one continuous
      temporally-correlated sequence, consecutive dataset days map to consecutive
      bank days, so a day and its temporal-lag neighbours stay correlated (the
      only seam is the single 2-year wrap boundary).
    - Space: latitude/longitude are wrapped modulo the bank size with random
      offsets, and each *tile* of the output is optionally flipped on either
      spatial axis. This covers an arbitrarily large field from a small bank
      while avoiding an obvious repeating pattern.

    The result is a lazy dask boolean cube chunked to
    ``(time_chunk, n_lat, n_lon)`` (spatial tiling is handled by the pipeline's
    own chunking downstream). All random choices come from ``rng`` so the cube is
    reproducible from the seed alone.
    """
    import dask.array as da

    if rng is None:
        rng = np.random.default_rng()

    bank_time, bank_lat, bank_lon = bank.shape

    # Underlying bool ndarray for point indexing inside the block function. The
    # bank is modest (a couple hundred MB unpacked) and shared across blocks.
    bank_values = np.asarray(bank.data if hasattr(bank, "data") else bank)
    if hasattr(bank_values, "compute"):
        bank_values = bank_values.compute()

    time_chunk = int(max(1, min(time_chunk, n_time)))

    t0 = int(rng.integers(0, bank_time))
    lat0 = int(rng.integers(0, bank_lat))
    lon0 = int(rng.integers(0, bank_lon))
    # Per spatial-tile flips keyed by a coarse tile grid so wrapping the small
    # bank across a large field does not produce an obvious repeat.
    flip_period = max(1, bank_lat), max(1, bank_lon)
    flip_lat = rng.integers(0, 2, size=(n_lat // flip_period[0] + 1,)).astype(
        bool
    )
    flip_lon = rng.integers(0, 2, size=(n_lon // flip_period[1] + 1,)).astype(
        bool
    )

    lat_idx = (lat0 + np.arange(n_lat)) % bank_lat
    lon_idx = (lon0 + np.arange(n_lon)) % bank_lon

    def _block(block_info=None):
        (ts, te), (las, lae), (los, loe) = block_info[None]["array-location"]
        t_src = (t0 + np.arange(ts, te)) % bank_time
        la_src = lat_idx[las:lae]
        lo_src = lon_idx[los:loe]
        out = bank_values[np.ix_(t_src, la_src, lo_src)].copy()
        # Apply per-tile flips based on which bank tile each row/col falls in.
        la_tile = (lat0 + np.arange(las, lae)) // flip_period[0]
        lo_tile = (lon0 + np.arange(los, loe)) // flip_period[1]
        for i, tile in enumerate(np.unique(la_tile)):
            if flip_lat[tile % len(flip_lat)]:
                rows = np.where(la_tile == tile)[0]
                out[:, rows, :] = out[:, rows[::-1], :]
        for tile in np.unique(lo_tile):
            if flip_lon[tile % len(flip_lon)]:
                cols = np.where(lo_tile == tile)[0]
                out[:, :, cols] = out[:, :, cols[::-1]]
        return out

    return da.map_blocks(
        _block,
        dtype=bool,
        chunks=(
            tuple(
                min(time_chunk, n_time - s)
                for s in range(0, n_time, time_chunk)
            ),
            (n_lat,),
            (n_lon,),
        ),
    )


def _resolve_chunks(ds, gridder):
    """Return dask chunks for the standardized dataset built in this pipeline.

    The chunks are chosen so that (a) the lazy graph stays small regardless of
    the incoming store's chunking (which we do not control) and (b) the chunks
    align with how ``ds_std`` is later consumed.

    Time is chunked into blocks of ``gridder.time_chunk`` (default 100) so
    per-frame ``sel``/``isel`` in gap-fill/visualization and the temporal patch
    length used by xbatcher both read a bounded run of time steps.

    For the spatial dims:

    - When the field is small enough to fit as a single spatial block (the
      common case, and whenever ``gridder.tile_size`` is ``"full"``), the whole
      ``lat``/``lon`` extent is one chunk. This gives the smallest possible
      graph and lets each xbatcher tile / frame read a single chunk.
    - When the field is large and xbatcher must tile it spatially, the chunks
      are aligned with ``gridder.tile_size`` so each spatial patch maps onto one
      (or a few) dask chunks instead of slicing into a single enormous
      full-field chunk. ``tile_size`` is a multiple of the U-Net downsampling
      factor, and the field has already been cropped to a multiple of it, so
      tile-aligned chunks tile the field cleanly.
    """
    default_time = 100
    time_chunk = default_time
    tile_size = None
    if gridder is not None:
        time_chunk = getattr(gridder, "time_chunk", default_time) or default_time
        tile_size = getattr(gridder, "tile_size", None)

    from .options import GridderOptions

    n_time = ds.sizes.get("time", time_chunk)
    time_chunk = int(max(1, min(time_chunk, n_time)))

    n_lat = ds.sizes.get("lat")
    n_lon = ds.sizes.get("lon")

    def _tile_len(requested, size):
        if requested is None or size is None:
            return -1
        try:
            requested = int(requested)
        except (TypeError, ValueError):
            return -1
        if requested <= 0 or requested >= size:
            return -1
        return requested

    if GridderOptions._is_full(tile_size):
        lat_chunk = -1
        lon_chunk = -1
    elif isinstance(tile_size, (tuple, list)) and len(tile_size) == 2:
        lat_chunk = _tile_len(tile_size[0], n_lat)
        lon_chunk = _tile_len(tile_size[1], n_lon)
    else:
        lat_chunk = -1
        lon_chunk = -1

    return {"time": time_chunk, "lat": lat_chunk, "lon": lon_chunk}


def prepare_model_data(ds, options, mode, *, dry_run=False):
    """Prepare model inputs, targets, and standardization statistics.

    ``options`` is the canonical source of every setting -- variable
    names/features, ``log_target``, ``n_temporal_lags``, ``add_geo``, the
    synthetic-cloud configuration, the output chunking (``options.gridder``),
    the train/validation split (``options.split``), and the standardization
    statistics. Nothing is passed as a loose keyword argument. ``options`` must
    be a full :class:`Options` object; ``mode`` is required and must be one of
    ``"train"``, ``"test"``, or ``"gapfill"``.

    In ``mode="train"``, if ``options.split`` has not been resolved yet, the
    train/validation dates are chosen automatically via
    :func:`mindthegap.train_validation_dates` (reading the split method,
    fractions, ``n_days``, and seed from ``options.split``); callers may still
    call it explicitly beforehand for full control.

    The spatial dimensions of ``ds`` are first cropped so their lengths are a
    multiple of the U-Net's downsampling factor (see
    :func:`unet_spatial_multiple`) via :func:`crop_to_multiple`, so callers no
    longer need to crop the field themselves.

    The returned dataset contains the transformed target, the target values
    the model can actually observe (``observed_target``), temporal lags of
    that observed target, seasonal channels, three per-pixel state flags,
    optional feature variables, and optional spherical coordinates. When the
    inputs are dask-backed the result is lazy and chunked from
    ``options.gridder`` so the lazy graph stays small and the chunks align with
    how the standardized dataset is consumed downstream (whole spatial frames
    for a block of time steps, or tile-aligned chunks when the field is large
    enough that xbatcher must tile it spatially).

    Channel semantics
    -----------------
    The flags tell the model how to use the inputs (or, more precisely, how the
    model has learned to use them):

    ``observed_target``
        The target values actually available to the model. Missing predictor
        values are later represented as 0. ``observed_target_m{n}`` /
        ``observed_target_p{n}`` are its temporal lags/leads.
    ``observed_flag``
        1 means there is a value in ``observed_target`` at that pixel and the
        model should use it.
    ``estimate_flag``
        1 means there is no observed value in the inputs at that pixel and the
        model should estimate it.
    ``unavailable_flag``
        1 means there is no information for that pixel, so the estimate is set
        to 0.
    ``land_flag``
        1 means land; the estimate is set to 0 (and masked later).

    During training
    ---------------
    Synthetic clouds are created over the observed data. Where synthetic clouds
    are, ``estimate_flag = 1``; during training there must be a true observed
    value for the pixels where ``estimate_flag = 1`` (that known value supplies
    the learning signal). ``unavailable_flag`` marks the true clouds or NaNs in
    the observed data. ``land_flag`` is included so the model can learn
    coastlines, islands, and similar structure.

    The intended per-pixel states are:

    ==================  ============  ============  ================  =========
    state               observed_flag estimate_flag unavailable_flag  land_flag
    ==================  ============  ============  ================  =========
    observed ocean            1             0              0              0
    synthetic gap             0             1              0              0
    true missing ocean        0             0              1              0
    land                      0             0              0              1
    ==================  ============  ============  ================  =========

    Modes
    -----
    ``mode="train"`` produces the self-supervised training inputs described
    above. Synthetic clouds are punched into the observed data
    (``estimate_flag=1`` there, with the true value retained as the learning
    target) and real clouds become ``unavailable_flag=1``. Standardization
    statistics are computed from the training dates in ``options.split`` and
    recorded on ``options.data`` (bounds, input channel names/order, transforms,
    standardization statistics, and target mean/std) so downstream functions and
    the saved model bundle can reproduce these inputs.

    Standardization is opt-in per group. ``options.data.features`` lists only
    *extra* predictor variables from ``ds`` (never the target or its
    ``observed_target`` / ``full_target`` variants -- that is rejected).
    ``options.data.std_features`` selects which of those features are
    standardized (each with its own statistics computed over the training
    dates). ``options.data.std_target`` selects whether the target is
    standardized: when ``True``, a single ``(mean, std)`` is computed from the
    masked ``observed_target`` over the training dates and applied to
    ``observed_target``, its temporal lags, and ``full_target`` so the
    standardized inputs and the training label share one consistent scale;
    ``target_mean`` / ``target_std`` record it. When ``False`` the target group
    is left in raw (or log) units and ``target_mean`` / ``target_std`` are 0 / 1.
    The returned dataset is
    subset to just the dates needed for model fitting -- the union of the
    training and validation dates in ``options.split`` -- rather than every date
    in ``ds``; :func:`make_generator` later splits it back into train/val via
    ``options.split``. Temporal lags are still computed from the full time
    series before subsetting, so the lag channels for the returned days reflect
    their true neighbours even when those neighbours are dropped. How the
    synthetic clouds are generated is controlled by ``options.data.cloud_mode``
    (with the related ``options.data.cloud_coverage`` / ``cloud_blob_sigma`` /
    ``cloud_time_sigma`` / ``missing_flag_shift`` / ``cloud_seed`` fields) --
    the cloud configuration is canonical on ``options`` and is not a call
    argument:

    - ``cloud_mode="synthetic_bank"`` (default) draws synthetic clouds from a
      precomputed *bank* -- a small, self-describing netCDF cube built offline
      with the same construction as :func:`synthetic_cloud_cube`. The bank is
      resolved from the packaged manifest by matching ``cloud_coverage`` /
      ``cloud_blob_sigma`` / ``cloud_time_sigma``, downloaded on first use and
      cached, then tiled/wrapped over the dataset grid with random day/space
      offsets and per-tile flips (all controlled by ``cloud_seed``). Because the
      bank is one continuous temporally-correlated sequence, a day and its
      ``n_temporal_lags`` neighbours stay correlated. This is memory-bounded and
      independent of record length. If no bank matches the requested parameters
      (or the download fails) it falls back to on-the-fly ``"synthetic"``
      generation with a warning.
    - ``cloud_mode="synthetic"`` generates random, spatially-blobby,
      temporally-correlated clouds *on the fly*. Cloud size is set by
      ``cloud_blob_sigma``, day-to-day persistence by ``cloud_time_sigma``
      (``0`` = independent per composite; larger = clouds that last and evolve),
      and the target covered fraction by ``cloud_coverage``. ``cloud_seed``
      (falling back to the run's global seed) makes the clouds reproducible. The
      full ``(time, lat, lon)`` field is smoothed with
      :func:`synthetic_cloud_cube`, so this is the original, more expensive
      path -- it materialises the whole cube in memory and its cost grows with
      the size of the record. Use ``"synthetic_bank"`` for large, multi-year
      records.
    - ``cloud_mode="shift"`` reuses the historical behaviour: the real cloud
      mask ``missing_flag`` is rolled forward by ``missing_flag_shift`` time
      steps and those pixels are hidden. This borrows a real cloud pattern from
      ``missing_flag_shift`` days ahead.

    Regardless of how the synthetic clouds are generated, each pixel ends up
    with at most one cloud type: synthetic clouds are only applied to real
    ocean observations, so a pixel that is already land or under a real cloud
    (``unavailable_flag=1``) keeps that state and ``estimate_flag`` never
    overlaps ``unavailable_flag`` or ``land_flag``.

    ``mode="test"`` builds the same self-supervised inputs as ``"train"``
    (synthetic clouds punched in, ``estimate``/``observed``/``unavailable``
    flags) but standardizes with the statistics already recorded on
    ``options.data`` (from a prior ``"train"`` run) instead of recomputing them,
    and does not modify ``options``. The whole passed-in ``ds`` is used; the
    split is not consulted. This yields a held-out evaluation set that is
    standardized identically to training.

    ``mode="gapfill"`` produces inference inputs for gap-filling. No synthetic
    clouds are created: ``observed_target`` is the real observed data,
    ``observed_flag`` marks the real observations, real cloud/NaN ocean pixels
    become ``estimate_flag=1`` (the pixels to fill), and ``unavailable_flag`` is
    0 everywhere. Every other channel (temporal lags, seasonal, geo, land) is
    computed identically to training so the inputs match the trained model. The
    recorded standardization statistics on ``options.data`` are reused so the
    inputs match training exactly; the whole passed-in ``ds`` is used and the
    split is not consulted.

    Returns the standardized ``output`` dataset. In ``mode="train"`` the
    standardization statistics (and all other resolved settings) are written to
    ``options.data`` -- ``options.data.standardization``,
    ``options.data.target_mean``, and ``options.data.target_std`` -- rather than
    returned separately, so ``options`` remains the single source of truth.

    When ``dry_run=True`` this is a fast, side-effect-free *probe*: the field
    is cropped and the full lazy channel graph is built so the returned dataset
    has the exact shape, ``input_names``/channel order, and chunking that a real
    run would produce, but the expensive/irreversible steps are skipped --
    standardization statistics are **not** computed, synthetic clouds are **not**
    generated, the train/validation split is **not** resolved, and ``options``
    is **not** mutated. Use it to inspect ``ds_std.sizes`` / ``ds_std.data_vars``
    (for example from :func:`mindthegap.set_up_gridder`) without paying for a
    full preparation. ``mode`` still selects which flag channels are built.
    """
    from .options import Options as _Options

    if mode not in ("train", "test", "gapfill"):
        raise ValueError(
            f"mode must be 'train', 'test', or 'gapfill', got {mode!r}"
        )
    if not isinstance(options, _Options):
        raise TypeError(
            "options must be a full mindthegap.Options object; every setting "
            "(variable names, cloud configuration, chunking, split, and "
            "standardization) is read from it"
        )

    full = options
    if not full.data.is_resolved() and not full.data.target_variable:
        raise ValueError(
            "options.data is not configured; call "
            "options.set_up_data_options(ds, target=..., missing_flag=..., "
            "land_flag=...) first"
        )

    # train_dates are used to compute standardization statistics; the returned
    # dataset is subset to the fitting dates (train + val). test and gapfill
    # reuse the recorded statistics and standardize the whole ds.
    train_dates = None
    fit_dates = None
    if mode == "train" and not dry_run:
        if not full.split.is_resolved():
            # Choose the train/validation dates from options.split (its method,
            # fractions, n_days, seed) when the caller has not already done so.
            train_validation_dates(ds["time"], full)
        split_selection = full.split.train_selection()
        available = pd.DatetimeIndex(
            pd.to_datetime(np.asarray(ds["time"].values))
        ).normalize()
        chosen = pd.DatetimeIndex(split_selection).normalize()
        missing_dates = chosen.difference(available)
        if len(missing_dates) > 0:
            sample = ", ".join(str(d.date()) for d in missing_dates[:3])
            raise ValueError(
                "options.split is inconsistent with ds: "
                f"{len(missing_dates)} training date(s) are not in the "
                f"dataset time coordinate (e.g. {sample}). Re-run "
                "mtg.train_validation_dates on this dataset's ds.time."
            )
        train_dates = split_selection
        # The returned dataset carries both the training and validation dates
        # (everything model fitting needs); make_generator later splits it back
        # into train/val via options.split. Statistics are still computed from
        # train_dates only below.
        val_selection = full.split.val_selection()
        fit_index = pd.to_datetime(
            np.concatenate(
                [
                    np.asarray(pd.to_datetime(split_selection)),
                    np.asarray(pd.to_datetime(val_selection)),
                ]
            )
        ).unique()
        # Preserve chronological order to keep the time coordinate monotonic.
        fit_dates = fit_index.sort_values()

    if mode in ("test", "gapfill") and not dry_run and not full.data.standardization:
        raise ValueError(
            f"mode={mode!r} reuses the standardization statistics recorded "
            "during training, but options.data.standardization is empty. Run "
            "prepare_model_data(ds, options, mode='train') first (or load a "
            "trained bundle into options)."
        )

    gridder = full.gridder
    verbose = full.verbose
    cloud_seed = (
        full.data.cloud_seed
        if full.data.cloud_seed is not None
        else full.resolved_seed()
    )
    data_options = full.data

    # Everything below reads the resolved data configuration from options.data.
    target_variable = data_options.target_variable
    missing_flag = data_options.missing_flag
    land_flag = data_options.land_flag
    features = list(data_options.features or [])
    std_features = list(data_options.std_features or [])
    std_target = bool(data_options.std_target)
    log_target = bool(data_options.log_target)
    n_temporal_lags = data_options.n_temporal_lags
    add_geo = bool(data_options.add_geo)
    cloud_mode = data_options.cloud_mode
    coverage = data_options.cloud_coverage
    blob_sigma = data_options.cloud_blob_sigma
    time_sigma = data_options.cloud_time_sigma
    missing_flag_shift = data_options.missing_flag_shift

    # Synthetic clouds are punched in for both train and test; gapfill has none.
    # A dry run never generates them -- it only needs the channel skeleton, so
    # the flag channels are built with the cheap gapfill logic (constant/derived
    # arrays) instead of the expensive synthetic-cloud path.
    make_synthetic = mode in ("train", "test") and not dry_run
    # test/gapfill reuse the recorded standardization instead of recomputing; a
    # dry run computes no statistics at all.
    reuse_standardization = mode in ("test", "gapfill") and not dry_run
    output_chunks = None

    if target_variable is None or missing_flag is None or land_flag is None:
        raise ValueError(
            "options.data must define target_variable, missing_flag, and "
            "land_flag; call options.set_up_data_options(...) first"
        )
    n_temporal_lags = 1 if n_temporal_lags is None else n_temporal_lags

    features = list(features or [])
    reserved_names = {
        target_variable,
        "full_target",
        "observed_target",
    }
    for lag in range(1, (n_temporal_lags or 0) + 1):
        reserved_names.add(f"observed_target_m{lag}")
        reserved_names.add(f"observed_target_p{lag}")
    illegal_features = [name for name in features if name in reserved_names]
    if illegal_features:
        raise ValueError(
            "options.data.features must contain only extra predictor "
            "variables, never the target or its variants; found "
            f"{illegal_features}. Use options.data.std_target to standardize "
            "the target."
        )
    unknown_std_features = [
        name for name in std_features if name not in features
    ]
    if unknown_std_features:
        raise ValueError(
            "options.data.std_features must be a subset of "
            f"options.data.features; unknown: {unknown_std_features}"
        )
    required = [target_variable, *features, missing_flag, land_flag]
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"Dataset is missing required variables: {missing}")
    for coord in ("time", "lat", "lon"):
        if coord not in ds.coords:
            raise ValueError(f"Required coordinate '{coord}' not found in ds")
    if n_temporal_lags < 0:
        raise ValueError("n_temporal_lags must be non-negative")

    from .model import unet_spatial_multiple

    # Crop the field so the spatial dims are a multiple of the U-Net's
    # downsampling factor; this used to be a separate caller step.
    ds = crop_to_multiple(ds, multiple=unet_spatial_multiple())

    # Pick the dask chunking used throughout the rest of the function so the
    # lazy graph stays small *and* aligns with how ``ds_std`` is consumed.
    #
    # The incoming store's chunks are outside our control and are often the
    # worst case for us -- e.g. one time step per chunk (``(1, lat, lon)``).
    # Building the temporal lags, broadcasts, and where-masks on that finest
    # chunking explodes the lazy Dask graph to millions of tasks for multi-year
    # records, which is slow and memory hungry even though nothing is computed
    # yet.
    #
    # Downstream, ``make_generator`` tiles ``ds_std`` with xbatcher over
    # ``gridder.time_chunk`` x ``gridder.tile_size`` windows and the
    # gap-fill/viz paths pull whole spatial frames for a block of time steps. If
    # the field fits comfortably as a single spatial block (the common case, and
    # ``tile_size="full"``), a "full spatial extent, coarse time" chunking is
    # ideal: one chunk per time block, cheap graph, and every consumer reads a
    # single chunk. For very large fields that xbatcher must tile spatially, we
    # instead align the dask chunks with the tile size so each xbatcher patch
    # maps onto one (or a few) chunks rather than slicing into a huge full-field
    # chunk. This ``output_chunks`` is reused for the final chunking so the
    # dataset is not rechunked twice.
    is_dask = any(
        getattr(ds[name].data, "__dask_graph__", None) is not None
        for name in required
        if name in ds
    )
    if output_chunks is None:
        output_chunks = _resolve_chunks(ds, gridder)
    if is_dask:
        chunk_spec = {
            dim: size
            for dim, size in output_chunks.items()
            if dim in ds.dims
        }
        ds = ds.chunk(chunk_spec)

    processed = ds[required].rename({target_variable: "full_target"})
    for flag in (missing_flag, land_flag):
        processed[flag] = processed[flag].broadcast_like(
            processed["full_target"]
        )
    if log_target:
        processed["full_target"] = np.log(
            processed["full_target"].where(processed["full_target"] > 0)
        )

    processed["land_flag"] = (processed[land_flag] == 1).astype("int8")
    if make_synthetic:
        # Self-supervised training/test: punch synthetic gaps into the observed
        # data. The hidden pixels become estimate_flag=1 (their true value is
        # known and supplies the learning/evaluation signal); real clouds become
        # unavailable_flag=1.
        processed["unavailable_flag"] = (processed[missing_flag] == 1).astype(
            "int8"
        )
        # Pixels that are real ocean observations and therefore eligible to be
        # hidden by a synthetic cloud: there must be a true value to learn from,
        # and the pixel must not already carry another cloud type. Applying this
        # single mask to the synthetic clouds below guarantees each pixel has at
        # most one cloud flag (a real cloud always wins over a synthetic one).
        eligible = (
            (processed[missing_flag] == 0)
            & (processed["land_flag"] == 0)
            & processed["full_target"].notnull()
        )
        bank_provenance = None
        if cloud_mode in ("synthetic_bank", "synthetic"):
            rng = np.random.default_rng(cloud_seed)
            n_t = processed.sizes["time"]
            n_la = processed.sizes["lat"]
            n_lo = processed.sizes["lon"]
            time_chunk = processed["full_target"].chunksizes.get(
                "time", (n_t,)
            )[0]

            cube = None
            if cloud_mode == "synthetic_bank":
                # Draw clouds from a precomputed bank: cheap and memory-bounded
                # regardless of record length. Fall back to on-the-fly
                # generation when no matching bank is published.
                from . import cloud_bank as _cb

                entry = _cb.find_bank_entry(
                    coverage=coverage,
                    blob_sigma=blob_sigma,
                    time_sigma=time_sigma,
                )
                if entry is None:
                    warnings.warn(
                        "cloud_mode='synthetic_bank' but no cloud bank matches "
                        f"coverage={coverage}, blob_sigma={blob_sigma}, "
                        f"time_sigma={time_sigma}; falling back to on-the-fly "
                        "synthetic cloud generation.",
                        RuntimeWarning,
                    )
                else:
                    try:
                        bank_path = _cb.fetch_bank(entry)
                        bank = _cb.open_bank(bank_path, chunk_time=time_chunk)
                        cube = _bank_to_cube(
                            bank,
                            n_t,
                            n_la,
                            n_lo,
                            time_chunk=time_chunk,
                            rng=rng,
                        )
                        bank_provenance = {
                            "cloud_bank_file": entry["filename"],
                            "cloud_bank_sha256": entry.get("sha256"),
                        }
                    except Exception as exc:
                        warnings.warn(
                            "Failed to load cloud bank "
                            f"{entry['filename']!r} ({exc}); falling back to "
                            "on-the-fly synthetic cloud generation.",
                            RuntimeWarning,
                        )
                        cube = None

            if cube is None:
                # On-the-fly generation (the original cloud_mode="synthetic"
                # path): smooth the full field once. This materialises an
                # (n_time, n_lat, n_lon) cube in memory and scales with the
                # length of the record -- use cloud_mode="synthetic_bank" for
                # large, multi-year records.
                cube = synthetic_cloud_cube(
                    n_t,
                    n_la,
                    n_lo,
                    coverage=coverage,
                    blob_sigma=blob_sigma,
                    time_sigma=time_sigma,
                    rng=rng,
                )
            synthetic_cloud = xr.DataArray(
                cube,
                dims=("time", "lat", "lon"),
                coords={
                    "time": processed["time"],
                    "lat": processed["lat"],
                    "lon": processed["lon"],
                },
            )
            estimate = synthetic_cloud
        else:
            # Historical behaviour: borrow the real cloud pattern from
            # ``missing_flag_shift`` time steps ahead and hide those pixels.
            shifted_missing = processed[missing_flag].roll(
                time=-missing_flag_shift,
                roll_coords=False,
            )
            estimate = shifted_missing == 0
        # Each pixel gets at most one cloud type. However the synthetic clouds
        # were created above, restrict them to pixels that are real ocean
        # observations (a known truth to learn from): drop any that fall on
        # land, on a real cloud (unavailable_flag), or on an already-missing
        # value. This guarantees estimate_flag and unavailable_flag never
        # overlap -- a pixel under a real cloud stays real, not synthetic.
        estimate = estimate & eligible
        processed["estimate_flag"] = estimate.astype("int8")
        processed["observed_target"] = processed["full_target"].where(
            ~estimate
        )
        processed["observed_flag"] = processed[
            "observed_target"
        ].notnull().astype("int8")
    else:
        # Gap-filling: no synthetic clouds. The observed target is the real
        # observed data (missing values are already NaN and become 0 later). Real
        # clouds/NaNs over ocean are exactly the pixels the model should fill, so
        # they become estimate_flag=1; nothing is unavailable at inference time.
        processed["observed_target"] = processed["full_target"]
        processed["observed_flag"] = processed[
            "full_target"
        ].notnull().astype("int8")
        processed["estimate_flag"] = (
            (processed["observed_flag"] == 0)
            & (processed["land_flag"] == 0)
        ).astype("int8")
        processed["unavailable_flag"] = xr.zeros_like(
            processed["land_flag"]
        )

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

    target_group = ["full_target", "observed_target"]
    lag_vars = []
    for lag in range(1, n_temporal_lags + 1):
        previous = f"observed_target_m{lag}"
        following = f"observed_target_p{lag}"
        processed[previous] = processed["observed_target"].shift(time=lag)
        processed[following] = processed["observed_target"].shift(time=-lag)
        lag_vars.extend([previous, following])
    target_group.extend(lag_vars)
    standardizable_vars = features + target_group

    means = {name: 0.0 for name in standardizable_vars}
    standard_deviations = {name: 1.0 for name in standardizable_vars}
    # applied_vars tracks which variables actually had standardization applied,
    # so the recorded ``standardization`` metadata (and test/gapfill reuse) is
    # exact. Feature standardization is controlled by options.data.std_features;
    # target standardization by options.data.std_target.
    applied_vars = []
    if reuse_standardization:
        # test/gapfill: reuse the standardization recorded at training time so
        # inputs match the trained model exactly; never recompute from the data.
        saved = data_options.standardization
        for name in standardizable_vars:
            if name in saved:
                means[name] = float(saved[name].get("mean", 0.0))
                standard_deviations[name] = float(saved[name].get("std", 1.0))
                if saved[name].get("applied"):
                    applied_vars.append(name)
    else:
        # Feature statistics: each standardized feature uses its own stats,
        # computed over the training dates. A dry run skips all statistic
        # computation (the expensive part) and leaves the identity transform.
        if std_features and not dry_run:
            feat_source = processed[std_features]
            if train_dates is not None:
                feat_source = feat_source.sel(time=train_dates)
            feat_means = feat_source.mean(
                dim=("time", "lat", "lon"), skipna=True
            ).compute()
            feat_stds = feat_source.std(
                dim=("time", "lat", "lon"), skipna=True
            ).compute()
            for name in std_features:
                means[name] = float(feat_means[name].item())
                standard_deviations[name] = float(feat_stds[name].item())
                applied_vars.append(name)
        # Target statistics: a single (mean, std) computed from the masked
        # observed_target over the training dates, then shared by the whole
        # target group (full_target, observed_target, and the temporal lags) so
        # the standardized inputs and the training label share one scale.
        if std_target and not dry_run:
            tgt_source = processed["observed_target"]
            if train_dates is not None:
                tgt_source = tgt_source.sel(time=train_dates)
            tgt_mean = float(
                tgt_source.mean(dim=("time", "lat", "lon"), skipna=True)
                .compute()
                .item()
            )
            tgt_std = float(
                tgt_source.std(dim=("time", "lat", "lon"), skipna=True)
                .compute()
                .item()
            )
            for name in target_group:
                means[name] = tgt_mean
                standard_deviations[name] = tgt_std
                applied_vars.append(name)

    std_vars = list(applied_vars)
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
            "estimate_flag",
            "unavailable_flag",
            "observed_flag",
            "land_flag",
        ]
    )
    chunks = {
        dim: size
        for dim, size in output_chunks.items()
        if dim in standardized.dims
    }
    output = standardized[output_vars].chunk(chunks)
    if mode == "train" and fit_dates is not None:
        # Return only the dates needed for model fitting (training + validation).
        # Temporal lags were already computed from the full time series above
        # (via shift), so the lag channels for the returned days still reflect
        # their true neighbours even when those neighbours are dropped here.
        # make_generator later splits this back into train/val via options.split.
        output = output.sel(time=fit_dates)

    if dry_run:
        # Pure query: return the lazy skeleton with the exact channel order and
        # shape, without mutating ``options`` or computing anything.
        return output

    if mode == "train":
        input_names = [name for name in output_vars if name != "full_target"]
        data_options.target = "full_target"
        data_options.lat_bounds = (
            float(output["lat"].min()),
            float(output["lat"].max()),
        )
        data_options.lon_bounds = (
            float(output["lon"].min()),
            float(output["lon"].max()),
        )
        data_options.input_names = input_names
        data_options.log_target = bool(log_target)
        data_options.n_temporal_lags = n_temporal_lags
        data_options.add_geo = bool(add_geo)
        data_options.std_features = list(std_features)
        data_options.std_target = bool(std_target)
        data_options.transforms = {
            "target": "natural logarithm" if log_target else "none",
            "temporal_lags": n_temporal_lags,
            "add_geo": bool(add_geo),
            "std_target": bool(std_target),
            "std_features": list(std_features),
            "cloud_mode": cloud_mode,
            "cloud_seed": cloud_seed,
        }
        if cloud_mode in ("synthetic_bank", "synthetic"):
            data_options.transforms.update(
                {
                    "cloud_coverage": coverage,
                    "cloud_blob_sigma": blob_sigma,
                    "cloud_time_sigma": time_sigma,
                }
            )
            if bank_provenance:
                data_options.transforms.update(bank_provenance)
        else:
            data_options.transforms["missing_flag_shift"] = missing_flag_shift
        data_options.standardization = {
            name: {
                "mean": float(means[name]),
                "std": float(standard_deviations[name]),
                "applied": name in std_vars,
            }
            for name in standardizable_vars
        }
        data_options.target_mean = float(means["full_target"])
        data_options.target_std = float(standard_deviations["full_target"])
        data_options.missing_value_handling = (
            "Missing predictor values are replaced with zero after "
            "standardization; mask channels identify land and missing data."
        )

    if verbose:
        input_names = [name for name in output_vars if name != "full_target"]
        print(f"Channels created ({len(input_names)} total):")
        for i, ch in enumerate(input_names, 1):
            print(f"  {i}. {ch}")
        print(
            "Target standardization: "
            f"mean={means['full_target']:.4f}, "
            f"std={standard_deviations['full_target']:.4f}"
        )
        print(f"Dataset is LAZY (not in memory): {output.chunks}")

    return output


def _random_train_val_indices(
    dates,
    n_train,
    n_val,
    min_day_difference=2,
    seed=None,
):
    """Return training and validation indices into the original date array.

    All selected dates, across both training and validation, differ by at least
    ``min_day_difference`` calendar days. For example, ``min_day_difference=2``
    allows Jan 1 and Jan 3.
    """
    dates = pd.to_datetime(np.asarray(dates))
    if dates.isna().any():
        raise ValueError("The date array contains missing datetime values.")
    if dates.has_duplicates:
        raise ValueError("The date array contains duplicate dates.")

    sort_order = np.argsort(dates.values)
    sorted_dates = dates[sort_order]
    sorted_days = sorted_dates.values.astype("datetime64[D]")

    n_dates = len(sorted_dates)
    n_total = n_train + n_val

    next_valid = np.searchsorted(
        sorted_days,
        sorted_days + np.timedelta64(min_day_difference, "D"),
        side="left",
    )

    max_n = 0
    i = 0
    while i < n_dates:
        max_n += 1
        i = next_valid[i]

    if n_total > max_n:
        raise ValueError(
            f"Requested {n_total} dates, but at most {max_n} can be "
            f"sampled with a {min_day_difference}-day minimum difference."
        )

    dp = [[0] * (n_total + 1) for _ in range(n_dates + 1)]
    for i in range(n_dates + 1):
        dp[i][0] = 1
    for i in range(n_dates - 1, -1, -1):
        for k in range(1, n_total + 1):
            dp[i][k] = dp[i + 1][k] + dp[next_valid[i]][k - 1]

    rng = np.random.default_rng(seed)
    selected_sorted_indices = []
    i = 0
    k = n_total
    while k > 0:
        skip_count = dp[i + 1][k]
        take_count = dp[next_valid[i]][k - 1]
        total_count = skip_count + take_count
        if rng.random() < take_count / total_count:
            selected_sorted_indices.append(i)
            i = next_valid[i]
            k -= 1
        else:
            i += 1

    selected_indices = sort_order[selected_sorted_indices]
    selected_indices = rng.permutation(selected_indices)
    train_indices = np.sort(selected_indices[:n_train])
    val_indices = np.sort(selected_indices[n_train:])
    return train_indices, val_indices, max_n


def train_validation_dates(
    times,
    options,
    *,
    method=None,
    n_days=None,
    train_fraction=None,
    val_fraction=None,
    n_train=None,
    n_val=None,
    train_slice=None,
    val_slice=None,
    min_day_difference=None,
    seed=None,
    verbose=None,
):
    """Choose training and validation dates and record them on ``options``.

    ``times`` is the dataset time coordinate (e.g. ``ds.time``). ``options`` is
    the full :class:`Options` object; the split configuration is read from and
    the chosen ``train_dates`` / ``val_dates`` are recorded on
    ``options.split``. The seed defaults to ``options.resolved_split_seed()``
    (the split's own seed, or the global seed when it is ``None``) and
    ``verbose`` defaults to ``options.verbose``. Returns ``options``.

    ``method`` defaults to ``options.split.method``. ``method="random"`` samples
    spaced-out dates for train/validation using ``min_day_difference`` (defaults
    to ``options.split.min_day_difference``). The counts come from ``n_train`` /
    ``n_val`` when supplied; otherwise ``train_fraction`` / ``val_fraction``
    divide up to ``n_days`` available dates (all default to ``options.split``).
    ``method="manual"`` selects dates falling in ``train_slice`` and
    ``val_slice`` (``slice("1997-01-01", "2000-01-01")``); it raises if a slice
    selects no dates. ``seed`` overrides the resolved default.
    """
    full_options = options
    split = full_options.split
    if seed is None:
        seed = full_options.resolved_split_seed()
    if verbose is None:
        verbose = full_options.verbose

    dates = pd.to_datetime(np.asarray(getattr(times, "values", times)))
    method = method if method is not None else split.method

    if method == "random":
        difference = (
            min_day_difference
            if min_day_difference is not None
            else split.min_day_difference
        )
        train_fraction = (
            train_fraction
            if train_fraction is not None
            else split.train_fraction
        )
        val_fraction = (
            val_fraction if val_fraction is not None else split.val_fraction
        )
        n_days = n_days if n_days is not None else split.n_days
        if n_days is None:
            total = len(dates)
        else:
            if (
                not isinstance(n_days, (int, np.integer))
                or isinstance(n_days, (bool, np.bool_))
                or n_days <= 0
            ):
                raise ValueError("n_days must be a positive integer")
            n_days = int(n_days)
            total = min(n_days, len(dates))
        if n_train is None:
            n_train = max(1, int(total * train_fraction))
        if n_val is None:
            n_val = max(1, int(total * val_fraction))
        train_idx, val_idx, _ = _random_train_val_indices(
            dates,
            n_train,
            n_val,
            min_day_difference=difference,
            seed=seed,
        )
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]
        split.min_day_difference = difference
        split.n_days = n_days
        split.train_fraction = train_fraction
        split.val_fraction = val_fraction
    elif method == "manual":
        if train_slice is None or val_slice is None:
            raise ValueError(
                "method='manual' requires train_slice and val_slice"
            )
        index = pd.DatetimeIndex(dates)
        train_dates = index[
            (index >= pd.to_datetime(train_slice.start))
            & (index <= pd.to_datetime(train_slice.stop))
        ]
        val_dates = index[
            (index >= pd.to_datetime(val_slice.start))
            & (index <= pd.to_datetime(val_slice.stop))
        ]
        if len(train_dates) == 0:
            raise ValueError(
                f"train_slice {train_slice.start} to {train_slice.stop} "
                "selects no dates in the dataset time range "
                f"({dates.min().date()} to {dates.max().date()})"
            )
        if len(val_dates) == 0:
            raise ValueError(
                f"val_slice {val_slice.start} to {val_slice.stop} "
                "selects no dates in the dataset time range "
                f"({dates.min().date()} to {dates.max().date()})"
            )
    else:
        raise ValueError(
            f"Unknown method {method!r}; choose 'random' or 'manual'"
        )

    split.method = method
    split.train_dates = [str(d.date()) for d in pd.to_datetime(train_dates)]
    split.val_dates = [str(d.date()) for d in pd.to_datetime(val_dates)]
    split.seed = seed
    if verbose:
        print(
            f"Split method: {method} "
            f"(train={len(split.train_dates)}, "
            f"val={len(split.val_dates)} dates)"
        )
        print(f"Training period: {split.training_period()}")
    return full_options
