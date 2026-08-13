"""Shared test fixtures and helpers.

``demo_data`` no longer generates synthetic data; it only loads real remote
datasets. Tests build their in-memory datasets with :func:`make_demo_ds`, which
produces the same ``(chlor_a, cloud_flag, land_flag)`` layout the tests relied
on, plus a matching ``metadata`` dict, without any network access.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def make_demo_ds(
    *,
    days=120,
    lat_size=16,
    lon_size=16,
    start="2020-01-01",
    seed=42,
    cloud_fraction=0.12,
):
    """Return ``(ds, metadata)`` for a deterministic synthetic demo dataset.

    Mirrors the layout the pipeline expects: a ``chlor_a`` target with
    ``cloud_flag`` and ``land_flag`` masks, spatial ``lat``/``lon`` coordinates,
    and a daily ``time`` coordinate. Deterministic given ``seed``.
    """
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
            "cloud_flag": (("time", "lat", "lon"), cloud.astype("int8")),
            "land_flag": (("time", "lat", "lon"), land.astype("int8")),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["chlor_a"].attrs["units"] = "mg m-3"

    metadata = {
        "dataset": {
            "name": "Synthetic",
            "product_id": "mindthegap-synthetic",
            "loader": "synthetic",
            "region": {
                "lat": [float(ds["lat"].min()), float(ds["lat"].max())],
                "lon": [float(ds["lon"].min()), float(ds["lon"].max())],
            },
            "available_period": (
                f"{pd.to_datetime(ds.time.values[0]).date()} to "
                f"{pd.to_datetime(ds.time.values[-1]).date()}"
            ),
            "data_source": "test fixture",
        },
        "target": {
            "name": "chlor_a",
            "units": "mg m-3",
        },
        "variables": {
            "target": "chlor_a",
            "features": [],
            "missing_flag": "cloud_flag",
            "land_flag": "land_flag",
        },
    }
    return ds, metadata


@pytest.fixture
def demo_ds():
    """Default in-memory synthetic dataset ``(ds, metadata)``."""
    return make_demo_ds()
