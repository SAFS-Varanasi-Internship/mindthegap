import json

import numpy as np
import pytest
import xarray as xr

from mindthegap.data import (
    _bank_to_cube,
    prepare_model_data,
)
from mindthegap import cloud_bank as cb
from mindthegap import Options, train_validation_dates
from conftest import make_demo_ds


def _full_options(ds, metadata, *, seed=1, **cloud_kwargs):
    """Full Options resolved for the demo dataset with cloud overrides.

    ``prepare_model_data`` reads every setting from ``options``; cloud
    configuration is set on ``options.data`` (never a call argument).
    """
    options = Options.default(data=ds, metadata=metadata, seed=seed)
    options.verbose = False
    for key, value in cloud_kwargs.items():
        setattr(options.data, key, value)
    train_validation_dates(ds.time, options, seed=seed, verbose=False)
    return options


def _write_bank(path, n_time=40, n_lat=32, n_lon=32, coverage=0.4, seed=0):
    """Write a small self-describing packed-bits bank file for tests."""
    from mindthegap.data import synthetic_cloud_cube

    cube = synthetic_cloud_cube(
        n_time,
        n_lat,
        n_lon,
        coverage=coverage,
        blob_sigma=4.0,
        time_sigma=2.0,
        rng=np.random.default_rng(seed),
    )
    packed = np.packbits(cube)
    ds = xr.Dataset(
        {"packed": ("packed_index", packed)},
        attrs={
            "generator": "mindthegap.data.synthetic_cloud_cube",
            "n_time": n_time,
            "n_lat": n_lat,
            "n_lon": n_lon,
            "coverage": coverage,
            "blob_sigma": 4.0,
            "time_sigma": 2.0,
            "seed": seed,
        },
    )
    ds.to_netcdf(path)
    return cube


def test_open_bank_roundtrips(tmp_path):
    path = tmp_path / "bank.nc"
    cube = _write_bank(path)

    bank = cb.open_bank(path, chunk_time=10)
    assert bank.dims == ("time", "lat", "lon")
    assert bank.dtype == np.dtype("bool")
    assert bank.shape == cube.shape
    np.testing.assert_array_equal(bank.values, cube)
    # Self-describing attributes survive.
    assert bank.attrs["generator"] == "mindthegap.data.synthetic_cloud_cube"
    assert bank.attrs["coverage"] == 0.4


def test_find_bank_entry_matches_params():
    manifest = {
        "banks": [
            {
                "filename": "a.nc",
                "coverage": 0.4,
                "blob_sigma": 6.0,
                "time_sigma": 2.0,
                "n_time": 730,
                "n_lat": 640,
                "n_lon": 640,
            }
        ]
    }
    entry = cb.find_bank_entry(
        coverage=0.4, blob_sigma=6.0, time_sigma=2.0, manifest=manifest
    )
    assert entry is not None and entry["filename"] == "a.nc"
    # No match -> None.
    assert (
        cb.find_bank_entry(
            coverage=0.3, blob_sigma=6.0, time_sigma=2.0, manifest=manifest
        )
        is None
    )


def test_bundled_manifest_loads():
    manifest = cb.load_manifest()
    assert "banks" in manifest
    assert isinstance(manifest["banks"], list)
    for entry in manifest["banks"]:
        for key in ("filename", "coverage", "blob_sigma", "time_sigma"):
            assert key in entry


def test_bank_to_cube_shape_coverage_and_reproducible(tmp_path):
    path = tmp_path / "bank.nc"
    _write_bank(path, n_time=40, n_lat=32, n_lon=32, coverage=0.4)
    bank = cb.open_bank(path, chunk_time=20)

    # Map onto a larger grid than the bank (tile + wrap).
    cube = _bank_to_cube(
        bank, n_time=100, n_lat=80, n_lon=96, time_chunk=25,
        rng=np.random.default_rng(7),
    )
    assert cube.shape == (100, 80, 96)
    # Chunked to (time_chunk, full-lat, full-lon).
    assert cube.chunks[0][0] == 25
    assert cube.chunks[1][0] == 80

    realized = cube.compute()
    assert realized.dtype == np.dtype("bool")
    assert abs(realized.mean() - 0.4) < 0.1

    again = _bank_to_cube(
        bank, n_time=100, n_lat=80, n_lon=96, time_chunk=25,
        rng=np.random.default_rng(7),
    ).compute()
    np.testing.assert_array_equal(realized, again)

    different = _bank_to_cube(
        bank, n_time=100, n_lat=80, n_lon=96, time_chunk=25,
        rng=np.random.default_rng(8),
    ).compute()
    assert not np.array_equal(realized, different)


def test_bank_to_cube_preserves_temporal_correlation(tmp_path):
    path = tmp_path / "bank.nc"
    _write_bank(path, n_time=60, n_lat=48, n_lon=48, coverage=0.4)
    bank = cb.open_bank(path, chunk_time=30)
    cube = _bank_to_cube(
        bank, n_time=120, n_lat=48, n_lon=48, time_chunk=40,
        rng=np.random.default_rng(3),
    ).compute()

    def corr(i):
        return np.corrcoef(cube[i].ravel(), cube[i + 1].ravel())[0, 1]

    # Adjacent days should be correlated (bank is a continuous sequence).
    adjacent = np.mean([corr(t) for t in range(0, 50)])
    assert adjacent > 0.3


def test_prepare_model_data_bank_falls_back_when_no_match():
    ds, metadata = make_demo_ds(days=16, lat_size=16, lon_size=16, seed=3)
    options = _full_options(
        ds,
        metadata,
        cloud_mode="synthetic_bank",
        # No published bank matches these params -> fall back.
        cloud_coverage=0.37,
        cloud_blob_sigma=6.0,
        cloud_time_sigma=2.0,
        cloud_seed=1,
    )
    with pytest.warns(RuntimeWarning):
        standardized = prepare_model_data(ds, options, mode="train")
    # Fallback still produces synthetic clouds over real ocean.
    estimate = standardized["estimate_flag"].values == 1
    assert estimate.any()


def test_prepare_model_data_bank_uses_cache(tmp_path, monkeypatch):
    # Point the cache at a temp dir and pre-place a matching bank so no network
    # is needed; the manifest is monkeypatched to reference it.
    monkeypatch.setenv("MINDTHEGAP_CACHE_DIR", str(tmp_path))
    bank_dir = tmp_path / "cloud_banks"
    bank_dir.mkdir(parents=True)
    bank_path = bank_dir / "test_bank.nc"
    _write_bank(bank_path, n_time=40, n_lat=32, n_lon=32, coverage=0.4)

    manifest = {
        "banks": [
            {
                "filename": "test_bank.nc",
                "coverage": 0.4,
                "blob_sigma": 4.0,
                "time_sigma": 2.0,
                "n_time": 40,
                "n_lat": 32,
                "n_lon": 32,
                "sha256": None,
            }
        ]
    }
    monkeypatch.setattr(cb, "load_manifest", lambda: manifest)

    ds, metadata = make_demo_ds(days=16, lat_size=16, lon_size=16, seed=3)
    options = _full_options(
        ds,
        metadata,
        cloud_mode="synthetic_bank",
        cloud_coverage=0.4,
        cloud_blob_sigma=4.0,
        cloud_time_sigma=2.0,
        cloud_seed=1,
    )
    standardized = prepare_model_data(ds, options, mode="train")
    estimate = standardized["estimate_flag"].values == 1
    land = standardized["land_flag"].values == 1
    unavailable = standardized["unavailable_flag"].values == 1
    assert estimate.any()
    assert not (estimate & land).any()
    assert not (estimate & unavailable).any()
