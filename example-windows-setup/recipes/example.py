#!/usr/bin/env python3
"""
A simple python example script to save a plot of the CHL variable at time index 1000 from the globcolour dataset.
"""

import icechunk as ic
import xarray as xr
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from pathlib import Path


# https://icechunk.io/en/latest/guides/performance/#setting-zarrs-async-concurrency-configuration
import zarr
zarr.config.set({"async.concurrency": 128})

url = "https://data.source.coop/fish-pace/globcolour/cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D"
storage = ic.http_storage(url)
repo = ic.Repository.open(storage)
auth = {p: ic.credentials.HttpAccess for p in repo.config.virtual_chunk_containers or []}
store = repo.reopen(authorize_virtual_chunk_access=auth).readonly_session("main").store

ds = xr.open_zarr(store,
                  consolidated=False,
                  chunks={"time": 1, "lat": -1, "lon": -1},  # NOT chunks={}
                  )


ds.CHL.isel(time=1000).load().plot.imshow(norm=LogNorm())

outdir = Path("outputs")
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / "CHL_t1000.png")
print('saved figure to', outdir / "CHL_t1000.png")
