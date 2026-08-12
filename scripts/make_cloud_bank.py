"""Generate a synthetic-cloud *bank* artifact for ``mindthegap``.

The bank is a precomputed ``(time, lat, lon)`` boolean cloud cube built with the
same construction as :func:`mindthegap.data.synthetic_cloud_cube` -- gaussian-
smoothed noise thresholded at the ``coverage`` quantile. ``prepare_model_data``
draws synthetic clouds from this bank (with random day/space offsets and flips)
instead of smoothing a full-record noise field, so the cost of ``cloud_mode=
"synthetic"`` no longer grows with the length of the dataset.

The cube is bit-packed with ``numpy.packbits`` and stored in a single, self-
describing netCDF file. The generation function name and every parameter are
written to the file attributes so the artifact records exactly how it was made.

Run::

    python scripts/make_cloud_bank.py

which writes ``assets/cloud_banks/<name>.nc``.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

import mindthegap


# Defaults: the synthetic-cloud settings currently used by prepare_model_data.
DEFAULTS = dict(
    n_time=730,          # 2 years of daily composites
    n_lat=640,
    n_lon=640,
    coverage=0.4,
    blob_sigma=6.0,
    time_sigma=2.0,
    seed=0,
    time_block=146,      # build in blocks to bound peak memory
)


def _build_cube(
    n_time,
    n_lat,
    n_lon,
    coverage,
    blob_sigma,
    time_sigma,
    seed,
    time_block,
):
    """Build the bool cloud cube with the ``synthetic_cloud_cube`` construction.

    Identical statistics to :func:`mindthegap.data.synthetic_cloud_cube`
    (gaussian-smoothed noise thresholded at the coverage quantile) but generated
    in time blocks -- with a small ``time_sigma`` overlap between blocks so the
    temporal smoothing is continuous -- so the full float noise field is never
    materialised at once. The coverage threshold is the exact global quantile of
    the whole smoothed field (this is a one-time offline build, so we can afford
    a full pass).
    """
    rng = np.random.default_rng(seed)

    # One reproducible noise field, drawn up front, smoothed block-wise.
    noise = rng.standard_normal((n_time, n_lat, n_lon)).astype("float32")
    depth = int(np.ceil(3 * time_sigma)) if time_sigma > 0 else 0

    field = np.empty((n_time, n_lat, n_lon), dtype="float32")
    for start in range(0, n_time, time_block):
        stop = min(start + time_block, n_time)
        lo = max(0, start - depth)
        hi = min(n_time, stop + depth)
        smoothed = gaussian_filter(
            noise[lo:hi], sigma=(time_sigma, blob_sigma, blob_sigma)
        )
        field[start:stop] = smoothed[start - lo : stop - lo]

    if coverage <= 0:
        return np.zeros((n_time, n_lat, n_lon), dtype=bool)
    threshold = float(np.quantile(field, 1.0 - coverage))
    return field > threshold


def make_bank(output_dir, **params):
    p = {**DEFAULTS, **params}
    cube = _build_cube(
        p["n_time"],
        p["n_lat"],
        p["n_lon"],
        p["coverage"],
        p["blob_sigma"],
        p["time_sigma"],
        p["seed"],
        p["time_block"],
    )
    achieved = float(cube.mean())

    packed = np.packbits(cube)
    created = datetime.now(timezone.utc).isoformat()

    ds = xr.Dataset(
        {"packed": ("packed_index", packed)},
        attrs={
            # How the synthetics were created -- self-describing provenance.
            "description": (
                "Bit-packed boolean synthetic-cloud bank for mindthegap "
                "(cloud_mode='synthetic'). Unpack with numpy.unpackbits and "
                "reshape to (n_time, n_lat, n_lon)."
            ),
            "generator": "mindthegap.data.synthetic_cloud_cube",
            "generator_construction": (
                "scipy.ndimage.gaussian_filter(standard_normal(shape), "
                "sigma=(time_sigma, blob_sigma, blob_sigma)) thresholded at the "
                "(1 - coverage) quantile"
            ),
            "mindthegap_version": getattr(mindthegap, "__version__", "unknown"),
            "created_utc": created,
            "n_time": int(p["n_time"]),
            "n_lat": int(p["n_lat"]),
            "n_lon": int(p["n_lon"]),
            "coverage": float(p["coverage"]),
            "coverage_achieved": achieved,
            "blob_sigma": float(p["blob_sigma"]),
            "time_sigma": float(p["time_sigma"]),
            "seed": int(p["seed"]),
            "packbits_axis": "C-order flatten of (n_time, n_lat, n_lon) bool",
            "dtype": "bool (packed as uint8 bits)",
        },
    )

    name = (
        f"clouds_blob{p['blob_sigma']:g}_time{p['time_sigma']:g}"
        f"_cov{int(round(p['coverage'] * 100))}"
        f"_{p['n_lat']}x{p['n_lon']}_{p['n_time']}d.nc"
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    ds.to_netcdf(
        path, encoding={"packed": {"zlib": True, "complevel": 9}}
    )

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {path}")
    print(f"  shape           = ({p['n_time']}, {p['n_lat']}, {p['n_lon']})")
    print(f"  coverage target = {p['coverage']}  achieved = {achieved:.4f}")
    print(f"  blob_sigma      = {p['blob_sigma']}")
    print(f"  time_sigma      = {p['time_sigma']}")
    print(f"  seed            = {p['seed']}")
    print(f"  size            = {size_mb:.2f} MB")
    print(f"  sha256          = {sha}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="assets/cloud_banks")
    for key in ("n_time", "n_lat", "n_lon", "seed", "time_block"):
        ap.add_argument(f"--{key}", type=int, default=DEFAULTS[key])
    for key in ("coverage", "blob_sigma", "time_sigma"):
        ap.add_argument(f"--{key}", type=float, default=DEFAULTS[key])
    args = ap.parse_args()
    make_bank(**vars(args))


if __name__ == "__main__":
    main()
