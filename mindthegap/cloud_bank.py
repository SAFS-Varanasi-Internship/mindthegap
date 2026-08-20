"""Loading precomputed synthetic-cloud *banks* for ``mindthegap``.

A cloud bank is a bit-packed boolean ``(n_time, n_lat, n_lon)`` cloud cube built
offline by ``scripts/make_cloud_bank.py`` with the same construction as
:func:`mindthegap.data.synthetic_cloud_cube`. ``prepare_model_data(cloud_mode=
"synthetic_bank")`` draws synthetic clouds from a bank -- tiling/wrapping it over
the dataset grid with random offsets and flips -- so the cost of generating
synthetic clouds no longer grows with the size of the dataset.

Banks are not shipped inside the wheel (they are ~10 MB each and there may be
several). Instead a small manifest (``cloud_banks_manifest.json``) ships with the
package and records, for each bank, the generation parameters plus the GitHub
release asset URL and sha256. :func:`fetch_bank` downloads a bank on first use
into a local cache, verifies its checksum, and reuses it thereafter.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

_MANIFEST_NAME = "cloud_banks_manifest.json"


def _cache_dir():
    """Return the local cache directory for downloaded cloud banks.

    Honours ``MINDTHEGAP_CACHE_DIR`` and falls back to ``platformdirs`` when
    available, otherwise ``~/.cache/mindthegap``.
    """
    override = os.environ.get("MINDTHEGAP_CACHE_DIR")
    if override:
        base = Path(override)
    else:
        try:
            import platformdirs

            base = Path(platformdirs.user_cache_dir("mindthegap"))
        except Exception:
            base = Path.home() / ".cache" / "mindthegap"
    path = base / "cloud_banks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest():
    """Return the bundled cloud-bank manifest as a dict."""
    try:
        from importlib.resources import files

        text = (files("mindthegap") / _MANIFEST_NAME).read_text()
    except Exception:
        text = (Path(__file__).with_name(_MANIFEST_NAME)).read_text()
    return json.loads(text)


def find_bank_entry(
    *,
    coverage,
    blob_sigma,
    time_sigma,
    n_lat=None,
    n_lon=None,
    n_time=None,
    manifest=None,
    tol=1e-6,
):
    """Return the manifest entry matching the requested parameters, or ``None``.

    ``coverage``/``blob_sigma``/``time_sigma`` are matched within ``tol``.
    ``n_lat``/``n_lon``/``n_time`` are matched exactly when supplied (``None``
    means "don't care").
    """
    manifest = manifest if manifest is not None else load_manifest()

    def close(a, b):
        return abs(float(a) - float(b)) <= tol

    for entry in manifest.get("banks", []):
        if not (
            close(entry["coverage"], coverage)
            and close(entry["blob_sigma"], blob_sigma)
            and close(entry["time_sigma"], time_sigma)
        ):
            continue
        if n_lat is not None and int(entry["n_lat"]) != int(n_lat):
            continue
        if n_lon is not None and int(entry["n_lon"]) != int(n_lon):
            continue
        if n_time is not None and int(entry["n_time"]) != int(n_time):
            continue
        return entry
    return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _asset_url(entry, manifest):
    if entry.get("url"):
        return entry["url"]
    base = manifest.get("asset_base_url", "").rstrip("/")
    return f"{base}/{entry['filename']}"


def fetch_bank(entry, *, manifest=None, cache_dir=None):
    """Return a local path to the bank file described by ``entry``.

    Looks in the local cache first; if the file is missing or fails its sha256
    check it is (re)downloaded from the release asset URL. Raises with a clear
    message if the download or checksum fails.
    """
    manifest = manifest if manifest is not None else load_manifest()
    cache_dir = Path(cache_dir) if cache_dir is not None else _cache_dir()
    path = cache_dir / entry["filename"]
    expected = entry.get("sha256")

    if path.exists() and (expected is None or _sha256(path) == expected):
        return path

    url = _asset_url(entry, manifest)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        from urllib.request import urlopen

        with urlopen(url) as response, open(tmp, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:  # pragma: no cover - network dependent
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to download cloud bank {entry['filename']!r} from {url}: "
            f"{exc}"
        ) from exc

    if expected is not None:
        got = _sha256(tmp)
        if got != expected:
            tmp.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {entry['filename']!r}: expected "
                f"{expected}, got {got}"
            )
    os.replace(tmp, path)
    return path


def open_bank(path, *, chunk_time=None):
    """Open a bank file and return a dask-backed boolean ``DataArray``.

    The packed bits are unpacked *once* into a boolean ``(n_time, n_lat, n_lon)``
    array which is then wrapped in a dask array (chunked along time by
    ``chunk_time``, default one chunk per time step's worth is avoided by using
    a moderate block) so downstream indexing only realises the regions it
    touches while the resident memory is just the (modest) unpacked bank.
    """
    import dask.array as da
    import xarray as xr

    with xr.open_dataset(path) as ds:
        attrs = dict(ds.attrs)
        n_time = int(attrs["n_time"])
        n_lat = int(attrs["n_lat"])
        n_lon = int(attrs["n_lon"])
        packed = ds["packed"].values

    n_bits = n_time * n_lat * n_lon
    # np.unpackbits returns uint8 (0/1); view it as bool (same 1-byte layout)
    # instead of astype() to avoid an extra full-size copy.
    unpacked = np.unpackbits(packed)[:n_bits]
    cube = unpacked.view(bool).reshape(n_time, n_lat, n_lon)

    if chunk_time is None:
        chunk_time = min(n_time, 100)
    chunk_time = int(max(1, min(chunk_time, n_time)))

    arr = da.from_array(cube, chunks=(chunk_time, n_lat, n_lon))
    return xr.DataArray(arr, dims=("time", "lat", "lon"), attrs=attrs)
