"""Self-supervised gap-fill training pieces.

Everything needed to train and score a gap-filling model against the satellite's
own observations (no gap-free truth): a masked loss that only scores real ocean,
the fake-cloud generators used to hide observed pixels, the PACE channel builder,
the season-aware / contiguous splits, and the fake-cloud metric.

Design notes worth carrying forward:
- The masked loss takes a **2-channel target** ``[value, mask]`` so no NaN ever
  reaches the graph (NaN targets + a ``tf.where`` masking loss crash cuDNN on the
  T4). Never compile with a *custom metric* on that GPU either; score after fit.
- Synthetic clouds must be **temporally correlated** (``time_sigma > 0``) to keep
  the prev/next-day channels honest; independent clouds leak (the model just copies
  the hidden pixel from the neighbouring day).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------------------------- #
# masked losses (2-channel target [value, mask]; score only observed ocean)
# --------------------------------------------------------------------------- #
def masked_mse(y_true, y_pred):
    """MSE over observed-ocean pixels only. ``y_true`` is 2 channels
    ``[value, mask]``; ``mask`` is 1 where the pixel should count."""
    y = y_true[..., 0:1]
    m = y_true[..., 1:2]
    return tf.reduce_sum(tf.square(y - y_pred) * m) / (tf.reduce_sum(m) + 1e-6)


def masked_mae(y_true, y_pred):
    """MAE counterpart of :func:`masked_mse` (2-channel target)."""
    y = y_true[..., 0:1]
    m = y_true[..., 1:2]
    return tf.reduce_sum(tf.abs(y - y_pred) * m) / (tf.reduce_sum(m) + 1e-6)


def target_with_mask(y):
    """Turn a target ``y`` (with NaN at land/gaps) into the 2-channel
    ``[filled_value, finite_mask]`` array the masked losses expect."""
    return np.stack([np.nan_to_num(y, nan=0.0), np.isfinite(y).astype("float32")], -1).astype("float32")


# --------------------------------------------------------------------------- #
# fake-cloud generators
# --------------------------------------------------------------------------- #
def keep_cloud_blobs(foot, span_frac=0.7, aspect=4.0):
    """Keep compact, cloud-shaped connected components of a boolean mask; drop the
    long/thin/edge-to-edge blobs that are orbital swaths.

    A stopgap swath filter until PACE's built-in orbital-swath flag is wired up.
    A blob is dropped if its bounding box spans >= ``span_frac`` of the domain in
    either direction, or is more than ``aspect`` : 1 elongated.
    """
    H, W = foot.shape
    lab, n = ndimage.label(foot)
    if n == 0:
        return foot
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        keep[i] = not ((h >= span_frac * H) or (w >= span_frac * W)
                       or max(h, w) / max(1, min(h, w)) > aspect)
    return keep[lab]


def synthetic_cloud_cube(T, H, W, coverage, blob_sigma=6.0, time_sigma=0.0, rng=None):
    """A ``(T, H, W)`` boolean synthetic-cloud cube: smooth gaussian noise
    thresholded to ~``coverage`` of pixels.

    ``blob_sigma`` sets cloud size; ``time_sigma`` sets day-to-day persistence
    (0 = independent per composite, larger = clouds that last and evolve, with
    cores that persist and edges that flicker). Correlated clouds keep the
    prev/next-day channels honest.
    """
    if rng is None:
        rng = np.random.default_rng()
    f = gaussian_filter(rng.standard_normal((T, H, W)), sigma=(time_sigma, blob_sigma, blob_sigma))
    return f > np.quantile(f, 1.0 - coverage)


# --------------------------------------------------------------------------- #
# PACE channel builder
# --------------------------------------------------------------------------- #
def build_pace_channels(chl_log, times, train_mask, n_days=1, cloud_mode="synthetic",
                        coverage=0.4, blob_sigma=6.0, time_sigma=0.0,
                        span_frac=0.7, aspect=4.0, cloud_len=3, seed=0, standardize_chl=True,
                        stats=None):
    """Build the self-supervised input channels and target from a PACE chlorophyll
    cube alone.

    ``chl_log`` is ``(T, H, W)`` log Chl-a with NaN at land and real gaps. Fake
    clouds hide observed ocean pixels:

    - ``cloud_mode='synthetic'``: gaussian-blob clouds (:func:`synthetic_cloud_cube`);
      set ``time_sigma > 0`` for realistic day-to-day persistence.
    - ``cloud_mode='shape'``: borrow a shifted day's real gaps and keep only compact
      blobs (:func:`keep_cloud_blobs`), i.e. clouds not swaths.

    Channels: ``sin_time``/``cos_time``, ``masked_CHL``, ``prev{k}``/``next{k}`` for
    ``k = 1..n_days`` (built from the *masked* CHL, no leak), and land / real-cloud /
    valid / fake-cloud flags. Numeric channels are standardized on ``train_mask``.

    Returns ``(ch, y, stats, order)``: the channel dict, the standardized target
    ``y`` (NaN kept at land/gaps), a ``stats`` dict (incl. ``stats['CHL'] = (mean,
    std)``), and ``order`` the channel names in model-input order.

    Pass ``stats=`` (a dict from a previous call) to standardize with SHARED stats
    instead of recomputing per call -- required when streaming many spatial chunks so
    every chunk uses one consistent normalization (and one CHL mean/std to invert with).
    """
    T, H, W = chl_log.shape
    gap = np.isnan(chl_log)
    land = gap.all(0)
    land3 = np.broadcast_to(land, chl_log.shape)
    real_cloud = gap & ~land3
    rng = np.random.default_rng(seed)

    if cloud_mode == "synthetic":
        cube = synthetic_cloud_cube(T, H, W, coverage, blob_sigma, time_sigma, rng)
        fake = cube & ~gap & ~land3
    else:  # shape: borrow a shifted day's gaps, keep compact blobs, hold per block
        fake = np.zeros((T, H, W), bool)
        for s in range(0, T, cloud_len):
            e = min(s + cloud_len, T)
            foot = keep_cloud_blobs(gap[(s + T // 2) % T] & ~land, span_frac, aspect)
            fake[s:e] = foot[None] & ~gap[s:e] & ~land3[s:e]

    masked = np.where(fake, np.nan, chl_log)
    doy = pd.to_datetime(times).dayofyear.to_numpy().astype("float32")
    ang = 2 * np.pi * doy / 365.0
    ch = {"sin_time": np.broadcast_to(np.sin(ang)[:, None, None], chl_log.shape).astype("float32").copy(),
          "cos_time": np.broadcast_to(np.cos(ang)[:, None, None], chl_log.shape).astype("float32").copy(),
          "masked_CHL": masked}
    for k in range(1, n_days + 1):
        p = np.roll(masked, k, 0)
        p[:k] = np.nan
        nx = np.roll(masked, -k, 0)
        nx[-k:] = np.nan
        ch[f"prev{k}_CHL"] = p
        ch[f"next{k}_CHL"] = nx

    numeric = list(ch.keys())
    valid = (~np.isnan(masked)) & ~land3
    ch.update({"land_flag": land3.astype("float32"),
               "real_cloud_flag": real_cloud.astype("float32"),
               "valid_CHL_flag": valid.astype("float32"),
               "fake_cloud_flag": fake.astype("float32")})

    # standardize numeric channels. If ``stats`` is given (shared/global stats, e.g. for
    # streaming many chunks with ONE consistent normalization), use it; else compute on
    # ``train_mask`` and return the computed dict.
    if stats is None:
        stats = {}
        for key in numeric:
            trd = ch[key][train_mask]
            m, s = float(np.nanmean(trd)), float(np.nanstd(trd))
            stats[key] = (m, s)
            ch[key] = np.nan_to_num((ch[key] - m) / (s + 1e-8), nan=0.0, posinf=0.0, neginf=0.0)
        ym, ys = ((float(np.nanmean(chl_log[train_mask])), float(np.nanstd(chl_log[train_mask])))
                  if standardize_chl else (0.0, 1.0))
        stats["CHL"] = (ym, ys)
    else:
        for key in numeric:
            m, s = stats[key]
            ch[key] = np.nan_to_num((ch[key] - m) / (s + 1e-8), nan=0.0, posinf=0.0, neginf=0.0)
        ym, ys = stats["CHL"]
    order = numeric + ["land_flag", "real_cloud_flag", "valid_CHL_flag", "fake_cloud_flag"]
    return ch, (chl_log - ym) / ys, stats, order


# --------------------------------------------------------------------------- #
# train / val / test splits
# --------------------------------------------------------------------------- #
SEASON_BREAKS = [40, 75, 225, 305]   # day-of-year cuts between the 4 Arabian Sea chl seasons


def season_of(doy):
    """Arabian Sea chlorophyll season (0-3) for a day-of-year: NE-monsoon bloom,
    spring low, SW-monsoon bloom, fall transition. Breaks are eyeballed, tunable."""
    if SEASON_BREAKS[0] <= doy < SEASON_BREAKS[1]:
        return 0
    if SEASON_BREAKS[1] <= doy < SEASON_BREAKS[2]:
        return 1
    if SEASON_BREAKS[2] <= doy < SEASON_BREAKS[3]:
        return 2
    return 3


def season_blocked_split(times, buf=2):
    """Season-aware blocked train/val/test masks over a time axis.

    Blocks are contiguous same-season runs. The first occurrence of each season
    goes to train (so train sees all four regimes); later occurrences rotate
    val/test. A ``buf``-step buffer drops boundary steps to limit leakage.
    Returns boolean masks ``(train, val, test)``.
    """
    doy = pd.to_datetime(times).dayofyear.to_numpy()
    seas = np.array([season_of(d) for d in doy])
    block = np.cumsum(np.r_[0, seas[1:] != seas[:-1]])
    role = np.array(["tr"] * len(seas), dtype=object)
    seen, rot = {}, 0
    for bid in range(block.max() + 1):
        m = block == bid
        s = int(seas[m][0])
        c = seen.get(s, 0)
        seen[s] = c + 1
        if c > 0:
            role[m] = ("val", "test")[rot % 2]
            rot += 1
    drop = np.array([len(set(role[max(0, i - buf):i + buf + 1])) > 1 for i in range(len(role))])
    return (role == "tr") & ~drop, (role == "val") & ~drop, (role == "test") & ~drop


def contiguous_split(T, frac_train=0.6, frac_val=0.2, buf=2):
    """Contiguous train/val/test masks of ``T`` steps with buffer gaps (a
    conservative future-holdout split). Returns boolean masks ``(train, val, test)``."""
    n_tr, n_va = int(frac_train * T), int(frac_val * T)
    tr = np.zeros(T, bool)
    va = np.zeros(T, bool)
    te = np.zeros(T, bool)
    tr[:n_tr] = True
    va[n_tr + buf:n_tr + buf + n_va] = True
    te[n_tr + buf + n_va + buf:] = True
    return tr, va, te


# --------------------------------------------------------------------------- #
# self-supervised metric
# --------------------------------------------------------------------------- #
def fake_cloud_mae(pred, truth, fake_flag):
    """Mean absolute error at the held-out fake-cloud pixels only.

    ``pred`` and ``truth`` are same-shape arrays in the same units; ``fake_flag``
    is the fake-cloud mask. Works for any predictor, so persistence's score is
    ``fake_cloud_mae(prev_day_value, truth, fake_flag)``.
    """
    m = (np.asarray(fake_flag) == 1) & np.isfinite(truth) & np.isfinite(pred)
    return float(np.mean(np.abs(pred[m] - truth[m])))
