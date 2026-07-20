"""Spatial patching and tiled inference.

Two jobs live here: pick *where* training crops go (the samplers), and stitch
tile predictions back into a full map at inference time (:func:`tiled_predict`).

The samplers all return a list of ``(row, col)`` top-left corners for a
``th x tw`` crop. Feed a sampler to :func:`make_crops` to materialize a training
set. For inference over a region too big for one forward pass, tile with
:func:`grid_positions` (or just call :func:`tiled_predict`, which blends the
overlaps to remove tile-boundary seams).
"""
import numpy as np
from scipy import ndimage


def grid_positions(H, W, th, tw, overlap=(0, 0)):
    """Fixed, evenly-spaced tile corners covering ``(H, W)``.

    Drops partial edge tiles (matches xbatcher), so a region that does not divide
    evenly leaves an untrained strip. ``overlap`` is ``(rows, cols)`` of tile
    overlap; larger overlap regains full coverage and adds edge-context.
    """
    oh, ow = overlap or (0, 0)
    sh, sw = th - oh, tw - ow
    n_h = (H - th) // sh + 1
    n_w = (W - tw) // sw + 1
    return [(i * sh, j * sw) for i in range(n_h) for j in range(n_w)]


def random_positions(H, W, th, tw, n, rng, ocean=None, min_ocean=0.0):
    """``n`` uniformly-random crop corners.

    If ``ocean`` (a ``(H, W)`` bool mask) is given and ``min_ocean > 0``, only
    keep crops that are at least ``min_ocean`` fraction ocean (ocean-aware
    sampling), so crops do not waste the window on land.
    """
    out, tries = [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        yy = int(rng.integers(0, H - th + 1))
        xx = int(rng.integers(0, W - tw + 1))
        if min_ocean <= 0 or ocean[yy:yy + th, xx:xx + tw].mean() >= min_ocean:
            out.append((yy, xx))
    return out


def coast_positions(ocean, coast, th, tw, n, rng, min_ocean=0.5):
    """``n`` crops anchored so an edge sits on the shoreline.

    ``coast`` is the set of ocean pixels touching land (see :func:`coast_mask`).
    For each sampled coast pixel we try placing it on the right / left / top edge
    of the crop and keep whichever placement extends into the most ocean.
    """
    ys, xs = np.where(coast)
    H, W = ocean.shape
    out = []
    for i in rng.choice(len(ys), size=n * 8, replace=True):
        r, c = int(ys[i]), int(xs[i])
        best, bf = None, -1
        for yy, xx in [(r - th // 2, c - tw + 1), (r - th // 2, c), (r, c - tw // 2)]:
            yy = int(np.clip(yy, 0, H - th))
            xx = int(np.clip(xx, 0, W - tw))
            f = ocean[yy:yy + th, xx:xx + tw].mean()
            if f > bf:
                bf, best = f, (yy, xx)
        if bf >= min_ocean:
            out.append(best)
        if len(out) >= n:
            break
    return out


def coast_mask(ocean):
    """Shoreline = ocean pixels adjacent to land, given an ``(H, W)`` ocean mask."""
    return ocean & ndimage.binary_dilation(~ocean)


def coast_weight(ocean, coast, scale=8.0, floor=0.15):
    """Flattened, normalized sampling distribution that is high near the coast and
    a small ``floor`` everywhere else.

    Sampling crop centres from this oversamples the shoreline while still covering
    the open interior (the coast has more error, so it earns more crops). Pass the
    result to :func:`coast_weighted_positions`. ``scale`` is the coastal band width
    in pixels; ``floor`` sets how much the open ocean is guaranteed.
    """
    dist = ndimage.distance_transform_edt(~coast)
    w = (np.exp(-dist / scale) + floor) * ocean
    return (w / w.sum()).ravel()


def coast_weighted_positions(H, W, th, tw, n, rng, ocean, flat, min_ocean=0.25):
    """``n`` crops whose centres are drawn from the coast-weighted distribution
    ``flat`` (from :func:`coast_weight`): full coverage, shoreline oversampled."""
    out = []
    for idx in rng.choice(flat.size, size=n * 4, p=flat):
        cy, cx = divmod(int(idx), W)
        yy = int(np.clip(cy - th // 2, 0, H - th))
        xx = int(np.clip(cx - tw // 2, 0, W - tw))
        if ocean[yy:yy + th, xx:xx + tw].mean() >= min_ocean:
            out.append((yy, xx))
        if len(out) >= n:
            break
    return out


def make_crops(Xf, Yf, th, tw, sampler, seed=0):
    """Cut a stack of ``th x tw`` crops out of full frames.

    ``Xf`` is ``(D, H, W, C)`` inputs, ``Yf`` is ``(D, H, W, K)`` targets.
    ``sampler`` is a callable ``sampler(rng) -> list[(row, col)]`` called once per
    day, so a random sampler yields fresh crops each day. Returns ``(Xc, Yc)``
    float32 arrays ready for ``tf.data.Dataset.from_tensor_slices``.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for d in range(Xf.shape[0]):
        for yy, xx in sampler(rng):
            xs.append(Xf[d, yy:yy + th, xx:xx + tw])
            ys.append(Yf[d, yy:yy + th, xx:xx + tw])
    return np.stack(xs).astype("float32"), np.stack(ys).astype("float32")


def tiled_predict(model, X, tile, stride):
    """Predict a full ``(H, W, C)`` input by sliding a tile window and averaging
    the overlaps (overlap-blend removes tile-boundary seams).

    Edges are clamped so the last row/col is always covered. ``tile`` is
    ``(th, tw)``; ``stride`` is ``(sh, sw)`` (use a stride smaller than the tile to
    blend). This is the inference path for a domain too large for one forward pass.
    """
    H, W, _ = X.shape
    th, tw = tile
    sh, sw = stride
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, H - th + 1, sh)) or [0]
    xs = list(range(0, W - tw + 1, sw)) or [0]
    if ys[-1] != H - th:
        ys.append(H - th)
    if xs[-1] != W - tw:
        xs.append(W - tw)
    for y in ys:
        for x in xs:
            p = model.predict(X[y:y + th, x:x + tw][np.newaxis, ...], verbose=0)[0, :, :, 0]
            acc[y:y + th, x:x + tw] += p
            cnt[y:y + th, x:x + tw] += 1
    return acc / np.maximum(cnt, 1)
