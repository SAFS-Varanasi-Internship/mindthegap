"""Recommend a memory-aware gridder (tile size, time chunk, batch size).

This module provides :func:`set_up_gridder`, an *optional* helper that
recommends how the prepared dataset should be tiled for training. It sizes the
spatial tile so a training batch fits in GPU memory (using the *actual*
:func:`mindthegap.UNet` architecture to estimate activation memory), sizes the
time chunk so a dask block fits comfortably in host RAM, and suggests a batch
size and tile overlap. The recommendation is returned and (optionally) applied
to ``options.gridder`` / ``options.fit``.

The function is deliberately conservative and self-describing: it never silently
mutates the configuration unless the user accepts the suggestion, and it refuses
with a clear explanation when even the smallest sensible tile cannot fit.
"""

from dataclasses import dataclass, replace
from typing import Optional
import math
import os
import shutil
import subprocess

from .model import UNet, unet_spatial_multiple


# Bytes per element for the standardized float32 dataset the pipeline produces.
_BYTES_PER_ELEMENT = 4

# Multiplier accounting for everything the training step keeps live in GPU
# memory beyond a single forward pass of activations: stored activations for the
# backward pass, gradients, and the Adam optimizer's moment buffers. A factor of
# ~3x the summed forward activation footprint is a reasonable, slightly
# conservative estimate for this fully-convolutional U-Net.
_TRAINING_MEMORY_MULTIPLIER = 3.0

# Working-copy multiplier for host RAM: loading a dask block materializes the
# block plus transient copies during standardization / channel assembly.
_RAM_WORKING_MULTIPLIER = 3.0


@dataclass
class GridderRecommendation:
    """Suggested gridder/fit values plus the reasoning behind them."""

    tile_size: tuple
    time_chunk: int
    batch_size: int
    overlap: Optional[tuple]
    n_channels: int
    field_shape: tuple
    n_tiles: tuple
    gpu_memory_gb: float
    ram_gb: float
    device: str

    def summary(self):
        """Return a human-readable multi-line explanation of the choice."""
        lat, lon = self.field_shape
        tlat, tlon = self.tile_size
        ntl, ntn = self.n_tiles
        whole = ntl == 1 and ntn == 1
        lines = [
            "set_up_gridder recommendation",
            "-----------------------------",
            f"  device            : {self.device}",
            f"  usable GPU memory : {self.gpu_memory_gb:.1f} GB",
            f"  usable host RAM   : {self.ram_gb:.1f} GB",
            f"  cropped field     : {lat} x {lon} (lat x lon)",
            f"  model channels    : {self.n_channels}",
            f"  tile_size         : ({tlat}, {tlon})"
            + ("  [whole field, single tile]" if whole else ""),
            f"  tiles over field  : {ntl} x {ntn}"
            + ("" if whole else f" ({ntl * ntn} tiles)"),
            f"  overlap           : {self.overlap}",
            f"  time_chunk        : {self.time_chunk}",
            f"  batch_size        : {self.batch_size}",
        ]
        return "\n".join(lines)


def _query_gpu_memory_gb():
    """Return total GPU memory in GB, or ``None`` when no GPU is visible.

    Uses ``nvidia-smi`` because TensorFlow's device-details API does not expose
    total memory. Returns ``None`` on any failure so callers fall back to RAM.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    first = out.stdout.strip().splitlines()
    if not first:
        return None
    try:
        # nvidia-smi reports MiB.
        return int(first[0].strip()) * (1024 ** 2) / 1e9
    except ValueError:
        return None


def _has_visible_gpu():
    """Return ``True`` when TensorFlow can see a GPU (best effort)."""
    try:
        import tensorflow as tf

        return bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        return False


def _system_ram_gb():
    """Return total host RAM in GB."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1e9
    except (ValueError, OSError, AttributeError):
        try:
            import psutil

            return psutil.virtual_memory().total / 1e9
        except Exception:
            return 4.0


def predicted_channels(options, ds=None):
    """Predict the number of model input channels ``prepare_model_data`` yields.

    When ``ds`` is given, this defers to :func:`mindthegap.prepare_model_data`
    in ``dry_run`` mode -- the authoritative source of truth for the channel
    set -- and counts the input channels of the lazy dataset it returns
    (everything except the ``full_target`` label). This stays correct even if
    the channel assembly changes.

    Without ``ds`` it falls back to a static mirror of the channel assembly:
    the seven base channels (observed target, two seasonal channels, three
    state flags, land flag), two channels per temporal lag, any extra
    ``features``, and three spherical geo channels when ``add_geo`` is set.
    """
    if ds is not None:
        from .data import prepare_model_data

        skeleton = prepare_model_data(ds, options, mode="train", dry_run=True)
        return sum(1 for name in skeleton.data_vars if name != "full_target")

    data = options.data
    n_lags = data.n_temporal_lags or 0
    n = 7
    n += 2 * n_lags
    n += len(data.features or [])
    if data.add_geo:
        n += 3
    return n


def estimate_tile_bytes(tile_size, n_channels, batch_size, *, build_fn=None):
    """Estimate peak GPU bytes to train one batch at ``tile_size``.

    Builds the *actual* :func:`mindthegap.UNet` at ``tile_size`` and sums the
    element counts of every layer's output tensor (the forward activations),
    then scales by ``batch_size``, ``float32`` byte size, and
    ``_TRAINING_MEMORY_MULTIPLIER`` (to account for stored activations,
    gradients, and optimizer state during the backward pass). Reading the built
    model keeps the estimate correct if the architecture changes.

    The activations are counted independently of ``batch_size`` (the batch axis
    is not counted per layer) and cached per ``(tile_size, n_channels)``, so the
    expensive model build happens once per distinct tile shape even when the
    tiling search probes the same tile at several batch sizes.
    """
    per_sample = _per_sample_activation_elements(
        tuple(tile_size), int(n_channels), build_fn
    )
    if per_sample is None:
        return None
    per_sample_bytes = per_sample * _BYTES_PER_ELEMENT
    return int(per_sample_bytes * batch_size * _TRAINING_MEMORY_MULTIPLIER)


_ACTIVATION_CACHE = {}


def _per_sample_activation_elements(tile_size, n_channels, build_fn):
    """Return summed per-sample activation element count, or ``None``.

    Cached by ``(tile_size, n_channels, build_fn)`` because building the Keras
    model is the dominant cost of the tiling search.
    """
    key = (tile_size, n_channels, build_fn)
    if key in _ACTIVATION_CACHE:
        return _ACTIVATION_CACHE[key]

    fn = build_fn if build_fn is not None else UNet
    tlat, tlon = tile_size
    try:
        model = fn((tlat, tlon, n_channels), verbose=False)
    except Exception:
        _ACTIVATION_CACHE[key] = None
        return None

    activation_elements = 0
    for layer in model.layers:
        output = getattr(layer, "output", None)
        if output is None:
            continue
        # A layer may have a single output tensor or a list of them.
        outputs = output if isinstance(output, list) else [output]
        for tensor in outputs:
            shape = getattr(tensor, "shape", None)
            if shape is None:
                continue
            elements = 1
            for dim in shape:
                # The batch axis is None; count per-sample elements and scale by
                # batch_size in the caller.
                if dim is None:
                    continue
                elements *= int(dim)
            activation_elements += elements

    _ACTIVATION_CACHE[key] = activation_elements
    return activation_elements


def _round_up_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def _candidate_tilings(field_shape, min_tile, multiple):
    """Yield ``(tile_lat, tile_lon, n_lat, n_lon)`` candidate tilings.

    Enumerates integer splits of the field along each axis. For ``n`` tiles on
    an axis the tile length is ``ceil(field / n)`` rounded up to ``multiple`` and
    clamped to the field length (so a single tile always covers the whole axis).
    Only tiles at or above ``min_tile`` are yielded, so the search never returns
    an artifact-prone tiny tile.
    """
    n_lat_field, n_lon_field = field_shape

    def axis_options(length):
        options = []
        seen = set()
        for n in range(1, length + 1):
            tile = _round_up_to_multiple(math.ceil(length / n), multiple)
            tile = min(tile, length)
            if tile < min_tile and tile < length:
                # Too small to be useful; but always allow a single full-axis
                # tile even if the field itself is below min_tile.
                if n != 1:
                    continue
            if tile in seen:
                continue
            seen.add(tile)
            # Number of tiles actually needed to cover the axis with this tile.
            n_cover = int(math.ceil(length / tile))
            options.append((tile, n_cover))
        return options

    for tile_lat, n_lat in axis_options(n_lat_field):
        for tile_lon, n_lon in axis_options(n_lon_field):
            yield tile_lat, tile_lon, n_lat, n_lon


def _choose_tile(
    field_shape,
    n_channels,
    usable_gpu_bytes,
    *,
    batch_floor,
    min_tile,
    multiple,
    build_fn=None,
):
    """Pick the largest-area tile whose training batch fits GPU memory.

    Prefers the whole cropped field when it fits at ``batch_floor``. Otherwise
    searches integer tilings, keeps those that fit, and returns the one with the
    largest area, breaking ties toward the tile whose aspect ratio is closest to
    the field's. Returns ``(tile_size, n_tiles)`` or ``None`` when nothing fits.
    """
    field_lat, field_lon = field_shape
    field_aspect = field_lat / field_lon if field_lon else 1.0

    def fits(tile):
        est = estimate_tile_bytes(
            tile, n_channels, batch_floor, build_fn=build_fn
        )
        if est is None:
            # Fall back to a coarse element-count estimate when the model
            # cannot be built.
            est = (
                tile[0]
                * tile[1]
                * n_channels
                * _BYTES_PER_ELEMENT
                * batch_floor
                * _TRAINING_MEMORY_MULTIPLIER
                * 20  # rough activation blow-up for the U-Net encoder/decoder
            )
        return est <= usable_gpu_bytes

    # Whole-field first.
    whole = (field_lat, field_lon)
    if fits(whole):
        return whole, (1, 1)

    best = None
    best_key = None
    for tile_lat, tile_lon, n_lat, n_lon in _candidate_tilings(
        field_shape, min_tile, multiple
    ):
        tile = (tile_lat, tile_lon)
        if tile == whole:
            continue
        if tile_lat < min_tile or tile_lon < min_tile:
            continue
        if not fits(tile):
            continue
        area = tile_lat * tile_lon
        aspect = tile_lat / tile_lon if tile_lon else 1.0
        aspect_penalty = abs(aspect - field_aspect)
        # Larger area is better; closer aspect breaks ties (negated so the max
        # key prefers a smaller penalty).
        key = (area, -aspect_penalty)
        if best_key is None or key > best_key:
            best_key = key
            best = (tile, (n_lat, n_lon))
    return best


def _choose_time_chunk(field_shape, n_channels, usable_ram_bytes, n_time):
    """Largest time chunk whose dask block fits comfortably in host RAM.

    A block spans ``time_chunk`` frames over the whole cropped field for every
    channel. The chunk is capped at the number of available time steps.
    """
    field_lat, field_lon = field_shape
    bytes_per_frame = (
        field_lat
        * field_lon
        * n_channels
        * _BYTES_PER_ELEMENT
        * _RAM_WORKING_MULTIPLIER
    )
    if bytes_per_frame <= 0:
        return int(n_time)
    max_frames = int(usable_ram_bytes // bytes_per_frame)
    max_frames = max(1, min(max_frames, int(n_time)))
    return max_frames


def set_up_gridder(
    ds,
    options,
    *,
    gpu_memory_gb=None,
    gpu_usable=0.7,
    ram_usable=0.5,
    batch_floor=8,
    min_tile=64,
    overlap=16,
    apply=None,
    build_fn=None,
):
    """Recommend a memory-aware gridder for training and (optionally) apply it.

    This *optional* helper sizes the training tile so one batch of the actual
    :func:`mindthegap.UNet` fits in GPU memory, sizes the dask time chunk so a
    block fits comfortably in host RAM, and suggests a batch size and tile
    overlap. Call it *after* :meth:`mindthegap.Options.set_up_data_options` so
    the model channel count is known; ``options.data`` must define the target,
    flags, and feature/lag/geo choices.

    Sizing logic:

    - The spatial tile is chosen to maximize field coverage: the whole cropped
      field is used when a batch of ``batch_floor`` samples fits; otherwise the
      field is split into an integer number of tiles and the largest tile that
      fits is chosen, preferring a tile whose aspect ratio matches the field.
      ``mtg.UNet`` uses batch normalization, so ``batch_floor`` (default 8)
      keeps the batch statistics stable rather than collapsing to a single
      sample.
    - Tiles are constrained to multiples of the U-Net downsampling factor and to
      at least ``min_tile`` (default 64) on each axis, because smaller tiles
      produce severe reconstruction artifacts with realistic cloud gaps.
    - The time chunk is the largest number of frames whose full-field block fits
      ``ram_usable`` of host RAM.
    - ``overlap`` (fixed, default 16 px) is suggested only when the field is
      tiled; a single whole-field tile needs no overlap.

    ``gpu_memory_gb`` overrides the queried GPU memory (falls back to host RAM
    when no GPU is visible). ``gpu_usable`` / ``ram_usable`` are the usable
    fractions of GPU / RAM.

    When even a ``min_tile`` x ``min_tile`` tile cannot fit a ``batch_floor``
    batch, a :class:`RuntimeError` is raised explaining that the user must lower
    ``options.fit.batch_size`` (or reduce channels / tile expectations) manually
    -- the batch floor is never silently relaxed.

    ``apply`` controls whether the recommendation is written to
    ``options.gridder`` / ``options.fit``: ``None`` (default) prompts the user
    interactively, ``True`` applies without prompting, ``False`` only returns the
    recommendation. Returns a :class:`GridderRecommendation`.
    """
    multiple = unet_spatial_multiple()
    # Probe prepare_model_data (dry_run) once: it is the source of truth for the
    # cropped field shape and channel set, so the tile-fitting math stays correct
    # even if channel assembly or cropping changes. It is fast (no statistics,
    # no synthetic clouds, no split) and does not mutate ``options``.
    from .data import prepare_model_data

    skeleton = prepare_model_data(ds, options, mode="train", dry_run=True)
    n_channels = sum(
        1 for name in skeleton.data_vars if name != "full_target"
    )
    field_shape = (
        int(skeleton.sizes["lat"]),
        int(skeleton.sizes["lon"]),
    )
    n_time = int(skeleton.sizes["time"]) if "time" in skeleton.sizes else 1

    has_gpu = _has_visible_gpu()
    if gpu_memory_gb is None:
        gpu_memory_gb = _query_gpu_memory_gb()
    ram_gb = _system_ram_gb()

    if gpu_memory_gb is not None and has_gpu:
        device = "GPU"
        compute_gb = gpu_memory_gb
    else:
        # CPU-only: treat a fraction of host RAM as the compute budget.
        device = "CPU"
        compute_gb = ram_gb

    usable_gpu_bytes = compute_gb * 1e9 * gpu_usable
    usable_ram_bytes = ram_gb * 1e9 * ram_usable

    chosen = _choose_tile(
        field_shape,
        n_channels,
        usable_gpu_bytes,
        batch_floor=batch_floor,
        min_tile=min_tile,
        multiple=multiple,
        build_fn=build_fn,
    )
    if chosen is None:
        smallest = min(min_tile, field_shape[0], field_shape[1])
        raise RuntimeError(
            f"Cannot fit a {smallest}x{smallest} tile with batch_size="
            f"{batch_floor} into {compute_gb * gpu_usable:.1f} GB of usable "
            f"{device} memory ({n_channels} channels). The batch floor is not "
            "relaxed automatically; set a smaller options.fit.batch_size "
            "manually (batch normalization may be unstable below ~4), reduce "
            "the number of feature channels, or use a larger-memory device."
        )

    tile_size, n_tiles = chosen
    whole_field = n_tiles == (1, 1)
    tile_overlap = None if whole_field else (overlap, overlap)

    time_chunk = _choose_time_chunk(
        field_shape, n_channels, usable_ram_bytes, n_time
    )

    recommendation = GridderRecommendation(
        tile_size=tuple(tile_size),
        time_chunk=int(time_chunk),
        batch_size=int(batch_floor),
        overlap=tile_overlap,
        n_channels=n_channels,
        field_shape=tuple(field_shape),
        n_tiles=tuple(n_tiles),
        gpu_memory_gb=compute_gb * gpu_usable,
        ram_gb=ram_gb * ram_usable,
        device=device,
    )

    if options.verbose:
        print(recommendation.summary())

    should_apply = _resolve_apply(apply, recommendation, options)
    if should_apply:
        _apply_recommendation(options, recommendation)
        if options.verbose:
            print("Applied recommendation to options.gridder / options.fit.")
    return recommendation


def _resolve_apply(apply, recommendation, options):
    """Decide whether to apply the recommendation.

    ``True``/``False`` are used directly. ``None`` prompts the user; when stdin
    is not interactive the prompt defaults to *not* applying so the call is
    non-destructive in scripts/notebooks-without-input.
    """
    if apply is True:
        return True
    if apply is False:
        return False
    try:
        response = input(
            "Apply these gridder settings to options? [y/N]: "
        )
    except (EOFError, OSError):
        return False
    return response.strip().lower() in ("y", "yes")


def _apply_recommendation(options, recommendation):
    """Write the recommendation onto ``options.gridder`` / ``options.fit``."""
    options.gridder.tile_size = recommendation.tile_size
    options.gridder.time_chunk = recommendation.time_chunk
    if recommendation.overlap is not None:
        options.gridder.overlap = recommendation.overlap
    options.fit = replace(options.fit, batch_size=recommendation.batch_size)
