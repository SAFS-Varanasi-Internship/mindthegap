# Re-export the primary functions (explicit, stable API)
from .utils import (
    crop_to_multiple,
    unstdize,
    compute_mae,
    compute_mse,
    make_tf_gen,
    build_standardized_lazy,
    build_standardized_lazy_new,
    make_xbatcher,
    UNet,
)
# Spatial patching + tiled inference
from .patching import (
    grid_positions,
    random_positions,
    coast_mask,
    coast_positions,
    coast_weight,
    coast_weighted_positions,
    make_crops,
    tiled_predict,
)
# Self-supervised gap-fill training pieces (masked loss, fake clouds, splits, metric)
from .selfsup import (
    masked_mse,
    masked_mae,
    target_with_mask,
    keep_cloud_blobs,
    synthetic_cloud_cube,
    build_pace_channels,
    season_of,
    season_blocked_split,
    contiguous_split,
    fake_cloud_mae,
)
# Expose the viz module as a submodule (lazy import by users)
from . import viz  # users can do: from mindthegap import viz; viz.plot_prediction_observed(...)

__all__ = [
    "crop_to_multiple",
    "unstdize",
    "compute_mae",
    "compute_mse",
    "make_tf_gen",
    "build_standardized_lazy",
    "build_standardized_lazy_new",
    "make_xbatcher",
    "UNet",
    # patching
    "grid_positions",
    "random_positions",
    "coast_mask",
    "coast_positions",
    "coast_weight",
    "coast_weighted_positions",
    "make_crops",
    "tiled_predict",
    # selfsup
    "masked_mse",
    "masked_mae",
    "target_with_mask",
    "keep_cloud_blobs",
    "synthetic_cloud_cube",
    "build_pace_channels",
    "season_of",
    "season_blocked_split",
    "contiguous_split",
    "fake_cloud_mae",
    "viz",
]
