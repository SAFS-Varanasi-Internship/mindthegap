# Re-export the primary functions (explicit, stable API)
from .data import (
    demo_data,
    crop_to_multiple,
    prepare_model_data,
    synthetic_cloud_cube,
    train_validation_dates,
    set_up_train_split,
)
from .evaluation import (
    unstdize,
    compute_mae,
    compute_mse,
)
from .model import (
    make_tf_gen,
    make_xbatcher,
    make_generator,
    fit_model,
    gapfill_std,
    UNet,
)
from .model_bundle import (
    load_model_bundle,
    load_bundle_metrics,
    save_model_bundle,
)
from .gridder import (
    set_up_gridder,
    GridderRecommendation,
)
from .validation import (
    validate_options,
    OptionsValidationError,
)
from .training import (
    train_model,
    TrainingResult,
)
from .session import set_up
from .options import (
    Options,
    DataOptions,
    GridderOptions,
    FitOptions,
    SplitOptions,
    CLOUD_MODE_OPTIONS,
    cloud_options_for,
)
from . import cloud_bank
# Expose the viz module as a submodule (lazy import by users)
from . import viz  # users can do: from mindthegap import viz; viz.plot_prediction_observed(...)

__all__ = [
    "demo_data",
    "crop_to_multiple",
    "unstdize",
    "compute_mae",
    "compute_mse",
    "make_tf_gen",
    "prepare_model_data",
    "synthetic_cloud_cube",
    "make_xbatcher",
    "unet_spatial_multiple",
    "make_generator",
    "train_validation_dates",
    "set_up_train_split",
    "fit_model",
    "gapfill_std",
    "UNet",
    "save_model_bundle",
    "load_model_bundle",
    "load_bundle_metrics",
    "set_up_gridder",
    "GridderRecommendation",
    "validate_options",
    "OptionsValidationError",
    "train_model",
    "TrainingResult",
    "set_up",
    "Options",
    "DataOptions",
    "GridderOptions",
    "FitOptions",
    "SplitOptions",
    "CLOUD_MODE_OPTIONS",
    "cloud_options_for",
    "cloud_bank",
    "viz",
]
