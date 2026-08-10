# Re-export the primary functions (explicit, stable API)
from .data import (
    demo_data,
    crop_to_multiple,
    prepare_model_data,
    train_validation_dates,
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
    create_model_bundle_metadata,
    load_model_bundle,
    options_from_bundle,
    save_model_bundle,
)
from .options import (
    Options,
    DataOptions,
    GridderOptions,
    FitOptions,
    SplitOptions,
)
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
    "make_xbatcher",
    "unet_spatial_multiple",
    "make_generator",
    "train_validation_dates",
    "fit_model",
    "gapfill_std",
    "UNet",
    "create_model_bundle_metadata",
    "save_model_bundle",
    "load_model_bundle",
    "options_from_bundle",
    "Options",
    "DataOptions",
    "GridderOptions",
    "FitOptions",
    "SplitOptions",
    "viz",
]
