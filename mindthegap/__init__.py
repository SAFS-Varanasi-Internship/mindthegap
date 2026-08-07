# Re-export the primary functions (explicit, stable API)
from .utils import (
    demo_data,
    crop_to_multiple,
    unstdize,
    compute_mae,
    compute_mse,
    make_tf_gen,
    build_standardized_lazy,
    make_xbatcher,
    fit_model,
    UNet,
)
from .model_bundle import (
    create_model_bundle_metadata,
    load_model_bundle,
    save_model_bundle,
)
from .options import (
    Options,
    DataOptions,
    GridderOptions,
    FitOptions,
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
    "build_standardized_lazy",
    "make_xbatcher",
    "fit_model",
    "UNet",
    "create_model_bundle_metadata",
    "save_model_bundle",
    "load_model_bundle",
    "Options",
    "DataOptions",
    "GridderOptions",
    "FitOptions",
    "viz",
]
