# Re-export the primary functions (explicit, stable API)
from .create_zarr import create_zarr, data_preprocessing, data_preprocessing_new
from .utils import (
    crop_to_multiple,
    unstdize,
    compute_mae,
    compute_mse,
    make_tf_gen,
    build_standardized_lazy,
    make_xbatcher,
    UNet,
)
from .model_bundle import (
    ModelBundle,
    save_model_bundle,
    load_model_bundle,
)
# Expose the viz module as a submodule (lazy import by users)
from . import viz  # users can do: from mindthegap import viz; viz.plot_prediction_observed(...)

__all__ = [
    "create_zarr",
    "data_preprocessing",
    "data_preprocessing_new",
    "crop_to_multiple",
    "unstdize",
    "compute_mae",
    "compute_mse",
    "make_tf_gen",
    "build_standardized_lazy",
    "make_xbatcher",
    "UNet",
    "ModelBundle",
    "save_model_bundle",
    "load_model_bundle",
    "viz",
]
