"""Session set-up: :func:`set_up` configures the runtime and returns options.

:func:`set_up` is the first of the three main user-facing functions (with
``mtg.train_model`` and ``mtg.save_model_bundle``). It prepares the compute
environment -- quieting TensorFlow logging, enabling GPU memory growth, and
reporting the device -- and returns a fresh :class:`~mindthegap.Options`::

    options = mtg.set_up()

It does not import numpy/xarray/etc. into the caller's namespace (a function
cannot); keep those imports in your own script or notebook.
"""

import os


def _configure_tensorflow(verbose):
    """Quiet TF logging, enable GPU memory growth, and report the device.

    Returns ``(n_gpus, tf_version)`` or ``(None, None)`` when TensorFlow is not
    importable. Imported lazily so basic package use does not require the ML
    stack.
    """
    # set_up deliberately reduces TF logging, so set it explicitly (setdefault
    # would leave a louder pre-existing value in place). Must happen before
    # TensorFlow is imported to take effect.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        import tensorflow as tf
    except Exception as error:  # pragma: no cover - depends on install
        if verbose:
            print(f"TensorFlow not available ({error}); skipping GPU set-up.")
        return None, None

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            # Memory growth must be set before GPUs are initialized; ignore if
            # TensorFlow has already claimed them this session.
            pass

    if verbose:
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Compute device: {'GPU' if gpus else 'CPU'}")
        print(f"GPUs available: {len(gpus)}")
    return len(gpus), tf.__version__


def set_up(*, smoke_test=False, seed=None, verbose=True):
    """Prepare the runtime and return a fresh :class:`~mindthegap.Options`.

    This is the recommended first call in a training session. It:

    1. quiets TensorFlow's C++ logging (``TF_CPP_MIN_LOG_LEVEL``),
    2. enables GPU memory growth (so TensorFlow grows its allocation instead of
       grabbing all device memory) and prints the detected device, and
    3. returns ``mtg.Options.default(smoke_test=..., seed=...)`` -- the object
       the rest of the pipeline (``set_up_data_options``, ``set_up_gridder``,
       ``train_model``, ``save_model_bundle``) configures and consumes.

    Parameters
    ----------
    smoke_test : bool, default False
        When true, configure a fast run (small tiles/time chunk, few epochs)
        for smoke-testing the pipeline on tiny synthetic data.
    seed : int, optional
        Global seed threaded through the per-stage seeds. When ``None`` a random
        integer is drawn once and recorded, so the run is random yet reproducible
        after the fact.
    verbose : bool, default True
        Print the TensorFlow version and detected device.

    Returns
    -------
    mindthegap.Options
        A fresh, valid configuration to populate and pass downstream.

    Notes
    -----
    Being a function, ``set_up`` cannot import ``numpy``/``xarray``/``tensorflow``
    into your namespace; keep those imports in your own script or notebook.
    """
    from .options import Options

    _configure_tensorflow(verbose)
    return Options.default(smoke_test=smoke_test, seed=seed)
