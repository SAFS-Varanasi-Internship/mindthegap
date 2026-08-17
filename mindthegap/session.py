"""Session set-up: :func:`set_up` configures the runtime and returns options.

:func:`set_up` is the first of the three main user-facing functions (with
``mtg.train_model`` and ``mtg.save_model_bundle``). It prepares the compute
environment -- quieting TensorFlow logging, enabling GPU memory growth, and
reporting the device -- and returns a fresh :class:`~mindthegap.Options`::

    options = mtg.set_up()

It does not import numpy/xarray/etc. into the caller's namespace (a function
cannot); keep those imports in your own script or notebook.
"""

import contextlib
import os
import sys


def _quiet_python_loggers():
    """Silence TF/absl *Python-side* log records (safe any time).

    Env vars only affect TensorFlow's native C++ logging and only when set
    before the first ``import tensorflow``; these loggers can be quieted even
    after TensorFlow is already imported.
    """
    import logging

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass


@contextlib.contextmanager
def _suppress_native_stderr(enabled):
    """Temporarily redirect the C-level ``stderr`` fd to ``os.devnull``.

    This is the only way to hide messages written straight to the native
    ``stderr`` by TensorFlow's CUDA/XLA plugins at import time (the
    ``cuFFT/cuDNN/cuBLAS factory ... already registered`` lines and the
    ``absl::InitializeLog()`` warning). It is heavy-handed -- it hides *all*
    native stderr for the duration, including genuine errors -- so it is opt-in.
    """
    if not enabled:
        yield
        return
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        # No real fd (e.g. captured stderr in tests/notebooks); nothing to do.
        yield
        return
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)


def _configure_tensorflow(verbose, quiet_native):
    """Quiet TF logging, enable GPU memory growth, and report the device.

    Returns ``(n_gpus, tf_version)`` or ``(None, None)`` when TensorFlow is not
    importable. Imported lazily so basic package use does not require the ML
    stack. When ``quiet_native`` the native-stderr plugin-registration noise
    emitted during the import is suppressed via an fd redirect.
    """
    # These only take effect when set *before* the first ``import tensorflow``.
    # Level 3 hides INFO/WARNING/ERROR from TF's own C++ logger; set explicitly
    # (not setdefault) so a louder pre-existing value is overridden.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
    try:
        with _suppress_native_stderr(quiet_native):
            import tensorflow as tf
    except Exception as error:  # pragma: no cover - depends on install
        if verbose:
            print(f"TensorFlow not available ({error}); skipping GPU set-up.")
        return None, None

    _quiet_python_loggers()

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


def set_up(*, smoke_test=False, seed=None, verbose=True, quiet_native=True):
    """Prepare the runtime and return a fresh :class:`~mindthegap.Options`.

    This is the recommended first call in a training session. It:

    1. quiets TensorFlow's C++ and Python logging (``TF_CPP_MIN_LOG_LEVEL`` plus
       the ``tensorflow``/``absl`` Python loggers),
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
    quiet_native : bool, default True
        Suppress the native-stderr CUDA/XLA plugin-registration noise emitted
        while TensorFlow is imported (the ``cuFFT/cuDNN/cuBLAS factory ...
        already registered`` lines and the ``absl::InitializeLog()`` warning) by
        redirecting the C-level ``stderr`` fd for the duration of the import.
        This hides *all* native stderr during that window, including genuine
        errors; set ``False`` to keep it. Note it only helps when ``set_up`` is
        called *before* TensorFlow is first imported -- the messages are printed
        on import, so importing TF earlier (e.g. an earlier cell) emits them
        before ``set_up`` can intercept.

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

    _configure_tensorflow(verbose, quiet_native)
    return Options.default(smoke_test=smoke_test, seed=seed)
