"""High-level training entry point: :func:`train_model` and its result object.

:func:`train_model` is one of the three main user-facing functions (with
``mtg.set_up`` and ``mtg.save_model_bundle``). Given a dataset and a configured
:class:`~mindthegap.Options`, it runs the whole fit pipeline --
:func:`~mindthegap.prepare_model_data`, xbatcher streaming setup
(:func:`~mindthegap.make_generator`), building :func:`~mindthegap.UNet`, and
:func:`~mindthegap.fit_model` -- and returns a :class:`TrainingResult`.

The result object is intentionally decoupled from any experiment tracker
(MLflow, W&B, DVC, ...). It exposes exactly the pieces those tools want, so
integration is a couple of plain calls with no tracker dependency in the
package::

    result = mtg.train_model(ds, options)
    mlflow.log_params(result.options.to_flat_dict())
    mlflow.log_metrics(result.metrics)
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import datetime

from .data import prepare_model_data
from .model import make_generator, UNet, fit_model
from .validation import validate_options


@dataclass
class TrainingResult:
    """Everything a run produces, kept separate by concern.

    Attributes
    ----------
    model :
        The fitted Keras model.
    options :
        The :class:`~mindthegap.Options` used for the run. This is the requested
        run *configuration*; after ``train_model`` it is also fully resolved
        (channel order, standardization, split). Serialize it for a tracker with
        ``result.options.to_flat_dict()``.
    metadata : dict
        Resolved dataset/runtime information discovered during the run (package
        version, timestamp, field shape, channel count, steps per epoch, device,
        ...). Distinct from ``options`` (requested config) and ``metrics`` (model
        performance).
    metrics : dict
        Final model-performance metrics suitable for
        ``tracker.log_metrics(...)`` -- e.g. ``{"val_loss": ..., "val_mae":
        ...}``. Produced by an explicit post-fit ``model.evaluate`` on the
        validation data (not just the last history value). Additional
        gap-filling metrics can be merged in later without changing the shape.
    history : dict
        The full epoch-by-epoch Keras training history (``loss``, ``val_loss``,
        ``mae``, ``val_mae``, ...) for plotting training curves.
    ds_std :
        The lazy standardized dataset produced by ``prepare_model_data`` (kept so
        callers can evaluate/visualize without recomputing it).
    """

    model: Any
    options: Any
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    ds_std: Any = None

    def summary(self):
        """Return a short human-readable summary of the run."""
        lines = ["TrainingResult", "--------------"]
        version = self.metadata.get("mindthegap_version", "unknown")
        lines.append(f"  mindthegap        : {version}")
        if "device" in self.metadata:
            lines.append(f"  device            : {self.metadata['device']}")
        if "field_shape" in self.metadata:
            lat, lon = self.metadata["field_shape"]
            lines.append(f"  cropped field     : {lat} x {lon} (lat x lon)")
        if "n_channels" in self.metadata:
            lines.append(
                f"  model channels    : {self.metadata['n_channels']}"
            )
        if "epochs_run" in self.metadata:
            lines.append(f"  epochs run        : {self.metadata['epochs_run']}")
        if self.metrics:
            metric_text = ", ".join(
                f"{key}={value:.6g}" for key, value in self.metrics.items()
            )
            lines.append(f"  metrics           : {metric_text}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def _package_version():
    """Return the installed ``mindthegap`` version, or ``"unknown"``."""
    from . import __version__

    return __version__


def _detect_device():
    """Return ``"GPU"`` or ``"CPU"`` (best effort, no hard TF import failure)."""
    try:
        import tensorflow as tf

        return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    except Exception:
        return "unknown"


def _dataset_nbytes(ds):
    """Best-effort in-memory size (bytes) of ``ds`` if fully loaded."""
    try:
        return int(ds.nbytes)
    except Exception:
        return None


def _available_ram_bytes():
    """Best-effort *available* (not total) host RAM in bytes, else ``None``."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


# Loading materializes the whole standardized field plus transient copies made
# while tf.data batches it, so require comfortable headroom before auto-loading.
_LOAD_RAM_HEADROOM = 2.0


def _should_load_data(ds_std, load_data, verbose):
    """Decide whether to eagerly ``load()`` ``ds_std`` into memory.

    ``load_data`` is ``True`` (force), ``False`` (never), or ``"auto"`` (load
    only when the standardized dataset comfortably fits in available RAM).
    Returns ``True``/``False`` and prints the reason when ``verbose``.
    """
    if load_data is True:
        return True
    if load_data is False:
        return False
    if load_data != "auto":
        raise ValueError(
            f"load_data must be True, False, or 'auto'; got {load_data!r}"
        )

    size = _dataset_nbytes(ds_std)
    avail = _available_ram_bytes()
    if size is None or avail is None:
        if verbose:
            print(
                "load_data='auto': cannot estimate dataset size or available "
                "RAM; streaming from dask (pass load_data=True to force)."
            )
        return False

    fits = size * _LOAD_RAM_HEADROOM <= avail
    if verbose:
        need_gb = size * _LOAD_RAM_HEADROOM / 1e9
        print(
            f"load_data='auto': standardized data ~{size / 1e9:.1f} GB, "
            f"available RAM ~{avail / 1e9:.1f} GB "
            f"(need ~{need_gb:.1f} GB with headroom): "
            + ("loading into memory." if fits else "streaming from dask.")
        )
    return fits


def train_model(
    ds, options, *, load_data="auto", callbacks=None, verbose=None
):
    """Train a gap-filling model end to end and return a :class:`TrainingResult`.

    This is the high-level fit entry point. Given a loaded dataset ``ds`` and a
    configured :class:`~mindthegap.Options`, it:

    1. validates that ``options`` has the data configuration it needs,
    2. runs :func:`~mindthegap.prepare_model_data` (``mode="train"``) to build
       the standardized, lazy dataset and resolve the channel order,
       standardization statistics, and (if not already chosen) the
       train/validation split,
    3. builds the streaming train/validation ``tf.data`` pipelines with
       :func:`~mindthegap.make_generator`,
    4. builds the fully-convolutional :func:`~mindthegap.UNet`,
    5. fits it with :func:`~mindthegap.fit_model` (compiling with
       ``options.fit`` and EarlyStopping), and
    6. runs an explicit post-fit ``model.evaluate`` on the validation data for
       the final metrics.

    Parameters
    ----------
    ds :
        The loaded dataset with ``time``/``lat``/``lon`` dimensions and the
        target/flag variables named in ``options.data``.
    options :
        A configured :class:`~mindthegap.Options`. ``options.data`` must define
        the target and flag variables (call
        :meth:`~mindthegap.Options.set_up_data_options` first). The gridder,
        split, and fit sections are used as configured (the split is resolved
        automatically if not already set).
    load_data : {"auto", True, False}, optional
        Whether to eagerly load the standardized dataset into host memory
        before streaming it into ``tf.data``. When the whole field fits in RAM
        this replays the standardization/channel pipeline once instead of on
        every tile every epoch, which is dramatically faster (dask replay per
        batch is the usual cause of slow steps). ``"auto"`` (default) loads only
        when the dataset comfortably fits in available RAM; ``True`` forces a
        load; ``False`` always streams lazily from dask.
    callbacks : list, optional
        Keras callbacks passed through to :func:`~mindthegap.fit_model`. When
        ``None`` the default EarlyStopping (on ``val_loss``) is used.
    verbose : bool, optional
        Verbosity; defaults to ``options.verbose``.

    Returns
    -------
    TrainingResult
        With ``.model``, ``.options`` (resolved), ``.metadata``, ``.metrics``,
        ``.history``, and ``.ds_std``.
    """
    # Fail early with a clear, how-to-fix message if the data config is missing.
    validate_options(options, requires=["data"])
    if load_data not in (True, False, "auto"):
        raise ValueError(
            f"load_data must be True, False, or 'auto'; got {load_data!r}"
        )

    if verbose is None:
        verbose = options.verbose

    # 1. Standardized, lazy dataset. This also resolves the split (if needed),
    #    channel order, and standardization statistics onto options.data.
    ds_std = prepare_model_data(ds, options, mode="train")

    # 1b. Optionally materialize the standardized field once. Streaming replays
    #     the whole crop/channel/standardization graph per tile per epoch, so
    #     when the data fits in RAM a single load() makes each step a cheap
    #     in-memory slice instead.
    if _should_load_data(ds_std, load_data, verbose):
        ds_std = ds_std.load()

    # 2. Streaming train/validation pipelines (validates split + gridder).
    train_dataset, val_dataset, train_steps, val_steps = make_generator(
        ds_std, options, verbose=verbose
    )

    # 3. Model: channel count comes from the resolved configuration.
    n_channels = len(options.data.input_names)
    model = UNet(
        (None, None, n_channels),
        verbose=verbose,
        tile_size=options.gridder.tile_size,
        input_names=options.data.input_names,
        out_channels=len(options.data.targets) or 1,
    )

    # 4. Fit (fit_model compiles with options.fit + metrics=["mae"]).
    history = fit_model(
        model,
        train_dataset,
        options,
        validation_data=val_dataset,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=verbose,
    )
    history_dict = {key: list(values) for key, values in history.history.items()}

    # 5. Explicit post-fit evaluation for the final metrics. Prefer the real
    #    evaluation over the last history value; fall back to history if the
    #    evaluation pass fails for any reason.
    metrics = _evaluate_metrics(
        model, val_dataset, val_steps, history_dict, verbose=verbose
    )

    field_shape = (int(ds_std.sizes["lat"]), int(ds_std.sizes["lon"]))
    metadata = {
        "mindthegap_version": _package_version(),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": _detect_device(),
        "field_shape": field_shape,
        "n_channels": n_channels,
        "n_train_time": int(len(options.split.train_dates)),
        "n_val_time": int(len(options.split.val_dates)),
        "train_steps": int(train_steps),
        "val_steps": int(val_steps),
        "epochs_run": len(history_dict.get("loss", [])),
        "input_names": list(options.data.input_names),
        "seed": options.resolved_seed(),
    }

    result = TrainingResult(
        model=model,
        options=options,
        metadata=metadata,
        metrics=metrics,
        history=history_dict,
        ds_std=ds_std,
    )
    if verbose:
        print(result.summary())
    return result


def _evaluate_metrics(model, val_dataset, val_steps, history_dict, *, verbose):
    """Return final validation metrics as ``{"val_<name>": value}``.

    Runs an explicit ``model.evaluate`` on the validation data (the issue's
    preferred approach) and prefixes each returned metric with ``val_`` so the
    keys are unambiguous for a tracker. Falls back to the best/last epoch of the
    Keras history if the evaluation pass is unavailable.
    """
    metrics = {}
    try:
        evaluated = model.evaluate(
            val_dataset,
            steps=val_steps,
            return_dict=True,
            verbose=1 if verbose else 0,
        )
    except Exception:
        evaluated = None

    if isinstance(evaluated, dict) and evaluated:
        for name, value in evaluated.items():
            key = name if name.startswith("val_") else f"val_{name}"
            metrics[key] = float(value)
        return metrics

    # Fallback: pull the best val_loss and the matching epoch's other val_*.
    val_loss = history_dict.get("val_loss")
    if val_loss:
        import numpy as np

        best_epoch = int(np.argmin(val_loss))
        for name, values in history_dict.items():
            if name.startswith("val_") and best_epoch < len(values):
                metrics[name] = float(values[best_epoch])
    return metrics
