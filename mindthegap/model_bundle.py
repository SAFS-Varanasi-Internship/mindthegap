"""Portable Keras model bundles for local and hosted inference.

A bundle is a directory containing:

``model.keras``
    The trained Keras model.
``options.json``
    The complete resolved :class:`~mindthegap.Options` (:meth:`Options.to_dict`
    serialized to JSON). ``options`` is the single source of truth for how the
    model was set up -- variable names, features, transforms, temporal lags,
    standardization statistics, channel order, split, and fit configuration.
``README.md``
    A human-readable model card generated from ``options`` and the repository
    source (git) information.
``make_dataset.py`` (optional)
    A verbatim record of the script/code the user ran to create the ``ds``
    passed to :func:`mindthegap.prepare_model_data`. ``options`` records every
    setting *except* how the raw dataset was built, so this file preserves that
    last piece for reproducibility.
``metrics.json`` (optional)
    The run's ``metrics``, ``metadata``, and ``history`` when a
    :class:`~mindthegap.TrainingResult` (from :func:`mindthegap.train_model`) is
    saved. Written for experiment tracking; the model and options load without
    it.
"""

import json
from pathlib import Path
import subprocess


MODEL_FILENAME = "model.keras"
OPTIONS_FILENAME = "options.json"
README_FILENAME = "README.md"
DATASET_SCRIPT_FILENAME = "make_dataset.py"
METRICS_FILENAME = "metrics.json"


def _git_value(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _source_metadata():
    repository = _git_value("remote", "get-url", "origin")
    if repository.startswith("git@github.com:"):
        repository = "https://github.com/" + repository.removeprefix(
            "git@github.com:"
        )
    if repository.endswith(".git"):
        repository = repository[:-4]
    return {
        "repository": repository,
        "git_commit": _git_value("rev-parse", "HEAD"),
    }


def _bounds_text(bounds):
    if not bounds:
        return "unknown"
    return f"{bounds[0]} to {bounds[1]}"


def _model_card(model_name, options, source, limitations, has_dataset_script):
    data = options.data
    inputs = "\n".join(
        f"- Channel {channel}: `{name}`"
        for channel, name in enumerate(data.input_names)
    )
    training_period = (
        options.split.training_period() or "the documented training period"
    )
    dataset_note = (
        "The exact script used to build the training dataset is recorded in "
        f"`{DATASET_SCRIPT_FILENAME}`."
        if has_dataset_script
        else "The raw dataset was created by the user; see "
        f"`{OPTIONS_FILENAME}` (`data.data_source`) for how it was obtained."
    )
    return f"""---
library_name: keras
tags:
  - gap-filling
  - ocean-color
---

# {model_name}

Keras model predicting **{data.target_name or data.target_variable}** for
**{data.source or 'the documented dataset'}**.

## Intended use

This model is intended for latitude {_bounds_text(data.lat_bounds)}, longitude
{_bounds_text(data.lon_bounds)} during {training_period}.

## Inputs

{inputs}

## Preprocessing

The complete resolved configuration is recorded in `{OPTIONS_FILENAME}`. Rebuild
it with `mindthegap.load_model_bundle(path)`, which returns the model and the
`Options` object, then run `mindthegap.prepare_model_data(ds, options,
mode="gapfill")` to reproduce the model inputs exactly.

Data source: `{data.data_source}`

{dataset_note}

## Limitations

{limitations}

## Source

Repository: {source['repository']}

Git commit: `{source['git_commit']}`
"""


def _resolve_dataset_script(dataset_script):
    """Return the source text to write to ``make_dataset.py``.

    ``dataset_script`` may be a path to an existing ``.py`` file (its contents
    are copied) or a string of Python source code (written verbatim).
    """
    if dataset_script is None:
        return None
    candidate = Path(dataset_script)
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        return candidate.read_text(encoding="utf-8")
    return str(dataset_script)


def save_model_bundle(
    model,
    path,
    options=None,
    *,
    model_name=None,
    limitations=None,
    dataset_script=None,
    overwrite=False,
):
    """Save a Keras model bundle with its resolved options as the source of truth.

    Writes ``model.keras``, ``options.json`` (the full resolved
    :class:`~mindthegap.Options`), and a generated ``README.md`` model card into
    ``path``. When ``dataset_script`` is provided it is written to
    ``make_dataset.py`` as a verbatim record of how the training dataset was
    created (``options`` records everything else).

    Parameters
    ----------
    model : keras.Model or mindthegap.TrainingResult
        The trained model to save, or the :class:`~mindthegap.TrainingResult`
        returned by :func:`mindthegap.train_model`. When a ``TrainingResult`` is
        passed, its ``.model`` and ``.options`` are used (so ``options`` may be
        omitted), and its ``.metrics``, ``.metadata``, and ``.history`` are
        written to ``metrics.json`` in the bundle for experiment tracking.
    path : str or Path
        Bundle directory to create.
    options : mindthegap.Options, optional
        The resolved configuration (must have been run through
        :func:`mindthegap.prepare_model_data` in ``mode="train"`` so the data
        section is fully populated). Required unless ``model`` is a
        ``TrainingResult`` (which already carries it); if both are given, the
        explicit ``options`` overrides the result's.
    model_name : str, optional
        Human-readable name for the model card. Defaults to a name derived from
        the dataset source.
    limitations : str, optional
        Free-text limitations for the model card.
    dataset_script : str or Path, optional
        Path to a ``.py`` file, or a string of Python source, recording how the
        raw dataset was built. Copied verbatim into ``make_dataset.py``.
    overwrite : bool, default False
        Overwrite an existing bundle.
    """
    from .validation import validate_options
    from .training import TrainingResult

    # Accept a TrainingResult directly: pull the fitted model, the resolved
    # options (unless explicitly overridden), and the run's metrics/metadata/
    # history so they can be persisted alongside the model.
    run_metrics = None
    if isinstance(model, TrainingResult):
        result = model
        model = result.model
        if options is None:
            options = result.options
        run_metrics = {
            "metrics": result.metrics,
            "metadata": result.metadata,
            "history": result.history,
        }

    if options is None:
        raise ValueError(
            "options is required: pass the resolved mindthegap.Options, or a "
            "mindthegap.TrainingResult (from mtg.train_model) that carries it."
        )

    # A saved bundle must carry the resolved channel order/standardization so it
    # can be reloaded and reused; validate_options explains how to produce them.
    validate_options(options, requires=["data_prepared"])

    bundle_path = Path(path)
    model_path = bundle_path / MODEL_FILENAME
    options_path = bundle_path / OPTIONS_FILENAME
    readme_path = bundle_path / README_FILENAME
    script_path = bundle_path / DATASET_SCRIPT_FILENAME
    metrics_path = bundle_path / METRICS_FILENAME

    tracked = [model_path, options_path, readme_path]
    if dataset_script is not None:
        tracked.append(script_path)
    if run_metrics is not None:
        tracked.append(metrics_path)
    existing = [file.name for file in tracked if file.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Bundle files already exist at {bundle_path}: "
            + ", ".join(existing)
            + "; use overwrite=True"
        )

    bundle_path.mkdir(parents=True, exist_ok=True)

    with options_path.open("w", encoding="utf-8") as file:
        json.dump(options.to_dict(), file, indent=2, sort_keys=False)

    if run_metrics is not None:
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(run_metrics, file, indent=2, sort_keys=False)

    resolved_name = model_name or (
        f"{options.data.source} U-Net gap filler"
        if options.data.source
        else "U-Net gap filler"
    )
    resolved_limitations = limitations or (
        "Use only with data matching the documented variables, domain, and "
        "preprocessing."
    )
    script_source = _resolve_dataset_script(dataset_script)
    readme_path.write_text(
        _model_card(
            resolved_name,
            options,
            _source_metadata(),
            resolved_limitations,
            has_dataset_script=script_source is not None,
        ),
        encoding="utf-8",
    )
    if script_source is not None:
        script_path.write_text(script_source, encoding="utf-8")

    model.save(model_path)
    return bundle_path


def load_model_bundle(path, compile=False):
    """Load a model bundle and return ``(model, options)``.

    ``options`` is reconstructed from ``options.json`` and is the full resolved
    :class:`~mindthegap.Options` used to train the model. Replay it through
    :func:`mindthegap.prepare_model_data` in ``mode="gapfill"`` to reproduce the
    model inputs exactly.
    """
    from .options import Options

    bundle_path = Path(path)
    required = (
        bundle_path / MODEL_FILENAME,
        bundle_path / OPTIONS_FILENAME,
        bundle_path / README_FILENAME,
    )
    missing = [file.name for file in required if not file.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete model bundle at {bundle_path}; missing: "
            + ", ".join(missing)
        )

    with (bundle_path / OPTIONS_FILENAME).open(encoding="utf-8") as file:
        options = Options.from_dict(json.load(file))

    import keras

    model = keras.models.load_model(
        bundle_path / MODEL_FILENAME,
        compile=compile,
    )
    # Keras defaults uncompiled GPU models to XLA JIT, which is not reliable
    # with this U-Net/cuDNN combination.
    if not compile:
        model.jit_compile = False

    print(f"Data loaded for this model with: {options.data.data_source}")
    return model, options
