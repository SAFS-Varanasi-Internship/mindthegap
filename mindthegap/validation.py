"""Validate that an :class:`~mindthegap.Options` has what a step needs.

The construction-time checks on the option dataclasses guarantee each field is
*individually* well-formed (a positive ``batch_size``, a known ``cloud_mode``,
and so on). They cannot know whether the *pipeline-resolved* pieces -- the data
configuration, the train/validation split, and the recorded standardization --
have actually been populated yet, because those are filled in later by
:meth:`~mindthegap.Options.set_up_data_options`,
:func:`~mindthegap.train_validation_dates`, and
:func:`~mindthegap.prepare_model_data`.

:func:`validate_options` is the single, reusable helper the pipeline functions
call to check exactly the parts of ``options`` they need before doing real
work. It reports *every* missing piece at once with a message that names the
option, says what it is for, and shows how to set it (with an example call),
so a user can fix the whole configuration in one pass instead of rerunning and
hitting one error at a time.

Typical use inside a pipeline function::

    from .validation import validate_options

    validate_options(options, requires=["data"])          # needs options.data
    validate_options(options, requires=["data", "split"])  # data + split

The ``requires`` names map to independently checkable requirements:

``"data"``
    ``options.data`` identifies the target and mask variables
    (``set_up_data_options`` has run).
``"data_prepared"``
    ``options.data`` is fully resolved, i.e. ``prepare_model_data(mode="train")``
    has recorded the input channel order and standardization statistics.
``"split"``
    ``options.split`` has resolved train/validation dates.
``"gridder"``
    ``options.gridder`` uses a supported method.
``"fit"``
    ``options.fit`` is present (its fields are validated on construction).
"""

from .options import Options


class OptionsValidationError(ValueError):
    """Raised when ``options`` is missing configuration a step requires.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    (and tests that match on ``ValueError``) keep working.
    """


def _check_data(options):
    """Return problems if ``options.data`` has no target/mask variables set."""
    data = options.data
    problems = []
    if data.target_variable is None:
        problems.append(
            "options.data.target_variable is not set. It names the variable in "
            "ds the model gap-fills (e.g. 'chlor_a'). Set it -- together with "
            "the cloud/land masks -- with options.set_up_data_options(ds, "
            "target='chlor_a', missing_flag='cloud_flag', land_flag='land_flag')."
        )
    if data.missing_flag is None:
        problems.append(
            "options.data.missing_flag is not set. It names the ds variable "
            "flagging missing/cloud pixels (1 = missing). Set it via "
            "options.set_up_data_options(ds, target=..., "
            "missing_flag='cloud_flag', land_flag=...)."
        )
    if data.land_flag is None:
        problems.append(
            "options.data.land_flag is not set. It names the ds variable "
            "flagging land pixels (1 = land). Set it via "
            "options.set_up_data_options(ds, target=..., missing_flag=..., "
            "land_flag='land_flag')."
        )
    return problems


def _check_data_prepared(options):
    """Return problems if ``prepare_model_data(mode='train')`` has not run."""
    data = options.data
    problems = []
    if not data.input_names:
        problems.append(
            "options.data.input_names is empty, so the model channel order is "
            "unknown. It is recorded by a training preparation pass; run "
            "ds_std = mtg.prepare_model_data(ds, options, mode='train') first "
            "(or load a trained bundle with mtg.load_model_bundle)."
        )
    if not data.standardization:
        problems.append(
            "options.data.standardization is empty, so there are no recorded "
            "statistics to reuse. Run ds_std = mtg.prepare_model_data(ds, "
            "options, mode='train') to compute them (or load a trained bundle "
            "with mtg.load_model_bundle)."
        )
    return problems


def _check_split(options):
    """Return problems if the train/validation split has not been resolved."""
    split = options.split
    if split.is_resolved():
        return []
    has_train = bool(split.train_dates)
    has_val = bool(split.val_dates)
    if has_train != has_val:
        # Partially set: one side has dates and the other does not.
        missing = "val_dates" if has_train else "train_dates"
        present = "train_dates" if has_train else "val_dates"
        return [
            f"options.split is partially set: {present} has dates but "
            f"{missing} is empty. Both are needed. Re-run "
            "mtg.set_up_train_split_options(ds, options) to choose them "
            f"together, or set options.split.{missing} to match."
        ]
    return [
        "options.split has no train/validation dates. They select which dates "
        "train the model and which validate it. Resolve them with "
        "mtg.set_up_train_split_options(ds, options) (a random split over all "
        "days; mode='train' also does this automatically), or set "
        "options.split.train_dates / options.split.val_dates manually with "
        "method='manual'."
    ]


def _check_gridder(options):
    """Return problems if the gridder method is unsupported."""
    if options.gridder.method != "xbatcher":
        return [
            f"options.gridder.method={options.gridder.method!r} is not "
            "supported; only 'xbatcher' is currently available. Set "
            "options.gridder = mtg.GridderOptions(method='xbatcher', "
            "tile_size=(64, 64)) (or run mtg.set_up_gridder_options(ds, "
            "options) for a memory-aware recommendation)."
        ]
    return []


def _check_fit(options):
    """``options.fit`` fields are validated on construction; nothing to add."""
    return []


_CHECKS = {
    "data": _check_data,
    "data_prepared": _check_data_prepared,
    "split": _check_split,
    "gridder": _check_gridder,
    "fit": _check_fit,
}


def validate_options(options, requires=("data",)):
    """Validate that ``options`` has the configuration a step requires.

    ``options`` must be a full :class:`~mindthegap.Options`. ``requires`` is an
    iterable of requirement names selecting which parts to check; see the module
    docstring for the available names (``"data"``, ``"data_prepared"``,
    ``"split"``, ``"gridder"``, ``"fit"``). Only the requested parts are
    checked, so a function can validate just what it needs (for example only
    ``options.data``, or ``options.data`` and ``options.split`` together).

    Every problem found across the requested requirements is collected and
    reported together in a single :class:`OptionsValidationError` (a
    :class:`ValueError` subclass), each with a message that names the missing
    option, explains what it is for, and shows how to set it. Returns ``options``
    unchanged when everything required is present, so callers may write
    ``options = validate_options(options, requires=[...])``.
    """
    if not isinstance(options, Options):
        raise TypeError(
            "options must be a mindthegap.Options object; got "
            f"{type(options).__name__}. Create one with mtg.Options.default()."
        )

    if isinstance(requires, str):
        requires = [requires]
    requested = list(requires)

    unknown = [name for name in requested if name not in _CHECKS]
    if unknown:
        available = ", ".join(sorted(_CHECKS))
        raise ValueError(
            f"Unknown validation requirement(s) {unknown}; choose from: "
            f"{available}."
        )

    problems = []
    for name in requested:
        problems.extend(_CHECKS[name](options))

    if problems:
        header = "options is missing configuration required for this step:"
        bullets = "\n".join(f"  - {problem}" for problem in problems)
        raise OptionsValidationError(f"{header}\n{bullets}")

    return options
