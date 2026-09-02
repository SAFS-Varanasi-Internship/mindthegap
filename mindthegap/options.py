"""Unified, serializable configuration for the gap-filling pipeline.

``Options`` is the canonical pipeline configuration and state. User-controlled
fitting choices live in :class:`GridderOptions` and :class:`FitOptions`, while
:class:`DataOptions` holds the canonical, pipeline-resolved data configuration
produced during loading and preprocessing.

The design follows normal Python conventions (dataclasses) rather than a custom
configuration mechanism, and is inspired by Icechunk's nested configuration API::

    import mindthegap as mtg

    options = mtg.Options.default()
    options.gridder = mtg.GridderOptions(tile_size=(128, 128), overlap=(16, 16))
    options.fit = mtg.FitOptions(epochs=50, batch_size=32)
    print(options)

The whole object round-trips through plain Python data for JSON/YAML::

    config = options.to_dict()
    options = mtg.Options.from_dict(config)
"""

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Optional


# The cloud parameters that apply to each ``cloud_mode``. ``cloud_seed`` is
# accepted for every mode (it makes the draw/shift reproducible). This is the
# single source of truth for which ``cloud_options`` keys are valid per mode, so
# set_up_data_options can accept one ``cloud_options`` dict and reject keys that
# do not apply to the chosen mode instead of exposing an argument per parameter.
CLOUD_MODE_OPTIONS = {
    "synthetic_bank": ("cloud_coverage", "cloud_blob_sigma", "cloud_time_sigma"),
    "synthetic": ("cloud_coverage", "cloud_blob_sigma", "cloud_time_sigma"),
    "shift": ("missing_flag_shift",),
}
# Valid for any mode.
_CLOUD_COMMON_OPTIONS = ("cloud_seed",)


def cloud_options_for(cloud_mode):
    """Return the ``cloud_options`` keys that apply to ``cloud_mode``.

    Includes the per-mode parameters plus the common ``cloud_seed``. Raises
    ``ValueError`` for an unknown mode.
    """
    if cloud_mode not in CLOUD_MODE_OPTIONS:
        valid = ", ".join(repr(m) for m in CLOUD_MODE_OPTIONS)
        raise ValueError(
            f"cloud_mode must be one of {valid}, got {cloud_mode!r}"
        )
    return CLOUD_MODE_OPTIONS[cloud_mode] + _CLOUD_COMMON_OPTIONS


# Which ``split_options`` keys are valid per split mode, so
# set_up_train_split_options can accept one ``split_options`` dict and reject
# keys that do not apply to the chosen mode instead of exposing an argument per
# parameter (mirroring CLOUD_MODE_OPTIONS).
SPLIT_MODE_OPTIONS = {
    "random": (
        "n_days",
        "train_fraction",
        "val_fraction",
        "n_train",
        "n_val",
        "min_day_difference",
    ),
    "manual": ("train_slice", "val_slice"),
}
# Valid for any mode.
_SPLIT_COMMON_OPTIONS = ("seed",)


def split_options_for(split_mode):
    """Return the ``split_options`` keys that apply to ``split_mode``.

    Includes the per-mode parameters plus the common ``seed``. Raises
    ``ValueError`` for an unknown mode.
    """
    if split_mode not in SPLIT_MODE_OPTIONS:
        valid = ", ".join(repr(m) for m in SPLIT_MODE_OPTIONS)
        raise ValueError(
            f"split_mode must be one of {valid}, got {split_mode!r}"
        )
    return SPLIT_MODE_OPTIONS[split_mode] + _SPLIT_COMMON_OPTIONS


def _to_plain(value):
    """Recursively convert a value into JSON/YAML-safe plain Python data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_plain(getattr(value, f.name)) for f in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _flatten(value, sep=".", _prefix=""):
    """Flatten nested dict data into a single dict of dotted keys.

    Lists/scalars are kept as leaf values; only nested dictionaries are
    expanded, so tracker-friendly keys such as ``"gridder.tile_size"`` map to a
    plain list rather than being split element-by-element.
    """
    flat = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{_prefix}{sep}{key}" if _prefix else str(key)
            flat.update(_flatten(item, sep=sep, _prefix=child))
    else:
        flat[_prefix] = value
    return flat


def _coerce_pair(value, name):
    """Return a two-element integer tuple, accepting scalars and sequences."""
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(
        f"{name} must be an int or a (lat, lon) pair, got {value!r}"
    )


@dataclass
class GridderOptions:
    """How the prepared dataset is split into spatial/temporal patches.

    ``tile_size`` may be an int, a ``(lat, lon)`` pair, or the string
    ``"full"``. Use ``"full"`` to treat the entire field as a single tile. It
    defaults to ``None`` meaning *unset*: no tiling has been chosen yet, so
    :func:`mindthegap.train_model` (via :func:`mindthegap.prepare_model_data`)
    sizes it automatically with :func:`mindthegap.set_up_gridder_options`. Once
    set (a pair or ``"full"``), the pipeline uses it exactly as given.
    """

    method: str = "xbatcher"
    tile_size: Optional[tuple] = None
    overlap: Optional[tuple] = None
    time_chunk: int = 100
    preload_batch: bool = False

    def __post_init__(self):
        if self.tile_size is not None and not self._is_full(self.tile_size):
            self.tile_size = _coerce_pair(self.tile_size, "tile_size")
        self.overlap = _coerce_pair(self.overlap, "overlap")
        if self.method != "xbatcher":
            raise ValueError(
                f"Unsupported gridder method {self.method!r}; "
                "only 'xbatcher' is currently supported"
            )
        if self.time_chunk <= 0:
            raise ValueError("time_chunk must be a positive integer")

    def is_resolved(self):
        """Return ``True`` once a tile size (a pair or ``"full"``) is set."""
        return self.tile_size is not None

    @staticmethod
    def _is_full(value):
        """Return True when ``value`` requests the full field as one tile."""
        return isinstance(value, str) and value.lower() == "full"

    def patch_dims(self):
        """Return the xbatcher ``input_dims`` mapping for this configuration."""
        return {
            "time": self.time_chunk,
            "lat": self.tile_size[0],
            "lon": self.tile_size[1],
        }

    def overlap_dims(self):
        """Return the xbatcher ``input_overlap`` mapping, or ``None``."""
        if self.overlap is None:
            return None
        return {"lat": self.overlap[0], "lon": self.overlap[1]}

    @classmethod
    def from_dict(cls, data):
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass
class FitOptions:
    """User-controlled model training configuration."""

    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    patience: int = 10
    loss: str = "mse"
    optimizer: str = "adam"
    shuffle_buffer: int = 512
    shuffle_seed: Optional[int] = None
    tf_seed: Optional[int] = None

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")

    @classmethod
    def from_dict(cls, data):
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass
class SplitOptions:
    """Train/validation temporal split as explicit selected dates.

    ``train_dates`` and ``val_dates`` are the resolved date selections (ISO
    strings) used to subset the standardized dataset. They are produced by
    :func:`mindthegap.train_validation_dates`, not by an implicit heuristic, so
    the split strategy is not configured until the user chooses one.
    ``method`` records how they were produced (``"random"`` or ``"manual"``)
    and also selects the strategy when :func:`mindthegap.train_validation_dates`
    is called without an explicit ``method``. ``n_days`` optionally caps the
    number of available dates used to size a random split (default ``None`` uses
    all dates), while ``train_fraction`` / ``val_fraction`` (default 80/20)
    divide those dates. ``seed`` (default ``None`` = inherit the global
    ``options.seed``) makes the random selection reproducible.
    """

    method: str = "random"
    n_days: Optional[int] = None
    train_fraction: float = 0.8
    val_fraction: float = 0.2
    train_dates: list = field(default_factory=list)
    val_dates: list = field(default_factory=list)
    min_day_difference: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.method not in ("random", "manual"):
            raise ValueError(
                f"Unsupported split method {self.method!r}; "
                "choose 'random' or 'manual'"
            )
        if self.n_days is not None and (
            not isinstance(self.n_days, int)
            or isinstance(self.n_days, bool)
            or self.n_days <= 0
        ):
            raise ValueError("n_days must be a positive integer")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if self.train_fraction + self.val_fraction > 1:
            raise ValueError(
                "train_fraction + val_fraction must not exceed 1"
            )

    def is_resolved(self):
        """Return ``True`` once train and validation dates have been chosen."""
        return bool(self.train_dates) and bool(self.val_dates)

    def train_selection(self):
        """Return a DatetimeIndex for selecting the training dates."""
        import pandas as pd

        return pd.to_datetime(self.train_dates)

    def val_selection(self):
        """Return a DatetimeIndex for selecting the validation dates."""
        import pandas as pd

        return pd.to_datetime(self.val_dates)

    def training_period(self):
        if not self.train_dates:
            return None
        dates = sorted(self.train_dates)
        return f"{dates[0]} to {dates[-1]}"

    @classmethod
    def from_dict(cls, data):
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass
class DataOptions:
    """Canonical, pipeline-resolved data configuration.

    This records the *resolved* configuration produced by loading and preparing
    the data, not the raw arguments originally supplied by the user. Downstream
    functions and saved model bundles rely on these values so the model inputs
    can be reproduced.

    Sections are empty until populated by the data-preparation pipeline; use
    :meth:`is_resolved` to check whether it has been populated.

    ``data_source`` records how the raw dataset was obtained: the exact
    ``demo_data(...)`` call when :func:`mindthegap.demo_data` was used, or the
    default ``"user manual"`` when the user loaded the data with their own
    script. It is carried into the saved model-bundle metadata and printed by
    :func:`mindthegap.load_model_bundle`.

    The synthetic-cloud configuration for ``mode="train"`` lives here so
    ``options`` remains the single source of truth for how the training data is
    created: ``cloud_mode`` selects the cloud source (``"synthetic_bank"`` --
    the default -- ``"synthetic"``, or ``"shift"``), ``cloud_coverage`` /
    ``cloud_blob_sigma`` / ``cloud_time_sigma`` parameterise the synthetic
    clouds, ``missing_flag_shift`` is used by ``cloud_mode="shift"``, and
    ``cloud_seed`` (``None`` = inherit the global ``options.seed``) makes the
    clouds reproducible. :func:`mindthegap.prepare_model_data` reads these and
    never takes them as call arguments.

    ``features`` lists *extra* variables from ``ds`` to include as model inputs;
    it must never contain the target (or its ``observed_target`` / ``full_target``
    variants). ``std_features`` selects which of those ``features`` are
    standardized. ``std_target`` selects whether the target is standardized: when
    ``True``, ``target_mean`` / ``target_std`` are computed from the masked
    ``observed_target`` over the training dates and that standardization is
    applied to ``observed_target``, its temporal lags, and ``full_target`` so the
    inputs and the training label share one consistent scale.
    """

    source: Optional[str] = None
    product_id: Optional[str] = None
    lat_bounds: Optional[tuple] = None
    lon_bounds: Optional[tuple] = None
    # target: Optional[str] = None
    # target_name: Optional[str] = None
    # target_units: Optional[str] = None
    # target_variable: Optional[str] = None
    """ CHANGE TO MAKE IT MULTIVAR OUTPUT """
    target: Optional[str] = None
    targets: list = field(default_factory=list)
    target_name: Optional[str] = None
    target_units: Optional[str] = None
    target_variable: Optional[str] = None
    target_variables: list = field(default_factory=list)
    
    missing_flag: Optional[str] = None
    land_flag: Optional[str] = None
    features: list = field(default_factory=list)
    std_features: list = field(default_factory=list)
    std_target: bool = False
    log_target: bool = False
    n_temporal_lags: int = 1
    add_geo: bool = False
    cloud_mode: str = "synthetic_bank"
    cloud_coverage: float = 0.4
    cloud_blob_sigma: float = 6.0
    cloud_time_sigma: float = 2.0
    missing_flag_shift: int = 10
    cloud_seed: Optional[int] = None
    input_names: list = field(default_factory=list)
    transforms: dict = field(default_factory=dict)
    standardization: dict = field(default_factory=dict)
    # target_mean: Optional[float] = None
    # target_std: Optional[float] = None
    """ CHANGE TO MAKE IT MULTIVAR OUTPUT """
    target_mean: Optional[float] = None
    target_std: Optional[float] = None
    target_means: dict = field(default_factory=dict)
    target_stds: dict = field(default_factory=dict)
    
    missing_value_handling: Optional[str] = None
    time_bounds: Optional[str] = None
    training_period: Optional[str] = None
    region_name: Optional[str] = None
    data_source: str = "user manual"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lat_bounds = _coerce_bounds(self.lat_bounds, "lat_bounds")
        self.lon_bounds = _coerce_bounds(self.lon_bounds, "lon_bounds")
        # self.input_names = list(self.input_names)
        # self.features = list(self.features)
        # self.std_features = list(self.std_features)
        """ CHANGE TO MAKE IT MULTIVAR OUTPUT """
        self.input_names = list(self.input_names)
        self.features = list(self.features)
        self.std_features = list(self.std_features)
        # Keep the singular/plural target fields in sync. The singular fields
        # are retained so existing callers and saved bundles keep working; the
        # plural fields are what the multi-target pipeline reads.
        self.targets = list(self.targets)
        self.target_variables = list(self.target_variables)
        if self.targets and self.target is None:
            self.target = self.targets[0]
        elif self.target and not self.targets:
            self.targets = [self.target]
        if self.target_variables and self.target_variable is None:
            self.target_variable = self.target_variables[0]
        elif self.target_variable and not self.target_variables:
            self.target_variables = [self.target_variable]
            
        if self.cloud_mode not in CLOUD_MODE_OPTIONS:
            valid = ", ".join(repr(m) for m in CLOUD_MODE_OPTIONS)
            raise ValueError(
                f"cloud_mode must be one of {valid}, got {self.cloud_mode!r}"
            )
        if not 0 <= self.cloud_coverage <= 1:
            raise ValueError("cloud_coverage must be between 0 and 1")

    def is_resolved(self):
        """Return ``True`` once the pipeline has populated the target/inputs."""
        # return self.target is not None and bool(self.input_names)
        """ CHANGE TO MAKE IT MULTIVAR OUTPUT """
        return bool(self.targets) and bool(self.input_names)

    def apply_dataset(
        self,
        ds,
        *,
        target_variable,
        missing_flag,
        land_flag,
        source=None,
        product_id=None,
        target_units=None,
        region_name=None,
        data_source="user manual",
    ):
        """Populate identity/variable configuration directly from a dataset.

        Called by :meth:`set_up_data_options` so that ``options`` -- not a
        separate metadata dict -- is the single source of truth for the variable
        names, dataset identity, spatial bounds, and time range. Standardization
        statistics and input channel order are filled in later by
        :func:`mindthegap.prepare_model_data`.
        """
        import pandas as pd

        self.source = source
        self.product_id = product_id
        self.data_source = data_source
        if region_name is not None:
            self.region_name = region_name

        # self.target_variable = target_variable
        # self.target_name = target_variable
        # self.missing_flag = missing_flag
        # self.land_flag = land_flag
        # self.target_units = (
        #     target_units
        #     if target_units is not None
        #     else ds[target_variable].attrs.get("units", "unknown")
        # )
        """ CHANGE TO MAKE IT MULTIVAR OUTPUT """
        targets = (
            [target_variable]
            if isinstance(target_variable, str)
            else list(target_variable)
        )
        self.target_variables = targets
        self.target_variable = targets[0]
        self.target_name = targets[0]
        self.missing_flag = missing_flag
        self.land_flag = land_flag
        self.target_units = (
            target_units
            if target_units is not None
            else ds[targets[0]].attrs.get("units", "unknown")
        )

        
        self.lat_bounds = (
            float(ds["lat"].min()),
            float(ds["lat"].max()),
        )
        self.lon_bounds = (
            float(ds["lon"].min()),
            float(ds["lon"].max()),
        )
        self.time_bounds = (
            f"{pd.to_datetime(ds.time.values[0]).date()} to "
            f"{pd.to_datetime(ds.time.values[-1]).date()}"
        )
        return self

    def load_from(self, ds, metadata):
        """Populate identity/variable configuration from a dataset + metadata.

        ``metadata`` is a mapping with ``dataset`` / ``target`` / ``variables``
        sections (the historical :func:`mindthegap.demo_data` return shape,
        still produced by test fixtures). Records the values that can be
        inferred from the loaded data so notebooks/tests do not have to unpack
        the metadata dict into floating variables. Standardization statistics
        and input channel order are filled in later by
        :func:`mindthegap.prepare_model_data`.
        """
        dataset_info = metadata.get("dataset", {})
        target_info = metadata.get("target", {})
        variables = metadata.get("variables", {})

        self.source = dataset_info.get("name")
        self.product_id = dataset_info.get("product_id")
        self.time_bounds = dataset_info.get("available_period")
        self.data_source = dataset_info.get("data_source", "user manual")
        if "region_name" in dataset_info:
            self.region_name = dataset_info["region_name"]

        self.target_variable = variables.get("target")
        self.target_name = target_info.get("name")
        self.missing_flag = variables.get("missing_flag")
        self.land_flag = variables.get("land_flag")
        self.features = list(variables.get("features", []))
        self.target_units = target_info.get("units")

        self.lat_bounds = (
            float(ds["lat"].min()),
            float(ds["lat"].max()),
        )
        self.lon_bounds = (
            float(ds["lon"].min()),
            float(ds["lon"].max()),
        )
        return self

    @classmethod
    def from_dict(cls, data):
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


def _coerce_bounds(value, name):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"{name} must be a (min, max) pair, got {value!r}")


@dataclass
class Options:
    """Top-level canonical pipeline configuration and state."""

    gridder: GridderOptions = field(default_factory=GridderOptions)
    fit: FitOptions = field(default_factory=FitOptions)
    split: SplitOptions = field(default_factory=SplitOptions)
    data: DataOptions = field(default_factory=DataOptions)
    verbose: bool = True
    smoke_test: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        # Materialize a concrete random global seed once at construction so a
        # "random" run is still reproducible after the fact (the value is
        # recorded on save). Passing an explicit ``seed`` pins it up front.
        if self.seed is None:
            import secrets

            self.seed = secrets.randbits(32)

    @classmethod
    def default(cls, data=None, metadata=None, smoke_test=False, seed=None):
        """Return a valid initial configuration with default fitting values.

        ``data`` starts unresolved and is populated by the pipeline. When a
        loaded dataset ``data`` (and its ``metadata``) is supplied, the
        data-dependent variable names/bounds are populated via
        :meth:`set_data_config`. The gridder is left **unset**
        (``options.gridder.tile_size is None``); it is sized automatically from
        the dataset/device by :func:`mindthegap.train_model` (via
        :func:`mindthegap.prepare_model_data`) with
        :func:`mindthegap.set_up_gridder_options`, or you may run that helper
        (or set ``options.gridder`` directly) yourself for full control. When
        ``smoke_test`` is true the run is configured to be fast: a small
        ``(16, 16)`` tile with ``time_chunk=10`` (so the tiny synthetic
        datasets used for smoke tests fit) and ``fit.epochs`` capped at 2. The
        user can still override any of these afterwards.

        ``seed`` is the single global seed inherited by the per-stage seeds
        (date sampling, ``tf.data`` shuffling, future synthetic clouds). When
        ``None`` (the default) a random integer is drawn once and stored, so
        the run is random by default yet reproducible after the fact (the value
        is recorded on save). Pass an explicit integer to pin it up front.
        TensorFlow's global RNG is *not* seeded here; call
        :meth:`seed_tensorflow` for a fully deterministic run.
        """
        options = cls(smoke_test=smoke_test, seed=seed)
        if smoke_test:
            options.gridder = replace(
                options.gridder, tile_size=(16, 16), time_chunk=10
            )
            options.fit = replace(options.fit, epochs=min(options.fit.epochs, 2))
        if data is not None:
            options.set_data_config(data=data, metadata=metadata)
        return options

    def resolved_seed(self):
        """Return the concrete global seed (always a materialized integer)."""
        return self.seed

    def resolved_split_seed(self):
        """Seed for train/validation date sampling (inherits global)."""
        return self.split.seed if self.split.seed is not None else self.seed

    def resolved_shuffle_seed(self):
        """Seed for ``tf.data`` shuffling (inherits global)."""
        return (
            self.fit.shuffle_seed
            if self.fit.shuffle_seed is not None
            else self.seed
        )

    def seed_tensorflow(self, seed):
        """Seed TensorFlow's global RNG for a deterministic training run.

        TensorFlow's global random state (model weight initialization, dropout)
        is process-global and must be set before the model is built; the layers
        used take no ``seed=`` argument. This helper is therefore the supported
        way to make TensorFlow deterministic: it applies
        ``tf.keras.utils.set_random_seed(seed)`` (so callers never import
        TensorFlow) and records the value on ``options.fit.tf_seed`` so it is
        serialized with the configuration. ``seed`` is required -- calling this
        is a deliberate opt-in; by default TensorFlow is left unseeded (random
        each run). Returns ``self`` for chaining.
        """
        if seed is None:
            raise ValueError(
                "seed_tensorflow requires an explicit integer seed"
            )
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        self.fit = replace(self.fit, tf_seed=seed)
        return self

    def set_up_data_options(
        self,
        ds,
        *,
        target,
        missing_flag,
        land_flag,
        log_target=False,
        features=(),
        std_features=(),
        std_target=False,
        add_geo=False,
        n_temporal_lags=1,
        cloud_mode=None,
        cloud_options=None,
    ):
        """Populate ``options.data`` from a loaded dataset (user-specified).

        The user specifies exactly what they want -- ``target``, ``missing_flag``
        and ``land_flag`` are required (no defaults); the remaining preprocessing
        choices are optional keyword arguments.         
        ``target`` may be a single variable name or a list of names; passing a
        list trains one model that predicts all of them jointly.
        This reads identity, spatial bounds, and time range directly from ``ds`` 
        (and its ``attrs`` written by:func:`mindthegap.demo_data`).

        Synthetic-cloud configuration is grouped rather than exposed as one
        argument per parameter. ``cloud_mode`` selects the cloud source and is
        *not* required (default ``options.data.cloud_mode`` --
        ``"synthetic_bank"``). ``cloud_options`` is an optional dict of only the
        parameters that apply to the selected mode:

        - ``"synthetic_bank"`` / ``"synthetic"``: ``cloud_coverage``,
          ``cloud_blob_sigma``, ``cloud_time_sigma`` (and ``cloud_seed``),
        - ``"shift"``: ``missing_flag_shift`` (and ``cloud_seed``).

        Passing a key that does not apply to the chosen ``cloud_mode`` raises a
        ``ValueError`` naming the valid keys; unset parameters keep their
        defaults. See :data:`mindthegap.options.CLOUD_MODE_OPTIONS`.

        This does **not** resolve the gridder or fit configuration from the
        dataset. The gridder is left unset (``options.gridder.tile_size is
        None``); it is sized automatically during
        :func:`mindthegap.prepare_model_data` (``mode="train"``) via
        :func:`mindthegap.set_up_gridder_options`, or set it yourself for full
        control. The train/validation split is also left unresolved.
        Standardization statistics and the final input channel order are filled
        in later by :func:`mindthegap.prepare_model_data`. Returns ``self`` for
        chaining.
        """
        # for name in (target, missing_flag, land_flag):
        #     if name not in ds:
        #         raise KeyError(
        #             f"{name!r} is not a variable in ds; pass variable names "
        #             "that exist in the dataset"
        #         )

        # self.data.apply_dataset(
        #     ds,
        #     target_variable=target,
        """ CHANGE TO MAKE IT MULTIVAR OUTPUT """
        targets = [target] if isinstance(target, str) else list(target)
        if not targets:
            raise ValueError("target must name at least one variable")
        for name in (*targets, missing_flag, land_flag):
            if name not in ds:
                raise KeyError(
                    f"{name!r} is not a variable in ds; pass variable names "
                    "that exist in the dataset"
                )
        self.data.apply_dataset(
            ds,
            target_variable=targets,
            missing_flag=missing_flag,
            land_flag=land_flag,
            source=ds.attrs.get("dataset_name"),
            product_id=ds.attrs.get("product_id"),
            region_name=ds.attrs.get("region_name"),
            data_source=ds.attrs.get("data_source", "user manual"),
        )
        self.data.features = list(features)
        self.data.std_features = list(std_features)
        self.data.std_target = bool(std_target)
        self.data.log_target = bool(log_target)
        self.data.add_geo = bool(add_geo)
        self.data.n_temporal_lags = int(n_temporal_lags)
        self._apply_cloud_options(cloud_mode, cloud_options)
        return self

    def _apply_cloud_options(self, cloud_mode, cloud_options):
        """Set ``cloud_mode`` and only the cloud parameters valid for it.

        ``cloud_mode`` defaults to the current ``options.data.cloud_mode``.
        ``cloud_options`` may contain only keys returned by
        :func:`cloud_options_for` for the chosen mode; anything else raises a
        ``ValueError`` so mode/parameter mismatches surface immediately.
        """
        resolved_mode = (
            self.data.cloud_mode if cloud_mode is None else cloud_mode
        )
        # Validates the mode and returns the keys that apply to it.
        allowed = cloud_options_for(resolved_mode)
        self.data.cloud_mode = resolved_mode

        options = dict(cloud_options or {})
        unknown = [key for key in options if key not in allowed]
        if unknown:
            raise ValueError(
                f"cloud_options {sorted(unknown)} do not apply to "
                f"cloud_mode={resolved_mode!r}; valid keys are "
                f"{list(allowed)}."
            )
        for key, value in options.items():
            setattr(self.data, key, value)
        # Re-run field validation (e.g. cloud_coverage range, cloud_mode).
        self.data.__post_init__()

    def set_data_config(self, data, metadata=None):
        """Populate all data-dependent configuration from a dataset.

        Populates ``options.data`` from the dataset and its loader
        ``metadata`` (variable names, identity, bounds, available period). The
        gridder/fit configuration and the train/validation split are *not*
        resolved here -- the gridder is used exactly as set on
        ``options.gridder`` (set it yourself if you want different tiling) and
        :func:`mindthegap.train_validation_dates` chooses the split. Returns
        ``self`` for chaining. Call after the dataset is loaded and cropped.
        """
        if metadata is not None:
            self.data.load_from(data, metadata)
        return self

    def to_dict(self):
        """Serialize the whole configuration to plain JSON/YAML-safe data."""
        return _to_plain(self)

    def to_flat_dict(self, sep="."):
        """Return a flat, tracker-friendly mapping of the configuration.

        Nested sections are flattened into dotted keys (for example
        ``"gridder.tile_size"`` or ``"fit.epochs"``) and scalar/list values are
        left as JSON-safe plain data, which is the shape experiment trackers
        such as MLflow / W&B expect for ``log_params``::

            mlflow.log_params(options.to_flat_dict())

        Nested dictionaries inside a section (for example the resolved
        ``data.standardization`` statistics) are flattened with the same
        separator. ``sep`` sets the key separator (default ``"."``).
        """
        return _flatten(self.to_dict(), sep=sep)

    @classmethod
    def from_dict(cls, data: dict) -> "Options":
        """Reconstruct :class:`Options` from :meth:`to_dict` output."""
        data = data or {}
        return cls(
            gridder=GridderOptions.from_dict(data.get("gridder", {})),
            fit=FitOptions.from_dict(data.get("fit", {})),
            split=SplitOptions.from_dict(data.get("split", {})),
            data=DataOptions.from_dict(data.get("data", {})),
            verbose=data.get("verbose", True),
            smoke_test=data.get("smoke_test", False),
            seed=data.get("seed"),
        )

    def __str__(self):
        return _render(self)

    __repr__ = __str__


def _render(options: "Options") -> str:
    lines = ["Options("]
    for section in fields(options):
        value = getattr(options, section.name)
        if not is_dataclass(value):
            lines.append(f"  {section.name} = {value!r}")
            continue
        lines.append(f"  {section.name}:")
        for item in fields(value):
            lines.append(f"    {item.name} = {getattr(value, item.name)!r}")
    lines.append(")")
    return "\n".join(lines)
