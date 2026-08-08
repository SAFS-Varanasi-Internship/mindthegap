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
    """How the prepared dataset is split into spatial/temporal patches."""

    method: str = "xbatcher"
    tile_size: tuple = (64, 64)
    overlap: Optional[tuple] = None
    time_chunk: int = 100
    preload_batch: bool = False
    tile_upper_limit: int = 64
    tile_multiple: int = 8

    def __post_init__(self):
        self.tile_size = _coerce_pair(self.tile_size, "tile_size")
        self.overlap = _coerce_pair(self.overlap, "overlap")
        if self.method != "xbatcher":
            raise ValueError(
                f"Unsupported gridder method {self.method!r}; "
                "only 'xbatcher' is currently supported"
            )
        if self.time_chunk <= 0:
            raise ValueError("time_chunk must be a positive integer")
    def _tile_length(self, size, chunk):
        """Largest tile length that fits the data, chunk, and cap."""
        available = min(size, chunk, self.tile_upper_limit)
        aligned = available - available % self.tile_multiple
        if aligned < self.tile_multiple:
            raise ValueError(
                f"dimension must contain at least {self.tile_multiple} cells"
            )
        return aligned

    def resolve_for(self, ds):
        """Return a copy with tile_size and time_chunk derived from ``ds``.

        Tile lengths are inferred from the dataset sizes and on-disk chunks,
        capped by ``tile_upper_limit`` and aligned to ``tile_multiple``. The
        time chunk targets roughly six blocks across the record.
        """
        chunk_sizes = getattr(ds, "chunksizes", {})

        def chunk0(dim):
            values = chunk_sizes.get(dim, (ds.sizes[dim],))
            return values[0] if values else ds.sizes[dim]

        tile_lat = self._tile_length(ds.sizes["lat"], chunk0("lat"))
        tile_lon = self._tile_length(ds.sizes["lon"], chunk0("lon"))
        time_chunk = min(100, max(1, ds.sizes["time"] // 6))
        return replace(
            self,
            tile_size=(tile_lat, tile_lon),
            time_chunk=time_chunk,
        )

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
    pixel_budget: int = 40 * 56 * 16

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")

    def resolve_for(self, tile_size):
        """Return a copy with ``batch_size`` scaled to the tile pixel budget.

        ``batch_size`` is capped so ``batch_size * tile_pixels`` stays within
        ``pixel_budget``.
        """
        pixels_per_tile = max(1, tile_size[0] * tile_size[1])
        batch_size = max(1, min(16, self.pixel_budget // pixels_per_tile))
        return replace(self, batch_size=batch_size)

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
    is called without an explicit ``method``. ``train_fraction`` /
    ``val_fraction`` (default 80/20 of the sampled dates) size the random
    split, and ``seed`` makes the random selection reproducible.
    """

    method: str = "random"
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
    """

    source: Optional[str] = None
    product_id: Optional[str] = None
    lat_bounds: Optional[tuple] = None
    lon_bounds: Optional[tuple] = None
    target: Optional[str] = None
    target_name: Optional[str] = None
    target_units: Optional[str] = None
    target_variable: Optional[str] = None
    missing_flag: Optional[str] = None
    land_flag: Optional[str] = None
    features: list = field(default_factory=list)
    log_target: bool = False
    n_temporal_lags: int = 1
    add_geo: bool = False
    input_names: list = field(default_factory=list)
    transforms: dict = field(default_factory=dict)
    standardization: dict = field(default_factory=dict)
    target_mean: Optional[float] = None
    target_std: Optional[float] = None
    missing_value_handling: Optional[str] = None
    available_period: Optional[str] = None
    training_period: Optional[str] = None
    region_name: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lat_bounds = _coerce_bounds(self.lat_bounds, "lat_bounds")
        self.lon_bounds = _coerce_bounds(self.lon_bounds, "lon_bounds")
        self.input_names = list(self.input_names)
        self.features = list(self.features)

    def is_resolved(self):
        """Return ``True`` once the pipeline has populated the target/inputs."""
        return self.target is not None and bool(self.input_names)

    def load_from(self, ds, metadata):
        """Populate identity/variable configuration from a dataset + metadata.

        ``metadata`` is the mapping returned alongside the dataset by
        :func:`mindthegap.demo_data`. This records the values that can be
        inferred from the loaded data (source identity, variable names,
        spatial bounds, available period) so notebooks do not have to unpack
        the metadata dict into floating variables. Standardization statistics
        and input channel order are filled in later by
        :func:`mindthegap.build_standardized_lazy`.
        """
        dataset_info = metadata.get("dataset", {})
        target_info = metadata.get("target", {})
        variables = metadata.get("variables", {})

        self.source = dataset_info.get("name")
        self.product_id = dataset_info.get("product_id")
        self.available_period = dataset_info.get("available_period")
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

    @classmethod
    def default(cls, data=None, metadata=None, smoke_test=False):
        """Return a valid initial configuration with default fitting values.

        ``data`` starts unresolved and is populated by the pipeline. When a
        loaded dataset ``data`` (and its ``metadata``) is supplied, the
        data-dependent sections are resolved immediately via
        :meth:`set_data_config`. When ``smoke_test`` is true the run is
        configured to be fast (``fit.epochs`` is capped at 2); the user can
        still override ``fit.epochs`` afterwards.
        """
        options = cls(smoke_test=smoke_test)
        if data is not None:
            options.set_data_config(data=data, metadata=metadata)
        return options

    def set_data_config(self, data, metadata=None):
        """Resolve all data-dependent configuration from a dataset.

        Populates ``options.data`` from the dataset and its loader
        ``metadata`` (variable names, identity, bounds, available period), then
        derives the gridder and fit sections from the dataset via
        :meth:`set_config`. The train/validation split is *not* set here; call
        :func:`mindthegap.train_validation_dates` to choose it. Returns
        ``self`` for chaining. Call after the dataset is loaded and cropped.
        """
        if metadata is not None:
            self.data.load_from(data, metadata)
        self.set_config(data)
        return self

    def set_config(self, ds):
        """Resolve dataset-derived heuristics for the non-data sections.

        Derives tile size and time chunk (``gridder``) and the batch size
        (``fit``) from the dataset so these heuristics live in the package
        rather than in notebooks. When ``smoke_test`` is set, ``fit.epochs`` is
        capped at 2 for a fast validation run. The train/validation split is
        left unresolved until the user selects one via
        :func:`mindthegap.train_validation_dates`. Returns ``self`` for
        chaining.
        """
        self.gridder = self.gridder.resolve_for(ds)
        self.fit = self.fit.resolve_for(self.gridder.tile_size)
        if self.smoke_test:
            self.fit = replace(self.fit, epochs=min(self.fit.epochs, 2))
        return self

    def to_dict(self):
        """Serialize the whole configuration to plain JSON/YAML-safe data."""
        return _to_plain(self)

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
