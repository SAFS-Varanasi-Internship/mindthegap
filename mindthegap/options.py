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

from dataclasses import dataclass, field, fields, is_dataclass
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
    target_units: Optional[str] = None
    input_names: list = field(default_factory=list)
    transforms: dict = field(default_factory=dict)
    standardization: dict = field(default_factory=dict)
    missing_value_handling: Optional[str] = None
    training_period: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lat_bounds = _coerce_bounds(self.lat_bounds, "lat_bounds")
        self.lon_bounds = _coerce_bounds(self.lon_bounds, "lon_bounds")
        self.input_names = list(self.input_names)

    def is_resolved(self):
        """Return ``True`` once the pipeline has populated the target/inputs."""
        return self.target is not None and bool(self.input_names)

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
    data: DataOptions = field(default_factory=DataOptions)

    @classmethod
    def default(cls):
        """Return a valid initial configuration with default fitting values.

        ``data`` starts unresolved and is populated by the pipeline.
        """
        return cls()

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
            data=DataOptions.from_dict(data.get("data", {})),
        )

    def __str__(self):
        return _render(self)

    __repr__ = __str__


def _render(options: "Options") -> str:
    lines = ["Options("]
    for section in fields(options):
        value = getattr(options, section.name)
        lines.append(f"  {section.name}:")
        for item in fields(value):
            lines.append(f"    {item.name} = {getattr(value, item.name)!r}")
    lines.append(")")
    return "\n".join(lines)
