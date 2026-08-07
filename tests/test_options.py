import numpy as np
import pytest

from mindthegap import (
    DataOptions,
    FitOptions,
    GridderOptions,
    Options,
    build_standardized_lazy,
    demo_data,
    make_xbatcher,
)


def test_default_options_are_valid():
    options = Options.default()

    assert options.gridder.method == "xbatcher"
    assert options.gridder.tile_size == (64, 64)
    assert options.fit.epochs == 50
    assert options.fit.batch_size == 16
    assert not options.data.is_resolved()


def test_options_roundtrip_through_dict():
    options = Options.default()
    options.gridder = GridderOptions(tile_size=(128, 128), overlap=(16, 16))
    options.fit = FitOptions(epochs=5, batch_size=32)

    restored = Options.from_dict(options.to_dict())

    assert restored == options
    assert restored.gridder.tile_size == (128, 128)
    assert restored.gridder.overlap == (16, 16)
    assert restored.fit.epochs == 5


def test_to_dict_is_json_safe():
    import json

    payload = json.dumps(Options.default().to_dict())

    assert json.loads(payload)["fit"]["batch_size"] == 16


def test_gridder_derives_patch_and_overlap_dims():
    gridder = GridderOptions(tile_size=32, overlap=4, time_chunk=10)

    assert gridder.tile_size == (32, 32)
    assert gridder.patch_dims() == {"time": 10, "lat": 32, "lon": 32}
    assert gridder.overlap_dims() == {"lat": 4, "lon": 4}


def test_gridder_overlap_none_returns_none():
    assert GridderOptions().overlap_dims() is None


def test_repr_lists_sections():
    text = str(Options.default())

    assert text.startswith("Options(")
    assert "gridder:" in text
    assert "epochs = 50" in text


@pytest.mark.parametrize(
    "section, kwargs, message",
    [
        (GridderOptions, {"method": "grid"}, "Unsupported gridder"),
        (GridderOptions, {"time_chunk": 0}, "time_chunk"),
        (FitOptions, {"epochs": 0}, "epochs"),
        (FitOptions, {"batch_size": -1}, "batch_size"),
        (FitOptions, {"learning_rate": 0}, "learning_rate"),
    ],
)
def test_validation_rejects_bad_values(section, kwargs, message):
    with pytest.raises(ValueError, match=message):
        section(**kwargs)


def test_data_options_from_dict_ignores_unknown_keys():
    data = DataOptions.from_dict({"target": "full_target", "bogus": 1})

    assert data.target == "full_target"


def test_build_standardized_lazy_populates_data_options():
    ds, _ = demo_data(days=12, lat_size=8, lon_size=8, seed=5)
    options = Options.default()

    output, stats = build_standardized_lazy(
        ds,
        target_variable="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        std_vars=[],
        options=options.data,
    )

    assert options.data.is_resolved()
    assert options.data.target == "full_target"
    assert "masked_target" in options.data.input_names
    assert "full_target" not in options.data.input_names
    assert options.data.transforms["target"] == "natural logarithm"
    assert set(options.data.standardization) == set(stats)
    assert options.data.lat_bounds[0] <= options.data.lat_bounds[1]

    restored = Options.from_dict(options.to_dict())
    assert restored.data == options.data


def test_make_xbatcher_accepts_gridder_options():
    ds, _ = demo_data(days=12, lat_size=16, lon_size=16, seed=1)
    output, _ = build_standardized_lazy(
        ds,
        target_variable="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        std_vars=[],
    )
    gridder = GridderOptions(tile_size=(8, 8), time_chunk=4)

    from_options = make_xbatcher(output, options=gridder)
    explicit = make_xbatcher(
        output,
        patch_dims={"time": 4, "lat": 8, "lon": 8},
    )

    assert len(from_options) == len(explicit)


def test_make_xbatcher_requires_config():
    ds, _ = demo_data(days=4, lat_size=8, lon_size=8, seed=2)
    with pytest.raises(ValueError, match="patch_dims or options"):
        make_xbatcher(ds)
