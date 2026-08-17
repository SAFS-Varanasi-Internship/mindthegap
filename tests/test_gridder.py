"""Tests for the memory-aware gridder recommendation (set_up_gridder_options)."""

import pytest

import mindthegap as mtg
from mindthegap.gridder import (
    GridderRecommendation,
    predicted_channels,
    estimate_tile_bytes,
    _choose_tile,
    _choose_time_chunk,
    set_up_gridder_options,
)
from mindthegap.model import unet_spatial_multiple
from conftest import make_demo_ds


def _prepared_options(ds, metadata, **data_kwargs):
    options = mtg.Options.default(data=ds, metadata=metadata)
    options.verbose = False
    options.set_up_data_options(
        ds,
        target="chlor_a",
        missing_flag="cloud_flag",
        land_flag="land_flag",
        **data_kwargs,
    )
    return options


def test_predicted_channels_base_case():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=1)
    options = _prepared_options(ds, metadata)
    # observed_target + 2 lags (m1,p1) + day_sin + day_cos + 3 flags + land = 9
    assert predicted_channels(options) == 9


def test_predicted_channels_with_geo_and_lags_and_features():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=1)
    ds = ds.assign(sst=ds["chlor_a"] * 0 + 1.0)
    options = _prepared_options(
        ds, metadata, add_geo=True, n_temporal_lags=2, features=["sst"]
    )
    # 7 base + 2*2 lags + 1 feature + 3 geo = 15
    assert predicted_channels(options) == 15


def test_predicted_channels_ds_probe_matches_static_and_is_pure():
    ds, metadata = make_demo_ds(days=20, lat_size=16, lon_size=16, seed=1)
    ds = ds.assign(sst=ds["chlor_a"] * 0 + 1.0)
    options = _prepared_options(
        ds, metadata, add_geo=True, n_temporal_lags=2, features=["sst"]
    )
    # The ds-based path defers to prepare_model_data(dry_run=True) -- the source
    # of truth -- and must agree with the static mirror without mutating options.
    assert predicted_channels(options, ds) == predicted_channels(options) == 15
    assert options.data.input_names == []
    assert not options.split.is_resolved()
    assert not options.data.standardization

def test_set_up_gridder_reports_cropped_field_shape():
    # set_up_gridder_options derives the field shape from prepare_model_data's dry-run
    # probe, which crops each axis to a multiple of the U-Net factor.
    ds, metadata = make_demo_ds(days=10, lat_size=100, lon_size=70, seed=1)
    options = _prepared_options(ds, metadata)
    rec = set_up_gridder_options(
        ds, options, gridder_options={"gpu_memory_gb": 8.0}, apply=False
    )
    multiple = unet_spatial_multiple()
    lat, lon = rec.field_shape
    assert lat % multiple == 0
    assert lon % multiple == 0
    assert (lat, lon) == (96, 64)


def test_estimate_tile_bytes_scales_with_area_and_batch():
    small = estimate_tile_bytes((64, 64), 9, 8)
    big = estimate_tile_bytes((128, 128), 9, 8)
    assert small is not None and big is not None
    # 2x each spatial dim -> ~4x the activations.
    assert big > small * 3.5
    # Doubling the batch roughly doubles the estimate.
    double_batch = estimate_tile_bytes((64, 64), 9, 16)
    assert double_batch == pytest.approx(small * 2, rel=0.01)


def test_choose_tile_prefers_whole_field_when_it_fits():
    multiple = unet_spatial_multiple()
    result = _choose_tile(
        (96, 96), 9, 20e9, batch_floor=8, min_tile=64, multiple=multiple
    )
    assert result == ((96, 96), (1, 1))


def test_choose_tile_tiles_when_field_too_big():
    multiple = unet_spatial_multiple()
    tile, n_tiles = _choose_tile(
        (1000, 800), 9, 2.8e9, batch_floor=8, min_tile=64, multiple=multiple
    )
    assert tile[0] % multiple == 0
    assert tile[1] % multiple == 0
    assert tile[0] >= 64 and tile[1] >= 64
    # The tile must actually fit the budget it was chosen for.
    assert estimate_tile_bytes(tile, 9, 8) <= 2.8e9
    assert n_tiles[0] >= 1 and n_tiles[1] >= 1


def test_choose_tile_returns_none_when_nothing_fits():
    multiple = unet_spatial_multiple()
    result = _choose_tile(
        (500, 500), 9, 1e6, batch_floor=8, min_tile=64, multiple=multiple
    )
    assert result is None


def test_choose_time_chunk_caps_at_available_time():
    chunk = _choose_time_chunk((64, 64), 9, 100e9, 40)
    assert chunk == 40


def test_choose_time_chunk_shrinks_under_ram_pressure():
    chunk = _choose_time_chunk((800, 1200), 9, 4e9, 500)
    assert 1 <= chunk < 500


def test_choose_time_chunk_targets_requested_frames():
    # A target below the RAM/available limits is used directly.
    chunk = _choose_time_chunk((64, 64), 9, 100e9, 1900, target_frames=500)
    assert chunk == 500


def test_choose_time_chunk_caps_target_at_ram_budget():
    # A large target is still capped so a block fits in RAM.
    capped = _choose_time_chunk((800, 1200), 9, 4e9, 2000, target_frames=1900)
    uncapped = _choose_time_chunk((800, 1200), 9, 4e9, 2000)
    assert capped == uncapped
    assert capped < 1900


def test_choose_time_chunk_caps_target_at_available_time():
    chunk = _choose_time_chunk((64, 64), 9, 100e9, 40, target_frames=500)
    assert chunk == 40


def test_set_up_gridder_whole_field_no_overlap():
    ds, metadata = make_demo_ds(days=40, lat_size=96, lon_size=96, seed=3)
    options = _prepared_options(ds, metadata)

    rec = set_up_gridder_options(
        ds, options, gridder_options={"gpu_memory_gb": 16.0}, apply=False
    )

    assert isinstance(rec, GridderRecommendation)
    assert rec.tile_size == (96, 96)
    assert rec.n_tiles == (1, 1)
    assert rec.overlap is None
    assert rec.batch_size == 8
    assert rec.n_channels == 9
    # apply=False must not mutate options (gridder stays unset).
    assert options.gridder.tile_size is None


def test_set_up_gridder_tiles_large_field_with_overlap():
    ds, metadata = make_demo_ds(days=40, lat_size=1000, lon_size=800, seed=3)
    options = _prepared_options(ds, metadata)

    rec = set_up_gridder_options(
        ds, options, gridder_options={"gpu_memory_gb": 4.0}, apply=False
    )

    assert rec.n_tiles != (1, 1)
    assert rec.overlap == (16, 16)
    assert rec.tile_size[0] % 8 == 0 and rec.tile_size[1] % 8 == 0


def test_set_up_gridder_apply_true_mutates_options():
    ds, metadata = make_demo_ds(days=40, lat_size=1000, lon_size=800, seed=3)
    options = _prepared_options(ds, metadata)

    rec = set_up_gridder_options(
        ds, options, gridder_options={"gpu_memory_gb": 4.0}, apply=True
    )

    assert options.gridder.tile_size == rec.tile_size
    assert options.gridder.time_chunk == rec.time_chunk
    assert options.gridder.overlap == rec.overlap
    assert options.fit.batch_size == rec.batch_size
    # The applied options must still round-trip through a dict.
    assert mtg.Options.from_dict(options.to_dict()).gridder.tile_size == (
        rec.tile_size
    )


def test_set_up_gridder_refuses_when_min_tile_cannot_fit():
    ds, metadata = make_demo_ds(days=10, lat_size=200, lon_size=200, seed=1)
    options = _prepared_options(ds, metadata)

    with pytest.raises(RuntimeError, match="batch_size"):
        set_up_gridder_options(
            ds, options, gridder_options={"gpu_memory_gb": 0.1}, apply=False
        )


def test_set_up_gridder_none_apply_defaults_to_no_apply_without_tty():
    # With no interactive stdin, apply=None must not mutate options.
    ds, metadata = make_demo_ds(days=40, lat_size=96, lon_size=96, seed=3)
    options = _prepared_options(ds, metadata)

    set_up_gridder_options(
        ds, options, gridder_options={"gpu_memory_gb": 16.0}, apply=None
    )

    assert options.gridder.tile_size is None


def test_set_up_gridder_rejects_unknown_gridder_options():
    ds, metadata = make_demo_ds(days=40, lat_size=96, lon_size=96, seed=3)
    options = _prepared_options(ds, metadata)

    with pytest.raises(ValueError, match="not recognized"):
        set_up_gridder_options(
            ds, options, gridder_options={"bogus": 1}, apply=False
        )


def test_set_up_gridder_uses_split_n_days_for_time_chunk():
    # When the split limits training to n_days, the time chunk targets that many
    # frames (here well within the RAM budget) instead of all available frames.
    ds, metadata = make_demo_ds(days=200, lat_size=96, lon_size=96, seed=3)
    options = _prepared_options(ds, metadata)
    options.split.n_days = 50

    rec = set_up_gridder_options(
        ds, options, gridder_options={"gpu_memory_gb": 16.0}, apply=False
    )

    assert rec.time_chunk == 50
