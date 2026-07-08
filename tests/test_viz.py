import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mindthegap.viz import _map_extent, plot_prediction_observed


class _DummyModel:
    def predict(self, x, verbose=0):
        return np.zeros((x.shape[0], x.shape[1], x.shape[2], 1), dtype=float)


class _DummyAxis:
    def __init__(self):
        self.imshow_calls = []
        self.set_extent_calls = []
        self.set_box_aspect_calls = []
        self.set_xticks_calls = []
        self.set_yticks_calls = []

    def imshow(self, *args, **kwargs):
        self.imshow_calls.append(kwargs)
        return object()

    def add_feature(self, *args, **kwargs):
        return None

    def set_extent(self, *args, **kwargs):
        self.set_extent_calls.append((args, kwargs))
        return None

    def set_box_aspect(self, *args, **kwargs):
        self.set_box_aspect_calls.append((args, kwargs))
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_xticks(self, *args, **kwargs):
        self.set_xticks_calls.append((args, kwargs))
        return None

    def set_yticks(self, *args, **kwargs):
        self.set_yticks_calls.append((args, kwargs))
        return None

    def set_title(self, *args, **kwargs):
        return None


class _DummyColorbarAxis:
    def set_ylabel(self, *args, **kwargs):
        return None


class _DummyFig:
    def add_axes(self, *args, **kwargs):
        return _DummyColorbarAxis()

    def colorbar(self, *args, **kwargs):
        return type("Colorbar", (), {"ax": _DummyColorbarAxis()})()

    def subplots_adjust(self, *args, **kwargs):
        return None


def test_map_extent_uses_pixel_edges():
    ds = xr.Dataset(coords={"lon": [10.0, 10.5, 11.0, 11.5], "lat": [4.0, 3.5, 3.0]})

    assert _map_extent(ds) == [9.75, 11.75, 2.75, 4.25]


def test_plot_prediction_observed_uses_coordinate_extent(monkeypatch, tmp_path):
    time = np.array(["2020-09-08"], dtype="datetime64[ns]")
    coords = {"time": time, "lat": [4.0, 3.5, 3.0], "lon": [10.0, 10.5, 11.0, 11.5]}
    shape = (1, 3, 4)

    zarr_stdized = xr.Dataset(
        data_vars={
            "feature": (("time", "lat", "lon"), np.zeros(shape)),
            "CHL": (("time", "lat", "lon"), np.zeros(shape)),
            "land_flag": (("time", "lat", "lon"), np.zeros(shape)),
            "real_cloud_flag": (("time", "lat", "lon"), np.zeros(shape)),
            "valid_CHL_flag": (("time", "lat", "lon"), np.ones(shape)),
            "fake_cloud_flag": (("time", "lat", "lon"), np.zeros(shape)),
        },
        coords=coords,
    )
    zarr_ds = xr.Dataset(
        data_vars={"CHL_cmes-level3": (("time", "lat", "lon"), np.ones(shape))},
        coords=coords,
    )

    np.save(tmp_path / "demo.npy", {"CHL": np.array([0.0, 1.0])})

    axes = np.array([[_DummyAxis(), _DummyAxis()], [_DummyAxis(), _DummyAxis()]])
    fig = _DummyFig()

    monkeypatch.setattr("mindthegap.viz.plt.subplots", lambda *args, **kwargs: (fig, axes))
    monkeypatch.setattr("mindthegap.viz.plt.show", lambda: None)

    plot_prediction_observed(zarr_stdized, zarr_ds, "demo", _DummyModel(), time[0], datadir=tmp_path)

    expected_extent = [9.75, 11.75, 2.75, 4.25]
    assert axes[0, 0].imshow_calls[0]["extent"] == expected_extent
    assert axes[0, 1].imshow_calls[0]["extent"] == expected_extent
    assert axes[0, 0].set_extent_calls[0][0][0] == expected_extent
    assert axes[0, 1].set_extent_calls[0][0][0] == expected_extent
    assert axes[0, 0].set_box_aspect_calls[0][0][0] == 0.75
    assert axes[0, 1].set_box_aspect_calls[0][0][0] == 0.75
    assert np.all((axes[0, 0].set_xticks_calls[0][0][0] >= expected_extent[0]) & (axes[0, 0].set_xticks_calls[0][0][0] <= expected_extent[1]))
    assert np.all((axes[0, 0].set_yticks_calls[0][0][0] >= expected_extent[2]) & (axes[0, 0].set_yticks_calls[0][0][0] <= expected_extent[3]))
