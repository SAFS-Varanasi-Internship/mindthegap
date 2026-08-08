import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mindthegap import viz


class _Model:
    def __init__(self):
        self.last_input = None

    def predict(self, values, verbose=0):
        self.last_input = values
        shape = (*values.shape[:3], 1)
        return np.zeros(shape, dtype="float32")


def _dataset():
    coords = {
        "time": np.array(["2020-09-08"], dtype="datetime64[ns]"),
        "lat": [4.0, 3.5, 3.0],
        "lon": [10.0, 10.5, 11.0, 11.5],
    }
    shape = (1, 3, 4)
    return xr.Dataset(
        {
            "channel_b": (("time", "lat", "lon"), np.full(shape, 2.0)),
            "channel_a": (("time", "lat", "lon"), np.full(shape, 1.0)),
            "full_target": (("time", "lat", "lon"), np.full(shape, 1.0)),
            "land_flag": (("time", "lat", "lon"), np.zeros(shape)),
            "true_missing_flag": (
                ("time", "lat", "lon"),
                np.zeros(shape),
            ),
            "synthetic_missing_flag": (
                ("time", "lat", "lon"),
                np.zeros(shape),
            ),
        },
        coords=coords,
    )


def _metadata():
    return {
        "inputs": [
            {"name": "channel_a", "channel": 0},
            {"name": "channel_b", "channel": 1},
        ],
        "preprocessing": {
            "standardization": {
                "full_target": {"mean": 2.0, "std": 3.0}
            }
        },
    }


def test_map_extent_uses_pixel_edges():
    assert viz._map_extent(_dataset()) == [9.75, 11.75, 2.75, 4.25]


def test_predict_frame_uses_metadata_order_and_unstandardizes():
    model = _Model()

    prediction = viz.predict_frame(
        _dataset(),
        model,
        _metadata(),
        "2020-09-08",
    )

    assert prediction.dims == ("lat", "lon")
    assert np.all(prediction.values == 2.0)
    assert np.all(model.last_input[..., 0] == 1.0)
    assert np.all(model.last_input[..., 1] == 2.0)


def test_observed_and_flag_frames():
    dataset = _dataset()
    dataset["land_flag"][0, 0, 0] = 1
    dataset["synthetic_missing_flag"][0, 0, 1] = 1
    dataset["true_missing_flag"][0, 0, 2] = 1

    observed = viz.observed_frame(
        dataset,
        _metadata(),
        "2020-09-08",
    )
    flags = viz.flag_frame(dataset, "2020-09-08")

    assert np.all(observed.values == 5.0)
    assert flags.values[0, :4].tolist() == [0, 1, 3, 2]


def test_individual_panels_and_composite_share_prediction():
    dataset = _dataset()
    metadata = _metadata()
    model = _Model()
    date = "2020-09-08"

    figure, axes = viz.plot_prediction_observed(
        dataset,
        model,
        metadata,
        date,
    )

    assert axes.shape == (2, 2)
    assert len(axes[0, 0].images) == 1
    assert len(axes[0, 1].images) == 1
    assert len(axes[1, 0].images) == 1
    assert len(axes[1, 1].images) == 1
    assert np.all(axes[0, 0].images[0].get_array() == 5.0)
    assert np.all(axes[1, 0].images[0].get_array() == 2.0)
    assert np.all(axes[1, 1].images[0].get_array() == 3.0)
    plt.close(figure)
