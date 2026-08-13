import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mindthegap import viz, Options


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
            "unavailable_flag": (
                ("time", "lat", "lon"),
                np.zeros(shape),
            ),
            "estimate_flag": (
                ("time", "lat", "lon"),
                np.zeros(shape),
            ),
        },
        coords=coords,
    )


def _options():
    options = Options.default(seed=1)
    options.data.target = "full_target"
    options.data.input_names = ["channel_a", "channel_b"]
    options.data.standardization = {
        "full_target": {"mean": 2.0, "std": 3.0, "applied": True}
    }
    return options


def test_map_extent_uses_pixel_edges():
    assert viz._map_extent(_dataset()) == [9.75, 11.75, 2.75, 4.25]


def test_predict_frame_uses_options_order_and_unstandardizes():
    model = _Model()

    prediction = viz.predict_frame(
        _dataset(),
        model,
        _options(),
        "2020-09-08",
    )

    assert prediction.dims == ("lat", "lon")
    assert np.all(prediction.values == 2.0)
    assert np.all(model.last_input[..., 0] == 1.0)
    assert np.all(model.last_input[..., 1] == 2.0)


def test_observed_and_flag_frames():
    dataset = _dataset()
    dataset["land_flag"][0, 0, 0] = 1
    dataset["estimate_flag"][0, 0, 1] = 1
    dataset["unavailable_flag"][0, 0, 2] = 1

    observed = viz.observed_frame(
        dataset,
        _options(),
        "2020-09-08",
    )
    flags = viz.flag_frame(dataset, "2020-09-08")

    assert np.all(observed.values == 5.0)
    assert flags.values[0, :4].tolist() == [0, 1, 3, 2]


def test_individual_panels_and_composite_share_prediction():
    dataset = _dataset()
    options = _options()
    model = _Model()
    date = "2020-09-08"

    figure, axes = viz.plot_prediction_observed(
        dataset,
        model,
        options,
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


def test_land_is_overlaid_without_masking_data():
    dataset = _dataset()
    dataset["land_flag"][0, 0, 0] = 1

    image = viz.plot_observed(
        dataset,
        _options(),
        "2020-09-08",
        colorbar=False,
    )
    ax = image.axes
    # A separate land overlay image is added on top of the data image.
    assert len(ax.images) == 2
    # The data image itself is untouched (no pixels set to NaN for land/clouds),
    # so observed values stay colored and gaps remain the white background.
    assert np.all(image.get_array() == 5.0)
    assert not np.ma.is_masked(image.get_array())
    plt.close(ax.figure)


def test_no_land_overlay_when_no_land():
    dataset = _dataset()  # land_flag all zero

    image = viz.plot_observed(
        dataset,
        _options(),
        "2020-09-08",
        colorbar=False,
    )
    ax = image.axes
    assert len(ax.images) == 1
    plt.close(ax.figure)
