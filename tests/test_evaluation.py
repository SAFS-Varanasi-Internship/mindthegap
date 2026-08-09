import numpy as np

from mindthegap.evaluation import compute_mae, compute_mse, unstdize


def test_unstdize_restores_original_scale():
    values = np.array([-1.0, 0.0, 1.0])

    np.testing.assert_array_equal(
        unstdize(values, mean=2.0, stdev=3.0),
        np.array([-1.0, 2.0, 5.0]),
    )


def test_metrics_ignore_nan_pairs():
    actual = np.array([1.0, 2.0, np.nan, 4.0])
    predicted = np.array([2.0, np.nan, 3.0, 2.0])

    assert compute_mae(actual, predicted) == 1.5
    assert compute_mse(actual, predicted) == 2.5
