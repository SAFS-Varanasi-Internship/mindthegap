"""Model evaluation and output transformation helpers."""

import numpy as np


def unstdize(stdized_image, mean, stdev):
    """Convert standardized values back to their original scale."""
    return stdized_image * stdev + mean


def compute_mae(y_true, y_pred):
    """Compute mean absolute error over pairs where neither value is NaN."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def compute_mse(y_true, y_pred):
    """Compute mean squared error over pairs where neither value is NaN."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean((y_true[mask] - y_pred[mask]) ** 2)
