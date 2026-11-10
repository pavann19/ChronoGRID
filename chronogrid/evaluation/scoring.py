"""Interval scoring for retained numerical regression fixtures."""

import numpy as np


def compute_winkler_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Computes Winkler Score (interval score) for central (1 - alpha) prediction intervals:
        S_alpha = (upper - lower) + (2/alpha)*(lower - y)*I(y < lower) + (2/alpha)*(y - upper)*I(y > upper)
    """
    width = upper - lower
    under_penalty = np.where(y_true < lower, (2.0 / alpha) * (lower - y_true), 0.0)
    over_penalty = np.where(y_true > upper, (2.0 / alpha) * (y_true - upper), 0.0)
    return width + under_penalty + over_penalty
