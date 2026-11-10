"""
Normalization engine for Data-Quality-Conditioned Conformal Prediction (DQ-CP).
Computes delivery-hour Median Absolute Deviation (MAD) scales strictly from training-period residuals.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


class DeliveryHourMADNormalizer:
    """
    Normalizes residuals by delivery-hour volatility:
        R(t, h) = |y(t, h) - y_hat(t, h)| / max(MAD_train(h), epsilon)

    MAD scales must be fitted exclusively on historical training-period residuals.
    """

    def __init__(
        self,
        hour_mad: Optional[Dict[int, float]] | None = None,
        global_mad: Optional[float] | None = None,
        epsilon: float = 1.0,
    ):
        """
        Initialize the normalizer.

        Args:
            hour_mad: Dict mapping delivery hour (0..23) to MAD scale in MW.
            global_mad: Global training residual MAD in MW (used as fallback).
            epsilon: Minimum allowable scale to prevent division by zero or near-zero.
        """
        self.hour_mad: Dict[int, float] = hour_mad or {}
        self.global_mad: float = global_mad if global_mad is not None else 500.0
        self.epsilon: float = max(1e-6, float(epsilon))

    def fit(
        self, training_residuals: np.ndarray, delivery_hours: np.ndarray
    ) -> "DeliveryHourMADNormalizer":
        """
        Fit MAD scales from training residuals.

        Args:
            training_residuals: Array of actual - predicted residuals on training data.
            delivery_hours: Array of integer delivery hours (0..23) corresponding to residuals.
        """
        if len(training_residuals) == 0:
            raise ValueError("Cannot fit normalizer on empty training residuals.")

        if len(training_residuals) != len(delivery_hours):
            raise ValueError(
                f"Length mismatch: {len(training_residuals)} residuals vs {len(delivery_hours)} hours."
            )

        res_clean = np.asarray(training_residuals, dtype=np.float64)
        hours_clean = np.asarray(delivery_hours, dtype=np.int64)

        if not np.all(np.isfinite(res_clean)):
            raise ValueError("Training residuals contain non-finite (NaN or Inf) values.")

        abs_res = np.abs(res_clean)
        self.global_mad = float(np.median(abs_res))
        if self.global_mad < self.epsilon:
            self.global_mad = self.epsilon

        df = pd.DataFrame({"abs_res": abs_res, "hour": hours_clean})
        grouped = df.groupby("hour")["abs_res"].median().to_dict()

        self.hour_mad = {}
        for h in range(24):
            val = grouped.get(h, self.global_mad)
            # Apply epsilon guard to prevent near-zero or zero scales
            if not np.isfinite(val) or val < self.epsilon:
                val = max(self.epsilon, self.global_mad)
            self.hour_mad[h] = float(val)

        return self

    def get_scale(self, delivery_hour: int) -> float:
        """Retrieve the positive, epsilon-guarded scale for a given delivery hour."""
        scale = self.hour_mad.get(int(delivery_hour), self.global_mad)
        if not np.isfinite(scale) or scale < self.epsilon:
            scale = max(self.epsilon, self.global_mad)
        return float(scale)

    def normalize(
        self, actuals: np.ndarray, predictions: np.ndarray, delivery_hours: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized non-conformity scores:
            R = |y - y_hat| / scale(h)
        """
        y = np.asarray(actuals, dtype=np.float64)
        y_hat = np.asarray(predictions, dtype=np.float64)
        h = np.asarray(delivery_hours, dtype=np.int64)

        if y.shape != y_hat.shape or y.shape != h.shape:
            raise ValueError(
                f"Shape mismatch: actuals {y.shape}, preds {y_hat.shape}, hours {h.shape}"
            )

        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(y_hat)):
            raise ValueError("Inputs contain non-finite values.")

        scales = np.array([self.get_scale(hour) for hour in h], dtype=np.float64)
        return np.abs(y - y_hat) / scales

    def scale_intervals(
        self, point_predictions: np.ndarray, delivery_hours: np.ndarray, conformal_quantile: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Invert normalized conformal quantile to absolute MW interval bounds:
            L = y_hat - q * scale(h)
            U = y_hat + q * scale(h)
        """
        y_hat = np.asarray(point_predictions, dtype=np.float64)
        h = np.asarray(delivery_hours, dtype=np.int64)
        q = float(conformal_quantile)

        if q < 0:
            raise ValueError(f"Conformal quantile must be non-negative, got {q}")

        scales = np.array([self.get_scale(hour) for hour in h], dtype=np.float64)
        half_width = q * scales

        lower_bound = y_hat - half_width
        upper_bound = y_hat + half_width

        return lower_bound, upper_bound

    def to_dict(self) -> Dict[str, Union[Dict[int, float], float]]:
        return {
            "hour_mad": {int(k): float(v) for k, v in self.hour_mad.items()},
            "global_mad": float(self.global_mad),
            "epsilon": float(self.epsilon),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeliveryHourMADNormalizer":
        hour_mad_raw = data.get("hour_mad", {})
        hour_mad = {int(k): float(v) for k, v in hour_mad_raw.items()}
        global_mad = float(data.get("global_mad", 500.0))
        epsilon = float(data.get("epsilon", 1.0))
        return cls(hour_mad=hour_mad, global_mad=global_mad, epsilon=epsilon)
