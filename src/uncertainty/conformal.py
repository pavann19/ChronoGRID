"""
Conformal Prediction engine for Data-Quality-Conditioned Conformal Prediction (DQ-CP).
Implements finite-sample Marginal Split Conformal and Simultaneous Daily Sup-Norm Conformal calibration.
"""

import math
from typing import Dict, Tuple, Optional
import numpy as np

from src.uncertainty.schemas import CalibrationTable
from src.uncertainty.normalization import DeliveryHourMADNormalizer


def compute_finite_sample_rank(n: int, alpha: float) -> Tuple[int, Optional[int]]:
    """
    Computes the standard finite-sample conformal quantile rank:
        k = ceil((n + 1) * (1 - alpha))

    Returns:
        (k_1_indexed, idx_0_indexed)
        If k > n (which occurs when alpha < 1 / (n + 1)), coverage >= 1 - alpha cannot be
        guaranteed by any finite order statistic. In this case, idx_0_indexed is None,
        indicating that an infinite quantile (unbounded interval) is mathematically required.
    """
    if n <= 0:
        raise ValueError(f"Sample size n must be positive, got {n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Miscoverage rate alpha must be in (0, 1), got {alpha}")

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        # Mathematically requires an infinite quantile; clamping to n-1 violates the finite-sample theorem
        return k, None
    idx = max(0, k - 1)
    return k, idx


class ConformalCalibrationPool:
    """
    Represents an independent calibration pool (Primary or Fallback).
    Maintains separate non-conformity scores for:
    1. Marginal Hourly Calibration: pooled 1D normalized residuals (length N_hourly).
    2. Simultaneous Daily Calibration: 1D sup-norm scores per daily forecast origin (length N_origins).
    """

    def __init__(
        self,
        model_id: str,
        normalizer: DeliveryHourMADNormalizer,
        marginal_scores: Optional[np.ndarray] = None,
        daily_sup_norm_scores: Optional[np.ndarray] = None,
        n_origins: int = 0,
        precomputed_marginal_quantiles: Optional[Dict[str, float]] = None,
        precomputed_simultaneous_quantiles: Optional[Dict[str, float]] = None
    ):
        self.model_id = model_id
        self.normalizer = normalizer
        self.marginal_scores = np.sort(np.asarray(marginal_scores, dtype=np.float64)) if marginal_scores is not None and len(marginal_scores) > 0 else np.array([])
        self.daily_sup_norm_scores = np.sort(np.asarray(daily_sup_norm_scores, dtype=np.float64)) if daily_sup_norm_scores is not None and len(daily_sup_norm_scores) > 0 else np.array([])
        self.n_origins = n_origins or len(self.daily_sup_norm_scores)
        self.precomputed_marginal_quantiles = precomputed_marginal_quantiles or {}
        self.precomputed_simultaneous_quantiles = precomputed_simultaneous_quantiles or {}

    @classmethod
    def from_calibration_table(cls, table: CalibrationTable) -> "ConformalCalibrationPool":
        """Instantiate a calibration pool directly from a serialized CalibrationTable."""
        normalizer = DeliveryHourMADNormalizer(
            hour_mad=table.hour_mad,
            global_mad=table.global_mad,
            epsilon=table.epsilon_guard
        )
        return cls(
            model_id=table.model_id,
            normalizer=normalizer,
            marginal_scores=np.array(table.marginal_scores, dtype=np.float64) if table.marginal_scores is not None else None,
            daily_sup_norm_scores=np.array(table.daily_sup_norm_scores, dtype=np.float64) if table.daily_sup_norm_scores is not None else None,
            n_origins=table.n_origins,
            precomputed_marginal_quantiles=table.marginal_quantiles,
            precomputed_simultaneous_quantiles=table.simultaneous_quantiles
        )

    @classmethod
    def from_calibration_data(
        cls,
        model_id: str,
        normalizer: DeliveryHourMADNormalizer,
        actuals: np.ndarray,
        predictions: np.ndarray,
        delivery_hours: np.ndarray,
        origin_ids: np.ndarray
    ) -> "ConformalCalibrationPool":
        """
        Fit an independent calibration pool from calibration set predictions.

        Args:
            model_id: Identifier of the model head (e.g. 'research_primary' or 'research_fallback')
            normalizer: Fitted DeliveryHourMADNormalizer
            actuals: 1D array of actual load in MW
            predictions: 1D array of point predictions in MW
            delivery_hours: 1D array of delivery hours (0..23)
            origin_ids: 1D array of daily origin identifiers (e.g. date strings or ints)
        """
        if len(actuals) != len(predictions) or len(actuals) != len(delivery_hours) or len(actuals) != len(origin_ids):
            raise ValueError("All calibration arrays must have identical length.")

        # Compute normalized non-conformity scores for every hour
        norm_scores = normalizer.normalize(actuals, predictions, delivery_hours)

        # Vector sup-norm per daily forecast origin
        df_cal = pd_df = {
            "origin": origin_ids,
            "hour": delivery_hours,
            "score": norm_scores
        }
        import pandas as pd
        df = pd.DataFrame(df_cal)
        daily_sup = df.groupby("origin")["score"].max().values

        return cls(
            model_id=model_id,
            normalizer=normalizer,
            marginal_scores=norm_scores,
            daily_sup_norm_scores=daily_sup,
            n_origins=len(daily_sup)
        )

    def get_marginal_quantile(self, alpha: float) -> Tuple[float, int]:
        """
        Compute the finite-sample pooled marginal non-conformity quantile:
            q_{1-alpha} = marginal_scores[ceil((n + 1)(1 - alpha)) - 1]
        If k > n, returns (float('inf'), k) indicating an unbounded interval.
        """
        n = len(self.marginal_scores)
        key = f"{1.0 - alpha:.2f}"
        if n == 0:
            if key in self.precomputed_marginal_quantiles:
                return float(self.precomputed_marginal_quantiles[key]), -1
            raise ValueError(f"Calibration pool '{self.model_id}' contains 0 marginal scores and no precomputed quantile for {key}.")

        k, idx = compute_finite_sample_rank(n, alpha)
        if idx is None:
            # k = n + 1: mathematically required infinite quantile
            return float("inf"), k
        q = float(self.marginal_scores[idx])
        return q, k

    def get_simultaneous_quantile(self, alpha: float) -> Tuple[float, int, bool]:
        """
        Compute vector Sup-Norm daily quantile:
            q_sim = daily_sup_norm_scores[ceil((N_origins + 1)(1 - alpha)) - 1]
        If k > n, returns (float('inf'), k, is_exploratory).

        Returns:
            (q_sim, rank_k, is_exploratory_flag)
        """
        n = len(self.daily_sup_norm_scores)
        key = f"{1.0 - alpha:.2f}"
        is_exploratory = (alpha <= 0.05 and (self.n_origins < 150))
        if n == 0:
            if key in self.precomputed_simultaneous_quantiles:
                return float(self.precomputed_simultaneous_quantiles[key]), -1, is_exploratory
            raise ValueError(f"Calibration pool '{self.model_id}' contains 0 daily origins and no precomputed quantile for {key}.")

        k, idx = compute_finite_sample_rank(n, alpha)
        if idx is None:
            # k = n + 1: mathematically required infinite quantile
            return float("inf"), k, is_exploratory
        q = float(self.daily_sup_norm_scores[idx])

        return q, k, is_exploratory

    def build_calibration_table(
        self,
        version: str,
        model_sha256: str,
        created_at: str,
        training_cutoff: str = "2024-03-31 23:00:00+00:00"
    ) -> CalibrationTable:
        """Export pool state into a serializable CalibrationTable."""
        q90_marg, k90_marg = self.get_marginal_quantile(0.10)
        q95_marg, k95_marg = self.get_marginal_quantile(0.05)
        q90_sim, k90_sim, _ = self.get_simultaneous_quantile(0.10)
        q95_sim, k95_sim, _ = self.get_simultaneous_quantile(0.05)

        return CalibrationTable(
            version=version,
            created_at=created_at,
            training_cutoff=training_cutoff,
            model_id=self.model_id,
            model_sha256=model_sha256,
            hour_mad=self.normalizer.hour_mad,
            global_mad=self.normalizer.global_mad,
            epsilon_guard=self.normalizer.epsilon,
            marginal_quantiles={"0.90": q90_marg, "0.95": q95_marg},
            simultaneous_quantiles={"0.90": q90_sim, "0.95": q95_sim},
            n_origins=self.n_origins,
            n_hourly_samples=len(self.marginal_scores),
            finite_sample_ranks={
                "marginal_0.90_k": k90_marg,
                "marginal_0.95_k": k95_marg,
                "simultaneous_0.90_k": k90_sim,
                "simultaneous_0.95_k": k95_sim
            },
            marginal_scores=[float(x) for x in self.marginal_scores],
            daily_sup_norm_scores=[float(x) for x in self.daily_sup_norm_scores]
        )
