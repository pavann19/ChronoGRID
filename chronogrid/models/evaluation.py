"""
Time Series Forecasting Evaluation Metrics (P5.4, P5.5, P5.6).
Computes MAE, RMSE, MAPE, sMAPE, R2, Peak-Hour Error, Bootstrap CIs, and Diebold-Mariano test.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray | None = None
) -> Dict[str, Any]:
    """
    Computes comprehensive energy forecasting metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Avoid division by zero in MAPE
    nonzero_mask = np.abs(y_true) > 1e-5
    yt = y_true[nonzero_mask]
    yp = y_pred[nonzero_mask]

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)

    # Symmetric MAPE (sMAPE)
    smape = float(np.mean(2.0 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp))) * 100.0)

    # R-squared
    r2 = float(r2_score(y_true, y_pred))

    metrics: dict[str, Any] = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 3),
        "smape": round(smape, 3),
        "r2": round(r2, 4),
    }

    # Peak hour analysis (German grid peak hours: 11:00 and 19:00 UTC)
    if timestamps is not None and len(timestamps) == len(y_true):
        ts_hours = pd.to_datetime(timestamps).hour
        peak_mask = (ts_hours == 11) | (ts_hours == 19)
        if peak_mask.any():
            metrics["peak_hours_mae"] = round(
                float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])), 2
            )

    return metrics


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% bootstrap confidence intervals for MAE and RMSE (ML Best Practices rule).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    mae_samples = np.empty(n_bootstrap)
    rmse_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        mae_samples[i] = mean_absolute_error(sample_true, sample_pred)
        rmse_samples[i] = np.sqrt(mean_squared_error(sample_true, sample_pred))

    alpha = (1.0 - confidence_level) / 2.0
    mae_ci = (
        float(np.percentile(mae_samples, alpha * 100)),
        float(np.percentile(mae_samples, (1.0 - alpha) * 100)),
    )
    rmse_ci = (
        float(np.percentile(rmse_samples, alpha * 100)),
        float(np.percentile(rmse_samples, (1.0 - alpha) * 100)),
    )

    return {
        "mae_ci_95": (round(mae_ci[0], 2), round(mae_ci[1], 2)),
        "rmse_ci_95": (round(rmse_ci[0], 2), round(rmse_ci[1], 2)),
    }


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_candidate: np.ndarray,
    h: int = 24,
    use_hln_correction: bool = True,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for statistical significance between two forecasting models.
    Supports multi-step forecast horizons (h) with Bartlett kernel autocovariance weighting
    and Harvey-Leybourne-Newbold (1997) small-sample Student's t adjustment.

    Args:
        y_true: Ground truth target values
        y_pred_baseline: Forecasts from baseline model
        y_pred_candidate: Forecasts from candidate model
        h: Forecast horizon (default 24 for day-ahead electricity forecasting)
        use_hln_correction: Whether to apply Harvey-Leybourne-Newbold (1997) correction

    Returns:
        (dm_statistic, p_value). Negative statistic indicates candidate has lower loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_baseline = np.asarray(y_pred_baseline, dtype=float)
    y_pred_candidate = np.asarray(y_pred_candidate, dtype=float)

    e_base = np.abs(y_true - y_pred_baseline)
    e_cand = np.abs(y_true - y_pred_candidate)
    d = e_cand - e_base  # Loss differential
    n = len(d)

    if n < 2:
        return 0.0, 1.0

    mean_d = float(np.mean(d))

    # Bartlett-weighted autocovariance for h-step horizon to ensure positive semi-definiteness
    gamma_0 = float(np.mean((d - mean_d) ** 2))
    long_run_var = gamma_0

    max_lag = max(1, min(h, n - 1))
    for k in range(1, max_lag):
        gamma_k = float(np.sum((d[k:] - mean_d) * (d[:-k] - mean_d)) / n)
        # Bartlett kernel weight (1 - k/h)
        weight = 1.0 - (k / float(max_lag))
        long_run_var += 2.0 * weight * gamma_k

    long_run_var = max(long_run_var, 1e-9)
    variance_mean_d = long_run_var / float(n)

    dm_stat = mean_d / np.sqrt(variance_mean_d)

    # Harvey, Leybourne, Newbold (1997) small sample adjustment
    if use_hln_correction and n > max_lag:
        hln_factor = np.sqrt(
            max(1e-9, (n + 1.0 - 2.0 * max_lag + (max_lag * (max_lag - 1.0)) / float(n)) / float(n))
        )
        dm_stat *= hln_factor
        df = n - 1
        p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=df)))
    else:
        p_value = float(2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat))))

    return float(dm_stat), float(p_value)


def compute_pairwise_statistical_comparisons(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    h: int = 24,
) -> pd.DataFrame:
    """
    Computes pairwise Diebold-Mariano tests across all model pairs with
    Holm-Bonferroni correction for multiple hypothesis testing.
    """
    models = list(predictions.keys())
    results = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m_a = models[i]
            m_b = models[j]
            dm_stat, p_val = diebold_mariano_test(y_true, predictions[m_a], predictions[m_b], h=h)
            results.append(
                {
                    "model_a": m_a,
                    "model_b": m_b,
                    "dm_statistic": round(dm_stat, 4),
                    "raw_p_value": p_val,
                    "better_model": m_b if dm_stat < 0 else m_a,
                }
            )

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Holm-Bonferroni step-down p-value adjustment
    m_tests = len(df)
    sorted_indices = df["raw_p_value"].argsort()
    adjusted_p_values = np.zeros(m_tests)

    # Walk backward or forward with (m - k + 1) multiplier
    for rank, idx in enumerate(sorted_indices):
        raw_p = df.loc[idx, "raw_p_value"]
        multiplier = m_tests - rank
        adjusted_p = min(1.0, raw_p * multiplier)
        adjusted_p_values[idx] = adjusted_p

    # Enforce monotonicity: p_adj[rank] >= p_adj[rank-1]
    sorted_adjs = adjusted_p_values[sorted_indices]
    for r in range(1, m_tests):
        if sorted_adjs[r] < sorted_adjs[r - 1]:
            sorted_adjs[r] = sorted_adjs[r - 1]
    adjusted_p_values[sorted_indices] = sorted_adjs

    df["adjusted_p_value"] = np.round(adjusted_p_values, 5)
    df["significant_alpha_05"] = df["adjusted_p_value"] < 0.05
    return df
