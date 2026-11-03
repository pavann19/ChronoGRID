"""
Time Series Forecasting Evaluation Metrics (P5.4, P5.5, P5.6).
Computes MAE, RMSE, MAPE, sMAPE, R2, Peak-Hour Error, Bootstrap CIs, and Diebold-Mariano test.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray = None) -> Dict[str, float]:
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

    metrics = {
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
            metrics["peak_hours_mae"] = round(float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])), 2)

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
    mae_ci = (float(np.percentile(mae_samples, alpha * 100)), float(np.percentile(mae_samples, (1.0 - alpha) * 100)))
    rmse_ci = (float(np.percentile(rmse_samples, alpha * 100)), float(np.percentile(rmse_samples, (1.0 - alpha) * 100)))

    return {
        "mae_ci_95": (round(mae_ci[0], 2), round(mae_ci[1], 2)),
        "rmse_ci_95": (round(rmse_ci[0], 2), round(rmse_ci[1], 2)),
    }


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_candidate: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for statistical significance between two forecasting models.
    H0: Both models have identical forecast accuracy.
    Returns: (dm_statistic, p_value). Negative statistic means candidate has lower loss.
    """
    e_base = np.abs(y_true - y_pred_baseline)
    e_cand = np.abs(y_true - y_pred_candidate)
    d = e_cand - e_base  # Loss differential

    mean_d = np.mean(d)
    n = len(d)
    
    # Autocovariance adjustment for h-step horizon
    autocov = np.var(d)
    for lag in range(1, h):
        gamma = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        autocov += 2.0 * gamma

    variance_d = max(autocov / n, 1e-9)
    dm_stat = float(mean_d / np.sqrt(variance_d))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat))))

    return dm_stat, p_value
