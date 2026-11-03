"""
Autoregressive and Rolling Window Statistics Feature Engineering (P5.3).
Strictly prevents data leakage under Day-Ahead Market (DAM) forecast issuance constraints.
All lags and rolling statistics are strictly backward-looking from issuance time (t - min_lag_hours).
"""

from typing import List
import numpy as np
import pandas as pd


def add_autoregressive_features(
    df: pd.DataFrame,
    target_col: str = "grid_load_mw",
    lags: List[int] = None,
    rolling_windows: List[int] = None,
    min_lag_hours: int = 24, # Day-Ahead Gate Closure: minimum 24h lag
) -> pd.DataFrame:
    """
    Computes autoregressive lag features and backward-looking rolling statistics.
    For Day-Ahead forecasting (min_lag_hours=24):
    - All lags are >= min_lag_hours (e.g. 24, 25, 48, 72, 168).
    - All rolling statistics are strictly computed on df[target_col].shift(min_lag_hours).
    This strictly guarantees ZERO data leakage from the forecast window [t - min_lag_hours + 1, t].
    """
    if lags is None:
        # Day-Ahead default lags: yesterday same hour (24), prior hour (25), 2 days ago (48), 3 days ago (72), last week (168)
        lags = [24, 25, 48, 72, 168]
    
    # Ensure all lags adhere to min_lag_hours
    lags = [lag for lag in lags if lag >= min_lag_hours]
    if not lags:
        lags = [min_lag_hours]

    if rolling_windows is None:
        rolling_windows = [6, 12, 24, 168]

    out = df.copy()

    # 1. Discrete Lags (>= min_lag_hours)
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    # 2. Rolling Window Statistics (Strictly Shifted by min_lag_hours)
    # At hour t, only observations up to t - min_lag_hours are available at issuance
    shifted_target = out[target_col].shift(min_lag_hours)

    for w in rolling_windows:
        rolling_obj = shifted_target.rolling(window=w, min_periods=max(1, w // 2))
        out[f"{target_col}_roll_mean_{w}"] = rolling_obj.mean()
        out[f"{target_col}_roll_std_{w}"] = rolling_obj.std().fillna(0.0)
        out[f"{target_col}_roll_min_{w}"] = rolling_obj.min()
        out[f"{target_col}_roll_max_{w}"] = rolling_obj.max()

    # 3. Ratio / Difference Features (Momentum strictly from known issuance data)
    if f"{target_col}_lag_{min_lag_hours}" in out.columns and f"{target_col}_lag_48" in out.columns:
        out[f"{target_col}_diff_24_48"] = out[f"{target_col}_lag_{min_lag_hours}"] - out[f"{target_col}_lag_48"]
    if f"{target_col}_lag_{min_lag_hours}" in out.columns and f"{target_col}_lag_168" in out.columns:
        out[f"{target_col}_diff_24_168"] = out[f"{target_col}_lag_{min_lag_hours}"] - out[f"{target_col}_lag_168"]

    return out
