"""
Autoregressive and Rolling Window Statistics Feature Engineering (P5.3).
Strictly prevents data leakage by computing rolling stats only over past shifted lags.
"""

from typing import List
import numpy as np
import pandas as pd


def add_autoregressive_features(
    df: pd.DataFrame,
    target_col: str = "grid_load_mw",
    lags: List[int] = None,
    rolling_windows: List[int] = None,
    min_lag_shift: int = 1,
) -> pd.DataFrame:
    """
    Computes autoregressive lag features and backward-looking rolling statistics.
    All rolling statistics are applied to df[target_col].shift(min_lag_shift) to guarantee zero leakage.
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168]
    if rolling_windows is None:
        rolling_windows = [6, 12, 24, 168]

    out = df.copy()

    # 1. Discrete Lags
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    # 2. Rolling Window Statistics (Strictly Backward-Looking)
    # Base shifted series to guarantee no leakage of current target value
    shifted_target = out[target_col].shift(min_lag_shift)

    for w in rolling_windows:
        rolling_obj = shifted_target.rolling(window=w, min_periods=max(1, w // 2))
        out[f"{target_col}_roll_mean_{w}"] = rolling_obj.mean()
        out[f"{target_col}_roll_std_{w}"] = rolling_obj.std().fillna(0.0)
        out[f"{target_col}_roll_min_{w}"] = rolling_obj.min()
        out[f"{target_col}_roll_max_{w}"] = rolling_obj.max()

    # 3. Ratio / Difference Features (Trend and Seasonality Momentum)
    # Ratio to yesterday same hour
    if f"{target_col}_lag_24" in out.columns and f"{target_col}_lag_1" in out.columns:
        out[f"{target_col}_diff_24"] = out[f"{target_col}_lag_1"] - out[f"{target_col}_lag_24"]
    # Ratio to last week same hour
    if f"{target_col}_lag_168" in out.columns and f"{target_col}_lag_24" in out.columns:
        out[f"{target_col}_diff_168"] = out[f"{target_col}_lag_24"] - out[f"{target_col}_lag_168"]

    return out
