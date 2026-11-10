"""
Temporal and Calendar Feature Engineering (P5.3).
Cyclical trigonometric encodings and German public holidays.
"""

import numpy as np
import pandas as pd


import holidays

# Compatibility inventory for the separately guarded historical models.
GERMAN_HOLIDAYS = {(d.year, d.month, d.day) for d in holidays.country_holidays("DE", years=range(2023, 2026))}


def add_temporal_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Computes cyclical trigonometric encodings and German calendar features.
    """
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)

    hours = ts.dt.hour.values
    dayofweek = ts.dt.dayofweek.values
    dayofyear = ts.dt.dayofyear.values
    month = ts.dt.month.values

    # Cyclical trigonometric features (preserves distance between 23h and 00h)
    out["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
    
    out["sin_dayofweek"] = np.sin(2 * np.pi * dayofweek / 7.0)
    out["cos_dayofweek"] = np.cos(2 * np.pi * dayofweek / 7.0)
    
    out["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    out["sin_dayofyear"] = np.sin(2 * np.pi * (dayofyear - 1) / 365.25)
    out["cos_dayofyear"] = np.cos(2 * np.pi * (dayofyear - 1) / 365.25)

    # Discrete indicators
    out["is_weekend"] = (dayofweek >= 5).astype(int)
    calendar = holidays.country_holidays("DE", years=sorted(set(ts.dt.year)))
    out["is_holiday"] = np.array([
        1 if t.date() in calendar else 0
        for t in ts
    ])
    out["is_working_day"] = ((out["is_weekend"] == 0) & (out["is_holiday"] == 0)).astype(int)

    return out

