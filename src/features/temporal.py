"""
Temporal and Calendar Feature Engineering (P5.3).
Cyclical trigonometric encodings and German public holidays.
"""

import numpy as np
import pandas as pd


# Official German Federal Holidays
GERMAN_HOLIDAYS = {
    # 2023
    (2023, 1, 1),   # Neujahr
    (2023, 4, 7),   # Karfreitag
    (2023, 4, 10),  # Ostermontag
    (2023, 5, 1),   # Tag der Arbeit
    (2023, 5, 18),  # Christi Himmelfahrt
    (2023, 5, 29),  # Pfingstmontag
    (2023, 10, 3),  # Tag der Deutschen Einheit
    (2023, 12, 25), # 1. Weihnachtstag
    (2023, 12, 26), # 2. Weihnachtstag
    # 2024
    (2024, 1, 1),   # Neujahr
    (2024, 3, 29),  # Karfreitag
    (2024, 4, 1),   # Ostermontag
    (2024, 5, 1),   # Tag der Arbeit
    (2024, 5, 9),   # Christi Himmelfahrt
    (2024, 5, 20),  # Pfingstmontag
    (2024, 10, 3),  # Tag der Deutschen Einheit
    (2024, 12, 25), # 1. Weihnachtstag
    (2024, 12, 26), # 2. Weihnachtstag
    # 2025
    (2025, 1, 1), (2025, 4, 18), (2025, 4, 21), (2025, 5, 1),
    (2025, 5, 29), (2025, 6, 9), (2025, 10, 3), (2025, 12, 25), (2025, 12, 26),
}


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
    out["is_holiday"] = np.array([
        1 if (t.year, t.month, t.day) in GERMAN_HOLIDAYS else 0
        for t in ts
    ])
    out["is_working_day"] = ((out["is_weekend"] == 0) & (out["is_holiday"] == 0)).astype(int)

    return out

