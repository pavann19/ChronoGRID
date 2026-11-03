"""
Tests for Feature Engineering and Leak-Free Transforms (P5.3).
"""

import numpy as np
import pandas as pd
from src.features.temporal import add_temporal_features
from src.features.weather_features import add_weather_features
from src.features.autoregressive import add_autoregressive_features
from src.features.engineer import FeatureEngineer


def test_temporal_features_cyclical_bounds():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 00:00:00", periods=24, freq="1h", tz="UTC"),
        "grid_load_mw": [50000.0] * 24,
    })
    feat = add_temporal_features(df)
    assert "sin_hour" in feat.columns
    assert "cos_hour" in feat.columns
    assert feat["sin_hour"].min() >= -1.0
    assert feat["sin_hour"].max() <= 1.0


def test_weather_degree_hours():
    df = pd.DataFrame({
        "temperature_c": [10.0, 18.0, 25.0],
        "wind_speed_ms": [5.0, 5.0, 5.0],
        "solar_radiation_wm2": [0.0, 200.0, 500.0],
    })
    feat = add_weather_features(df, heating_base=18.0, cooling_base=22.0)
    assert feat["heating_degree_hours"].iloc[0] == 8.0 # 18 - 10
    assert feat["heating_degree_hours"].iloc[1] == 0.0
    assert feat["cooling_degree_hours"].iloc[2] == 3.0 # 25 - 22


def test_autoregressive_lags_leak_free():
    df = pd.DataFrame({
        "grid_load_mw": [100.0, 200.0, 300.0, 400.0, 500.0]
    })
    feat = add_autoregressive_features(df, target_col="grid_load_mw", lags=[1, 2], rolling_windows=[2])
    # At index 2 (value 300.0), lag_1 should be 200.0, lag_2 should be 100.0
    assert feat["grid_load_mw_lag_1"].iloc[2] == 200.0
    assert feat["grid_load_mw_lag_2"].iloc[2] == 100.0
    # Rolling mean for window 2 at index 2 is over past values (100.0, 200.0) -> mean = 150.0 (NOT including 300.0)
    assert feat["grid_load_mw_roll_mean_2"].iloc[2] == 150.0
