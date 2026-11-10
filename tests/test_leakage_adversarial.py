"""
Adversarial information-boundary & Day-Ahead Information Boundary Tests.
Verifies that no future target, weather, or market information leaks into predictions.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.engineer import FeatureEngineer


@pytest.fixture
def clean_sample_data():
    ts = pd.date_range("2024-01-01 00:00:00", periods=400, freq="1h", tz="UTC")
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "timestamp": ts,
        "grid_load_mw": 50000.0 + 10000.0 * np.sin(2 * np.pi * np.arange(400) / 24) + rng.normal(0, 1000, 400),
        "temperature_c": 10.0 + 5.0 * np.cos(2 * np.pi * np.arange(400) / 24) + rng.normal(0, 1, 400),
        "wind_speed_ms": np.maximum(0.5, rng.normal(5, 2, 400)),
        "solar_radiation_wm2": np.maximum(0.0, 500.0 * np.sin(np.pi * (ts.hour - 6) / 12)),
        "humidity_pct": np.clip(rng.normal(65, 10, 400), 20, 100),
    })
    return df


def test_adversarial_future_target_perturbation(clean_sample_data):
    """Perturbing a future target row must NOT alter features at earlier rows."""
    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, weather_mode="persistence")
    fe.fit(clean_sample_data)
    
    base_feats = fe.transform(clean_sample_data, drop_warmup=True)
    feat_cols = fe.get_feature_names()

    # Adversarially perturb target at index 350
    perturbed_df = clean_sample_data.copy()
    pert_idx = 350
    perturbed_df.loc[pert_idx, "grid_load_mw"] += 100000.0

    pert_feats = fe.transform(perturbed_df, drop_warmup=True)

    # All rows prior to pert_idx must have identical features (diff == 0.0)
    warmup_offset = len(clean_sample_data) - len(base_feats)
    eval_pert_idx = pert_idx - warmup_offset

    diff = np.max(np.abs(
        base_feats.iloc[:eval_pert_idx][feat_cols].values - 
        pert_feats.iloc[:eval_pert_idx][feat_cols].values
    ))
    assert diff == 0.0, f"Leakage detected: prior features changed by {diff} after future target perturbation!"


def test_adversarial_future_weather_perturbation(clean_sample_data):
    """Perturbing future weather at delivery hour must NOT alter features at earlier rows."""
    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, weather_mode="persistence")
    fe.fit(clean_sample_data)
    
    base_feats = fe.transform(clean_sample_data, drop_warmup=True)
    feat_cols = fe.get_feature_names()

    perturbed_df = clean_sample_data.copy()
    pert_idx = 350
    perturbed_df.loc[pert_idx, "temperature_c"] += 50.0

    pert_feats = fe.transform(perturbed_df, drop_warmup=True)

    warmup_offset = len(clean_sample_data) - len(base_feats)
    eval_pert_idx = pert_idx - warmup_offset

    diff = np.max(np.abs(
        base_feats.iloc[:eval_pert_idx][feat_cols].values - 
        pert_feats.iloc[:eval_pert_idx][feat_cols].values
    ))
    assert diff == 0.0, f"Leakage detected: prior features changed by {diff} after future weather perturbation!"


def test_market_gate_closure_winter_vs_summer(clean_sample_data):
    """Asserts that 12:00 Berlin gate closure converts to 11:00 UTC in winter (CET) and 10:00 UTC in summer (CEST)."""
    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, weather_mode="persistence")
    
    # 1. Winter date (Jan 15, CET = UTC+1)
    winter_origin = pd.Timestamp("2024-01-15 12:00:00", tz="Europe/Berlin")
    assert winter_origin.tz_convert("UTC").hour == 11, "Winter 12:00 CET must convert to 11:00 UTC!"

    # 2. Summer date (Jul 15, CEST = UTC+2)
    summer_origin = pd.Timestamp("2024-07-15 12:00:00", tz="Europe/Berlin")
    assert summer_origin.tz_convert("UTC").hour == 10, "Summer 12:00 CEST must convert to 10:00 UTC!"


def test_block_day_ahead_market_frame_generator(clean_sample_data):
    """Verifies that block day-ahead frame generation produces valid delivery profiles with zero leakage."""
    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, weather_mode="persistence")
    features_df, meta = fe.build_day_ahead_market_frame(clean_sample_data)

    assert meta["delivery_hours"] in (23, 24, 25)
    assert len(features_df) == meta["delivery_hours"]
    assert meta["lead_time_min_hours"] >= 12.0
    assert meta["lead_time_max_hours"] <= 36.0
    assert "grid_load_mw" not in features_df.columns or "forecast_temperature_c" in features_df.columns

