"""
Edge Cases and Robustness Verification Test Suite:
1. Daylight-saving transitions (Spring 23h local day & Autumn 25h local day).
2. Missing weather observations (1h, 6h, 24h, 72h dropouts).
3. Forecast horizon failures and boundary conditions.
4. Zero future data leakage verification.
"""

import zoneinfo

import numpy as np
import pandas as pd
import pytest

from chronogrid.features.engineer import FeatureEngineer
from chronogrid.pipeline.cleaning import DataCleaner
from chronogrid.serving.schemas import DayAheadForecastRequest

# ==============================================================================
# 1. Daylight-Saving Transitions (DST)
# ==============================================================================


def test_dst_spring_transition_utc_continuity():
    """
    In Germany (Europe/Berlin), Spring DST occurs on the last Sunday of March (e.g. 2024-03-31).
    Clock jumps from 02:00 CET to 03:00 CEST (23-hour local day).
    In UTC, physical time is strictly continuous (24 distinct hours).
    """
    # Create UTC hourly sequence covering the transition day
    utc_range = pd.date_range("2024-03-30 00:00:00", "2024-04-01 00:00:00", freq="1h", tz="UTC")
    assert len(utc_range) == 49  # 2 days + 1 hour unbroken

    # Convert to Europe/Berlin timezone
    berlin_tz = zoneinfo.ZoneInfo("Europe/Berlin")
    berlin_times = [ts.astimezone(berlin_tz) for ts in utc_range]

    # On March 31, local Berlin time skips 02:00 CET -> 03:00 CEST (23 local hours)
    march31_berlin = [t for t in berlin_times if t.day == 31 and t.month == 3]
    assert len(march31_berlin) == 23  # Exactly 23 local hours in Berlin due to skipped hour!

    # But in UTC, physical time flows continuously with exactly 24 hours
    march31_utc = [t for t in utc_range if t.day == 31 and t.month == 3]
    assert len(march31_utc) == 24  # Exactly 24 continuous hours in UTC!

    # Test pipeline cleaning on DST transition data
    df_dst = pd.DataFrame(
        {
            "timestamp": utc_range,
            "grid_load_mw": np.random.uniform(40000, 65000, len(utc_range)),
            "temperature_c": np.random.uniform(5, 18, len(utc_range)),
        }
    )
    cleaner = DataCleaner()
    cleaned, stats = cleaner.clean(df_dst)
    assert len(cleaned) == 49
    assert stats["missing_hours_inserted"] == 0
    assert cleaned["timestamp"].is_monotonic_increasing


def test_dst_autumn_transition_disambiguation():
    """
    In Germany, Autumn DST occurs on the last Sunday of October (e.g. 2023-10-29).
    Clock falls back from 03:00 CEST to 02:00 CET (25-hour local day).
    If unflagged local time creates duplicate 02:00 hours, DataCleaner must deduplicate
    and maintain strict UTC monotonic ordering.
    """
    utc_range = pd.date_range("2023-10-28 00:00:00", "2023-10-30 00:00:00", freq="1h", tz="UTC")
    df_autumn = pd.DataFrame(
        {
            "timestamp": utc_range,
            "grid_load_mw": np.random.uniform(40000, 65000, len(utc_range)),
            "temperature_c": np.random.uniform(5, 15, len(utc_range)),
        }
    )
    # Simulate accidental duplicate local timestamp
    dup_row = df_autumn.iloc[[20]].copy()
    df_autumn = pd.concat([df_autumn, dup_row], ignore_index=True)

    cleaner = DataCleaner()
    cleaned, stats = cleaner.clean(df_autumn)
    assert stats["duplicates_removed"] == 1
    assert len(cleaned) == len(utc_range)
    assert cleaned["timestamp"].is_monotonic_increasing


# ==============================================================================
# 2. Missing Weather Observations Handling
# ==============================================================================


@pytest.mark.parametrize("gap_hours", [1, 3, 6, 24, 72])
def test_missing_weather_observations_resilience(gap_hours):
    """
    Tests sensor failure dropouts of 1h, 3h, 6h, 24h, and 72h.
    Pipeline must impute minor gaps, extrapolate/persist multi-day gaps,
    and allow downstream feature engineering and inference to execute without crash.
    """
    n_hours = 500
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "grid_load_mw": np.random.uniform(45000, 70000, n_hours),
            "temperature_c": np.random.uniform(-5, 15, n_hours),
            "wind_speed_ms": np.random.uniform(1, 10, n_hours),
            "solar_radiation_wm2": np.random.uniform(0, 600, n_hours),
            "humidity_pct": np.random.uniform(40, 95, n_hours),
        }
    )

    # Inject missing gap in weather parameters
    gap_start = 300
    gap_end = gap_start + gap_hours
    df.loc[gap_start:gap_end, "temperature_c"] = np.nan
    df.loc[gap_start:gap_end, "solar_radiation_wm2"] = np.nan

    # 1. Cleaning stage
    cleaner = DataCleaner(max_interpolation_gap_hours=3)
    cleaned, stats = cleaner.clean(df)
    assert not cleaned["temperature_c"].isnull().any()
    assert not cleaned["solar_radiation_wm2"].isnull().any()

    # If gap > 3h, verify quality flag was set
    if gap_hours > 3:
        assert len(cleaned) < len(df)
        assert (cleaned["quality_flag"] == "CAUSAL_FORWARD_FILL").any()

    # 2. Feature engineering stage must run without error or NaNs
    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24)
    feat_df = fe.fit_transform(cleaned, drop_warmup=True)
    assert len(feat_df) > 0
    assert not feat_df[fe.get_feature_names()].isnull().any().any()


# ==============================================================================
# 3. Forecast Horizon Failures & Boundary Conditions
# ==============================================================================


def test_forecast_horizon_invalid_bounds():
    """Tests Pydantic validation on illegal forecast horizons."""
    # Zero or negative horizon
    with pytest.raises(Exception):
        DayAheadForecastRequest(horizon_hours=0)
    with pytest.raises(Exception):
        DayAheadForecastRequest(horizon_hours=-5)

    # Exceeding maximum allowed horizon (168 hours / 1 week)
    with pytest.raises(Exception):
        DayAheadForecastRequest(horizon_hours=169)


def test_insufficient_history_depth_failure():
    """
    Tests that requesting feature transformation with insufficient historical lag depth
    (e.g. only 10 hours of history when 168 hours of weekly lags are required)
    is caught and handled safely.
    """
    short_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC"),
            "grid_load_mw": np.random.uniform(40000, 60000, 10),
            "temperature_c": np.random.uniform(5, 15, 10),
        }
    )
    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, lags=[24, 168])
    fe.fit(short_df)

    # Transforming with drop_warmup=True on only 10 rows must return an empty dataframe
    # rather than crashing, signaling insufficient history depth
    transformed = fe.transform(short_df, drop_warmup=True)
    assert len(transformed) == 0


# ==============================================================================
# 4. Zero Future Data Leakage Verification
# ==============================================================================


def test_zero_future_leakage_day_ahead():
    """
    Mathematically verifies that no feature relies on contemporaneous realised observations
    or any data within the day-ahead forecast issuance window [t - 23, t].
    """
    n_hours = 300
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_hours, freq="1h", tz="UTC"),
            "grid_load_mw": np.arange(n_hours, dtype=float)
            * 100.0,  # Monotonically increasing load
            "temperature_c": np.random.uniform(-5, 25, n_hours),
            "wind_speed_ms": np.random.uniform(1, 12, n_hours),
            "solar_radiation_wm2": np.random.uniform(0, 800, n_hours),
            "humidity_pct": np.random.uniform(40, 90, n_hours),
        }
    )

    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24)
    feat_df = fe.fit_transform(df, drop_warmup=True)

    # 1. Assert verify_zero_future_leakage passes
    assert fe.verify_zero_future_leakage(feat_df) is True

    # 2. Assert contemporaneous raw variables are strictly absent from feature_cols
    forbidden = {
        "grid_load_mw",
        "temperature_c",
        "wind_speed_ms",
        "solar_radiation_wm2",
        "humidity_pct",
        "solar_generation_mw",
        "wind_generation_mw",
    }
    for f in fe.get_feature_names():
        assert f not in forbidden, f"Forbidden contemporaneous variable in features: {f}"

    # 3. Assert all grid load lag features are >= 24 hours
    for f in fe.get_feature_names():
        if "grid_load_mw_lag_" in f:
            lag = int(f.split("_lag_")[-1])
            assert lag >= 24, f"Lag {lag} violates 24h day-ahead issuance constraint!"
