"""
Adversarial Reliability Test Suite for ChronoGrid DDIR (Historical evaluation).
Verifies deterministic validation, state transitions, dual-head routing, anti-flapping hysteresis,
and high-concurrency resilience across all 10 failure regimes.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronogrid.reliability.router import DualHeadInferenceRouter
from chronogrid.reliability.state_machine import ReliabilityState


@pytest.fixture
def router():
    """Initializes DualHeadInferenceRouter with pre-compiled ONNX models."""
    return DualHeadInferenceRouter(
        primary_onnx_path="artifacts/models/champion_model.onnx",
        primary_metadata_path="artifacts/models/champion_metadata.json",
        fallback_onnx_path="artifacts/models/load_only_model.onnx",
        fallback_metadata_path="artifacts/models/load_only_metadata.json",
        recovery_threshold=2,
        max_freshness_hours=28.0,
    )


@pytest.fixture
def valid_weather_payload():
    """Standard fresh and physically valid weather payload."""
    now_utc = datetime.now(timezone.utc)
    return {
        "timestamp": (now_utc - timedelta(hours=2)).isoformat(),
        "temperature_c": 18.5,
        "wind_speed_ms": 4.2,
        "solar_radiation_wm2": 450.0,
        "humidity_pct": 55.0,
    }


@pytest.fixture
def sample_feature_df(router):
    """Synthetic feature row covering all 43 primary features."""
    row = {f: 1.0 for f in router.primary_engine.feature_names}
    row["forecast_temperature_c"] = 18.5
    row["forecast_wind_speed_ms"] = 4.2
    row["forecast_solar_radiation_wm2"] = 450.0
    row["forecast_humidity_pct"] = 55.0
    return pd.DataFrame([row])


# ==============================================================================
# Regime 1: Normal Healthy Operation
# ==============================================================================
def test_ddir_regime_1_normal_healthy(router, sample_feature_df, valid_weather_payload):
    result = router.route_and_predict(sample_feature_df, weather_payload=valid_weather_payload)
    assert result.routing_state == ReliabilityState.HEALTHY
    assert result.active_head == "PRIMARY_WEATHER"
    assert result.fallback_reason is None
    assert result.weather_quality_score >= 0.95
    assert len(result.predictions) == 1
    assert result.predictions[0] > 0
    assert result.features_used_count == 43


# ==============================================================================
# Regime 2: Complete Missingness
# ==============================================================================
def test_ddir_regime_2_complete_missingness(router, sample_feature_df):
    # Pass None as weather payload
    result = router.route_and_predict(sample_feature_df, weather_payload=None)
    assert result.routing_state == ReliabilityState.UNAVAILABLE
    assert result.active_head == "FALLBACK_LOAD_ONLY"
    assert (
        "unavailable" in result.fallback_reason.lower()
        or "missing" in result.fallback_reason.lower()
    )
    assert result.weather_quality_score == 0.0
    assert len(result.predictions) == 1
    assert result.predictions[0] > 0
    assert result.features_used_count == 34


# ==============================================================================
# Regime 3: Delayed Weather Ingestion
# ==============================================================================
def test_ddir_regime_3_delayed_weather(router, sample_feature_df):
    now_utc = datetime.now(timezone.utc)
    delayed_payload = {
        "timestamp": (now_utc - timedelta(hours=30)).isoformat(),  # 30h old > 28h gate closure SLA
        "temperature_c": 15.0,
        "wind_speed_ms": 3.5,
        "solar_radiation_wm2": 300.0,
        "humidity_pct": 60.0,
    }
    result = router.route_and_predict(sample_feature_df, weather_payload=delayed_payload)
    assert result.routing_state == ReliabilityState.STALE
    assert result.active_head == "FALLBACK_LOAD_ONLY"
    assert "delayed" in result.fallback_reason.lower() or "stale" in result.fallback_reason.lower()
    assert len(result.predictions) == 1


# ==============================================================================
# Regime 4: Stale / Frozen Telemetry
# ==============================================================================
def test_ddir_regime_4_stale_telemetry(router, sample_feature_df):
    now_utc = datetime.now(timezone.utc)
    stale_payload = {
        "timestamp": (now_utc - timedelta(hours=48)).isoformat(),  # 48h old > 36h threshold
        "temperature_c": 12.0,
        "wind_speed_ms": 5.0,
        "solar_radiation_wm2": 0.0,
        "humidity_pct": 75.0,
    }
    result = router.route_and_predict(sample_feature_df, weather_payload=stale_payload)
    assert result.routing_state == ReliabilityState.STALE
    assert result.active_head == "FALLBACK_LOAD_ONLY"
    assert "stale" in result.fallback_reason.lower()
    assert len(result.predictions) == 1


# ==============================================================================
# Regime 5: Malformed Schema
# ==============================================================================
def test_ddir_regime_5_malformed_schema(router, sample_feature_df):
    corrupt_payload = {
        "temperature_c": "CORRUPTED_STRING_VALUE",
        "wind_speed_ms": None,
        "solar_radiation_wm2": "INVALID",
        "humidity_pct": 50.0,
    }
    result = router.route_and_predict(sample_feature_df, weather_payload=corrupt_payload)
    assert result.routing_state == ReliabilityState.INVALID
    assert result.active_head == "FALLBACK_LOAD_ONLY"
    assert (
        "corrupted" in result.fallback_reason.lower()
        or "malformed" in result.fallback_reason.lower()
    )
    assert len(result.predictions) == 1


# ==============================================================================
# Regime 6: Physical Outliers
# ==============================================================================
def test_ddir_regime_6_physical_outliers(router, sample_feature_df):
    outlier_payload = {
        "temperature_c": 999.0,  # Physically impossible for Central Europe
        "wind_speed_ms": -5.0,  # Negative wind speed
        "solar_radiation_wm2": -100.0,  # Negative solar radiation
        "humidity_pct": 180.0,  # > 100% humidity
    }
    result = router.route_and_predict(sample_feature_df, weather_payload=outlier_payload)
    assert result.routing_state == ReliabilityState.INVALID
    assert result.active_head == "FALLBACK_LOAD_ONLY"
    assert "outlier" in result.fallback_reason.lower()
    assert len(result.predictions) == 1


# ==============================================================================
# Regime 7: Partial Missingness
# ==============================================================================
def test_ddir_regime_7_partial_missingness(router, sample_feature_df):
    # Temperature intact, but secondary wind_speed and humidity missing
    partial_payload = {
        "temperature_c": 21.0,
        "wind_speed_ms": None,
        "solar_radiation_wm2": 500.0,
        "humidity_pct": None,
    }
    result = router.route_and_predict(sample_feature_df, weather_payload=partial_payload)
    # Temperature is intact with 2/4 channels missing
    assert result.routing_state in [ReliabilityState.DEGRADED, ReliabilityState.UNAVAILABLE]
    assert len(result.predictions) == 1
    assert result.predictions[0] > 0


# ==============================================================================
# Regime 8: Provider Network Timeout / Exception
# ==============================================================================
def test_ddir_regime_8_provider_timeout(router, sample_feature_df):
    # Empty dict simulating HTTP connection reset
    result = router.route_and_predict(sample_feature_df, weather_payload={})
    assert result.routing_state == ReliabilityState.UNAVAILABLE
    assert result.active_head == "FALLBACK_LOAD_ONLY"
    assert len(result.predictions) == 1


# ==============================================================================
# Regime 9: Anti-Flapping Hysteresis Recovery (Distinct Updates Required)
# ==============================================================================
def test_ddir_regime_9_hysteresis_recovery(router, sample_feature_df, valid_weather_payload):
    now_utc = datetime.now(timezone.utc)
    p_healthy_0 = {**valid_weather_payload, "timestamp": (now_utc - timedelta(hours=4)).isoformat()}
    p_healthy_1 = {**valid_weather_payload, "timestamp": (now_utc - timedelta(hours=3)).isoformat()}
    p_healthy_2 = {
        **valid_weather_payload,
        "timestamp": (now_utc - timedelta(hours=2)).isoformat(),
        "temperature_c": 19.1,
    }
    p_duplicate = {**p_healthy_1}  # Duplicate of 1

    # 1. Start Healthy at T-4h
    res1 = router.route_and_predict(sample_feature_df, weather_payload=p_healthy_0)
    assert res1.routing_state == ReliabilityState.HEALTHY
    assert res1.active_head == "PRIMARY_WEATHER"

    # 2. Inject Failure -> Immediate demotion to fallback
    res2 = router.route_and_predict(sample_feature_df, weather_payload=None)
    assert res2.routing_state == ReliabilityState.UNAVAILABLE
    assert res2.active_head == "FALLBACK_LOAD_ONLY"

    # 3. First distinct advancing update at T-3h -> Held on Fallback (K=1 < K_threshold=2)
    res3 = router.route_and_predict(sample_feature_df, weather_payload=p_healthy_1)
    assert res3.active_head == "FALLBACK_LOAD_ONLY"
    assert "recovery in progress" in res3.fallback_reason.lower()
    assert res3.consecutive_healthy_count == 1

    # 4. Replay duplicate update at T-3h -> Held on Fallback, counter does NOT increment!
    res_dup = router.route_and_predict(sample_feature_df, weather_payload=p_duplicate)
    assert res_dup.active_head == "FALLBACK_LOAD_ONLY"
    assert "recovery held" in res_dup.fallback_reason.lower()
    assert res_dup.consecutive_healthy_count == 1

    # 5. Second distinct advancing healthy update at T-2h -> Promoted to Primary (K=2 >= K_threshold=2)
    res4 = router.route_and_predict(sample_feature_df, weather_payload=p_healthy_2)
    assert res4.routing_state == ReliabilityState.HEALTHY
    assert res4.active_head == "PRIMARY_WEATHER"
    assert res4.consecutive_healthy_count == 2


# ==============================================================================
# Regime 10: High Concurrency Stress Test
# ==============================================================================
def test_ddir_regime_10_concurrent_requests(router, sample_feature_df, valid_weather_payload):
    n_threads = 20
    requests = []

    for i in range(n_threads):
        if i % 3 == 0:
            payload = valid_weather_payload
        elif i % 3 == 1:
            payload = None
        else:
            payload = {"temperature_c": 999.0}
        requests.append(payload)

    def worker(p):
        return router.route_and_predict(sample_feature_df, weather_payload=p)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(worker, requests))

    assert len(results) == n_threads
    for r in results:
        # Every single request must successfully return valid predictions (100% FCR)
        assert len(r.predictions) == 1
        assert r.predictions[0] > 0
        assert r.routing_state in [
            ReliabilityState.HEALTHY,
            ReliabilityState.UNAVAILABLE,
            ReliabilityState.INVALID,
        ]


def test_ddir_reports_elapsed_times(router, sample_feature_df, valid_weather_payload, monkeypatch):
    # Exercise real validation/routing/inference, while measuring controlled intervals.
    # CI scheduling and coverage instrumentation must not become latency guarantees.
    ticks = iter([10.0, 10.001, 10.003, 10.004, 10.009])
    monkeypatch.setattr(
        "chronogrid.reliability.router.time",
        SimpleNamespace(perf_counter=lambda: next(ticks)),
    )
    result = router.route_and_predict(sample_feature_df, weather_payload=valid_weather_payload)
    assert result.validation_time_ms == pytest.approx(2.0)
    assert result.inference_time_ms == pytest.approx(5.0)
    assert result.total_time_ms == pytest.approx(9.0)
    assert result.routing_state == ReliabilityState.HEALTHY
    assert len(result.predictions) == 1 and np.isfinite(result.predictions[0])
