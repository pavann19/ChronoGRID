"""
ChronoGrid DDIR Historical evaluation Remediation Regression Suite.
Verifies all 5 highest-priority fixes supported by reproducible audit evidence:
1. Observability Blast Radius Shielding (RISK-02)
2. Future Timestamp Validation & Routing to INVALID (RISK-05)
3. Frozen Sensor Reading Detection & Anti-Replay Hardening (RISK-06)
4. Cold-Start State Inversion Prevention via Atomic Persistence (RISK-04)
5. Validation/State Machine Fail-Safe Exception Handling (Tests 4.4 & 4.5)
6. Prometheus Multiprocess Metric Collector Integration (RISK-03)
"""

from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
import pandas as pd

from src.reliability.router import DualHeadInferenceRouter
from src.reliability.validator import WeatherInputValidator
from src.reliability.state_machine import ReliabilityState, ReliabilityStateMachine
from src.reliability.recovery import HysteresisRecoveryManager
from src.reliability.persistence import DDIRStatePersistence, DDIRPersistedState
from src.serving.app import load_engine, forecast_resilient, metrics, DDIR_REQUEST_COUNT, DDIR_QUALITY_GAUGE
from src.serving.schemas import ResilientForecastRequest


@pytest.fixture(scope="module")
def initialized_engine():
    load_engine()


# =============================================================================
# 1. Observability Blast Radius Shielding (RISK-02)
# =============================================================================

def test_remediation_observability_shielding_counter_failure(initialized_engine):
    """
    Audit Finding: RISK-02 (CRITICAL). Unhandled exceptions in Prometheus metric collection abort forecasts.
    Fix: Wrap metric collection in try-except in src/serving/app.py.
    Verification: Request succeeds with HTTP 200 equivalent and complete 24-hour predictions despite Prometheus error.
    """
    req = ResilientForecastRequest(
        start_timestamp="2024-07-20T14:00:00Z",
        horizon_hours=24,
        weather_payload=None,
    )
    with patch.object(DDIR_REQUEST_COUNT, "labels", side_effect=RuntimeError("Prometheus Lock Error")):
        resp = forecast_resilient(req)
        assert resp is not None
        assert len(resp.forecast_load_mw) == 24
        assert resp.active_head == "FALLBACK_LOAD_ONLY"
        assert resp.routing_state == "UNAVAILABLE"


def test_remediation_observability_shielding_gauge_failure(initialized_engine):
    """
    Audit Finding: RISK-02 (CRITICAL). Gauge update failure must not abort forecasts.
    """
    req = ResilientForecastRequest(
        start_timestamp="2024-07-20T14:00:00Z",
        horizon_hours=24,
        weather_payload=None,
    )
    with patch.object(DDIR_QUALITY_GAUGE, "set", side_effect=ValueError("MMap corrupt")):
        resp = forecast_resilient(req)
        assert resp is not None
        assert len(resp.forecast_load_mw) == 24


# =============================================================================
# 2. Future Timestamp Validation & Routing to INVALID (RISK-05)
# =============================================================================

def test_remediation_future_timestamp_rejected():
    """
    Audit Finding: RISK-05 (HIGH). Clamping negative age to 0.0h allows far-future timestamps to be accepted.
    Fix: Detect timestamps > 300s into the future, mark is_valid=False, quality=0.0, and route to INVALID.
    """
    validator = WeatherInputValidator()
    now_ref = datetime(2026, 11, 3, 16, 0, 0, tzinfo=timezone.utc)
    future_ts = (now_ref + timedelta(days=365)).isoformat()

    report = validator.validate({
        "timestamp": future_ts,
        "temperature_c": 18.0,
        "wind_speed_ms": 4.0,
        "solar_radiation_wm2": 300.0,
        "humidity_pct": 50.0,
    }, current_time=now_ref)

    assert report.is_valid is False
    assert report.quality_score == 0.0
    assert any("future" in v.lower() for v in report.violations)

    state, reason = ReliabilityStateMachine.evaluate(report)
    assert state == ReliabilityState.INVALID
    assert "future" in reason.lower() or "temporal provenance" in reason.lower()


def test_remediation_future_timestamp_routes_to_fallback():
    """
    Verify that an input with future timestamp forces DualHeadInferenceRouter to Fallback head.
    """
    router = DualHeadInferenceRouter()
    now_ref = datetime(2026, 11, 3, 16, 0, 0, tzinfo=timezone.utc)
    future_ts = (now_ref + timedelta(days=30)).isoformat()
    raw_df = pd.DataFrame([{f: 1.0 for f in router.primary_engine.feature_names}])

    res = router.route_and_predict(
        features=raw_df,
        weather_payload={
            "timestamp": future_ts,
            "temperature_c": 18.0,
            "wind_speed_ms": 4.0,
            "solar_radiation_wm2": 300.0,
            "humidity_pct": 50.0,
        },
        current_time=now_ref,
    )

    assert res.active_head == "FALLBACK_LOAD_ONLY"
    assert res.routing_state == ReliabilityState.INVALID
    assert "temporal provenance" in res.fallback_reason.lower() or "future" in res.fallback_reason.lower()


# =============================================================================
# 3. Frozen Sensor Reading Detection & Anti-Replay Hardening (RISK-06)
# =============================================================================

def test_remediation_frozen_sensor_held_on_fallback():
    """
    Audit Finding: RISK-06 (MEDIUM). Advancing timestamp header on frozen physical sensor bypassed anti-replay.
    Fix: Telemetry content fingerprint excluding timestamp detects identical sensor values across advancing timestamps.
    Verification: Recovery is held on Fallback head; K counter does not reach threshold.
    """
    router = DualHeadInferenceRouter(recovery_threshold=2)
    router.recovery_manager.reset(ReliabilityState.UNAVAILABLE)

    now_ref = datetime(2026, 11, 3, 16, 0, 0, tzinfo=timezone.utc)
    raw_df = pd.DataFrame([{f: 1.0 for f in router.primary_engine.feature_names}])

    frozen_vals = {"temperature_c": 15.0, "wind_speed_ms": 4.0, "solar_radiation_wm2": 250.0, "humidity_pct": 60.0}

    # Update 1: T-2h
    r1 = router.route_and_predict(raw_df, weather_payload={"timestamp": (now_ref - timedelta(hours=2)).isoformat(), **frozen_vals}, current_time=now_ref)
    assert r1.active_head == "FALLBACK_LOAD_ONLY"
    assert r1.consecutive_healthy_count == 1

    # Update 2: T-1h (advancing timestamp, but identical readings)
    r2 = router.route_and_predict(raw_df, weather_payload={"timestamp": (now_ref - timedelta(hours=1)).isoformat(), **frozen_vals}, current_time=now_ref)
    assert r2.active_head == "FALLBACK_LOAD_ONLY"
    assert r2.consecutive_healthy_count == 1
    assert "frozen" in r2.fallback_reason.lower() or "variance" in r2.fallback_reason.lower()

    # Update 3: Genuine fresh readings with variance (T-0.5h) -> Recovery succeeds!
    varied_vals = {"temperature_c": 16.4, "wind_speed_ms": 4.8, "solar_radiation_wm2": 290.0, "humidity_pct": 57.0}
    r3 = router.route_and_predict(raw_df, weather_payload={"timestamp": (now_ref - timedelta(minutes=30)).isoformat(), **varied_vals}, current_time=now_ref)
    assert r3.active_head == "PRIMARY_WEATHER"
    assert r3.consecutive_healthy_count == 2
    assert r3.routing_state == ReliabilityState.HEALTHY


# =============================================================================
# 4. Cold-Start State Inversion Prevention via Atomic Persistence (RISK-04)
# =============================================================================

def test_remediation_cold_start_state_inversion_eliminated():
    """
    Audit Finding: RISK-04 (HIGH). Worker reboot during ongoing outage starts with start_healthy=True.
    Fix: Atomic file-backed state persistence restores outage state upon restart.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "ddir_state.json")

        # Initial instance experiences failure
        router_1 = DualHeadInferenceRouter(state_file_path=state_file, recovery_threshold=2)
        router_1.recovery_manager.reset(ReliabilityState.UNAVAILABLE)
        assert router_1.recovery_manager.current_effective_state == ReliabilityState.UNAVAILABLE

        # Verify state file was written
        assert os.path.exists(state_file)

        # Simulate worker crash and respawn (new router instance with same state file)
        router_2 = DualHeadInferenceRouter(state_file_path=state_file, recovery_threshold=2)

        # Cold-start state inversion eliminated: router_2 correctly restores UNAVAILABLE state!
        assert router_2.recovery_manager.current_effective_state == ReliabilityState.UNAVAILABLE
        assert router_2.recovery_manager.consecutive_healthy_count == 0


def test_atomic_state_persistence_file_safety():
    """
    Verifies that DDIRStatePersistence writes atomically and handles corrupt files gracefully.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.json")
        sp = DDIRStatePersistence(state_path)

        state = DDIRPersistedState(
            effective_state="STALE",
            consecutive_healthy_count=0,
            recovery_threshold=2,
            last_healthy_timestamp="2024-07-20T12:00:00Z",
            last_healthy_fingerprint="abc123",
            last_healthy_content_fingerprint="def456",
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        assert sp.save_state(state) is True
        loaded = sp.load_state()
        assert loaded is not None
        assert loaded.effective_state == "STALE"
        assert loaded.consecutive_healthy_count == 0

        # Corrupt file test
        with open(state_path, "w") as f:
            f.write("{ INVALID JSON DATA")
        corrupt_loaded = sp.load_state()
        assert corrupt_loaded is None  # Fails gracefully without throwing


# =============================================================================
# 5. Fail-Safe Exception Handling in Validation & State Machine (Tests 4.4 & 4.5)
# =============================================================================

def test_remediation_validator_exception_fail_safe():
    """
    Audit Finding: Test 4.4. Unhandled validator exceptions crash the entire inference pipeline.
    Fix: Catch unexpected validator exceptions in router, log warning, and safely demote to load-only fallback.
    """
    router = DualHeadInferenceRouter()
    raw_df = pd.DataFrame([{f: 1.0 for f in router.primary_engine.feature_names}])

    with patch.object(router.validator, "validate", side_effect=ZeroDivisionError("Internal Math Bug")):
        res = router.route_and_predict(raw_df, weather_payload={"temperature_c": 15.0})
        # Verifies fail-safe fallback: Does NOT crash; serves load-only forecast!
        assert res.active_head == "FALLBACK_LOAD_ONLY"
        assert res.routing_state == ReliabilityState.INVALID
        assert "ZeroDivisionError" in res.fallback_reason or "Internal validation error" in res.fallback_reason
        assert len(res.predictions) == len(raw_df)


def test_remediation_state_machine_exception_fail_safe():
    """
    Audit Finding: Test 4.5. Unhandled state machine exceptions crash the entire inference pipeline.
    Fix: Catch unexpected state machine exceptions in router and safely demote to fallback.
    """
    router = DualHeadInferenceRouter()
    raw_df = pd.DataFrame([{f: 1.0 for f in router.primary_engine.feature_names}])

    with patch.object(ReliabilityStateMachine, "evaluate", side_effect=TypeError("Unexpected State Machine Bug")):
        res = router.route_and_predict(raw_df, weather_payload={"temperature_c": 15.0})
        assert res.active_head == "FALLBACK_LOAD_ONLY"
        assert res.routing_state == ReliabilityState.INVALID
        assert len(res.predictions) == len(raw_df)


# =============================================================================
# 6. Prometheus Multiprocess Mode Integration (RISK-03)
# =============================================================================

def test_remediation_prometheus_multiprocess_metrics_endpoint():
    """
    Audit Finding: RISK-03 (HIGH). Multi-worker Prometheus scraping serves conflicting single-process registries.
    Fix: Support PROMETHEUS_MULTIPROC_DIR in /metrics endpoint.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": tmpdir}):
            resp = metrics()
            assert resp.status_code == 200
            assert resp.media_type.startswith("text/plain")
