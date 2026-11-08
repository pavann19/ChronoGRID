"""
Tests for ChronoGrid DDIR Historical evaluation: Deployment & Multi-Process Resilience Audit.
Verifies multi-process isolation, cold-start state inversion, frozen sensor bypass,
future timestamp validation anomaly, observability blast radius, and bounded metric cardinality.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest
import pandas as pd

from src.reliability.router import DualHeadInferenceRouter
from src.reliability.validator import WeatherInputValidator
from src.reliability.state_machine import ReliabilityState
from src.serving.app import load_engine, forecast_resilient, DDIR_REQUEST_COUNT
from src.serving.schemas import ResilientForecastRequest


@pytest.fixture(scope="module")
def initialized_engine():
    load_engine()


def test_multi_process_state_isolation():
    # Simulate two independent router instances as would exist in separate processes
    router_a = DualHeadInferenceRouter(
        primary_onnx_path="artifacts/models/champion_model.onnx",
        primary_metadata_path="artifacts/models/champion_metadata.json",
        fallback_onnx_path="artifacts/models/load_only_model.onnx",
        fallback_metadata_path="artifacts/models/load_only_metadata.json",
    )
    router_b = DualHeadInferenceRouter(
        primary_onnx_path="artifacts/models/champion_model.onnx",
        primary_metadata_path="artifacts/models/champion_metadata.json",
        fallback_onnx_path="artifacts/models/load_only_model.onnx",
        fallback_metadata_path="artifacts/models/load_only_metadata.json",
    )

    # Inject failure into router_a only
    router_a.recovery_manager.reset(ReliabilityState.UNAVAILABLE)

    # Verify state isolation: router_b is unaffected and remains HEALTHY
    assert router_a.recovery_manager.current_effective_state == ReliabilityState.UNAVAILABLE
    assert router_b.recovery_manager.current_effective_state == ReliabilityState.HEALTHY


def test_cold_start_state_inversion_vulnerability():
    # Freshly instantiated router defaults to start_healthy=True
    router = DualHeadInferenceRouter(
        primary_onnx_path="artifacts/models/champion_model.onnx",
        primary_metadata_path="artifacts/models/champion_metadata.json",
        fallback_onnx_path="artifacts/models/load_only_model.onnx",
        fallback_metadata_path="artifacts/models/load_only_metadata.json",
    )
    assert router.recovery_manager.current_effective_state == ReliabilityState.HEALTHY
    assert router.recovery_manager.consecutive_healthy_count == 2


def test_frozen_sensor_timestamp_remediation():
    router = DualHeadInferenceRouter(
        primary_onnx_path="artifacts/models/champion_model.onnx",
        primary_metadata_path="artifacts/models/champion_metadata.json",
        fallback_onnx_path="artifacts/models/load_only_model.onnx",
        fallback_metadata_path="artifacts/models/load_only_metadata.json",
        recovery_threshold=2,
    )
    router.recovery_manager.reset(ReliabilityState.UNAVAILABLE)

    now_ref = datetime(2026, 11, 3, 16, 0, 0, tzinfo=timezone.utc)
    raw_df = pd.DataFrame([{f: 1.0 for f in router.primary_engine.feature_names}])

    # Sensor reading is frozen, but timestamps advance
    frozen = {"temperature_c": 15.0, "wind_speed_ms": 4.0, "solar_radiation_wm2": 250.0, "humidity_pct": 60.0}
    r1 = router.route_and_predict(raw_df, weather_payload={"timestamp": (now_ref - timedelta(hours=2)).isoformat(), **frozen}, current_time=now_ref)
    assert r1.active_head == "FALLBACK_LOAD_ONLY"
    assert r1.consecutive_healthy_count == 1

    r2 = router.route_and_predict(raw_df, weather_payload={"timestamp": (now_ref - timedelta(hours=1)).isoformat(), **frozen}, current_time=now_ref)
    # Remediation Verified: Identical physical readings with advancing timestamps are flagged as static/frozen!
    # Recovery is held on Fallback, preventing frozen sensor from hijacking Primary!
    assert r2.active_head == "FALLBACK_LOAD_ONLY"
    assert r2.consecutive_healthy_count == 1
    assert "frozen" in r2.fallback_reason.lower() or "variance" in r2.fallback_reason.lower()


def test_future_timestamp_validation_remediation():
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

    # Remediation Verified: Future timestamp is rejected with invalid temporal provenance
    assert report.is_valid is False
    assert report.quality_score == 0.0
    assert any("future" in v.lower() for v in report.violations)


def test_observability_unhandled_exception_blast_radius_remediation(initialized_engine):
    req = ResilientForecastRequest(
        start_timestamp="2024-07-20T14:00:00Z",
        horizon_hours=24,
        weather_payload=None
    )
    # Remediation Verified: Even if Prometheus throws an internal error, the forecast request succeeds!
    with patch.object(DDIR_REQUEST_COUNT, "labels", side_effect=RuntimeError("Prometheus Lock Error")):
        resp = forecast_resilient(req)
        assert len(resp.forecast_load_mw) == 24
        assert resp.active_head == "FALLBACK_LOAD_ONLY"


def test_metric_cardinality_is_strictly_bounded():
    labels = DDIR_REQUEST_COUNT._labelnames
    assert labels == ("state", "head")
    # 5 reliability states * 2 active heads = 10 series maximum
    assert len(labels) == 2
    assert "timestamp" not in labels
    assert "request_id" not in labels

