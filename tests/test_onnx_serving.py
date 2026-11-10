"""
Tests for ONNX Engine and FastAPI Serving Endpoints (P5.8).
"""

import numpy as np

from chronogrid.serving.app import forecast_resilient, health, load_engine, metrics
from chronogrid.serving.onnx_engine import OnnxInferenceEngine
from chronogrid.serving.schemas import ResilientForecastRequest


def test_fastapi_health_endpoint():
    resp = health()
    assert resp.status == "healthy"
    assert resp.service == "ChronoGrid Inference Engine"


def test_fastapi_metrics_endpoint():
    resp = metrics()
    assert resp.status_code == 200
    assert b"chronogrid_inference_requests_total" in resp.body


def test_fastapi_resilient_endpoint():
    load_engine()
    req = ResilientForecastRequest(horizon_hours=24, weather_payload=None)
    resp = forecast_resilient(req)
    assert resp.routing_state == "UNAVAILABLE"
    assert resp.active_head == "FALLBACK_LOAD_ONLY"
    assert len(resp.forecast_load_mw) == 24
    assert resp.forecast_load_mw[0] > 0


def test_onnx_engine_predictions():
    engine = OnnxInferenceEngine()
    dummy_input = [{f: 1.0 for f in engine.feature_names}]
    preds = engine.predict(dummy_input)
    assert len(preds) == 1
    assert isinstance(preds[0], (float, np.floating))
