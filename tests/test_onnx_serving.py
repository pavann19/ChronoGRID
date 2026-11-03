"""
Tests for ONNX Engine and FastAPI Serving Endpoints (P5.8).
"""

import numpy as np
from src.serving.app import health, metrics
from src.serving.onnx_engine import OnnxInferenceEngine


def test_fastapi_health_endpoint():
    resp = health()
    assert resp.status == "healthy"
    assert resp.service == "ChronoGrid Inference Engine"


def test_fastapi_metrics_endpoint():
    resp = metrics()
    assert resp.status_code == 200
    assert b"chronogrid_inference_requests_total" in resp.body


def test_onnx_engine_predictions():
    engine = OnnxInferenceEngine()
    dummy_input = [{f: 1.0 for f in engine.feature_names}]
    preds = engine.predict(dummy_input)
    assert len(preds) == 1
    assert isinstance(preds[0], (float, np.floating))
