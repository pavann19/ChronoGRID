from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chronogrid.serving import app as service
from chronogrid.serving.registry_binding import verify_model_binding


@pytest.mark.parametrize(
    "method,path,token",
    [
        ("post", "/predict", "CHRONOGRID_DIAGNOSTICS_TOKEN"),
        ("get", "/model/info", "CHRONOGRID_DIAGNOSTICS_TOKEN"),
        ("get", "/metrics", "CHRONOGRID_DIAGNOSTICS_TOKEN"),
        ("post", "/forecast/day-ahead", "CHRONOGRID_LEGACY_REPLAY_TOKEN"),
        ("post", "/forecast/resilient", "CHRONOGRID_LEGACY_REPLAY_TOKEN"),
        ("post", "/forecast/dqcp", "CHRONOGRID_LEGACY_REPLAY_TOKEN"),
        ("post", "/forecast/dqcp/feedback", "CHRONOGRID_LEGACY_REPLAY_TOKEN"),
        ("get", "/forecast/future-route", "CHRONOGRID_LEGACY_REPLAY_TOKEN"),
    ],
)
def test_all_forecast_and_diagnostic_routes_require_auth(monkeypatch, method, path, token):
    client = TestClient(service.app)
    monkeypatch.delenv(token, raising=False)
    assert getattr(client, method)(path).status_code == 503
    monkeypatch.setenv(token, "test-secret")
    assert getattr(client, method)(path).status_code == 401
    assert (
        getattr(client, method)(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    )
    assert client.get("/health").status_code == 200


def test_onnx_tampering_rejected_against_registry(tmp_path):
    root = Path(__file__).resolve().parents[1]
    model = root / "artifacts/models/champion_model.onnx"
    metadata = root / "artifacts/models/champion_metadata.json"
    verify_model_binding(model, metadata)
    altered = tmp_path / model.name
    altered.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_model_binding(altered, metadata)


def test_holidays_outside_historical_table():
    import pandas as pd

    from chronogrid.features.temporal import add_temporal_features

    frame = add_temporal_features(
        pd.DataFrame(
            {"timestamp": pd.to_datetime(["2031-12-25T12:00Z", "2031-12-24T12:00Z"], utc=True)}
        )
    )
    assert frame.is_holiday.tolist() == [1, 0]
