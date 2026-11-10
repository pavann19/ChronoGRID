import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chronogrid.operations.artifact_registry import ArtifactRegistry, sha
from chronogrid.operations.registry_cli import main
from chronogrid.serving import app as service

ROOT = Path(__file__).resolve().parents[1]


def register(registry, name, note=None):
    model = (ROOT / "artifacts/models/champion_model.onnx").read_bytes()
    metadata = (ROOT / "artifacts/models/champion_metadata.json").read_bytes()
    if note:
        metadata = json.dumps(json.loads(metadata) | dict(review_note=note)).encode()
    return registry.register(
        name,
        model,
        metadata,
        expected_model_sha256=sha(model),
        expected_metadata_sha256=sha(metadata),
    )


def test_optin_registry_serves_one_bundle_without_legacy_database_initialization(
    tmp_path, monkeypatch
):
    registry = ArtifactRegistry.create(tmp_path / "registry.sqlite")
    a = register(registry, name="ChronoGrid_LoadForecaster")
    registry.activate("ChronoGrid_LoadForecaster", a, expected_current=None)
    for name in ("ENGINE", "ROUTER", "DQCP_ENGINE", "CHAMPION_META"):
        monkeypatch.setattr(service, name, getattr(service, name))
    monkeypatch.setenv("CHRONOGRID_ARTIFACT_REGISTRY", str(registry.path))
    monkeypatch.setenv("CHRONOGRID_DIAGNOSTICS_TOKEN", "isolated-registry-test")

    def forbidden():
        raise AssertionError("opt-in registry must not initialize the legacy application DB")

    monkeypatch.setattr(service, "init_db", forbidden)
    with TestClient(
        service.app, headers={"Authorization": "Bearer isolated-registry-test"}
    ) as client:
        info = client.get("/model/info").json()
        assert info["champion_metadata"]["bundle_id"] == a
        assert info["champion_metadata"]["registry_identity_verified"] is True
        assert service.ROUTER is None and service.DQCP_ENGINE is None
        response = client.post(
            "/predict", json={"features": [{n: 0 for n in service.ENGINE.feature_names}]}
        )
        assert response.status_code == 200
        assert response.json()["evidence_kind"] == "caller_supplied_features"
        assert response.headers["X-ChronoGrid-Feature-Parity"] == "not-established"
        b = register(registry, name="ChronoGrid_LoadForecaster", note="new immutable revision")
        registry.activate("ChronoGrid_LoadForecaster", b, expected_current=a)
        assert client.get("/model/info").json()["champion_metadata"]["bundle_id"] == a


def test_registry_load_error_fails_closed_without_legacy_fallback(tmp_path, monkeypatch):
    for name in ("ENGINE", "ROUTER", "DQCP_ENGINE", "CHAMPION_META"):
        monkeypatch.setattr(service, name, getattr(service, name))
    missing = tmp_path / "missing.sqlite"
    monkeypatch.setenv("CHRONOGRID_ARTIFACT_REGISTRY", str(missing))
    with pytest.raises(FileNotFoundError):
        service.load_engine()
    assert service.ENGINE is None and service.ROUTER is None and service.DQCP_ENGINE is None
    assert not missing.exists()


def test_registry_cli_requires_explicit_create_and_does_not_overwrite(tmp_path, capsys):
    path = tmp_path / "registry.sqlite"
    assert main(["--db", str(path), "verify"]) == 2
    assert not path.exists()
    assert main(["--db", str(path), "init"]) == 0
    before = path.read_bytes()
    assert main(["--db", str(path), "init"]) == 2
    assert path.read_bytes() == before
