from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from scripts.audit_legacy_artifact_binding import ARM_INPUTS, build_report


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_root(root: Path) -> Path:
    for arm, inputs in ARM_INPUTS.items():
        served = root / inputs["served_model"]
        research = root / inputs["research_model"]
        calibration = root / inputs["calibration"]
        served.parent.mkdir(parents=True, exist_ok=True)
        research.parent.mkdir(parents=True, exist_ok=True)
        calibration.parent.mkdir(parents=True, exist_ok=True)
        served.write_bytes(f"served-{arm}".encode())
        research.write_bytes(f"research-{arm}".encode())
        calibration.write_text(
            json.dumps(
                {
                    "model_sha256": _sha(research),
                    "feature_code_sha256": None,
                }
            ),
            encoding="utf-8",
        )

    database = root / "data/chronogrid.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE model_registry (model_name TEXT, stage TEXT, is_champion INTEGER, metadata_json TEXT)"
        )
        connection.execute(
            "INSERT INTO model_registry VALUES (?, ?, ?, ?)",
            ("primary", "Production", 1, json.dumps({"onnx_sha256": ""})),
        )
    return database


def test_report_separates_research_binding_from_served_model_binding(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    database = _audit_root(root)

    report = build_report(root)

    assert report["status"] == "FAIL_BINDING_NOT_ESTABLISHED"
    assert report["arms"]["primary"]["calibration_matches_research_model"] is True
    assert report["arms"]["primary"]["calibration_matches_served_model"] is False
    assert report["arms"]["fallback"]["calibration_matches_research_model"] is True
    assert report["arms"]["fallback"]["calibration_matches_served_model"] is False
    assert report["registry"]["has_onnx_sha256_column"] is False
    assert _sha(database) == report["registry"]["sha256_after"]


def test_registry_audit_is_read_only(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    database = _audit_root(root)
    before = _sha(database)

    report = build_report(root)

    assert report["registry"]["opened_read_only"] is True
    assert report["registry"]["bytes_unchanged"] is True
    assert _sha(database) == before
