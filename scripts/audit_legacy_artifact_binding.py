"""Audit legacy model/calibration artifact binding without modifying inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts/engineering/v1_improvement_phase/legacy_artifact_binding_audit_v1.json"
)
ARM_INPUTS = {
    "primary": {
        "served_model": "artifacts/models/champion_model.onnx",
        "research_model": "artifacts/models/research/research_primary_model_thru_20240331.joblib",
        "calibration": "artifacts/calibration/primary_calibration_table.json",
    },
    "fallback": {
        "served_model": "artifacts/models/load_only_model.onnx",
        "research_model": "artifacts/models/research/research_fallback_model_thru_20240331.joblib",
        "calibration": "artifacts/calibration/fallback_calibration_table.json",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def inspect_registry(database_path: Path) -> dict[str, Any]:
    before_hash = sha256_file(database_path)
    uri_path = quote(database_path.resolve().as_posix(), safe="/:")
    connection = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
            ("table", "model_registry"),
        ).fetchone()
        if table is None:
            raise ValueError("model_registry table is missing")

        columns = [row["name"] for row in connection.execute('PRAGMA table_info("model_registry")')]
        indexes: list[dict[str, Any]] = []
        for row in connection.execute('PRAGMA index_list("model_registry")'):
            index_name = row["name"]
            index_columns = [
                column["name"]
                for column in connection.execute(
                    f"PRAGMA index_info({_quote_identifier(index_name)})"
                )
            ]
            definition = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
                ("index", index_name),
            ).fetchone()
            indexes.append(
                {
                    "name": index_name,
                    "unique": bool(row["unique"]),
                    "partial": bool(row["partial"]),
                    "columns": index_columns,
                    "sql": definition["sql"] if definition else None,
                }
            )

        champion_guard = False
        for index in indexes:
            columns_in_index = index["columns"]
            index_sql = (index["sql"] or "").lower().replace(" ", "")
            if not index["unique"]:
                continue
            enforces_champion_uniqueness = (
                (columns_in_index == ["model_name"] and not index["partial"])
                or set(columns_in_index) == {"model_name", "is_champion"}
                or (
                    columns_in_index == ["model_name"]
                    and index["partial"]
                    and "is_champion" in index_sql
                    and "=1" in index_sql
                )
            )
            if enforces_champion_uniqueness:
                champion_guard = True

        row_count = connection.execute('SELECT COUNT(*) FROM "model_registry"').fetchone()[0]
        stage_counts: dict[str, int] = {}
        if "stage" in columns:
            for row in connection.execute(
                'SELECT "stage", COUNT(*) AS count FROM "model_registry" GROUP BY "stage" ORDER BY "stage"'
            ):
                stage_counts[str(row["stage"])] = int(row["count"])
        champion_count = None
        if "is_champion" in columns:
            champion_count = int(
                connection.execute(
                    'SELECT COUNT(*) FROM "model_registry" WHERE "is_champion" = 1'
                ).fetchone()[0]
            )
        champion_metadata_has_onnx_hash = None
        if {"is_champion", "metadata_json"}.issubset(columns):
            champion_metadata_has_onnx_hash = False
            metadata_rows = connection.execute(
                'SELECT "metadata_json" FROM "model_registry" WHERE "is_champion" = 1'
            )
            for row in metadata_rows:
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                champion_metadata_has_onnx_hash = champion_metadata_has_onnx_hash or bool(
                    metadata.get("onnx_sha256")
                )
    finally:
        connection.close()

    after_hash = sha256_file(database_path)
    return {
        "path": database_path.relative_to(ROOT).as_posix()
        if database_path.is_relative_to(ROOT)
        else str(database_path),
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "bytes_unchanged": before_hash == after_hash,
        "opened_read_only": True,
        "row_count": int(row_count),
        "stage_counts": stage_counts,
        "champion_row_count": champion_count,
        "columns": columns,
        "has_onnx_sha256_column": "onnx_sha256" in columns,
        "champion_metadata_has_onnx_sha256": champion_metadata_has_onnx_hash,
        "indexes": indexes,
        "unique_champion_constraint_detected": champion_guard,
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm, inputs in ARM_INPUTS.items():
        served_path = root / inputs["served_model"]
        research_path = root / inputs["research_model"]
        calibration_path = root / inputs["calibration"]
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        served_hash = sha256_file(served_path)
        research_hash = sha256_file(research_path)
        calibration_hash = calibration.get("model_sha256")
        arms[arm] = {
            "served_model": {"path": inputs["served_model"], "sha256": served_hash},
            "research_model": {"path": inputs["research_model"], "sha256": research_hash},
            "calibration": {
                "path": inputs["calibration"],
                "sha256": sha256_file(calibration_path),
                "model_sha256": calibration_hash,
                "feature_code_sha256": calibration.get("feature_code_sha256"),
            },
            "calibration_matches_served_model": calibration_hash == served_hash,
            "calibration_matches_research_model": calibration_hash == research_hash,
        }

    registry = inspect_registry(root / "data/chronogrid.db")
    all_served_bindings_match = all(
        arm["calibration_matches_served_model"] for arm in arms.values()
    )
    feature_bindings_present = all(
        bool(arm["calibration"]["feature_code_sha256"]) for arm in arms.values()
    )
    binding_controls_present = (
        registry["has_onnx_sha256_column"]
        and registry["champion_metadata_has_onnx_sha256"] is True
        and registry["unique_champion_constraint_detected"]
    )
    passed = (
        all_served_bindings_match
        and feature_bindings_present
        and binding_controls_present
        and registry["bytes_unchanged"]
    )
    return {
        "format": "chronogrid-legacy-artifact-binding-audit-v1",
        "audit_date_utc": datetime.now(UTC).date().isoformat(),
        "scope": "Read-only checksum and registry-schema audit of the legacy serving/calibration artifacts. No models, calibration tables, labels, or database rows were modified.",
        "status": "PASS" if passed else "FAIL_BINDING_NOT_ESTABLISHED",
        "arms": arms,
        "registry": registry,
        "checks": {
            "calibration_hash_matches_served_model_for_all_arms": all_served_bindings_match,
            "feature_code_hash_present_for_all_arms": feature_bindings_present,
            "registry_enforces_artifact_hash_and_unique_champion": binding_controls_present,
            "registry_bytes_unchanged": registry["bytes_unchanged"],
        },
        "limitations": [
            "This report establishes current file-hash and SQLite-schema relationships only; it does not exercise a running service or prove which file a deployed process has loaded.",
            "The audit does not validate calibration methodology, interval coverage, feature availability, runtime reload behavior, or production security controls.",
            "A failed binding check is a legacy production-readiness blocker, not a reason to alter the frozen research experiment.",
        ],
    }


def write_report(output: Path, report: dict[str, Any]) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    digest = write_report(args.output, report)
    print(f"Legacy artifact binding: {report['status']}; sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
