"""Verify model and metadata bytes against the reviewed local registry."""

import hashlib
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "artifacts/models/serving_registry.json"


def verify_model_binding(model_path, metadata_path, registry_path=None):
    registry = json.loads(Path(registry_path or REGISTRY).read_bytes())
    for path in (Path(model_path), Path(metadata_path)):
        expected = registry["sha256"].get(path.name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected or actual != expected:
            raise ValueError(f"Serving registry SHA-256 mismatch: {path.name}")
