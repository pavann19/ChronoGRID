"""Read hash-addressed historical source evidence; never execute it."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def historical_source(path, expected_sha256):
    current = ROOT / path
    raw = current.read_bytes() if current.is_file() else b""
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raw = (
            ROOT / "artifacts/reports/portfolio_cleanup/frozen_sources" / (expected_sha256 + ".py")
        ).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Historical source checksum mismatch")
    return raw
