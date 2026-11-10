"""Locate optional historical bundles and unpack the retained weather test fixture."""

import hashlib
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEATHER_ARCHIVE_SHA256 = "b9d4bd1e50c8f5710af11ed8329ab91c99e5ea341656f0089f255db14d32a9cf"


def external_evidence_root():
    return Path(os.environ.get("CHRONOGRID_EVIDENCE_ROOT", str(ROOT / ".tools/evidence/restored")))


def weather_replay_root():
    root = ROOT / "artifacts/experiments/weather_v4"
    archive = root / "replay_evidence.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != WEATHER_ARCHIVE_SHA256:
        raise ValueError("Weather replay archive checksum mismatch")
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = (root / member.filename).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Unsafe weather archive path")
            raw = source.read(member)
            if path.exists():
                if path.read_bytes() != raw:
                    raise ValueError("Weather replay fixture drift: " + member.filename)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
    return root
