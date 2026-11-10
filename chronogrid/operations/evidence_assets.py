"""Locate optional historical model-audit inputs outside the tracked tree."""

import os
from pathlib import Path


def external_evidence_root():
    return Path(
        os.environ.get(
            "CHRONOGRID_EVIDENCE_ROOT",
            Path(__file__).resolve().parents[2] / ".tools/evidence/restored",
        )
    )
