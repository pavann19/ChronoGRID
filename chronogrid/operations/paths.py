"""Explicit asset workspace resolution for source and installed-code execution."""
import os
from pathlib import Path

def workspace(path=None,required=True):
    selected=path if path is not None else os.getenv('CHRONOGRID_WORKSPACE')
    candidate=Path(selected).expanduser().resolve() if selected is not None else Path(__file__).resolve().parents[2]
    if (candidate/'frontend/operations/index.html').is_file() and (candidate/'artifacts/reports/research_closure_evidence_manifest.json').is_file():
        return candidate
    if selected is not None or required:
        raise ValueError('Research assets unavailable: provide --workspace / CHRONOGRID_WORKSPACE pointing to a verified ChronoGrid checkout or asset workspace')
    return None
