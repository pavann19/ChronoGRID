"""Model identity checks must run before any adaptive-state loading."""
import hashlib
import json
from pathlib import Path

import pytest

from src.uncertainty.engine import load_dqcp_engine

ROOT = Path(__file__).resolve().parents[1]


def fixture_artifacts(tmp_path):
    paths = {}
    for head in ('primary', 'fallback'):
        model = tmp_path / f'{head}.onnx'
        model.write_bytes(f'synthetic model {head}'.encode())
        paths[head] = model
        table = json.loads((ROOT / f'artifacts/calibration/{head}_calibration_table.json').read_text())
        table['model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
        (tmp_path / f'{head}_calibration_table.json').write_text(json.dumps(table))
    return paths


def test_matching_models_load_without_changing_quantiles(tmp_path):
    paths = fixture_artifacts(tmp_path)
    engine = load_dqcp_engine(tmp_path, tmp_path / 'state.json', paths)
    assert engine is not None
    original = json.loads((tmp_path / 'primary_calibration_table.json').read_text())
    assert engine.primary_pool.precomputed_marginal_quantiles == original['marginal_quantiles']
    assert not list(tmp_path.glob('state*'))


@pytest.mark.parametrize('head', ['primary', 'fallback'])
def test_tampered_model_rejected_before_state_loading(tmp_path, head):
    paths = fixture_artifacts(tmp_path)
    paths[head].write_bytes(b'tampered')
    # Invalid state would fail if loaded; identity rejection must happen first.
    state = tmp_path / 'state_primary.json'
    state.write_text('invalid state')
    assert load_dqcp_engine(tmp_path, tmp_path / 'state.json', paths) is None
    assert state.read_text() == 'invalid state'


def test_missing_artifact_and_incomplete_mapping_rejected(tmp_path):
    paths = fixture_artifacts(tmp_path)
    assert load_dqcp_engine(tmp_path, model_paths={'primary': paths['primary']}) is None
    paths['fallback'].unlink()
    assert load_dqcp_engine(tmp_path, model_paths=paths) is None


def test_frozen_tables_refuse_legacy_served_models(tmp_path):
    assert load_dqcp_engine(aci_state_file=tmp_path / 'state.json') is None
    assert not list(tmp_path.iterdir())
