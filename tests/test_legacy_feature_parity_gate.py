from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
import json

import pytest

from src.operations.capture import delivery_grid
from src.serving.feature_parity import unavailable_features, legacy_dependencies, require_legacy_prospective_parity
from scripts.audit_legacy_feature_parity import audit, ROOT


@pytest.mark.parametrize('day,unavailable', [('2024-06-17', 12), ('2024-03-31', 11), ('2024-10-27', 13)])
def test_completed_hour_lag24_boundary_across_dst(day, unavailable):
    day = date.fromisoformat(day)
    gate = datetime.combine(day - timedelta(days=1), time(12), ZoneInfo('Europe/Berlin'))
    names = ['grid_load_mw_lag_24']
    assert sum(bool(unavailable_features(names, ts, gate)) for ts in delivery_grid(day)) == unavailable


def test_unknown_feature_and_unverified_provenance_fail_closed():
    with pytest.raises(ValueError, match='unknown'):
        legacy_dependencies(['invented'])
    report = dict(decision='PASS', unavailable_delivery_hours=0, mismatched_cells=0,
                  original_training_feature_provenance_verified=False)
    with pytest.raises(ValueError, match='not established'):
        require_legacy_prospective_parity(report)


def test_frozen_feature_paths_still_expose_the_measured_mismatch():
    before = (ROOT / 'src/features/engineer.py').read_bytes()
    report, rows = audit()
    assert report['per_head']['primary']['feature_count'] == 43
    assert report['per_head']['fallback']['feature_count'] == 34
    assert report['unavailable_delivery_hours'] == 12
    assert report['unavailable_head_delivery_hours'] == 24
    assert report['mismatched_cells'] > 0
    assert any(not r['numerical_match'] and r['dependency_available_at_gate'] for r in rows)
    assert (ROOT / 'src/features/engineer.py').read_bytes() == before
    with pytest.raises(ValueError):
        require_legacy_prospective_parity(report)
