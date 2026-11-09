"""Issuance boundaries use absolute instants, including German DST transitions."""
from datetime import datetime, timezone

import pytest

from src.operations.availability import observed_history


def evidence():
    source = dict(source='smard-normalized-hourly', available_at='2026-10-25T02:00:00Z',
                  captured_at='2026-10-25T02:00:00Z')
    data = dict(format='smard-hourly-mw-v1', unit='MW', rows=[
        dict(timestamp='2026-10-25T02:00:00+02:00', load_mw=100),
        dict(timestamp='2026-10-25T02:00:00+01:00', load_mw=101),
    ])
    return data, source, datetime(2026, 10, 25, 2, tzinfo=timezone.utc)


def test_dst_repeated_local_hour_has_two_distinct_utc_observations():
    rows = observed_history(*evidence())
    assert [t.hour for t in rows] == [0, 1]
    assert list(rows.values()) == [100, 101]


@pytest.mark.parametrize('field,value', [
    ('captured_at', '2026-10-25T02:00:01Z'),
    ('available_at', '2026-10-25T02:00:01Z'),
    ('available_at', '2026-10-25T02:00:00'),
])
def test_future_or_naive_provenance_rejected(field, value):
    data, source, issued = evidence()
    source[field] = value
    with pytest.raises(ValueError):
        observed_history(data, source, issued)


@pytest.mark.parametrize('value', [True, '100', None, -1, float('nan'), float('inf')])
def test_bad_load_cannot_reach_feature_generation(value):
    data, source, issued = evidence()
    data['rows'][0]['load_mw'] = value
    with pytest.raises(ValueError, match='numeric'):
        observed_history(data, source, issued)


@pytest.mark.parametrize('timestamp,match', [
    ('2026-10-25T00:00:00Z', 'duplicate'),
    ('2026-10-25T02:00:00Z', 'future'),
    ('2026-10-25T01:30:00Z', 'non-hourly'),
    ('2026-10-25T01:00:00', 'timezone-aware'),
])
def test_invalid_observation_timestamps_rejected(timestamp, match):
    data, source, issued = evidence()
    data['rows'][1]['timestamp'] = timestamp
    with pytest.raises(ValueError, match=match):
        observed_history(data, source, issued)


def test_empty_and_stale_sources_rejected():
    data, source, issued = evidence()
    with pytest.raises(ValueError, match='stale'):
        observed_history(data, source, datetime(2026, 10, 28, tzinfo=timezone.utc))
    data['rows'] = []
    with pytest.raises(ValueError, match='empty'):
        observed_history(data, source, issued)
