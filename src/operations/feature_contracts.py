"""Runtime binding and explicit dependencies for the two frozen load-only adapters.

The sidecar records a present-day review of existing artifacts. It is not evidence
that the original training run recorded these complete bindings.
"""
from datetime import timedelta
import hashlib
import json
from pathlib import Path

from src.evaluation.prospective import utc

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = 'src/operations/feature_contract_bindings.json'
SOURCE_PATHS = ('src/features/engineer.py', 'src/features/temporal.py',
                'src/features/autoregressive.py', 'src/features/weather_features.py',
                'src/operations/local_calendar.py', 'src/operations/calendar_support.py',
                'src/operations/availability.py', 'src/operations/feature_contracts.py')
CALENDAR = ('sin_hour', 'cos_hour', 'sin_dayofweek', 'cos_dayofweek',
            'sin_month', 'cos_month', 'sin_dayofyear', 'cos_dayofyear',
            'is_weekend', 'is_holiday', 'is_working_day')


def feature_dependencies():
    """Elapsed UTC hours before delivery; observations label interval starts.

    The legacy diff_24_168 name actually subtracts lag168 from lag48 under
    min_lag_hours=48. Preserve that behavior and disclose it explicitly.
    """
    rules = {name: [] for name in CALENDAR}
    rules.update({f'grid_load_mw_lag_{lag}': [lag] for lag in (48, 72, 168)})
    for window in (6, 12, 24, 168):
        for statistic in ('mean', 'std', 'min', 'max'):
            rules[f'grid_load_mw_roll_{statistic}_{window}'] = list(range(48, 48 + window))
    rules['grid_load_mw_diff_24_168'] = [48, 168]
    return rules


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_binding(experiment_dir, metadata, *, root=ROOT):
    """Reject metadata, artifact, policy, schema or source drift before unpickling."""
    root = Path(root)
    raw = (root / MANIFEST).read_bytes()
    manifest = json.loads(raw)
    policy = metadata['feature_policy']
    binding = manifest['policies'].get(policy)
    if manifest['schema_version'] != '1.0' or binding is None:
        raise ValueError('feature contract binding unavailable')
    if (set(binding['artifacts']) != {'fallback.joblib', 'fallback_calibration.json', 'fallback_metadata.json'}
            or set(binding['sources']) != set(SOURCE_PATHS)):
        raise ValueError('incomplete feature contract binding inventory')
    for name, expected in binding['artifacts'].items():
        artifact = (Path(experiment_dir) / name).read_bytes()
        if sha(artifact) != expected:
            raise ValueError('feature contract artifact/metadata binding mismatch')
        if name == 'fallback_metadata.json' and json.loads(artifact) != metadata:
            raise ValueError('parsed metadata binding mismatch')
    for name, expected in binding['sources'].items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError('feature implementation checksum binding mismatch')
    rules = feature_dependencies()
    if (binding['dependencies'] != rules or metadata['feature_names'] != list(rules)
            or metadata['min_lag_hours'] != 48 or metadata['mode'] != 'load_only'):
        raise ValueError('feature contract schema/dependencies mismatch')
    code_hash = sha(json.dumps(binding['sources'], sort_keys=True, separators=(',', ':')).encode())
    return binding, raw, code_hash


def validate_dependencies(binding, deliveries, history, available_at, issued_at):
    """Every required value must exist and its hourly interval must be complete."""
    available, issued = utc(available_at), utc(issued_at)
    if available > issued:
        raise ValueError('feature source unavailable at issuance')
    if binding['dependencies'] != feature_dependencies():
        raise ValueError('unrecognized feature dependency contract')
    checked = 0
    for delivery in deliveries:
        delivery = utc(delivery)
        for feature, offsets in binding['dependencies'].items():
            for offset in offsets:
                source_time = delivery - timedelta(hours=offset)
                if source_time + timedelta(hours=1) > available:
                    raise ValueError(f'{feature}: dependency unavailable at issuance')
                if source_time not in history:
                    raise ValueError(f'{feature}: missing source dependency; no imputation')
                checked += 1
    return checked
