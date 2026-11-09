"""Pre-inference validation of observed load vintages; no imputation or revision merging."""
from datetime import datetime, timedelta
import math

from src.evaluation.prospective import utc


def observed_history(data: dict, source: dict, issued_at: datetime) -> dict[datetime, float]:
    """Validate local first-seen evidence, not an independent publication-time proof.

    Each timestamp labels the start of a completed one-hour observation. Duplicate
    instants are rejected even when expressed using different timezone offsets.
    Future weather forecasts require a different contract and cannot use this one.
    """
    issued = utc(issued_at)
    available = utc(datetime.fromisoformat(source['available_at']))
    captured = utc(datetime.fromisoformat(source['captured_at']))
    if available > captured or captured > issued:
        raise ValueError('source unavailable at issuance or captured before availability')
    if (source['source'] != 'smard-normalized-hourly'
            or data.get('format') != 'smard-hourly-mw-v1' or data.get('unit') != 'MW'):
        raise ValueError('source contract mismatch')
    rows = {}
    for row in data['rows']:
        timestamp = utc(datetime.fromisoformat(row['timestamp']))
        value = row['load_mw']
        if timestamp.minute or timestamp.second or timestamp.microsecond:
            raise ValueError('non-hourly source observation')
        if timestamp in rows:
            raise ValueError('duplicate source instant; revisions require a separate snapshot')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('invalid numeric load observation')
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid numeric load observation')
        if timestamp + timedelta(hours=1) > available:
            raise ValueError('unfinished/future source observation')
        rows[timestamp] = float(value)
    if not rows:
        raise ValueError('empty source history')
    if issued - (max(rows) + timedelta(hours=1)) > timedelta(hours=48):
        raise ValueError('load source is stale (>48 hours)')
    return rows
