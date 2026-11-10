"""Deterministic actual linkage from captured, unit-checked SMARD snapshots."""

import json
import math
from datetime import datetime, timedelta

from chronogrid.evaluation.prospective import ActualObservation, utc
from chronogrid.operations.contracts import Forecast
from chronogrid.operations.journal import Conflict


def link_smard(store, run_id, snapshot_sha256):
    store.get("run:" + run_id)
    snap = store.get("snapshot:" + snapshot_sha256)
    data = json.loads(store.snapshot_bytes(snapshot_sha256))
    if (
        data.get("format") != "smard-hourly-mw-v1"
        or data.get("unit") != "MW"
        or snap["source"] != "smard-normalized-hourly"
    ):
        raise ValueError("wrong observation source contract")
    available = utc(datetime.fromisoformat(snap["available_at"]))
    values = {}
    for row in data["rows"]:
        t = utc(datetime.fromisoformat(row["timestamp"]))
        raw = row["load_mw"]
        if (
            isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or float(raw) < 0
            or t in values
            or t.minute
            or t.second
            or t.microsecond
        ):
            raise ValueError("invalid/duplicate hourly actual observation")
        if t + timedelta(hours=1) > available:
            raise ValueError("snapshot contains unfinished observation")
        values[t] = float(raw)
    linked = duplicates = unavailable = 0
    for row in store.rows("forecast"):
        p = Forecast.model_validate(row["payload"])
        if p.run_id != run_id:
            continue
        for delivery in p.deliveries:
            t = utc(delivery.timestamp)
            if t not in values:
                unavailable += 1
                continue  # unknown is not zero and not a final missing vintage
            a = ActualObservation(
                prediction_sha256=row["sha256"],
                delivery_timestamp=t,
                observation_end=t + timedelta(hours=1),
                available_at=available,
                source=snap["source"],
                source_snapshot_sha256=snapshot_sha256,
                revision=snapshot_sha256[:32],
                value_mw=values[t],
                status="observed",
                quality_flags=(),
            )
            try:
                store.actual(p.run_id, p.delivery_date, p.arm, a)
                linked += 1
            except Conflict:
                duplicates += 1
    return dict(
        linked=linked,
        duplicate_revisions=duplicates,
        unavailable_hours=unavailable,
        source_snapshot=snapshot_sha256,
    )
