import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_operations_evidence import forecast, setup

from chronogrid.operations.journal import Journal, encode
from chronogrid.operations.observations import link_smard
from chronogrid.operations.reporting import score_run
from chronogrid.operations.worker import run_cycle

ROOT = Path(__file__).resolve().parents[1]


def test_actual_collector_revision_linkage_cycle_after_window_and_cli(tmp_path):
    j, clock, run, source = setup(tmp_path)
    p = forecast(j, clock, source)
    key = j.commit(p)
    clock.t = datetime(2027, 1, 16, 12, tzinfo=timezone.utc)
    rows = [dict(timestamp=d.timestamp.isoformat(), load_mw=100) for d in p.deliveries]
    snapshot = j.snapshot(
        encode(dict(format="smard-hourly-mw-v1", unit="MW", rows=rows)),
        "smard-normalized-hourly",
        clock(),
    )
    first = link_smard(j, "run1", snapshot)
    second = link_smard(j, "run1", snapshot)
    assert first["linked"] == 24 and second["duplicate_revisions"] == 24
    cycle = run_cycle(j, "run1", {"main": "weekly"}, fetch=lambda _: snapshot)
    assert (
        cycle["forecasts"]["main"]["status"] == "outside_registered_window"
        and cycle["actuals"]["linked"] == 0
    )
    assert j.rows("forecast")[0]["sha256"] == key
    clock.t = datetime(2027, 2, 1, tzinfo=timezone.utc)
    assert score_run(j, "run1", clock(), 100)["arms"]["main"]["overall"]["mae_mw"] == 0
    # Exercise the genuine CLI argument parser/read path without network or real labels.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/chronogrid_ops.py"), "--db", str(j.path), "verify"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sqlite_integrity"] == "ok"


def test_collector_rejects_duplicates_future_units_and_nonfinite_actuals(tmp_path):
    j, clock, _, _ = setup(tmp_path)
    clock.t = datetime(2027, 1, 16, 12, tzinfo=timezone.utc)
    good = dict(timestamp="2027-01-15T10:00:00+00:00", load_mw=100)
    for data in [
        dict(format="smard-hourly-mw-v1", unit="kW", rows=[good]),
        dict(format="smard-hourly-mw-v1", unit="MW", rows=[good, good]),
        dict(
            format="smard-hourly-mw-v1",
            unit="MW",
            rows=[good | {"timestamp": "2027-01-17T00:00:00+00:00"}],
        ),
        dict(format="smard-hourly-mw-v1", unit="MW", rows=[good | {"load_mw": -1}]),
    ]:
        sh = j.snapshot(encode(data), "smard-normalized-hourly", clock())
        with pytest.raises(ValueError):
            link_smard(j, "run1", sh)
    assert not j.rows("actual")


def test_cycle_isolates_model_arm_failure_and_keeps_valid_baseline(tmp_path):
    from test_operations_evidence import Clock

    from chronogrid.operations.contracts import Run
    from chronogrid.operations.journal import digest

    clock = Clock()
    clock.t = datetime(2027, 1, 14, 9, 58, tzinfo=timezone.utc)
    j = Journal(tmp_path / "cycle.sqlite", clock)
    model = digest((ROOT / "chronogrid/operations/capture.py").read_bytes())
    j.register(
        Run(
            run_id="isolated",
            experiment_kind="prospective",
            protocol_sha256="a" * 64,
            start_date="2027-01-15",
            end_date="2027-01-15",
            arms=("broken", "baseline"),
            allowed_model_sha256=(model,),
            demand_low_mw=40000,
            demand_high_mw=60000,
        )
    )
    t = datetime(2027, 1, 14, 10, tzinfo=timezone.utc)
    rows = [
        dict(timestamp=(t - timedelta(hours=i + 2)).isoformat(), load_mw=50000 + i)
        for i in reversed(range(336))
    ]
    snapshot = j.snapshot(
        encode(dict(format="smard-hourly-mw-v1", unit="MW", rows=rows)),
        "smard-normalized-hourly",
        clock(),
    )
    clock.t = t
    result = run_cycle(
        j, "isolated", {"broken": "fallback48", "baseline": "weekly"}, fetch=lambda _: snapshot
    )
    assert result["forecasts"]["broken"]["status"] == "failed"
    assert result["forecasts"]["baseline"]["status"] == "committed_local_only"
    assert len(j.rows("forecast")) == 1 and j.rows("forecast")[0]["payload"]["arm"] == "baseline"
    assert len(j.rows("worker_error")) == 1
    j.verify()
