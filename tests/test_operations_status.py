import sqlite3
from datetime import datetime, timezone

import pytest
from test_operations_evidence import forecast, setup

from chronogrid.operations.journal import IntegrityFailure, digest
from chronogrid.operations.status import ReadOnlyJournal, journal_status


def test_status_never_initializes_missing_journal(tmp_path):
    path = tmp_path / "missing" / "journal.sqlite"
    with pytest.raises(FileNotFoundError):
        journal_status(path)
    assert not path.parent.exists()


def test_status_gate_boundary_and_unchanged_evidence(tmp_path):
    store, clock, _, _ = setup(tmp_path)
    before = digest(store.path.read_bytes())
    clock.t = datetime(2027, 1, 14, 11, tzinfo=timezone.utc)
    report = journal_status(store.path, clock=clock)
    assert report["runs"][0]["slots"]["due"] == 1
    assert report["attention_required"]
    clock.t = datetime(2027, 1, 14, 11, 0, 1, tzinfo=timezone.utc)
    assert journal_status(store.path, clock=clock)["runs"][0]["slots"]["missed_gate"] == 1
    assert before == digest(store.path.read_bytes())
    assert not store.rows("failure")


def test_readonly_connection_rejects_writes_and_unknown_run(tmp_path):
    store, clock, _, _ = setup(tmp_path)
    with ReadOnlyJournal(store.path).connection() as db:
        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO events VALUES (999,'bad','bad','bad')")
    with pytest.raises(KeyError):
        journal_status(store.path, run_id="unknown", clock=clock)


def test_cli_status_uses_readonly_path(tmp_path, capsys, monkeypatch):
    import json

    from chronogrid.operations.cli import main

    store, _, _, _ = setup(tmp_path)
    before = digest(store.path.read_bytes())

    def forbidden(*args, **kwargs):
        raise AssertionError("status must bypass Journal initialization")

    monkeypatch.setattr("chronogrid.operations.cli.Journal", forbidden)
    main(["--db", str(store.path), "status", "--run", "run1"])
    assert json.loads(capsys.readouterr().out)["runs"][0]["run_id"] == "run1"
    assert digest(store.path.read_bytes()) == before


def test_status_rejects_tampered_evidence(tmp_path):
    store, clock, _, _ = setup(tmp_path)
    with store.connection() as db:
        db.execute("DROP TRIGGER immutable_objects_update")
        db.execute("UPDATE objects SET payload='{}' WHERE kind='run'")
    with pytest.raises(IntegrityFailure):
        journal_status(store.path, clock=clock)


def test_committed_origin_stays_committed_after_gate(tmp_path):
    store, clock, _, snapshot = setup(tmp_path)
    store.commit(forecast(store, clock, snapshot))
    clock.t = datetime(2027, 1, 20, tzinfo=timezone.utc)
    result = journal_status(store.path, clock=clock)
    assert result["runs"][0]["slots"]["committed"] == 1
    assert result["runs"][0]["missing_slots"] == []
    assert not result["attention_required"]


def test_synthetic_slots_do_not_become_missed_prospective_forecasts(tmp_path):
    store, clock, _, _ = setup(tmp_path, kind="synthetic_fault")
    clock.t = datetime(2027, 1, 20, tzinfo=timezone.utc)
    result = journal_status(store.path, clock=clock)
    assert result["runs"][0]["slots"]["uncommitted_replay_or_synthetic"] == 1
    assert not result["attention_required"]


@pytest.mark.parametrize(
    "day,instant",
    [("2027-03-29", "2027-03-28T10:00:00+00:00"), ("2027-11-01", "2027-10-31T11:00:00+00:00")],
)
def test_gate_uses_german_timezone_across_dst(tmp_path, day, instant):
    from chronogrid.operations.contracts import Run
    from chronogrid.operations.journal import Journal

    store = Journal(tmp_path / "journal.sqlite")
    store.register(
        Run(
            run_id="dst",
            experiment_kind="prospective",
            protocol_sha256="a" * 64,
            start_date=day,
            end_date=day,
            arms=("main",),
            allowed_model_sha256=("a" * 64,),
            demand_low_mw=90,
            demand_high_mw=110,
        )
    )
    report = journal_status(store.path, clock=lambda: datetime.fromisoformat(instant))
    assert report["runs"][0]["slots"]["due"] == 1
