from datetime import datetime,timedelta,timezone
from chronogrid.operations.worker import run_once
from test_operations_evidence import setup
import pytest
from chronogrid.operations.journal import Conflict

def test_worker_never_backfills_missed_gate(tmp_path):
    store,clock,_,_=setup(tmp_path)
    clock.t=datetime(2027,1,14,13,tzinfo=timezone.utc)
    def never(_): raise AssertionError('must not fetch after missed gate')
    result=run_once(store,'run1','main',fetch=never)
    assert result['status']=='missed_gate' and not store.rows('forecast')
    assert len(store.rows('failure'))==1
    clock.t=datetime(2027,1,16,10,tzinfo=timezone.utc)
    assert run_once(store,'run1','main',fetch=never)['status']=='outside_registered_window'


def test_non_forecast_conflict_is_not_reported_as_concurrent_commit(tmp_path, monkeypatch):
    store, clock, _, snapshot = setup(tmp_path)
    clock.t = datetime(2027, 1, 14, 10, tzinfo=timezone.utc)
    def conflict(*args):
        raise Conflict('snapshot identity conflict')
    monkeypatch.setattr('chronogrid.operations.worker.capture_weekly', conflict)
    with pytest.raises(Conflict, match='snapshot identity'):
        run_once(store, 'run1', 'main', fetch=lambda _: snapshot)
    assert not store.rows('forecast')
