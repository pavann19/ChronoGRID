"""Read-only, transaction-consistent journal status; no network or model loading."""
from contextlib import contextmanager
from datetime import datetime, time, timedelta
from pathlib import Path
import json
import sqlite3
from zoneinfo import ZoneInfo

from chronogrid.evaluation.prospective import utc
from chronogrid.operations.contracts import Run
from chronogrid.operations.journal import Journal, now_utc


class ReadOnlyJournal(Journal):
    def __init__(self, path, clock=now_utc):
        self.path = Path(path).resolve()
        self.clock = clock
        if not self.path.is_file():
            raise FileNotFoundError('Journal does not exist; status never creates one')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=15,
                             isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        try:
            yield db
        finally:
            db.close()


def journal_status(path, *, run_id=None, clock=now_utc):
    store = ReadOnlyJournal(path, clock)
    now = utc(clock())
    berlin = ZoneInfo('Europe/Berlin')
    with store.connection() as db:
        db.execute('BEGIN')
        store.journal_mode = db.execute('PRAGMA journal_mode').fetchone()[0]
        integrity = store._verify_db(db)
        rows = [dict(r) | {'payload': json.loads(r['payload'])}
                for r in db.execute('SELECT * FROM objects')]
        runs = [Run.model_validate(r['payload']) for r in rows if r['kind'] == 'run'
                and (run_id is None or r['payload']['run_id'] == run_id)]
        if run_id is not None and not runs:
            raise KeyError('Unknown registered run: ' + run_id)
        keys = {r['key'] for r in rows}
        output = []
        for run in runs:
            states = dict(committed=0, missed_gate=0, due=0, future=0)
            missing = []
            day = run.start_date
            while day <= run.end_date:
                gate = datetime.combine(day - timedelta(days=1), time(12), berlin)
                for arm in run.arms:
                    if f'forecast:{run.run_id}:{day}:{arm}' in keys:
                        states['committed'] += 1
                        continue
                    if run.experiment_kind != 'prospective':
                        state = 'uncommitted_replay_or_synthetic'
                    elif now > utc(gate):
                        state = 'missed_gate'
                    elif now.astimezone(berlin).date() == gate.date():
                        state = 'due'
                    else:
                        state = 'future'
                    states[state] = states.get(state, 0) + 1
                    missing.append(dict(delivery_date=str(day), arm=arm, status=state,
                                        issuance_gate=gate.isoformat()))
                day += timedelta(days=1)
            failures = [r for r in rows if r['kind'] in ('failure', 'worker_error')
                        and r['payload'].get('run_id') == run.run_id]
            output.append(dict(run_id=run.run_id, experiment_kind=run.experiment_kind,
                               scheduled_slots=sum(states.values()), slots=states,
                               missing_slots=missing, recorded_failures=len(failures)))
        db.execute('COMMIT')
    return dict(format='chronogrid-operations-status-v1', checked_at=now.isoformat(),
                integrity=integrity, runs=sorted(output, key=lambda r: r['run_id']),
                attention_required=any(r['slots']['missed_gate'] or r['slots']['due']
                                       for r in output),
                scope='Journal integrity and scheduling only; no model, source, security, '
                      'coverage, scheduler installation or production readiness certification',
                production_approved=False)
