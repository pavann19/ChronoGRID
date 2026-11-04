"""Run once under an OS scheduler, or bounded local polling; never fabricate missed origins."""
from datetime import timedelta,datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from src.operations.capture import capture_smard,capture_weekly,capture_fallback48
from src.operations.journal import Conflict

ROOT=Path(__file__).resolve().parents[2]
def run_once(store,run_id,arm,model='weekly',fetch=capture_smard):
    run=store.get('run:'+run_id);now=store.clock().astimezone(ZoneInfo('Europe/Berlin')); day=now.date()+timedelta(days=1)
    if arm not in run['arms']: raise ValueError('unregistered arm')
    if run['experiment_kind']!='prospective': raise ValueError('worker requires prospective run')
    if not run['start_date']<=str(day)<=run['end_date']: return dict(status='outside_registered_window')
    key=f'forecast:{run_id}:{day}:{arm}'
    try: store.get(key);return dict(status='already_committed')
    except KeyError: pass
    gate=now.replace(hour=12,minute=0,second=0,microsecond=0)
    if now>gate:
        store.failure(run_id,day,arm,'missed issuance gate; no backfill');return dict(status='missed_gate')
    try:
        snapshot=fetch(store)
        capture=capture_weekly if model=='weekly' else capture_fallback48
        args=(store,run_id,day,arm,snapshot)
        _,h=capture(*args) if model=='weekly' else capture(*args,ROOT/'artifacts/experiments/issuance_v2')
        return dict(status='committed_local_only',prediction_sha256=h)
    except Conflict: return dict(status='concurrent_origin_already_committed')
    except Exception as exc:
        store.failure(run_id,day,arm,str(exc));raise
