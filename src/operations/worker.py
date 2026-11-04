"""Bounded once/cycle worker; no scheduler installed and no missed-origin backfill."""
from datetime import timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from src.operations.capture import capture_smard,capture_weekly,capture_fallback48
from src.operations.journal import Conflict,IntegrityFailure

ROOT=Path(__file__).resolve().parents[2]

def run_once(store,run_id,arm,model='weekly',fetch=capture_smard):
    from src.operations.adaptive import capture_adaptive48,StaleAdaptiveState
    if model not in ('weekly','fallback48','adaptive48'):raise ValueError('unrecognized worker model')
    run=store.get('run:'+run_id);now=store.clock().astimezone(ZoneInfo('Europe/Berlin'));day=now.date()+timedelta(days=1)
    if arm not in run['arms']:raise ValueError('unregistered arm')
    if run['experiment_kind']!='prospective':raise ValueError('worker requires prospective run')
    if not run['start_date']<=str(day)<=run['end_date']:return dict(status='outside_registered_window')
    key=f'forecast:{run_id}:{day}:{arm}'
    try:store.get(key);return dict(status='already_committed')
    except KeyError:pass
    gate=now.replace(hour=12,minute=0,second=0,microsecond=0)
    if now>gate:
        store.failure(run_id,day,arm,'missed issuance gate; no backfill');return dict(status='missed_gate')
    try:
        snapshot=fetch(store)
        capture={'weekly':capture_weekly,'fallback48':capture_fallback48,'adaptive48':capture_adaptive48}[model]
        for attempt in range(3):
            args=(store,run_id,day,arm,snapshot)
            try:
                _,h=capture(*args) if model=='weekly' else capture(*args,ROOT/'artifacts/experiments/issuance_v2')
                return dict(status='committed_local_only',prediction_sha256=h,adaptive_retries=attempt)
            except StaleAdaptiveState:
                if attempt==2:raise
    except StaleAdaptiveState as exc:
        store.failure(run_id,day,arm,str(exc));return dict(status='adaptive_state_contended_retry_later')
    except Conflict:return dict(status='concurrent_origin_already_committed')
    except Exception as exc:
        store.failure(run_id,day,arm,str(exc));raise

def run_cycle(store,run_id,models,fetch=capture_smard,max_feedback_origins=16):
    """Fetch once, append actuals, settle eligible feedback, then issue tomorrow's vectors."""
    from src.operations.adaptive import AdaptiveStreams
    from src.operations.observations import link_smard
    store.verify()
    run=store.get('run:'+run_id)
    if run['experiment_kind']!='prospective':raise ValueError('cycle worker requires prospective run')
    if not isinstance(models,dict) or set(models)!=set(run['arms']) or any(m not in ('weekly','fallback48','adaptive48') for m in models.values()):raise ValueError('explicit model choice required for every registered arm')
    if not 1<=max_feedback_origins<=367:raise ValueError('invalid feedback batch budget')
    snapshot=fetch(store)
    actuals=link_smard(store,run_id,snapshot);feedback={};forecasts={}
    for arm,model in models.items():
        operation='feedback' if model=='adaptive48' else 'forecast'
        try:
            if model=='adaptive48':
                rows=[]
                for _ in range(max_feedback_origins):
                    result=AdaptiveStreams(store).settle(run_id,arm);rows.append(result)
                    if result['status'] not in ('processed','skipped'):break
                feedback[arm]=rows
            operation='forecast'
            forecasts[arm]=run_once(store,run_id,arm,model,fetch=lambda _:snapshot)
        except IntegrityFailure:
            raise # corruption is a journal-wide safety failure, not an isolated arm outage
        except Exception as exc:
            with store.transaction() as db:
                seq=db.execute('SELECT count(*) FROM events').fetchone()[0]+1
                store._add(db,f'worker-error:{seq}','worker_error',dict(run_id=run_id,arm=arm,operation=operation,reason=str(exc)))
            forecasts[arm]=dict(status='failed',operation=operation,reason=str(exc))
    return dict(source_snapshot=snapshot,actuals=actuals,feedback=feedback,forecasts=forecasts,guarantee=False)
