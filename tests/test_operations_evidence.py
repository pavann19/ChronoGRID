from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone,date
from pathlib import Path
import json, sqlite3, subprocess,sys
import pytest
from src.operations.contracts import Run,Forecast
from src.operations.journal import Journal,Conflict,IntegrityFailure,digest,encode
from src.evaluation.prospective import ActualObservation
from src.operations.capture import delivery_grid,parse_smard
from src.operations.reporting import score_run
D='a'*64
class Clock:
    def __init__(self): self.t=datetime(2027,1,1,tzinfo=timezone.utc)
    def __call__(self): return self.t

def setup(tmp_path,kind='prospective',end='2027-01-15',arms=('main',)):
    clock=Clock(); store=Journal(tmp_path/'journal.sqlite',clock)
    run=Run(run_id='run1',experiment_kind=kind,protocol_sha256=D,start_date='2027-01-15',end_date=end,arms=arms,allowed_model_sha256=(D,),demand_low_mw=90,demand_high_mw=110)
    store.register(run)
    snapshot=store.snapshot(b'{"test":"synthetic fixture"}','fixture',clock())
    return store,clock,run,snapshot

def forecast(store,clock,snapshot,day=date(2027,1,15),arm='main',kind='prospective',bounded=False):
    source=store.get('snapshot:'+snapshot); start=delivery_grid(day)[0]
    clock.t=start-timedelta(hours=13)
    data=dict(schema_version='2.0',experiment_kind=kind,run_id='run1',arm=arm,feature_policy='test-fixture',record_id='record_'+str(day)+'_'+arm,issued_at=clock(),issuance_timezone='UTC',delivery_date=day,model_id='fixture',model_sha256=D,feature_code_sha256=D,protocol_sha256=D,route='fallback',routing_reason='fixture',data_quality_flags=(),outage=False,inputs=(dict(source='fixture',snapshot_sha256=snapshot,event_time_start='2026-12-31T00:00:00Z',event_time_end='2026-12-31T01:00:00Z',available_at=source['available_at'],captured_at=source['captured_at'],quality_flags=(),outage=False),),calibration_method='unavailable',calibration_state_version='none',calibration_state_sha256=None,nominal_coverage=.9,effective_alpha=None,deliveries=tuple(dict(timestamp=t,point_mw=100,lower_mw=None,upper_mw=None,interval_status='unavailable',quality_flags=()) for t in delivery_grid(day)))
    if bounded:
        cal=store.snapshot(encode({'model_sha256':D}),'test-calibration',clock())
        data.update(calibration_method='pooled_marginal_mad',calibration_state_sha256=cal,effective_alpha=.1)
        data['deliveries']=tuple(dict(x,lower_mw=90,upper_mw=110,interval_status='bounded') for x in data['deliveries'])
    return Forecast.model_validate(data)

def link(store,clock,p,key,y=100,revision='v1'):
    for t in [d.timestamp for d in p.deliveries]:
        clock.t=t+timedelta(hours=2)
        raw=encode(dict(timestamp=t.isoformat(),value=y,revision=revision))
        sh=store.snapshot(raw,'actual-fixture',clock())
        store.actual(p.run_id,p.delivery_date,p.arm,ActualObservation(prediction_sha256=key,delivery_timestamp=t,observation_end=t+timedelta(hours=1),available_at=clock(),source='actual-fixture',source_snapshot_sha256=sh,revision=revision,value_mw=y,status='observed',quality_flags=()))

def test_transaction_race_restart_backup_export(tmp_path):
    store,clock,run,snapshot=setup(tmp_path); p=forecast(store,clock,snapshot)
    def write(_):
        try: return Journal(store.path,clock).commit(p)
        except Conflict: return 'duplicate'
    with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(write,range(16)))
    assert results.count('duplicate')==15
    before=store.verify(); assert Journal(store.path).verify()==before
    store.backup(tmp_path/'backup.sqlite'); assert Journal(tmp_path/'backup.sqlite').verify()==before
    store.export(tmp_path/'export.json'); restored=Journal.restore(tmp_path/'export.json',tmp_path/'restore.sqlite')
    assert restored.verify()==before
    with pytest.raises(Conflict): Journal.restore(tmp_path/'export.json',tmp_path/'restore.sqlite')
    with store.connection() as db:
        with pytest.raises(sqlite3.IntegrityError): db.execute("DELETE FROM objects")
        with pytest.raises(sqlite3.IntegrityError): db.execute("UPDATE objects SET payload='{}'")

def test_killed_writer_rolls_back(tmp_path):
    store,clock,_,_=setup(tmp_path); before=store.verify()
    code="import sqlite3,os,sys; c=sqlite3.connect(sys.argv[1]); c.execute('BEGIN IMMEDIATE'); c.execute(\"INSERT INTO objects VALUES ('dead','failure','{}','x','x')\"); os._exit(23)"
    child=subprocess.run([sys.executable,'-c',code,str(store.path)])
    assert child.returncode==23 and store.verify()==before

def test_export_tampering_detected(tmp_path):
    store,*_=setup(tmp_path); path=tmp_path/'export.json'; store.export(path)
    data=json.loads(path.read_text()); data['objects'][0]['payload']='{}'; path.write_text(json.dumps(data))
    with pytest.raises(IntegrityFailure): Journal.restore(path,tmp_path/'bad.sqlite')
    assert Journal(tmp_path/'bad.sqlite').verify()['objects']==0

def test_source_identity_protocol_and_late_commit(tmp_path):
    store,clock,_,snapshot=setup(tmp_path); p=forecast(store,clock,snapshot)
    for update in ({'protocol_sha256':'b'*64},{'model_sha256':'b'*64},{'experiment_kind':'synthetic_fault'}):
        with pytest.raises(ValueError): store.commit(p.model_copy(update=update))
    changed=p.inputs[0].model_copy(update={'source':'pretend'})
    with pytest.raises(ValueError): store.commit(p.model_copy(update={'inputs':(changed,)}))
    clock.t=clock.t.replace(hour=12)
    with pytest.raises(ValueError,match='gate'): store.commit(p)
    assert not store.rows('forecast')

def test_actual_linkage_and_vintages(tmp_path):
    store,clock,_,snapshot=setup(tmp_path); p=forecast(store,clock,snapshot,bounded=True); h=store.commit(p)
    link(store,clock,p,h,y=120)
    clock.t=datetime(2027,1,30,tzinfo=timezone.utc)
    report=score_run(store,'run1',clock(),replicates=100)['arms']['main']
    assert report['overall']['mae_mw']==20 and report['overall']['rmse_mw']==20
    assert report['overall']['covered']==0 and report['overall']['missed']==24
    assert report['overall']['mean_interval_width_mw']==20
    assert report['overall']['mean_winkler_score']==pytest.approx(220)
    assert report['minimum_complete_daily_coverage']==0
    assert report['bootstrap'][0]['status']=='insufficient_complete_origins'
    # Revisions append, cannot rewrite first collected actual.
    link(store,clock,p,h,y=100,revision='v2')
    clock.t=datetime(2027,1,30,tzinfo=timezone.utc)
    assert score_run(store,'run1',clock(),100)['arms']['main']['overall']['mae_mw']==20
    with pytest.raises(Conflict): link(store,clock,p,h,y=100,revision='v2')

def test_missing_and_uncalibrated_never_covered(tmp_path):
    store,clock,_,snapshot=setup(tmp_path,end='2027-01-16'); p=forecast(store,clock,snapshot); h=store.commit(p); link(store,clock,p,h)
    clock.t=datetime(2027,2,1,tzinfo=timezone.utc)
    report=score_run(store,'run1',clock(),100)['arms']['main']
    assert report['scheduled_hours']==48 and report['missing_forecast_hours']==24
    assert report['unavailable_interval_hours']==24 and report['overall']['interval_observations']==0
    assert report['overall']['marginal_coverage'] is None and report['minimum_complete_daily_coverage'] is None
    assert report['overall']['point_observations']==24

def test_smard_units_duplicates_and_missing():
    rows,missing=parse_smard(b'{"series":[[1704067200000,100],[1704070800000,null]]}')
    assert list(rows.values())==[100] and missing==1
    with pytest.raises(ValueError): parse_smard(b'{"series":[[1704067200000,100],[1704067200000,101]]}')
    with pytest.raises(ValueError): parse_smard(b'{"series":[[1704067200001,100]]}')

def test_calendar_blocks_with_missing_origins(tmp_path):
    store,clock,_,snapshot=setup(tmp_path,kind='synthetic_fault',end='2027-02-18')
    for i in range(35):
        if i==10: continue
        p=forecast(store,clock,snapshot,day=date(2027,1,15)+timedelta(days=i),kind='synthetic_fault',bounded=True)
        h=store.commit(p); link(store,clock,p,h,y=100 if i%2 else 120)
    clock.t=datetime(2027,3,1,tzinfo=timezone.utc)
    report=score_run(store,'run1',clock(),100)['arms']['main']
    assert len(report['per_origin'])==35 and report['missing_forecast_hours']==24
    assert len(report['bootstrap'])==36 and report['complete_origins']==34
    assert all(x['status']=='empirical' for x in report['bootstrap'])
    assert report['overall']['marginal_coverage']==pytest.approx(17/34)

def test_actual_cross_prediction_rejection(tmp_path):
    store,clock,_,snapshot=setup(tmp_path);p=forecast(store,clock,snapshot);h=store.commit(p)
    t=p.deliveries[0].timestamp;clock.t=t+timedelta(hours=2)
    sh=store.snapshot(b'{"value":1}','actual-fixture',clock())
    actual=ActualObservation(prediction_sha256='b'*64,delivery_timestamp=t,observation_end=t+timedelta(hours=1),available_at=clock(),source='actual-fixture',source_snapshot_sha256=sh,revision='r1',value_mw=1,status='observed',quality_flags=())
    with pytest.raises(ValueError,match='hash'):store.actual('run1',p.delivery_date,'main',actual)

def test_actual_processes_compete_for_same_registered_slot(tmp_path):
    store,clock,_,snapshot=setup(tmp_path);p=forecast(store,clock,snapshot);payload=tmp_path/'forecast.json';payload.write_text(p.model_dump_json())
    code="from pathlib import Path;from datetime import datetime;from src.operations.journal import Journal,Conflict;from src.operations.contracts import Forecast;import sys;p=Forecast.model_validate_json(Path(sys.argv[2]).read_bytes());j=Journal(Path(sys.argv[1]),lambda:datetime.fromisoformat(sys.argv[3]));\ntry:j.commit(p);print('committed')\nexcept Conflict:print('duplicate')"
    procs=[subprocess.Popen([sys.executable,'-c',code,str(store.path),str(payload),clock().isoformat()],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
    outputs=[p.communicate(timeout=30) for p in procs]
    assert all(p.returncode==0 for p in procs),outputs
    assert sum('committed' in out for out,err in outputs)==1
    assert sum('duplicate' in out for out,err in outputs)==3
    assert len(store.rows('forecast'))==1
