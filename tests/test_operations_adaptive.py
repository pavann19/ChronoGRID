from concurrent.futures import ThreadPoolExecutor
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import json
import pytest
from chronogrid.operations.adaptive import AdaptiveConfig,AdaptiveStreams,StaleAdaptiveState,capture_adaptive48
from chronogrid.operations.journal import Journal,encode,digest
from chronogrid.operations.contracts import Run,Forecast
from chronogrid.operations.capture import delivery_grid
from chronogrid.evaluation.prospective import ActualObservation
from chronogrid.uncertainty.schemas import CalibrationTable
from test_operations_evidence import setup,forecast,Clock,D

ROOT=Path(__file__).resolve().parents[1]

def configured(tmp_path,arms=('marginal','simultaneous'),end='2027-01-15',heads=None,**policy):
    j,clock,run,snapshot=setup(tmp_path,kind='synthetic_fault',end=end,arms=arms)
    table=CalibrationTable.model_validate_json((ROOT/'artifacts/experiments/issuance_v2/fallback_calibration.json').read_bytes()).model_copy(update={'model_sha256':D,'model_id':'fixture'})
    base=j.snapshot(table.model_dump_json().encode(),'aci-base-fixture',clock())
    streams=AdaptiveStreams(j)
    for arm in arms:
        kind='simultaneous_daily' if arm=='simultaneous' else 'marginal_hourly'
        streams.register(AdaptiveConfig(run_id=run.run_id,arm=arm,head=(heads or {}).get(arm,'fallback'),model_sha256=D,base_calibration_sha256=base,feature_policy='test-fixture',target_kind=kind,**policy))
    return j,clock,run,snapshot,streams

def issue(j,clock,snapshot,streams,arm='marginal',day=date(2027,1,15),head='fallback'):
    p=forecast(j,clock,snapshot,day=day,arm=arm,kind='synthetic_fault',bounded=True)
    cfg,state,h,version=streams.prepare(p.run_id,arm)
    p=Forecast.model_validate(p.model_dump() | dict(route=head,calibration_method=cfg.method,calibration_state_sha256=h,calibration_state_version=version,effective_alpha=state['alpha']))
    return p,j.commit(p)

def actuals(j,clock,p,h,misses=1,revision='v1',all_at=None,flags=()):
    for i,d in enumerate(p.deliveries):
        clock.t=all_at or d.timestamp+timedelta(hours=2)
        value=120 if i<misses else 100
        source=j.snapshot(encode(dict(timestamp=d.timestamp.isoformat(),value=value,revision=revision)),'aci-actual-fixture',clock())
        j.actual(p.run_id,p.delivery_date,p.arm,ActualObservation(prediction_sha256=h,delivery_timestamp=d.timestamp,observation_end=d.timestamp+timedelta(hours=1),available_at=clock(),source='aci-actual-fixture',source_snapshot_sha256=source,revision=revision,value_mw=value,status='observed',quality_flags=flags))

def test_distinct_marginal_and_simultaneous_signals_timestamp_guard_and_duplicates(tmp_path):
    j,clock,_,snapshot,streams=configured(tmp_path)
    records={a:issue(j,clock,snapshot,streams,a) for a in ('marginal','simultaneous')}
    for p,h in records.values():actuals(j,clock,p,h)
    end=delivery_grid(date(2027,1,15))[-1]+timedelta(hours=1)
    clock.t=end+timedelta(hours=48)-timedelta(seconds=1)
    assert streams.settle('run1','marginal')['status']=='not_eligible'
    assert not j.rows('aci_step')
    clock.t=end+timedelta(hours=48)
    marginal=streams.settle('run1','marginal');simultaneous=streams.settle('run1','simultaneous')
    assert marginal['error_signal']==pytest.approx(1/24) and marginal['alpha_after']==pytest.approx(.1+.01*(.1-1/24))
    assert simultaneous['error_signal']==1 and simultaneous['alpha_after']==pytest.approx(.091)
    count=j.verify()['objects'];duplicate=streams.settle('run1','marginal',date(2027,1,15))
    assert duplicate['duplicate'] and duplicate['covered_hours']==23 and duplicate['alpha_after']==marginal['alpha_after']
    assert j.verify()['objects']==count
    # A later revised vector cannot rewrite the original transition.
    p,h=records['marginal'];actuals(j,clock,p,h,misses=0,revision='v2',all_at=end+timedelta(hours=50))
    assert streams.settle('run1','marginal',p.delivery_date)['missed_hours']==1
    j.verify()

def test_pending_labels_deadline_skip_and_no_forecast_gap(tmp_path):
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal',),end='2027-01-16')
    p,h=issue(j,clock,snapshot,streams)
    end=delivery_grid(p.delivery_date)[-1]+timedelta(hours=1)
    clock.t=end+timedelta(hours=48)
    assert streams.settle('run1','marginal')['status']=='pending_actuals'
    with pytest.raises(ValueError,match='chronological'):streams.settle('run1','marginal',date(2027,1,16))
    clock.t=end+timedelta(hours=168)
    skipped=streams.settle('run1','marginal')
    assert skipped['reason']=='labels_missing_at_deadline' and skipped['alpha_after']==.1
    # Next day had no forecast, so it is explicitly skipped without inventing an error.
    skipped2=streams.settle('run1','marginal')
    assert skipped2['reason']=='no_forecast' and skipped2['error_signal'] is None
    assert streams.settle('run1','marginal')['status']=='done'
    actuals(j,clock,p,h,misses=24,all_at=end+timedelta(hours=170))
    assert streams.state('run1','marginal')['alpha']==.1
    assert streams.settle('run1','marginal',p.delivery_date)['status']=='skipped'
    j.verify()

def test_concurrent_feedback_restart_and_portable_recovery(tmp_path):
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal',))
    p,h=issue(j,clock,snapshot,streams);actuals(j,clock,p,h,misses=24)
    clock.t=delivery_grid(p.delivery_date)[-1]+timedelta(hours=49)
    def settle(_):return AdaptiveStreams(Journal(j.path,clock)).settle('run1','marginal',p.delivery_date)
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(settle,range(12)))
    assert len(j.rows('aci_step'))==1 and sum(bool(r.get('duplicate')) for r in rows)==11
    assert all(r['alpha_after']==pytest.approx(.091) for r in rows)
    assert AdaptiveStreams(Journal(j.path,clock)).state('run1','marginal')['alpha']==pytest.approx(.091)
    path=tmp_path/'adaptive.json';j.export(path);restored=Journal.restore(path,tmp_path/'restored.sqlite')
    assert AdaptiveStreams(restored).state('run1','marginal')==streams.state('run1','marginal')

def test_stale_issuance_basis_static_bypass_and_head_isolation(tmp_path):
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal','primary'),end='2027-01-19',heads={'primary':'primary'})
    p,h=issue(j,clock,snapshot,streams);actuals(j,clock,p,h,misses=24)
    future=forecast(j,clock,snapshot,day=date(2027,1,19),arm='marginal',kind='synthetic_fault',bounded=True)
    with pytest.raises(ValueError,match='versioned'):j.commit(future)
    cfg,state,old_hash,version=streams.prepare('run1','marginal')
    future=Forecast.model_validate(future.model_dump()|dict(calibration_state_sha256=old_hash,calibration_state_version=version,effective_alpha=state['alpha']))
    streams.settle('run1','marginal',p.delivery_date)
    with pytest.raises(StaleAdaptiveState):j.commit(future)
    assert streams.state('run1','primary')['alpha']==.1
    cfg,new,h,version=streams.prepare('run1','marginal')
    current=Forecast.model_validate(future.model_dump()|dict(calibration_state_sha256=h,calibration_state_version=version,effective_alpha=new['alpha'],issued_at=clock()))
    j.commit(current);j.verify()

def test_invalid_quality_actuals_cannot_drive_learning(tmp_path):
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal',))
    p,h=issue(j,clock,snapshot,streams);actuals(j,clock,p,h,misses=24,flags=('invalid',))
    clock.t=delivery_grid(p.delivery_date)[-1]+timedelta(hours=49)
    assert streams.settle('run1','marginal')['status']=='pending_actuals'
    assert streams.state('run1','marginal')['alpha']==.1

def test_real_frozen_model_adaptive_capture_on_isolated_synthetic_history(tmp_path):
    clock=Clock();clock.t=datetime(2027,1,14,9,58,tzinfo=timezone.utc);j=Journal(tmp_path/'model.sqlite',clock)
    experiment=ROOT/'artifacts/experiments/issuance_v2';weights=(experiment/'fallback.joblib').read_bytes();cal=(experiment/'fallback_calibration.json').read_bytes()
    sha=digest(weights);base=j.snapshot(cal,'issuance-v2-static-calibration',clock())
    run=Run(run_id='adaptive',experiment_kind='synthetic_fault',protocol_sha256=D,start_date='2027-01-15',end_date='2027-01-15',arms=('adaptive',),allowed_model_sha256=(sha,),demand_low_mw=40000,demand_high_mw=60000);j.register(run)
    streams=AdaptiveStreams(j);streams.register(AdaptiveConfig(run_id='adaptive',arm='adaptive',head='fallback',model_sha256=sha,base_calibration_sha256=base,feature_policy='fixed_gate_lag48_weather48_rolling48_v2',target_kind='marginal_hourly'))
    t=datetime(2027,1,14,10,tzinfo=timezone.utc)
    rows=[dict(timestamp=(t-timedelta(hours=i+2)).isoformat(),load_mw=50000+i) for i in reversed(range(336))]
    snapshot=j.snapshot(encode(dict(format='smard-hourly-mw-v1',unit='MW',rows=rows)),'smard-normalized-hourly',clock());clock.t=t
    p,h=capture_adaptive48(j,run.run_id,date(2027,1,15),'adaptive',snapshot,experiment)
    assert p.effective_alpha==.1 and p.calibration_state_version.startswith('aci-v1:')
    assert len(p.deliveries)==24 and j.verify()['objects']>0

def test_dst_feedback_uses_actual_23_or_25_hour_denominator(tmp_path):
    for day,hours in [(date(2027,3,28),23),(date(2027,10,31),25)]:
        folder=tmp_path/str(day);clock=Clock();j=Journal(folder/'j.sqlite',clock)
        run=Run(run_id='run1',experiment_kind='synthetic_fault',protocol_sha256=D,start_date=day,end_date=day,arms=('marginal',),allowed_model_sha256=(D,),demand_low_mw=90,demand_high_mw=110);j.register(run)
        source=j.snapshot(b'{"dst":"fixture"}','fixture',clock())
        table=CalibrationTable.model_validate_json((ROOT/'artifacts/experiments/issuance_v2/fallback_calibration.json').read_bytes()).model_copy(update={'model_sha256':D,'model_id':'fixture'})
        base=j.snapshot(table.model_dump_json().encode(),'aci-base-fixture',clock());streams=AdaptiveStreams(j)
        streams.register(AdaptiveConfig(run_id=run.run_id,arm='marginal',head='fallback',model_sha256=D,base_calibration_sha256=base,feature_policy='test-fixture',target_kind='marginal_hourly'))
        p,h=issue(j,clock,source,streams,day=day);assert len(p.deliveries)==hours
        actuals(j,clock,p,h,misses=1);clock.t=delivery_grid(day)[-1]+timedelta(hours=49)
        result=streams.settle('run1','marginal');assert result['error_signal']==pytest.approx(1/hours)
        assert result['required_hours']==hours;j.verify()

def test_application_timestamp_is_atomic_when_wall_clock_advances(tmp_path):
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal',));p,h=issue(j,clock,snapshot,streams);actuals(j,clock,p,h,misses=0)
    clock.t=delivery_grid(p.delivery_date)[-1]+timedelta(hours=49)
    def tick():
        now=clock.t;clock.t+=timedelta(microseconds=1);return now
    j.clock=tick;streams.settle('run1','marginal');j.verify()

def test_first_collected_revision_uses_event_sequence_not_revision_spelling(tmp_path):
    from chronogrid.operations.reporting import score_run
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal',));p,h=issue(j,clock,snapshot,streams)
    same_time=delivery_grid(p.delivery_date)[-1]+timedelta(hours=2)
    actuals(j,clock,p,h,misses=24,revision='z_first',all_at=same_time)
    actuals(j,clock,p,h,misses=0,revision='a_later',all_at=same_time)
    clock.t=delivery_grid(p.delivery_date)[-1]+timedelta(hours=49)
    assert streams.settle('run1','marginal')['missed_hours']==24
    clock.t=datetime(2027,2,1,tzinfo=timezone.utc)
    report=score_run(j,'run1',clock(),100)['arms']['marginal']
    assert report['overall']['mae_mw']==20 and report['overall']['covered']==0

def test_cannot_retrofit_config_after_static_forecast_and_reject_invalid_config(tmp_path):
    j,clock,run,snapshot=setup(tmp_path,kind='synthetic_fault',arms=('marginal',))
    p=forecast(j,clock,snapshot,arm='marginal',kind='synthetic_fault',bounded=True);j.commit(p)
    table=CalibrationTable.model_validate_json((ROOT/'artifacts/experiments/issuance_v2/fallback_calibration.json').read_bytes()).model_copy(update={'model_sha256':D})
    base=j.snapshot(table.model_dump_json().encode(),'aci-base-fixture',clock())
    cfg=AdaptiveConfig(run_id=run.run_id,arm='marginal',head='fallback',model_sha256=D,base_calibration_sha256=base,feature_policy='test-fixture',target_kind='marginal_hourly')
    with pytest.raises(ValueError,match='retrofit'):AdaptiveStreams(j).register(cfg)
    with pytest.raises(ValueError):AdaptiveConfig.model_validate(cfg.model_dump()|{'minimum_delay_hours':200,'label_deadline_hours':168})

@pytest.mark.parametrize('misses,expected',[(0,.11),(24,.01)])
def test_registered_clipping_is_exact_and_not_tuned(tmp_path,misses,expected):
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal',),gamma=.5,alpha_max=.11)
    p,h=issue(j,clock,snapshot,streams);actuals(j,clock,p,h,misses=misses)
    clock.t=delivery_grid(p.delivery_date)[-1]+timedelta(hours=49)
    assert streams.settle('run1','marginal')['alpha_after']==pytest.approx(expected)
    j.verify()

def test_restore_rejects_semantically_invalid_but_rehashed_feedback_atomically(tmp_path):
    from chronogrid.operations.journal import IntegrityFailure
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal',))
    p,h=issue(j,clock,snapshot,streams);actuals(j,clock,p,h,misses=1)
    clock.t=delivery_grid(p.delivery_date)[-1]+timedelta(hours=49);streams.settle('run1','marginal')
    export=tmp_path/'valid.json';j.export(export);data=json.loads(export.read_bytes())
    changed=next(o for o in data['objects'] if o['kind']=='aci_step')
    body=json.loads(changed['payload']);body['alpha_after']=.4
    changed['payload']=encode(body).decode();changed['sha256']=digest(encode(body))
    objects={o['key']:o for o in data['objects']};previous='0'*64
    for event in data['events']:
        o=objects[event['object_key']];event['previous']=previous
        event['sha256']=digest(encode(dict(seq=event['seq'],key=o['key'],kind=o['kind'],payload_sha256=o['sha256'],recorded_at=o['recorded_at'],previous=previous)))
        previous=event['sha256']
    bad=tmp_path/'invalid.json';bad.write_bytes(encode(data))
    with pytest.raises(IntegrityFailure):Journal.restore(bad,tmp_path/'rejected.sqlite')
    assert Journal(tmp_path/'rejected.sqlite').verify()['objects']==0

def test_feedback_is_exactly_once_across_real_processes(tmp_path):
    import subprocess,sys
    j,clock,_,snapshot,streams=configured(tmp_path,arms=('marginal',));p,h=issue(j,clock,snapshot,streams);actuals(j,clock,p,h,misses=24)
    clock.t=delivery_grid(p.delivery_date)[-1]+timedelta(hours=49)
    code="from pathlib import Path;from datetime import datetime,date;from chronogrid.operations.journal import Journal;from chronogrid.operations.adaptive import AdaptiveStreams;import sys,json;j=Journal(Path(sys.argv[1]),lambda:datetime.fromisoformat(sys.argv[2]));r=AdaptiveStreams(j).settle('run1','marginal',date(2027,1,15));print(json.dumps(r))"
    children=[subprocess.Popen([sys.executable,'-c',code,str(j.path),clock().isoformat()],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
    outputs=[child.communicate(timeout=30) for child in children]
    assert all(child.returncode==0 for child in children),outputs
    rows=[json.loads(out) for out,err in outputs]
    assert len(j.rows('aci_step'))==1 and sum(bool(r.get('duplicate')) for r in rows)==3
    assert all(r['alpha_after']==pytest.approx(.091) for r in rows)
    j.verify()
