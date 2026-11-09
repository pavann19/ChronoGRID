from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd
from src.features.engineer import FeatureEngineer
from src.operations.journal import Journal,digest,encode
from src.operations.contracts import Run
from src.operations.capture import capture_weekly,delivery_grid
from src.operations.models import CheckedOnnx,static_intervals
from src.uncertainty.schemas import CalibrationTable

ROOT=Path(__file__).resolve().parents[1]

def test_weekly_capture_actual_inputs_and_no_fake_intervals(tmp_path):
    t=datetime(2027,1,14,10,tzinfo=timezone.utc); state=[t-timedelta(minutes=2)]
    store=Journal(tmp_path/'j.sqlite',lambda:state[0]); h=digest((ROOT/'src/operations/capture.py').read_bytes())
    store.register(Run(run_id='weekly',experiment_kind='prospective',protocol_sha256='a'*64,start_date='2027-01-15',end_date='2027-01-15',arms=('baseline',),allowed_model_sha256=(h,),demand_low_mw=90,demand_high_mw=110))
    rows=[dict(timestamp=(t-timedelta(hours=i+2)).isoformat(),load_mw=100+i) for i in reversed(range(336))]
    data=dict(format='smard-hourly-mw-v1',unit='MW',rows=rows)
    snap=store.snapshot(encode(data),'smard-normalized-hourly',state[0]);state[0]=t
    p,key=capture_weekly(store,'weekly',date(2027,1,15),'baseline',snap)
    assert p.model_sha256==h and len(p.deliveries)==24
    assert all(d.interval_status=='unavailable' and d.lower_mw is None for d in p.deliveries)
    expected={x['timestamp']:x['load_mw'] for x in rows}
    assert [d.point_mw for d in p.deliveries]==[expected[(x-timedelta(hours=168)).isoformat()] for x in delivery_grid(date(2027,1,15))]
    assert len(store.rows('forecast'))==1 and len(key)==64
    future=data|{'rows':rows+[dict(timestamp=(t+timedelta(hours=1)).isoformat(),load_mw=999)]}
    state[0]=t-timedelta(seconds=1);bad=store.snapshot(encode(future),'smard-normalized-hourly',state[0]);state[0]=t
    with pytest.raises(ValueError,match='future'): capture_weekly(store,'weekly',date(2027,1,15),'baseline',bad)

def test_48hour_features_invariant_to_post_gate_observations():
    raw=pd.read_parquet(ROOT/'data/raw/genuine_smard_dwd_observations.parquet');raw['timestamp']=pd.to_datetime(raw.timestamp,utc=True)
    start=pd.Timestamp('2024-06-17T00:00:00',tz='Europe/Berlin');gate=start-pd.DateOffset(days=1)+pd.Timedelta(hours=12)
    fe=FeatureEngineer(min_lag_hours=48,lags=[48,72,168],weather_mode='persistence');base=fe.fit_transform(raw,drop_warmup=True)
    altered=raw.copy(); mask=altered.timestamp>=gate.tz_convert('UTC')
    for col in ['grid_load_mw','temperature_c','wind_speed_ms','solar_radiation_wm2','humidity_pct']: altered.loc[mask,col]=999999
    changed=fe.transform(altered,drop_warmup=True)
    delivery=base.timestamp.between(start.tz_convert('UTC'),(start+pd.Timedelta(hours=23)).tz_convert('UTC'))
    assert delivery.sum()==24
    np.testing.assert_allclose(base.loc[delivery,fe.get_feature_names()],changed.loc[delivery,fe.get_feature_names()])

def test_strict_onnx_inputs_and_calibration_compatibility():
    model=ROOT/'artifacts/models/champion_model.onnx';meta=ROOT/'artifacts/models/champion_metadata.json'
    engine=CheckedOnnx(model,meta,digest(model.read_bytes()),digest(meta.read_bytes()))
    with pytest.raises(ValueError,match='schema'): engine.predict([{'invented':0}])
    with pytest.raises(ValueError,match='non-finite'): engine.predict([{name:float('nan') for name in engine.engine.feature_names}])
    table=CalibrationTable.model_validate_json((ROOT/'artifacts/calibration/primary_calibration_table.json').read_bytes())
    with pytest.raises(ValueError,match='identity'): static_intervals(np.ones(24),np.arange(24),table,engine.sha256,'pooled_marginal_mad',.9,'policy','policy')
    with pytest.raises(ValueError,match='policy'): static_intervals(np.ones(24),np.arange(24),table,table.model_sha256,'pooled_marginal_mad',.9,'new','old')
    result=static_intervals(np.ones(25),np.arange(25)%24,table,table.model_sha256,'pooled_marginal_mad',.9,'same','same')
    assert len(result['lower'])==25 and not result['guarantee']

def test_fallback48_capture_uses_model_bound_calibration(tmp_path):
    from src.operations.capture import capture_fallback48
    t=datetime(2027,1,14,10,tzinfo=timezone.utc); state=[t-timedelta(minutes=2)]
    store=Journal(tmp_path/'j.sqlite',lambda:state[0]); experiment=ROOT/'artifacts/experiments/issuance_v2'
    modelsha=digest((experiment/'fallback.joblib').read_bytes())
    store.register(Run(run_id='model48',experiment_kind='prospective',protocol_sha256='a'*64,start_date='2027-01-15',end_date='2027-01-15',arms=('fallback48',),allowed_model_sha256=(modelsha,),demand_low_mw=40000,demand_high_mw=60000))
    rows=[dict(timestamp=(t-timedelta(hours=i+2)).isoformat(),load_mw=50000+i) for i in reversed(range(336))]
    snap=store.snapshot(encode(dict(format='smard-hourly-mw-v1',unit='MW',rows=rows)),'smard-normalized-hourly',state[0]);state[0]=t
    p,key=capture_fallback48(store,'model48',date(2027,1,15),'fallback48',snap,experiment)
    assert p.model_sha256==modelsha and p.calibration_state_sha256 is not None
    from src.operations.feature_contracts import MANIFEST,load_binding
    binding_input=next(x for x in p.inputs if x.source=='feature-contract-binding-v1')
    assert store.snapshot_bytes(binding_input.snapshot_sha256)==(ROOT/MANIFEST).read_bytes()
    metadata=json.loads((experiment/'fallback_metadata.json').read_bytes())
    assert p.feature_code_sha256==load_binding(experiment,metadata)[2]
    assert all(d.interval_status=='bounded' and d.lower_mw<=d.point_mw<=d.upper_mw for d in p.deliveries)
    assert store.verify()['objects']>=5

def test_future_holiday_calendar_fails_closed_without_changing_model_features():
    from src.operations.calendar_support import assert_frozen_calendar_supported,nationwide_holidays
    from src.features.temporal import GERMAN_HOLIDAYS
    for year in (2023,2024,2025):
        assert {(d.year,d.month,d.day) for d in nationwide_holidays(year)}=={d for d in GERMAN_HOLIDAYS if d[0]==year}
    assert_frozen_calendar_supported(delivery_grid(date(2026,11,5)))
    with pytest.raises(ValueError,match='calendar'):assert_frozen_calendar_supported(delivery_grid(date(2026,12,25)))
