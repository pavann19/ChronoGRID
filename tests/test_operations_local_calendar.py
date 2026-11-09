from datetime import date
import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path
from datetime import datetime,timedelta,timezone
from src.operations.capture import delivery_grid
from src.operations.local_calendar import holidays,add_local_calendar,LocalCalendarFeatures

def test_2026_nationwide_holidays_and_local_midnight():
    assert holidays(2026)==frozenset(date.fromisoformat(d) for d in ['2026-01-01','2026-04-03','2026-04-06','2026-05-01','2026-05-14','2026-05-25','2026-10-03','2026-12-25','2026-12-26'])
    frame=add_local_calendar(pd.DataFrame({'timestamp':pd.to_datetime(['2026-12-24T22:00Z','2026-12-24T23:00Z','2026-12-25T23:00Z'],utc=True)}))
    assert frame.is_holiday.tolist()==[0,1,1]
    assert frame.sin_hour.iloc[1]==pytest.approx(0)
    assert frame.is_working_day.tolist()==[1,0,0]
    with pytest.raises(ValueError,match='policy years'):holidays(2031)

@pytest.mark.parametrize('day,length',[('2026-03-29',23),('2026-10-25',25)])
def test_dst_local_clock_keeps_unique_utc_grid_and_elapsed_lags(day,length):
    grid=delivery_grid(date.fromisoformat(day));assert len(grid)==length
    ts=pd.date_range(grid[0]-pd.Timedelta(hours=216),grid[-1],freq='h')
    frame=pd.DataFrame({'timestamp':ts,'grid_load_mw':np.arange(len(ts),dtype=float)})
    out=LocalCalendarFeatures()._generate_raw_features(frame).set_index('timestamp').loc[pd.DatetimeIndex(grid)]
    assert out.index.is_unique
    assert np.array_equal(out.grid_load_mw_lag_48.values,frame.set_index('timestamp').loc[pd.DatetimeIndex(grid)-pd.Timedelta(hours=48),'grid_load_mw'].values)
    local=pd.DatetimeIndex(grid).tz_convert('Europe/Berlin')
    assert np.allclose(out.sin_hour,np.sin(2*np.pi*local.hour/24))
    if length==25:assert sum(local.hour==2)==2
    else:assert sum(local.hour==2)==0
    changed=frame.copy();changed.loc[changed.timestamp>=grid[0],'grid_load_mw']=999999
    altered=LocalCalendarFeatures()._generate_raw_features(changed).set_index('timestamp').loc[pd.DatetimeIndex(grid)]
    cols=[c for c in out if c!='grid_load_mw']
    pd.testing.assert_frame_equal(out[cols],altered[cols])

@pytest.mark.parametrize('value',['2026-12-25',None,'not-a-time'])
def test_reject_naive_missing_invalid_timestamps(value):
    with pytest.raises((ValueError,TypeError)):add_local_calendar(pd.DataFrame({'timestamp':[value]}))

def test_reject_gaps_before_row_shift():
    with pytest.raises(ValueError,match='consecutive UTC'):
        LocalCalendarFeatures()._generate_raw_features(pd.DataFrame({'timestamp':pd.to_datetime(['2026-01-01T00:00Z','2026-01-01T02:00Z'],utc=True),'grid_load_mw':[1,2]}))

@pytest.mark.parametrize('day',['2026-12-25','2027-03-28','2027-10-31'])
def test_real_v3_model_capture_on_isolated_synthetic_history(tmp_path,day):
    from src.operations.capture import capture_fallback48
    from src.operations.journal import Journal,digest,encode
    from src.operations.contracts import Run
    root=Path(__file__).resolve().parents[1]
    experiment=root/'artifacts/experiments/local_calendar_v3'
    d=date.fromisoformat(day)
    issued=datetime.combine(d-timedelta(days=1),datetime.min.time(),tzinfo=timezone.utc)+timedelta(hours=9)
    state=[issued-timedelta(minutes=2)]
    store=Journal(tmp_path/'journal.sqlite',lambda:state[0])
    modelsha=digest((experiment/'fallback.joblib').read_bytes())
    store.register(Run(run_id='calendar',experiment_kind='synthetic_fault',protocol_sha256='a'*64,start_date=d,end_date=d,arms=('local',),allowed_model_sha256=(modelsha,),demand_low_mw=40000,demand_high_mw=60000))
    rows=[dict(timestamp=(issued-timedelta(hours=i+2)).isoformat(),load_mw=50000+i) for i in reversed(range(336))]
    snap=store.snapshot(encode(dict(format='smard-hourly-mw-v1',unit='MW',rows=rows)),'smard-normalized-hourly',state[0])
    state[0]=issued
    p,h=capture_fallback48(store,'calendar',d,'local',snap,experiment)
    assert p.feature_policy=='fixed_gate_load48_local_calendar_v3'
    assert p.experiment_kind=='synthetic_fault'
    assert tuple(x.timestamp for x in p.deliveries)==delivery_grid(d)
    assert all(x.interval_status=='bounded' and x.lower_mw<=x.point_mw<=x.upper_mw for x in p.deliveries)
    assert store.verify()['sqlite_integrity']=='ok'
    # Source-code/metadata drift must fail before unpickling or issuing a second arm.
    import shutil
    copied=tmp_path/'bad';shutil.copytree(experiment,copied)
    metadata=json.loads((copied/'fallback_metadata.json').read_bytes())
    metadata['feature_sources']['src/operations/local_calendar.py']='0'*64
    (copied/'fallback_metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='binding mismatch'):
        capture_fallback48(store,'calendar',d,'local',snap,copied)
