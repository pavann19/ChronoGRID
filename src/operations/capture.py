"""Source-preserving SMARD capture and issuance-safe seasonal baseline."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import math
import requests
from src.evaluation.prospective import Delivery, InputSnapshot, utc
from src.operations.contracts import Forecast
from src.operations.journal import Journal, digest, encode

SMARD='https://www.smard.de/app/chart_data/410/DE'
MAX_BYTES=8_000_000

def fetch_json(url):
    # URL is built only from fixed SMARD base plus validated integer slice IDs.
    with requests.get(url,timeout=(5,20),stream=True) as response:
        response.raise_for_status(); content=bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content)>MAX_BYTES: raise ValueError('source response exceeds capture limit')
        return bytes(content)

def parse_smard(content: bytes):
    """Hourly series are MWh per one-hour interval, numerically equal to average MW."""
    payload=json.loads(content); rows={}; missing=0
    for timestamp,value in payload['series']:
        ts=datetime.fromtimestamp(timestamp/1000,timezone.utc)
        if ts.minute or ts.second or ts.microsecond: raise ValueError('non-hourly SMARD timestamp')
        if value is None:
            missing+=1; continue
        if isinstance(value,bool) or not math.isfinite(float(value)) or float(value)<0:
            raise ValueError('invalid consumption')
        key=ts.isoformat(); val=float(value)
        if key in rows and rows[key]!=val: raise ValueError('conflicting duplicate consumption')
        rows[key]=val
    return rows,missing

def capture_smard(store: Journal, weeks=3):
    if not 1<=weeks<=8: raise ValueError('weeks outside bounded capture window')
    index_raw=fetch_json(SMARD+'/index_hour.json')
    observed=utc(store.clock())
    index_hash=store.snapshot(index_raw,'smard-index',observed)
    stamps=json.loads(index_raw)['timestamps']
    selected=sorted({int(x) for x in stamps if int(x)<=int(observed.timestamp()*1000)})[-weeks:]
    if len(selected)<weeks: raise ValueError('insufficient SMARD index history')
    merged={}; hashes=[]; missing=0
    for stamp in selected:
        raw=fetch_json(f'{SMARD}/410_DE_hour_{stamp}.json'); fetched=utc(store.clock())
        hashes.append(store.snapshot(raw,'smard-hourly-response',fetched))
        rows,nulls=parse_smard(raw); missing+=nulls
        for ts,val in rows.items():
            if utc(datetime.fromisoformat(ts))+timedelta(hours=1)>fetched: continue
            if ts in merged and merged[ts]!=val: raise ValueError('conflicting source slices; preserve and investigate revision')
            merged[ts]=val
    if not merged: raise ValueError('no completed load observations')
    # A normalized snapshot never conceals upstream raw hashes or missingness.
    content=encode(dict(format='smard-hourly-mw-v1',unit='MW',source_resolution='hour',raw_sha256=hashes,index_sha256=index_hash,source_first_seen_only=True,missing_source_values=missing,rows=[dict(timestamp=k,load_mw=v) for k,v in sorted(merged.items())]))
    return store.snapshot(content,'smard-normalized-hourly',utc(store.clock()))

def delivery_grid(day: date):
    zone=ZoneInfo('Europe/Berlin')
    start=datetime.combine(day,datetime.min.time(),zone).astimezone(timezone.utc)
    end=datetime.combine(day+timedelta(days=1),datetime.min.time(),zone).astimezone(timezone.utc)
    return tuple(start+timedelta(hours=i) for i in range(int((end-start).total_seconds()/3600)))

def capture_weekly(store: Journal,run_id: str,day: date,arm: str,snapshot_sha256: str):
    """Explicit uncalibrated baseline. Never reuses research calibration for this policy."""
    run=store.get('run:'+run_id)
    source=store.get('snapshot:'+snapshot_sha256)
    dataset=json.loads(store.snapshot_bytes(snapshot_sha256))
    if dataset.get('format')!='smard-hourly-mw-v1' or dataset.get('unit')!='MW': raise ValueError('wrong source/unit contract')
    if source['source']!='smard-normalized-hourly': raise ValueError('unapproved source')
    rows={}; issued=utc(store.clock())
    for row in dataset['rows']:
        ts=utc(datetime.fromisoformat(row['timestamp'])); value=float(row['load_mw'])
        if ts in rows or ts.minute or ts.second or ts.microsecond or not math.isfinite(value) or value<0: raise ValueError('invalid hourly history')
        if ts+timedelta(hours=1)>utc(datetime.fromisoformat(source['available_at'])): raise ValueError('source contains unfinished/future actuals')
        rows[ts]=value
    if not rows: raise ValueError('empty source')
    # Do not present an old 2024 dataset as today's usable source.
    if issued-(max(rows)+timedelta(hours=1))>timedelta(hours=48): raise ValueError('load source is stale (>48 hours)')
    points=[]
    for ts in delivery_grid(day):
        lag=ts-timedelta(hours=168)
        if lag not in rows or lag+timedelta(hours=1)>issued: raise ValueError('weekly lag unavailable at issuance')
        points.append(Delivery(timestamp=ts,point_mw=rows[lag],lower_mw=None,upper_mw=None,interval_status='unavailable',quality_flags=('uncalibrated_baseline',)))
    code_sha=digest(Path(__file__).read_bytes())
    p=Forecast(run_id=run_id,arm=arm,experiment_kind=run['experiment_kind'],record_id=f'{run_id}_{day}_{arm}',protocol_sha256=run['protocol_sha256'],issued_at=issued,issuance_timezone='UTC',delivery_date=day,model_id='weekly-persistence-168h-v2',model_sha256=code_sha,feature_code_sha256=code_sha,feature_policy='exact_utc_weekly_lag_168h_no_imputation',route='fallback',routing_reason='predeclared load-only baseline; primary issuance-safe feature policy not validated',data_quality_flags=('uncalibrated_baseline',),outage=False,inputs=(InputSnapshot(source=source['source'],snapshot_sha256=snapshot_sha256,event_time_start=min(rows),event_time_end=max(rows),available_at=datetime.fromisoformat(source['available_at']),captured_at=datetime.fromisoformat(source['captured_at']),quality_flags=(),outage=False),),calibration_method='unavailable',calibration_state_version='not-calibrated',calibration_state_sha256=None,nominal_coverage=.9,effective_alpha=None,deliveries=tuple(points))
    return p,store.commit(p)

def capture_fallback48(store: Journal,run_id: str,day: date,arm: str,snapshot_sha256: str,experiment_dir: Path):
    """Frozen V2 load-only model with source availability and exact calibration identity."""
    import pandas as pd
    import numpy as np
    import joblib
    from src.features.engineer import FeatureEngineer
    from src.uncertainty.schemas import CalibrationTable
    from src.operations.models import static_intervals
    metadata=json.loads((experiment_dir/'fallback_metadata.json').read_text())
    model_bytes=(experiment_dir/'fallback.joblib').read_bytes()
    cal_bytes=(experiment_dir/'fallback_calibration.json').read_bytes()
    if digest(model_bytes)!=metadata['model_sha256'] or digest(cal_bytes)!=metadata['calibration_sha256']:
        raise ValueError('V2 artifact identity mismatch')
    if metadata['feature_policy']!='fixed_gate_lag48_weather48_rolling48_v2' or metadata['min_lag_hours']!=48:
        raise ValueError('unregistered V2 feature policy')
    table=CalibrationTable.model_validate_json(cal_bytes)
    if table.model_sha256!=metadata['model_sha256']: raise ValueError('V2 model/calibration mismatch')
    state_hash=store.snapshot(cal_bytes,'issuance-v2-static-calibration',utc(store.clock()))
    source=store.get('snapshot:'+snapshot_sha256)
    data=json.loads(store.snapshot_bytes(snapshot_sha256))
    if data.get('format')!='smard-hourly-mw-v1' or data.get('unit')!='MW' or source['source']!='smard-normalized-hourly':
        raise ValueError('source contract mismatch')
    issued=utc(store.clock()); grid=delivery_grid(day)
    history=pd.DataFrame(data['rows']).rename(columns={'load_mw':'grid_load_mw'})
    history['timestamp']=pd.to_datetime(history.timestamp,utc=True)
    if history.timestamp.duplicated().any() or not np.isfinite(history.grid_load_mw).all(): raise ValueError('invalid source rows')
    history=history.sort_values('timestamp')
    if ((history.timestamp+pd.Timedelta(hours=1))>pd.Timestamp(source['available_at'])).any(): raise ValueError('future source observation')
    if issued-history.timestamp.iloc[-1].to_pydatetime()-timedelta(hours=1)>timedelta(hours=48): raise ValueError('stale source')
    last_needed=grid[-1]-timedelta(hours=48)
    if last_needed+timedelta(hours=1)>issued or history.timestamp.iloc[-1]<last_needed: raise ValueError('48h lag unavailable at issuance')
    earliest=grid[0]-timedelta(hours=216)
    index=pd.date_range(earliest,grid[-1],freq='h')
    full=history.set_index('timestamp').reindex(index).rename_axis('timestamp').reset_index()
    if full.loc[full.timestamp<=pd.Timestamp(last_needed),'grid_load_mw'].isna().any(): raise ValueError('missing source history; no interpolation allowed')
    fe=FeatureEngineer(min_lag_hours=48,lags=[48,72,168],weather_mode='load_only')
    features=fe._generate_raw_features(full).set_index('timestamp').loc[pd.DatetimeIndex(grid),metadata['feature_names']]
    if not np.isfinite(features.values).all(): raise ValueError('missing issuance-safe features')
    model=joblib.load(experiment_dir/'fallback.joblib'); predictions=model.predict(features.values)
    intervals=static_intervals(predictions,np.array([t.hour for t in grid]),table,metadata['model_sha256'],'pooled_marginal_mad',.9,metadata['feature_policy'],metadata['feature_policy'])
    points=tuple(Delivery(timestamp=t,point_mw=float(p),lower_mw=intervals['lower'][i] if intervals['lower'] is not None else None,upper_mw=intervals['upper'][i] if intervals['upper'] is not None else None,interval_status=intervals['status'],quality_flags=('research_calibration_temporal_assumptions_unverified',)) for i,(t,p) in enumerate(zip(grid,predictions)))
    run=store.get('run:'+run_id)
    p=Forecast(run_id=run_id,arm=arm,experiment_kind=run['experiment_kind'],record_id=f'{run_id}_{day}_{arm}',protocol_sha256=run['protocol_sha256'],issued_at=issued,issuance_timezone='UTC',delivery_date=day,model_id=metadata['model_id'],model_sha256=metadata['model_sha256'],feature_code_sha256=digest(Path(__file__).read_bytes()),feature_policy=metadata['feature_policy'],route='fallback',routing_reason='load-only V2 arm; no genuine prospective weather-vintage integration',data_quality_flags=('research_calibration_temporal_assumptions_unverified',),outage=False,inputs=(InputSnapshot(source=source['source'],snapshot_sha256=snapshot_sha256,event_time_start=history.timestamp.iloc[0].to_pydatetime(),event_time_end=history.timestamp.iloc[-1].to_pydatetime(),available_at=datetime.fromisoformat(source['available_at']),captured_at=datetime.fromisoformat(source['captured_at']),quality_flags=(),outage=False),),calibration_method='pooled_marginal_mad',calibration_state_version=table.version,calibration_state_sha256=state_hash,nominal_coverage=.9,effective_alpha=.1,deliveries=points)
    return p,store.commit(p)
