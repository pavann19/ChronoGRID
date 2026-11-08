"""Single-run weather evidence; no learned primary model or historical availability fiction."""
from datetime import datetime,date,timedelta,timezone
from urllib.parse import urlencode
import json,math
from pydantic import Field,FiniteFloat,model_validator
from src.evaluation.prospective import Frozen,utc,InputSnapshot
from src.operations.journal import encode,IntegrityFailure
from src.operations.capture import fetch_json,delivery_grid
from src.operations.weather import parse_forecast

ENDPOINT='https://single-runs-api.open-meteo.com/v1/forecast'
MODEL='icon_eu'
VARIABLES='temperature_2m,relative_humidity_2m,wind_speed_10m,direct_radiation'
FORMAT='open-meteo-icon-eu-single-run-v1'
SOURCE='open-meteo-icon-eu-single-run'

class RunRequest(Frozen):
    initialization: datetime
    latitude: FiniteFloat=Field(default=50.1109,ge=-90,le=90)
    longitude: FiniteFloat=Field(default=8.6821,ge=-180,le=180)
    @model_validator(mode='after')
    def policy(self):
        t=utc(self.initialization)
        if t.hour or t.minute or t.second or t.microsecond:raise ValueError('registered weather policy requires 00 UTC initialization')
        return self
    def parameters(self):
        return dict(latitude=self.latitude,longitude=self.longitude,models=MODEL,run=utc(self.initialization).strftime('%Y-%m-%dT%H:%M'),hourly=VARIABLES,timezone='UTC',wind_speed_unit='ms',forecast_days=3)
    def url(self):return ENDPOINT+'?'+urlencode(self.parameters())

def normalize(raw,request):
    body=json.loads(raw)
    if not isinstance(body,dict) or body.get('error'):raise ValueError('weather provider error or multiple-location response')
    lat,lon=body.get('latitude'),body.get('longitude')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (lat,lon)):
        raise ValueError('missing weather response coordinates')
    if abs(lat-request.latitude)>.25 or abs(lon-request.longitude)>.25:raise ValueError('weather grid point differs from requested location')
    rows=parse_forecast(raw)
    if len(rows)!=72:raise ValueError('single-run policy requires complete 72-hour timestamp grid')
    start=utc(request.initialization)
    for i,row in enumerate(rows):
        t=datetime.fromisoformat(row['timestamp'])
        if t!=start+timedelta(hours=i):raise ValueError('weather grid does not start at requested initialization')
        for key in VARIABLES.split(','):
            value=row[key]
            if value is None:continue
            if isinstance(value,bool) or not isinstance(value,(int,float)):raise ValueError('weather values must be numbers or null')
            limits={'temperature_2m':(-100,70),'relative_humidity_2m':(0,100),'wind_speed_10m':(0,150),'direct_radiation':(0,1500)}
            if not limits[key][0]<=value<=limits[key][1]:raise ValueError('weather value outside registered physical guard')
        row['lead_hours']=i
    return dict(response_grid_point=dict(latitude=lat,longitude=lon,elevation_m=body.get('elevation')),rows=rows)

def capture_single_run(store,request: RunRequest,fetch=fetch_json):
    request=RunRequest.model_validate(request)
    if utc(request.initialization)>utc(store.clock()):raise ValueError('future weather initialization')
    spec=dict(format='open-meteo-single-run-request-v1',request=json.loads(request.model_dump_json()),parameters=request.parameters(),url=request.url())
    request_sha=store.snapshot(encode(spec),SOURCE+'-request',utc(store.clock()))
    try:
        raw=fetch(request.url());received=utc(store.clock())
        raw_sha=store.snapshot(raw,SOURCE+'-response',received)
        normalized=normalize(raw,request)
    except IntegrityFailure:raise
    except Exception as exc:
        with store.transaction() as db:
            seq=db.execute('SELECT count(*) FROM events').fetchone()[0]+1
            store._add(db,f'weather-error:{seq}','weather_error',dict(request_sha256=request_sha,model=MODEL,initialization=utc(request.initialization).isoformat(),reason=str(exc)))
        raise
    content=dict(format=FORMAT,request_sha256=request_sha,raw_sha256=raw_sha,source_model_initialization_time=utc(request.initialization).isoformat(),model=MODEL,identity_evidence='requested_run_via_provider_contract; model/initialization not echoed in response',source_publication_time=None,availability='first_seen_local_capture_only',radiation_semantics='direct_radiation at t is mean over [t-1h,t); other fields instantaneous at t',**normalized)
    return store.snapshot(encode(content),SOURCE,utc(store.clock()))

def aligned_features(request,rows,day):
    """Shared variable/time alignment; caller separately establishes availability."""
    indexed={datetime.fromisoformat(r['timestamp']):r for r in rows};features=[]
    for t in delivery_grid(day):
        start=indexed.get(t);end=indexed.get(t+timedelta(hours=1))
        if start is None or end is None:raise ValueError('weather run does not span delivery day')
        values=dict(temperature_instant_c=start['temperature_2m'],humidity_instant_pct=start['relative_humidity_2m'],wind_instant_ms=start['wind_speed_10m'],direct_radiation_delivery_mean_wm2=end['direct_radiation'])
        if any(v is None for v in values.values()):raise ValueError('missing delivery weather; no imputation')
        features.append(dict(timestamp=t.isoformat(),radiation_valid_at=(t+timedelta(hours=1)).isoformat(),lead_hours=int((t-utc(request.initialization)).total_seconds()/3600),**values))
    return features

def delivery_weather(store,snapshot_sha256,day: date,issued_at: datetime):
    """Assemble future features only from actually captured evidence; no forecast issued."""
    issued=utc(issued_at);now=utc(store.clock());grid=delivery_grid(day)
    from zoneinfo import ZoneInfo
    gate=datetime.combine(day-timedelta(days=1),datetime.min.time(),ZoneInfo('Europe/Berlin')).replace(hour=12)
    if issued.astimezone(ZoneInfo('Europe/Berlin')).date()!=day-timedelta(days=1) or issued>utc(gate) or issued>=grid[0] or issued>now:raise ValueError('weather selection outside real day-ahead issuance window')
    content=json.loads(store.snapshot_bytes(snapshot_sha256));meta=store.get('snapshot:'+snapshot_sha256)
    if content.get('format')!=FORMAT or meta['source']!=SOURCE:raise ValueError('not a single-run snapshot')
    request_sha=content['request_sha256'];raw_sha=content['raw_sha256']
    spec=json.loads(store.snapshot_bytes(request_sha));request=RunRequest.model_validate(spec['request'])
    if spec.get('format')!='open-meteo-single-run-request-v1' or spec['url']!=request.url() or spec['parameters']!=request.parameters():raise ValueError('weather request lineage mismatch')
    if content['source_model_initialization_time']!=utc(request.initialization).isoformat() or content['model']!=MODEL:raise ValueError('weather model/run mismatch')
    reconstructed=normalize(store.snapshot_bytes(raw_sha),request)
    if content['rows']!=reconstructed['rows'] or content['response_grid_point']!=reconstructed['response_grid_point']:raise ValueError('normalized weather differs from raw response')
    inputs=[]
    for h,source in [(request_sha,SOURCE+'-request'),(raw_sha,SOURCE+'-response'),(snapshot_sha256,SOURCE)]:
        item=store.get('snapshot:'+h)
        if item['source']!=source or max(utc(datetime.fromisoformat(item['available_at'])),utc(datetime.fromisoformat(item['captured_at'])))>issued:
            raise ValueError('weather evidence unavailable at issuance or wrong source')
        inputs.append(InputSnapshot(source=source,snapshot_sha256=h,event_time_start=request.initialization,event_time_end=request.initialization,available_at=item['available_at'],captured_at=item['captured_at'],quality_flags=('provider_run_identity_not_independently_attested','single_grid_point_not_national_aggregate'),outage=False))
    age=issued-utc(request.initialization)
    if not timedelta(0)<=age<=timedelta(hours=36):raise ValueError('stale weather run')
    features=aligned_features(request,reconstructed['rows'],day)
    return dict(format='chronogrid-delivery-weather-input-v1',delivery_date=day.isoformat(),snapshot_sha256=snapshot_sha256,initialization=utc(request.initialization).isoformat(),model=MODEL,issued_at=issued.isoformat(),features=features,inputs=[json.loads(x.model_dump_json()) for x in inputs],scope='input evidence only; no trained weather primary, calibration, prediction or independent timestamp proof')
