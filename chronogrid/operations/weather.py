"""Archive genuine current forecast responses without relabeling them historical vintages."""
from datetime import datetime,timezone,timedelta
from urllib.parse import urlencode
import json,math
from chronogrid.operations.capture import fetch_json
from chronogrid.operations.journal import encode
from chronogrid.evaluation.prospective import utc

PARAMS=dict(latitude=50.1109,longitude=8.6821,hourly='temperature_2m,relative_humidity_2m,wind_speed_10m,direct_radiation',timezone='UTC',wind_speed_unit='ms',models='icon_seamless',forecast_days=3)

def parse_forecast(raw: bytes):
    payload=json.loads(raw)
    if payload.get('utc_offset_seconds')!=0: raise ValueError('forecast response not UTC')
    units=payload['hourly_units']
    expected={'temperature_2m':'°C','relative_humidity_2m':'%','wind_speed_10m':'m/s','direct_radiation':'W/m²'}
    if any(units.get(k)!=v for k,v in expected.items()): raise ValueError('weather unit mismatch')
    hourly=payload['hourly']; times=hourly['time']; rows=[]; seen=set()
    if any(len(hourly[k])!=len(times) for k in expected): raise ValueError('weather series length mismatch')
    for i,ts in enumerate(times):
        t=datetime.fromisoformat(ts)
        if t.tzinfo is not None and t.utcoffset()!=timedelta(0): raise ValueError('weather timestamp offset contradicts UTC contract')
        t=t.replace(tzinfo=timezone.utc)
        if rows and t!=datetime.fromisoformat(rows[-1]['timestamp'])+timedelta(hours=1): raise ValueError('weather time gap or non-monotonic order')
        if t in seen or t.minute or t.second: raise ValueError('duplicate/non-hourly weather timestamp')
        seen.add(t); values={k:hourly[k][i] for k in expected}
        if any(v is not None and not math.isfinite(float(v)) for v in values.values()): raise ValueError('nonfinite weather forecast')
        rows.append(dict(timestamp=t.isoformat(),**values))
    return rows

def capture_weather(store):
    url='https://api.open-meteo.com/v1/forecast?'+urlencode(PARAMS)
    raw=fetch_json(url); captured=utc(store.clock())
    raw_hash=store.snapshot(raw,'open-meteo-icon-seamless-response',captured)
    rows=parse_forecast(raw)
    normalized=encode(dict(format='open-meteo-icon-forecast-v1',source='Open-Meteo DWD ICON seamless',parameters=PARAMS,raw_sha256=raw_hash,source_model_initialization_time=None,availability='first_seen_local_capture; source generationtime_ms is computation latency, not issuance',rows=rows))
    return store.snapshot(normalized,'open-meteo-icon-seamless-forecast',utc(store.clock()))
