import json
import pytest
from chronogrid.operations.weather import parse_forecast

def test_weather_vintage_units_and_missingness():
    p=dict(utc_offset_seconds=0,hourly_units={'temperature_2m':'°C','relative_humidity_2m':'%','wind_speed_10m':'m/s','direct_radiation':'W/m²'},hourly={'time':['2026-11-05T00:00','2026-11-05T01:00'],'temperature_2m':[10,None],'relative_humidity_2m':[80,80],'wind_speed_10m':[4,5],'direct_radiation':[0,0]})
    rows=parse_forecast(json.dumps(p).encode());assert rows[1]['temperature_2m'] is None and rows[0]['timestamp'].endswith('+00:00')
    p['hourly_units']['wind_speed_10m']='km/h'
    with pytest.raises(ValueError,match='unit'):parse_forecast(json.dumps(p).encode())
