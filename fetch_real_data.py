"""
Probe real SMARD and DWD/Open-Meteo APIs.
"""
import requests
import json
import pandas as pd
from datetime import datetime, timezone

print("Testing SMARD API...")
# SMARD Total load filter=410, region=DE, resolution=hour
index_url = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
try:
    r = requests.get(index_url, timeout=10)
    print("SMARD index status:", r.status_code)
    if r.status_code == 200:
        data = r.json()
        timestamps = data.get("timestamps", [])
        print(f"SMARD timestamps available: {len(timestamps)} slices")
        if timestamps:
            # Look at a recent slice
            recent_ts = timestamps[-1]
            data_url = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{recent_ts}.json"
            r2 = requests.get(data_url, timeout=10)
            print("SMARD data slice status:", r2.status_code)
            if r2.status_code == 200:
                s_data = r2.json()
                series = s_data.get("series", [])
                print(f"SMARD slice {recent_ts} has {len(series)} hourly records")
                if series:
                    t0 = datetime.fromtimestamp(series[0][0]/1000, tz=timezone.utc)
                    t1 = datetime.fromtimestamp(series[-1][0]/1000, tz=timezone.utc)
                    print(f"Range: {t0} to {t1}, first val: {series[0][1]} MW, last val: {series[-1][1]} MW")
except Exception as e:
    print("SMARD error:", e)

print("\nTesting Open-Meteo German Weather Archive (DWD Station / Centroid)...")
# Frankfurt centroid (Lat 50.1109, Lon 8.6821)
weather_url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 50.1109,
    "longitude": 8.6821,
    "start_date": "2024-01-01",
    "end_date": "2024-01-07",
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,direct_radiation",
    "timezone": "UTC"
}
try:
    rw = requests.get(weather_url, params=params, timeout=10)
    print("Weather archive status:", rw.status_code)
    if rw.status_code == 200:
        wdata = rw.json()
        times = wdata.get("hourly", {}).get("time", [])
        temps = wdata.get("hourly", {}).get("temperature_2m", [])
        print(f"Weather hours fetched: {len(times)}, temp sample: {temps[:5]}")
except Exception as e:
    print("Weather error:", e)

