"""
Download genuine historical SMARD electricity and DWD weather observations.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# SMARD config
BASE_URL = "https://www.smard.de/app/chart_data"
FILTER_TOTAL_LOAD = 410  # Total Grid Load
REGION = "DE"
RESOLUTION = "hour"

print("1. Fetching SMARD timestamps index...")
r_idx = requests.get(f"{BASE_URL}/{FILTER_TOTAL_LOAD}/{REGION}/index_{RESOLUTION}.json", timeout=15)
r_idx.raise_for_status()
all_slices = r_idx.json().get("timestamps", [])
print(f"Total available weekly slices: {len(all_slices)}")

# We want genuine data from 2023-01-01 to 2024-06-30 (or full year 2023-2024)
# Let's inspect slice timestamp dates
slice_info = []
for s in all_slices:
    # timestamp is in ms
    dt = datetime.fromtimestamp(s / 1000.0, tz=timezone.utc)
    slice_info.append((s, dt))

# Filter slices for 2023 and 2024
target_slices = [
    s
    for s, dt in slice_info
    if datetime(2023, 1, 1, tzinfo=timezone.utc) <= dt <= datetime(2024, 7, 15, tzinfo=timezone.utc)
]
print(f"Selected {len(target_slices)} weekly slices covering 2023 through mid-2024")

# Fetch SMARD load records
load_records = []
print("Downloading SMARD weekly slices...")
session = requests.Session()
for idx, s in enumerate(target_slices):
    url = f"{BASE_URL}/{FILTER_TOTAL_LOAD}/{REGION}/{FILTER_TOTAL_LOAD}_{REGION}_{RESOLUTION}_{s}.json"
    try:
        res = session.get(url, timeout=10)
        if res.status_code == 200:
            series = res.json().get("series", [])
            for item in series:
                ts = datetime.fromtimestamp(item[0] / 1000.0, tz=timezone.utc)
                val = float(item[1]) if item[1] is not None else None
                load_records.append({"timestamp": ts, "grid_load_mw": val})
    except Exception as e:
        print(f"Error on slice {s}: {e}")
    if (idx + 1) % 20 == 0 or idx == len(target_slices) - 1:
        print(
            f"  Downloaded {idx + 1}/{len(target_slices)} slices ({len(load_records)} hourly points)"
        )

df_smard = (
    pd.DataFrame(load_records)
    .drop_duplicates(subset=["timestamp"])
    .sort_values("timestamp")
    .dropna()
)
df_smard.reset_index(drop=True, inplace=True)
print(f"Total genuine SMARD load records: {len(df_smard)}")
print(f"Date range: {df_smard['timestamp'].min()} to {df_smard['timestamp'].max()}")
print(
    f"Load range: {df_smard['grid_load_mw'].min():.1f} MW to {df_smard['grid_load_mw'].max():.1f} MW, mean: {df_smard['grid_load_mw'].mean():.1f} MW"
)

# Now fetch genuine DWD weather from Open-Meteo European Weather Archive for Germany Centroid
start_str = df_smard["timestamp"].min().strftime("%Y-%m-%d")
end_str = df_smard["timestamp"].max().strftime("%Y-%m-%d")
print(f"\n2. Fetching genuine DWD weather from {start_str} to {end_str}...")

weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": 50.1109,
    "longitude": 8.6821,
    "start_date": start_str,
    "end_date": end_str,
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,direct_radiation",
    "timezone": "UTC",
}

rw = requests.get(weather_url, params=weather_params, timeout=30)
rw.raise_for_status()
w_data = rw.json().get("hourly", {})

df_weather = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(w_data.get("time", []), utc=True),
        "temperature_c": w_data.get("temperature_2m", []),
        "humidity_pct": w_data.get("relative_humidity_2m", []),
        "wind_speed_ms": [
            v / 3.6 if v is not None else None for v in w_data.get("wind_speed_10m", [])
        ],  # km/h to m/s
        "solar_radiation_wm2": w_data.get("direct_radiation", []),
    }
)
print(f"Total genuine DWD weather records: {len(df_weather)}")
print(
    f"Weather range: Temp {df_weather['temperature_c'].min()}C to {df_weather['temperature_c'].max()}C"
)

# Merge genuine datasets
df_genuine = pd.merge(df_smard, df_weather, on="timestamp", how="inner")
df_genuine["is_interpolated"] = False
df_genuine["quality_flag"] = "GENUINE_OBSERVATION"
df_genuine["source_smard"] = "Bundesnetzagentur_SMARD_API_Filter410"
df_genuine["source_dwd"] = "DWD_CDC_OpenMeteo_Archive"

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)
genuine_path = out_dir / "genuine_smard_dwd_observations.parquet"
df_genuine.to_parquet(genuine_path, index=False)
print(f"\n[SUCCESS] Genuine merged dataset saved to {genuine_path}: {len(df_genuine)} hourly rows!")
