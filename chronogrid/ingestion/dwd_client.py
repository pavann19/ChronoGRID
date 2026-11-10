"""
DWD (Deutscher Wetterdienst) OpenData Meteorological Ingestion Client.
Provides 2m temperature, relative humidity, wind speed, and solar irradiance.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DwdClient:
    """
    Client for German Weather Service (DWD) OpenData Climate Data Center (CDC).
    Also supports Open-Meteo German archive integration as a modern high-speed proxy.
    """
    CDC_BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
    OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

    # Representative German grid load centroid: Frankfurt am Main (Lat 50.1109, Lon 8.6821)
    DEFAULT_LAT = 50.1109
    DEFAULT_LON = 8.6821

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def fetch_hourly_weather(
        self,
        start_date: str,
        end_date: str,
        latitude: float = DEFAULT_LAT,
        longitude: float = DEFAULT_LON,
    ) -> pd.DataFrame:
        """
        Fetches hourly weather data for the specified date range.
        Columns: timestamp, temperature_c, humidity_pct, wind_speed_ms, solar_radiation_wm2
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,direct_radiation",
            "timezone": "UTC",
        }
        try:
            resp = requests.get(self.OPEN_METEO_ARCHIVE, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            times = pd.to_datetime(hourly.get("time", []), utc=True)
            
            df = pd.DataFrame({
                "timestamp": times,
                "temperature_c": hourly.get("temperature_2m", []),
                "humidity_pct": hourly.get("relative_humidity_2m", []),
                "wind_speed_ms": [v / 3.6 if v is not None else None for v in hourly.get("wind_speed_10m", [])],  # km/h to m/s
                "solar_radiation_wm2": hourly.get("direct_radiation", []),
            })
            return df
        except Exception as e:
            logger.warning(f"Could not fetch live weather from archive ({e}).")
            return pd.DataFrame(columns=[
                "timestamp", "temperature_c", "humidity_pct", "wind_speed_ms", "solar_radiation_wm2"
            ])

