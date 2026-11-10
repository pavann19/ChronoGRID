"""
SMARD Electricity Data Ingestion Client (Bundesnetzagentur).
Fetches electricity market data: total grid load, generation by type, and spot price.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class SmardClient:
    """
    Client for the German Federal Network Agency (Bundesnetzagentur) SMARD API.
    API Docs: https://www.smard.de/en/downloadcenter/data-download
    """

    BASE_URL = "https://www.smard.de/app/chart_data"

    # Known SMARD filter IDs
    FILTER_TOTAL_LOAD = 410  # Gesamte Last (Grid Load)
    FILTER_SOLAR = 1226  # Photovoltaik
    FILTER_WIND_ONSHORE = 1225  # Wind Onshore
    FILTER_WIND_OFFSHORE = 1224  # Wind Offshore
    FILTER_DAY_AHEAD_PRICE = 4169  # Day-Ahead Grosshandelspreise

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def fetch_time_series(
        self,
        filter_id: int,
        region: str = "DE",
        resolution: str = "hour",
        timestamp_ms: Optional[int] | None = None,
    ) -> pd.DataFrame:
        """
        Fetches time series from SMARD endpoint.
        Returns DataFrame with ['timestamp', 'value'].
        """
        # If timestamp not provided, get available timestamps index
        if timestamp_ms is None:
            index_url = f"{self.BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
            try:
                resp = requests.get(index_url, timeout=self.timeout)
                resp.raise_for_status()
                timestamps = resp.json().get("timestamps", [])
                if not timestamps:
                    raise ValueError(f"No timestamps returned for filter {filter_id}")
                timestamp_ms = timestamps[-1]  # Most recent slice
            except Exception as e:
                logger.warning(f"Could not fetch index from SMARD ({e}).")
                return pd.DataFrame(columns=["timestamp", "value"])

        data_url = f"{self.BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp_ms}.json"
        try:
            resp = requests.get(data_url, timeout=self.timeout)
            resp.raise_for_status()
            series = resp.json().get("series", [])
            if not series:
                return pd.DataFrame(columns=["timestamp", "value"])

            records = []
            for item in series:
                ts = datetime.fromtimestamp(item[0] / 1000.0, tz=timezone.utc)
                val = float(item[1]) if item[1] is not None else None
                records.append({"timestamp": ts, "value": val})

            df = pd.DataFrame(records)
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch data slice {timestamp_ms} from SMARD: {e}")
            return pd.DataFrame(columns=["timestamp", "value"])
