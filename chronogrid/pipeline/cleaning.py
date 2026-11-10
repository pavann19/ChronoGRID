"""
Data Cleaning and Normalization Module (P5.2).
Handles deduplication, chronological reindexing, gap interpolation, and outlier handling.
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans raw electricity and weather time series.
    """
    def __init__(self, max_interpolation_gap_hours: int = 3):
        self.max_interpolation_gap_hours = max_interpolation_gap_hours

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Executes cleaning pipeline:
        1. Ensures UTC timezone.
        2. Deduplicates timestamps.
        3. Sorts chronologically.
        4. Reindexes to unbroken hourly frequency.
        5. Performs controlled interpolation with quality tracking.
        """
        cleaning_stats = {
            "initial_rows": len(df),
            "duplicates_removed": 0,
            "missing_hours_inserted": 0,
            "interpolated_values_count": 0,
            "final_rows": 0,
        }

        cleaned = df.copy()

        # 1. Standardize timestamp to UTC datetime
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True)

        # 2. Deduplicate
        initial_count = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last")
        cleaning_stats["duplicates_removed"] = initial_count - len(cleaned)

        # 3. Sort chronologically
        cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

        # 4. Construct unbroken hourly grid
        start_ts = cleaned["timestamp"].min()
        end_ts = cleaned["timestamp"].max()
        full_grid = pd.date_range(start=start_ts, end=end_ts, freq="1h", tz="UTC")

        cleaning_stats["missing_hours_inserted"] = len(full_grid) - len(cleaned)
        
        cleaned = cleaned.set_index("timestamp").reindex(full_grid)
        cleaned.index.name = "timestamp"

        # Initialize tracking columns if missing
        if "is_interpolated" not in cleaned.columns:
            cleaned["is_interpolated"] = False
        if "quality_flag" not in cleaned.columns:
            cleaned["quality_flag"] = "VALID"

        # 5. Targeted interpolation on numeric series (with DatetimeIndex)
        numeric_cols = [
            "grid_load_mw", "day_ahead_price_eur_mwh", "solar_generation_mw",
            "wind_generation_mw", "temperature_c", "humidity_pct",
            "wind_speed_ms", "solar_radiation_wm2"
        ]

        for col in numeric_cols:
            if col in cleaned.columns:
                null_mask = cleaned[col].isnull()
                if null_mask.any():
                    filled = cleaned[col].ffill(limit=self.max_interpolation_gap_hours)
                    filled_mask = null_mask & filled.notna()
                    cleaned.loc[filled_mask, "is_interpolated"] = True
                    cleaned.loc[filled_mask, "quality_flag"] = "CAUSAL_FORWARD_FILL"
                    cleaned[col] = filled
                    cleaning_stats["interpolated_values_count"] += int(filled_mask.sum())

        # Drop unknown leading values and observations beyond the causal fill bound.
        cleaned = cleaned.dropna(subset=[c for c in numeric_cols if c in cleaned])

        # Reset index back to column
        cleaned.reset_index(inplace=True)

        # Ensure solar radiation at night is strictly 0
        hours = cleaned["timestamp"].dt.hour
        solar_night_mask = (hours < 4) | (hours > 21)
        if "solar_radiation_wm2" in cleaned.columns:
            cleaned.loc[solar_night_mask, "solar_radiation_wm2"] = 0.0
        if "solar_generation_mw" in cleaned.columns:
            cleaned.loc[solar_night_mask, "solar_generation_mw"] = 0.0

        cleaning_stats["final_rows"] = len(cleaned)
        logger.info(f"Cleaning complete: {cleaning_stats}")
        return cleaned, cleaning_stats
