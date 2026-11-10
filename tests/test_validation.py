"""
Tests for Data Validation and Cleaning Pipelines (P5.2).
"""

import numpy as np
import pandas as pd
from chronogrid.pipeline.validation import DataValidator
from chronogrid.pipeline.cleaning import DataCleaner


def test_validator_detects_out_of_bounds():
    validator = DataValidator()
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC"),
        "grid_load_mw": [50000.0] * 9 + [150000.0], # 150 GW is physically impossible
        "temperature_c": [12.0] * 10,
    })
    report = validator.validate(df)
    assert report["range_violations"]["grid_load_mw"] == 1


def test_cleaner_deduplicates_and_interpolates():
    cleaner = DataCleaner(max_interpolation_gap_hours=3)
    times = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": [times[0], times[1], times[1], times[3], times[4]], # Duplicate times[1], missing times[2]
        "grid_load_mw": [50000.0, 52000.0, 52000.0, np.nan, 56000.0],
        "temperature_c": [10.0, 11.0, 11.0, 12.0, 13.0],
    })
    cleaned, stats = cleaner.clean(df)
    assert len(cleaned) == 5
    assert stats["duplicates_removed"] == 1
    assert stats["missing_hours_inserted"] == 1
    assert not cleaned["grid_load_mw"].isnull().any()

