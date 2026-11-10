"""
Data Validation Module for SMARD & DWD Datasets (P5.2).
Validates schema, physical ranges, temporal continuity, and completeness.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class HourlyRecordValidator(BaseModel):
    """Pydantic validator for an individual hourly record."""
    timestamp: pd.Timestamp
    grid_load_mw: float = Field(ge=20000.0, le=120000.0)
    temperature_c: float = Field(ge=-35.0, le=50.0)
    humidity_pct: float = Field(ge=0.0, le=100.0)
    wind_speed_ms: float = Field(ge=0.0, le=65.0)
    solar_radiation_wm2: float = Field(ge=0.0, le=1300.0)

    model_config = {
        "arbitrary_types_allowed": True
    }


class DataValidator:
    """
    Comprehensive dataset validation suite.
    """
    RANGES = {
        "grid_load_mw": (20000.0, 120000.0),
        "temperature_c": (-35.0, 50.0),
        "humidity_pct": (0.0, 100.0),
        "wind_speed_ms": (0.0, 65.0),
        "solar_radiation_wm2": (0.0, 1300.0),
    }

    def __init__(self, max_null_fraction: float = 0.05):
        self.max_null_fraction = max_null_fraction

    def validate(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Runs complete validation suite. Returns detailed validation summary report.
        """
        report = {
            "total_rows": len(df),
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "null_counts": {},
            "range_violations": {},
            "temporal_issues": {},
        }

        if df.empty:
            report["is_valid"] = False
            report["errors"].append("Dataset is empty.")
            return report

        # 1. Required Columns Check
        required_cols = ["timestamp", "grid_load_mw", "temperature_c"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report["is_valid"] = False
            report["errors"].append(f"Missing required columns: {missing_cols}")
            return report

        # 2. Null Fraction Check
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_frac = null_count / len(df)
            report["null_counts"][col] = null_count
            if null_frac > self.max_null_fraction:
                report["is_valid"] = False
                report["errors"].append(
                    f"Column '{col}' has null fraction {null_frac:.3%} exceeding threshold {self.max_null_fraction:.3%}"
                )
            elif null_count > 0:
                report["warnings"].append(f"Column '{col}' contains {null_count} nulls.")

        # 3. Physical Plausibility Range Checks
        for col, (min_val, max_val) in self.RANGES.items():
            if col in df.columns:
                valid_series = df[col].dropna()
                violations = int(((valid_series < min_val) | (valid_series > max_val)).sum())
                report["range_violations"][col] = violations
                if violations > 0:
                    report["warnings"].append(
                        f"Column '{col}' has {violations} values outside physical range [{min_val}, {max_val}]."
                    )

        # 4. Temporal Checks (Duplicates & Gaps)
        ts_series = pd.to_datetime(df["timestamp"], utc=True)
        dup_count = int(ts_series.duplicated().sum())
        report["temporal_issues"]["duplicate_timestamps"] = dup_count
        if dup_count > 0:
            report["warnings"].append(f"Found {dup_count} duplicate timestamps.")

        # Temporal Monotonicity
        is_monotonic = ts_series.is_monotonic_increasing
        report["temporal_issues"]["is_monotonic_increasing"] = bool(is_monotonic)
        if not is_monotonic:
            report["warnings"].append("Timestamps are not strictly monotonically increasing (re-sorting required).")

        # Temporal Gaps Check on sorted unique timestamps
        sorted_unique_ts = ts_series.drop_duplicates().sort_values()
        if len(sorted_unique_ts) > 1:
            diffs = sorted_unique_ts.diff().dropna()
            hourly_delta = pd.Timedelta(hours=1)
            gap_count = int((diffs > hourly_delta).sum())
            report["temporal_issues"]["gap_count"] = gap_count
            if gap_count > 0:
                max_gap = diffs.max()
                report["warnings"].append(f"Found {gap_count} temporal gaps (largest gap: {max_gap}).")

        return report

