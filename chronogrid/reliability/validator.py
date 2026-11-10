"""
Weather Input Validator for ChronoGrid DDIR.
Inspects incoming weather telemetry against physical, temporal, schema, and completeness bounds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Detailed audit report produced by WeatherInputValidator."""

    is_valid: bool
    quality_score: float  # [0.0, 1.0]
    violations: List[str] = field(default_factory=list)
    missing_rate: float = 0.0
    freshness_age_hours: Optional[float] = None
    channel_states: Dict[str, str] = field(default_factory=dict)
    can_interpolate: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class WeatherInputValidator:
    """
    Deterministic validator for external weather telemetry.
    Checks physical boundaries (WMO standards), schema, completeness, and temporal freshness.
    """

    # WMO physical validity ranges for Central Europe / Germany
    PHYSICAL_BOUNDS = {
        "temperature_c": (-40.0, 48.0),  # Ambient air temperature (°C)
        "wind_speed_ms": (0.0, 60.0),  # 10m wind speed (m/s)
        "solar_radiation_wm2": (0.0, 1250.0),  # Surface solar irradiance (W/m²)
        "humidity_pct": (0.0, 100.0),  # Relative humidity (%)
    }

    REQUIRED_CHANNELS = ["temperature_c", "wind_speed_ms", "solar_radiation_wm2", "humidity_pct"]

    def __init__(
        self,
        max_freshness_hours: float = 28.0,
        stale_threshold_hours: float = 36.0,
        max_missing_rate: float = 0.15,
    ):
        self.max_freshness_hours = max_freshness_hours
        self.stale_threshold_hours = stale_threshold_hours
        self.max_missing_rate = max_missing_rate

    def validate(
        self,
        weather_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]],
        current_time: Optional[datetime] | None = None,
        data_timestamp: Optional[Union[str, datetime]] | None = None,
    ) -> ValidationReport:
        """
        Validates a weather telemetry payload.
        """
        violations = []
        channel_states = {}

        # 1. Null / Missing Payload Check
        if weather_data is None:
            return ValidationReport(
                is_valid=False,
                quality_score=0.0,
                violations=["Weather payload is null/missing (Provider Unavailable)"],
                missing_rate=1.0,
                freshness_age_hours=None,
                channel_states={c: "MISSING" for c in self.REQUIRED_CHANNELS},
                can_interpolate=False,
            )

        # Convert to standardized DataFrame or list of dicts
        if isinstance(weather_data, dict):
            if not weather_data:
                return ValidationReport(
                    is_valid=False,
                    quality_score=0.0,
                    violations=["Weather payload is empty dictionary"],
                    missing_rate=1.0,
                    channel_states={c: "MISSING" for c in self.REQUIRED_CHANNELS},
                )
            rows = [weather_data]
        elif isinstance(weather_data, list):
            if len(weather_data) == 0:
                return ValidationReport(
                    is_valid=False,
                    quality_score=0.0,
                    violations=["Weather payload is empty list"],
                    missing_rate=1.0,
                    channel_states={c: "MISSING" for c in self.REQUIRED_CHANNELS},
                )
            rows = weather_data
        elif isinstance(weather_data, pd.DataFrame):
            if weather_data.empty:
                return ValidationReport(
                    is_valid=False,
                    quality_score=0.0,
                    violations=["Weather DataFrame is empty"],
                    missing_rate=1.0,
                    channel_states={c: "MISSING" for c in self.REQUIRED_CHANNELS},
                )
            rows = weather_data.to_dict(orient="records")
        else:
            return ValidationReport(
                is_valid=False,
                quality_score=0.0,
                violations=[f"Unsupported weather payload type: {type(weather_data).__name__}"],
                missing_rate=1.0,
            )

        # 2. Schema and Channel Completeness Check
        total_fields = len(rows) * len(self.REQUIRED_CHANNELS)
        missing_count = 0
        outlier_count = 0
        type_error_count = 0

        # Inspect data_timestamp from payload if present
        if data_timestamp is None and "timestamp" in rows[0]:
            data_timestamp = rows[0]["timestamp"]

        for ch in self.REQUIRED_CHANNELS:
            ch_missing = 0
            ch_outlier = 0
            low_b, high_b = self.PHYSICAL_BOUNDS[ch]

            for row in rows:
                val = row.get(ch)
                if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                    ch_missing += 1
                    missing_count += 1
                else:
                    try:
                        f_val = float(val)
                        if f_val < low_b or f_val > high_b:
                            ch_outlier += 1
                            outlier_count += 1
                    except (ValueError, TypeError):
                        type_error_count += 1
                        missing_count += 1

            if ch_missing == len(rows):
                channel_states[ch] = "MISSING"
                violations.append(f"Channel '{ch}' completely missing")
            elif ch_outlier > 0:
                channel_states[ch] = "OUTLIER"
                violations.append(
                    f"Channel '{ch}' contains {ch_outlier} physical outlier(s) outside [{low_b}, {high_b}]"
                )
            elif ch_missing > 0:
                channel_states[ch] = "PARTIAL"
                violations.append(f"Channel '{ch}' has {ch_missing}/{len(rows)} missing values")
            else:
                channel_states[ch] = "OK"

        if type_error_count > 0:
            violations.append(f"Payload contains {type_error_count} malformed non-numeric values")

        missing_rate = missing_count / max(1, total_fields)
        outlier_rate = outlier_count / max(1, total_fields)

        # 3. Temporal Freshness Check
        freshness_age_hours = None
        now_utc = current_time or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        if data_timestamp is not None:
            try:
                if isinstance(data_timestamp, str):
                    try:
                        ts = datetime.fromisoformat(data_timestamp)
                    except Exception:
                        ts = pd.to_datetime(data_timestamp, utc=True).to_pydatetime()
                elif isinstance(data_timestamp, datetime):
                    ts = (
                        data_timestamp
                        if data_timestamp.tzinfo is not None
                        else data_timestamp.replace(tzinfo=timezone.utc)
                    )
                else:
                    ts = None

                if ts is not None:
                    delta_seconds = (now_utc - ts).total_seconds()
                    if delta_seconds < -300.0:  # More than 5 minutes ahead of current time
                        future_lead_hours = abs(delta_seconds) / 3600.0
                        freshness_age_hours = -round(future_lead_hours, 1)
                        violations.append(
                            f"Weather data timestamp is in the FUTURE: {future_lead_hours:.1f}h ahead of current system time (Invalid temporal provenance)"
                        )
                    else:
                        freshness_age_hours = max(0.0, delta_seconds / 3600.0)
                        if freshness_age_hours > self.stale_threshold_hours:
                            violations.append(
                                f"Weather data is STALE: age {freshness_age_hours:.1f}h exceeds stale threshold {self.stale_threshold_hours}h"
                            )
                        elif freshness_age_hours > self.max_freshness_hours:
                            violations.append(
                                f"Weather data is DELAYED: age {freshness_age_hours:.1f}h exceeds freshness SLA {self.max_freshness_hours}h"
                            )
            except Exception as e:
                violations.append(f"Failed to parse weather timestamp '{data_timestamp}': {e}")

        # 4. Quality Score Calculation
        # Q in [0.0, 1.0]
        freshness_penalty = 0.0
        if freshness_age_hours is not None:
            if freshness_age_hours < 0:
                freshness_penalty = 1.0  # Future timestamp is completely penalized
            else:
                freshness_penalty = min(0.3, (freshness_age_hours / 48.0) * 0.3)

        q_score = 1.0 - (0.4 * missing_rate + 0.3 * outlier_rate + freshness_penalty)
        if type_error_count > 0:
            q_score = min(q_score, 0.2)
        if any("future" in v.lower() for v in violations):
            q_score = 0.0
        q_score = max(0.0, min(1.0, float(q_score)))

        # 5. Interpolation & Recovery Feasibility
        # If temperature is OK and only secondary channels (humidity, wind) are missing, we can interpolate
        can_interpolate = (
            channel_states.get("temperature_c") == "OK"
            and missing_rate <= self.max_missing_rate
            and outlier_rate == 0.0
            and (
                freshness_age_hours is None
                or (0.0 <= freshness_age_hours <= self.max_freshness_hours)
            )
            and not any("future" in v.lower() for v in violations)
        )

        is_valid = len(violations) == 0

        return ValidationReport(
            is_valid=is_valid,
            quality_score=round(q_score, 3),
            violations=violations,
            missing_rate=round(missing_rate, 3),
            freshness_age_hours=round(freshness_age_hours, 1)
            if freshness_age_hours is not None
            else None,
            channel_states=channel_states,
            can_interpolate=can_interpolate,
            details={
                "n_rows": len(rows),
                "type_errors": type_error_count,
                "outlier_count": outlier_count,
            },
        )
