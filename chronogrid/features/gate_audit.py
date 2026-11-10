"""Event-time availability audit; publication latency needs separate source vintages."""
from datetime import timedelta
import re
import pandas as pd


def audit_gate_availability(delivery_timestamps, feature_names, min_lag_hours=48, assert_available=True):
    """Resolve latest raw source time for every feature at every delivery hour.

    Input must be timezone aware and hourly. Calendar values are known at the gate. Observations label hourly interval starts;
    availability uses the interval end (not the beginning).
    Differences inherit the newest lag. Rolling windows end at t-min_lag_hours.
    Unknown feature names fail closed rather than being assumed available.
    """
    times = pd.DatetimeIndex(delivery_timestamps)
    if times.tz is None:
        raise ValueError("Timezone-aware delivery timestamps required")
    times = times.tz_convert("UTC")
    calendar = {"sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "sin_month", "cos_month", "sin_dayofyear", "cos_dayofyear", "is_weekend", "is_holiday", "is_working_day"}
    rows = []
    for delivery in times:
        day = delivery.tz_convert("Europe/Berlin").date()
        gate = pd.Timestamp(str(day-timedelta(days=1))+" 12:00", tz="Europe/Berlin").tz_convert("UTC")
        for name in feature_names:
            if name in calendar:
                source = gate
            elif "_lag_" in name:
                lag = int(name.rsplit("_lag_", 1)[1])
                source = delivery-pd.Timedelta(hours=lag)
            elif "_roll_" in name or name.startswith("forecast_"):
                source = delivery-pd.Timedelta(hours=min_lag_hours)
            elif "_diff_" in name:
                lags = [int(x) for x in re.findall(r"\d+", name.rsplit("_diff_", 1)[1])]
                source = delivery-pd.Timedelta(hours=min(min_lag_hours, max(lags)))
            else:
                raise ValueError(f"Unknown feature provenance: {name}")
            # Observations label interval starts; the complete hourly value is known at its end.
            if name not in calendar:
                source += pd.Timedelta(hours=1)
            rows.append({"delivery_utc": delivery, "gate_utc": gate, "feature": name, "latest_source_utc": source, "available": source <= gate})
    result = pd.DataFrame(rows)
    if assert_available and not result.available.all():
        raise ValueError(f"Post-gate source observations: {int((~result.available).sum())}")
    return result
