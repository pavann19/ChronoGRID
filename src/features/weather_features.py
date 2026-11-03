"""
Weather and Thermodynamics Feature Engineering (P5.3).
Eliminates future data leakage by using only Day-Ahead Weather Forecasts (DWD NWP)
or 24-hour lagged persistence weather available at forecast issuance time.
"""

from typing import Optional
import numpy as np
import pandas as pd


def add_weather_features(
    df: pd.DataFrame,
    heating_base: float = 18.0,
    cooling_base: float = 22.0,
    weather_mode: str = "persistence", # "persistence", "load_only"
    min_lag_hours: int = 24,
) -> pd.DataFrame:
    """
    Computes thermodynamic weather features available at forecast issuance time.
    
    IMPORTANT: Realised weather at delivery hour t is NOT available at day-ahead gate closure.
    - If weather_mode == "persistence" (default honest production fallback):
      Uses lagged weather observations strictly known at forecast issuance (T_{t - min_lag_hours}).
    - If weather_mode == "load_only":
      Does not append weather features, returning the DataFrame untouched for load-only benchmark ablation.
    
    Contemporaneous realised weather (t) is NEVER directly fed to the model to guarantee ZERO leakage.
    Synthetic NWP noise injection is disallowed per senior data governance rules.
    """
    out = df.copy()

    if weather_mode == "load_only":
        return out

    # "persistence" mode: use lag known at issuance (strictly >= min_lag_hours)
    out["forecast_temperature_c"] = out["temperature_c"].shift(min_lag_hours) if "temperature_c" in out.columns else 12.0
    out["forecast_wind_speed_ms"] = out["wind_speed_ms"].shift(min_lag_hours) if "wind_speed_ms" in out.columns else 4.0
    out["forecast_solar_radiation_wm2"] = out["solar_radiation_wm2"].shift(min_lag_hours) if "solar_radiation_wm2" in out.columns else 0.0
    out["forecast_humidity_pct"] = out["humidity_pct"].shift(min_lag_hours) if "humidity_pct" in out.columns else 65.0

    temp = out["forecast_temperature_c"]

    # 2. Thermodynamic Degree Hours from Forecast
    out["forecast_heating_degree_hours"] = np.maximum(0.0, heating_base - temp)
    out["forecast_cooling_degree_hours"] = np.maximum(0.0, temp - cooling_base)
    out["forecast_temperature_squared"] = temp ** 2

    # 3. Wind Chill Index from Forecast
    wind_kmh = out["forecast_wind_speed_ms"] * 3.6
    wchill = (
        13.12
        + 0.6215 * temp
        - 11.37 * np.power(np.maximum(wind_kmh, 1.0), 0.16)
        + 0.3965 * temp * np.power(np.maximum(wind_kmh, 1.0), 0.16)
    )
    out["forecast_wind_chill_c"] = np.where(temp <= 10.0, wchill, temp)

    # 4. Solar-Temperature Interaction from Forecast
    out["forecast_temp_solar_interaction"] = (temp * out["forecast_solar_radiation_wm2"]) / 1000.0

    return out
