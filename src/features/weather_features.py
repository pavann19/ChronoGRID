"""
Weather and Thermodynamics Feature Engineering (P5.3).
Heating/Cooling degree hours, wind chill index, and solar interactions.
"""

import numpy as np
import pandas as pd


def add_weather_features(
    df: pd.DataFrame,
    heating_base: float = 18.0,
    cooling_base: float = 22.0,
) -> pd.DataFrame:
    """
    Computes thermodynamic weather features relevant for electricity demand.
    """
    out = df.copy()

    if "temperature_c" in out.columns:
        temp = out["temperature_c"]
        # Heating Degree Hours (HDH): captures space heating / heat pump demand
        out["heating_degree_hours"] = np.maximum(0.0, heating_base - temp)
        
        # Cooling Degree Hours (CDH): captures air conditioning load
        out["cooling_degree_hours"] = np.maximum(0.0, temp - cooling_base)

        # Extreme temperature quadratic terms (U-shaped electricity demand curve)
        out["temperature_squared"] = temp ** 2

        # Wind Chill Index (Environment Canada formula adapted for Celsius)
        if "wind_speed_ms" in out.columns:
            wind_kmh = out["wind_speed_ms"] * 3.6
            # Valid for T <= 10C and wind > 4.8 km/h
            wchill = (
                13.12
                + 0.6215 * temp
                - 11.37 * np.power(np.maximum(wind_kmh, 1.0), 0.16)
                + 0.3965 * temp * np.power(np.maximum(wind_kmh, 1.0), 0.16)
            )
            out["wind_chill_c"] = np.where(temp <= 10.0, wchill, temp)

        # Solar Radiation Interaction (solar warmth reduces net grid heating demand)
        if "solar_radiation_wm2" in out.columns:
            out["temp_solar_interaction"] = (temp * out["solar_radiation_wm2"]) / 1000.0

    return out
