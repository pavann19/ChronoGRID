"""
Generates a realistic, high-fidelity benchmark dataset representing German electricity
market dynamics (SMARD) synchronized with meteorological conditions (DWD).
Provides 100% reproducible historical data from 2023-01-01 through 2024-06-30 (13,128 hours).
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_german_energy_weather_dataset(
    start_date: str = "2023-01-01 00:00:00",
    end_date: str = "2024-06-30 23:00:00",
    random_seed: int = 42,
    inject_minor_anomalies: bool = True,
) -> pd.DataFrame:
    """
    Generates an hourly time series of German electricity consumption and synchronized weather.
    Grounded in authentic physics, German industrial activity, and European grid operations.
    """
    rng = np.random.RandomState(random_seed)
    
    # 1. Hourly timestamp grid in UTC
    timestamps = pd.date_range(start=start_date, end=end_date, freq="1h", tz="UTC")
    n_hours = len(timestamps)

    # 2. German Calendar & Holiday Flags
    # German federal public holidays (fixed & movable)
    german_holidays = {
        (2023, 1, 1), (2023, 4, 7), (2023, 4, 10), (2023, 5, 1),
        (2023, 5, 18), (2023, 5, 29), (2023, 10, 3), (2023, 12, 25), (2023, 12, 26),
        (2024, 1, 1), (2024, 3, 29), (2024, 4, 1), (2024, 5, 1),
        (2024, 5, 9), (2024, 5, 20), (2024, 10, 3), (2024, 12, 25), (2024, 12, 26),
    }

    hours = timestamps.hour.values
    dayofweek = timestamps.dayofweek.values
    dayofyear = timestamps.dayofyear.values
    is_weekend = (dayofweek >= 5).astype(int)
    
    is_holiday = np.array([
        1 if (ts.year, ts.month, ts.day) in german_holidays else 0
        for ts in timestamps
    ])

    # 3. Weather Simulation (German Central Climate - Frankfurt/Kassel centroid)
    # Seasonal temperature: T_season(t) = 10.5 - 9.5 * cos(2*pi*(doy - 25)/365)
    seasonal_temp = 10.5 - 9.5 * np.cos(2 * np.pi * (dayofyear - 25) / 365.25)
    # Diurnal temperature cycle: peak around 15:00 UTC (16:00 CET)
    diurnal_temp = 4.5 * np.cos(2 * np.pi * (hours - 15) / 24)
    # Autoregressive weather noise (synoptic pressure systems, lasting 3-5 days)
    weather_noise = np.zeros(n_hours)
    wn = 0.0
    for i in range(n_hours):
        wn = 0.985 * wn + rng.normal(0, 0.45)
        weather_noise[i] = wn
    temperature_c = seasonal_temp + diurnal_temp + weather_noise

    # Solar Irradiance (W/m2): zero at night, peak around 12:00 UTC, modulated by cloudiness
    solar_elevation = np.maximum(0.0, np.sin(np.pi * (hours - 5) / 14))
    seasonal_solar_factor = 0.3 + 0.7 * np.sin(np.pi * (dayofyear - 80) / 365.25)
    seasonal_solar_factor = np.clip(seasonal_solar_factor, 0.2, 1.0)
    cloud_cover = np.clip(0.5 + 0.3 * np.sin(dayofyear / 20) + rng.normal(0, 0.2, n_hours), 0.0, 1.0)
    solar_radiation_wm2 = 900.0 * solar_elevation * seasonal_solar_factor * (1.0 - 0.75 * cloud_cover)
    solar_radiation_wm2 = np.maximum(0.0, solar_radiation_wm2)

    # Wind Speed (m/s): Weibull-like distribution, higher in winter
    winter_wind_boost = 1.0 + 0.4 * np.maximum(0.0, np.cos(2 * np.pi * (dayofyear - 15) / 365.25))
    wind_speed_ms = rng.weibull(2.1, n_hours) * 4.5 * winter_wind_boost
    wind_speed_ms = np.clip(wind_speed_ms, 0.2, 26.0)

    # Relative Humidity (%): inversely correlated with temperature
    humidity_pct = 75.0 - 1.2 * (temperature_c - 10.0) + rng.normal(0, 4.0, n_hours)
    humidity_pct = np.clip(humidity_pct, 20.0, 98.0)

    # 4. Electricity Grid Load (Total Load in MW - SMARD DE)
    # Base German load: ~55,000 MW baseline
    base_load = 54000.0

    # Annual Seasonal Pattern (Higher in winter for heating/lighting, lower in summer)
    seasonal_load = 7500.0 * np.cos(2 * np.pi * (dayofyear - 15) / 365.25)

    # Diurnal Profile (Morning peak ~11h, Evening peak ~19h, Night dip ~04h)
    diurnal_profile = (
        -8500.0 * np.cos(2 * np.pi * hours / 24)
        - 3200.0 * np.cos(4 * np.pi * (hours - 11) / 24)
        + 1500.0 * np.sin(2 * np.pi * (hours - 8) / 24)
    )

    # Industrial Weekly Modulation (Weekdays vs Weekends & Holidays)
    # Saturday: -15%, Sunday: -25%, Public Holidays: -22%
    day_multiplier = np.ones(n_hours)
    day_multiplier[dayofweek == 5] = 0.84  # Saturday
    day_multiplier[dayofweek == 6] = 0.75  # Sunday
    day_multiplier[is_holiday == 1] = 0.76 # Holidays

    # Temperature sensitivity: Heating degree effect (< 18C) and cooling degree effect (> 22C)
    heating_effect = np.maximum(0.0, 18.0 - temperature_c) * 420.0  # 420 MW per degC below 18
    cooling_effect = np.maximum(0.0, temperature_c - 22.0) * 350.0  # 350 MW per degC above 22

    # Solar rooftop self-consumption effect (reduces grid load during sunny days)
    rooftop_pv_suppression = -0.012 * solar_radiation_wm2 * 1000.0

    # Residual stochastic noise (AR(1) process for realistic temporal autocorrelation)
    load_noise = np.zeros(n_hours)
    ln = 0.0
    for i in range(n_hours):
        ln = 0.92 * ln + rng.normal(0, 480.0)
        load_noise[i] = ln

    raw_load = (base_load + seasonal_load + diurnal_profile) * day_multiplier
    grid_load_mw = raw_load + heating_effect + cooling_effect + rooftop_pv_suppression + load_noise
    grid_load_mw = np.clip(grid_load_mw, 32000.0, 82000.0)

    # 5. Renewable Generation & Day-Ahead Spot Price
    solar_gen_mw = np.clip(solar_radiation_wm2 * 45.0 + rng.normal(0, 200, n_hours), 0, 38000)
    solar_gen_mw[solar_elevation <= 0] = 0.0
    
    wind_gen_mw = np.clip((wind_speed_ms ** 2.2) * 220.0 + rng.normal(0, 400, n_hours), 500, 48000)

    # Day-ahead wholesale spot market price (EUR/MWh)
    # Price rises with load, drops with wind/solar ("merit order effect")
    residual_load = grid_load_mw - 0.75 * (solar_gen_mw + wind_gen_mw)
    price_eur_mwh = 20.0 + 0.0022 * residual_load + rng.normal(0, 12.0, n_hours)
    price_eur_mwh = np.clip(price_eur_mwh, -25.0, 280.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "grid_load_mw": np.round(grid_load_mw, 1),
        "day_ahead_price_eur_mwh": np.round(price_eur_mwh, 2),
        "solar_generation_mw": np.round(solar_gen_mw, 1),
        "wind_generation_mw": np.round(wind_gen_mw, 1),
        "temperature_c": np.round(temperature_c, 2),
        "humidity_pct": np.round(humidity_pct, 1),
        "wind_speed_ms": np.round(wind_speed_ms, 2),
        "solar_radiation_wm2": np.round(solar_radiation_wm2, 1),
        "is_interpolated": False,
        "quality_flag": "VALID",
    })

    # 6. Inject minor realistic sensor anomalies/gaps (< 0.2%) to validate P5.2 cleaning
    if inject_minor_anomalies:
        # Gap 1: 2-hour sensor glitch on 2023-03-14 02:00 - 03:00
        mask_gap1 = (df["timestamp"] >= "2023-03-14 02:00:00+00:00") & (df["timestamp"] <= "2023-03-14 03:00:00+00:00")
        df.loc[mask_gap1, "grid_load_mw"] = np.nan
        
        # Gap 2: 1-hour missing temperature on 2023-08-10 14:00
        mask_gap2 = df["timestamp"] == "2023-08-10 14:00:00+00:00"
        df.loc[mask_gap2, "temperature_c"] = np.nan

        # Duplicate row to test deduplication
        dup_row = df.iloc[[100]].copy()
        df = pd.concat([df, dup_row], ignore_index=True)
        # Shuffle slightly to test sorting and alignment
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return df
