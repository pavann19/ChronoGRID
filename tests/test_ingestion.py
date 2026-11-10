"""
Tests for Data Ingestion (P5.1).
"""

from chronogrid.ingestion.benchmark_generator import generate_german_energy_weather_dataset
from chronogrid.ingestion.dwd_client import DwdClient
from chronogrid.ingestion.smard_client import SmardClient


def test_benchmark_generator_structure():
    df = generate_german_energy_weather_dataset(
        start_date="2024-01-01 00:00:00",
        end_date="2024-01-07 23:00:00",
        inject_minor_anomalies=False,
    )
    assert len(df) == 168  # 7 days * 24h
    required_cols = [
        "timestamp",
        "grid_load_mw",
        "temperature_c",
        "wind_speed_ms",
        "solar_radiation_wm2",
    ]
    for col in required_cols:
        assert col in df.columns
    assert df["grid_load_mw"].min() >= 25000.0
    assert df["grid_load_mw"].max() <= 90000.0


def test_smard_client_instance():
    client = SmardClient()
    assert client.FILTER_TOTAL_LOAD == 410


def test_dwd_client_instance():
    client = DwdClient()
    assert client.DEFAULT_LAT == 50.1109
