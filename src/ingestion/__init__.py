"""Ingestion module for SMARD and DWD data."""
from src.ingestion.smard_client import SmardClient
from src.ingestion.dwd_client import DwdClient
from src.ingestion.benchmark_generator import generate_german_energy_weather_dataset
from src.ingestion.ingest_pipeline import IngestionPipeline

__all__ = [
    "SmardClient",
    "DwdClient",
    "generate_german_energy_weather_dataset",
    "IngestionPipeline",
]

