"""Ingestion module for SMARD and DWD data."""

from chronogrid.ingestion.benchmark_generator import generate_german_energy_weather_dataset
from chronogrid.ingestion.dwd_client import DwdClient
from chronogrid.ingestion.ingest_pipeline import IngestionPipeline
from chronogrid.ingestion.smard_client import SmardClient

__all__ = [
    "SmardClient",
    "DwdClient",
    "generate_german_energy_weather_dataset",
    "IngestionPipeline",
]
