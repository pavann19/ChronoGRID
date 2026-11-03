"""
Database ORM models for ChronoGrid.
Supports both PostgreSQL and SQLite.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text, Index
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ElectricityRecord(Base):
    """Hourly electricity and weather aligned record."""
    __tablename__ = "electricity_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, unique=True, index=True)
    
    # SMARD electricity metrics
    grid_load_mw = Column(Float, nullable=False)
    day_ahead_price_eur_mwh = Column(Float, nullable=True)
    solar_generation_mw = Column(Float, nullable=True)
    wind_generation_mw = Column(Float, nullable=True)
    
    # DWD weather metrics
    temperature_c = Column(Float, nullable=True)
    humidity_pct = Column(Float, nullable=True)
    wind_speed_ms = Column(Float, nullable=True)
    solar_radiation_wm2 = Column(Float, nullable=True)
    
    # Data Quality flags
    is_interpolated = Column(Boolean, default=False, nullable=False)
    quality_flag = Column(String(32), default="VALID", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_records_timestamp", "timestamp"),
    )


class PipelineRun(Base):
    """Tracks execution of data pipelines (P5.2)."""
    __tablename__ = "pipeline_runs"

    run_id = Column(String(64), primary_key=True)
    pipeline_name = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False)  # SUCCESS, FAILED, RUNNING
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    rows_ingested = Column(Integer, default=0)
    rows_validated = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)


class ModelRegistryRecord(Base):
    """Tracks model versions and deployment lifecycle stages (P5.7 & P5.10)."""
    __tablename__ = "model_registry"

    model_id = Column(String(64), primary_key=True)
    model_name = Column(String(64), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    model_type = Column(String(32), nullable=False)  # LightGBM, XGBoost, etc.
    stage = Column(String(32), default="Staging", nullable=False)  # Staging, Production, Archived
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    artifact_uri = Column(String(255), nullable=True)
    onnx_uri = Column(String(255), nullable=True)
    is_champion = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    promoted_at = Column(DateTime, nullable=True)
    metadata_json = Column(Text, nullable=True)


class DriftReportRecord(Base):
    """Stores historical drift monitoring evaluations (P5.9)."""
    __tablename__ = "drift_reports"

    report_id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    reference_start = Column(DateTime, nullable=False)
    reference_end = Column(DateTime, nullable=False)
    current_start = Column(DateTime, nullable=False)
    current_end = Column(DateTime, nullable=False)
    overall_drift_score = Column(Float, nullable=False)
    drift_detected = Column(Boolean, default=False, nullable=False)
    retraining_triggered = Column(Boolean, default=False, nullable=False)
    feature_metrics_json = Column(Text, nullable=False)
    evidently_report_path = Column(String(255), nullable=True)
