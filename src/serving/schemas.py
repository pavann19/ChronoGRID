"""
Pydantic Request and Response Schemas for FastAPI Serving (P5.8).
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Payload containing feature inputs for inference."""
    features: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries or a single observation.",
        example=[{
            "temperature_c": 14.5,
            "humidity_pct": 68.0,
            "wind_speed_ms": 4.2,
            "solar_radiation_wm2": 320.0,
            "sin_hour": 0.5,
            "cos_hour": 0.866,
            "grid_load_mw_lag_24": 58400.0,
            "grid_load_mw_lag_168": 59100.0
        }]
    )


class PredictResponse(BaseModel):
    """Inference response."""
    predictions: List[float]
    model_version: int
    model_type: str
    inference_time_ms: float
    unit: str = "MW"

    model_config = {"protected_namespaces": ()}


class DayAheadForecastRequest(BaseModel):
    """Request for 24-hour horizon roll-forward forecast."""
    start_timestamp: Optional[str] = Field(
        None,
        description="Starting timestamp in ISO 8601 (e.g. 2024-06-01T00:00:00Z). If omitted, uses latest available."
    )
    horizon_hours: int = Field(24, ge=1, le=168)


class DayAheadForecastResponse(BaseModel):
    """Day-ahead hourly forecast sequence."""
    timestamps: List[str]
    forecast_load_mw: List[float]
    model_version: int
    model_type: str
    generated_at: str

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    service: str
    onnx_loaded: bool
    version: str = "1.0.0"
