"""
Pydantic Request and Response Schemas for FastAPI Serving (P5.8).
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, FiniteFloat


class PredictRequest(BaseModel):
    """Payload containing feature inputs for inference."""

    features: List[Dict[str, FiniteFloat]] = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Complete finite model feature dictionaries; consult /model/info. No implicit padding.",
    )


class PredictResponse(BaseModel):
    """Inference response."""

    predictions: List[float]
    model_version: Optional[int]
    model_type: str
    inference_time_ms: float
    unit: str = "MW"
    evidence_kind: Literal["caller_supplied_features"] = "caller_supplied_features"

    model_config = {"protected_namespaces": ()}


class DayAheadForecastRequest(BaseModel):
    """Request for 24-hour horizon roll-forward forecast."""

    start_timestamp: Optional[str] = Field(
        None,
        description="Starting timestamp in ISO 8601 (e.g. 2024-06-01T00:00:00Z). If omitted, uses latest available.",
    )
    horizon_hours: int = Field(24, ge=1, le=168)


class DayAheadForecastResponse(BaseModel):
    """Day-ahead hourly forecast sequence."""

    timestamps: List[str]
    forecast_load_mw: List[float]
    model_version: Optional[int]
    model_type: str
    generated_at: str

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    service: str
    onnx_loaded: bool
    version: str = "2.0-research-diagnostics"


class ResilientForecastRequest(BaseModel):
    """Request for DDIR-protected resilient day-ahead forecasting."""

    start_timestamp: Optional[str] = Field(
        None,
        description="Starting timestamp in ISO 8601 (e.g. 2024-06-01T00:00:00Z). If omitted, uses latest available.",
    )
    horizon_hours: int = Field(24, ge=1, le=168)
    weather_payload: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="External weather telemetry to validate and route. If omitted or null, gracefully routes to Fallback Load-Only head.",
    )


class ResilientForecastResponse(BaseModel):
    """Resilient forecast response with DDIR audit provenance."""

    timestamps: List[str]
    forecast_load_mw: List[float]
    routing_state: str
    active_head: str
    fallback_reason: Optional[str]
    weather_quality_score: float
    weather_freshness_seconds: Optional[float]
    inference_time_ms: float
    validation_time_ms: float
    total_time_ms: float
    features_used_count: int
    model_version: Optional[int]
    model_type: str
    generated_at: str

    model_config = {"protected_namespaces": ()}


class DQCPForecastRequest(BaseModel):
    """Request payload for research-only DQ-CP conformal forecast."""

    start_timestamp: Optional[str] = Field(
        None, description="Start timestamp in ISO 8601 UTC format"
    )
    weather_payload: Optional[List[Dict[str, Any]]] = Field(
        None, description="24-hour weather observation payload"
    )
    target_coverage: float = Field(0.90, description="Desired coverage level: 0.90 or 0.95")
    calibration_method: str = Field(
        "pooled_marginal_mad", description="'pooled_marginal_mad' or 'daily_sup_norm'"
    )
    forecast_origin: Optional[str] = Field(
        None, description="Date/timestamp identifier for forecast origin"
    )

    model_config = {"protected_namespaces": ()}


class DQCPForecastResponse(BaseModel):
    """Response payload for research-only DQ-CP conformal forecast."""

    timestamps: List[str] = Field(..., description="Hourly delivery timestamps")
    forecast_mw: List[float] = Field(..., description="24-hour point predictions in MW")
    lower_bound_mw: Optional[List[float]] = Field(None, description="Lower prediction bounds in MW")
    upper_bound_mw: Optional[List[float]] = Field(None, description="Upper prediction bounds in MW")
    target_coverage: float = Field(..., description="Target coverage level")
    calibration_method: str = Field(..., description="Calibration method applied")
    model_id: str = Field(..., description="Active model head identifier")
    calibration_version: str = Field(..., description="Calibration table version")
    reliability_state: str = Field(..., description="DDIR reliability state")
    forecast_origin: Optional[str] = Field(None, description="Origin identifier")
    uncertainty_available: bool = Field(
        ..., description="Whether calibrated intervals could be produced"
    )
    is_unbounded: bool = Field(
        False, description="Whether prediction interval is infinite (unbounded)"
    )
    mean_interval_width_mw: Optional[float] = Field(
        None, description="Mean interval width across 24h"
    )
    current_alpha: float = Field(..., description="Active miscoverage parameter")
    degradation_flag: Optional[str] = Field(None, description="Degradation diagnostic message")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    total_time_ms: float = Field(..., description="Total execution time in milliseconds")
    generated_at: str = Field(..., description="UTC ISO generation timestamp")

    model_config = {"protected_namespaces": ()}


class DQCPFeedbackRequest(BaseModel):
    """Request payload for delayed-feedback settlement submission."""

    forecast_origin: str = Field(..., description="Forecast origin date (e.g. '2024-04-02')")
    actual_load_mw: List[FiniteFloat] = Field(
        ...,
        min_length=24,
        max_length=24,
        description="Legacy 24-hour replay actuals in MW, not independently verified source records",
    )
    settlement_timestamp: str = Field(..., description="UTC timestamp of settlement publication")
    effective_delay_days: int = Field(
        2, description="Observed settlement delay in days (must be >= 2)"
    )
    head: Optional[Literal["primary", "fallback"]] = Field(
        None, description="Optional target head ('primary' or 'fallback')"
    )

    model_config = {"protected_namespaces": ()}


class DQCPFeedbackResponse(BaseModel):
    """Response payload for delayed-feedback settlement submission."""

    forecast_origin: str
    settlement_timestamp: str
    effective_delay_hours: int
    miscoverage_occurred: bool
    empirical_coverage: float
    alpha_before: float
    alpha_after: float
    gamma: float
    status: str

    model_config = {"protected_namespaces": ()}
