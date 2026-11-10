"""
Data schemas for the Data-Quality-Conditioned Conformal Prediction (DQ-CP) engine.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class UncertaintyResult(BaseModel):
    """Result of DQ-CP conformal interval estimation."""

    model_config = {"protected_namespaces": ()}

    forecast_mw: List[float] = Field(..., description="24-hour day-ahead point predictions in MW")
    lower_bound_mw: Optional[List[float]] = Field(
        None, description="24-hour lower prediction bounds in MW"
    )
    upper_bound_mw: Optional[List[float]] = Field(
        None, description="24-hour upper prediction bounds in MW"
    )
    target_coverage: float = Field(
        0.90, description="Nominal target coverage level (e.g. 0.90 or 0.95)"
    )
    calibration_method: str = Field(
        ..., description="Method used ('pooled_marginal_mad' or 'daily_sup_norm')"
    )
    model_id: str = Field(..., description="Model artifact identifier")
    calibration_version: str = Field(..., description="Calibration table version string")
    reliability_state: str = Field(
        ..., description="DDIR reliability state (HEALTHY, DEGRADED, STALE, UNAVAILABLE, INVALID)"
    )
    forecast_origin: Optional[str] = Field(
        None, description="ISO timestamp or date of forecast issuance cutoff"
    )
    uncertainty_available: bool = Field(
        True, description="Whether valid uncertainty bounds could be generated"
    )
    is_unbounded: bool = Field(
        False,
        description="Whether the required conformal quantile is infinite (unbounded interval)",
    )
    mean_interval_width_mw: Optional[float] = Field(
        None, description="Average interval width across the 24 hours"
    )
    current_alpha: float = Field(
        0.10, description="Active nominal miscoverage parameter alpha (may be adapted by ACI)"
    )
    degradation_flag: Optional[str] = Field(
        None, description="Diagnostic annotation if inputs are degraded"
    )


class CalibrationTable(BaseModel):
    """Serialized calibration parameters for a specific model head."""

    model_config = {"protected_namespaces": ()}

    version: str = Field(..., description="Semantic calibration version, e.g. 'v1.0-research'")
    created_at: str = Field(..., description="UTC ISO timestamp of calibration table generation")
    training_cutoff: str = Field(
        "2024-03-31 23:00:00+00:00", description="Upper boundary timestamp of training set"
    )
    model_id: str = Field(
        ..., description="Model identifier: research_primary or research_fallback"
    )
    model_sha256: str = Field(
        ..., description="SHA-256 fingerprint of the calibrated model artifact"
    )
    hour_mad: Dict[int, float] = Field(
        ..., description="Delivery-hour MAD scales in MW (keys 0..23)"
    )
    global_mad: float = Field(..., description="Global training residual MAD in MW")
    epsilon_guard: float = Field(
        1.0, description="Minimum allowable MAD scale to avoid division by zero"
    )
    marginal_quantiles: Dict[str, float] = Field(
        ..., description="Pooled normalized residual quantiles (keys '0.90', '0.95')"
    )
    simultaneous_quantiles: Dict[str, float] = Field(
        ..., description="Sup-norm daily vector quantiles (keys '0.90', '0.95')"
    )
    n_origins: int = Field(..., description="Number of daily forecast origins in calibration set")
    n_hourly_samples: int = Field(
        ..., description="Number of hourly observations in calibration set"
    )
    finite_sample_ranks: Dict[str, int] = Field(
        ..., description="Rank indices k used for each quantile"
    )
    marginal_scores: Optional[List[float]] = Field(
        None, description="Raw sorted marginal non-conformity scores"
    )
    daily_sup_norm_scores: Optional[List[float]] = Field(
        None, description="Raw sorted daily sup-norm scores"
    )


class DelayedFeedbackRecord(BaseModel):
    """Audit record for delayed-feedback Adaptive Conformal Inference (ACI)."""

    forecast_origin: str = Field(
        ..., description="Origin date/timestamp of the forecast being evaluated (e.g. '2024-04-02')"
    )
    settlement_timestamp: str = Field(
        ..., description="UTC timestamp when verified ground truth arrived"
    )
    effective_delay_hours: int = Field(48, description="Observed feedback delay in hours")
    miscoverage_occurred: bool = Field(
        ..., description="Whether any/majority of delivery hours suffered miscoverage"
    )
    empirical_coverage: float = Field(
        ..., description="Fraction of delivery hours covered by the interval (0.0 to 1.0)"
    )
    alpha_before: float = Field(..., description="Miscoverage parameter alpha before update")
    alpha_after: float = Field(..., description="Miscoverage parameter alpha after update")
    gamma: float = Field(0.01, description="ACI step size used for this update")
    status: str = Field(
        "PROCESSED",
        description="Audit status: 'PROCESSED', 'DUPLICATE_IGNORED', or 'REJECTED_REVISION'",
    )
