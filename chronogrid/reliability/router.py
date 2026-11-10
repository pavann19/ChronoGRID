"""
Dual-Head Inference Router for ChronoGrid DDIR.
Coordinates validation, reliability state transitions, and deterministic execution
across Primary (Weather-Enabled) and Fallback (Load-Only) ONNX engines.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from chronogrid.serving.onnx_engine import OnnxInferenceEngine
from chronogrid.reliability.state_machine import ReliabilityState, ReliabilityStateMachine
from chronogrid.reliability.validator import WeatherInputValidator, ValidationReport
from chronogrid.reliability.recovery import HysteresisRecoveryManager

logger = logging.getLogger(__name__)


_UNSPECIFIED = object()


@dataclass
class DDIRInferenceResult:
    """Rich auditable result returned by DualHeadInferenceRouter."""
    predictions: List[float]
    routing_state: ReliabilityState
    active_head: str  # "PRIMARY_WEATHER" or "FALLBACK_LOAD_ONLY"
    fallback_reason: Optional[str]
    weather_quality_score: float
    weather_freshness_seconds: Optional[float]
    consecutive_healthy_count: int
    inference_time_ms: float
    validation_time_ms: float
    total_time_ms: float
    features_used_count: int
    validation_report: Optional[ValidationReport] = None
    audit_metadata: Dict[str, Any] = field(default_factory=dict)


class DualHeadInferenceRouter:
    """
    Sub-millisecond Dual-Head Inference Router.
    Validates weather inputs and routes to Primary ONNX (43 features) or Fallback ONNX (34 features).
    """

    def __init__(
        self,
        primary_onnx_path: str = "artifacts/models/champion_model.onnx",
        primary_metadata_path: str = "artifacts/models/champion_metadata.json",
        fallback_onnx_path: str = "artifacts/models/load_only_model.onnx",
        fallback_metadata_path: str = "artifacts/models/load_only_metadata.json",
        recovery_threshold: int = 2,
        max_freshness_hours: float = 28.0,
        state_file_path: Optional[str] = None,
    ):
        self.primary_onnx_path = Path(primary_onnx_path)
        self.primary_metadata_path = Path(primary_metadata_path)
        self.fallback_onnx_path = Path(fallback_onnx_path)
        self.fallback_metadata_path = Path(fallback_metadata_path)
        self.state_file_path = state_file_path

        # Load ONNX engines
        self.primary_engine = OnnxInferenceEngine(
            onnx_model_path=str(self.primary_onnx_path),
            metadata_path=str(self.primary_metadata_path),
        )
        self.fallback_engine = OnnxInferenceEngine(
            onnx_model_path=str(self.fallback_onnx_path),
            metadata_path=str(self.fallback_metadata_path),
        )

        self.validator = WeatherInputValidator(max_freshness_hours=max_freshness_hours)
        self.recovery_manager = HysteresisRecoveryManager(
            recovery_threshold=recovery_threshold,
            state_file_path=state_file_path,
        )

        logger.info(
            f"DualHeadInferenceRouter initialized. Primary Features: {len(self.primary_engine.feature_names)}, "
            f"Fallback Features: {len(self.fallback_engine.feature_names)}, Recovery Threshold: {recovery_threshold}"
        )

    def route_and_predict(
        self,
        features: Union[pd.DataFrame, List[Dict[str, Any]], np.ndarray],
        weather_payload: Any = _UNSPECIFIED,
        data_timestamp: Optional[Union[str, datetime]] = None,
        current_time: Optional[datetime] = None,
    ) -> DDIRInferenceResult:
        """
        Validates weather telemetry, evaluates reliability state, and executes dual-head prediction.
        """
        t_start = time.perf_counter()

        # 1. Validation Step
        t_val_start = time.perf_counter()
        
        # If weather_payload is unspecified, attempt to extract it from features
        if weather_payload is _UNSPECIFIED:
            if isinstance(features, pd.DataFrame):
                # Check for direct weather columns in DataFrame
                weather_cols = [c for c in ["temperature_c", "wind_speed_ms", "solar_radiation_wm2", "humidity_pct"] if c in features.columns]
                if weather_cols:
                    weather_payload = features[weather_cols].copy()
                else:
                    # Check for forecast_* weather columns
                    alias_map = {
                        "forecast_temperature_c": "temperature_c",
                        "forecast_wind_speed_ms": "wind_speed_ms",
                        "forecast_solar_radiation_wm2": "solar_radiation_wm2",
                        "forecast_humidity_pct": "humidity_pct"
                    }
                    present_aliases = [c for c in alias_map if c in features.columns]
                    if present_aliases:
                        weather_payload = features[present_aliases].rename(columns=alias_map).copy()
                    else:
                        weather_payload = None
            else:
                weather_payload = None

        try:
            report = self.validator.validate(
                weather_data=weather_payload,
                current_time=current_time,
                data_timestamp=data_timestamp,
            )
            raw_state, raw_reason = ReliabilityStateMachine.evaluate(report)
        except Exception as exc:
            logger.warning(f"Validator/State machine error: {exc}. Executing safe fallback to load-only model.")
            report = ValidationReport(
                is_valid=False,
                quality_score=0.0,
                violations=[f"Internal validation error: {str(exc)}"],
                missing_rate=1.0,
            )
            raw_state = ReliabilityState.INVALID
            raw_reason = f"Internal validation error: {str(exc)}"

        t_val_end = time.perf_counter()
        validation_time_ms = (t_val_end - t_val_start) * 1000.0

        # 2. State Machine & Hysteresis Recovery Evaluation
        # Compute source timestamp, raw payload fingerprint, and telemetry content fingerprint
        src_ts = data_timestamp
        if src_ts is None and isinstance(weather_payload, dict) and "timestamp" in weather_payload:
            src_ts = weather_payload["timestamp"]
        elif src_ts is None and isinstance(weather_payload, list) and len(weather_payload) > 0 and isinstance(weather_payload[0], dict):
            src_ts = weather_payload[0].get("timestamp")

        fingerprint = None
        content_fingerprint = None
        if weather_payload is not None:
            try:
                import hashlib
                fingerprint = hashlib.sha256(str(weather_payload).encode()).hexdigest()[:16]

                # Content fingerprint: hash telemetry readings excluding timestamp / date headers
                content_obj = None
                if isinstance(weather_payload, dict):
                    content_obj = {k: v for k, v in sorted(weather_payload.items()) if k.lower() not in ("timestamp", "time", "datetime", "date")}
                elif isinstance(weather_payload, list) and len(weather_payload) > 0 and isinstance(weather_payload[0], dict):
                    content_obj = [{k: v for k, v in sorted(r.items()) if k.lower() not in ("timestamp", "time", "datetime", "date")} for r in weather_payload]
                elif isinstance(weather_payload, pd.DataFrame):
                    cols = [c for c in weather_payload.columns if c.lower() not in ("timestamp", "time", "datetime", "date")]
                    content_obj = weather_payload[cols].to_dict(orient="records")
                else:
                    content_obj = weather_payload

                if content_obj is not None:
                    content_fingerprint = hashlib.sha256(str(content_obj).encode()).hexdigest()[:16]
            except Exception:
                fingerprint = None
                content_fingerprint = None

        effective_state, active_head, final_reason = self.recovery_manager.process_state(
            raw_state=raw_state,
            raw_reason=raw_reason,
            source_timestamp=src_ts,
            payload_fingerprint=fingerprint,
            content_fingerprint=content_fingerprint,
        )

        # 3. Model Execution
        t_inf_start = time.perf_counter()
        fallback_reason = None

        if active_head == "PRIMARY_WEATHER":
            # Primary Head requires 43 features
            try:
                preds = self._predict_engine(self.primary_engine, features)
                features_used_count = len(self.primary_engine.feature_names)
            except Exception as e:
                logger.warning(f"Primary ONNX head encountered inference error: {e}. Executing immediate fallback.")
                # Immediate fail-safe to Fallback Head
                preds = self._predict_engine(self.fallback_engine, features)
                active_head = "FALLBACK_LOAD_ONLY"
                effective_state = ReliabilityState.DEGRADED
                fallback_reason = f"Primary head runtime error: {str(e)}"
                features_used_count = len(self.fallback_engine.feature_names)
        else:
            # Fallback Head requires 34 features
            preds = self._predict_engine(self.fallback_engine, features)
            fallback_reason = final_reason
            features_used_count = len(self.fallback_engine.feature_names)

        t_inf_end = time.perf_counter()
        inference_time_ms = (t_inf_end - t_inf_start) * 1000.0
        total_time_ms = (t_inf_end - t_start) * 1000.0

        freshness_secs = (report.freshness_age_hours * 3600.0) if report.freshness_age_hours is not None else None
        recovery_status = self.recovery_manager.get_status()

        return DDIRInferenceResult(
            predictions=[round(float(p), 1) for p in preds],
            routing_state=effective_state,
            active_head=active_head,
            fallback_reason=fallback_reason,
            weather_quality_score=report.quality_score,
            weather_freshness_seconds=freshness_secs,
            consecutive_healthy_count=recovery_status["consecutive_healthy_count"],
            inference_time_ms=round(inference_time_ms, 3),
            validation_time_ms=round(validation_time_ms, 3),
            total_time_ms=round(total_time_ms, 3),
            features_used_count=features_used_count,
            validation_report=report,
            audit_metadata={
                "raw_state": raw_state.value,
                "recovery_threshold": recovery_status["recovery_threshold"],
                "violations": report.violations,
                "missing_rate": report.missing_rate,
            }
        )

    def _predict_engine(self, engine: OnnxInferenceEngine, features: Union[pd.DataFrame, List[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        """Extracts required schema subset for target engine and runs prediction."""
        if isinstance(features, pd.DataFrame):
            if features.columns.duplicated().any():
                raise ValueError('Duplicate feature columns are not allowed.')
            missing_cols=[c for c in engine.feature_names if c not in features.columns]
            if missing_cols:
                raise ValueError(f'Missing required model features: {missing_cols}')
            X=features[engine.feature_names].values.astype(np.float32)
            return engine.predict(X)

        elif isinstance(features, list):
            # List of dicts
            missing=[name for row in features for name in engine.feature_names if name not in row]
            if missing:raise ValueError(f'Missing required model features: {sorted(set(missing))}')
            return engine.predict([{name:row[name] for name in engine.feature_names} for row in features])
        else:
            # Direct numpy array
            X = np.asarray(features, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.ndim!=2:raise ValueError('Expected a one- or two-dimensional feature array.')
            if engine is self.fallback_engine and X.shape[1]==len(self.primary_engine.feature_names):
                indices=[self.primary_engine.feature_names.index(name) for name in engine.feature_names]
                X=X[:,indices]
            return engine.predict(X)
