"""
FastAPI Model Serving Application (P5.8 & P5.11).
Ultra-low latency inference via ONNX Runtime with Prometheus metrics integration.
"""

import os
import secrets
import hashlib
import json
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

from src.serving.schemas import (
    PredictRequest,
    PredictResponse,
    DayAheadForecastRequest,
    DayAheadForecastResponse,
    HealthResponse,
    ResilientForecastRequest,
    ResilientForecastResponse,
    DQCPForecastRequest,
    DQCPForecastResponse,
    DQCPFeedbackRequest,
    DQCPFeedbackResponse,
)
from src.serving.onnx_engine import OnnxInferenceEngine
from src.db.database import get_session, init_db
from src.db.models import ModelRegistryRecord
from src.uncertainty import load_dqcp_engine, DQCPUncertaintyEngine

logger = logging.getLogger(__name__)

# Prometheus Metrics Definitions (P5.11)
REQUEST_COUNT = Counter(
    "chronogrid_inference_requests_total",
    "Total number of inference requests served",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "chronogrid_inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_LOAD = Histogram(
    "chronogrid_predicted_load_mw",
    "Distribution of predicted grid load in MW",
    buckets=[30000, 40000, 50000, 60000, 70000, 80000, 90000]
)
CHAMPION_VERSION = Gauge(
    "chronogrid_champion_model_version",
    "Version number of the current active champion model"
)
DDIR_REQUEST_COUNT = Counter(
    "chronogrid_ddir_requests_total",
    "Total number of DDIR resilient inference requests",
    ["state", "head"]
)
DDIR_QUALITY_GAUGE = Gauge(
    "chronogrid_ddir_weather_quality_score",
    "Current weather telemetry quality score (0.0 to 1.0)"
)

# DQ-CP Research Uncertainty Metrics
DQCP_CALIBRATION_POOL_SIZE = Gauge(
    "chronogrid_dqcp_calibration_pool_size",
    "Number of calibration samples in the active pool",
    ["head"]
)
DQCP_INTERVAL_WIDTH = Histogram(
    "chronogrid_dqcp_interval_width_mw",
    "Distribution of prediction interval widths in MW",
    buckets=[500, 1000, 2000, 3000, 4000, 5000, 7500, 10000]
)
DQCP_ACI_ALPHA = Gauge(
    "chronogrid_dqcp_aci_alpha",
    "Current active ACI miscoverage parameter alpha"
)
DQCP_FEEDBACK_COUNT = Counter(
    "chronogrid_dqcp_coverage_feedback_total",
    "Total ground-truth feedback settlements received",
    ["status"]
)
DQCP_FALLBACK_INTERVAL_COUNT = Counter(
    "chronogrid_dqcp_fallback_interval_total",
    "Total intervals generated using fallback calibration pool"
)
DQCP_UNBOUNDED_INTERVAL_COUNT = Counter(
    "chronogrid_dqcp_unbounded_interval_total",
    "Total unbounded prediction intervals generated when k exceeds pool size"
)
DQCP_CALIBRATION_VERSION_INFO = Gauge(
    "chronogrid_dqcp_calibration_version_info",
    "Version indicator of the active DQ-CP calibration table",
    ["version"]
)

# Global engine and router instances
ENGINE: Optional[OnnxInferenceEngine] = None
ROUTER: Optional[Any] = None
DQCP_ENGINE: Optional[DQCPUncertaintyEngine] = None
CHAMPION_META = {"version": None, "model_type": "Unknown", "promoted_at": None, "registry_identity_verified": False}


def load_engine():
    """Initializes ONNX engine, DDIR router, and registry metadata."""
    global ENGINE, ROUTER, CHAMPION_META, DQCP_ENGINE
    CHAMPION_META = {"version": None, "model_type": "Unknown", "promoted_at": None, "registry_identity_verified": False}
    CHAMPION_VERSION.set(0) # zero is the unverified registry-version sentinel, not a model release
    onnx_file = Path("artifacts/models/champion_model.onnx")
    meta_file = Path("artifacts/models/champion_metadata.json")
    fallback_onnx = Path("artifacts/models/load_only_model.onnx")
    fallback_meta = Path("artifacts/models/load_only_metadata.json")

    if onnx_file.exists():
        ENGINE = OnnxInferenceEngine(
            onnx_model_path=str(onnx_file),
            metadata_path=str(meta_file),
        )
        actual_sha = hashlib.sha256(onnx_file.read_bytes()).hexdigest()
        CHAMPION_META = {"version": None, "model_type": ENGINE.metadata.get("model_type", "Unknown"), "promoted_at": None, "artifact_sha256": actual_sha, "registry_identity_verified": False}
        if fallback_onnx.exists():
            try:
                from src.reliability.router import DualHeadInferenceRouter
                state_file = os.environ.get("DDIR_STATE_FILE", "data/ddir_runtime_state.json")
                ROUTER = DualHeadInferenceRouter(
                    primary_onnx_path=str(onnx_file),
                    primary_metadata_path=str(meta_file),
                    fallback_onnx_path=str(fallback_onnx),
                    fallback_metadata_path=str(fallback_meta),
                    recovery_threshold=2,
                    state_file_path=state_file,
                )
            except Exception:
                ROUTER = None
        else:
            ROUTER = None
        # Fetch version from DB
        try:
            with get_session() as session:
                rec = session.query(ModelRegistryRecord).filter_by(is_champion=True).first()
                if rec:
                    CHAMPION_META["registry_record"] = {"version": rec.version, "model_type": rec.model_type, "onnx_uri": rec.onnx_uri}
                    registry_metadata = json.loads(rec.metadata_json or "{}")
                    if registry_metadata.get("onnx_sha256") == actual_sha:
                        CHAMPION_META.update(version=rec.version, promoted_at=str(rec.promoted_at), registry_identity_verified=True)
                        CHAMPION_VERSION.set(rec.version)
        except Exception:
            pass
    else:
        ENGINE = None

    # Load DQ-CP research uncertainty engine
    try:
        DQCP_ENGINE = load_dqcp_engine(
            aci_state_file=Path(os.environ.get("DQCP_STATE_FILE", "data/dqcp_aci_state.json")),
            model_paths={"primary": onnx_file, "fallback": fallback_onnx},
        )
        if DQCP_ENGINE:
            try:
                n_prim = DQCP_ENGINE.primary_pool.n_origins if DQCP_ENGINE.primary_pool else 0
                n_fall = DQCP_ENGINE.fallback_pool.n_origins if DQCP_ENGINE.fallback_pool else 0
                DQCP_CALIBRATION_POOL_SIZE.labels(head="primary").set(n_prim)
                DQCP_CALIBRATION_POOL_SIZE.labels(head="fallback").set(n_fall)
                DQCP_ACI_ALPHA.set(DQCP_ENGINE.aci_manager.get_current_alpha())
                DQCP_CALIBRATION_VERSION_INFO.labels(version=DQCP_ENGINE.version).set(1.0)
            except Exception:
                pass
    except Exception as exc:
        logger.warning(f"Could not initialize DQ-CP engine: {exc}")
        DQCP_ENGINE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    load_engine()
    yield
    # Shutdown


app = FastAPI(
    title="ChronoGrid Legacy Model Diagnostics",
    description="Caller-feature ONNX inference and explicitly authorized research replay/simulation. Prospective evidence uses the registered operations worker.",
    version="2.0-research-diagnostics",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CHRONOGRID_CORS_ORIGINS", "http://127.0.0.1:8765,http://localhost:8765").split(",") if origin.strip() and origin.strip()!="*"],
    allow_credentials=False,
    allow_methods=["GET","POST"],
    allow_headers=["Authorization","Content-Type"],
)


LEGACY_REPLAY_ROUTES = {"/forecast/day-ahead", "/forecast/resilient", "/forecast/dqcp", "/forecast/dqcp/feedback"}

@app.middleware("http")
async def research_boundary(request: Request, call_next):
    # Protect future diagnostic routes by default; probes reveal only service readiness.
    if request.method != "OPTIONS" and request.url.path not in {"/health", "/ready"} and request.url.path not in LEGACY_REPLAY_ROUTES:
        expected = os.getenv("CHRONOGRID_DIAGNOSTICS_TOKEN", "")
        if not expected.strip():
            return JSONResponse(status_code=503, content={"detail": "Diagnostic authentication is not configured."})
        supplied = request.headers.get("Authorization", "")
        if not secrets.compare_digest(supplied.encode("utf-8"), ("Bearer " + expected).encode("utf-8")):
            return JSONResponse(status_code=401, content={"detail": "Diagnostic authorization required."}, headers={"WWW-Authenticate": "Bearer"})
    legacy = request.url.path in LEGACY_REPLAY_ROUTES and request.method == "POST"
    if legacy:
        if os.getenv("CHRONOGRID_ENABLE_LEGACY_REPLAY") != "1":
            return JSONResponse(status_code=410, content={"detail":"Legacy simulated/replay forecasting is retired by default. Use the registered prospective capture worker."})
        expected = os.getenv("CHRONOGRID_LEGACY_REPLAY_TOKEN", "")
        if not expected:
            return JSONResponse(status_code=503, content={"detail":"Legacy replay requires an explicitly configured authorization token."})
        supplied = request.headers.get("Authorization", "")
        if not secrets.compare_digest(supplied.encode("utf-8"), ("Bearer " + expected).encode("utf-8")):
            return JSONResponse(status_code=401, content={"detail":"Replay authorization required."},headers={"WWW-Authenticate":"Bearer"})
    response = await call_next(request)
    if legacy:
        response.headers["X-ChronoGrid-Evidence-Kind"] = "legacy_replay_or_synthetic"
        response.headers["X-ChronoGrid-Research-Only"] = "true"
        if request.url.path.startswith("/forecast/dqcp"):
            response.headers["X-ChronoGrid-Calibration-Compatibility"] = "not-validated"
    elif request.url.path == "/predict":
        response.headers["X-ChronoGrid-Evidence-Kind"] = "caller_supplied_features"
    return response


@app.middleware("http")
async def diagnostic_response_headers(request: Request, call_next):
    # Registered last so authentication rejections receive these headers too.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe."""
    return HealthResponse(
        status="healthy",
        service="ChronoGrid Inference Engine",
        onnx_loaded=ENGINE is not None,
    )


@app.get("/ready")
def ready():
    """Readiness probe."""
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="ONNX model is not loaded yet.")
    return {"ready": True, "scope": "ONNX engine loaded; not prospective forecast validity or production readiness"}


@app.get("/model/info")
def model_info():
    """Returns active champion model information."""
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model engine not loaded.")
    return {
        "champion_metadata": CHAMPION_META,
        "n_features": len(ENGINE.feature_names),
        "feature_names": ENGINE.feature_names,
        "input_name": ENGINE.input_name,
        "output_name": ENGINE.output_name,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, response: Response):
    """
    Sub-millisecond batch or single electricity load inference.
    """
    start_t = time.perf_counter()
    if ENGINE is None:
        load_engine()
        if ENGINE is None:
            REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(status_code=503, detail="ONNX model not loaded.")

    try:
        preds = ENGINE.predict(payload.features)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        latency_s = latency_ms / 1000.0

        # Metrics recording
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency_s)
        for p in preds:
            PREDICTION_LOAD.observe(float(p))

        # Response headers
        response.headers["X-Inference-Time-Ms"] = f"{latency_ms:.2f}"
        response.headers["X-Model-Version"] = str(CHAMPION_META["version"]) if CHAMPION_META["version"] is not None else "unverified"

        return PredictResponse(
            predictions=[round(float(p), 1) for p in preds],
            model_version=CHAMPION_META["version"],
            model_type=CHAMPION_META["model_type"],
            inference_time_ms=round(latency_ms, 2),
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/forecast/day-ahead", response_model=DayAheadForecastResponse)
def forecast_day_ahead(payload: DayAheadForecastRequest):
    """
    Simulates operational 24-hour day-ahead horizon rolling forecast.
    """
    if ENGINE is None:
        load_engine()
        if ENGINE is None:
            raise HTTPException(status_code=503, detail="ONNX model not loaded.")

    horizon = payload.horizon_hours
    # Read processed canonical parquet to obtain latest state if start_timestamp not given
    try:
        df = pd.read_parquet("data/processed/smard_dwd_hourly.parquet")
    except Exception:
        raise HTTPException(status_code=500, detail="Processed dataset not available for day-ahead forecast.")

    start_ts = pd.to_datetime(payload.start_timestamp, utc=True) if payload.start_timestamp else df["timestamp"].max()
    future_timestamps = pd.date_range(start=start_ts + pd.Timedelta(hours=1), periods=horizon, freq="1h", tz="UTC")

    from src.features.engineer import FeatureEngineer
    fe = FeatureEngineer(min_lag_hours=24)
    fe.feature_cols = ENGINE.feature_names
    fe.fitted = True

    # Build simulated future frame using recent history for lags (at least 168h + 24h lag depth)
    recent_history = df.tail(336).copy()
    future_rows = []
    
    last_known_load = recent_history["grid_load_mw"].iloc[-1]
    
    for i, ts in enumerate(future_timestamps):
        # Look up or simulate weather for ts
        simulated_temp = 15.0 + 5.0 * np.cos(2 * np.pi * (ts.hour - 15) / 24)
        simulated_solar = max(0.0, 700.0 * np.sin(np.pi * (ts.hour - 5) / 14)) if 5 <= ts.hour <= 19 else 0.0
        future_rows.append({
            "timestamp": ts,
            "grid_load_mw": last_known_load,  # Placeholder, will use lag
            "temperature_c": simulated_temp,
            "solar_radiation_wm2": simulated_solar,
            "wind_speed_ms": 4.5,
            "humidity_pct": 65.0,
        })

    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([recent_history, future_df], ignore_index=True)
    combined_feats = fe._generate_raw_features(combined)
    eval_feats = combined_feats.iloc[-horizon:][ENGINE.feature_names]

    preds = ENGINE.predict(eval_feats.values)

    return DayAheadForecastResponse(
        timestamps=[ts.isoformat() for ts in future_timestamps],
        forecast_load_mw=[round(float(p), 1) for p in preds],
        model_version=CHAMPION_META["version"],
        model_type=CHAMPION_META["model_type"],
        generated_at=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    )


@app.post("/forecast/resilient", response_model=ResilientForecastResponse)
def forecast_resilient(payload: ResilientForecastRequest):
    """
    DDIR-protected resilient day-ahead rolling forecast.
    Inspects weather telemetry and dynamically routes to Primary or Fallback ONNX head.
    """
    global ROUTER
    if ROUTER is None:
        load_engine()
        if ROUTER is None:
            raise HTTPException(status_code=503, detail="DDIR dual-head router is not loaded.")

    horizon = payload.horizon_hours
    try:
        df = pd.read_parquet("data/processed/smard_dwd_hourly.parquet")
    except Exception:
        raise HTTPException(status_code=500, detail="Processed dataset not available for day-ahead forecast.")

    start_ts = pd.to_datetime(payload.start_timestamp, utc=True) if payload.start_timestamp else df["timestamp"].max()
    future_timestamps = pd.date_range(start=start_ts + pd.Timedelta(hours=1), periods=horizon, freq="1h", tz="UTC")

    from src.features.engineer import FeatureEngineer
    fe = FeatureEngineer(min_lag_hours=24)
    fe.feature_cols = ROUTER.primary_engine.feature_names
    fe.fitted = True

    recent_history = df.tail(336).copy()
    last_known_load = recent_history["grid_load_mw"].iloc[-1]

    future_rows = []
    # If weather_payload is provided, match to future_timestamps
    weather_by_ts = {}
    if payload.weather_payload:
        for w in payload.weather_payload:
            if "timestamp" in w:
                try:
                    w_ts = pd.to_datetime(w["timestamp"], utc=True)
                    weather_by_ts[w_ts] = w
                except Exception:
                    pass

    for i, ts in enumerate(future_timestamps):
        w_dict = weather_by_ts.get(ts, {})
        future_rows.append({
            "timestamp": ts,
            "grid_load_mw": last_known_load,
            "temperature_c": w_dict.get("temperature_c", 15.0),
            "solar_radiation_wm2": w_dict.get("solar_radiation_wm2", 0.0),
            "wind_speed_ms": w_dict.get("wind_speed_ms", 4.0),
            "humidity_pct": w_dict.get("humidity_pct", 65.0),
        })

    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([recent_history, future_df], ignore_index=True)
    combined_feats = fe._generate_raw_features(combined)
    eval_feats = combined_feats.iloc[-horizon:].copy()

    # Route and predict through DDIR
    res = ROUTER.route_and_predict(
        features=eval_feats,
        weather_payload=payload.weather_payload,
        current_time=start_ts.to_pydatetime() if hasattr(start_ts, "to_pydatetime") else None,
    )

    # Record Prometheus metrics safely (shielded against observability failures)
    try:
        DDIR_REQUEST_COUNT.labels(state=res.routing_state.value, head=res.active_head).inc()
        DDIR_QUALITY_GAUGE.set(res.weather_quality_score)
    except Exception as exc:
        logger.warning(f"Failed to record Prometheus metrics in forecast_resilient: {exc}")

    return ResilientForecastResponse(
        timestamps=[ts.isoformat() for ts in future_timestamps],
        forecast_load_mw=res.predictions,
        routing_state=res.routing_state.value,
        active_head=res.active_head,
        fallback_reason=res.fallback_reason,
        weather_quality_score=res.weather_quality_score,
        weather_freshness_seconds=res.weather_freshness_seconds,
        inference_time_ms=res.inference_time_ms,
        validation_time_ms=res.validation_time_ms,
        total_time_ms=res.total_time_ms,
        features_used_count=res.features_used_count,
        model_version=CHAMPION_META["version"],
        model_type=CHAMPION_META["model_type"],
        generated_at=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    )


@app.post("/forecast/dqcp", response_model=DQCPForecastResponse)
def forecast_dqcp(payload: DQCPForecastRequest):
    """
    Research-only endpoint generating Data-Quality-Conditioned Conformal Prediction (DQ-CP) intervals.
    Conditioned dynamically on DDIR reliability states without altering the frozen v1.0.0 champion.
    """
    t_start = time.perf_counter()
    if ROUTER is None:
        raise HTTPException(status_code=503, detail="DDIR dual-head router is not initialized.")
    if DQCP_ENGINE is None:
        raise HTTPException(status_code=503, detail="DQ-CP Uncertainty Engine is not initialized.")

    raw_path = Path("data/raw/genuine_smard_dwd_observations.parquet")
    if not raw_path.exists():
        raise HTTPException(status_code=500, detail="Historical reference dataset missing.")

    df = pd.read_parquet(raw_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    horizon = 24
    if payload.start_timestamp:
        try:
            start_ts = pd.to_datetime(payload.start_timestamp, utc=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid start_timestamp: {exc}")
    else:
        start_ts = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)

    future_timestamps = [start_ts + pd.Timedelta(hours=i) for i in range(horizon)]

    from src.features.engineer import FeatureEngineer
    fe = FeatureEngineer(min_lag_hours=24, weather_mode="persistence")
    recent_history = df.tail(336).copy()
    last_known_load = recent_history["grid_load_mw"].iloc[-1]

    future_rows = []
    weather_by_ts = {}
    if payload.weather_payload:
        for w in payload.weather_payload:
            if "timestamp" in w:
                try:
                    w_ts = pd.to_datetime(w["timestamp"], utc=True)
                    weather_by_ts[w_ts] = w
                except Exception:
                    pass

    for ts in future_timestamps:
        w_dict = weather_by_ts.get(ts, {})
        future_rows.append({
            "timestamp": ts,
            "grid_load_mw": last_known_load,
            "temperature_c": w_dict.get("temperature_c", 15.0),
            "solar_radiation_wm2": w_dict.get("solar_radiation_wm2", 0.0),
            "wind_speed_ms": w_dict.get("wind_speed_ms", 4.0),
            "humidity_pct": w_dict.get("humidity_pct", 65.0),
        })

    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([recent_history, future_df], ignore_index=True)
    combined_feats = fe._generate_raw_features(combined)
    eval_feats = combined_feats.iloc[-horizon:].copy()

    # Route point forecast through DDIR
    res = ROUTER.route_and_predict(
        features=eval_feats,
        weather_payload=payload.weather_payload,
        current_time=start_ts.to_pydatetime() if hasattr(start_ts, "to_pydatetime") else None,
    )

    delivery_hours = np.array([ts.hour for ts in future_timestamps], dtype=np.int64)
    forecast_origin = payload.forecast_origin or future_timestamps[0].date().isoformat()

    # Generate DQ-CP uncertainty envelope
    unc_res = DQCP_ENGINE.generate_interval(
        point_predictions=np.array(res.predictions, dtype=np.float64),
        delivery_hours=delivery_hours,
        reliability_state=res.routing_state,
        forecast_head=res.active_head,
        calibration_method=payload.calibration_method,
        target_coverage=payload.target_coverage,
        forecast_origin=forecast_origin
    )

    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    # Record Prometheus metrics safely
    try:
        if unc_res.uncertainty_available and unc_res.mean_interval_width_mw and not unc_res.is_unbounded:
            DQCP_INTERVAL_WIDTH.observe(unc_res.mean_interval_width_mw)
        if unc_res.is_unbounded:
            DQCP_UNBOUNDED_INTERVAL_COUNT.inc()
        DQCP_ACI_ALPHA.set(unc_res.current_alpha)
        if unc_res.model_id == "research_fallback":
            DQCP_FALLBACK_INTERVAL_COUNT.inc()
    except Exception as exc:
        logger.warning(f"Failed to record Prometheus metrics in /forecast/dqcp: {exc}")

    return DQCPForecastResponse(
        timestamps=[ts.isoformat() for ts in future_timestamps],
        forecast_mw=unc_res.forecast_mw,
        lower_bound_mw=unc_res.lower_bound_mw,
        upper_bound_mw=unc_res.upper_bound_mw,
        target_coverage=unc_res.target_coverage,
        calibration_method=unc_res.calibration_method,
        model_id=unc_res.model_id,
        calibration_version=unc_res.calibration_version,
        reliability_state=unc_res.reliability_state,
        forecast_origin=unc_res.forecast_origin,
        uncertainty_available=unc_res.uncertainty_available,
        is_unbounded=unc_res.is_unbounded,
        mean_interval_width_mw=unc_res.mean_interval_width_mw,
        current_alpha=unc_res.current_alpha,
        degradation_flag=unc_res.degradation_flag,
        inference_time_ms=res.inference_time_ms,
        total_time_ms=t_total_ms,
        generated_at=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    )


@app.post("/forecast/dqcp/feedback", response_model=DQCPFeedbackResponse)
def submit_dqcp_feedback(payload: DQCPFeedbackRequest):
    """
    Submits verified ground-truth settlement data for delayed-feedback ACI adaptation.
    Legacy declared-delay compatibility only; this does not verify actual source availability.
    The registered journal worker supplies the prospective feedback path.
    """
    if DQCP_ENGINE is None:
        raise HTTPException(status_code=503, detail="DQ-CP Uncertainty Engine is not initialized.")

    actuals = np.array(payload.actual_load_mw, dtype=np.float64)
    if len(actuals) != 24:
        raise HTTPException(status_code=400, detail=f"Expected 24 hourly load actuals, got {len(actuals)}.")

    try:
        rec = DQCP_ENGINE.submit_feedback(
            forecast_origin=payload.forecast_origin,
            actuals=actuals,
            settlement_timestamp=payload.settlement_timestamp,
            effective_delay_days=payload.effective_delay_days,
            head=payload.head
        )
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except KeyError as key_err:
        raise HTTPException(status_code=404, detail=str(key_err))

    if rec is None or rec.status == "DUPLICATE_IGNORED":
        alpha_curr = DQCP_ENGINE.primary_aci_manager.get_current_alpha()
        try:
            DQCP_FEEDBACK_COUNT.labels(status="duplicate").inc()
        except Exception:
            pass
        return DQCPFeedbackResponse(
            forecast_origin=payload.forecast_origin,
            settlement_timestamp=payload.settlement_timestamp,
            effective_delay_hours=int(payload.effective_delay_days * 24),
            miscoverage_occurred=False,
            empirical_coverage=1.0,
            alpha_before=alpha_curr,
            alpha_after=alpha_curr,
            gamma=DQCP_ENGINE.primary_aci_manager.gamma,
            status="duplicate_ignored"
        )
    elif rec.status == "REJECTED_REVISION":
        try:
            DQCP_FEEDBACK_COUNT.labels(status="rejected_revision").inc()
        except Exception:
            pass
        return DQCPFeedbackResponse(
            forecast_origin=payload.forecast_origin,
            settlement_timestamp=payload.settlement_timestamp,
            effective_delay_hours=int(payload.effective_delay_days * 24),
            miscoverage_occurred=False,
            empirical_coverage=1.0,
            alpha_before=rec.alpha_before,
            alpha_after=rec.alpha_after,
            gamma=rec.gamma,
            status="rejected_revision"
        )

    try:
        status_label = "miscovered" if rec.miscoverage_occurred else "covered"
        DQCP_FEEDBACK_COUNT.labels(status=status_label).inc()
        DQCP_ACI_ALPHA.set(rec.alpha_after)
    except Exception as exc:
        logger.warning(f"Failed to record Prometheus metrics in /forecast/dqcp/feedback: {exc}")

    return DQCPFeedbackResponse(
        forecast_origin=rec.forecast_origin,
        settlement_timestamp=rec.settlement_timestamp,
        effective_delay_hours=rec.effective_delay_hours,
        miscoverage_occurred=rec.miscoverage_occurred,
        empirical_coverage=rec.empirical_coverage,
        alpha_before=rec.alpha_before,
        alpha_after=rec.alpha_after,
        gamma=rec.gamma,
        status="processed"
    )


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint with multiprocess support."""
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        from prometheus_client import CollectorRegistry, multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

