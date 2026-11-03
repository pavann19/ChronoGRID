"""
FastAPI Model Serving Application (P5.8 & P5.11).
Ultra-low latency inference via ONNX Runtime with Prometheus metrics integration.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

from src.serving.schemas import (
    PredictRequest,
    PredictResponse,
    DayAheadForecastRequest,
    DayAheadForecastResponse,
    HealthResponse,
)
from src.serving.onnx_engine import OnnxInferenceEngine
from src.db.database import get_session, init_db
from src.db.models import ModelRegistryRecord

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

# Global engine instance
ENGINE: Optional[OnnxInferenceEngine] = None
CHAMPION_META = {"version": 1, "model_type": "Unknown", "promoted_at": None}


def load_engine():
    """Initializes ONNX engine and registry metadata."""
    global ENGINE, CHAMPION_META
    onnx_file = Path("artifacts/models/champion_model.onnx")
    meta_file = Path("artifacts/models/champion_metadata.json")

    if onnx_file.exists():
        ENGINE = OnnxInferenceEngine(
            onnx_model_path=str(onnx_file),
            metadata_path=str(meta_file),
        )
        # Fetch version from DB
        try:
            with get_session() as session:
                rec = session.query(ModelRegistryRecord).filter_by(is_champion=True).first()
                if rec:
                    CHAMPION_META = {
                        "version": rec.version,
                        "model_type": rec.model_type,
                        "promoted_at": str(rec.promoted_at),
                    }
                    CHAMPION_VERSION.set(rec.version)
        except Exception:
            pass
    else:
        ENGINE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    load_engine()
    yield
    # Shutdown


app = FastAPI(
    title="ChronoGrid Production ML Forecasting API",
    description="High-Throughput Electricity Demand Forecasting Engine (SMARD & DWD) with ONNX Runtime.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    return {"ready": True}


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
        response.headers["X-Model-Version"] = str(CHAMPION_META["version"])

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


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

