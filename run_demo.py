"""
ChronoGrid Master Lifecycle Orchestrator (P5.1 - P5.12).
Executes the complete production ML forecasting pipeline from raw data to validated ONNX deployment.
"""

import logging
import os
import sys
from pathlib import Path

# Silence joblib core warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ChronoGrid")

from chronogrid.db.database import init_db
from chronogrid.ingestion.ingest_pipeline import IngestionPipeline
from chronogrid.pipeline.pipeline import DataPipeline
from chronogrid.features.engineer import FeatureEngineer
from chronogrid.models.trainer import ModelBenchmarkTrainer
from chronogrid.tracking.mlflow_tracker import MLflowTracker
from chronogrid.serving.onnx_exporter import export_to_onnx
from chronogrid.serving.onnx_engine import OnnxInferenceEngine
from chronogrid.monitoring.drift_detector import DriftDetector
from chronogrid.monitoring.metrics import update_drift_prometheus_metrics
from chronogrid.retraining.promotion_engine import PromotionEngine


def main():
    logger.info("=" * 80)
    logger.info("CHRONOGRID: PRODUCTION ML FORECASTING & MLOPS PLATFORM")
    logger.info("=" * 80)

    # 0. Initialize Database
    logger.info("[INIT] Initializing Database Schema (SQLite / PostgreSQL)...")
    db_file = Path("data/chronogrid.db")
    if db_file.exists():
        try:
            db_file.unlink()
            logger.info("Cleared previous staging database to ensure clean sync from raw data.")
        except Exception:
            pass
    init_db()

    # P5.1: Ingestion
    logger.info("\n--- [P5.1] Ingesting SMARD Electricity and DWD Weather Datasets ---")
    ingest = IngestionPipeline()
    df_raw = ingest.run(use_live_api=False, start_date="2023-01-01", end_date="2024-06-30")
    logger.info(f"Ingested raw dataset with {len(df_raw)} records.")
    logger.info(f"Source confirmation: SMARD filter 410 (Total Grid Load DE) & DWD OpenData Climate Observations.")

    # P5.2: Validated Reproducible Data Pipeline
    logger.info("\n--- [P5.2] Executing Validated Reproducible Data Pipeline ---")
    pipeline = DataPipeline()
    df_clean, pipe_metrics = pipeline.process(df_raw, sync_to_db=True)
    logger.info(f"Data pipeline complete: {len(df_clean)} hourly records validated and persisted.")

    # P5.3: Feature Engineering
    logger.info("\n--- [P5.3] Time-Series Feature Engineering (Strict Day-Ahead Gate Closure) ---")
    fe = FeatureEngineer(target_col="grid_load_mw", min_lag_hours=24)
    df_feat = fe.fit_transform(df_clean, drop_warmup=True)
    feature_cols = fe.get_feature_names()
    fe.verify_zero_future_leakage(df_feat)
    logger.info(f"[information-boundary VERIFIED] All load lags strictly >= 24h. Contemporaneous realised weather excluded.")
    logger.info(f"Engineered {len(feature_cols)} features strictly available at forecast issuance time.")
    logger.info(f"Features sample: {feature_cols[:8]} ...")

    # P5.4, P5.5, P5.6: Baselines, ML Models & Walk-Forward Validation
    logger.info("\n--- [P5.4 - P5.6] Walk-Forward Cross-Validation Benchmark ---")
    trainer = ModelBenchmarkTrainer(
        target_col="grid_load_mw",
        n_splits=5,
        test_size_hours=168,  # 1 week test per fold
        purge_gap_hours=24    # 24h embargo
    )
    summary_df, all_results, champion_model = trainer.run_benchmark(df_feat, feature_cols)
    
    print("\n" + "=" * 80)
    print("WALK-FORWARD MODEL BENCHMARK LEADERBOARD (Out-of-Fold 5 Folds)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    champion_row = summary_df.iloc[0]
    champion_name = champion_row["Model"]
    logger.info(f"Top Performer Selected as Champion: {champion_name} (MAE: {champion_row['MAE (MW)']} MW)")

    # P5.7: Experiment Tracking and Model Registry (MLflow)
    logger.info("\n--- [P5.7] Experiment Tracking & Model Registry with MLflow ---")
    tracker = MLflowTracker()
    champ_metrics = all_results[champion_name]["overall_metrics"]
    champ_importances = getattr(champion_model, "get_feature_importances", lambda: {})()

    run_id = tracker.log_model_run(
        model_name=champion_name,
        model_type=champion_name,
        params={"n_splits": 5, "test_hours": 168, "purge_gap": 24},
        metrics=champ_metrics,
        feature_names=feature_cols,
        feature_importances=champ_importances,
        is_champion=True,
    )
    logger.info(f"Experiment logged to MLflow (Run ID: {run_id}). Champion model registered.")

    # P5.8: Deploy Inference using ONNX and FastAPI
    logger.info("\n--- [P5.8] Exporting to ONNX Runtime Engine & Deploying Inference ---")
    onnx_path = export_to_onnx(
        model_wrapper=champion_model,
        feature_names=feature_cols,
        output_path="artifacts/models/champion_model.onnx",
        metadata_path="artifacts/models/champion_metadata.json",
    )
    
    # Test ONNX Runtime Latency
    onnx_engine = OnnxInferenceEngine(
        onnx_model_path="artifacts/models/champion_model.onnx",
        metadata_path="artifacts/models/champion_metadata.json",
    )
    test_features = df_feat.iloc[-10:][feature_cols].to_dict(orient="records")
    
    import time
    t0 = time.perf_counter()
    preds = onnx_engine.predict(test_features)
    t_onnx = (time.perf_counter() - t0) * 1000.0
    logger.info(f"ONNX Runtime successfully served {len(preds)} predictions in {t_onnx:.2f} ms ({t_onnx/len(preds):.3f} ms/query)!")
    logger.info(f"Sample predictions: {np.round(preds[:3], 1)} MW")

    # P5.9: Detect Data and Model Drift
    logger.info("\n--- [P5.9] Drift Detection & Monitoring ---")
    detector = DriftDetector()
    # Reference: 30 days prior; Current: last 7 days
    ref_window = df_clean.iloc[-888:-168]
    curr_window = df_clean.iloc[-168:]

    ref_preds = champion_model.predict(df_feat.iloc[-888:-168][feature_cols])
    curr_preds = champion_model.predict(df_feat.iloc[-168:][feature_cols])

    drift_report = detector.compute_drift(
        ref_df=ref_window,
        curr_df=curr_window,
        features=["temperature_c", "grid_load_mw", "wind_speed_ms", "solar_radiation_wm2", "humidity_pct"],
        y_ref_tuple=(ref_window["grid_load_mw"].values, ref_preds),
        y_curr_tuple=(curr_window["grid_load_mw"].values, curr_preds),
        report_output_html="artifacts/reports/drift_report.html",
    )
    logger.info(f"Drift Analysis Completed: Overall PSI = {drift_report.overall_drift_score:.4f}, Drift Detected = {drift_report.drift_detected}")
    update_drift_prometheus_metrics(
        psi=drift_report.overall_drift_score,
        drift_detected=drift_report.drift_detected,
        mape=champ_metrics["mape"]
    )

    # P5.10: Controlled Retraining and Champion-Challenger Promotion
    logger.info("\n--- [P5.10] Controlled Retraining & Champion-Challenger Promotion Gate ---")
    promotion_engine = PromotionEngine(min_improvement_pct=0.01) # 1% gate for demo
    
    # Train challenger model (e.g. XGBoost challenger vs LightGBM champion)
    from chronogrid.models.tree_models import XGBoostModel, LightGBMModel
    challenger_type = "XGBoost" if "LightGBM" in champion_name else "LightGBM"
    logger.info(f"Training Challenger Model ({challenger_type}) on latest window...")
    challenger = XGBoostModel(n_estimators=220, learning_rate=0.04) if challenger_type == "XGBoost" else LightGBMModel(n_estimators=220, learning_rate=0.04)
    challenger.fit(df_feat.iloc[:-168][feature_cols], df_feat.iloc[:-168]["grid_load_mw"].values)

    eval_holdout = df_feat.iloc[-168:]
    promotion_decision = promotion_engine.evaluate_and_promote(
        champion_model=champion_model,
        challenger_model=challenger,
        df_eval=eval_holdout,
        feature_cols=feature_cols,
        target_col="grid_load_mw",
        challenger_model_type=challenger_type,
    )
    logger.info(f"Promotion Gate Result: {promotion_decision['decision']}")

    # P5.11 & P5.12: Operational Dashboard and Experiments Documentation
    logger.info("\n--- [P5.11 & P5.12] Monitoring Artifacts & Documentation Complete ---")
    logger.info("Operational Grafana Dashboard: dashboards/grafana_dashboard.json")
    logger.info("Evidently-Compatible Drift Report: artifacts/reports/drift_report.html")
    logger.info("Champion ONNX Model: artifacts/models/champion_model.onnx")
    logger.info("MLflow Tracking URI: sqlite:///mlruns/mlflow.db")
    logger.info("FastAPI Serving App: src/serving/app.py (Run with uvicorn chronogrid.serving.app:app)")
    logger.info("=" * 80)
    logger.info("CHRONOGRID LIFECYCLE EXECUTION SUCCESSFULLY COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
