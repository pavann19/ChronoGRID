# ChronoGrid — Production ML Forecasting & MLOps Platform

[![Tests](https://img.shields.io/badge/tests-18%2F18%20passed-brightgreen.svg)]()
[![Inference](https://img.shields.io/badge/ONNX%20latency-0.17ms-blue.svg)]()
[![Target](https://img.shields.io/badge/Dataset-SMARD%20%2B%20DWD%20Germany-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)]()

> An end-to-end production ML forecasting and MLOps system built using real-world German electricity market data (**Bundesnetzagentur SMARD**) synchronized with meteorological conditions (**Deutscher Wetterdienst DWD**).

---

## Task Completion Matrix

| Task | Description | Status | Implementation Details |
| :--- | :--- | :---: | :--- |
| **P5.1** | Ingest SMARD electricity & DWD weather datasets | **DONE** | `src/ingestion/smard_client.py`, `dwd_client.py`, and `benchmark_generator.py` |
| **P5.2** | Build validated, reproducible data pipelines | **DONE** | `src/pipeline/validation.py`, `cleaning.py`, and PostgreSQL/SQLite ORM sync |
| **P5.3** | Perform time-series feature engineering | **DONE** | Cyclical hour/day/month encodings, holidays, HDH/CDH, wind chill, backward lags ($t-1 \dots t-168$), rolling stats |
| **P5.4** | Establish statistical and ML baselines | **DONE** | Naive 24h, Weekly 168h seasonal naive, Moving Average, and Ridge regression in `src/models/baselines.py` |
| **P5.5** | Train LightGBM, XGBoost and scikit-learn models | **DONE** | LightGBM, XGBoost, HistGradientBoosting, and Random Forest in `src/models/tree_models.py` |
| **P5.6** | Implement walk-forward validation | **DONE** | 5-fold expanding window cross-validation with 24h purge embargo and 95% bootstrap CIs |
| **P5.7** | Track experiments and versions with MLflow | **DONE** | MLflow tracking (`sqlite:///mlruns/mlflow.db`) and DB Model Registry (`src/tracking/mlflow_tracker.py`) |
| **P5.8** | Deploy inference using FastAPI and ONNX | **DONE** | ONNX export (`onnxmltools`), ONNX Runtime engine (0.17 ms/query), and FastAPI endpoints (`/predict`, `/forecast/day-ahead`) |
| **P5.9** | Detect data and model drift | **DONE** | Kolmogorov-Smirnov test, PSI, Wasserstein distance, CUSUM residual drift, and Evidently HTML report (`src/monitoring/drift_detector.py`) |
| **P5.10** | Implement controlled retraining & model promotion | **DONE** | Champion-Challenger evaluation gate with Diebold-Mariano test and atomic promotion in `src/retraining/promotion_engine.py` |
| **P5.11** | Add operational dashboards and monitoring | **DONE** | Prometheus metrics exporter (`/metrics`), Grafana dashboard JSON (`dashboards/grafana_dashboard.json`), and Streamlit UI |
| **P5.12** | Document reproducible experiments & performance | **DONE** | Comprehensive reports in `docs/architecture.md`, `docs/experiments_report.md`, and `docs/runbook.md` |

---

## Tech Stack

- **Core ML & Data**: Python 3.13, Pandas, NumPy, Scipy, Scikit-learn, XGBoost, LightGBM, Statsmodels.
- **Orchestration**: Apache Airflow DAG (`configs/airflow_pipeline.py`).
- **Database & Storage**: SQLAlchemy ORM with PostgreSQL (`psycopg2`) & SQLite support, Apache Parquet.
- **Experiment Tracking**: MLflow tracking server and Model Registry.
- **Model Serving**: FastAPI, Uvicorn, ONNX, ONNX Runtime (`onnxruntime`).
- **Monitoring & Drift**: Evidently-compatible HTML reporting, Kolmogorov-Smirnov test, Population Stability Index (PSI), Wasserstein distance.
- **Operational Metrics**: Prometheus Client (`prometheus_client`), Grafana Dashboard JSON.

---

## Quickstart

### 1. Run Complete Lifecycle in One Command
```powershell
.\.venv\Scripts\python run_demo.py
```

### 2. Run Test Suite (18 Tests)
```powershell
.\.venv\Scripts\python -m pytest tests/ -v
```

### 3. Launch FastAPI Serving
```powershell
.\.venv\Scripts\uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### 4. Launch Interactive Operations Dashboard
```powershell
.\.venv\Scripts\streamlit run src/monitoring/dashboard.py
```

---

## Walk-Forward Benchmark Results (Out-of-Fold 5 Folds)

```
================================================================================
               Model  MAE (MW)  RMSE (MW)  MAPE (%)     R2         MAE 95% CI Diebold-Mariano vs Naive
             XGBoost    640.71     836.31     1.449 0.9860   (603.67, 677.76)           p=0.0000 (***)
            LightGBM    646.73     848.16     1.472 0.9856   (608.61, 686.09)           p=0.0000 (***)
HistGradientBoosting    667.39     884.89     1.514 0.9843   (627.87, 707.06)           p=0.0000 (***)
       Random_Forest    754.54    1008.08     1.696 0.9797     (709.9, 796.9)           p=0.0000 (***)
    Ridge_Regression    829.31    1081.19     1.943 0.9766     (782.3, 875.6)           p=0.0000 (***)
         Weekly_168h   1783.09    2591.10     3.983 0.8657 (1662.26, 1912.72)           p=0.0000 (***)
           Naive_24h   4180.15    5857.29     9.518 0.3137  (3892.8, 4448.77)                 Baseline
      Moving_Avg_24h   5271.68    6340.10    12.242 0.1959 (5018.76, 5532.82)           p=0.0000 (***)
================================================================================
```
