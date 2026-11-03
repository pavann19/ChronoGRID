# ChronoGrid Production Runbook

Operational manual for operating, deploying, monitoring, and maintaining ChronoGrid.

---

## 1. Environment Setup

### Prerequisites
- Python 3.11+ (Python 3.13 tested)
- SQLite (included) or PostgreSQL 14+
- 4GB+ RAM

### Virtual Environment Initialization
```powershell
# Create virtual environment
python -m venv --system-site-packages .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install requirements
.\.venv\Scripts\pip install -r requirements.txt
```

---

## 2. Running the Complete End-to-End Pipeline

To execute the entire lifecycle (Ingestion -> Validation -> Feature Engineering -> Benchmark -> ONNX Export -> Drift Check -> Retraining Gate):

```powershell
.\.venv\Scripts\python run_demo.py
```

---

## 3. Starting the Production FastAPI Inference Service

```powershell
# Launch FastAPI via Uvicorn
.\.venv\Scripts\uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 2
```

### Endpoints
- `GET http://localhost:8000/health`: Liveness probe.
- `GET http://localhost:8000/ready`: Readiness probe.
- `GET http://localhost:8000/model/info`: Current active champion version & features.
- `POST http://localhost:8000/predict`: Sub-millisecond batch inference.
- `POST http://localhost:8000/forecast/day-ahead`: Automated 24h rolling forecast.
- `GET http://localhost:8000/metrics`: Prometheus scrapable metrics.

### Example Inference Query
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {
        "temperature_c": 12.5,
        "wind_speed_ms": 4.5,
        "solar_radiation_wm2": 350.0,
        "humidity_pct": 65.0,
        "grid_load_mw_lag_24": 62000.0,
        "grid_load_mw_lag_168": 61500.0,
        "sin_hour": 0.5,
        "cos_hour": 0.866
      }
    ]
  }'
```

---

## 4. Monitoring & Telemetry

### Grafana Dashboard
- Import `dashboards/grafana_dashboard.json` into your Grafana instance.
- Pre-configured panels:
  - Active Champion Version (`chronogrid_champion_model_version`)
  - Inference QPS (`sum(rate(chronogrid_inference_requests_total[1m]))`)
  - Latency p50, p95, p99 (`histogram_quantile(..., chronogrid_inference_latency_seconds_bucket)`)
  - Drift Telemetry (`chronogrid_overall_psi_score`)

### Drift Inspection
Open `artifacts/reports/drift_report.html` in any browser to inspect feature-by-feature Kolmogorov-Smirnov statistics, PSI scores, and concept drift metrics.

### Streamlit Interactive Operations Dashboard
```powershell
.\.venv\Scripts\streamlit run src/monitoring/dashboard.py
```

---

## 5. Automated Orchestration via Airflow

The production DAG is configured in `configs/airflow_pipeline.py`.
To inspect or trigger with an Airflow scheduler:
```powershell
airflow dags trigger chronogrid_production_forecasting_pipeline
```

---

## 6. Running Unit & Integration Tests

```powershell
.\.venv\Scripts\python -m pytest tests/ -v
```
All 18 tests cover data ingestion, validation ranges, gap interpolation, leak-free lags, model training, ONNX serving, drift detection, and champion-challenger promotion gates.
