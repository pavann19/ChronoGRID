# ChronoGrid Production Runbook

**Operational Manual**: Deployment, Orchestration, Zero-Leakage Verification, Serving, Monitoring, and Disaster Recovery for the ChronoGrid ML Forecasting Platform.

---

## 1. Environment Setup

### Prerequisites
- Python 3.11+ (Python 3.13 validated)
- SQLite (included) or PostgreSQL 14+
- Local machine with 4GB+ RAM (24GB optimal for large tree benchmarks)

### Virtual Environment Initialization
```powershell
# Create virtual environment
python -m venv --system-site-packages .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
.\.venv\Scripts\pip install -r requirements.txt
```

---

## 2. Reproducing Independent Technical Audits & Verification

ChronoGrid includes automated audit scripts to reproduce all technical claims independently:

```powershell
# 1. Generate Data Provenance & Weather Vintage Audits
.\.venv\Scripts\python scripts/generate_provenance_and_vintage_audits.py

# 2. Run Comprehensive ONNX Performance Benchmark (10,000 queries) & Parity Suite
.\.venv\Scripts\python scripts/benchmark_onnx_and_parity.py

# 3. Execute 10-Point Adversarial Zero-Leakage Suite, 5-Fold Walk-Forward, and Governance Audits
.\.venv\Scripts\python scripts/run_full_audit_verification.py

# 4. Execute Full Automated Test Suite (34 passed tests)
.\.venv\Scripts\python -m pytest tests/ -v
```

Audit deliverables generated in `artifacts/reports/`:
- `leakage_verification.json`: 10-point adversarial leakage test results.
- `data_provenance_report.json`: SMARD API & DWD station 01420/10637 validation.
- `weather_vintage_audit.json`: Realised DWD observations vs NWP audit and persistence weather rules.
- `model_evaluation_report.json`: 5-fold walk-forward cross-validation leaderboard.
- `statistical_comparison.json`: Pairwise Harvey-Leybourne-Newbold Diebold-Mariano tests with Holm-Bonferroni correction.
- `onnx_performance.json`: Latency percentiles (p50, p95, p99), batch scalability, and throughput.
- `onnx_parity.json`: Numerical divergence verification between Python model and ONNX Runtime.
- `drift_verification.json`: 5-tier covariate, prediction, target, residual, and peak-hour drift telemetry.
- `promotion_verification.json`: Champion-challenger gate and atomic rollback verification.
- `test_results.json`: Summary of 34 automated unit and integration tests.

---

## 3. Running the Complete Master Lifecycle

To execute the entire production pipeline (Ingestion -> Validation -> Zero-Leakage Featurization -> Walk-Forward Benchmark -> ONNX Export -> Multi-Tier Drift Monitoring -> Retraining Gate):

```powershell
.\.venv\Scripts\python run_demo.py
```

---

## 4. Operational Day-Ahead Market (DAM) Rules

1. **Market Cutoff (Gate Closure)**:
   - Commercial auction bids close at **12:00 local time (`Europe/Berlin`) on Day $D-1$**.
   - In UTC:
     - Winter time (CET, UTC+1): **11:00 UTC**.
     - Summer daylight saving time (CEST, UTC+2): **10:00 UTC**.
2. **Delivery Period**:
   - The delivery day comprises the complete local day from **00:00 to 23:00 local time on Day $D$**.
   - During DST transitions, delivery spans **23 local hours in March (Spring)** and **25 local hours in October (Autumn)**.
3. **Information Boundary**:
   - For all 24 delivery hours of Day $D$, only data observed prior to gate closure ($T \le 12:00\text{ local on } D-1$) is available.
   - Realised future weather is prohibited. ChronoGrid operates with either:
     - **Active Production Mode**: Persistence weather known at issuance ($T \le T_{\text{origin}}$).
     - **Ablation Mode**: Pure load-only forecasting (excluding meteorological features).

---

## 5. Starting the Production FastAPI Inference Service

```powershell
# Launch FastAPI via Uvicorn
.\.venv\Scripts\uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 2
```

### Key Endpoints
- `GET http://localhost:8000/health`: Liveness probe.
- `GET http://localhost:8000/ready`: Readiness probe (verifies ONNX session initialized).
- `GET http://localhost:8000/model/info`: Active champion version, feature schema, and input dimensions.
- `POST http://localhost:8000/predict`: Sub-millisecond batch load predictions.
- `POST http://localhost:8000/forecast/day-ahead`: Automated 24h day-ahead forecast generation.
- `GET http://localhost:8000/metrics`: Prometheus scrapable metrics.

### Sample Prediction Query
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {
        "sin_hour": 0.5,
        "cos_hour": 0.866,
        "grid_load_mw_lag_24": 62500.0,
        "grid_load_mw_lag_168": 61200.0,
        "forecast_temperature_c": 14.5,
        "forecast_wind_speed_ms": 4.2
      }
    ]
  }'
```

---

## 6. Disaster Recovery & Rollback Procedure

If a freshly promoted champion model exhibits unexpected degradation in production:

1. **Trigger Immediate Rollback**:
   Execute the automated rollback method via Python or CLI:
   ```python
   from src.retraining.promotion_engine import PromotionEngine
   engine = PromotionEngine()
   result = engine.rollback_to_previous_champion()
   print(result)
   ```
2. **Mechanism**:
   - Locates the latest backup artifact in `artifacts/models/backup/champion_backup_*.onnx`.
   - Restores the previous ONNX binary and metadata JSON to `artifacts/models/champion_model.onnx`.
   - Updates the SQLite/PostgreSQL `model_registry` table, setting the degraded model to `Rollback_Demoted` and reactivating the previous champion as `Production` / `is_champion=True`.
   - The FastAPI engine automatically hot-reloads the restored ONNX session on the next request.

---

## 7. Monitoring & Operational Dashboards

- **Grafana Dashboard**: Import `dashboards/grafana_dashboard.json` for live monitoring of QPS, latency percentiles (p50/p95/p99), predicted load distributions, and drift scores.
- **Evidently-Compatible Drift Report**: Open `artifacts/reports/drift_report.html` for diagnostic visual charts of Kolmogorov-Smirnov distributions and PSI histograms.
- **Streamlit Interactive UI**: Run `.\.venv\Scripts\streamlit run src/monitoring/dashboard.py`.
