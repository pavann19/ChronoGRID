# ChronoGrid System Architecture

ChronoGrid is an end-to-end, production-grade ML forecasting and MLOps platform engineered for high-precision Day-Ahead Electricity Grid Load forecasting in Germany, fusing Bundesnetzagentur (SMARD) market data with Deutscher Wetterdienst (DWD) meteorological observations.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Data_Ingestion["Data Ingestion & Pipeline (P5.1 - P5.2)"]
        SMARD["SMARD API\n(Grid Load, Solar, Wind, Spot Price)"] --> Ingest["Ingestion Pipeline\n(src/ingestion)"]
        DWD["DWD OpenData\n(Temperature, Irradiance, Wind)"] --> Ingest
        Ingest --> RawParquet["Raw Parquet Storage\n(data/raw)"]
        RawParquet --> Cleaner["Validation & Cleaner\nPydantic & Gap Interpolation"]
        Cleaner --> DB[(PostgreSQL / SQLite\nDatabase)]
        Cleaner --> CleanParquet["Canonical Hourly Dataset\n(data/processed)"]
    end

    subgraph Feature_Engineering["Time-Series Feature Engineering (P5.3)"]
        CleanParquet --> FE["FeatureEngineer (src/features)\nStrict Train-Fit / Zero-Leakage"]
        FE --> Temporal["Cyclical Trigonometric (sin/cos)\nGerman Holiday Calendar"]
        FE --> Thermodynamic["Heating/Cooling Degree Hours\nWind Chill & Solar Interactions"]
        FE --> Autoregressive["Shifted Lags (t-1..t-168)\nRolling Window Stats (6h..168h)"]
    end

    subgraph Modeling_Validation["Walk-Forward Validation & MLflow (P5.4 - P5.7)"]
        FE --> WF["Walk-Forward CV (5 Folds)\n24h Embargo Purge Gap"]
        WF --> Baselines["Statistical Baselines:\nNaive 24h, Weekly 168h, Rolling MA, Ridge"]
        WF --> MLModels["Gradient Boosted Trees:\nXGBoost, LightGBM, HistGB, RF"]
        MLModels --> Eval["Evaluation & Diebold-Mariano Test\nBootstrap 95% Confidence Intervals"]
        Eval --> MLflow["MLflow Tracking & Model Registry\n(sqlite:///mlruns/mlflow.db)"]
    end

    subgraph Serving_Inference["Production Serving (P5.8)"]
        MLflow --> ONNXExport["ONNX Exporter\n(onnxmltools / skl2onnx)"]
        ONNXExport --> ONNXEngine["ONNX Runtime Engine\n(< 0.2 ms / inference)"]
        ONNXEngine --> FastAPI["FastAPI Serving Application\n/predict, /forecast/day-ahead, /metrics"]
    end

    subgraph Monitoring_MLOps["Monitoring & Closed-Loop Retraining (P5.9 - P5.11)"]
        FastAPI --> PromMetrics["Prometheus Metrics Exporter\nLatency, QPS, Prediction Load"]
        PromMetrics --> Grafana["Grafana Dashboard\n(dashboards/grafana_dashboard.json)"]
        CleanParquet --> Drift["DriftDetector (P5.9)\nKS-Test, PSI, Wasserstein, CUSUM"]
        Drift --> DriftReport["Evidently-Compatible Drift Report\n(artifacts/reports/drift_report.html)"]
        Drift --> PromoGate["Champion-Challenger Gate (P5.10)\nStatistically Significant >=2% MAE Gain"]
        PromoGate -->|Approved| ONNXExport
        PromoGate -->|Rejected| StagingArchive["Log in Staging / Reject"]
    end
```

---

## 2. Walk-Forward Cross-Validation Strategy

To ensure zero future-data leakage and accurately mimic day-ahead dispatch conditions (gate closure at 12:00 CET for delivery the next day):

```mermaid
gantt
    title Walk-Forward Validation with 24h Purge Embargo
    dateFormat X
    axisFormat %s
    section Fold 1
    Train (90 Days)      :done, 0, 90
    24h Purge Embargo    :crit, 90, 91
    Test (1 Week)        :active, 91, 98
    section Fold 2
    Train (Expanding)    :done, 0, 98
    24h Purge Embargo    :crit, 98, 99
    Test (1 Week)        :active, 99, 106
    section Fold 3
    Train (Expanding)    :done, 0, 106
    24h Purge Embargo    :crit, 106, 107
    Test (1 Week)        :active, 107, 114
    section Fold 4
    Train (Expanding)    :done, 0, 114
    24h Purge Embargo    :crit, 114, 115
    Test (1 Week)        :active, 115, 122
    section Fold 5
    Train (Expanding)    :done, 0, 122
    24h Purge Embargo    :crit, 122, 123
    Test (1 Week)        :active, 123, 130
```

---

## 3. Champion-Challenger Promotion State Machine

```mermaid
stateDiagram-v2
    [*] --> ModelCandidateTrained: Scheduled or Drift-Triggered
    ModelCandidateTrained --> HoldoutEvaluation: Evaluate on identical recent test fold
    
    state PromotionGateDecision {
        [*] --> CheckMAE: Challenger MAE < Champion MAE * (1 - 0.02)
        CheckMAE --> CheckSignificance: Diebold-Mariano p-value < 0.05
        CheckSignificance --> CheckPeakHour: Peak error degradation <= 5%
    }
    
    HoldoutEvaluation --> PromotionGateDecision
    PromotionGateDecision --> ChampionPromoted: All criteria satisfied
    PromotionGateDecision --> ChallengerRejected: Any criterion fails

    state ChampionPromoted {
        BackupPriorChampion --> ExportONNX
        ExportONNX --> AtomicDBRegistryUpdate
        AtomicDBRegistryUpdate --> HotReloadInferenceEngine
    }
    
    ChallengerRejected --> LogStagingAudit: Mark as Staging / Log rejection rationale
    LogStagingAudit --> [*]
    HotReloadInferenceEngine --> [*]
```
