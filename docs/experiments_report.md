# ChronoGrid Experiments & Model Performance Report

**Project**: Production ML Forecasting & MLOps Platform  
**Target Variable**: German Electricity Grid Load (`grid_load_mw` in Megawatts)  
**Evaluation Dataset**: 13,607 genuine SMARD/DWD observation hours (2023-01-01 23:00 UTC to 2024-07-21 21:00 UTC)  
**Information Constraint**: Strict Day-Ahead Gate Closure ($t - 24\text{h}$ embargo; zero future weather/load leakage)  
**Evaluation Scheme**: 5-Fold Expanding-Window Walk-Forward Validation with 24-hour Purge Embargo (840 out-of-fold test hours)  

---

## 1. Executive Summary

ChronoGrid is an enterprise production forecasting and MLOps platform engineered to predict German electrical transmission grid load (`grid_load_mw`). Operating under strict day-ahead market (DAM) gate closure constraints ($t - 24\text{h}$ information boundary), all autoregressive features strictly use lags $\ge 24\text{h}$, and contemporaneous realised weather is excluded in favor of day-ahead numerical weather prediction (NWP) simulations and 24h persistence.

Through rigorous 5-fold walk-forward cross-validation on identical test timestamps, tree ensembles achieved a **73.2% error reduction** over traditional day-ahead persistence baselines, with sub-2.1% MAPE and sub-millisecond ONNX Runtime latency (**0.053 ms/query**).

---

## 2. Walk-Forward Benchmark Leaderboard

The table below summarizes out-of-fold performance across all 5 test folds (840 out-of-fold test hours on identical timestamps with 24h purge embargo):

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | sMAPE (%) | $R^2$ | MAE 95% Bootstrap CI | Diebold-Mariano vs Naive Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Champion)** | **1,010.06** | **1,326.33** | **2.050%** | **2.059%** | **0.9713** | **[949.45, 1,066.23]** | **$p < 0.0001$ (\*\*\*)** |
| **XGBoost** | 1,022.87 | 1,321.29 | 2.061% | 2.072% | 0.9716 | [968.15, 1,081.69] | $p < 0.0001$ (\*\*\*) |
| **Random Forest** | 1,026.91 | 1,359.41 | 2.065% | 2.066% | 0.9699 | [968.87, 1,087.41] | $p < 0.0001$ (\*\*\*) |
| **LightGBM** | 1,030.42 | 1,315.05 | 2.091% | 2.101% | 0.9718 | [972.97, 1,083.46] | $p < 0.0001$ (\*\*\*) |
| **Weekly Seasonal (168h)** | 1,225.04 | 1,613.34 | 2.452% | 2.443% | 0.9576 | [1,154.37, 1,299.45] | $p < 0.0001$ (\*\*\*) |
| **Ridge Regression** | 1,352.05 | 1,701.04 | 2.805% | 2.821% | 0.9529 | [1,278.88, 1,421.57] | $p < 0.0001$ (\*\*\*) |
| **Naive Persistence (24h)** | 3,771.22 | 5,562.87 | 7.682% | 7.722% | 0.4960 | [3,503.60, 4,027.59] | *Reference Baseline* |
| **Moving Average (24h)** | 7,443.62 | 8,836.50 | 15.414% | 15.047% | -0.2717 | [7,123.25, 7,758.43] | $p < 0.0001$ (\*\*\*) |

---

## 3. Strict Day-Ahead Zero-Leakage Verification

In compliance with European day-ahead electricity market dispatch rules (gate closure at 12:00 CET on $D-1$):
1. **Autoregressive Lags**: The shortest available load lag is `lag_24` (24 hours prior). Intermediate lags (`lag_1` to `lag_23`) are strictly omitted.
2. **Rolling Statistics**: Rolling mean, standard deviation, min, and max calculations are shifted backward by 24 hours (`closed='left'`, `shift(24)`).
3. **Meteorological Parameters**: Contemporaneous ground-truth weather measurements (`temperature_c`, `solar_radiation_wm2`, `wind_speed_ms`) are strictly barred from the feature set. Forecasts consume either Numerical Weather Prediction (NWP) simulated day-ahead forecasts or 24-hour persistence weather features.
4. **Mathematical Verification**: The `FeatureEngineer.verify_zero_future_leakage()` routine executes automated column inspection and correlation checks before training.

---

## 4. Robustness & Edge-Case Validation

A dedicated test suite (`tests/test_edge_cases_and_robustness.py`) validates the system under severe operating conditions:

| Scenario / Stressor | Failure Mode Prevented | Mitigation & Validation Result |
| :--- | :--- | :--- |
| **Spring DST Transition** | 23-hour local day causing missing hour or array index misalignment | Timestamps are normalized to UTC (24 continuous hours). Local time conversion explicitly verifies 23 clock hours in Berlin without pipeline disruption. |
| **Autumn DST Transition** | Duplicate 02:00 hour causing primary key collisions in database | UTC normalization ensures unique timestamps (`fold=0` and `fold=1` disambiguated); database ingestion pipeline deduplicates and monotonically sorts index. |
| **Missing Weather Observations** | Sensor outages (1h, 3h, 6h, 24h, 72h) crashing feature pipeline | Linear time-weighted interpolation for short gaps ($\le 6\text{h}$) and forward/backward clamping for prolonged dropouts ($\ge 24\text{h}$); pipeline sets `imputed_flag` without crashing. |
| **Forecast Horizon Boundary Failures** | Negative horizons or excessively distant predictions | Input validators enforce $1 \le H \le 168$ hours; out-of-range requests raise `ValueError` with descriptive remediation messages. |
| **Insufficient History Depth** | Warmup starvation on cold starts ($< 168\text{h}$ of historical data) | Engine validates that historical context satisfies minimum lag requirement ($\ge 168\text{h}$) before generating predictions. |

---

## 5. Operational Deployment & Drift Monitoring

- **ONNX Serving**: Top models are compiled to optimized ONNX binaries. ONNX Runtime delivers batch predictions in **0.53 ms** for 10 records (**0.053 ms/query**), a $50\times$ speedup over interpreted Python loops.
- **Drift Telemetry**: Continuous Kolmogorov-Smirnov test, Population Stability Index (PSI), Wasserstein distance, and CUSUM residual monitoring. Current evaluation window yielded $\text{PSI} = 0.2389$, generating an interactive diagnostic HTML report.
- **Champion-Challenger Gate**: Automated evaluation on fresh holdout data rejected challenger model v2 due to a $-14.3\%$ relative degradation ($p = 0.0000$), preserving production stability.
