# ChronoGrid Experiments & Model Performance Report

**Project**: Production ML Forecasting & MLOps Platform  
**Target Variable**: German Electricity Grid Load (`grid_load_mw` in Megawatts)  
**Evaluation Dataset**: 13,128 hours (2023-01-01 00:00:00 UTC to 2024-06-30 23:00:00 UTC)  
**Evaluation Scheme**: 5-Fold Expanding-Window Walk-Forward Validation with 24-hour Purge Embargo  

---

## 1. Executive Summary

Electricity demand forecasting is central to transmission system operations (TSO), intraday scheduling, and reserve dispatch. ChronoGrid fuses SMARD grid telemetry with DWD meteorological parameters to predict hourly German electricity load with sub-1.5% Mean Absolute Percentage Error (MAPE). 

Through rigorous walk-forward cross-validation, gradient boosted tree architectures (XGBoost and LightGBM) achieved an **84.7% error reduction** over traditional day-ahead naive baselines, and were deployed via ONNX Runtime delivering **sub-millisecond inference latencies (0.17 ms/query)**.

---

## 2. Walk-Forward Benchmark Leaderboard

The table below summarizes out-of-fold performance across all 5 test folds (840 out-of-fold test hours with 24h purge gaps):

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | sMAPE (%) | $R^2$ | MAE 95% Bootstrap CI | Diebold-Mariano vs Naive Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **640.71** | **836.31** | **1.449%** | **1.450%** | **0.9860** | **[603.67, 677.76]** | **$p < 0.0001$ (\*\*\*)** |
| **LightGBM** | 646.73 | 848.16 | 1.472% | 1.472% | 0.9856 | [608.61, 686.09] | $p < 0.0001$ (\*\*\*) |
| **HistGradientBoosting** | 667.39 | 884.89 | 1.514% | 1.513% | 0.9843 | [627.87, 707.06] | $p < 0.0001$ (\*\*\*) |
| **Random Forest** | 754.54 | 1008.08 | 1.696% | 1.694% | 0.9797 | [709.90, 796.90] | $p < 0.0001$ (\*\*\*) |
| **Ridge Regression** | 829.31 | 1081.19 | 1.943% | 1.951% | 0.9766 | [782.30, 875.60] | $p < 0.0001$ (\*\*\*) |
| **Weekly Seasonal (168h)** | 1,783.09 | 2,591.10 | 3.983% | 3.995% | 0.8657 | [1662.26, 1912.72] | $p < 0.0001$ (\*\*\*) |
| **Naive Persistence (24h)** | 4,180.15 | 5,857.29 | 9.518% | 9.596% | 0.3137 | [3892.80, 4448.77] | *Reference Baseline* |
| **Moving Average (24h)** | 5,271.68 | 6,340.10 | 12.242% | 11.893% | 0.1959 | [5018.76, 5532.82] | $p < 0.0001$ (\*\*\*) |

---

## 3. Key Findings & Insights

### A. Non-Linear Interaction vs Linear Models
- Standard regularized linear regression (Ridge) achieved an MAE of 829.31 MW.
- Non-linear tree ensembles (XGBoost, LightGBM) improved MAE to **640.71 MW**, demonstrating that thermodynamic thresholds (Heating Degree Hours $< 18^\circ\text{C}$, Cooling Degree Hours $> 22^\circ\text{C}$) and quadratic solar-temperature interactions exhibit strong non-linear behavior that linear models fail to fully capture.

### B. Statistical Significance
- The Diebold-Mariano test was computed between out-of-fold loss differentials.
- The test statistic comparing XGBoost against the standard Naive 24h baseline yielded $DM = -18.42$, $p = 0.0000$, confirming that the accuracy improvement is statistically significant beyond random noise.

### C. Top Feature Importances (Gain-based)
1. `grid_load_mw_lag_24` (Yesterday same hour load): **38.4%**
2. `grid_load_mw_lag_168` (Last week same day & hour): **22.1%**
3. `cos_hour` / `sin_hour` (Diurnal circadian cycle): **14.2%**
4. `temperature_c` & `heating_degree_hours` (Space heating sensitivity): **11.8%**
5. `is_working_day` / `is_holiday` (Industrial production drop): **6.5%**
6. `solar_radiation_wm2` & rooftop self-consumption: **4.2%**
7. `wind_speed_ms` / `wind_chill_c`: **2.8%**

---

## 4. Operational Trade-Offs

| Model Engine | Model Format | Size on Disk | Inference Latency (10 queries) | Inference Latency / Query | Memory Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | ONNX (`.onnx`) | **16.5 KB** | **1.71 ms** | **0.171 ms** | **~42 MB** |
| LightGBM | ONNX (`.onnx`) | 3.0 KB | 1.85 ms | 0.185 ms | ~38 MB |
| Random Forest | Scikit-learn (`.joblib`) | 48.2 MB | 22.40 ms | 2.240 ms | ~180 MB |

**Conclusion**: ONNX conversion reduces model artifact size by $>99\%$ compared to uncompressed pickle/joblib trees, and ONNX Runtime execution provider delivers $>10\times$ speedup over Python interpreter inference loops.

---

## 5. Drift Detection & Retraining Telemetry

During testing, empirical weather distributions shifted from late spring to high-summer conditions:
- **Temperature Drift**: Kolmogorov-Smirnov $p = 0.0001$, $\text{PSI} = 0.324$ (Triggered `ALERT`).
- **Solar Irradiance Drift**: $\text{PSI} = 0.281$ (Triggered `ALERT`).
- **Overall Population Stability Index**: $\text{PSI} = 0.2926$ ($> 0.25$ alert threshold).
- **Automated Retraining Gate**:
  - Challenger model (LightGBM) trained on newest window and evaluated against Champion (XGBoost).
  - Challenger achieved MAE of 963.8 MW vs Champion MAE of 640.7 MW on holdout evaluation (improvement: $-50.4\%$).
  - **Promotion Gate Action**: **REJECT_CHALLENGER**. The platform successfully blocked regression, logged the challenger in Staging for auditability, and maintained the production Champion intact.
