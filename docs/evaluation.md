# Evaluation

The corrected retrospective benchmark uses 5 expanding folds, a 24-hour purge and 35 consecutive summer Berlin delivery days (2024-06-17 to 2024-07-21). There is no winter or heating-season evidence. Models stay fixed during each test week; features update at the previous-day noon market gate. The corrected default uses 48-hour load lags. The earlier 24-hour variant is retained only as an explicitly leaky comparison.

Random Forest MAE was 1,043.55 MW with post-gate lags and 1,146.79 MW with the corrected pipeline; weekly persistence MAE is 1,225.04 MW. These values come from [metrics.csv](../artifacts/reports/benchmark_corrected/metrics.csv). The saved [predictions](../artifacts/reports/benchmark_corrected/per_hour_predictions.csv.gz), [folds](../artifacts/reports/benchmark_corrected/folds.csv), [gate audit](../artifacts/reports/benchmark_corrected/gate_audit.csv) and [metadata](../artifacts/reports/benchmark_corrected/report.json) record the evaluated hours, protocol and uncertainty estimates. The learned models in the paired comparisons are statistically indistinguishable from weekly persistence in this window: all paired 95% intervals include zero. This does not establish equivalence or a prospective advantage.

| Comparison (MAE difference) | Observed difference (MW) | Paired 95% CI (MW) | Excludes zero? |
|---|---:|---:|---|
| RandomForest minus Weekly_168h | -78.2489 | [-180.9474, 15.7669] | No |
| RandomForest minus HistGradientBoosting | -11.9465 | [-110.2582, 82.8351] | No |
| HistGradientBoosting minus Weekly_168h | -66.3024 | [-219.6237, 84.2748] | No |
| LightGBM minus Weekly_168h | -41.3045 | [-207.1093, 115.0052] | No |
| XGBoost minus Weekly_168h | 54.7214 | [-152.3170, 277.6310] | No |

The paired differences use 10,000 circular moving-block resamples with NumPy `RandomState(42)`: 18 blocks of two complete Berlin delivery days per replicate, truncated to 35 days, with shared sampled hours across models and percentile 95% intervals. Negative differences favor the first model. Results are saved in [paired_differences.csv](../artifacts/reports/benchmark_corrected/paired_differences.csv). Run `make paired-bootstrap` (or `python scripts/paired_bootstrap.py`) to reproduce these comparisons from the saved predictions without retraining or downloading data.

Run `python scripts/benchmark_corrected.py` to repeat the benchmark and `python scripts/audit_gate.py` to audit completed observation intervals. The audit does not reconstruct publication delays or historical revisions. Weather is persistence from historical observations, rather than archived forecast vintages.

A latest-revision SMARD TSO series (filter 411) was tried as an external baseline. It showed a systematic +1,051 MW mean bias (largest between 11:00 and 16:00), no timestamp misalignment (zero shift was optimal) and irregular one-hour dips on some days, so its scope or vintage does not match the realised-load series (filter 410). It is not used for any comparison.

Run `python -m pytest`, `python -m ruff check chronogrid tests scripts`, `python -m ruff format --check chronogrid tests scripts` and `python -m mypy chronogrid` for source checks. These commands validate local behavior. Hosted results are available through the [CI badge](https://github.com/pavann19/ChronoGRID/actions/workflows/ci.yml/badge.svg?branch=master) and [CI workflow runs](https://github.com/pavann19/ChronoGRID/actions/workflows/ci.yml); the [workflow definition](../.github/workflows/ci.yml) covers Python 3.11, 3.12 and 3.13. Build `deployment/Dockerfile` to check the locked diagnostic container environment.

Older numerical fixtures under `artifacts/evaluation_legacy/` support regression tests for the previous fixed-model protocol. They are not the corrected headline result.

There is no untouched annual holdout or externally timestamped prospective evaluation. The corrected benchmark has not been promoted into the diagnostic serving artifacts. See [provenance and limitations](provenance_and_limitations.md).
