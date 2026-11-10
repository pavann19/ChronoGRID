# Evaluation

The corrected retrospective benchmark uses 5 expanding folds, a 24-hour purge and 35 reused Berlin delivery days. Models stay fixed during each test week; features update at the previous-day noon market gate. The corrected default uses 48-hour load lags. The earlier 24-hour variant is retained only as an explicitly leaky comparison.

Random Forest MAE is 1,146.79 MW; weekly persistence MAE is 1,225.04 MW. These values come from [metrics.csv](../artifacts/reports/benchmark_corrected/metrics.csv). The saved [predictions](../artifacts/reports/benchmark_corrected/per_hour_predictions.csv.gz), [folds](../artifacts/reports/benchmark_corrected/folds.csv), [gate audit](../artifacts/reports/benchmark_corrected/gate_audit.csv) and [metadata](../artifacts/reports/benchmark_corrected/report.json) record the evaluated hours, protocol and uncertainty estimates. Individual confidence intervals do not establish paired statistical superiority.

Run `python scripts/benchmark_corrected.py` to repeat the benchmark and `python scripts/audit_gate.py` to audit completed observation intervals. The audit does not reconstruct publication delays or historical revisions. Weather is persistence from historical observations, rather than archived forecast vintages. The TSO series is a downloaded revision without an original issuance receipt.

Run `python -m pytest`, `python -m ruff check chronogrid tests scripts`, `python -m ruff format --check chronogrid tests scripts` and `python -m mypy chronogrid` for source checks. These commands validate local behavior; this document does not assert a hosted CI result. Build `deployment/Dockerfile` to check the locked diagnostic container environment.

Older numerical fixtures under `artifacts/evaluation_legacy/` support regression tests for the previous fixed-model protocol. They are not the corrected headline result.

There is no untouched annual holdout or externally timestamped prospective evaluation. The corrected benchmark has not been promoted into the diagnostic serving artifacts. See [provenance and limitations](provenance_and_limitations.md).
