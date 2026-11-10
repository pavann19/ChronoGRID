# ADR 0007 — retire unlabeled simulated HTTP forecasts and reject implicit feature padding

Status: accepted for implementation; historical baseline, reports and sealed engineering_v1 remain unchanged.

Inspection found that the legacy day-ahead HTTP handler generates synthetic temperature/solar values, the resilient handler fills weather/load placeholders from a historical tail, and the DQCP handler combines production ONNX points with research calibration. Caller-supplied delay integers are not independent settlement evidence. The new prospective path must remain the authoritative capture path; those older routes must not present themselves as live prospective forecasts.

Decision: legacy forecasting/feedback HTTP routes return 410 by default. An operator can explicitly enable local replay/simulation with a configured bearer token; responses are labeled research replay/synthetic and never written as prospective records. Keep the underlying Python research functions and old artifacts reproducible. The low-level /predict endpoint is labeled caller-supplied features and requires a complete finite feature schema and bounded batch, without implicit zero-filled missing features.

Fix the shared ONNX engine and router input boundary: reject missing/duplicate feature columns, wrong dimensions, empty/nonfinite matrices and nonfinite outputs. Project primary-to-fallback numeric arrays by feature-name indices, not positional prefix truncation (weather features are interleaved). No model weights or calibration quantiles change. Numeric arrays already matching an engine's width retain that documented order contract.

CORS becomes an explicit localhost allowlist without credentials/wildcards. Legacy replay is a deliberate local research compatibility mode, not a production service. Shared-state prospective capture continues through the newer journal/worker; do not rewrite frozen ACI managers to hide old findings.

Validation: actual model predictions using name-based versus dataframe fallback projections; missing/extra/NaN/shape rejection; HTTP default retirement, missing/mismatched token, correct opt-in labels and new read-only API isolation; full regression. Original global environment remains unmodified. A new isolated service dependency profile will distinguish runtime requirements from optional training/export/statistical tools.
