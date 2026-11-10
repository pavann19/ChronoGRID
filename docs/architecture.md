# Architecture

I separate historical evaluation from model serving. The corrected feature pipeline defaults to elapsed-hour load lags and rolling windows whose newest observation is complete before the previous-day noon gate in Europe/Berlin. Calendar features are deterministic; weather is a lagged persistence input. I reindex missing hours before shifting so a row offset cannot silently become the wrong elapsed-time lag.

```mermaid
flowchart LR
    A[SMARD load and weather archive] --> B[Causal cleaning]
    B --> C[Load lags and persistence weather]
    C --> D[Gate availability audit]
    D --> E[Expanding fold training and evaluation]
    E --> F[Reviewed model registry]
    F --> G[Authenticated ONNX diagnostics]
    G --> H[Metrics and local evidence journal]
```

`chronogrid/features/gate_audit.py` resolves the latest dependency for each delivery-hour feature and fails closed for unknown names. The leakage suite checks transformations by perturbing raw observations; the data-integrity suite separately checks timezone, DST, horizon and schema behavior. Publication-time evidence is outside this event-time audit.

`scripts/benchmark_corrected.py` trains fresh models in expanding folds and writes hourly predictions, fold boundaries, gate counts, metrics and block-bootstrap intervals. Its post-gate comparison requires `allow_post_gate_lags=True`. Model selection in this retrospective experiment does not change the served artifact.

`chronogrid/serving/app.py` authenticates diagnostics and every forecast path. Local artifacts are checked against `artifacts/models/serving_registry.json` before ONNX Runtime loads them; the optional transactional artifact registry verifies immutable model/metadata bundles. Health probes remain open. Legacy replay is disabled by default.

The optional operations journal records local forecasts, source snapshots and feedback with identities and transactional checks. Local timestamps and hashes do not provide external attestation. I retain frozen evidence as byte-addressed snapshots and compact test fixtures, while bulky historical outputs live in an ignored evidence bundle.
