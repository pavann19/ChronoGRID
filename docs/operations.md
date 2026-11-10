# Running the diagnostic service

Install, test and run the corrected benchmark using the commands in the README. The service is an authenticated local diagnostic API; its historical ONNX artifacts are separate from the corrected benchmark winner.

`chronogrid-registry` registers and checks immutable model/metadata bundles. Registry checks reject incompatible identities before loading. Back up the SQLite registry before migration or recovery work.

`/health` and `/ready` are unauthenticated liveness and model-load probes. Predictions and model information require the diagnostics token. Historical replay requires a separate token and remains disabled by default. A probe does not establish forecast validity or suitability for an operated service.

The Python dependencies are declared in `pyproject.toml` and the service distributions and hashes are pinned in `deployment/requirements-service.lock`. `scripts/lock_dependencies.py` generates a new lock from an explicitly selected installed environment. The container verifier builds committed source without a local wheelhouse and probes the isolated runtime.

See [evaluation](evaluation.md), [architecture](architecture.md) and [the runbook](runbook.md).
