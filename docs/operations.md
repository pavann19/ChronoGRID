# Operations

I use ChronoGRID as a local forecasting research demonstration. I do not claim production readiness or managed operation. The root README contains the supported install, test, benchmark, serving and Docker commands.

The `chronogrid-ops` command manages explicitly selected local journals and workspaces; `chronogrid-registry` manages reviewed diagnostic artifact bundles. A journal records forecast/model/source identities and applies feedback transactionally. Health and readiness are probes, not assertions of forecast quality or external evidence custody.

The diagnostics token protects prediction, model information and metrics. Forecast routes require the replay token before their enablement state is exposed. I keep replay disabled unless I am running an explicitly labelled historical or synthetic exercise. The corrected benchmark does not automatically promote a model, change a registry or publish an artifact.

I declare dependencies in `pyproject.toml`; the runtime lockfiles under `deployment/` pin distributions and hashes. `scripts/lock_operations_dependencies.py` selects the operations or service profile from that declaration when creating a new lock. Local development may have a different installed environment, so I record which checks actually ran and do not imply locked-environment coverage from them.

I keep historical working notes in `docs/archive/`, and large evidence outside Git tracking. The optional evidence bundle can be restored for old audit scripts. Current acceptance evidence and unresolved checks are listed in `artifacts/reports/cleanup_report.md`. Recovery and credential handling are described in [the runbook](runbook.md).
