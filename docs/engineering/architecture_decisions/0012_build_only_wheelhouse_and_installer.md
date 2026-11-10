# ADR: Build-time locked dependencies

Status: current implementation reviewed during portfolio cleanup.

I declare dependencies in pyproject.toml and retain hash-locked operations and service profiles. Docker installs a transferable virtual environment in a build stage and copies the installed environment into the runtime.

The build needs no ignored local wheelhouse. I avoid BuildKit-only mount syntax so the same source can be built by a classic Docker backend. A build result does not imply vulnerability clearance.
