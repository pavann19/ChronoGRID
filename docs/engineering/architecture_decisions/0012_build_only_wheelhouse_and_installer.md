# ADR 0012 — Keep the operations wheelhouse and installer out of the runtime image

Status: accepted, local engineering checkpoint 2026-11-04.

## Context

The previous offline build copied the Linux wheelhouse into a final image layer and left pip available at runtime. The first Trivy image scan reported eight Python-package advisories associated with pip-vendored modules, alongside a larger Debian OS-package inventory. Runtime processes do not need to install or update packages.

## Decision

Use a read-only BuildKit context bind mount to install from the existing hash-locked wheelhouse, run `pip check`, then uninstall pip and remove `ensurepip` before adding application files. Pin both the Python base digest and Dockerfile frontend digest. Keep the runtime account non-root as before.

## Consequences

The wheelhouse is available during the build but is not stored in a final image layer. The final image no longer includes the Python package installer or its vendored modules, and the local image shrank by 324,831,917 bytes (27.2927%) compared with the earlier audit build. Imports, API readiness and image/rootfs vulnerability rows were verified against the resulting image.

This does not remediate the pinned Debian 13.7 OS-package findings. BuildKit and the pinned syntax frontend are required; the offline wheelhouse must still match the hash-locked requirements. A future base-image update must be evaluated and rescanned rather than assumed safe.
