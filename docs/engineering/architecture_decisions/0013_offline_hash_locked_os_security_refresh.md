# ADR: OS refresh policy after evidence export

Status: current implementation reviewed during portfolio cleanup.

I supersede the old dependency on locally stored OS-package payloads. The operations Dockerfile refreshes packages from the configured OS repositories at build time, while Python distributions remain hash-locked.

OS repository state is mutable, so I do not claim bit-for-bit image reproducibility. The earlier offline package review remains in the historical notes and exported evidence rather than as a hidden build dependency.
