# ADR 0011 — Code package, explicit research assets, optional toolchains

Status: accepted for local implementation and verification; no release publication.

The original project metadata depends unconditionally on training, export, database and tracking tools, relies on implicit package discovery and describes production readiness. Source-relative asset paths also fail after wheel installation. A working source checkout and a passing inherited environment do not prove an installable package.

Declare the actual src Python package explicitly, retain its import path for compatibility, and use an unreleased 2.0.0.dev0 distribution version. Default dependencies match the isolated operations profile. Diagnostics, training/statistics, ONNX export, tracking, Parquet and dashboard tooling become explicit extras. Broad legacy requirements remain a separate compatibility path; this change does not repair the old installed environment by assertion.

Ship Python code and accurate research metadata only. Do not bundle raw datasets, mutable journals, model weights, historical reports or console assets into the wheel. Resolve those through an explicit --workspace / CHRONOGRID_WORKSPACE or a recognized source checkout. The console must fail clearly when assets are absent; never substitute fabricated data. Preserve exact feature-source bytes so model identity checks also work in an installed package. Keep the frozen baseline/tag and all prior reports unchanged.

Provide the full operations CLI as a console entry point, retaining the original script as a compatibility wrapper. Standalone journal commands require an explicit database path. Read-only commands refuse nonexistent journals instead of creating empty success evidence. Source-dependent workers receive the selected workspace explicitly. A loopback-only serve command exposes the existing read-only console.

Accept: inspect wheel contents/metadata; build and install in a clean environment; run CLI/help/journal recovery away from the checkout; reject missing assets/journals; exercise console and model workers with explicit workspace; preserve frozen hashes; run meaningful package and full regressions. Resolve/audit optional research dependencies separately; do not represent an extra's declaration as tested compatibility. No push, package upload, release tag or public service.

References: [setuptools package discovery](https://setuptools.pypa.io/en/stable/userguide/package_discovery.html), [Python project metadata specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/).
