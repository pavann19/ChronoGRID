# Changelog

All notable changes are documented here using [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.1] - 2026-11-10

### Fixed
- Correct README grammar and link evaluation documentation to hosted CI results.
- Rename the legacy evaluation fixture tests and replace the deleted-script reference.
- Leave the published `v0.2.0` tag unchanged.

## [0.2.0] - 2026-11-10

### Fixed
- Correct the day-ahead information boundary with completed-hour checks and 48-hour lags.
- Replace future-dependent filling and rerun the benchmark on matching delivery hours.
- Require authentication for forecast routes and bind serving artifacts to a checksum registry.

### Changed
- Rename the Python package to `chronogrid` and consolidate packaging and dependency locks.
- Consolidate validation into one CI workflow and separate public findings from local working notes.

### Added
- Save corrected metrics, predictions, fold boundaries and gate audits under `artifacts/reports/benchmark_corrected/`.
- Document the protocol and remaining limits in [evaluation](docs/evaluation.md).

## [0.1.0] - 2026-11-03

### Added
- Implement data ingestion, forecasting, ONNX diagnostics and research evaluation.

### Known issues
- A later audit found post-gate lag features in the initial evaluation; see 0.2.0 for the correction.
