# Changelog

All notable changes to the **ChronoGrid** platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-03

### Added
- **Day-Ahead Market (DAM) Framework**: Explicit gate closure at 12:00 CET/CEST on Day $D-1$ for the complete $12\text{--}35\text{h}$ delivery horizon of Day $D$ (`FeatureEngineer.build_day_ahead_market_frame`).
- **Zero-Leakage Invariance Suite**: 10-point adversarial perturbation suite testing future target spikes, weather spikes, DST transitions, rolling window shifts, and schema parity.
- **Statistical Significance Multi-Step Adjustment**: Harvey-Leybourne-Newbold (1997) autocorrelation-corrected Diebold-Mariano test ($h=24$) with Holm-Bonferroni family-wise error rate control.
- **ONNX Benchmark & Parity Verification**: 10,000-query latency profile (p50: 0.0162 ms, throughput: 51,421 QPS) and numerical divergence verification ($< 0.0032\text{ MW}$).
- **Automated Rollback Governance**: Atomic staging model promotion with instant backup rollback capability in `ModelPromotionEngine`.
- **CI/CD Pipeline**: GitHub Actions workflow testing Python 3.11, 3.12, and 3.13 across test execution, zero-leakage verification, and ONNX serving validation.
- **Project Packaging**: Added `pyproject.toml` and package metadata in `src/__init__.py`.

### Changed
- **Weather Baseline**: Deprecated and completely removed synthetic Gaussian AR(1) NWP noise injection. Instituted honest persistence weather ($T \le T_{\text{origin}}$) and load-only ablation modes.
- **Champion Selection**: Verified Random Forest as reproducible champion (MAE 1,043.55 MW, MAPE 2.101%), beating naive persistence by 72.3%.
- **Test Suite**: Expanded test coverage from 18 to 34 automated unit and integration tests (100% pass rate).

### Fixed
- Fixed skl2onnx boolean attribute coercion for modern ONNX 1.17+ runtime compatibility.
- Fixed rolling window off-by-one boundary leaks by shifting all target rolling statistics by $\ge 24\text{h}$ with `closed='left'`.
- Fixed German DST 23-hour Spring and 25-hour Autumn transition indexing via timezone-aware local grids.
