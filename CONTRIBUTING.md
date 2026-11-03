# Contributing to ChronoGrid

Thank you for your interest in contributing to **ChronoGrid**!

We welcome contributions from machine learning engineers, data engineers, energy market analysts, and infrastructure specialists. To maintain the scientific rigor and production reliability of the platform, please review the following guidelines.

---

## Code of Conduct

Maintain professional, constructive, and respectful collaboration in all interactions.

---

## Development Workflow

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/<your-username>/ChronoGRID.git
   cd ChronoGRID
   ```

2. **Environment Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install pytest pytest-cov ruff
   ```

3. **Branching**:
   Use descriptive, kebab-case branch names prefixed with the change type:
   - `feat/feature-name`
   - `fix/bug-description`
   - `docs/doc-update`
   - `perf/optimization`

4. **Code Standards**:
   - Python 3.11+ compatibility.
   - Run code formatting and linting:
     ```bash
     ruff check src tests --fix
     ```
   - All public functions and classes must include clear docstrings.

---

## Zero-Leakage Mandate

ChronoGrid simulates realistic Day-Ahead Market (DAM) forecasting conditions. Any PR modifying feature generation, data ingestion, or model evaluation must adhere to:

1. **Information Boundary**: No observation newer than forecast gate closure ($T \le T_{\text{gate}}$) may enter feature sets.
2. **Shifted Statistics**: Rolling windows and autoregressive lags must enforce $\ge 24\text{h}$ lag bounds relative to delivery timestamps.
3. **Weather Governance**: Realised weather at delivery time is strictly prohibited as a predictive feature. Use only persistence weather ($T \le T_{\text{gate}}$) or genuine archived NWP forecasts.
4. **Adversarial Test Verification**: Ensure all 10 adversarial leakage checks pass:
   ```bash
   python -c "
   import sys; sys.path.insert(0, '.')
   import pandas as pd
   from src.features.engineer import FeatureEngineer
   df = pd.read_parquet('data/raw/genuine_smard_dwd_observations.parquet')
   fe = FeatureEngineer(min_lag_hours=24, weather_mode='persistence')
   fe.fit(df)
   res = fe.run_adversarial_leakage_suite(df)
   assert res['suite_status'] == 'ALL_TESTS_PASSED'
   print('Zero-Leakage Suite Passed')
   "
   ```

---

## Running Tests

All automated tests must pass before submitting a pull request:

```bash
pytest tests/ -v
```

---

## Commit Guidelines

ChronoGrid follows [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add new capability`
- `fix: resolve issue`
- `docs: update documentation`
- `test: add unit or integration tests`
- `perf: optimize runtime or memory`
- `refactor: restructure code without changing behavior`
