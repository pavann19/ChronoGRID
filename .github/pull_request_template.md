## Description

Brief summary of the changes introduced by this pull request.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement / Refactoring

## Zero-Leakage & Information Boundary Checklist

- [ ] This PR does NOT introduce future target or weather leakage into model features.
- [ ] All rolling statistics and lags respect the minimum 24-hour information boundary ($T \le T_{\text{gate}}$).
- [ ] The adversarial zero-leakage test suite has been executed and passes 10/10.

## Testing & Quality

- [ ] All unit and integration tests pass (`pytest tests/ -v`).
- [ ] New tests have been added covering the new functionality.
- [ ] Linter checks pass (`ruff check src tests`).
