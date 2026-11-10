# Contributing

I review changes against the forecasting information boundary and reproducible evidence.
Start with the README and current architecture decisions, then use a separate branch.

Install with `make install`. Before proposing a change, run `make lint`, `make test`
and `make audit`. A benchmark change also needs `make benchmark` and saved hourly
outputs, fold boundaries, source identity and uncertainty intervals. Do not replace
frozen results or a model/calibration hash merely to make a check pass.

I distinguish local tests, clean-source builds, hosted CI and production operation.
Describe what actually ran and any remaining limitation. Accuracy claims must respect
the market gate and disclose reused evaluation dates.

Keep runtime databases, credentials and bulky historical outputs outside Git tracking.
The evidence-export script and history-cleanup note describe the optional archive.
A model promotion or public release requires its own explicit review.
