# Changelog

## 0.2.0 - 2026-11-10

I corrected the day-ahead information boundary, added a completed-hour gate audit,
replaced future-dependent filling, and reran the benchmark on identical delivery hours.
I authenticated forecast routes, bound served artifacts to a checksum registry, and
renamed the Python package to `chronogrid`. I consolidated dependencies and CI,
normalized source line endings, and separated historical working notes from current docs.
Validation and remaining limits are recorded in `artifacts/reports/cleanup_report.md`.

## 1.0.0 - 2026-11-03 (Superseded)

The original release used post-gate load features. Its day-ahead accuracy and
information-boundary claims are withdrawn. The original Git tag remains unchanged for inspection.
The corrected development release is 0.2.0; this does not imply production operation.
