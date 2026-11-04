# ADR 0001 — extend research with a separate transactional operations path

Status: accepted for local implementation, 2026-11-04.

Problem: closure files are hash-pinned, while the old evidence helper can leave partial exclusive-create files and has no run/arm uniqueness. Legacy inference mixes incompatible production model/research calibrator identities. Rewriting old evidence would break provenance.

Decision: new src/operations package and operations API, with SQLite WAL + synchronous FULL, BEGIN IMMEDIATE writers, foreign keys, immutable SQL tables, content hashes and a transactionally chained journal. Register protocol/run/origin/arm before issuance. Keep immutable source bytes and actual revisions separately. Use SQLite online backup, validate exported manifests and chains; do not copy a live WAL database file. File owners can still rewrite SQLite: this is tamper evidence under a trusted local operator, not WORM or HA.

Alternatives: file-per-record is too weak for cross-file transactions; PostgreSQL/object-lock is reasonable for multi-host deployment but not justified by a demonstrated local workload. A remote independent timestamp authority adds chronology proof, not atomic local storage. Integrate RFC3161 verification locally; no arbitrary remote service submission or paid provisioning is enabled by default.

Compatibility: old endpoint remains historical/research code, not the prospective capture API. The new interface does not expose it as a trustworthy calibrated prospective forecast. Existing research artifacts are unchanged. Model/feature-policy changes invalidate old interval assumptions even if the predictor binary matches.

Validation: competing writers, killed transaction, reload/backup/export, corruption, source availability, duplicate origin/arm, timestamp/actual linkage, API/UI empty-data states. Independent receipts are accepted only after signature/chain/imprint/time validation against explicitly configured trust, never from caller-provided boolean flags.

Sources: [SQLite WAL](https://sqlite.org/wal.html), [transactions](https://sqlite.org/lang_transaction.html), [backup API](https://sqlite.org/backup.html), [RFC3161](https://www.rfc-editor.org/rfc/rfc3161.html). Source behavior must be checked against actual local versions; these references are design rationale, not validation results.
