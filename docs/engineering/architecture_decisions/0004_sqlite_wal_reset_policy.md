# ADR 0004 — safe journal mode on bundled SQLite versions

Status: accepted and implemented during final validation, 2026-11-04.

The host bundles SQLite 3.49.1 and the pinned Python 3.13.18 container bundles SQLite 3.46.1. Upstream documents a rare WAL-reset race fixed in 3.51.3 and later, with explicit 3.44.6/3.50.7 backports. Passing finite concurrency tests cannot establish absence of this race.

Decision: automatically use WAL only for known-safe version ranges; otherwise use SQLite DELETE rollback journaling with synchronous FULL and BEGIN IMMEDIATE. Never silently permit an unsafe WAL override. A vendor backport without a recognizable fixed version is treated conservatively. Startup migration of the local runtime database changes storage journal mode, not any immutable prediction payload or event hash. If another writer prevents the mode change, startup fails rather than proceeding unsafely.

Tradeoff: rollback journaling can block readers during commits and may reduce concurrency. Repeat local bounded benchmarks and competing-writer/restart/export tests; record mode and actual SQLite version in integrity output. Original WAL benchmark remains preserved as an earlier version, not presented as the final configuration's performance.

Scope: single-host local research remains unchanged; no multi-host HA claim. The historic local pilot was recorded under earlier WAL configuration; its export and hash chain remain preserved. Future writes use the conservative policy.

Source: [SQLite WAL-reset documentation](https://sqlite.org/wal.html#walresetbug).
