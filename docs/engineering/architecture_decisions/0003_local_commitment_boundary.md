# ADR 0003 — local pilot versus independent chronology

Status: implemented locally; external authority receipt not obtained.

Two genuine pre-delivery forecast records were committed for 2026-11-05 at 2026-11-04 06:27:54.327936 UTC (weekly) and 06:27:59.125523 UTC (V2 fallback), before the German noon gate. Run definition preceded issuance; both share a raw-source-backed SMARD snapshot. No actuals exist at this checkpoint; initial report has zero scored observations and 24 not-mature hours per arm, with null performance metrics.

The local journal hash head and verified export/backup establish current content and transaction history under a trusted local operator. They do not establish independently witnessed time. Local RFC3161 integration is exercised with an ephemeral test CA, never mislabeled as a public independent authority. External TSA submission would disclose hashes; no remote submission or paid authority has been authorized or attempted.

Decision: retain tiers `historical_replay`, `synthetic_fault`, and `prospective` with explicit `independent_timestamp=false` on the local pilot. Any independent evidence claim requires retained nonce-bearing request, signed token, pinned trust anchor identity, matching forecast hash and appropriate generation time. No success is inferred from a receipt-shaped JSON flag. Trust/revocation policy and external witness infrastructure remain unresolved operational requirements.
