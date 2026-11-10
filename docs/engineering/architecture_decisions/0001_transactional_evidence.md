# ADR: Transactional local evidence

Status: current implementation reviewed during portfolio cleanup.

I commit forecast records, snapshot identities and event records in a transaction. Duplicate or conflicting identities fail rather than silently replacing an earlier forecast.

A local hash and timestamp support reproducibility; they do not provide independent timestamp attestation or immutable external custody.
