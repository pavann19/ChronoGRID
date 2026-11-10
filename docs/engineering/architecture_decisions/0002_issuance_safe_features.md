# ADR: Completed-hour gate availability

Status: current implementation reviewed during portfolio cleanup.

I use the Berlin previous-day noon gate and compare each observed feature dependency at the end of its source hour. The session audit reproduces the old post-gate finding in artifacts/reports/portfolio_cleanup/phase1_gate_audit.json. I default FeatureEngineer to the corrected lag policy and require allow_post_gate_lags for the old comparison.

I keep the old fixed-model experiment distinct from the new expanding-fold benchmark. Only the new run supplies current headline metrics; event-time checks do not establish publication-time availability.
