# ADR 0006 — transactional delayed adaptive feedback

Status: accepted for implementation after sealed checkpoint engineering_v1. Old ACI code, calibration, labels and experiments stay unchanged.

Gap: the original file-backed ACI manager trusts caller-supplied delay days, permits replacing pending bands, and uses an any-hour-failed signal even when its endpoint is described as marginal. New work must not rewrite those historical results or assume theoretical validity of a daily batched, clipped, delayed variant.

Decision: append-only stream configuration per run/arm/head in the shared journal. Pin base calibration/model/feature policy and predeclared target, gamma, clipping, minimum feedback delay and label deadline. Compute minimum delay from the actual delivery interval end; it is a study policy, not an asserted universal SMARD publication rule. Feedback references immutable issued forecasts and first observed actual-record hashes; never accepts caller-supplied coverage, delay, bounds or actual vectors.

Process scheduled origins chronologically. Missing forecasts/intervals are explicit skips; incomplete labels wait until the registered deadline, then skip without inventing error. Late revisions do not roll back learned state. A skipped origin cannot later be silently replayed as if feedback occurred earlier. Duplicate settlement returns the original recorded transition. Primary/fallback and distinct target arms have separate streams.

Marginal-hourly feedback uses the missed-hour fraction for a complete delivery vector. Simultaneous feedback uses the any-hour-missed indicator. The daily batch, clipping and delayed feedback are explicit algorithm choices, not an unconditional transfer of a paper's guarantee. DST vectors use their actual 23/24/25 lengths. Parameters are frozen at registration; no tuning against historical holdout outcomes.

Issue against a versioned alpha snapshot. Commit compares the snapshot's basis with the latest transactional stream state and rejects stale computations; feedback and forecast identity decisions serialize across local processes. Adaptive intervals use the original frozen base-score pool at the registered alpha; base files are never modified. A new run/arm is required; do not retrofit adaptation into the already committed pilot.

Validation: hand-calculated error signals, actual timestamp guards, chronological ordering/gaps/deadlines, duplicate/revision behavior, isolated heads, concurrent settlement, restart/export/restore, stale issuance basis rejection and a real V2 model capture on synthetic isolated history. No prospective performance or theoretical guarantee is claimed from these tests.

Reference: [Gibbs and Candes, Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170). This implementation's additional batching/delay/clipping/missing-data choices require their own empirical validation.
