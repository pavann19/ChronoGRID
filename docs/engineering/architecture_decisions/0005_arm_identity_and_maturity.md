# ADR 0005 — origin arms and maturity windows

New multi-model run registrations require an explicit immutable arm-to-model checksum map. A single-model run can infer its only possible mapping. The original one-origin pilot registration held an allowed model set without per-arm pins; it remains preserved and readable as legacy evidence, but cannot receive new forecast commitments. This is a limitation of that pilot, not a silent retroactive correction. Its two arm/model assignments can still be inspected in the immutable records and original capture implementation commit.

Bootstrap samples are restricted to fully mature scheduled calendar days at the report cutoff. Missing past days remain zero-count calendar entries, preserving real gaps; future scheduled days are not artificial zero-observation samples. Overall hourly point/coverage summaries may include partially mature origins, so bootstrap scope is labeled separately. Dependence assumptions and sparse-sample limits still apply.
