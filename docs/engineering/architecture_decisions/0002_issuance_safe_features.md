# ADR 0002 — fixed-cutoff features and honest model selection

Status: implemented as separate retrospective experiment, 2026-11-04.

The new diagnostic finds that 420/840 historical hourly targets have lag-24 observation intervals ending after a fixed D-1 noon gate, even with zero publication delay. A delivery-relative lag test is insufficient proof of a fixed-origin information boundary. Frozen reports are preserved; their numerical aggregates are not altered.

Decision: new experiment artifacts/experiments/issuance_v2 uses lag 48/72/168, rolling windows shifted 48 hours and weather persistence shifted 48 hours. Same HGB hyperparameters for primary/fallback; no hyperparameter search. Train through 2024-03-31, calibrate on 74 complete German local dates 2024-04-03 through 2024-06-15, evaluate the previously inspected 35 dates. Original model/calibration files are untouched. Training-only constant removal is recorded in metadata.

Benefits: complete vector features can be generated from information before the gate without filling future demand with observed labels or invented defaults. New adapter checks exact features, finite inputs and model/metadata/calibration identities. Alternative recursive filling changes feature distributions and complicates calibration; lag48 is simpler to audit. It does not prove source publication availability; the live collector separately preserves first-seen snapshots.

Negative result retained: V2 primary/fallback MAE 1234.332679/1236.869729 MW versus weekly persistence 1225.0375 MW. Paired weather difference 2.537050 MW, two-day-block interval [-64.950649,84.199416] MW; inconclusive. No winner promotion. Old 1068.72 MW result is not compared as a like-for-like fixed-cutoff prospective improvement.

Prospective local pilot can record weekly persistence without intervals and a separately named V2 load-only arm with explicitly research-only static intervals. No old calibration reused for new model identities. Weather-vintage collection, fresh independent seasons, and future interval reliability remain unverified. The new experiment is historical replay on reused dates, not a new independent holdout.

References: [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api), [historical forecast](https://open-meteo.com/en/docs/historical-forecast-api), [previous runs](https://open-meteo.com/en/docs/previous-runs-api). These are different source products; source retrieval today is not a historical availability receipt.
