# ADR 0008 — Local calendar experiment, separate from frozen V2

Status: accepted for a separately versioned research experiment.

The frozen temporal transform uses UTC hour/date and a holiday table ending in 2025. Its guard correctly rejects unsupported holidays but prevents future holiday issuance. Replacing that table in place would change the feature/calibration contract of existing models.

Introduce a load-only V3 policy using Europe/Berlin hour, weekday, month, day of year and nationwide holiday dates. Keep lag offsets and rolling windows in elapsed UTC hours, including on DST days. The two repeated autumn hours share local clock features; their timestamps remain distinct. State-specific and school holidays are explicitly excluded. Calendar rules are a versioned modelling policy, not an assertion that future legislation cannot change.

Train one new HGB model with the same fixed settings, source data, training cutoff, calibration dates and reused evaluation dates as V2. No search, tuning, winner promotion or recalibration of existing models. Fit residual scales on training only; group calibration residuals by actual German local delivery dates rather than assuming all days contain 24 hours. Retain UTC horizon residual scaling for compatibility, stated explicitly. Compare point errors with frozen V2 fallback and weekly persistence on exactly the same timestamps, with paired daily block uncertainty.

Accept: known 2026 holidays, local midnight boundaries, spring/autumn DST, naive timestamp rejection, unchanged UTC lag values, no future-history dependence, model/calibration identity, capture of a future Christmas vector on isolated synthetic input, and old manifest/archive verification. Historical replay is insufficient for prospective model promotion. Never modify the V2 experiment or existing pilot.

Primary reference: [Bundesbank 2026 holiday schedule](https://www.bundesbank.de/resource/blob/764518/f839f63a78a1fec2d3176b0b64044216/mL/feiertage-in-deutschland-1-data.docx). The nationwide intersection excludes regional and additional central-bank non-business days.
