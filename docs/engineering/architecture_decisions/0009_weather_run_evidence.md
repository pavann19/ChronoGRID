# ADR 0009 — Single weather runs and delivery alignment

Status: accepted for a separate input-evidence adapter, not primary model promotion.

The current ICON-seamless snapshot has no identifiable initialization time and can splice runs. Open-Meteo now offers a dedicated Single Runs API with an explicit initialization parameter. Its documentation distinguishes initialization from publication; archive retrieval today does not prove availability at an old cutoff. Most non-ECMWF runs start in April 2026, so this source cannot silently replace the frozen 2023–2024 weather inputs.

Use one explicit ICON-EU model and a predeclared 00 UTC run policy. Preserve request, raw response and normalized snapshot hashes independently. The API body does not echo initialization/model identity in the successful probe; label identity as the provider's requested-run contract, not independent verification. Keep provider and local-clock trust explicit. First-seen capture is the only operational availability evidence; generationtime_ms and HTTP Date are not issuance receipts.

Record variable semantics. Temperature, humidity and wind are instantaneous at valid time. Direct radiation is the mean of the preceding hour. Align the latter at t+1h to the German load delivery interval [t,t+1h); do not silently join all four at t. Preserve missing values. Require every 23/24/25-hour delivery vector, before-gate capture, freshness, geographic consistency, exact units and source-chain verification before considering it usable.

This source is a Frankfurt grid-point prototype, not a national weather aggregate. A new model/calibration must train against matching forecast variables, locations and run-selection policy. No interpolation into the persistence-trained primary, synthetic weather relabeling, performance claim or pilot replacement. Archive retrieval remains historical source evidence even when the requested run is old. Collecting fresh runs can seed a genuinely prospective study; insufficient labels remain a real limitation.

Sources: [Single Runs API](https://open-meteo.com/en/docs/single-runs-api), [availability metadata](https://open-meteo.com/en/docs/model-updates), [hourly weather variable definitions](https://open-meteo.com/en/docs).
