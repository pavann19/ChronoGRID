# Provenance and limitations

I evaluated `data/raw/genuine_smard_dwd_observations.parquet`, SHA-256 `d02425d161333cbc067388bdaf639287c11f90164d91aee72da8ed3de198e817`. The saved run contains 13,607 rows spanning 2023-01-01T23:00:00+00:00 through 2024-07-21T21:00:00+00:00; these values and library versions are recorded in `artifacts/reports/benchmark_corrected/report.json`.

The load collector requests SMARD's German hourly grid-load series. The weather collector requests Open-Meteo's historical archive without pinning a source model in the request. I cannot substantiate the earlier assertion that these are direct observations from a named DWD station. I retain the original dataset filename, labels and bytes to preserve identity, and withdraw that provenance claim. Open-Meteo documents this product as historical reanalysis: [product documentation](https://open-meteo.com/en/docs/historical-weather-api).

The corrected evaluation is retrospective. It reuses 35 previously inspected local dates, with 5 expanding folds and a 24-hour embargo. The new mean errors are not interchangeable with the earlier fixed-training-cutoff experiment. I did not tune models against a fresh holdout, establish prospective reliability, or measure a managed production service.

Weather is persistence rather than an archived NWP vintage. A source's observation interval ending before the gate is necessary but does not prove that its value was actually published by then. Historical revisions, source availability delays and vintage identity remain unresolved. The external series is excluded from comparisons; see the [baseline limitation](evaluation.md).

The bootstrap resamples consecutive delivery-day blocks and quantifies variation on these reused observations. It cannot create missing seasons, establish deployment performance or validate a causal explanation. All point metrics and intervals come from the saved benchmark outputs; I do not claim statistical superiority from overlapping individual intervals.

The served artifacts retain compatibility guards and model/calibration bindings. The corrected benchmark remains retrospective. Further evaluation needs archived forecast vintages, untouched delivery dates and timestamped external forecasts.

Source attribution and licences are in [NOTICE](../NOTICE). The corrected protocol is described in [evaluation](evaluation.md).

The main benchmark retains the existing UTC calendar feature convention; the separately versioned local-calendar adapter uses Europe/Berlin.
