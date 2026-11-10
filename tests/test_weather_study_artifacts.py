from pathlib import Path

import pandas as pd

from scripts.audit_weather_v4_research import audit


def test_fixed_weather_replay_reconstructs_without_training():
    result = audit()
    assert result["status"] == "verified_without_retraining"
    assert result["decision"] == "NO_GO_FOR_PROMOTION"
    assert result["evaluation_hours"] == 817 and result["excluded_embargo_origins"] == 4


def test_dst_horizon_totals_and_actual_day_lengths():
    root = Path(__file__).resolve().parents[1] / "artifacts/experiments/weather_v4"
    origins = pd.read_csv(root / "per_origin_intervals.csv")
    horizons = pd.read_csv(root / "per_utc_horizon.csv")
    daily = origins[
        (origins["head"] == "primary")
        & (origins.method == "pooled_marginal_mad")
        & (origins.nominal == 0.9)
    ]
    assert len(daily) == 34 and daily.total.sum() == 817
    assert daily[daily.delivery_date == "2026-10-25"].total.iloc[0] == 25
    h = horizons[
        (horizons["head"] == "primary")
        & (horizons.method == "pooled_marginal_mad")
        & (horizons.nominal == 0.9)
    ]
    assert len(h) == 24 and h.covered.sum() == daily.covered.sum() == 455
    assert h[h.utc_delivery_hour == 22].total.iloc[0] == 35
    assert set(h[h.utc_delivery_hour != 22].total) == {34}
