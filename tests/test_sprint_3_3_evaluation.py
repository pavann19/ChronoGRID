# Repository scripts bootstrap the checkout before importing the installed package.
# ruff: noqa: E402
"""
Unit and regression tests for ChronoGrid Historical evaluationOut-of-Sample Evaluation and Benchmark.
Verifies data partition integrity, point forecast parity, conformal coverage guarantees,
Daily-Origin Block Bootstrap outputs, segregated synthetic degradation resilience,
per-origin and per-horizon granularity, and objective assessment scorecard.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

from scripts.run_sprint_3_3_evaluation import (
    compute_winkler_score,
)


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def benchmark_data():
    report_path = ROOT_DIR / "artifacts/evaluation_legacy/benchmark.json"
    assert report_path.exists(), (
        f"Benchmark report not found at {report_path}. Run scripts/run_sprint_3_3_evaluation.py first."
    )
    with open(report_path, "r") as f:
        return json.load(f)


def test_evaluation_partition_and_data_hygiene(benchmark_data):
    """Verify evaluation partition timestamps, row count, and artifact checksums."""
    meta = benchmark_data["benchmark_metadata"]
    part = meta["experiment_configuration"]["evaluation_partition"]
    assert part["total_hours"] == 840
    assert part["total_daily_origins"] == 35
    assert part["start_timestamp_utc"] == "2024-06-16 22:00:00+00:00"
    assert part["end_timestamp_utc"] == "2024-07-21 21:00:00+00:00"

    checksums = meta["artifact_checksums"]
    assert (
        checksums["production_champion_onnx"]
        == "1c28a97074fbc579f94eb3ab5ae72752bba97057c077eece647c2d1e5b8fcd91"
    )
    assert (
        checksums["production_fallback_onnx"]
        == "fa2913cf584e985e9b23d4e61ec082bb90ba93780d062eb8ed6ed910b45c6e45"
    )
    assert (
        checksums["research_primary_joblib"]
        == "352abc4f302a43163532a222701efb6d2b1bb2d99df3f33d047faa9029ce758b"
    )
    assert (
        checksums["research_fallback_joblib"]
        == "3c12480e7849bf14f44f5e4777dbe6d40edf7b22131f08b9fa19ae80603b21d9"
    )


def test_point_forecast_performance(benchmark_data):
    """Verify point forecasting accuracy and primary vs fallback advantage."""
    pts = benchmark_data["point_forecast_performance"]
    prim = pts["primary_weather_enabled"]
    fall = pts["fallback_load_only"]

    assert 1000.0 < prim["mae_mw"] < 1150.0, f"Primary MAE {prim['mae_mw']} outside expected range"
    assert 1000.0 < fall["mae_mw"] < 1150.0, f"Fallback MAE {fall['mae_mw']} outside expected range"
    assert prim["mae_mw"] < fall["mae_mw"], (
        "Primary model should outperform Fallback on point forecast MAE"
    )
    assert prim["rmse_mw"] < fall["rmse_mw"], (
        "Primary model should outperform Fallback on point forecast RMSE"
    )
    assert pts["delta_mae_mw"] > 0.0


def test_marginal_conformal_coverage_guarantees(benchmark_data):
    """Verify that empirical marginal coverage meets or exceeds nominal targets."""
    conf = benchmark_data["conformal_evaluations"]

    # Primary Static
    p_stat_90 = conf["primary_static"]["marginal_90"]
    p_stat_95 = conf["primary_static"]["marginal_95"]
    assert p_stat_90["empirical_marginal_coverage"] >= 0.90
    assert p_stat_95["empirical_marginal_coverage"] >= 0.95
    assert p_stat_95["mean_interval_width_mw"] > p_stat_90["mean_interval_width_mw"]

    # Fallback Static
    f_stat_90 = conf["fallback_static"]["marginal_90"]
    f_stat_95 = conf["fallback_static"]["marginal_95"]
    assert f_stat_90["empirical_marginal_coverage"] >= 0.90
    assert f_stat_95["empirical_marginal_coverage"] >= 0.95
    assert f_stat_95["mean_interval_width_mw"] > f_stat_90["mean_interval_width_mw"]

    # Delayed-Feedback ACI
    p_aci_90 = conf["primary_delayed_feedback_aci"]["marginal_90"]
    f_aci_90 = conf["fallback_delayed_feedback_aci"]["marginal_90"]
    assert p_aci_90["empirical_marginal_coverage"] >= 0.90
    assert f_aci_90["empirical_marginal_coverage"] >= 0.90


def test_simultaneous_daily_coverage_guarantees(benchmark_data):
    """Verify simultaneous 24h vector band properties under Sup-Norm Conformal."""
    conf = benchmark_data["conformal_evaluations"]

    p_sim_90 = conf["primary_static"]["simultaneous_90"]
    p_sim_95 = conf["primary_static"]["simultaneous_95_exploratory"]

    # Simultaneous coverage must be valid (>= 0.90)
    assert p_sim_90["empirical_simultaneous_coverage"] >= 0.90
    assert p_sim_95["empirical_simultaneous_coverage"] >= 0.95

    # Simultaneous bands must be wider than marginal intervals to cover entire 24h vectors
    p_marg_90 = conf["primary_static"]["marginal_90"]
    assert p_sim_90["mean_interval_width_mw"] > p_marg_90["mean_interval_width_mw"]
    assert p_sim_95["mean_interval_width_mw"] > p_sim_90["mean_interval_width_mw"]


def test_daily_origin_block_bootstrap_consistency(benchmark_data):
    """Verify Daily-Origin Block Bootstrap confidence intervals and block length sensitivity."""
    p_boot = benchmark_data["conformal_evaluations"]["primary_static"]["marginal_90"][
        "block_bootstrap"
    ]

    for b in [1, 2, 3, 5]:
        b_key = f"block_length_{b}"
        assert b_key in p_boot
        boot = p_boot[b_key]
        assert boot["n_bootstraps"] == 10000
        ci = boot["marginal_coverage"]["ci_95"]
        assert 0.85 < ci[0] < ci[1] < 1.0
        # Check that empirical mean is within the CI
        mean_cov = boot["marginal_coverage"]["mean"]
        assert ci[0] <= mean_cov <= ci[1]


def test_segregated_synthetic_degradation_experiments(benchmark_data):
    """Verify that synthetic degradation experiments are properly segregated and resilient."""
    assert "synthetic_degradation_experiments" in benchmark_data
    stress = benchmark_data["synthetic_degradation_experiments"]

    assert "synthetic_regime_dropout_10days" in stress
    dropout = stress["synthetic_regime_dropout_10days"]

    assert dropout["experiment_type"] == "SYNTHETIC_FAULT_INJECTION"
    assert dropout["overall_empirical_coverage"] >= 0.90
    assert dropout["outage_period_coverage"] >= 0.90
    assert dropout["fallback_switchover_count"] == 10


def test_per_origin_and_per_horizon_completeness(benchmark_data):
    """Verify that per-origin (N=35) and per-horizon (h=24) metrics are fully populated."""
    origins = benchmark_data["per_origin_results"]["primary_static_marginal_90"]
    assert len(origins) == 35
    for orig in origins:
        assert 1 <= orig["origin_index"] <= 35
        assert 0.0 <= orig["marginal_coverage_rate"] <= 1.0
        assert orig["mean_interval_width_mw"] > 0.0

    horizons = benchmark_data["per_horizon_metrics"]["primary_static_marginal_90"]
    assert len(horizons) == 24
    for h in range(24):
        h_key = f"h{h:02d}"
        assert h_key in horizons
        assert horizons[h_key]["total_days_count"] == 35
        assert 0.0 <= horizons[h_key]["empirical_coverage"] <= 1.0


def test_chronological_aci_event_logs_structure(benchmark_data):
    """Verify chronological ACI event logs under 48h settlement delay."""
    logs = benchmark_data["chronological_aci_event_logs"]["primary_head"]
    assert len(logs) > 0

    settled_events = [e for e in logs if e["event_type"] == "GROUND_TRUTH_SETTLED"]
    assert len(settled_events) == 33  # 35 origins - 2 day delay = 33 settled origins
    for ev in settled_events:
        assert ev["effective_delay_days"] == 2
        assert "alpha_before" in ev
        assert "alpha_after" in ev
        assert ev["status"] == "PROCESSED"


def test_objective_assessments_scorecard(benchmark_data):
    """Verify objective scorecard statuses."""
    assessments = benchmark_data["objective_assessments"]
    assert len(assessments) == 8

    # Ensure rigorous statuses
    assert assessments["objective_1_point_forecast_baseline_parity"]["status"] == "PASS"
    assert assessments["objective_2_marginal_conformal_coverage"]["status"] == "PASS"
    assert assessments["objective_3_simultaneous_daily_vector_coverage_90"]["status"] == "PASS"
    assert (
        assessments["objective_4_simultaneous_daily_vector_coverage_95_exploratory"]["status"]
        == "CONDITIONAL"
    )
    assert assessments["objective_5_delayed_feedback_aci_stability"]["status"] == "PASS"
    assert assessments["objective_6_independent_fallback_calibration"]["status"] == "PASS"
    assert assessments["objective_7_block_bootstrap_uncertainty_quantification"]["status"] == "PASS"
    assert assessments["objective_8_synthetic_degradation_fault_tolerance"]["status"] == "PASS"


def test_winkler_score_mathematical_properties():
    """Verify that Winkler Score properly penalizes miscoverage and equals width when covered."""
    y = np.array([50.0, 40.0, 70.0])
    lower = np.array([45.0, 45.0, 45.0])
    upper = np.array([65.0, 65.0, 65.0])
    alpha = 0.10

    scores = compute_winkler_score(y, lower, upper, alpha)
    assert np.isclose(scores[0], 20.0)
    assert np.isclose(scores[1], 120.0)
    assert np.isclose(scores[2], 120.0)
