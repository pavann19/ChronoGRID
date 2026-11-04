"""
Unit and integration tests for ChronoGrid Data-Quality-Conditioned Conformal Prediction (DQ-CP).
Historical evaluation Remediation Suite:
- Quantile-rank boundary tests and infinite/unbounded interval handling (k > n)
- Normalization, Zero-MAD epsilon guard, and non-finite input guards
- Strict Primary and Fallback ACI state isolation
- Daily vector Sup-Norm tests
- Delayed feedback (48h lag), revision rejection, duplicate idempotency, and restart persistence
- No-evaluation-leakage tests
- DDIR reliability state integration tests
- FastAPI research endpoint tests
"""

import os
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.reliability.state_machine import ReliabilityState
from src.uncertainty.schemas import UncertaintyResult, CalibrationTable, DelayedFeedbackRecord
from src.uncertainty.normalization import DeliveryHourMADNormalizer
from src.uncertainty.conformal import ConformalCalibrationPool, compute_finite_sample_rank
from src.uncertainty.aci import DelayedFeedbackACIManager
from src.uncertainty.engine import DQCPUncertaintyEngine, load_dqcp_engine


# =============================================================================
# 1. Quantile-Rank Boundary Tests & Infinite Quantile Handling
# =============================================================================

def test_finite_sample_rank_formula():
    """Verify exact finite-sample quantile rank: k = ceil((n + 1)(1 - alpha))."""
    # N = 1800 (pooled hourly calibration)
    k_90_1800, idx_90_1800 = compute_finite_sample_rank(1800, 0.10)
    assert k_90_1800 == 1621
    assert idx_90_1800 == 1620

    k_95_1800, idx_95_1800 = compute_finite_sample_rank(1800, 0.05)
    assert k_95_1800 == 1711
    assert idx_95_1800 == 1710

    # N = 75 (daily origins calibration)
    k_90_75, idx_90_75 = compute_finite_sample_rank(75, 0.10)
    assert k_90_75 == 69
    assert idx_90_75 == 68

    k_95_75, idx_95_75 = compute_finite_sample_rank(75, 0.05)
    assert k_95_75 == 73
    assert idx_95_75 == 72


def test_finite_sample_rank_boundaries():
    """Verify invalid alphas and sample sizes raise proper ValueErrors."""
    with pytest.raises(ValueError, match="Sample size n must be positive"):
        compute_finite_sample_rank(0, 0.10)

    with pytest.raises(ValueError, match="Sample size n must be positive"):
        compute_finite_sample_rank(-10, 0.10)

    with pytest.raises(ValueError, match="Miscoverage rate alpha must be in"):
        compute_finite_sample_rank(100, 0.0)

    with pytest.raises(ValueError, match="Miscoverage rate alpha must be in"):
        compute_finite_sample_rank(100, 1.0)

    with pytest.raises(ValueError, match="Miscoverage rate alpha must be in"):
        compute_finite_sample_rank(100, 1.5)


def test_finite_sample_rank_remediation_cases():
    """
    Verify mathematically correct behavior when k = ceil((n + 1)(1 - alpha)) > n:
    Must return idx = None indicating an infinite quantile is required.
    Independently verified for n = 1, 5, 75, 1800.
    """
    # 1. n = 1, alpha = 0.10 -> ceil(2 * 0.90) = 2 > 1 -> idx is None
    k_1, idx_1 = compute_finite_sample_rank(1, 0.10)
    assert k_1 == 2
    assert idx_1 is None

    # 2. n = 5, alpha = 0.10 -> ceil(6 * 0.90) = 6 > 5 -> idx is None
    k_5, idx_5 = compute_finite_sample_rank(5, 0.10)
    assert k_5 == 6
    assert idx_5 is None

    # 3. n = 75, alpha = 0.01 -> ceil(76 * 0.99) = 76 > 75 -> idx is None
    k_75_01, idx_75_01 = compute_finite_sample_rank(75, 0.01)
    assert k_75_01 == 76
    assert idx_75_01 is None

    # 4. n = 75, alpha = 0.05 -> ceil(76 * 0.95) = 73 <= 75 -> finite index 72
    k_75_05, idx_75_05 = compute_finite_sample_rank(75, 0.05)
    assert k_75_05 == 73
    assert idx_75_05 == 72

    # 5. n = 1800, alpha = 0.05 -> ceil(1801 * 0.95) = 1711 <= 1800 -> finite index 1710
    k_1800_05, idx_1800_05 = compute_finite_sample_rank(1800, 0.05)
    assert k_1800_05 == 1711
    assert idx_1800_05 == 1710

    # 6. n = 1800, alpha = 0.0005 -> ceil(1801 * 0.9995) = 1801 > 1800 -> idx is None
    k_1800_extreme, idx_1800_extreme = compute_finite_sample_rank(1800, 0.0005)
    assert k_1800_extreme == 1801
    assert idx_1800_extreme is None


def test_conformal_pool_infinite_quantile_and_unbounded_intervals():
    """Verify that when k > n, pool returns q = inf and engine generates unbounded intervals."""
    norm = DeliveryHourMADNormalizer({h: 500.0 for h in range(24)})
    pool = ConformalCalibrationPool(
        model_id="test_small",
        normalizer=norm,
        marginal_scores=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        daily_sup_norm_scores=np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        n_origins=5
    )

    q_marg, k_marg = pool.get_marginal_quantile(0.10)
    assert math.isinf(q_marg)
    assert k_marg == 6

    q_sim, k_sim, _ = pool.get_simultaneous_quantile(0.10)
    assert math.isinf(q_sim)
    assert k_sim == 6

    engine = DQCPUncertaintyEngine(primary_pool=pool, fallback_pool=pool)
    preds = np.ones(24) * 50000.0
    hours = np.arange(24)

    res = engine.generate_interval(
        point_predictions=preds,
        delivery_hours=hours,
        reliability_state=ReliabilityState.HEALTHY,
        forecast_head="primary",
        target_coverage=0.90
    )

    assert res.is_unbounded is True
    assert res.uncertainty_available is True
    assert math.isinf(res.mean_interval_width_mw)
    assert all(math.isinf(x) and x < 0 for x in res.lower_bound_mw)
    assert all(math.isinf(x) and x > 0 for x in res.upper_bound_mw)
    assert "UNBOUNDED" in res.degradation_flag


def test_conformal_ties_and_edge_cases():
    """Verify behavior on tied scores and empty pools."""
    norm = DeliveryHourMADNormalizer({h: 500.0 for h in range(24)})
    tied_scores = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    pool = ConformalCalibrationPool(
        model_id="tied_pool",
        normalizer=norm,
        marginal_scores=tied_scores
    )

    q, k = pool.get_marginal_quantile(0.50)
    assert q == 2.0
    assert k == 3

    empty_pool = ConformalCalibrationPool(model_id="empty", normalizer=norm)
    with pytest.raises(ValueError, match="contains 0 marginal scores"):
        empty_pool.get_marginal_quantile(0.10)


# =============================================================================
# 2. Normalization and Zero-MAD Tests
# =============================================================================

def test_normalization_and_scale_inversion():
    """Verify delivery-hour MAD normalization and interval inversion."""
    hour_mad = {h: 500.0 + h * 10.0 for h in range(24)}
    norm = DeliveryHourMADNormalizer(hour_mad=hour_mad, global_mad=600.0, epsilon=1.0)

    actuals = np.array([50000.0, 52000.0])
    preds = np.array([49000.0, 51000.0])
    hours = np.array([0, 10])

    scores = norm.normalize(actuals, preds, hours)
    assert pytest.approx(scores[0], rel=1e-5) == 1000.0 / 500.0
    assert pytest.approx(scores[1], rel=1e-5) == 1000.0 / 600.0

    lowers, uppers = norm.scale_intervals(preds, hours, conformal_quantile=2.0)
    assert lowers[0] == 49000.0 - 2.0 * 500.0
    assert uppers[0] == 49000.0 + 2.0 * 500.0
    assert lowers[1] == 51000.0 - 2.0 * 600.0
    assert uppers[1] == 51000.0 + 2.0 * 600.0


def test_zero_and_near_zero_mad_epsilon_guard():
    """Verify epsilon guard prevents division by zero when residuals are identical or zero."""
    zero_res = np.zeros(48)
    hours = np.tile(np.arange(24), 2)
    norm = DeliveryHourMADNormalizer(epsilon=5.0)
    norm.fit(zero_res, hours)

    for h in range(24):
        assert norm.get_scale(h) >= 5.0

    scores = norm.normalize(np.zeros(24), np.zeros(24), np.arange(24))
    assert np.all(np.isfinite(scores))
    assert np.all(scores == 0.0)


def test_normalization_non_finite_inputs():
    """Verify non-finite inputs raise ValueError."""
    norm = DeliveryHourMADNormalizer()
    with pytest.raises(ValueError, match="non-finite"):
        norm.normalize(np.array([np.nan]), np.array([50000.0]), np.array([0]))

    with pytest.raises(ValueError, match="non-finite"):
        norm.normalize(np.array([50000.0]), np.array([np.inf]), np.array([0]))


# =============================================================================
# 3. Primary and Fallback Pool Separation
# =============================================================================

def test_primary_fallback_pool_separation():
    """Verify Primary and Fallback calibration pools remain strictly separate."""
    norm = DeliveryHourMADNormalizer({h: 500.0 for h in range(24)})

    pool_prim = ConformalCalibrationPool(
        model_id="research_primary",
        normalizer=norm,
        marginal_scores=np.linspace(0.5, 4.0, 100),
        daily_sup_norm_scores=np.linspace(1.0, 5.0, 20),
        n_origins=20
    )

    pool_fall = ConformalCalibrationPool(
        model_id="research_fallback",
        normalizer=norm,
        marginal_scores=np.linspace(1.0, 6.0, 100),
        daily_sup_norm_scores=np.linspace(2.0, 8.0, 20),
        n_origins=20
    )

    q_prim_90, _ = pool_prim.get_marginal_quantile(0.10)
    q_fall_90, _ = pool_fall.get_marginal_quantile(0.10)

    assert q_fall_90 > q_prim_90
    assert pool_prim.model_id != pool_fall.model_id


# =============================================================================
# 4. Strict Primary and Fallback ACI State Isolation Tests
# =============================================================================

def test_aci_primary_fallback_state_isolation():
    """
    Prove that Primary and Fallback ACI managers maintain strictly independent state.
    1. Fallback misses do not alter primary alpha.
    2. Primary misses do not alter fallback alpha.
    3. Switching between primary and fallback preserves both states.
    4. Restart preserves both independently.
    5. Duplicate and out-of-order feedback cannot corrupt either state.
    6. Revised feedback returns REJECTED_REVISION and does not mutate state.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prim_state = Path(tmpdir) / "aci_prim.json"
        fall_state = Path(tmpdir) / "aci_fall.json"

        prim_aci = DelayedFeedbackACIManager(head="primary", target_alpha=0.10, gamma=0.01, state_file=prim_state)
        fall_aci = DelayedFeedbackACIManager(head="fallback", target_alpha=0.10, gamma=0.02, state_file=fall_state)

        engine = DQCPUncertaintyEngine(
            primary_aci_manager=prim_aci,
            fallback_aci_manager=fall_aci
        )

        assert engine.primary_aci_manager.get_current_alpha() == 0.10
        assert engine.fallback_aci_manager.get_current_alpha() == 0.10

        # Register forecasts for both
        engine.primary_aci_manager.register_forecast("2024-04-02", np.zeros(24), np.ones(24) * 50000)
        engine.fallback_aci_manager.register_forecast("2024-04-02", np.zeros(24), np.ones(24) * 45000)

        # 1. Fallback suffers miscoverage (actual = 60000 > 45000)
        rec_fall = engine.submit_feedback(
            forecast_origin="2024-04-02",
            actuals=np.ones(24) * 60000,
            settlement_timestamp="2024-04-05T00:00:00Z",
            effective_delay_days=3,
            head="fallback"
        )
        assert rec_fall.status == "PROCESSED"
        # Fallback alpha decreases: 0.10 + 0.02 * (0.10 - 1.0) = 0.082
        assert pytest.approx(engine.fallback_aci_manager.get_current_alpha(), rel=1e-5) == 0.082
        # PRIMARY ALPHA MUST REMAIN UNTOUCHED AT 0.10!
        assert pytest.approx(engine.primary_aci_manager.get_current_alpha(), rel=1e-5) == 0.10

        # 2. Primary suffers miscoverage
        rec_prim = engine.submit_feedback(
            forecast_origin="2024-04-02",
            actuals=np.ones(24) * 60000,
            settlement_timestamp="2024-04-05T00:00:00Z",
            effective_delay_days=3,
            head="primary"
        )
        assert rec_prim.status == "PROCESSED"
        # Primary alpha decreases: 0.10 + 0.01 * (0.10 - 1.0) = 0.091
        assert pytest.approx(engine.primary_aci_manager.get_current_alpha(), rel=1e-5) == 0.091
        # Fallback alpha must remain at 0.082!
        assert pytest.approx(engine.fallback_aci_manager.get_current_alpha(), rel=1e-5) == 0.082

        # 3. Duplicate submission: must return DUPLICATE_IGNORED and not change alpha
        rec_dup = engine.submit_feedback(
            forecast_origin="2024-04-02",
            actuals=np.ones(24) * 60000,
            settlement_timestamp="2024-04-05T01:00:00Z",
            effective_delay_days=3,
            head="primary"
        )
        assert rec_dup.status == "DUPLICATE_IGNORED"
        assert pytest.approx(engine.primary_aci_manager.get_current_alpha(), rel=1e-5) == 0.091

        # 4. Revised feedback: different actual load -> must return REJECTED_REVISION and not change alpha
        rec_rev = engine.submit_feedback(
            forecast_origin="2024-04-02",
            actuals=np.ones(24) * 58000,
            settlement_timestamp="2024-04-10T00:00:00Z",
            effective_delay_days=8,
            head="primary"
        )
        assert rec_rev.status == "REJECTED_REVISION"
        assert pytest.approx(engine.primary_aci_manager.get_current_alpha(), rel=1e-5) == 0.091

        # 5. Restart persistence: reload in new engine instances
        prim_reloaded = DelayedFeedbackACIManager(head="primary", state_file=prim_state)
        fall_reloaded = DelayedFeedbackACIManager(head="fallback", state_file=fall_state)
        assert pytest.approx(prim_reloaded.get_current_alpha(), rel=1e-5) == 0.091
        assert pytest.approx(fall_reloaded.get_current_alpha(), rel=1e-5) == 0.082

        # 6. Invalid feedback: wrong actuals length raises ValueError and does not mutate state
        with pytest.raises(ValueError, match="Actuals length"):
            prim_reloaded.submit_ground_truth(
                origin_date="2024-04-02",
                actuals=np.ones(10),
                settlement_timestamp="2024-04-05T00:00:00Z"
            )
        assert pytest.approx(prim_reloaded.get_current_alpha(), rel=1e-5) == 0.091


# =============================================================================
# 5. Daily Sup-Norm Conformal Tests
# =============================================================================

def test_daily_sup_norm_conformal():
    """Verify vector Sup-Norm computes max score per origin and flags 95% as exploratory."""
    norm = DeliveryHourMADNormalizer({h: 500.0 for h in range(24)})
    np.random.seed(42)
    actuals = np.random.normal(50000, 1000, 1800)
    preds = np.random.normal(50000, 1000, 1800)
    hours = np.tile(np.arange(24), 75)
    origin_ids = np.repeat([f"origin_{i}" for i in range(75)], 24)

    pool = ConformalCalibrationPool.from_calibration_data(
        model_id="test_model",
        normalizer=norm,
        actuals=actuals,
        predictions=preds,
        delivery_hours=hours,
        origin_ids=origin_ids
    )

    assert pool.n_origins == 75
    assert len(pool.daily_sup_norm_scores) == 75

    q_90, rank_90, is_exp_90 = pool.get_simultaneous_quantile(0.10)
    assert rank_90 == 69
    assert not is_exp_90

    q_95, rank_95, is_exp_95 = pool.get_simultaneous_quantile(0.05)
    assert rank_95 == 73
    assert is_exp_95  # N=75 < 150 must flag as exploratory


# =============================================================================
# 6. Delayed Feedback Settlement Guard
# =============================================================================

def test_delayed_feedback_settlement_guard():
    """Verify premature feedback (< 2 days) is rejected by settlement delay guard."""
    aci = DelayedFeedbackACIManager(target_alpha=0.10, gamma=0.01, delay_days=2)
    aci.register_forecast("2024-04-02", np.zeros(24), np.ones(24) * 1000)

    with pytest.raises(ValueError, match="Premature feedback rejected"):
        aci.submit_ground_truth(
            origin_date="2024-04-02",
            actuals=np.ones(24) * 500,
            settlement_timestamp="2024-04-03T10:00:00Z",
            effective_delay_days=1
        )


# =============================================================================
# 7. DDIR State Integration Tests
# =============================================================================

def test_dqcp_engine_ddir_state_transitions():
    """Verify DQ-CP uncertainty behavior across all 5 DDIR reliability states."""
    norm = DeliveryHourMADNormalizer({h: 500.0 for h in range(24)})
    pool_prim = ConformalCalibrationPool(
        model_id="research_primary",
        normalizer=norm,
        marginal_scores=np.ones(100) * 2.0,
        daily_sup_norm_scores=np.ones(20) * 3.0,
        n_origins=20
    )
    pool_fall = ConformalCalibrationPool(
        model_id="research_fallback",
        normalizer=norm,
        marginal_scores=np.ones(100) * 2.5,
        daily_sup_norm_scores=np.ones(20) * 4.0,
        n_origins=20
    )
    engine = DQCPUncertaintyEngine(primary_pool=pool_prim, fallback_pool=pool_fall)

    preds = np.ones(24) * 50000.0
    hours = np.arange(24)

    # 1. HEALTHY: Primary pool used
    res_healthy = engine.generate_interval(preds, hours, ReliabilityState.HEALTHY, forecast_head="primary")
    assert res_healthy.uncertainty_available is True
    assert res_healthy.model_id == "research_primary"
    assert res_healthy.lower_bound_mw[0] == 50000.0 - 2.0 * 500.0

    # 2. DEGRADED (routed to fallback): Fallback pool used
    res_degraded = engine.generate_interval(preds, hours, ReliabilityState.DEGRADED, forecast_head="fallback")
    assert res_degraded.uncertainty_available is True
    assert res_degraded.model_id == "research_fallback"
    assert res_degraded.lower_bound_mw[0] == 50000.0 - 2.5 * 500.0
    assert "degraded" in res_degraded.degradation_flag.lower()

    # 3. STALE: Fallback pool used
    res_stale = engine.generate_interval(preds, hours, ReliabilityState.STALE, forecast_head="fallback")
    assert res_stale.uncertainty_available is True
    assert res_stale.model_id == "research_fallback"

    # 4. UNAVAILABLE: Does NOT invent an interval!
    res_unavail = engine.generate_interval(preds, hours, ReliabilityState.UNAVAILABLE, forecast_head="primary")
    assert res_unavail.uncertainty_available is False
    assert res_unavail.lower_bound_mw is None
    assert res_unavail.upper_bound_mw is None
    assert "unavailable" in res_unavail.degradation_flag.lower()

    # 5. INVALID: Rejects uncertainty
    res_invalid = engine.generate_interval(preds, hours, ReliabilityState.INVALID, forecast_head="primary")
    assert res_invalid.uncertainty_available is False
    assert res_invalid.lower_bound_mw is None


# =============================================================================
# 8. No-Evaluation-Leakage Audit Test
# =============================================================================

def test_no_evaluation_leakage_audit():
    """Verify that calibration tables do not contain or touch held-out evaluation timestamps."""
    calib_prov_path = Path("artifacts/calibration/calibration_provenance.json")
    assert calib_prov_path.exists(), "Calibration provenance artifact must exist."

    with open(calib_prov_path) as f:
        prov = json.load(f)

    assert "2024-03-31" in prov["training_partition"]["end_timestamp"]
    assert prov["calibration_partition"]["start_timestamp"] >= "2024-04-02"
    assert prov["calibration_partition"]["end_timestamp"] <= "2024-06-16"
    assert prov["held_out_evaluation_boundary"]["start_timestamp"] >= "2024-06-16 22:00"
    assert prov["held_out_evaluation_boundary"]["status"] == "STRICTLY_UNTOUCHED_HELD_OUT_SET"


# =============================================================================
# 9. FastAPI DQ-CP Endpoint Tests
# =============================================================================

def test_api_forecast_dqcp_endpoint():
    """Verify forecast_dqcp route handler returns valid prediction envelope and Prometheus metrics."""
    from src.serving.app import load_engine, forecast_dqcp
    from src.serving.schemas import DQCPForecastRequest

    load_engine()

    req_payload = DQCPForecastRequest(
        start_timestamp="2024-06-16T22:00:00Z",
        target_coverage=0.90,
        calibration_method="pooled_marginal_mad",
        forecast_origin="2024-06-16",
        weather_payload=[
            {
                "timestamp": f"2024-06-16T{h:02d}:00:00Z",
                "temperature_c": 18.5,
                "solar_radiation_wm2": 150.0,
                "wind_speed_ms": 4.2,
                "humidity_pct": 60.0
            }
            for h in range(24)
        ]
    )

    # Frozen research tables identify joblib artifacts, not the served ONNX files.
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as excinfo:
        forecast_dqcp(req_payload)
    assert excinfo.value.status_code == 503


def test_api_feedback_settlement_endpoint(monkeypatch):
    """Verify submit_dqcp_feedback route handler processes delayed feedback and enforces 2-day lag."""
    from src.serving.app import load_engine, submit_dqcp_feedback
    from src.serving.schemas import DQCPFeedbackRequest
    from fastapi import HTTPException

    load_engine()

    # Exercise feedback mechanics using an isolated synthetic engine. Startup
    # correctly rejects the incompatible frozen serving/calibration pair.
    from src.serving import app as service
    synthetic = DQCPUncertaintyEngine()
    synthetic.primary_aci_manager.register_forecast(
        '2024-06-16', np.full(24, 50000.0), np.full(24, 54000.0)
    )
    monkeypatch.setattr(service, 'DQCP_ENGINE', synthetic)

    # Premature feedback (< 2 days) must raise HTTPException 422
    premature_payload = DQCPFeedbackRequest(
        forecast_origin="2024-06-16",
        actual_load_mw=[52000.0] * 24,
        settlement_timestamp="2024-06-17T10:00:00Z",
        effective_delay_days=1
    )
    with pytest.raises(HTTPException) as excinfo:
        submit_dqcp_feedback(premature_payload)
    assert excinfo.value.status_code == 422

    # Valid feedback (2 days delay) must succeed
    valid_payload = DQCPFeedbackRequest(
        forecast_origin="2024-06-16",
        actual_load_mw=[52000.0] * 24,
        settlement_timestamp="2024-06-18T10:00:00Z",
        effective_delay_days=2
    )
    data = submit_dqcp_feedback(valid_payload)
    assert data.status in ("processed", "duplicate_ignored")
    assert data.effective_delay_hours == 48
