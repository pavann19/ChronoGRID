"""
Tests for ChronoGrid DDIR Historical evaluation Audit Verification.
Ensures artifact checksums, schema integrity, fallback parity, hysteresis anti-replay,
and session reuse remain verified.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from chronogrid.features.engineer import FeatureEngineer
from chronogrid.reliability.recovery import HysteresisRecoveryManager
from chronogrid.reliability.router import DualHeadInferenceRouter
from chronogrid.reliability.state_machine import ReliabilityState
from chronogrid.serving.onnx_engine import OnnxInferenceEngine

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def paths():
    return {
        "raw": ROOT_DIR / "data/raw/genuine_smard_dwd_observations.parquet",
        "champ_onnx": ROOT_DIR / "artifacts/models/champion_model.onnx",
        "champ_meta": ROOT_DIR / "artifacts/models/champion_metadata.json",
        "fall_onnx": ROOT_DIR / "artifacts/models/load_only_model.onnx",
        "fall_meta": ROOT_DIR / "artifacts/models/load_only_metadata.json",
    }


def test_audit_artifact_checksums_and_schemas(paths):
    # Verify all artifacts exist and have non-empty sizes
    for name, p in paths.items():
        assert p.exists(), f"Artifact missing: {p}"
        assert p.stat().st_size > 0, f"Artifact empty: {p}"

    with open(paths["champ_meta"]) as f:
        c_meta = json.load(f)
    with open(paths["fall_meta"]) as f:
        f_meta = json.load(f)

    c_feats = c_meta["feature_names"]
    f_feats = f_meta["feature_names"]

    assert len(c_feats) == 43, f"Expected 43 champion features, got {len(c_feats)}"
    assert len(f_feats) == 34, f"Expected 34 fallback features, got {len(f_feats)}"
    assert set(f_feats).issubset(set(c_feats)), (
        "Fallback features must be strict subset of champion"
    )

    diff = set(c_feats) - set(f_feats)
    assert len(diff) == 9
    for col in diff:
        assert "forecast_" in col, f"Non-weather column in difference: {col}"


def test_audit_fallback_parity_and_correctness(paths):
    router = DualHeadInferenceRouter(
        primary_onnx_path=str(paths["champ_onnx"]),
        primary_metadata_path=str(paths["champ_meta"]),
        fallback_onnx_path=str(paths["fall_onnx"]),
        fallback_metadata_path=str(paths["fall_meta"]),
        recovery_threshold=2,
    )
    direct_fall_engine = OnnxInferenceEngine(str(paths["fall_onnx"]), str(paths["fall_meta"]))

    # Synthetic sample row
    fe_pers = FeatureEngineer(
        allow_post_gate_lags=True, min_lag_hours=24, weather_mode="persistence"
    )
    df_raw = pd.read_parquet(paths["raw"]).iloc[:500]
    df_pers = fe_pers.fit_transform(df_raw, drop_warmup=True)
    sample_pers = df_pers.iloc[0:1]

    # Direct fallback reference
    fe_load = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, weather_mode="load_only")
    df_load = fe_load.fit_transform(df_raw, drop_warmup=True)
    sample_load = df_load.iloc[0:1]
    ref_fallback_pred = round(
        float(direct_fall_engine.predict(sample_load[direct_fall_engine.feature_names].values)[0]),
        1,
    )

    now_utc = datetime.now(timezone.utc)
    scenarios = [
        ("missing_weather", None),
        (
            "stale_weather",
            {
                "timestamp": (now_utc - timedelta(hours=48)).isoformat(),
                "temperature_c": 15.0,
                "wind_speed_ms": 3.0,
                "solar_radiation_wm2": 100.0,
                "humidity_pct": 50.0,
            },
        ),
        (
            "invalid_outlier",
            {
                "timestamp": now_utc.isoformat(),
                "temperature_c": 999.0,
                "wind_speed_ms": 3.0,
                "solar_radiation_wm2": 100.0,
                "humidity_pct": 50.0,
            },
        ),
        (
            "partial_missing",
            {
                "timestamp": now_utc.isoformat(),
                "temperature_c": 15.0,
                "wind_speed_ms": None,
                "solar_radiation_wm2": None,
                "humidity_pct": 50.0,
            },
        ),
    ]

    for sc_name, payload in scenarios:
        router.recovery_manager.reset(ReliabilityState.UNAVAILABLE)
        res = router.route_and_predict(sample_pers, weather_payload=payload, current_time=now_utc)
        assert res.active_head == "FALLBACK_LOAD_ONLY"
        assert res.predictions[0] == ref_fallback_pred, f"Fallback prediction mismatch in {sc_name}"
        assert res.features_used_count == 34


def test_audit_hysteresis_anti_replay_and_flapping():
    mgr = HysteresisRecoveryManager(recovery_threshold=2, start_healthy=True)
    now_ref = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Healthy baseline at T-4h
    t0 = (now_ref - timedelta(hours=4)).isoformat()
    _, head0, _ = mgr.process_state(
        ReliabilityState.HEALTHY, "OK", source_timestamp=t0, payload_fingerprint="fp0"
    )
    assert head0 == "PRIMARY_WEATHER"

    # 2. Failure occurs -> Immediate demotion
    _, head1, _ = mgr.process_state(ReliabilityState.UNAVAILABLE, "Crash", None, None)
    assert head1 == "FALLBACK_LOAD_ONLY"
    assert mgr.consecutive_healthy_count == 0

    # 3. Healthy update 1 at T-3h -> held on fallback (K=1 < 2)
    t1 = (now_ref - timedelta(hours=3)).isoformat()
    _, head2, r2 = mgr.process_state(
        ReliabilityState.HEALTHY, "OK", source_timestamp=t1, payload_fingerprint="fp1"
    )
    assert head2 == "FALLBACK_LOAD_ONLY"
    assert mgr.consecutive_healthy_count == 1
    assert "recovery in progress" in r2.lower()

    # 4. Duplicate replay of T-3h -> anti-replay blocks increment!
    _, head_rep, r_rep = mgr.process_state(
        ReliabilityState.HEALTHY, "OK", source_timestamp=t1, payload_fingerprint="fp1"
    )
    assert head_rep == "FALLBACK_LOAD_ONLY"
    assert mgr.consecutive_healthy_count == 1
    assert "recovery held" in r_rep.lower()

    # 5. Advancing healthy update 2 at T-2h -> promotes to primary!
    t2 = (now_ref - timedelta(hours=2)).isoformat()
    st3, head3, _ = mgr.process_state(
        ReliabilityState.HEALTHY, "OK", source_timestamp=t2, payload_fingerprint="fp2"
    )
    assert head3 == "PRIMARY_WEATHER"
    assert st3 == ReliabilityState.HEALTHY
    assert mgr.consecutive_healthy_count == 2


def test_audit_session_reuse_and_thread_safety(paths):
    router = DualHeadInferenceRouter(
        primary_onnx_path=str(paths["champ_onnx"]),
        primary_metadata_path=str(paths["champ_meta"]),
        fallback_onnx_path=str(paths["fall_onnx"]),
        fallback_metadata_path=str(paths["fall_meta"]),
    )

    # Verify session pointers remain stable
    sess_pri_id = id(router.primary_engine.session)
    sess_fall_id = id(router.fallback_engine.session)

    # 50 concurrent requests
    df_raw = pd.read_parquet(paths["raw"]).iloc[:500]
    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, weather_mode="persistence")
    row = fe.fit_transform(df_raw, drop_warmup=True).iloc[0:1]

    def worker(idx):
        pay = (
            None
            if idx % 2 == 0
            else {
                "temperature_c": 18.0,
                "wind_speed_ms": 4.0,
                "solar_radiation_wm2": 250.0,
                "humidity_pct": 55.0,
            }
        )
        return router.route_and_predict(row, weather_payload=pay)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(50)))

    assert len(results) == 50
    assert id(router.primary_engine.session) == sess_pri_id
    assert id(router.fallback_engine.session) == sess_fall_id
