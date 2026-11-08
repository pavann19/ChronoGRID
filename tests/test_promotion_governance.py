"""
Promotion Engine Governance & Rollback Tests.
Verifies rejection of inferior models, atomic promotion, and rollback capabilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.tree_models import LightGBMModel, XGBoostModel
from src.retraining.promotion_engine import PromotionEngine


@pytest.fixture
def dummy_train_eval_data(isolated_application_database):
    rng = np.random.RandomState(42)
    X = pd.DataFrame({f"f_{i}": rng.randn(200) for i in range(10)})
    y = 50000.0 + rng.randn(200) * 1000.0
    df = X.copy()
    df["grid_load_mw"] = y
    df["timestamp"] = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    return df, [f"f_{i}" for i in range(10)]


def test_promotion_engine_rejects_degraded_model(dummy_train_eval_data):
    df, feat_cols = dummy_train_eval_data
    engine = PromotionEngine(min_improvement_pct=0.05)

    champ = LightGBMModel(n_estimators=30, learning_rate=0.05)
    champ.fit(df.iloc[:150][feat_cols], df.iloc[:150]["grid_load_mw"].values)

    weak_challenger = LightGBMModel(n_estimators=2, learning_rate=0.001)
    weak_challenger.fit(df.iloc[:150][feat_cols], df.iloc[:150]["grid_load_mw"].values)

    eval_holdout = df.iloc[150:]
    decision = engine.evaluate_and_promote(
        champion_model=champ,
        challenger_model=weak_challenger,
        df_eval=eval_holdout,
        feature_cols=feat_cols,
        target_col="grid_load_mw",
        challenger_model_type="LightGBM_Weak",
    )

    assert decision["decision"] == "REJECT_CHALLENGER"
    assert decision["passed_promotion_gate"] is False


def test_atomic_promotion_and_rollback(dummy_train_eval_data):
    df, feat_cols = dummy_train_eval_data
    engine = PromotionEngine(min_improvement_pct=0.01)

    model = XGBoostModel(n_estimators=10, learning_rate=0.05)
    model.fit(df.iloc[:150][feat_cols], df.iloc[:150]["grid_load_mw"].values)

    # 1. Promote to generate active model & backup
    engine._execute_atomic_promotion(
        challenger_model=model,
        challenger_model_type="XGBoost_GovernanceTest",
        feature_cols=feat_cols,
        metrics={"mae": 900.0, "rmse": 1200.0, "mape": 1.8, "r2": 0.98},
        params={"n_estimators": 10},
    )

    # 2. Rollback to restore previous backup
    res = engine.rollback_to_previous_champion()
    assert res["status"] == "ROLLBACK_SUCCESSFUL"
    assert res["restored_backup"] is not None

