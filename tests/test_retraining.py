"""
Tests for Retraining and Promotion Logic (P5.10).
"""

import numpy as np
import pandas as pd

from src.retraining.promotion_engine import PromotionEngine


class MockModel:
    def __init__(self, error_scale: float):
        self.error_scale = error_scale

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 50000.0) + self.error_scale


def test_promotion_engine_rejects_inferior_challenger(isolated_application_database):
    engine = PromotionEngine(min_improvement_pct=0.02)
    df_eval = pd.DataFrame({
        "target": np.random.normal(50000, 1000, 100),
        "f1": np.random.randn(100),
    })
    # Champion error: 50, Challenger error: 200 (inferior)
    champ = MockModel(error_scale=50.0)
    chall = MockModel(error_scale=200.0)

    decision = engine.evaluate_and_promote(
        champion_model=champ,
        challenger_model=chall,
        df_eval=df_eval,
        feature_cols=["f1"],
        target_col="target",
    )
    assert decision["passed_promotion_gate"] is False
    assert decision["decision"] == "REJECT_CHALLENGER"
