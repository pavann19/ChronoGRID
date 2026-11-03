"""
Tests for Models, Baselines, and Metrics (P5.4, P5.5, P5.6).
"""

import numpy as np
import pandas as pd
from src.models.evaluation import compute_metrics, diebold_mariano_test
from src.models.baselines import NaivePersistence24h, WeeklySeasonalPersistence168h
from src.models.tree_models import LightGBMModel, XGBoostModel


def test_metrics_calculation():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0]) # Errors: +10, -10, +10 -> MAE=10
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == 10.0
    assert m["rmse"] == 10.0
    assert 0.0 < m["mape"] < 10.0
    assert m["r2"] > 0.95


def test_diebold_mariano_significance():
    y_true = np.random.normal(50000, 5000, 200)
    # Poor baseline
    pred_base = y_true + np.random.normal(0, 2000, 200)
    # Excellent candidate
    pred_cand = y_true + np.random.normal(0, 100, 200)

    dm_stat, p_val = diebold_mariano_test(y_true, pred_base, pred_cand)
    assert dm_stat < 0.0 # Candidate has lower loss
    assert p_val < 0.001 # Highly significant


def test_tree_models_fit_predict():
    X = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y = np.random.randn(50)

    lgb_m = LightGBMModel(n_estimators=10)
    lgb_m.fit(X, y)
    p_lgb = lgb_m.predict(X)
    assert len(p_lgb) == 50

    xgb_m = XGBoostModel(n_estimators=10)
    xgb_m.fit(X, y)
    p_xgb = xgb_m.predict(X)
    assert len(p_xgb) == 50
