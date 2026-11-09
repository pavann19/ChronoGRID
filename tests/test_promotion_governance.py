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


def test_legacy_multi_file_mutation_is_retired(tmp_path):
    engine = PromotionEngine(models_dir=str(tmp_path/'models'),backup_dir=str(tmp_path/'backup'))
    with pytest.raises(RuntimeError,match='retired'):
        engine._execute_atomic_promotion(None,'unused',[],{}, {})
    with pytest.raises(RuntimeError,match='retired'):
        engine.rollback_to_previous_champion()
    assert not list((tmp_path/'models').iterdir())
    assert not list((tmp_path/'backup').iterdir())


def test_passing_metric_review_does_not_execute_a_release(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.retraining import promotion_engine as module
    engine=PromotionEngine(models_dir=str(tmp_path/'models'),backup_dir=str(tmp_path/'backup'))
    monkeypatch.setattr(module,'diebold_mariano_test',lambda *args:(-3.0,.001))
    def forbidden(*args,**kwargs):
        raise AssertionError('metric review must not execute artifact or registry mutation')
    monkeypatch.setattr(module,'get_session',forbidden)
    monkeypatch.setattr(engine,'_execute_atomic_promotion',forbidden)
    frame=pd.DataFrame(dict(f=np.arange(48),grid_load_mw=np.full(48,100.0),
                            timestamp=pd.date_range('2024-01-01',periods=48,freq='h',tz='UTC')))
    champ=SimpleNamespace(predict=lambda x:np.full(len(x),110.0))
    challenger=SimpleNamespace(predict=lambda x:np.full(len(x),101.0))
    decision=engine.evaluate_and_promote(champ,challenger,frame,['f'])
    assert decision['passed_promotion_gate'] is True
    assert decision['decision']=='RELEASE_REVIEW_REQUIRED'
    assert decision['promotion_executed'] is False and decision['production_approved'] is False
    assert not list((tmp_path/'models').iterdir())

