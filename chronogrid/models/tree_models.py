"""
Tree-Based Gradient Boosting and Ensemble Forecasting Models (P5.5).
Wraps LightGBM, XGBoost, and Scikit-learn models with unified interface.
"""

from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


class LightGBMModel:
    """LightGBM Forecasting Model."""

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ):
        self.name = "LightGBM"
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
            **kwargs,
        }
        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self) -> Dict[str, float]:
        if hasattr(self.model, "feature_importances_") and self.feature_names_:
            imp = self.model.feature_importances_
            total = sum(imp) or 1.0
            return {name: float(val / total) for name, val in zip(self.feature_names_, imp)}
        return {}


class XGBoostModel:
    """XGBoost Forecasting Model."""

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ):
        self.name = "XGBoost"
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            **kwargs,
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self) -> Dict[str, float]:
        if hasattr(self.model, "feature_importances_") and self.feature_names_:
            imp = self.model.feature_importances_
            total = sum(imp) or 1.0
            return {name: float(val / total) for name, val in zip(self.feature_names_, imp)}
        return {}


class RandomForestModel:
    """Scikit-learn Random Forest Regressor."""

    def __init__(
        self, n_estimators: int = 150, max_depth: int = 14, random_state: int = 42, **kwargs
    ):
        self.name = "RandomForest"
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_jobs": -1,
            **kwargs,
        }
        self.model = RandomForestRegressor(**self.params)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self) -> Dict[str, float]:
        if hasattr(self.model, "feature_importances_") and self.feature_names_:
            imp = self.model.feature_importances_
            total = sum(imp) or 1.0
            return {name: float(val / total) for name, val in zip(self.feature_names_, imp)}
        return {}


class HistGradientBoostingModel:
    """Scikit-learn Histogram Gradient Boosting Regressor."""

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        random_state: int = 42,
        **kwargs,
    ):
        self.name = "HistGradientBoosting"
        self.params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "random_state": random_state,
            **kwargs,
        }
        self.model = HistGradientBoostingRegressor(**self.params)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
