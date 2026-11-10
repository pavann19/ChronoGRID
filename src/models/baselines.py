"""
Statistical and Machine Learning Baselines (P5.4).
Provides standard benchmark references: Naive 24h, Weekly Seasonal Naive, Moving Average, and Ridge.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class NaivePersistence24h:
    """Predicts yesterday's same-hour value: y_hat(t) = y(t-24)."""
    def __init__(self, target_col: str = "grid_load_mw"):
        self.target_col = target_col
        self.name = "Naive_Persistence_24h"

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag_col = f"{self.target_col}_lag_24"
        if lag_col in X.columns:
            return X[lag_col].ffill(limit=3).values
        raise ValueError(f"Required lag column '{lag_col}' not found in features.")


class WeeklySeasonalPersistence168h:
    """Predicts last week's same-day and hour value: y_hat(t) = y(t-168)."""
    def __init__(self, target_col: str = "grid_load_mw"):
        self.target_col = target_col
        self.name = "Weekly_Seasonal_168h"

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag_col = f"{self.target_col}_lag_168"
        if lag_col in X.columns:
            return X[lag_col].ffill(limit=3).values
        raise ValueError(f"Required lag column '{lag_col}' not found in features.")


class MovingAverageBaseline:
    """Predicts rolling mean of past 24 hours."""
    def __init__(self, target_col: str = "grid_load_mw"):
        self.target_col = target_col
        self.name = "Moving_Average_24h"

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        roll_col = f"{self.target_col}_roll_mean_24"
        if roll_col in X.columns:
            return X[roll_col].ffill(limit=3).values
        raise ValueError(f"Required column '{roll_col}' not found in features.")


class RidgeBaseline:
    """Standard regularized linear regression baseline."""
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.name = "Ridge_Regression_Baseline"
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
