"""
Unified Feature Engineering Pipeline (P5.3).
Fits scaling and preprocessing strictly on training data to prevent leakage.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features.temporal import add_temporal_features
from src.features.weather_features import add_weather_features
from src.features.autoregressive import add_autoregressive_features


class FeatureEngineer:
    """
    Feature pipeline for time series electricity forecasting.
    """
    def __init__(
        self,
        target_col: str = "grid_load_mw",
        lags: List[int] = None,
        rolling_windows: List[int] = None,
        scale_features: bool = False,
    ):
        self.target_col = target_col
        self.lags = lags or [1, 2, 3, 24, 48, 168]
        self.rolling_windows = rolling_windows or [6, 12, 24, 168]
        self.scale_features = scale_features
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = []
        self.fitted: bool = False

    def _generate_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all deterministic feature transformations."""
        out = add_temporal_features(df)
        out = add_weather_features(out)
        out = add_autoregressive_features(
            out,
            target_col=self.target_col,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
        )
        return out

    def fit(self, df_train: pd.DataFrame) -> "FeatureEngineer":
        """
        Fits scalers and determines feature column schema strictly from training data.
        """
        df_feat = self._generate_raw_features(df_train)
        
        # Determine candidate feature columns (exclude non-feature metadata)
        exclude_cols = {
            "timestamp", self.target_col, "is_interpolated", "quality_flag", "id", "created_at"
        }
        self.feature_cols = [c for c in df_feat.columns if c not in exclude_cols]

        if self.scale_features:
            self.scaler = StandardScaler()
            # Drop NaN rows caused by max lag for fitting scaler
            clean_train = df_feat[self.feature_cols].dropna()
            self.scaler.fit(clean_train)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, drop_warmup: bool = False) -> pd.DataFrame:
        """
        Transforms dataset using fitted parameters.
        """
        if not self.fitted:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform().")

        df_feat = self._generate_raw_features(df)

        if self.scale_features and self.scaler is not None:
            df_feat[self.feature_cols] = self.scaler.transform(df_feat[self.feature_cols])

        if drop_warmup:
            # Drop rows with NaN caused by max lag (e.g. 168 hours)
            max_lag = max(self.lags + self.rolling_windows)
            df_feat = df_feat.iloc[max_lag:].reset_index(drop=True)

        return df_feat

    def fit_transform(self, df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
        """Convenience method to fit and transform."""
        return self.fit(df).transform(df, drop_warmup=drop_warmup)

    def get_feature_names(self) -> List[str]:
        """Returns ordered list of feature column names."""
        return list(self.feature_cols)
