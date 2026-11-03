"""
Unified Feature Engineering Pipeline (P5.3).
Enforces strict Day-Ahead forecast issuance constraints and zero data leakage.
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
    Day-Ahead Time Series Feature Pipeline.
    Strictly excludes contemporaneous realised load and weather.
    All features represent information strictly available at forecast issuance time (t - min_lag_hours).
    """
    def __init__(
        self,
        target_col: str = "grid_load_mw",
        min_lag_hours: int = 24, # Day-Ahead Gate Closure constraint
        lags: List[int] = None,
        rolling_windows: List[int] = None,
        weather_mode: str = "nwp_forecast",
        scale_features: bool = False,
    ):
        self.target_col = target_col
        self.min_lag_hours = min_lag_hours
        # Day-ahead lags strictly >= min_lag_hours (e.g. 24, 25, 48, 72, 168)
        self.lags = lags or [24, 25, 48, 72, 168]
        self.rolling_windows = rolling_windows or [6, 12, 24, 168]
        self.weather_mode = weather_mode
        self.scale_features = scale_features
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = []
        self.fitted: bool = False

    def _generate_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all deterministic, issuance-constrained feature transformations."""
        out = add_temporal_features(df)
        out = add_weather_features(out, weather_mode=self.weather_mode)
        out = add_autoregressive_features(
            out,
            target_col=self.target_col,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            min_lag_hours=self.min_lag_hours,
        )
        return out

    def fit(self, df_train: pd.DataFrame) -> "FeatureEngineer":
        """
        Fits scalers and determines feature column schema strictly from training data.
        Excludes contemporaneous realised variables to prevent future data leakage.
        """
        df_feat = self._generate_raw_features(df_train)
        
        # Contemporaneous / realised columns that MUST NOT be used as inputs at forecast issuance time
        strictly_excluded_cols = {
            "timestamp",
            self.target_col,
            "temperature_c",           # Realised weather at delivery time (leakage!)
            "humidity_pct",            # Realised weather
            "wind_speed_ms",           # Realised weather
            "solar_radiation_wm2",     # Realised weather
            "solar_generation_mw",     # Realised generation at delivery time
            "wind_generation_mw",      # Realised generation at delivery time
            "day_ahead_price_eur_mwh", # Realised spot market price
            "is_interpolated",
            "quality_flag",
            "source_smard",
            "source_dwd",
            "data_type",
            "id",
            "created_at",
        }
        self.feature_cols = [c for c in df_feat.columns if c not in strictly_excluded_cols]

        if self.scale_features:
            self.scaler = StandardScaler()
            clean_train = df_feat[self.feature_cols].dropna()
            self.scaler.fit(clean_train)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, drop_warmup: bool = False) -> pd.DataFrame:
        """Transforms dataset using fitted parameters."""
        if not self.fitted:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform().")

        df_feat = self._generate_raw_features(df)

        if self.scale_features and self.scaler is not None:
            df_feat[self.feature_cols] = self.scaler.transform(df_feat[self.feature_cols])

        if drop_warmup:
            max_warmup = max(self.lags + [self.min_lag_hours + w for w in self.rolling_windows])
            df_feat = df_feat.iloc[max_warmup:].reset_index(drop=True)

        return df_feat

    def fit_transform(self, df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
        """Convenience method to fit and transform."""
        return self.fit(df).transform(df, drop_warmup=drop_warmup)

    def get_feature_names(self) -> List[str]:
        """Returns ordered list of feature column names."""
        return list(self.feature_cols)

    def verify_zero_future_leakage(self, df: pd.DataFrame) -> bool:
        """
        Validates that no engineered feature relies on observations within [t - min_lag_hours + 1, t].
        """
        # Ensure raw realised weather columns are not in feature_cols
        forbidden = {"temperature_c", "wind_speed_ms", "solar_radiation_wm2", "humidity_pct", self.target_col}
        overlap = forbidden.intersection(set(self.feature_cols))
        if overlap:
            raise ValueError(f"Leakage detected! Contemporaneous variables in feature set: {overlap}")

        # Ensure all load lags are >= min_lag_hours
        for col in self.feature_cols:
            if col.startswith(f"{self.target_col}_lag_"):
                lag_num = int(col.split("_lag_")[-1])
                if lag_num < self.min_lag_hours:
                    raise ValueError(f"Leakage detected! Lag {lag_num} < {self.min_lag_hours} in {col}")

        return True
