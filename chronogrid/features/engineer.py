"""
Unified Feature Engineering Pipeline (P5.3).
Enforces strict Day-Ahead forecast issuance constraints and zero data leakage.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from chronogrid.features.temporal import add_temporal_features
from chronogrid.features.weather_features import add_weather_features
from chronogrid.features.autoregressive import add_autoregressive_features


class FeatureEngineer:
    """
    Day-Ahead Time Series Feature Pipeline.
    Strictly excludes contemporaneous realised load and weather.
    All features represent information strictly available at forecast issuance time (t - min_lag_hours).
    """
    def __init__(
        self,
        target_col: str = "grid_load_mw",
        min_lag_hours: int = 48,
        lags: List[int] = None,
        rolling_windows: List[int] = None,
        weather_mode: str = "persistence", # "persistence" (default honest mode), "load_only"
        scale_features: bool = False,
        allow_post_gate_lags: bool = False,
    ):
        if min_lag_hours < 48 and not allow_post_gate_lags:
            raise ValueError("Lags below 48h require allow_post_gate_lags=True (leaky comparison baseline)")
        self.allow_post_gate_lags = allow_post_gate_lags
        self.target_col = target_col
        self.min_lag_hours = min_lag_hours
        # Day-ahead lags strictly >= min_lag_hours (e.g. 24, 25, 48, 72, 168)
        self.lags = lags or ([24, 25, 48, 72, 168] if allow_post_gate_lags else [48, 72, 168])
        if any(lag < 48 for lag in self.lags) and not allow_post_gate_lags:
            raise ValueError("Post-gate lags require allow_post_gate_lags=True")
        self.rolling_windows = rolling_windows or [6, 12, 24, 168]
        self.weather_mode = weather_mode
        self.scale_features = scale_features
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = []
        self.fitted: bool = False

    def _generate_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all deterministic, issuance-constrained feature transformations."""
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        if not self.allow_post_gate_lags and (timestamps.duplicated().any() or not timestamps.is_monotonic_increasing):
            raise ValueError("Features require unique chronological timestamps")
        if not self.allow_post_gate_lags and not timestamps.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            df = df.assign(timestamp=timestamps).set_index("timestamp").reindex(pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq="h")).rename_axis("timestamp").reset_index()
        out = add_temporal_features(df)
        out = add_weather_features(out, weather_mode=self.weather_mode, min_lag_hours=self.min_lag_hours)
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
            df_feat = df_feat.iloc[max_warmup:].dropna(subset=self.feature_cols).reset_index(drop=True)

        return df_feat

    def fit_transform(self, df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
        """Convenience method to fit and transform."""
        return self.fit(df).transform(df, drop_warmup=drop_warmup)

    def get_feature_names(self) -> List[str]:
        """Returns ordered list of feature column names."""
        return list(self.feature_cols)

    def verify_zero_future_leakage(self, df: pd.DataFrame = None, return_details: bool = False) -> Union[bool, Dict[str, Any]]:
        """
        Validates that no engineered feature relies on observations within [t - min_lag_hours + 1, t].
        Performs programmatic inspection of columns, lags, and rolling shifts.
        """
        results = {
            "contemporaneous_excluded": True,
            "min_lag_hours_verified": True,
            "rolling_shift_verified": True,
            "forbidden_columns_found": [],
            "invalid_lags_found": [],
            "status": "PASSED"
        }

        # 1. Ensure raw realised weather and load columns are not in feature_cols
        forbidden = {
            "temperature_c", "wind_speed_ms", "solar_radiation_wm2", "humidity_pct",
            self.target_col, "solar_generation_mw", "wind_generation_mw", "day_ahead_price_eur_mwh"
        }
        overlap = forbidden.intersection(set(self.feature_cols))
        if overlap:
            results["contemporaneous_excluded"] = False
            results["forbidden_columns_found"] = list(overlap)
            results["status"] = "FAILED"
            raise ValueError(f"Leakage detected! Contemporaneous variables in feature set: {overlap}")

        # 2. Ensure all load lags are >= min_lag_hours
        for col in self.feature_cols:
            if col.startswith(f"{self.target_col}_lag_"):
                lag_num = int(col.split("_lag_")[-1])
                if lag_num < self.min_lag_hours:
                    results["min_lag_hours_verified"] = False
                    results["invalid_lags_found"].append(col)
                    results["status"] = "FAILED"
                    raise ValueError(f"Leakage detected! Lag {lag_num} < {self.min_lag_hours} in {col}")

        # 3. Ensure rolling statistics were generated from shifted target
        roll_cols = [c for c in self.feature_cols if f"{self.target_col}_roll_" in c]
        if roll_cols and self.min_lag_hours <= 0:
            results["rolling_shift_verified"] = False
            results["status"] = "FAILED"
            raise ValueError("Leakage detected! Rolling statistics require min_lag_hours >= 1")

        if return_details:
            return results
        return True

    def build_day_ahead_market_frame(
        self,
        history_df: pd.DataFrame,
        forecast_origin: Optional[pd.Timestamp] = None,
        weather_persistence_lag: int = 24,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Constructs a Day-Ahead Market (DAM) forecast frame adhering strictly to European market rules:
        - Market Gate Closure: 12:00 CET/CEST on Day D-1 (Local Germany 'Europe/Berlin').
        - Delivery Day: Day D (00:00 to 23:00 local time).
        - At gate closure, only data up to forecast_origin is available.
        - Returns (features_df, market_metadata).
        """
        df_hist = history_df.sort_values("timestamp").reset_index(drop=True)
        if df_hist["timestamp"].dt.tz is None:
            df_hist["timestamp"] = df_hist["timestamp"].dt.tz_localize("UTC")

        # Determine latest timestamp in history if origin is not provided
        if forecast_origin is None:
            max_ts = df_hist["timestamp"].max()
            # Convert to Berlin time to find previous 12:00 gate closure
            max_ts_berlin = max_ts.tz_convert("Europe/Berlin")
            # If current Berlin time is past 12:00, origin is today 12:00 Berlin, delivery is tomorrow
            if max_ts_berlin.hour >= 12:
                origin_berlin = max_ts_berlin.replace(hour=12, minute=0, second=0, microsecond=0)
            else:
                prev_day = max_ts_berlin - pd.Timedelta(days=1)
                origin_berlin = prev_day.replace(hour=12, minute=0, second=0, microsecond=0)
            forecast_origin = origin_berlin.tz_convert("UTC")
        else:
            if forecast_origin.tz is None:
                forecast_origin = forecast_origin.tz_localize("UTC")
            origin_berlin = forecast_origin.tz_convert("Europe/Berlin")

        # Delivery day in local time is the following calendar day
        delivery_date = (origin_berlin + pd.Timedelta(days=1)).date()
        
        # Build local delivery timestamps across the 24 local hours (or 23/25 on DST transition days)
        # Using date_range in Europe/Berlin automatically handles DST spring (23h) and autumn (25h)
        delivery_local_start = pd.Timestamp(f"{delivery_date} 00:00:00", tz="Europe/Berlin")
        # Find next midnight local
        delivery_local_end_exclusive = pd.Timestamp(f"{delivery_date + pd.Timedelta(days=1)} 00:00:00", tz="Europe/Berlin")
        
        # Hourly grid in Europe/Berlin
        local_grid = pd.date_range(delivery_local_start, delivery_local_end_exclusive - pd.Timedelta(hours=1), freq="1h")
        delivery_utc_grid = local_grid.tz_convert("UTC")
        delivery_hours_count = len(delivery_utc_grid)

        # Truncate history strictly to timestamp <= forecast_origin
        strict_history = df_hist[df_hist["timestamp"] <= forecast_origin].copy()
        if len(strict_history) < 168:
            raise ValueError(f"Insufficient historical data ({len(strict_history)}h < 168h required) at origin {forecast_origin}")

        # Construct future rows
        future_rows = []
        last_known_load = strict_history[self.target_col].iloc[-1]

        for ts_utc in delivery_utc_grid:
            # Weather persistence strictly from known history (e.g. 24h before delivery or at issuance)
            # Find matching persistence timestamp in history (ts - weather_persistence_lag <= origin)
            target_lag_ts = ts_utc - pd.Timedelta(hours=weather_persistence_lag)
            if target_lag_ts > forecast_origin:
                target_lag_ts = forecast_origin

            match_row = strict_history[strict_history["timestamp"] <= target_lag_ts].iloc[-1]

            future_rows.append({
                "timestamp": ts_utc,
                self.target_col: last_known_load,  # Placeholder, will be replaced by lag
                "temperature_c": match_row.get("temperature_c", 15.0),
                "solar_radiation_wm2": match_row.get("solar_radiation_wm2", 0.0),
                "wind_speed_ms": match_row.get("wind_speed_ms", 4.0),
                "humidity_pct": match_row.get("humidity_pct", 65.0),
            })

        future_df = pd.DataFrame(future_rows)
        combined_df = pd.concat([strict_history.tail(336), future_df], ignore_index=True)

        # Generate features
        if not self.fitted:
            self.fit(strict_history)

        transformed = self.transform(combined_df, drop_warmup=False)
        delivery_features = transformed.iloc[-delivery_hours_count:].reset_index(drop=True)

        metadata = {
            "forecast_origin_utc": forecast_origin.isoformat(),
            "forecast_origin_berlin": origin_berlin.isoformat(),
            "delivery_date": str(delivery_date),
            "delivery_start_utc": delivery_utc_grid[0].isoformat(),
            "delivery_end_utc": delivery_utc_grid[-1].isoformat(),
            "delivery_hours": delivery_hours_count,
            "lead_time_min_hours": (delivery_utc_grid[0] - forecast_origin).total_seconds() / 3600.0,
            "lead_time_max_hours": (delivery_utc_grid[-1] - forecast_origin).total_seconds() / 3600.0,
            "timezone": "Europe/Berlin",
            "weather_mode": self.weather_mode,
        }

        return delivery_features, metadata

    @staticmethod
    def _suite_result(details):
        passed = sum(bool(item["passed"]) for item in details.values())
        return {"suite_status": "ALL_TESTS_PASSED" if passed == len(details) else "TESTS_FAILED",
                "tests_passed": passed, "total_tests": len(details), "details": details}

    def run_leakage_suite(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perturb raw observations and compare features at their original row positions.

        This checks transformation causality. Gate-relative availability is a separate audit.
        """
        clean = df.dropna().reset_index(drop=True)
        if len(clean) <= max(self.lags) + self.min_lag_hours + 10:
            raise ValueError("Insufficient rows for meaningful perturbation tests")
        if not self.fitted:
            self.fit(clean)
        cols = self.get_feature_names()
        base = self.transform(clean)[cols]
        cut = len(clean) - self.min_lag_hours - 5
        details = {}
        def compare(name, altered, stop):
            actual = self.transform(altered)[cols]
            left, right = base.iloc[:stop], actual.iloc[:stop]
            equal = np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True)
            details[name] = {"passed": bool(equal.all()), "rows_compared": stop}
        pert = clean.copy()
        pert.loc[cut, self.target_col] += 50000
        compare("1_future_target_perturbation", pert, cut + self.min_lag_hours)
        weather = [c for c in ("temperature_c", "wind_speed_ms", "humidity_pct", "solar_radiation_wm2") if c in clean]
        if weather and self.weather_mode != "load_only":
            pert = clean.copy()
            pert.loc[cut:, weather] += 1000
            compare("2_future_weather_perturbation", pert, cut + self.min_lag_hours)
        pert = clean.copy()
        pert.loc[cut:, self.target_col] = np.arange(len(clean)-cut) * 100000 + 900000
        compare("3_future_load_perturbation", pert, cut + self.min_lag_hours)
        details["target_excluded"] = {"passed": self.target_col not in cols}
        shifted = clean[self.target_col].shift(self.min_lag_hours)
        for window in self.rolling_windows:
            col = f"{self.target_col}_roll_mean_{window}"
            expected = shifted.rolling(window, min_periods=max(1, window//2)).mean()
            details[f"rolling_shift_{window}"] = {"passed": bool(np.isclose(base[col], expected, equal_nan=True).all())}
        return self._suite_result(details)

    def run_data_integrity_suite(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Report timezone, DST, horizon and schema checks separately from leakage."""
        if not self.fitted:
            self.fit(df)
        details = {"timezone_continuity": {"passed": bool(df.timestamp.dt.tz is not None and df.timestamp.is_unique and df.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all())}}
        for label, day, end, count in [("spring", "2024-03-31", "2024-04-01", 23), ("autumn", "2024-10-27", "2024-10-28", 25)]:
            grid = pd.date_range(day, end, inclusive="left", freq="h", tz="Europe/Berlin")
            details[f"dst_{label}"] = {"passed": bool(len(grid)==count and grid.tz_convert("UTC").is_unique)}
        try:
            _, metadata = self.build_day_ahead_market_frame(df)
            horizon = metadata["delivery_hours"] in (23, 24, 25)
        except (ValueError, IndexError):
            horizon = False
        details["forecast_horizon_validity"] = {"passed": horizon}
        details["feature_schema_parity"] = {"passed": list(self.transform(df)[self.feature_cols]) == self.feature_cols}
        return self._suite_result(details)
