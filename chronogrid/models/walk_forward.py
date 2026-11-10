"""
Walk-Forward Validation Engine for Time Series Forecasting (P5.6).
Implements expanding and rolling window CV with purge/embargo gap to prevent leakage.
"""

from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pandas as pd

from chronogrid.models.evaluation import compute_bootstrap_confidence_intervals, compute_metrics


class TimeSeriesWalkForwardValidator:
    """
    Chronological Walk-Forward Cross-Validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size_hours: int = 168,  # 1 week per fold test set
        min_train_size_hours: int = 2160,  # 90 days minimum initial training
        purge_gap_hours: int = 24,  # 24h day-ahead embargo gap
        window_type: str = "expanding",  # "expanding" or "rolling"
    ):
        self.n_splits = n_splits
        self.test_size_hours = test_size_hours
        self.min_train_size_hours = min_train_size_hours
        self.purge_gap_hours = purge_gap_hours
        self.window_type = window_type

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) for each fold.
        """
        n_total = len(df)
        total_test_span = self.n_splits * self.test_size_hours

        # Determine starting index for first test fold
        first_test_start = n_total - total_test_span
        if first_test_start < self.min_train_size_hours + self.purge_gap_hours:
            first_test_start = self.min_train_size_hours + self.purge_gap_hours

        for i in range(self.n_splits):
            test_start = first_test_start + i * self.test_size_hours
            test_end = min(test_start + self.test_size_hours, n_total)

            if test_start >= n_total:
                break

            train_end = test_start - self.purge_gap_hours

            if self.window_type == "expanding":
                train_start = 0
            else:  # rolling
                train_start = max(0, train_end - self.min_train_size_hours)

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def evaluate_model(
        self,
        model_factory,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "grid_load_mw",
    ) -> Dict[str, Any]:
        """
        Runs walk-forward validation across all folds for a given model factory.
        Collects out-of-fold predictions and summary statistics.
        """
        fold_metrics = []
        oof_predictions = []
        oof_actuals = []
        oof_timestamps = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(df)):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            X_train = df_train[feature_cols]
            y_train = df_train[target_col].values

            X_test = df_test[feature_cols]
            y_test = df_test[target_col].values
            ts_test = df_test["timestamp"].values if "timestamp" in df_test.columns else None

            # Instantiate and fit fresh model
            model = model_factory()
            model.fit(X_train, y_train)

            # Predict
            preds = np.asarray(model.predict(X_test), dtype=float)
            # Drop unknown predictions together with their labels and timestamps.
            # Keeping this mask here preserves alignment; predict() cannot drop X rows alone.
            eligible = np.isfinite(preds) & np.isfinite(y_test)
            dropped = int((~eligible).sum())
            if not eligible.any():
                raise ValueError("No causally available predictions in this fold")
            preds, y_test = preds[eligible], y_test[eligible]
            if ts_test is not None:
                ts_test = ts_test[eligible]

            metrics = compute_metrics(y_test, preds, timestamps=ts_test)
            metrics["unavailable_rows_dropped"] = dropped
            metrics["fold"] = fold_idx + 1
            metrics["train_start"] = (
                str(df_train["timestamp"].iloc[0])
                if "timestamp" in df_train.columns
                else train_idx[0]
            )
            metrics["train_end"] = (
                str(df_train["timestamp"].iloc[-1])
                if "timestamp" in df_train.columns
                else train_idx[-1]
            )
            metrics["test_start"] = (
                str(df_test["timestamp"].iloc[0]) if "timestamp" in df_test.columns else test_idx[0]
            )
            metrics["test_end"] = (
                str(df_test["timestamp"].iloc[-1])
                if "timestamp" in df_test.columns
                else test_idx[-1]
            )
            fold_metrics.append(metrics)

            oof_predictions.extend(preds)
            oof_actuals.extend(y_test)
            if ts_test is not None:
                oof_timestamps.extend(ts_test)

        oof_act = np.array(oof_actuals)
        oof_pred = np.array(oof_predictions)
        overall_metrics = compute_metrics(
            oof_act, oof_pred, timestamps=np.array(oof_timestamps) if oof_timestamps else None
        )
        bootstrap_ci = compute_bootstrap_confidence_intervals(oof_act, oof_pred)
        overall_metrics.update(bootstrap_ci)

        return {
            "overall_metrics": overall_metrics,
            "fold_metrics": fold_metrics,
            "oof_actuals": oof_act,
            "oof_predictions": oof_pred,
            "oof_timestamps": oof_timestamps,
        }
