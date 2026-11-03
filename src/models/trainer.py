"""
Unified Model Training, Evaluation, and Benchmark Pipeline (P5.4, P5.5, P5.6).
Compares baselines and ML models through walk-forward validation and selects the champion.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from src.models.baselines import (
    NaivePersistence24h,
    WeeklySeasonalPersistence168h,
    MovingAverageBaseline,
    RidgeBaseline,
)
from src.models.tree_models import (
    LightGBMModel,
    XGBoostModel,
    RandomForestModel,
    HistGradientBoostingModel,
)
from src.models.walk_forward import TimeSeriesWalkForwardValidator
from src.models.evaluation import diebold_mariano_test

logger = logging.getLogger(__name__)


class ModelBenchmarkTrainer:
    """
    Trains and compares multiple time series forecasting models.
    """
    def __init__(
        self,
        target_col: str = "grid_load_mw",
        n_splits: int = 5,
        test_size_hours: int = 168,
        purge_gap_hours: int = 24,
    ):
        self.target_col = target_col
        self.validator = TimeSeriesWalkForwardValidator(
            n_splits=n_splits,
            test_size_hours=test_size_hours,
            purge_gap_hours=purge_gap_hours,
            window_type="expanding",
        )

    def run_benchmark(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, any], any]:
        """
        Runs complete benchmark suite across baselines and tree models.
        Returns:
            - summary_df: comparison table (MAE, RMSE, MAPE, sMAPE, R2, DM p-value)
            - all_results: dictionary with fold metrics and OOF predictions
            - champion_model: fitted champion model instance
        """
        logger.info(f"Starting Walk-Forward Benchmark ({self.validator.n_splits} folds)...")
        
        # 1. Model Factories
        model_factories = {
            "Naive_24h": lambda: NaivePersistence24h(self.target_col),
            "Weekly_168h": lambda: WeeklySeasonalPersistence168h(self.target_col),
            "Moving_Avg_24h": lambda: MovingAverageBaseline(self.target_col),
            "Ridge_Regression": lambda: RidgeBaseline(alpha=10.0),
            "Random_Forest": lambda: RandomForestModel(n_estimators=100, max_depth=12),
            "HistGradientBoosting": lambda: HistGradientBoostingModel(max_iter=150, learning_rate=0.05),
            "XGBoost": lambda: XGBoostModel(n_estimators=200, learning_rate=0.05, max_depth=6),
            "LightGBM": lambda: LightGBMModel(n_estimators=200, learning_rate=0.05, num_leaves=31),
        }

        all_results = {}
        summary_rows = []

        # Run walk-forward validation for each model
        for name, factory in model_factories.items():
            logger.info(f"Evaluating model: {name}")
            res = self.validator.evaluate_model(
                factory,
                df,
                feature_cols=feature_cols,
                target_col=self.target_col,
            )
            all_results[name] = res
            m = res["overall_metrics"]
            summary_rows.append({
                "Model": name,
                "MAE (MW)": m["mae"],
                "RMSE (MW)": m["rmse"],
                "MAPE (%)": m["mape"],
                "sMAPE (%)": m["smape"],
                "R2": m["r2"],
                "MAE 95% CI": str(m["mae_ci_95"]),
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("MAE (MW)").reset_index(drop=True)

        # 2. Diebold-Mariano test against top baseline (Naive_24h)
        base_oof = all_results["Naive_24h"]["oof_predictions"]
        y_true = all_results["Naive_24h"]["oof_actuals"]
        
        dm_stats = []
        for idx, row in summary_df.iterrows():
            m_name = row["Model"]
            cand_oof = all_results[m_name]["oof_predictions"]
            if m_name == "Naive_24h":
                dm_stats.append("Baseline")
            else:
                dm_stat, p_val = diebold_mariano_test(y_true, base_oof, cand_oof)
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                dm_stats.append(f"p={p_val:.4f} ({sig})")

        summary_df["Diebold-Mariano vs Naive"] = dm_stats

        # 3. Select Champion Model (lowest MAE)
        champion_name = summary_df.iloc[0]["Model"]
        logger.info(f"Champion model selected: {champion_name} (MAE: {summary_df.iloc[0]['MAE (MW)']} MW)")

        # 4. Retrain Champion on entire dataset (excluding final test holdout if needed)
        champion_factory = model_factories[champion_name]
        champion_model = champion_factory()
        
        clean_df = df.dropna(subset=feature_cols + [self.target_col])
        champion_model.fit(clean_df[feature_cols], clean_df[self.target_col].values)

        return summary_df, all_results, champion_model
