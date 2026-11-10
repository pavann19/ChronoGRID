"""
Legacy champion/challenger metric review. Automatic artifact and registry mutation
is retired; use explicit diagnostic ArtifactRegistry bundles for local lifecycle work.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from chronogrid.db.database import get_session as get_session
from chronogrid.models.evaluation import compute_metrics, diebold_mariano_test

logger = logging.getLogger(__name__)


class PromotionEngine:
    """
    Legacy metric evaluator. Passing metrics do not authorize a production release.
    """

    def __init__(
        self,
        min_improvement_pct: float = 0.02,  # Must improve MAE by >= 2%
        significance_alpha: float = 0.05,
        max_peak_error_degradation_pct: float = 0.05,
        model_name: str = "ChronoGrid_LoadForecaster",
        models_dir: str = "artifacts/models",
        backup_dir: str = "artifacts/models/backup",
    ):
        self.min_improvement_pct = min_improvement_pct
        self.significance_alpha = significance_alpha
        self.max_peak_error_degradation_pct = max_peak_error_degradation_pct
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.backup_dir = Path(backup_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_promote(
        self,
        champion_model: Any,
        challenger_model: Any,
        df_eval: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "grid_load_mw",
        challenger_model_type: str = "LightGBM",
        challenger_params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Returns a metric review; passing metrics never executes an artifact promotion.
        """
        logger.info("Evaluating Champion vs Challenger on validation holdout...")

        X_eval = df_eval[feature_cols]
        y_true = df_eval[target_col].values
        timestamps = df_eval["timestamp"].values if "timestamp" in df_eval.columns else None

        # Predictions
        champ_preds = champion_model.predict(X_eval)
        chall_preds = challenger_model.predict(X_eval)

        # Performance Metrics
        champ_metrics = compute_metrics(y_true, champ_preds, timestamps=timestamps)
        chall_metrics = compute_metrics(y_true, chall_preds, timestamps=timestamps)

        # 1. Metric Improvement Check
        champ_mae = champ_metrics["mae"]
        chall_mae = chall_metrics["mae"]
        improvement_pct = (champ_mae - chall_mae) / max(champ_mae, 1e-5)

        # 2. Statistical Significance (Diebold-Mariano test)
        dm_stat, p_value = diebold_mariano_test(y_true, champ_preds, chall_preds)
        is_statistically_significant = bool(p_value < self.significance_alpha and dm_stat < 0)

        # 3. Peak Hour Degradation Safety Check
        champ_peak = champ_metrics.get("peak_hours_mae", champ_mae)
        chall_peak = chall_metrics.get("peak_hours_mae", chall_mae)
        peak_degradation = (chall_peak - champ_peak) / max(champ_peak, 1e-5)
        peak_safety_passed = bool(peak_degradation <= self.max_peak_error_degradation_pct)

        # Promotion Gate Decision
        passed_gate = bool(
            (improvement_pct >= self.min_improvement_pct)
            and is_statistically_significant
            and peak_safety_passed
        )

        decision_report = {
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "champion_mae": champ_mae,
            "challenger_mae": chall_mae,
            "improvement_pct": round(improvement_pct * 100.0, 2),
            "diebold_mariano_statistic": round(dm_stat, 4),
            "p_value": round(p_value, 5),
            "is_statistically_significant": is_statistically_significant,
            "peak_hour_safety_passed": peak_safety_passed,
            "passed_promotion_gate": passed_gate,
            "decision": "RELEASE_REVIEW_REQUIRED" if passed_gate else "REJECT_CHALLENGER",
            "promotion_executed": False,
            "production_approved": False,
            "champion_metrics": champ_metrics,
            "challenger_metrics": chall_metrics,
        }

        logger.info(
            f"Promotion gate decision: {decision_report['decision']} (Improvement: {decision_report['improvement_pct']}%, p={p_value:.4f})"
        )

        if passed_gate:
            logger.info("Metric gate passed; automatic legacy artifact mutation is retired.")
        else:
            self._log_rejected_challenger(
                challenger_model_type=challenger_model_type,
                feature_cols=feature_cols,
                metrics=chall_metrics,
                params=challenger_params or {},
                decision_report=decision_report,
            )

        return decision_report

    def _execute_atomic_promotion(
        self,
        challenger_model: Any,
        challenger_model_type: str,
        feature_cols: List[str],
        metrics: Dict[str, Any],
        params: Dict[str, Any],
    ):
        """Retired: file swaps and SQL transactions are not one atomic release."""
        raise RuntimeError(
            "Legacy multi-file promotion/rollback is retired; use ArtifactRegistry diagnostic bundles."
        )

    def rollback_to_previous_champion(self) -> Dict[str, Any]:
        """Retired: file swaps and SQL transactions are not one atomic release."""
        raise RuntimeError(
            "Legacy multi-file promotion/rollback is retired; use ArtifactRegistry diagnostic bundles."
        )

    def _log_rejected_challenger(
        self,
        challenger_model_type: str,
        feature_cols: List[str],
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        decision_report: Dict[str, Any],
    ):
        """Legacy registry writes are retired; the caller retains the returned review report."""
        logger.info("Rejected challenger review returned without mutating the legacy registry.")
