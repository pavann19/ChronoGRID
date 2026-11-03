"""
Controlled Retraining and Champion-Challenger Model Promotion Engine (P5.10).
Evaluates challenger against champion under strict statistical gates before atomic promotion.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.db.database import get_session
from src.db.models import ModelRegistryRecord
from src.models.evaluation import compute_metrics, diebold_mariano_test
from src.models.tree_models import LightGBMModel, XGBoostModel
from src.serving.onnx_exporter import export_to_onnx

logger = logging.getLogger(__name__)


class PromotionEngine:
    """
    Orchestrates challenger retraining, evaluation gate, and atomic model promotion.
    """
    def __init__(
        self,
        min_improvement_pct: float = 0.02, # Must improve MAE by >= 2%
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
        champion_model: any,
        challenger_model: any,
        df_eval: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "grid_load_mw",
        challenger_model_type: str = "LightGBM",
        challenger_params: Dict[str, any] = None,
    ) -> Dict[str, any]:
        """
        Evaluates Champion vs Challenger on holdout test set and executes atomic promotion if gate passes.
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
            "decision": "PROMOTE_TO_PRODUCTION" if passed_gate else "REJECT_CHALLENGER",
            "champion_metrics": champ_metrics,
            "challenger_metrics": chall_metrics,
        }

        logger.info(f"Promotion gate decision: {decision_report['decision']} (Improvement: {decision_report['improvement_pct']}%, p={p_value:.4f})")

        if passed_gate:
            self._execute_atomic_promotion(
                challenger_model=challenger_model,
                challenger_model_type=challenger_model_type,
                feature_cols=feature_cols,
                metrics=chall_metrics,
                params=challenger_params or {},
            )
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
        challenger_model: any,
        challenger_model_type: str,
        feature_cols: List[str],
        metrics: Dict[str, any],
        params: Dict[str, any],
    ):
        """
        Executes atomic promotion:
        1. Backs up active champion ONNX & metadata.
        2. Exports challenger to a staging temporary file and verifies ONNX validity.
        3. Atomically replaces production files.
        4. Updates database registry within a transaction with rollback guard.
        """
        champ_onnx = self.models_dir / "champion_model.onnx"
        champ_meta = self.models_dir / "champion_metadata.json"
        temp_onnx = self.models_dir / "challenger_staging.tmp.onnx"
        temp_meta = self.models_dir / "challenger_staging.tmp.json"

        # 1. Backup current champion
        backup_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        last_backup_onnx = None
        last_backup_meta = None
        if champ_onnx.exists():
            last_backup_onnx = self.backup_dir / f"champion_backup_{backup_tag}.onnx"
            shutil.copy2(champ_onnx, last_backup_onnx)
            if champ_meta.exists():
                last_backup_meta = self.backup_dir / f"champion_backup_{backup_tag}.json"
                shutil.copy2(champ_meta, last_backup_meta)
            logger.info(f"Active champion backed up to {last_backup_onnx}.")

        # 2. Export challenger to staging temporary file
        try:
            export_to_onnx(
                model_wrapper=challenger_model,
                feature_names=feature_cols,
                output_path=str(temp_onnx),
                metadata_path=str(temp_meta),
            )
            # Verify temporary ONNX file can be parsed by ONNX Runtime
            import onnxruntime as ort
            test_sess = ort.InferenceSession(str(temp_onnx), providers=["CPUExecutionProvider"])
            del test_sess
        except Exception as e:
            if temp_onnx.exists():
                temp_onnx.unlink()
            if temp_meta.exists():
                temp_meta.unlink()
            raise RuntimeError(f"Atomic promotion aborted: challenger ONNX verification failed ({e})") from e

        # 3. Atomically swap staging files to production destination
        if temp_onnx.exists():
            shutil.move(str(temp_onnx), str(champ_onnx))
        if temp_meta.exists():
            shutil.move(str(temp_meta), str(champ_meta))

        # 4. Update Model Registry in DB
        try:
            with get_session() as session:
                # Demote current champion
                session.query(ModelRegistryRecord).filter_by(
                    model_name=self.model_name, is_champion=True
                ).update({"is_champion": False, "stage": "Archived"})

                latest_rec = (
                    session.query(ModelRegistryRecord)
                    .filter_by(model_name=self.model_name)
                    .order_by(ModelRegistryRecord.version.desc())
                    .first()
                )
                next_version = (latest_rec.version + 1) if latest_rec else 1

                new_champion_rec = ModelRegistryRecord(
                    model_id=f"model_{challenger_model_type}_v{next_version}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                    model_name=self.model_name,
                    version=next_version,
                    model_type=challenger_model_type,
                    stage="Production",
                    mae=float(metrics["mae"]),
                    rmse=float(metrics["rmse"]),
                    mape=float(metrics["mape"]),
                    r2=float(metrics["r2"]),
                    onnx_uri=str(champ_onnx),
                    is_champion=True,
                    promoted_at=datetime.utcnow(),
                    metadata_json=json.dumps({"params": params, "features": feature_cols}, default=str),
                )
                session.add(new_champion_rec)
            logger.info(f"Model v{next_version} ({challenger_model_type}) ATOMICALLY PROMOTED to Production!")
        except Exception as e:
            logger.error(f"Database commit failed during promotion: {e}. Executing immediate artifact rollback...")
            if last_backup_onnx and last_backup_onnx.exists():
                shutil.copy2(last_backup_onnx, champ_onnx)
            if last_backup_meta and last_backup_meta.exists():
                shutil.copy2(last_backup_meta, champ_meta)
            raise

    def rollback_to_previous_champion(self) -> Dict[str, any]:
        """
        Restores the previous champion ONNX model and metadata from backup directory
        and updates the database registry.
        """
        champ_onnx = self.models_dir / "champion_model.onnx"
        champ_meta = self.models_dir / "champion_metadata.json"

        # Find latest backup in backup directory
        backup_onnx_files = sorted(self.backup_dir.glob("champion_backup_*.onnx"))
        if not backup_onnx_files:
            raise FileNotFoundError("No champion backup found in backup directory to rollback to.")

        latest_backup_onnx = backup_onnx_files[-1]
        backup_tag = latest_backup_onnx.stem.replace("champion_backup_", "")
        latest_backup_meta = self.backup_dir / f"champion_backup_{backup_tag}.json"

        # Restore files
        shutil.copy2(latest_backup_onnx, champ_onnx)
        if latest_backup_meta.exists():
            shutil.copy2(latest_backup_meta, champ_meta)

        restored_version = None
        # Update DB registry
        with get_session() as session:
            # Mark current failing champion as Rollback_Demoted
            session.query(ModelRegistryRecord).filter_by(
                model_name=self.model_name, is_champion=True
            ).update({"is_champion": False, "stage": "Rollback_Demoted"})

            # Restore the most recent Archived model
            prev_model = (
                session.query(ModelRegistryRecord)
                .filter(ModelRegistryRecord.model_name == self.model_name, ModelRegistryRecord.stage == "Archived")
                .order_by(ModelRegistryRecord.version.desc())
                .first()
            )
            if prev_model:
                prev_model.is_champion = True
                prev_model.stage = "Production"
                restored_version = prev_model.version

        logger.info(f"Successfully rolled back to champion backup {latest_backup_onnx.name} (v{restored_version}).")
        return {
            "status": "ROLLBACK_SUCCESSFUL",
            "restored_backup": latest_backup_onnx.name,
            "restored_version": restored_version,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _log_rejected_challenger(
        self,
        challenger_model_type: str,
        feature_cols: List[str],
        metrics: Dict[str, any],
        params: Dict[str, any],
        decision_report: Dict[str, any],
    ):
        """Logs rejected challenger in staging without changing production champion."""
        with get_session() as session:
            latest_rec = (
                session.query(ModelRegistryRecord)
                .filter_by(model_name=self.model_name)
                .order_by(ModelRegistryRecord.version.desc())
                .first()
            )
            next_version = (latest_rec.version + 1) if latest_rec else 1

            challenger_rec = ModelRegistryRecord(
                model_id=f"rejected_{challenger_model_type}_v{next_version}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                model_name=self.model_name,
                version=next_version,
                model_type=challenger_model_type,
                stage="Staging",
                mae=float(metrics["mae"]),
                rmse=float(metrics["rmse"]),
                mape=float(metrics["mape"]),
                r2=float(metrics["r2"]),
                is_champion=False,
                metadata_json=json.dumps({
                    "params": params,
                    "features": feature_cols,
                    "rejection_reason": decision_report,
                }, default=str),
            )
            session.add(challenger_rec)
            logger.info(f"Challenger model v{next_version} logged in Staging (Rejected: did not satisfy gate).")

