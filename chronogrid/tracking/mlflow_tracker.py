"""
MLflow Experiment Tracking and Model Registry Management (P5.7).
Logs parameters, metrics, artifacts, and manages model versions & stages in registry.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from chronogrid.db.database import get_session
from chronogrid.db.models import ModelRegistryRecord

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Manages MLflow experiment runs and Model Registry.
    """
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns/mlflow.db",
        experiment_name: str = "ChronoGrid_Electricity_Load_Forecasting",
        model_name: str = "ChronoGrid_LoadForecaster",
        artifact_dir: str = "artifacts/mlflow",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        Path("mlruns").mkdir(parents=True, exist_ok=True)

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient()
            try:
                exp = mlflow.get_experiment_by_name(self.experiment_name)
                if exp is None:
                    self.experiment_id = mlflow.create_experiment(
                        self.experiment_name,
                        artifact_location=str(self.artifact_dir.resolve())
                    )
                else:
                    self.experiment_id = exp.experiment_id
            except Exception as e:
                logger.warning(f"MLflow experiment init warning: {e}")
                self.experiment_id = "default"
        else:
            logger.warning("MLflow library not available. Using local artifact fallback.")
            self.client = None
            self.experiment_id = "local_fallback"

    def log_model_run(
        self,
        model_name: str,
        model_type: str,
        params: Dict[str, any],
        metrics: Dict[str, any],
        feature_names: List[str],
        feature_importances: Dict[str, float] = None,
        model_artifact_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        stage: str = "Staging",
        is_champion: bool = False,
    ) -> str:
        """
        Logs run parameters, metrics, artifacts to MLflow and mirrors to DB Model Registry.
        """
        run_id = f"run_{model_name}_{int(pd.Timestamp.utcnow().timestamp())}"

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(experiment_id=self.experiment_id, run_name=model_name) as run:
                    run_id = run.info.run_id
                    # Log params
                    for k, v in params.items():
                        mlflow.log_param(k, v)
                    mlflow.log_param("model_type", model_type)
                    mlflow.log_param("num_features", len(feature_names))

                    # Log metrics
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            mlflow.log_metric(k, float(v))

                    # Log feature importances artifact
                    if feature_importances:
                        imp_path = self.artifact_dir / f"{run_id}_feature_importances.json"
                        with open(imp_path, "w") as f:
                            json.dump(feature_importances, f, indent=2)
                        mlflow.log_artifact(str(imp_path))

                    # Log ONNX artifact if available
                    if onnx_path and os.path.exists(onnx_path):
                        mlflow.log_artifact(onnx_path)

            except Exception as e:
                logger.warning(f"Error logging to MLflow server: {e}")

        # Mirror run in ChronoGrid DB Model Registry
        with get_session() as session:
            # Determine next version
            latest_rec = (
                session.query(ModelRegistryRecord)
                .filter_by(model_name=self.model_name)
                .order_by(ModelRegistryRecord.version.desc())
                .first()
            )
            next_version = (latest_rec.version + 1) if latest_rec else 1

            if is_champion:
                # If promoting to champion, mark previous champion false
                session.query(ModelRegistryRecord).filter_by(
                    model_name=self.model_name, is_champion=True
                ).update({"is_champion": False, "stage": "Archived"})

            reg_rec = ModelRegistryRecord(
                model_id=run_id,
                model_name=self.model_name,
                version=next_version,
                model_type=model_type,
                stage="Production" if is_champion else stage,
                mae=float(metrics.get("mae", 0.0)),
                rmse=float(metrics.get("rmse", 0.0)),
                mape=float(metrics.get("mape", 0.0)),
                r2=float(metrics.get("r2", 0.0)),
                artifact_uri=model_artifact_path,
                onnx_uri=onnx_path,
                is_champion=is_champion,
                promoted_at=pd.Timestamp.utcnow().to_pydatetime() if is_champion else None,
                metadata_json=json.dumps({
                    "params": params,
                    "metrics": metrics,
                    "features": feature_names,
                }, default=str),
            )
            session.add(reg_rec)
            logger.info(
                f"Registered model {self.model_name} v{next_version} ({model_type}) - Stage: {reg_rec.stage}"
            )

        return run_id

    def get_champion_model_info(self) -> Optional[Dict[str, any]]:
        """Retrieves active Production Champion model metadata."""
        with get_session() as session:
            rec = (
                session.query(ModelRegistryRecord)
                .filter_by(model_name=self.model_name, is_champion=True)
                .first()
            )
            if rec:
                return {
                    "model_id": rec.model_id,
                    "version": rec.version,
                    "model_type": rec.model_type,
                    "stage": rec.stage,
                    "mae": rec.mae,
                    "rmse": rec.rmse,
                    "mape": rec.mape,
                    "onnx_uri": rec.onnx_uri,
                    "promoted_at": str(rec.promoted_at),
                }
            return None

