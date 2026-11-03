"""Models and evaluation package."""
from src.models.evaluation import (
    compute_metrics,
    compute_bootstrap_confidence_intervals,
    diebold_mariano_test,
)
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
from src.models.trainer import ModelBenchmarkTrainer

__all__ = [
    "compute_metrics",
    "compute_bootstrap_confidence_intervals",
    "diebold_mariano_test",
    "NaivePersistence24h",
    "WeeklySeasonalPersistence168h",
    "MovingAverageBaseline",
    "RidgeBaseline",
    "LightGBMModel",
    "XGBoostModel",
    "RandomForestModel",
    "HistGradientBoostingModel",
    "TimeSeriesWalkForwardValidator",
    "ModelBenchmarkTrainer",
]
