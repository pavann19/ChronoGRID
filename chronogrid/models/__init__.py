"""Models and evaluation package."""

from chronogrid.models.baselines import (
    MovingAverageBaseline,
    NaivePersistence24h,
    RidgeBaseline,
    WeeklySeasonalPersistence168h,
)
from chronogrid.models.evaluation import (
    compute_bootstrap_confidence_intervals,
    compute_metrics,
    diebold_mariano_test,
)
from chronogrid.models.trainer import ModelBenchmarkTrainer
from chronogrid.models.tree_models import (
    HistGradientBoostingModel,
    LightGBMModel,
    RandomForestModel,
    XGBoostModel,
)
from chronogrid.models.walk_forward import TimeSeriesWalkForwardValidator

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
