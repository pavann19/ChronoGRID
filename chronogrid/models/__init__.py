"""Models and evaluation package."""
from chronogrid.models.evaluation import (
    compute_metrics,
    compute_bootstrap_confidence_intervals,
    diebold_mariano_test,
)
from chronogrid.models.baselines import (
    NaivePersistence24h,
    WeeklySeasonalPersistence168h,
    MovingAverageBaseline,
    RidgeBaseline,
)
from chronogrid.models.tree_models import (
    LightGBMModel,
    XGBoostModel,
    RandomForestModel,
    HistGradientBoostingModel,
)
from chronogrid.models.walk_forward import TimeSeriesWalkForwardValidator
from chronogrid.models.trainer import ModelBenchmarkTrainer

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

