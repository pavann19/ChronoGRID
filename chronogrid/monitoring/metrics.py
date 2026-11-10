"""
Prometheus and Operational Metrics Exporter (P5.11).
"""

from prometheus_client import Gauge

# Operational metrics
OVERALL_PSI_SCORE = Gauge(
    "chronogrid_overall_psi_score",
    "Current Population Stability Index score across key features"
)
DRIFT_DETECTED_FLAG = Gauge(
    "chronogrid_drift_detected_flag",
    "Flag indicating whether drift was detected in latest check (1=True, 0=False)"
)
RESIDUAL_MAPE = Gauge(
    "chronogrid_residual_mape",
    "Latest residual Mean Absolute Percentage Error on ground truth labels"
)


def update_drift_prometheus_metrics(psi: float, drift_detected: bool, mape: float = None):
    """Updates operational Prometheus gauges for drift and model error."""
    OVERALL_PSI_SCORE.set(float(psi))
    DRIFT_DETECTED_FLAG.set(1.0 if drift_detected else 0.0)
    if mape is not None:
        RESIDUAL_MAPE.set(float(mape))

