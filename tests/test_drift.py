"""
Tests for Drift Detection (P5.9).
"""

import numpy as np
import pandas as pd

from src.monitoring.drift_detector import DriftDetector


def test_psi_identical_distributions():
    detector = DriftDetector()
    rng = np.random.RandomState(42)
    s1 = rng.normal(50, 10, 1000)
    s2 = rng.normal(50, 10, 1000)
    psi = detector.calculate_psi(s1, s2)
    assert psi < 0.05 # Negligible drift


def test_psi_shifted_distribution():
    detector = DriftDetector()
    rng = np.random.RandomState(42)
    s1 = rng.normal(50, 10, 1000)
    s2 = rng.normal(85, 10, 1000) # Massive shift
    psi = detector.calculate_psi(s1, s2)
    assert psi > 0.50 # High drift alert


def test_drift_report_generation(tmp_path, isolated_application_database):
    detector = DriftDetector()
    ref_df = pd.DataFrame({"temperature_c": np.random.normal(15, 5, 200)})
    curr_df = pd.DataFrame({"temperature_c": np.random.normal(30, 5, 200)}) # Heatwave drift
    report = detector.compute_drift(
        ref_df,
        curr_df,
        features=["temperature_c"],
        report_output_html=str(tmp_path / "drift_report.html"),
    )
    assert report.drift_detected is True
    assert "temperature_c" in report.feature_metrics
    assert report.feature_metrics["temperature_c"].status == "ALERT"

