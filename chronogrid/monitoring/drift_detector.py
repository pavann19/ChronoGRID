"""
Data and Model Drift Detection Suite (P5.9).
Computes Kolmogorov-Smirnov test, Population Stability Index (PSI), Wasserstein Distance,
and Residual Concept Drift with Evidently-compatible HTML and JSON reporting.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from chronogrid.db.database import get_session
from chronogrid.db.models import DriftReportRecord

logger = logging.getLogger(__name__)


@dataclass
class FeatureDriftResult:
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    wasserstein_distance: float
    drift_detected: bool
    status: str  # STABLE, WARNING, ALERT


@dataclass
class DriftReport:
    report_id: str
    timestamp: str
    overall_drift_score: float
    drift_detected: bool
    retraining_triggered: bool
    feature_metrics: Dict[str, FeatureDriftResult]
    residual_drift: Optional[Dict[str, float]] = None


class DriftDetector:
    """
    Evaluates covariate and concept drift between reference and current windows.
    """

    def __init__(
        self,
        psi_threshold_warning: float = 0.10,
        psi_threshold_alert: float = 0.25,
        ks_alpha: float = 0.05,
        mape_degradation_alert: float = 0.20,
    ):
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_alert = psi_threshold_alert
        self.ks_alpha = ks_alpha
        self.mape_degradation_alert = mape_degradation_alert

    def calculate_psi(
        self, reference: np.ndarray, current: np.ndarray, num_buckets: int = 10
    ) -> float:
        """
        Calculates Population Stability Index (PSI) between reference and current samples.
        """
        ref = reference[~np.isnan(reference)]
        curr = current[~np.isnan(current)]
        if len(ref) == 0 or len(curr) == 0:
            return 0.0

        # Create quantile bins based on reference distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(ref, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Ensure bins are strictly increasing (handle discrete ties)
        bins = np.unique(bins)
        if len(bins) < 3:
            return 0.0

        ref_counts = np.histogram(ref, bins=bins)[0]
        curr_counts = np.histogram(curr, bins=bins)[0]

        # Convert to percentages with epsilon smoothing to prevent div by zero
        eps = 1e-4
        ref_pct = (ref_counts / len(ref)) + eps
        curr_pct = (curr_counts / len(curr)) + eps

        ref_pct /= np.sum(ref_pct)
        curr_pct /= np.sum(curr_pct)

        psi_val: float = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(max(0.0, psi_val))

    def evaluate_feature(
        self, feature_name: str, ref_series: pd.Series, curr_series: pd.Series
    ) -> FeatureDriftResult:
        """
        Evaluates drift metrics for a single feature.
        """
        ref_vals = ref_series.dropna().values
        curr_vals = curr_series.dropna().values

        if len(ref_vals) < 5 or len(curr_vals) < 5:
            return FeatureDriftResult(
                feature_name=feature_name,
                ks_statistic=0.0,
                ks_pvalue=1.0,
                psi=0.0,
                wasserstein_distance=0.0,
                drift_detected=False,
                status="STABLE",
            )

        # 1. Kolmogorov-Smirnov test
        ks_res = stats.ks_2samp(ref_vals, curr_vals)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        # 2. Population Stability Index (PSI)
        psi = self.calculate_psi(ref_vals, curr_vals)

        # 3. 1-Wasserstein Distance (Earth Mover's Distance)
        try:
            w_dist = float(stats.wasserstein_distance(ref_vals, curr_vals))
        except Exception:
            w_dist = 0.0

        # Drift decision logic
        if psi >= self.psi_threshold_alert or (
            ks_pval < self.ks_alpha and psi >= self.psi_threshold_warning
        ):
            status = "ALERT"
            drift_detected = True
        elif psi >= self.psi_threshold_warning or ks_pval < self.ks_alpha:
            status = "WARNING"
            drift_detected = True
        else:
            status = "STABLE"
            drift_detected = False

        return FeatureDriftResult(
            feature_name=feature_name,
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pval, 5),
            psi=round(psi, 4),
            wasserstein_distance=round(w_dist, 4),
            drift_detected=drift_detected,
            status=status,
        )

    def evaluate_residual_drift(
        self,
        y_ref: np.ndarray,
        y_ref_pred: np.ndarray,
        y_curr: np.ndarray,
        y_curr_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluates concept drift in prediction residuals between reference and current windows.
        """
        ref_mae = float(np.mean(np.abs(y_ref - y_ref_pred)))
        curr_mae = float(np.mean(np.abs(y_curr - y_curr_pred)))
        ref_mape = float(np.mean(np.abs((y_ref - y_ref_pred) / y_ref)) * 100.0)
        curr_mape = float(np.mean(np.abs((y_curr - y_curr_pred) / y_curr)) * 100.0)

        degradation_pct = (curr_mae - ref_mae) / max(ref_mae, 1e-5)

        # CUSUM statistic on current residuals
        residuals = y_curr - y_curr_pred
        cusum_stat = float(
            np.max(np.abs(np.cumsum(residuals - np.mean(residuals))))
            / (np.std(residuals) * np.sqrt(len(residuals)) + 1e-5)
        )

        return {
            "ref_mae": round(ref_mae, 2),
            "curr_mae": round(curr_mae, 2),
            "ref_mape": round(ref_mape, 3),
            "curr_mape": round(curr_mape, 3),
            "mae_degradation_pct": round(degradation_pct * 100.0, 2),
            "cusum_score": round(cusum_stat, 3),
            "concept_drift_detected": bool(
                degradation_pct >= self.mape_degradation_alert or cusum_stat > 3.0
            ),
        }

    def compute_drift(
        self,
        ref_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        features: Optional[List[str]] | None = None,
        y_ref_tuple: Optional[Tuple[np.ndarray, np.ndarray]] | None = None,
        y_curr_tuple: Optional[Tuple[np.ndarray, np.ndarray]] | None = None,
        report_output_html: str = "artifacts/reports/drift_report.html",
    ) -> DriftReport:
        """
        Executes complete drift analysis, generates HTML report, and persists to DB.
        """
        report_id = f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if features is None:
            # Default to key numeric columns
            features = [
                c
                for c in [
                    "temperature_c",
                    "grid_load_mw",
                    "wind_speed_ms",
                    "solar_radiation_wm2",
                    "humidity_pct",
                ]
                if c in ref_df.columns and c in curr_df.columns
            ]

        feature_metrics = {}
        alert_count = 0
        warning_count = 0

        for feat in features:
            res = self.evaluate_feature(feat, ref_df[feat], curr_df[feat])
            feature_metrics[feat] = res
            if res.status == "ALERT":
                alert_count += 1
            elif res.status == "WARNING":
                warning_count += 1

        overall_drift_score = (
            float(np.mean([m.psi for m in feature_metrics.values()])) if feature_metrics else 0.0
        )

        # Evaluate residual drift if predictions provided
        residual_metrics = None
        concept_drift = False
        if y_ref_tuple is not None and y_curr_tuple is not None:
            residual_metrics = self.evaluate_residual_drift(
                y_ref_tuple[0], y_ref_tuple[1], y_curr_tuple[0], y_curr_tuple[1]
            )
            concept_drift = bool(residual_metrics["concept_drift_detected"])

        # Trigger retraining if: >= 1 feature ALERT, or overall PSI > 0.25, or concept drift detected
        drift_detected = bool(alert_count > 0 or warning_count >= 2 or concept_drift)
        retraining_triggered = bool(
            alert_count > 0 or overall_drift_score >= self.psi_threshold_alert or concept_drift
        )

        report = DriftReport(
            report_id=report_id,
            timestamp=datetime.utcnow().isoformat(),
            overall_drift_score=round(overall_drift_score, 4),
            drift_detected=drift_detected,
            retraining_triggered=retraining_triggered,
            feature_metrics=feature_metrics,
            residual_drift=residual_metrics,
        )

        # Generate HTML report
        self._export_html_report(report, ref_df, curr_df, report_output_html)

        # Persist to database
        try:
            with get_session() as session:
                rec = DriftReportRecord(
                    report_id=report_id,
                    reference_start=pd.to_datetime(ref_df["timestamp"].min()).to_pydatetime()
                    if "timestamp" in ref_df.columns
                    else datetime.utcnow(),
                    reference_end=pd.to_datetime(ref_df["timestamp"].max()).to_pydatetime()
                    if "timestamp" in ref_df.columns
                    else datetime.utcnow(),
                    current_start=pd.to_datetime(curr_df["timestamp"].min()).to_pydatetime()
                    if "timestamp" in curr_df.columns
                    else datetime.utcnow(),
                    current_end=pd.to_datetime(curr_df["timestamp"].max()).to_pydatetime()
                    if "timestamp" in curr_df.columns
                    else datetime.utcnow(),
                    overall_drift_score=report.overall_drift_score,
                    drift_detected=report.drift_detected,
                    retraining_triggered=report.retraining_triggered,
                    feature_metrics_json=json.dumps(
                        {k: asdict(v) for k, v in feature_metrics.items()}
                    ),
                    evidently_report_path=str(report_output_html),
                )
                session.add(rec)
        except Exception as e:
            logger.warning(f"Could not persist drift report to DB: {e}")

        return report

    def _export_html_report(
        self, report: DriftReport, ref_df: pd.DataFrame, curr_df: pd.DataFrame, output_path: str
    ):
        """Generates a standalone, styled HTML drift dashboard."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        rows_html = ""
        for feat, m in report.feature_metrics.items():
            badge_color = (
                "#dc2626"
                if m.status == "ALERT"
                else ("#d97706" if m.status == "WARNING" else "#16a34a")
            )
            rows_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 12px; font-weight: 600;">{feat}</td>
                <td style="padding: 12px;"><span style="background: {badge_color}; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600;">{m.status}</span></td>
                <td style="padding: 12px; font-family: monospace;">{m.psi:.4f}</td>
                <td style="padding: 12px; font-family: monospace;">{m.ks_statistic:.4f}</td>
                <td style="padding: 12px; font-family: monospace;">{m.ks_pvalue:.5f}</td>
                <td style="padding: 12px; font-family: monospace;">{m.wasserstein_distance:.4f}</td>
            </tr>
            """

        residual_html = ""
        if report.residual_drift:
            res = report.residual_drift
            res_badge = "#dc2626" if res["concept_drift_detected"] else "#16a34a"
            residual_html = f"""
            <div style="margin-top: 24px; padding: 18px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0;">
                <h3 style="margin-top: 0; color: #1e293b;">Model Concept & Residual Drift</h3>
                <p>Status: <span style="background: {res_badge}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{"DRIFT DETECTED" if res["concept_drift_detected"] else "STABLE"}</span></p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 12px;">
                    <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #cbd5e1;"><strong>Ref MAE:</strong> {res["ref_mae"]} MW</div>
                    <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #cbd5e1;"><strong>Current MAE:</strong> {res["curr_mae"]} MW</div>
                    <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #cbd5e1;"><strong>Degradation:</strong> {res["mae_degradation_pct"]}%</div>
                    <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #cbd5e1;"><strong>CUSUM Score:</strong> {res["cusum_score"]}</div>
                </div>
            </div>
            """

        status_banner = "#fef2f2" if report.retraining_triggered else "#f0fdf4"
        banner_border = "#f87171" if report.retraining_triggered else "#4ade80"
        banner_title = (
            "RETRAINING TRIGGERED: SIGNIFICANT DRIFT DETECTED"
            if report.retraining_triggered
            else "STABLE: NO ACTION REQUIRED"
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ChronoGrid Drift Detection Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f1f5f9; margin: 0; padding: 24px; color: #0f172a; }}
        .card {{ background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 24px; max-width: 1000px; margin: 0 auto; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; text-align: left; }}
        th {{ background: #f8fafc; padding: 12px; font-weight: 600; border-bottom: 2px solid #e2e8f0; }}
    </style>
</head>
<body>
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 16px;">
            <div>
                <h1 style="margin: 0; font-size: 24px; color: #0f172a;">ChronoGrid Drift Monitoring Report</h1>
                <p style="margin: 4px 0 0 0; color: #64748b;">Report ID: {report.report_id} | Generated: {report.timestamp}</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 28px; font-weight: 700; color: #0284c7;">{report.overall_drift_score:.4f}</div>
                <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">Overall PSI Score</div>
            </div>
        </div>

        <div style="margin-top: 20px; padding: 16px; border-radius: 8px; background: {status_banner}; border-left: 6px solid {banner_border};">
            <strong style="font-size: 16px;">{banner_title}</strong>
            <p style="margin: 4px 0 0 0; font-size: 14px;">Reference Window: {len(ref_df)} hours | Current Window: {len(curr_df)} hours</p>
        </div>

        {residual_html}

        <h3 style="margin-top: 24px; color: #1e293b;">Feature Distribution Metrics</h3>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Drift Status</th>
                    <th>PSI</th>
                    <th>KS Statistic</th>
                    <th>KS p-value</th>
                    <th>Wasserstein Dist</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Drift report exported to {out_p}")
