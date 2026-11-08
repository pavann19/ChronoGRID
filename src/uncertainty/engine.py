"""
Data-Quality-Conditioned Conformal Prediction (DQ-CP) Uncertainty Engine.
Integrates delivery-hour normalization, finite-sample conformal calibration pools,
delayed-feedback ACI, and DDIR reliability-state conditioning.
"""

import math
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np

from src.reliability.state_machine import ReliabilityState
from src.uncertainty.schemas import UncertaintyResult, CalibrationTable, DelayedFeedbackRecord
from src.uncertainty.normalization import DeliveryHourMADNormalizer
from src.uncertainty.conformal import ConformalCalibrationPool
from src.uncertainty.aci import DelayedFeedbackACIManager


class DQCPUncertaintyEngine:
    """
    Core engine providing calibrated prediction intervals conditioned on input data quality.

    State Policies:
    - HEALTHY: Primary model point forecast + Primary conformal calibration pool.
    - DEGRADED: Conditioned on active head; primary or fallback conformal pool with degradation annotation.
    - STALE: Fallback model point forecast + Fallback conformal calibration pool.
    - UNAVAILABLE: Does NOT invent an interval. Returns uncertainty_available = False.
    - INVALID: Rejects uncertainty. Returns uncertainty_available = False.

    Maintains completely isolated ACI state managers for Primary and Fallback model heads.
    """

    def __init__(
        self,
        primary_pool: Optional[ConformalCalibrationPool] = None,
        fallback_pool: Optional[ConformalCalibrationPool] = None,
        primary_aci_manager: Optional[DelayedFeedbackACIManager] = None,
        fallback_aci_manager: Optional[DelayedFeedbackACIManager] = None,
        aci_manager: Optional[DelayedFeedbackACIManager] = None,
        version: str = "v1.0-research"
    ):
        self.primary_pool = primary_pool
        self.fallback_pool = fallback_pool
        # Isolate primary and fallback ACI state managers
        self.primary_aci_manager = primary_aci_manager or aci_manager or DelayedFeedbackACIManager(head="primary")
        self.fallback_aci_manager = fallback_aci_manager or DelayedFeedbackACIManager(head="fallback")
        self.version = version

    @property
    def aci_manager(self) -> DelayedFeedbackACIManager:
        """Backwards compatibility alias for primary ACI manager."""
        return self.primary_aci_manager

    def generate_interval(
        self,
        point_predictions: np.ndarray,
        delivery_hours: np.ndarray,
        reliability_state: ReliabilityState,
        forecast_head: str = "primary",
        calibration_method: str = "pooled_marginal_mad",
        target_coverage: float = 0.90,
        forecast_origin: Optional[str] = None
    ) -> UncertaintyResult:
        """
        Generate calibrated uncertainty bounds conditioned on DDIR reliability state.

        Args:
            point_predictions: 1D array of 24 point predictions in MW
            delivery_hours: 1D array of 24 delivery hours (0..23)
            reliability_state: Active ReliabilityState from DDIR router
            forecast_head: 'primary' or 'fallback'
            calibration_method: 'pooled_marginal_mad' (Target A) or 'daily_sup_norm' (Target B)
            target_coverage: Desired marginal or simultaneous coverage (default 0.90)
            forecast_origin: Origin identifier (date string, e.g. '2024-04-02')

        Returns:
            UncertaintyResult
        """
        preds = np.asarray(point_predictions, dtype=np.float64)
        hours = np.asarray(delivery_hours, dtype=np.int64)

        if len(preds) != 24 or len(hours) != 24:
            raise ValueError(f"DQ-CP requires exactly 24 hourly steps, got {len(preds)}.")

        # Route to appropriate model head, pool, and isolated ACI manager
        is_fallback = (forecast_head == "fallback" or (reliability_state in (ReliabilityState.STALE, ReliabilityState.DEGRADED) and forecast_head == "fallback"))
        active_aci = self.fallback_aci_manager if is_fallback else self.primary_aci_manager
        current_alpha = active_aci.get_current_alpha()

        # Handle INVALID state
        if reliability_state == ReliabilityState.INVALID:
            return UncertaintyResult(
                forecast_mw=[float(x) for x in preds],
                lower_bound_mw=None,
                upper_bound_mw=None,
                target_coverage=target_coverage,
                calibration_method=calibration_method,
                model_id=forecast_head,
                calibration_version=self.version,
                reliability_state=reliability_state.value,
                forecast_origin=forecast_origin,
                uncertainty_available=False,
                is_unbounded=False,
                mean_interval_width_mw=None,
                current_alpha=current_alpha,
                degradation_flag="Input validation failed: INVALID state."
            )

        # Handle UNAVAILABLE state - DO NOT invent an interval!
        if reliability_state == ReliabilityState.UNAVAILABLE:
            return UncertaintyResult(
                forecast_mw=[float(x) for x in preds],
                lower_bound_mw=None,
                upper_bound_mw=None,
                target_coverage=target_coverage,
                calibration_method=calibration_method,
                model_id=forecast_head,
                calibration_version=self.version,
                reliability_state=reliability_state.value,
                forecast_origin=forecast_origin,
                uncertainty_available=False,
                is_unbounded=False,
                mean_interval_width_mw=None,
                current_alpha=current_alpha,
                degradation_flag="Weather telemetry completely unavailable; uncertainty generation halted."
            )

        if is_fallback:
            active_pool = self.fallback_pool
            model_id = "research_fallback"
        else:
            active_pool = self.primary_pool
            model_id = "research_primary"

        if active_pool is None:
            return UncertaintyResult(
                forecast_mw=[float(x) for x in preds],
                lower_bound_mw=None,
                upper_bound_mw=None,
                target_coverage=target_coverage,
                calibration_method=calibration_method,
                model_id=model_id,
                calibration_version=self.version,
                reliability_state=reliability_state.value,
                forecast_origin=forecast_origin,
                uncertainty_available=False,
                is_unbounded=False,
                mean_interval_width_mw=None,
                current_alpha=current_alpha,
                degradation_flag="Calibration pool not loaded for requested head."
            )

        # Compute conformal quantile
        is_exploratory = False
        if calibration_method == "daily_sup_norm":
            q, rank_k, is_exploratory = active_pool.get_simultaneous_quantile(current_alpha)
        elif calibration_method == "pooled_marginal_mad":
            q, rank_k = active_pool.get_marginal_quantile(current_alpha)
        else:
            raise ValueError(f"Unknown calibration method: '{calibration_method}'")

        # Check for infinite quantile / unbounded interval
        is_unbounded = math.isinf(q)
        if is_unbounded:
            lowers = np.full(len(preds), -np.inf, dtype=np.float64)
            uppers = np.full(len(preds), np.inf, dtype=np.float64)
            mean_width = float("inf")
        else:
            lowers, uppers = active_pool.normalizer.scale_intervals(preds, hours, q)
            mean_width = float(np.mean(uppers - lowers))

        # Build degradation annotation if applicable
        degradation_flag = None
        if reliability_state == ReliabilityState.DEGRADED:
            degradation_flag = f"Telemetry degraded; active head '{forecast_head}' using separate {model_id} pool."
        elif reliability_state == ReliabilityState.STALE:
            degradation_flag = "Weather telemetry stale; DDIR routed to fallback pool."
        if is_exploratory:
            degradation_flag = (degradation_flag + " " if degradation_flag else "") + "[EXPLORATORY: 95% simultaneous daily band]"
        if is_unbounded:
            degradation_flag = (degradation_flag + " " if degradation_flag else "") + "[UNBOUNDED: Required finite-sample rank exceeds sample size (k > n). Interval is infinite.]"

        # Register forecast with head-specific ACI manager
        if forecast_origin:
            active_aci.register_forecast(forecast_origin, lowers, uppers)

        return UncertaintyResult(
            forecast_mw=[float(x) for x in preds],
            lower_bound_mw=[float(x) for x in lowers],
            upper_bound_mw=[float(x) for x in uppers],
            target_coverage=target_coverage,
            calibration_method=calibration_method,
            model_id=model_id,
            calibration_version=self.version,
            reliability_state=reliability_state.value,
            forecast_origin=forecast_origin,
            uncertainty_available=True,
            is_unbounded=is_unbounded,
            mean_interval_width_mw=mean_width,
            current_alpha=current_alpha,
            degradation_flag=degradation_flag
        )

    def submit_feedback(
        self,
        forecast_origin: str,
        actuals: np.ndarray,
        settlement_timestamp: str,
        effective_delay_days: int = 2,
        head: Optional[str] = None
    ) -> Optional[DelayedFeedbackRecord]:
        """Submit ground-truth settlement for delayed ACI feedback to the appropriate model head."""
        origin_key = str(forecast_origin).strip()
        target_head = head
        if not target_head:
            # Deterministically identify which manager holds this forecast origin
            if origin_key in self.primary_aci_manager.pending_forecasts or origin_key in self.primary_aci_manager.processed_origins:
                target_head = "primary"
            elif origin_key in self.fallback_aci_manager.pending_forecasts or origin_key in self.fallback_aci_manager.processed_origins:
                target_head = "fallback"
            else:
                target_head = "primary"

        mgr = self.primary_aci_manager if target_head == "primary" else self.fallback_aci_manager
        return mgr.submit_ground_truth(
            origin_date=forecast_origin,
            actuals=actuals,
            settlement_timestamp=settlement_timestamp,
            effective_delay_days=effective_delay_days
        )


def load_dqcp_engine(
    calibration_dir: Optional[Path] = None,
    aci_state_file: Optional[Path] = None,
    model_paths: Optional[Dict[str, Path]] = None,
) -> Optional[DQCPUncertaintyEngine]:
    """
    Factory to load and initialize DQCPUncertaintyEngine from serialized calibration tables.
    Maintains isolated persistence files for primary and fallback ACI managers.
    """
    import json
    import hashlib
    from pathlib import Path

    calib_path = Path(calibration_dir) if calibration_dir else Path("artifacts/calibration")
    prim_file = calib_path / "primary_calibration_table.json"
    fall_file = calib_path / "fallback_calibration_table.json"

    if not prim_file.exists() or not fall_file.exists():
        return None

    try:
        with open(prim_file, "r") as f:
            prim_data = json.load(f)
        with open(fall_file, "r") as f:
            fall_data = json.load(f)

        table_prim = CalibrationTable(**prim_data)
        table_fall = CalibrationTable(**fall_data)

        # Verify identity before constructing pools or touching adaptive state.
        paths = model_paths if model_paths is not None else {
            "primary": Path("artifacts/models/champion_model.onnx"),
            "fallback": Path("artifacts/models/load_only_model.onnx"),
        }
        if set(paths) != {"primary", "fallback"}:
            raise ValueError("Both primary and fallback model artifact paths are required")
        for head, table in (("primary", table_prim), ("fallback", table_fall)):
            actual_hash = hashlib.sha256(Path(paths[head]).read_bytes()).hexdigest()
            if table.model_sha256 != actual_hash:
                raise ValueError(f"Calibration model identity mismatch for {head}")

        pool_prim = ConformalCalibrationPool.from_calibration_table(table_prim)
        pool_fall = ConformalCalibrationPool.from_calibration_table(table_fall)

        base_state = Path(aci_state_file) if aci_state_file else Path("data/dqcp_aci_state.json")
        prim_state_file = base_state.with_name(f"{base_state.stem}_primary{base_state.suffix}")
        fall_state_file = base_state.with_name(f"{base_state.stem}_fallback{base_state.suffix}")

        primary_aci = DelayedFeedbackACIManager(
            target_alpha=0.10,
            gamma=0.01,
            delay_days=2,
            head="primary",
            state_file=prim_state_file
        )
        fallback_aci = DelayedFeedbackACIManager(
            target_alpha=0.10,
            gamma=0.01,
            delay_days=2,
            head="fallback",
            state_file=fall_state_file
        )

        return DQCPUncertaintyEngine(
            primary_pool=pool_prim,
            fallback_pool=pool_fall,
            primary_aci_manager=primary_aci,
            fallback_aci_manager=fallback_aci,
            version=table_prim.version
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load DQCP engine: {exc}")
        return None
