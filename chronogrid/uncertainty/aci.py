"""
Delayed-Feedback Adaptive Conformal Inference (ACI) manager for DQ-CP.
Enforces a 48-hour (2-day) transmission grid settlement feedback lag,
preventing premature information leakage and ensuring deterministic alpha updates.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

from chronogrid.uncertainty.schemas import DelayedFeedbackRecord


class DelayedFeedbackACIManager:
    """
    Manages online Adaptive Conformal Inference with physical settlement delay:
        alpha_t = clip(alpha_{t-1} + gamma * (alpha_target - err_{t - delay}), alpha_min, alpha_max)

    In European power markets (SMARD), physical grid load is verified with a 48h settlement delay.
    Maintains independent, namespaced state for each model head ('primary' vs 'fallback').
    """

    def __init__(
        self,
        target_alpha: float = 0.10,
        gamma: float = 0.01,
        delay_days: int = 2,
        alpha_min: float = 0.01,
        alpha_max: float = 0.50,
        head: str = "primary",
        state_file: Optional[Path] = None
    ):
        self.target_alpha = float(target_alpha)
        self.current_alpha = float(target_alpha)
        self.gamma = float(gamma)
        self.delay_days = int(delay_days)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.head = str(head)
        self.state_file = Path(state_file) if state_file else None

        # Tracks pending forecasts: origin_date -> dict(lower_bound, upper_bound, issuance_date)
        self.pending_forecasts: Dict[str, Dict[str, List[float]]] = {}
        # Tracks processed origin dates to guarantee idempotency against duplicate feedback
        self.processed_origins: Set[str] = set()
        # Tracks settled actuals to distinguish duplicates from revised observations
        self.settled_actuals: Dict[str, List[float]] = {}
        # Audit log of every state update
        self.update_history: List[DelayedFeedbackRecord] = []

        if self.state_file and self.state_file.exists():
            self.load_state(self.state_file)

    def register_forecast(
        self,
        origin_date: str,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray
    ) -> None:
        """
        Register a generated 24-hour prediction band for future ground-truth settlement.

        Args:
            origin_date: Unique date identifier for this forecast origin, e.g. "2024-04-02"
            lower_bound: Array of 24 lower interval bounds in MW
            upper_bound: Array of 24 upper interval bounds in MW
        """
        origin_key = str(origin_date).strip()
        self.pending_forecasts[origin_key] = {
            "lower": [float(x) for x in lower_bound],
            "upper": [float(x) for x in upper_bound]
        }
        self._auto_save()

    def submit_ground_truth(
        self,
        origin_date: str,
        actuals: np.ndarray,
        settlement_timestamp: str,
        effective_delay_days: Optional[int] = None
    ) -> Optional[DelayedFeedbackRecord]:
        """
        Process settled ground-truth load for a historical forecast origin.

        Requirements:
        1. Feedback cannot be applied if the observed delay is less than delay_days (settlement guard).
        2. Updates are strictly idempotent: duplicates are ignored.
        3. If origin is missing from pending forecasts, raises KeyError.

        Args:
            origin_date: The origin date being settled, e.g. "2024-04-02"
            actuals: Array of 24 true observed load values in MW
            settlement_timestamp: ISO UTC timestamp when ground-truth arrived
            effective_delay_days: Elapsed days between forecast issuance and settlement
        """
        origin_key = str(origin_date).strip()

        # Settlement delay guard: prevent premature feedback leakage
        observed_delay = effective_delay_days if effective_delay_days is not None else self.delay_days
        if observed_delay < self.delay_days:
            raise ValueError(
                f"Premature feedback rejected: Observed delay is {observed_delay} days, "
                f"but SMARD settlement guard requires at least {self.delay_days} days."
            )

        if origin_key not in self.pending_forecasts:
            raise KeyError(f"No pending forecast registered for origin '{origin_key}'.")

        expected_len = len(self.pending_forecasts[origin_key]["lower"])
        if len(actuals) != expected_len:
            raise ValueError(f"Actuals length {len(actuals)} does not match forecast length {expected_len}.")

        # Idempotency and revision check
        if origin_key in self.processed_origins:
            old_y = self.settled_actuals.get(origin_key)
            new_y = [float(x) for x in actuals]
            if old_y is not None and len(new_y) == len(old_y) and np.allclose(new_y, old_y):
                # Exact duplicate: return idempotent status without state modification
                return DelayedFeedbackRecord(
                    forecast_origin=origin_key,
                    settlement_timestamp=settlement_timestamp,
                    effective_delay_hours=int(observed_delay * 24),
                    miscoverage_occurred=False,
                    empirical_coverage=1.0,
                    alpha_before=self.current_alpha,
                    alpha_after=self.current_alpha,
                    gamma=self.gamma,
                    status="DUPLICATE_IGNORED"
                )
            else:
                # Revision detected: reject explicitly to maintain monotonic audit trail
                return DelayedFeedbackRecord(
                    forecast_origin=origin_key,
                    settlement_timestamp=settlement_timestamp,
                    effective_delay_hours=int(observed_delay * 24),
                    miscoverage_occurred=False,
                    empirical_coverage=1.0,
                    alpha_before=self.current_alpha,
                    alpha_after=self.current_alpha,
                    gamma=self.gamma,
                    status="REJECTED_REVISION"
                )

        forecast = self.pending_forecasts[origin_key]
        lowers = np.array(forecast["lower"], dtype=np.float64)
        uppers = np.array(forecast["upper"], dtype=np.float64)
        y = np.array(actuals, dtype=np.float64)

        # Determine hourly coverage
        covered = (y >= lowers) & (y <= uppers)
        empirical_coverage = float(np.mean(covered))
        # Miscoverage occurred if any hour failed coverage (vector error)
        miscoverage_occurred = not np.all(covered)

        # ACI error signal: 1 if miscoverage occurred, 0 otherwise
        err_signal = 1.0 if miscoverage_occurred else 0.0

        alpha_before = self.current_alpha
        # State update: alpha_{t+1} = alpha_t + gamma * (alpha_target - err)
        # Note: If err=1 (miscoverage), alpha decreases -> 1 - alpha increases -> interval widens!
        new_alpha = alpha_before + self.gamma * (self.target_alpha - err_signal)
        new_alpha = float(np.clip(new_alpha, self.alpha_min, self.alpha_max))
        self.current_alpha = new_alpha

        # Mark as processed and store settled actuals
        self.processed_origins.add(origin_key)
        self.settled_actuals[origin_key] = [float(x) for x in y]

        record = DelayedFeedbackRecord(
            forecast_origin=origin_key,
            settlement_timestamp=settlement_timestamp,
            effective_delay_hours=int(observed_delay * 24),
            miscoverage_occurred=miscoverage_occurred,
            empirical_coverage=empirical_coverage,
            alpha_before=alpha_before,
            alpha_after=new_alpha,
            gamma=self.gamma,
            status="PROCESSED"
        )
        self.update_history.append(record)
        self._auto_save()
        return record

    def get_current_alpha(self) -> float:
        """Returns the current active miscoverage parameter."""
        return float(self.current_alpha)

    def _auto_save(self) -> None:
        if self.state_file:
            self.save_state(self.state_file)

    def save_state(self, path: Path) -> None:
        """Atomically persist state to JSON."""
        state_dict = {
            "head": self.head,
            "target_alpha": self.target_alpha,
            "current_alpha": self.current_alpha,
            "gamma": self.gamma,
            "delay_days": self.delay_days,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "pending_forecasts": self.pending_forecasts,
            "processed_origins": list(self.processed_origins),
            "settled_actuals": self.settled_actuals,
            "update_history": [r.dict() for r in self.update_history]
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(state_dict, f, indent=2)
        tmp_path.replace(path)

    def load_state(self, path: Path) -> None:
        """Load state from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.head = str(data.get("head", self.head))
        self.target_alpha = float(data.get("target_alpha", 0.10))
        self.current_alpha = float(data.get("current_alpha", 0.10))
        self.gamma = float(data.get("gamma", 0.01))
        self.delay_days = int(data.get("delay_days", 2))
        self.alpha_min = float(data.get("alpha_min", 0.01))
        self.alpha_max = float(data.get("alpha_max", 0.50))
        self.pending_forecasts = data.get("pending_forecasts", {})
        self.processed_origins = set(data.get("processed_origins", []))
        self.settled_actuals = data.get("settled_actuals", {})
        self.update_history = [DelayedFeedbackRecord(**r) for r in data.get("update_history", [])]

    def reset(self) -> None:
        """Reset ACI state to initial conditions."""
        self.current_alpha = self.target_alpha
        self.pending_forecasts.clear()
        self.processed_origins.clear()
        self.settled_actuals.clear()
        self.update_history.clear()
        self._auto_save()
