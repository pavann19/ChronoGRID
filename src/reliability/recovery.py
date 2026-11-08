"""
Hysteresis Recovery Manager for ChronoGrid DDIR.
Eliminates flapping by requiring K consecutive healthy cycles before transitioning back to Primary Head.
Includes static/frozen telemetry sensor detection and persistent state backing.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
from typing import Any, Optional, Tuple

from src.reliability.state_machine import ReliabilityState
from src.reliability.persistence import DDIRStatePersistence, DDIRPersistedState

logger = logging.getLogger(__name__)


class HysteresisRecoveryManager:
    """
    Manages state transitions with anti-flapping hysteresis.
    - Demotion to Fallback Head occurs immediately upon the first failure (N = 1).
    - Recovery to Primary Head requires K distinct, advancing, healthy cycles (default K = 2).
    - Prevents replayed, duplicate, or frozen/static sensor telemetry from triggering recovery.
    - Supports atomic file-backed state persistence to eliminate cold-start state inversion.
    """

    def __init__(
        self,
        recovery_threshold: int = 2,
        start_healthy: bool = True,
        state_file_path: Optional[str] = None,
    ):
        self.recovery_threshold = max(1, recovery_threshold)
        self.state_file_path = state_file_path
        self.persistence = DDIRStatePersistence(state_file_path) if state_file_path else None
        self._lock = threading.Lock()

        # Default in-memory state
        self.consecutive_healthy_count: int = self.recovery_threshold if start_healthy else 0
        self.current_effective_state: ReliabilityState = (
            ReliabilityState.HEALTHY if start_healthy else ReliabilityState.UNAVAILABLE
        )
        self.last_healthy_timestamp = None
        self.last_healthy_fingerprint: Optional[str] = None
        self.last_healthy_content_fingerprint: Optional[str] = None

        # Restore from persisted state if available
        if self.persistence:
            loaded = self.persistence.load_state()
            if loaded:
                try:
                    self.current_effective_state = ReliabilityState(loaded.effective_state)
                    self.consecutive_healthy_count = loaded.consecutive_healthy_count
                    self.recovery_threshold = loaded.recovery_threshold
                    self.last_healthy_timestamp = loaded.last_healthy_timestamp
                    self.last_healthy_fingerprint = loaded.last_healthy_fingerprint
                    self.last_healthy_content_fingerprint = loaded.last_healthy_content_fingerprint
                    logger.info(
                        f"Restored DDIR state from {state_file_path}: state={self.current_effective_state.value}, "
                        f"healthy_count={self.consecutive_healthy_count}"
                    )
                except Exception as e:
                    logger.warning(f"Error restoring persisted DDIR state: {e}")

    def _persist(self):
        """Persists current state to file if persistence is enabled."""
        if self.persistence:
            rec = DDIRPersistedState(
                effective_state=self.current_effective_state.value,
                consecutive_healthy_count=self.consecutive_healthy_count,
                recovery_threshold=self.recovery_threshold,
                last_healthy_timestamp=str(self.last_healthy_timestamp) if self.last_healthy_timestamp else None,
                last_healthy_fingerprint=self.last_healthy_fingerprint,
                last_healthy_content_fingerprint=self.last_healthy_content_fingerprint,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self.persistence.save_state(rec)

    def process_state(
        self,
        raw_state: ReliabilityState,
        raw_reason: str,
        source_timestamp: Any = None,
        payload_fingerprint: Optional[str] = None,
        content_fingerprint: Optional[str] = None,
    ) -> Tuple[ReliabilityState, str, str]:
        """
        Processes a raw evaluated state through the hysteresis window.
        Recovery requires distinct, advancing, sufficiently fresh source updates with active variance.

        Returns:
            (effective_state, active_head, final_reason)
            active_head is either "PRIMARY_WEATHER" or "FALLBACK_LOAD_ONLY"
        """
        with self._lock:
            # 1. Healthy Candidate State
            if raw_state == ReliabilityState.HEALTHY:
                is_duplicate = False
                duplicate_msg = ""

                # Check duplicate raw payload fingerprint
                if payload_fingerprint is not None and payload_fingerprint == self.last_healthy_fingerprint:
                    is_duplicate = True
                    duplicate_msg = "Duplicate payload fingerprint detected; distinct source update required"

                # Check non-advancing timestamp
                if source_timestamp is not None and self.last_healthy_timestamp is not None:
                    if str(source_timestamp) <= str(self.last_healthy_timestamp):
                        is_duplicate = True
                        duplicate_msg = (
                            f"Non-advancing timestamp ({source_timestamp} <= {self.last_healthy_timestamp}); "
                            f"distinct source update required"
                        )

                # Check static/frozen telemetry readings across advancing timestamps
                if (
                    not is_duplicate
                    and source_timestamp is not None
                    and self.last_healthy_timestamp is not None
                    and str(source_timestamp) > str(self.last_healthy_timestamp)
                    and content_fingerprint is not None
                    and self.last_healthy_content_fingerprint is not None
                    and content_fingerprint == self.last_healthy_content_fingerprint
                ):
                    is_duplicate = True
                    duplicate_msg = (
                        "Static/frozen telemetry readings detected across advancing timestamps; "
                        "active variance required for recovery"
                    )

                if is_duplicate and self.consecutive_healthy_count < self.recovery_threshold:
                    return (
                        self.current_effective_state,
                        "FALLBACK_LOAD_ONLY",
                        f"Recovery held: {duplicate_msg} ({self.consecutive_healthy_count}/{self.recovery_threshold} distinct updates verified)",
                    )

                # Valid distinct healthy observation with variance
                if not is_duplicate:
                    self.consecutive_healthy_count += 1
                    self.last_healthy_timestamp = source_timestamp
                    self.last_healthy_fingerprint = payload_fingerprint
                    if content_fingerprint is not None:
                        self.last_healthy_content_fingerprint = content_fingerprint
                    self._persist()

                if self.consecutive_healthy_count >= self.recovery_threshold:
                    self.current_effective_state = ReliabilityState.HEALTHY
                    self._persist()
                    return (
                        ReliabilityState.HEALTHY,
                        "PRIMARY_WEATHER",
                        raw_reason,
                    )
                else:
                    # In recovery window: still hold on Fallback head
                    return (
                        self.current_effective_state,
                        "FALLBACK_LOAD_ONLY",
                        f"Recovery in progress ({self.consecutive_healthy_count}/{self.recovery_threshold} distinct healthy updates required)",
                    )

            # 2. Degraded with Intercept Capability
            elif raw_state == ReliabilityState.DEGRADED:
                if self.consecutive_healthy_count >= self.recovery_threshold:
                    self.current_effective_state = ReliabilityState.DEGRADED
                    self._persist()
                    return (
                        ReliabilityState.DEGRADED,
                        "PRIMARY_WEATHER",
                        raw_reason,
                    )
                else:
                    self.consecutive_healthy_count = 0
                    self.current_effective_state = ReliabilityState.DEGRADED
                    self._persist()
                    return (
                        ReliabilityState.DEGRADED,
                        "FALLBACK_LOAD_ONLY",
                        f"Degraded telemetry during recovery: {raw_reason}",
                    )

            # 3. Disruption / Outage / Corrupt State -> Immediate Demotion
            else:
                self.consecutive_healthy_count = 0
                self.current_effective_state = raw_state
                self.last_healthy_timestamp = None
                self.last_healthy_fingerprint = None
                self.last_healthy_content_fingerprint = None
                self._persist()
                return (
                    raw_state,
                    "FALLBACK_LOAD_ONLY",
                    raw_reason,
                )

    def reset(self, initial_state: ReliabilityState = ReliabilityState.HEALTHY):
        """Resets the recovery manager state."""
        with self._lock:
            self.consecutive_healthy_count = self.recovery_threshold if initial_state == ReliabilityState.HEALTHY else 0
            self.current_effective_state = initial_state
            self.last_healthy_timestamp = None
            self.last_healthy_fingerprint = None
            self.last_healthy_content_fingerprint = None
            self._persist()

    def get_status(self) -> dict:
        """Returns internal state for observability."""
        with self._lock:
            return {
                "consecutive_healthy_count": self.consecutive_healthy_count,
                "recovery_threshold": self.recovery_threshold,
                "current_effective_state": self.current_effective_state.value,
            }
