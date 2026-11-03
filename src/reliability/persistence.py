"""
State persistence for ChronoGrid DDIR.
Persists runtime reliability state across worker crashes, respawns, and restarts.
Uses atomic file writes (tempfile + os.replace) to ensure zero corruption.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "data/ddir_runtime_state.json"


@dataclass
class DDIRPersistedState:
    """Serializable record of DDIR reliability state."""
    effective_state: str
    consecutive_healthy_count: int
    recovery_threshold: int
    last_healthy_timestamp: Optional[str]
    last_healthy_fingerprint: Optional[str]
    last_healthy_content_fingerprint: Optional[str]
    updated_at: str


class DDIRStatePersistence:
    """Atomic file-backed state persistence for DDIR."""

    def __init__(self, state_file_path: Optional[str] = None):
        self.state_file_path = Path(state_file_path or DEFAULT_STATE_FILE)

    def load_state(self) -> Optional[DDIRPersistedState]:
        """Loads and parses persisted state from disk if available."""
        if not self.state_file_path.exists():
            return None
        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DDIRPersistedState(
                effective_state=data["effective_state"],
                consecutive_healthy_count=int(data["consecutive_healthy_count"]),
                recovery_threshold=int(data.get("recovery_threshold", 2)),
                last_healthy_timestamp=data.get("last_healthy_timestamp"),
                last_healthy_fingerprint=data.get("last_healthy_fingerprint"),
                last_healthy_content_fingerprint=data.get("last_healthy_content_fingerprint"),
                updated_at=data.get("updated_at", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to read DDIR state file {self.state_file_path}: {e}")
            return None

    def save_state(self, state: DDIRPersistedState) -> bool:
        """Atomically persists state using tempfile + os.replace."""
        try:
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_dir = self.state_file_path.parent
            with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
                json.dump(asdict(state), tf, indent=2)
                temp_name = tf.name
            os.replace(temp_name, self.state_file_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to atomically persist DDIR state to {self.state_file_path}: {e}")
            return False

    def clear(self) -> bool:
        """Deletes state file if present."""
        try:
            if self.state_file_path.exists():
                self.state_file_path.unlink()
            return True
        except Exception as e:
            logger.warning(f"Failed to clear DDIR state file: {e}")
            return False

