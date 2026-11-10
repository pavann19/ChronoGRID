"""
ChronoGrid DDIR (Degraded & Delayed Input Reliability) Subsystem.
Provides deterministic input validation, fault state machines, anti-flapping recovery,
and dual-head inference routing for resilient electricity demand forecasting.
"""

from chronogrid.reliability.persistence import DDIRPersistedState, DDIRStatePersistence
from chronogrid.reliability.recovery import HysteresisRecoveryManager
from chronogrid.reliability.router import DDIRInferenceResult, DualHeadInferenceRouter
from chronogrid.reliability.state_machine import ReliabilityState
from chronogrid.reliability.validator import ValidationReport, WeatherInputValidator

__all__ = [
    "ReliabilityState",
    "WeatherInputValidator",
    "ValidationReport",
    "HysteresisRecoveryManager",
    "DualHeadInferenceRouter",
    "DDIRInferenceResult",
    "DDIRStatePersistence",
    "DDIRPersistedState",
]
