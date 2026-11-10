"""
Reliability State Machine for ChronoGrid DDIR.
Maps ValidationReport to explicit deterministic states: HEALTHY, DEGRADED, STALE, UNAVAILABLE, INVALID.
"""

from enum import Enum
from typing import Tuple
from chronogrid.reliability.validator import ValidationReport


class ReliabilityState(str, Enum):
    """Explicit operational states for external weather telemetry."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class ReliabilityStateMachine:
    """
    Deterministic state evaluation engine for weather telemetry.
    """

    @staticmethod
    def evaluate(report: ValidationReport) -> Tuple[ReliabilityState, str]:
        """
        Evaluates a ValidationReport into an explicit ReliabilityState and explanation reason.

        Returns:
            (ReliabilityState, reason_string)
        """
        # 1. Complete Missingness / Null Payload -> UNAVAILABLE
        if report.missing_rate >= 0.99:
            reason = "Weather telemetry is completely unavailable or null"
            if report.violations:
                reason += f": {report.violations[0]}"
            return ReliabilityState.UNAVAILABLE, reason

        # 2. Future Timestamp / Invalid Temporal Provenance -> INVALID
        future_violations = [v for v in report.violations if "future" in v.lower()]
        if future_violations:
            return ReliabilityState.INVALID, f"Invalid temporal provenance: {future_violations[0]}"

        # 3. Corrupt / Outliers / Schema Mismatch -> INVALID
        outlier_count = report.details.get("outlier_count", 0)
        type_errors = report.details.get("type_errors", 0)
        if outlier_count > 0 or type_errors > 0 or any("OUTLIER" in s for s in report.channel_states.values()):
            reasons = [v for v in report.violations if "outlier" in v.lower() or "malformed" in v.lower()]
            reason = f"Corrupted telemetry: {'; '.join(reasons)}" if reasons else "Physical boundary outliers detected"
            return ReliabilityState.INVALID, reason

        # 3. Staleness / Delayed Ingestion -> STALE
        if report.freshness_age_hours is not None:
            if report.freshness_age_hours > 36.0:
                return (
                    ReliabilityState.STALE,
                    f"Weather data is STALE (age {report.freshness_age_hours:.1f}h > 36h threshold)",
                )
            elif report.freshness_age_hours > 28.0:
                return (
                    ReliabilityState.STALE,
                    f"Weather data is DELAYED (age {report.freshness_age_hours:.1f}h > 28h gate closure SLA)",
                )

        # 4. Partial Missingness with Intercept Feasibility -> DEGRADED
        if report.can_interpolate and 0.0 < report.missing_rate <= 0.15:
            reasons = [v for v in report.violations if "missing" in v.lower() or "partial" in v.lower()]
            return (
                ReliabilityState.DEGRADED,
                f"Minor channel missingness ({report.missing_rate * 100:.1f}%): {'; '.join(reasons)}",
            )

        # 5. Missing critical channel (e.g. temperature missing) without outliers -> UNAVAILABLE
        if report.missing_rate > 0.15 or report.channel_states.get("temperature_c") != "OK":
            return (
                ReliabilityState.UNAVAILABLE,
                f"Critical weather channels missing (rate {report.missing_rate * 100:.1f}% > 15%)",
            )

        # 6. 100% Intact & Fresh -> HEALTHY
        if report.is_valid and report.quality_score >= 0.95:
            return ReliabilityState.HEALTHY, "Weather telemetry is fully intact, fresh, and within valid boundaries"

        # Fallback for minor infractions
        if report.quality_score >= 0.70:
            return ReliabilityState.DEGRADED, f"Telemetry degraded (Quality score {report.quality_score:.2f})"
        else:
            return ReliabilityState.INVALID, f"Telemetry invalid (Quality score {report.quality_score:.2f})"

