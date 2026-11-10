"""Fail closed when the frozen UTC calendar cannot represent a future holiday."""

from datetime import date, timedelta

from dateutil.easter import easter

from chronogrid.features.temporal import GERMAN_HOLIDAYS


def nationwide_holidays(year):
    if not 1900 <= year <= 2099:
        raise ValueError("calendar guard outside validated year range")
    spring = easter(year)
    return {
        date(year, 1, 1),
        date(year, 5, 1),
        date(year, 10, 3),
        date(year, 12, 25),
        date(year, 12, 26),
        spring - timedelta(days=2),
        spring + timedelta(days=1),
        spring + timedelta(days=39),
        spring + timedelta(days=50),
    }


def assert_frozen_calendar_supported(deliveries):
    # Preserve the trained model's UTC-date convention; do not silently migrate
    # to German-local date features or invent a new calibration policy.
    for t in deliveries:
        d = t.date()
        legacy = (d.year, d.month, d.day) in GERMAN_HOLIDAYS
        if legacy != (d in nationwide_holidays(d.year)):
            raise ValueError(
                "frozen calendar lacks this holiday; a separately versioned feature/calibration experiment is required"
            )
