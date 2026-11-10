"""UTC delivery hours for a Berlin calendar day, including DST changes."""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def delivery_grid(day: date):
    zone = ZoneInfo("Europe/Berlin")
    start = datetime.combine(day, datetime.min.time(), zone).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), zone).astimezone(
        timezone.utc
    )
    return tuple(
        start + timedelta(hours=i) for i in range(int((end - start).total_seconds() / 3600))
    )
