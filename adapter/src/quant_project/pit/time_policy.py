"""Timezone-safe announcement and available-at policy calculations."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
AVAILABLE_AT_POLICIES = {"conservative_next_session_v1"}


class PITTimeError(ValueError):
    pass


def _parse_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as error:
        raise PITTimeError(f"invalid date value: {value!r}") from error


def parse_announcement(value, precision: str) -> datetime:
    """Parse provider announcement time; date precision is a Shanghai-local date."""
    if precision not in {"timestamp", "date"}:
        raise PITTimeError(f"unsupported announcement_precision: {precision!r}")
    if value is None or value == "":
        raise PITTimeError("announcement_at is required")
    if precision == "date":
        local_date = _parse_date(value)
        return datetime.combine(local_date, time.min, SHANGHAI)
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise PITTimeError(f"invalid announcement timestamp: {value!r}") from error
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise PITTimeError("timestamp announcement_at must include a timezone")
    return stamp.astimezone(SHANGHAI)


def normalize_as_of(value) -> datetime:
    """Dates mean end-of-day Shanghai time; timestamps must carry an offset."""
    if value is None or value == "":
        raise PITTimeError("as_of is required")
    if isinstance(value, datetime):
        stamp = value
    else:
        raw = str(value)
        if len(raw) == 10:
            try:
                return datetime.combine(date.fromisoformat(raw), time.max, SHANGHAI)
            except ValueError as error:
                raise PITTimeError(f"invalid as_of date: {value!r}") from error
        try:
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as error:
            raise PITTimeError(f"invalid as_of timestamp: {value!r}") from error
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise PITTimeError("as_of timestamp must include a timezone")
    return stamp.astimezone(SHANGHAI)


def available_at_for_announcement(announcement_at, *, precision: str, trading_sessions,
                                  policy: str = "conservative_next_session_v1") -> datetime:
    """Return the open of the next calendar trading session after announcement day."""
    if policy not in AVAILABLE_AT_POLICIES:
        raise PITTimeError(f"unsupported availability policy: {policy!r}")
    announcement = parse_announcement(announcement_at, precision)
    sessions = sorted({_parse_date(value) for value in trading_sessions})
    next_day = next((session for session in sessions if session > announcement.date()), None)
    if next_day is None:
        raise PITTimeError(f"trading calendar has no session after {announcement.date().isoformat()}")
    return datetime.combine(next_day, time(9, 30), SHANGHAI)
