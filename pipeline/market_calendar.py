"""NYSE session helpers backed by an exchange calendar, including early closes."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

try:
    import pandas_market_calendars as mcal
except ImportError as exc:  # pragma: no cover - exercised only with missing dependency
    raise ImportError(
        "NYSE session handling requires pandas-market-calendars; install "
        "pipeline/requirements.txt"
    ) from exc


@lru_cache(maxsize=1)
def _nyse():
    return mcal.get_calendar("NYSE")


def _session_index(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    schedule = _nyse().schedule(start_date=pd.Timestamp(start).date(), end_date=pd.Timestamp(end).date())
    sessions = pd.DatetimeIndex(schedule.index)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def nyse_schedule(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Return open/close timestamps; shortened sessions retain their early close."""
    return _nyse().schedule(start_date=pd.Timestamp(start).date(), end_date=pd.Timestamp(end).date())


def session_close_time(value: str | pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(value).normalize()
    schedule = nyse_schedule(day, day)
    if schedule.empty:
        raise ValueError(f"{day.date()} is not an NYSE session")
    return pd.Timestamp(schedule.iloc[0]["market_close"]).tz_convert("America/New_York")


def nyse_sessions(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return _session_index(start, end)


def is_nyse_session(value: str | pd.Timestamp) -> bool:
    day = pd.Timestamp(value).normalize()
    return day in _session_index(day, day)


def next_nyse_session(value: str | pd.Timestamp, *, include_date: bool = True) -> pd.Timestamp:
    day = pd.Timestamp(value).normalize()
    start = day if include_date else day + pd.Timedelta(days=1)
    sessions = _session_index(start, start + pd.Timedelta(days=14))
    if sessions.empty:
        raise ValueError(f"could not find an NYSE session at or after {start.date()}")
    return sessions[0]


def previous_nyse_session(value: str | pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(value).normalize()
    sessions = _session_index(day - pd.Timedelta(days=14), day - pd.Timedelta(days=1))
    if sessions.empty:
        raise ValueError(f"could not find an NYSE session before {day.date()}")
    return sessions[-1]


def first_session_on_or_after(value: str | pd.Timestamp) -> pd.Timestamp:
    return next_nyse_session(value, include_date=True)


def first_available_session_after_filing(filed_date: str | pd.Timestamp) -> pd.Timestamp:
    """Conservative rule when SEC supplies a filing date but no acceptance time."""
    return next_nyse_session(filed_date, include_date=False)
