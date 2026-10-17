"""Shared point-in-time timeline rules for live signals and offline diagnostics."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence


def _utc(value: object, field: str, filing_key: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"filing {filing_key} is missing {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"filing {filing_key} has invalid {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"filing {filing_key} has timezone-naive {field}")
    return parsed.astimezone(timezone.utc)


def cycle_timeline(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Return shared publication, observation and signal cutoffs from persisted evidence.

    Ordinary reports become usable only after both publication and required-facts readiness.
    Evidenced corporate-action resolutions contribute their own recorded time, never a fabricated filing time.
    """
    published: list[datetime] = []
    observed: list[datetime] = []
    available: list[datetime] = []
    for row in rows:
        filing_key = row.get("filing_key", "<unknown>")
        resolution = row.get("recorded_at")
        if resolution:
            resolved_at = _utc(resolution, "resolution recorded_at", filing_key)
            observed.append(resolved_at)
            available.append(resolved_at)
            continue
        published_at = _utc(row.get("filed_at"), "filed_at", filing_key)
        first_seen_at = _utc(row.get("first_seen_at"), "first_seen_at", filing_key)
        facts_ready_at = _utc(row.get("facts_ready_at"), "facts_ready_at", filing_key)
        if facts_ready_at < published_at:
            raise ValueError(f"filing {filing_key} facts-ready time precedes publication")
        published.append(published_at)
        observed.append(first_seen_at)
        available.extend((published_at, facts_ready_at))
    if not published:
        raise ValueError("cycle has no ordinary published filing timestamp")
    if not observed or not available:
        raise ValueError("cycle has no persisted availability evidence")
    latest_published = max(published)
    latest_seen = max(observed)
    signal_as_of = max(latest_seen, max(available))
    return {
        "published_at": latest_published.isoformat(),
        "first_seen_at": latest_seen.isoformat(),
        "snapshot_as_of": signal_as_of.isoformat(),
    }
