"""Productive iXBRL fallback for the fundamentals pipeline (rule version ``ixbrl-fallback-v1``).

SEC companyfacts lags behind accepted filings and omits dimensioned or extension facts. For an explicitly
configured set of CIKs this module reads the primary iXBRL document of the most recent 10-Q/10-K filings
and returns facts in the same shape as companyfacts entries, so that ``sec_edgar.get_quarterly_fundamentals``
can fill *only periods that companyfacts does not provide*.

Selection rules (nothing is guessed):
  * only the reporting period of the filing itself (``reportDate``); comparative periods are ignored;
  * durations: a discrete quarter (80-100 days); year-to-date and annual values are never differenced here;
  * undimensioned facts, except the per-share fields EPS_Basic/SharesOutstanding, which use the single
    share-class member configured for the traded security (``traded_class_member``); other classes are
    never merged or converted;
  * Revenue and COGS: if several candidate tags carry different values the field stays missing (the
    definitions differ and the choice would be a guess); other fields use the adapter's tag preference order;
  * availability: the SEC acceptance time of the filing (filing date alone is never enough);
  * every entry carries tag, context, unit, period, accession, URL and document hash (``_prov``).
"""
from __future__ import annotations

import functools

import pandas as pd

try:
    from . import ixbrl, sec_edgar as se
except ImportError:  # pragma: no cover - script execution
    import ixbrl  # type: ignore
    import sec_edgar as se  # type: ignore

RULE_VERSION = "ixbrl-fallback-v1"
CLASS_FIELDS = {"EPS_Basic", "SharesOutstanding"}
STRICT_FIELDS = {"Revenue", "COGS"}
CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"
FORMS = ("10-Q", "10-K")


def _config() -> dict:
    return se.IXBRL_FALLBACK


def enabled_for(cik: int) -> bool:
    cfg = _config()
    return bool(cfg.get("enabled")) and int(cik) in {int(c) for c in cfg.get("ciks", [])}


def recent_filings(cik: int, cutoff: pd.Timestamp | None, count: int) -> list[dict]:
    recent = ((se.get_submissions(cik, max_age_seconds=6 * 3600) or {}).get("filings") or {}).get("recent") or {}
    keys = ("form", "accessionNumber", "reportDate", "filingDate", "acceptanceDateTime", "primaryDocument",
            "isInlineXBRL")
    rows = [dict(zip(keys, v)) for v in zip(*[recent.get(k, []) for k in keys])]
    rows = [r for r in rows if r["form"] in FORMS and r.get("reportDate") and r.get("isInlineXBRL")
            and str(r["primaryDocument"]).endswith(".htm")]
    for r in rows:
        r["accepted"] = pd.Timestamp(r["acceptanceDateTime"]).tz_convert("UTC")
    if cutoff is not None:
        rows = [r for r in rows if r["accepted"] <= cutoff]
    rows.sort(key=lambda r: r["accepted"], reverse=True)
    return rows[:count]


@functools.lru_cache(maxsize=64)
def _facts(cik: int, accession: str, document: str) -> tuple:
    return tuple(ixbrl.filing_facts(cik, accession, document))


def _pick(facts: list[dict], field: str, tags: list[str], *, instant: bool, period_end: str,
          class_member: str | None) -> dict | None:
    dims = ((CLASS_AXIS, class_member),) if class_member and field in CLASS_FIELDS else ()
    if instant:
        unit_part = "shares" if field == "SharesOutstanding" else "USD"
    else:
        unit_part = "USD/xbrli:shares" if field == "EPS_Basic" else "USD"
    found: dict[str, list[dict]] = {}
    for tag in tags:
        hits = [f for f in ixbrl.select(facts, [f"us-gaap:{tag}"], period_end=period_end, days=None if instant else (80, 100),
                                        instant=instant, dimensions=dims, unit_contains=unit_part)
                if instant or field == "EPS_Basic" or "/" not in f["unit"]]
        if hits:
            found[tag] = hits
    if not found:
        return None
    values = {round(h["value"], 6) for hits in found.values() for h in hits}
    if len(values) > 1 and field in STRICT_FIELDS:
        return None
    first = next(t for t in tags if t in found)
    candidates = {round(h["value"], 6) for h in found[first]}
    if len(candidates) != 1:
        return None
    return found[first][0]


def fallback_entries(cik: int, tag_maps: dict[str, tuple[list[str], bool]], as_of=None) -> dict[str, list[dict]]:
    """{field: [companyfacts-style entry]} from the latest filings; empty if not configured for the CIK."""
    if not enabled_for(cik):
        return {}
    cfg = _config()
    cutoff = None
    if se._POINT_IN_TIME is not None:
        cutoff = se._POINT_IN_TIME[0]
    elif as_of is not None:
        cutoff = pd.Timestamp(as_of).tz_localize("UTC") if pd.Timestamp(as_of).tzinfo is None else pd.Timestamp(as_of)
    class_member = (cfg.get("traded_class_member") or {}).get(str(int(cik)))
    out: dict[str, list[dict]] = {}
    for filing in recent_filings(cik, cutoff, int(cfg.get("recent_filings", 2))):
        facts = list(_facts(int(cik), filing["accessionNumber"], filing["primaryDocument"]))
        period_end = filing["reportDate"]
        for field, (tags, instant) in tag_maps.items():
            fact = _pick(facts, field, tags, instant=instant, period_end=period_end, class_member=class_member)
            if fact is None:
                continue
            entry = {"val": fact["value"], "end": period_end, "filed": filing["filingDate"],
                     "accn": filing["accessionNumber"], "form": filing["form"],
                     "_prov": {"source": "ixbrl", "rule": RULE_VERSION, "tag": fact["tag"],
                               "context_id": fact["context_id"], "unit": fact["unit"],
                               "period_start": fact["period_start"], "period_end": fact["period_end"],
                               "dimensions": [list(d) for d in fact["dimensions"]],
                               "accession": fact["accession"], "source_url": fact["source_url"],
                               "document_sha256": fact["document_sha256"],
                               "accepted_at": filing["accepted"].isoformat()}}
            if not instant:
                entry["start"] = fact["period_start"]
            out.setdefault(field, []).append(entry)
    return out
