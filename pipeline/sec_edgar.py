"""Free fundamentals from SEC EDGAR XBRL "company facts" API.

No API key needed, but SEC requires a descriptive User-Agent and asks for
<= 10 requests/second. Responses are cached to disk so re-runs are cheap.
"""
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import requests

try:
    from .market_calendar import first_available_session_after_filing
except ImportError:
    from market_calendar import first_available_session_after_filing  # type: ignore[no-redef]

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "secfacts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SEC blocks generic/blank user agents; set a descriptive project name and contact
# through the environment instead of committing a personal address.
USER_AGENT = os.getenv(
    "DYNAMIC_STOCK_USER_AGENT",
    "dynamic-stock-recommendation-research/1.0 (configure DYNAMIC_STOCK_USER_AGENT)",
)
HEADERS = {"User-Agent": USER_AGENT}

# Candidate XBRL tags per financial statement line item, in preference order
# (companies vary in which tag they file under, especially in older filings).
DURATION_TAGS = {
    "Revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "RevenuesNetOfInterestExpense", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "COGS": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "NetIncome": ["NetIncomeLoss", "ProfitLoss"],
    "OperatingIncome": ["OperatingIncomeLoss"],
    "DepreciationAmortization": ["DepreciationDepletionAndAmortization",
                                  "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"],
    "CFO": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "EPS_Basic": ["EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"],
}

INSTANT_TAGS = {
    "Assets": ["Assets"],
    "AssetsCurrent": ["AssetsCurrent"],
    "LiabilitiesCurrent": ["LiabilitiesCurrent"],
    "Liabilities": ["Liabilities"],
    "StockholdersEquity": ["StockholdersEquity",
                           "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "LongTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "InventoryNet": ["InventoryNet"],
    "AccountsPayableCurrent": ["AccountsPayableCurrent", "AccountsPayableTradeCurrent"],
    "CashAndEquivalents": ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalentsAtFairValue"],
    "SharesOutstanding": ["CommonStockSharesOutstanding", "CommonStockSharesIssued"],
}

# Version of the quarter derivation rules; bump on every change so derived data stays traceable.
DERIVATION_RULE_VERSION = "q-derive-v2"
# Only flow quantities in a monetary unit are additive: a quarter may be the cumulative value minus the
# previous cumulative value of the same fiscal-year start. EPS, average share counts, ratios and balance
# sheet stocks are never differenced.
ADDITIVE_FIELDS = frozenset({"Revenue", "COGS", "NetIncome", "OperatingIncome",
                             "DepreciationAmortization", "CFO"})

# Fallback tags outside the us-gaap taxonomy, tried only if the field is still missing.
DEI_FALLBACK_TAGS = {
    "SharesOutstanding": ["EntityCommonStockSharesOutstanding"],
}


# iXBRL fallback configuration (see ixbrl_fallback.py); disabled unless registered from the reviewed config.
IXBRL_FALLBACK: dict = {}


def register_ixbrl_fallback(config: dict | None) -> None:
    IXBRL_FALLBACK.clear()
    IXBRL_FALLBACK.update(config or {})


# Documented legal successors (e.g. holding-company reorganisations): {successor_cik: (predecessor_ciks)}.
# Only entries registered explicitly from the reviewed configuration are merged.
CIK_SUCCESSORS: dict[int, tuple[int, ...]] = {}


def register_successors(mapping: dict[int, tuple[int, ...]]) -> None:
    CIK_SUCCESSORS.clear()
    CIK_SUCCESSORS.update({int(k): tuple(int(c) for c in v) for k, v in mapping.items()})


def get_company_facts(cik: int, max_age_seconds: float = 24 * 60 * 60) -> dict | None:
    """Company facts; for a registered successor the predecessor history is merged and deduplicated."""
    own = _fetch_company_facts(cik, max_age_seconds)
    predecessors = CIK_SUCCESSORS.get(int(cik), ())
    if not predecessors:
        return own
    merged = json.loads(json.dumps(own or {"facts": {"us-gaap": {}}}))
    taxonomy = merged.setdefault("facts", {}).setdefault("us-gaap", {})
    for old in predecessors:
        old_facts = _fetch_company_facts(old, max_age_seconds)
        for tag, body in (((old_facts or {}).get("facts") or {}).get("us-gaap") or {}).items():
            target = taxonomy.setdefault(tag, {k: v for k, v in body.items() if k != "units"})
            target.setdefault("units", {})
            for unit, entries in (body.get("units") or {}).items():
                existing = target["units"].setdefault(unit, [])
                seen = {json.dumps(e, sort_keys=True) for e in existing}
                existing.extend(e for e in entries if json.dumps(e, sort_keys=True) not in seen)
    return merged


def _fetch_company_facts(cik: int, max_age_seconds: float = 24 * 60 * 60) -> dict | None:
    """Fetch (and cache) the raw SEC "company facts" JSON for a CIK."""
    cache_file = CACHE_DIR / f"{cik}.json"
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < max_age_seconds:
        return json.loads(cache_file.read_text())

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException:
        return json.loads(cache_file.read_text()) if cache_file.exists() else None
    time.sleep(0.15)  # stay well under SEC's 10 req/s fair-use limit
    if resp.status_code != 200:
        return json.loads(cache_file.read_text()) if cache_file.exists() else None
    data = resp.json()
    cache_file.write_text(json.dumps(data))
    return data


SUBMISSIONS_DIR = CACHE_DIR.parent / "secsubmissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
_POINT_IN_TIME = None  # (cutoff, {accession: acceptance timestamp}) while a snapshot is built


def require_user_agent() -> None:
    """Real SEC access needs a declared contact; the placeholder default is refused."""
    if "configure DYNAMIC_STOCK_USER_AGENT" in USER_AGENT or "@" not in USER_AGENT:
        raise RuntimeError(
            "set DYNAMIC_STOCK_USER_AGENT to 'Name contact@example.org' before real SEC access"
        )


def get_submissions(cik: int, max_age_seconds: float = 3600.0) -> dict | None:
    """Submissions index; for a registered successor both filing histories, deduplicated by accession."""
    own = _fetch_submissions(cik, max_age_seconds)
    predecessors = CIK_SUCCESSORS.get(int(cik), ())
    if not predecessors:
        return own
    merged = json.loads(json.dumps(own or {"filings": {"recent": {}}}))
    recent = merged["filings"].setdefault("recent", {})
    keys = list(recent) or None
    for old in predecessors:
        old_recent = (((_fetch_submissions(old, max_age_seconds) or {}).get("filings") or {}).get("recent")) or {}
        keys = keys or list(old_recent)
        known = set(recent.get("accessionNumber", []))
        for index, accession in enumerate(old_recent.get("accessionNumber", [])):
            if accession in known:
                continue
            for key in keys:
                recent.setdefault(key, []).append((old_recent.get(key) or [None] * (index + 1))[index])
            known.add(accession)
    return merged


def _fetch_submissions(cik: int, max_age_seconds: float = 3600.0) -> dict | None:
    """Fetch (and cache) the SEC submissions index containing acceptance timestamps."""
    cache_file = SUBMISSIONS_DIR / f"{cik}.json"
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < max_age_seconds:
        return json.loads(cache_file.read_text())
    require_user_agent()
    try:
        resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                            headers=HEADERS, timeout=30)
    except requests.RequestException:
        return json.loads(cache_file.read_text()) if cache_file.exists() else None
    time.sleep(0.15)
    if resp.status_code != 200:
        return json.loads(cache_file.read_text()) if cache_file.exists() else None
    data = resp.json()
    cache_file.write_text(json.dumps(data))
    return data


def acceptance_times(submissions: dict | None) -> dict[str, pd.Timestamp]:
    """Map accession number -> SEC acceptance timestamp (UTC) for 10-Q/10-K style forms."""
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    result = {}
    for accn, stamp in zip(recent.get("accessionNumber", []), recent.get("acceptanceDateTime", [])):
        if accn and stamp:
            result[accn] = pd.Timestamp(stamp).tz_convert("UTC")
    return result


@contextmanager
def point_in_time(cutoff, acceptance: dict):
    """Only use facts accepted by ``cutoff``; same-day filings need a proven acceptance time."""
    global _POINT_IN_TIME
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("point-in-time cutoff must be timezone-aware")
    previous, _POINT_IN_TIME = _POINT_IN_TIME, (cutoff.tz_convert("UTC"), acceptance)
    try:
        yield
    finally:
        _POINT_IN_TIME = previous


def _best_by_key(entries, key_fn, as_of=None):
    """Select the latest version that was available by the requested date.

    SEC Company Facts contains amendments and comparative facts. A date-only
    ``filed`` timestamp is conservatively usable beginning at the next NYSE
    session; filtering happens before quarter/YTD derivations.
    """
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else None
    best = {}
    for e in entries:
        filed = e.get("filed")
        if not filed:
            continue
        available_at = first_available_session_after_filing(filed)
        if _POINT_IN_TIME is not None:
            pit_cutoff, acceptance = _POINT_IN_TIME
            filed_day = pd.Timestamp(filed).normalize()
            if filed_day >= pit_cutoff.tz_localize(None).normalize():
                accepted = acceptance.get(e.get("accn", ""))
                if accepted is None or accepted > pit_cutoff:
                    continue
            available_at = min(available_at, pit_cutoff.tz_localize(None).normalize())
        elif cutoff is not None and available_at > cutoff:
            continue
        k = key_fn(e)
        candidate = {**e, "_available_at": available_at}
        rank = (e["filed"], e.get("accn", ""))
        if k not in best or rank > (best[k]["filed"], best[k].get("accn", "")):
            best[k] = candidate
    return best


def _discrete_quarters(units: list, as_of=None, derive: bool = True) -> dict:
    """Turn XBRL duration facts into one discrete value per quarter end.

    Many filers report cash-flow and income items as year-to-date totals, and Q4 only exists inside
    the annual figure. The ``fy``/``fp`` labels of a fact describe the *filing*, not the period (a
    comparative fact repeats in the next year's filings under the later ``fy``), so quarters are
    derived from the period dates themselves: within one fiscal-year start, a quarter equals the
    cumulative value minus the previous cumulative value (3M, 6M, 9M, 12M).
    """
    entries = [e for e in units
               if e.get("form") in ("10-Q", "10-K") and "start" in e and "end" in e]
    for e in entries:
        e["_start"] = pd.Timestamp(e["start"])
        e["_end"] = pd.Timestamp(e["end"])
        e["_days"] = (e["_end"] - e["_start"]).days

    cumulative = _best_by_key(
        [e for e in entries if 55 <= e["_days"] <= 380],
        lambda e: (e["start"], e["end"]),
        as_of=as_of,
    )
    by_start: dict[pd.Timestamp, dict[pd.Timestamp, dict]] = {}
    for entry in cumulative.values():
        by_start.setdefault(entry["_start"], {})[entry["_end"]] = entry

    quarters: dict[pd.Timestamp, dict] = {}
    for start, ends in by_start.items():
        ordered = sorted(ends)
        for index, end in enumerate(ordered):
            entry = ends[end]
            if entry["_days"] <= 100:
                quarters.setdefault(end.normalize(), entry)
                continue
            if not derive:
                continue
            previous = next((ends[p] for p in ordered[:index] if 80 <= (end - p).days <= 100), None)
            if previous is None:
                continue
            latest = entry if entry["_available_at"] >= previous["_available_at"] else previous
            quarters.setdefault(end.normalize(), {
                **latest, "val": entry["val"] - previous["val"], "form": "derived",
                "_start": previous["_end"], "_end": entry["_end"],
            })
    # A directly reported discrete quarter (any start) wins over a derived value for the same end.
    for entry in cumulative.values():
        if 55 <= entry["_days"] <= 100:
            quarters[entry["_end"].normalize()] = entry
    return quarters


def _instant_values(units: list, as_of=None) -> dict:
    """One balance-sheet snapshot value per period-end date (latest filing wins)."""
    entries = [e for e in units if e.get("form") in ("10-Q", "10-K") and "end" in e]
    best = _best_by_key(entries, lambda e: e["end"], as_of=as_of)
    return {pd.Timestamp(k).normalize(): v for k, v in best.items()}


def _emit(series: dict, field: str, merged: dict) -> None:
    if not merged:
        return
    series[field] = {period: entry["val"] for period, entry in merged.items()}
    series[f"{field}_filed_at"] = {period: entry["filed"] for period, entry in merged.items()}
    series[f"{field}_available_at"] = {
        period: entry["_available_at"].strftime("%Y-%m-%d") for period, entry in merged.items()}
    series[f"{field}_form"] = {period: entry.get("form", "") for period, entry in merged.items()}
    series[f"{field}_accn"] = {period: entry.get("accn", "") for period, entry in merged.items()}
    prov = {period: json.dumps(entry["_prov"], sort_keys=True)
            for period, entry in merged.items() if "_prov" in entry}
    if prov:
        series[f"{field}_ixbrl_provenance"] = prov


def get_quarterly_fundamentals(cik: int, as_of=None) -> pd.DataFrame | None:
    """Return a DataFrame indexed by fiscal-quarter-end date with raw XBRL line items.

    Companies often switch the XBRL tag they file a concept under over the years
    (e.g. "Revenues" pre-2018 vs "RevenueFromContractWithCustomerExcludingAssessedTax"
    after ASC 606 adoption), so every candidate tag is merged rather than stopping
    at the first one found, keeping whichever tag actually has data for a given quarter.
    """
    """Return quarter facts, optionally filtered to an information cutoff.

    If ``as_of`` is omitted, latest currently available filings are returned;
    use an explicit decision-date cutoff to avoid restatement look-ahead.
    """
    facts = get_company_facts(cik)
    if facts is None or "us-gaap" not in facts.get("facts", {}):
        return None
    gaap = facts["facts"]["us-gaap"]

    try:
        from . import ixbrl_fallback as fb
    except ImportError:  # pragma: no cover - script execution
        import ixbrl_fallback as fb  # type: ignore
    fallback = fb.fallback_entries(
        cik, {**{f: (t, False) for f, t in DURATION_TAGS.items()}, **{f: (t, True) for f, t in INSTANT_TAGS.items()}},
        as_of=as_of)

    series = {}
    for field, tags in DURATION_TAGS.items():
        merged = {}
        for tag in tags:
            if tag not in gaap:
                continue
            unit_key = "USD/shares" if "USD/shares" in gaap[tag]["units"] else "USD"
            if unit_key in gaap[tag]["units"]:
                derive = field in ADDITIVE_FIELDS and unit_key == "USD"
                for k, v in _discrete_quarters(gaap[tag]["units"][unit_key], as_of=as_of, derive=derive).items():
                    merged.setdefault(k, v)
        if field in fallback:
            for k, v in _discrete_quarters(fallback[field], as_of=as_of, derive=False).items():
                merged.setdefault(k, v)
        _emit(series, field, merged)
    for field, tags in INSTANT_TAGS.items():
        merged = {}
        for tag in tags:
            if tag in gaap:
                unit_key = "shares" if "shares" in gaap[tag]["units"] else "USD"
                if unit_key in gaap[tag]["units"]:
                    for k, v in _instant_values(gaap[tag]["units"][unit_key], as_of=as_of).items():
                        merged.setdefault(k, v)
        if field in fallback:
            for k, v in _instant_values(fallback[field], as_of=as_of).items():
                merged.setdefault(k, v)
        if not merged and field in DEI_FALLBACK_TAGS:
            dei = facts.get("facts", {}).get("dei", {})
            for tag in DEI_FALLBACK_TAGS[field]:
                if tag in dei and "shares" in dei[tag]["units"]:
                    for k, v in _instant_values(dei[tag]["units"]["shares"], as_of=as_of).items():
                        merged.setdefault(k, v)
        _emit(series, field, merged)

    if not series:
        return None
    df = pd.DataFrame(series)
    df = df.sort_index()
    availability_columns = [name for name in df.columns if name.endswith("_available_at")]
    df["available_at"] = df[availability_columns].max(axis=1) if availability_columns else pd.NaT
    df["first_seen_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    df.index.name = "period_end"
    return df
