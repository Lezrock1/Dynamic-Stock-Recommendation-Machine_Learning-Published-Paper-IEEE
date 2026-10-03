"""Free fundamentals from SEC EDGAR XBRL "company facts" API.

No API key needed, but SEC requires a descriptive User-Agent and asks for
<= 10 requests/second. Responses are cached to disk so re-runs are cheap.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

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
                "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "COGS": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "NetIncome": ["NetIncomeLoss", "ProfitLoss"],
    "OperatingIncome": ["OperatingIncomeLoss"],
    "DepreciationAmortization": ["DepreciationDepletionAndAmortization",
                                  "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"],
    "CFO": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "EPS_Basic": ["EarningsPerShareBasic"],
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

# Fallback tags outside the us-gaap taxonomy, tried only if the field is still missing.
DEI_FALLBACK_TAGS = {
    "SharesOutstanding": ["EntityCommonStockSharesOutstanding"],
}


def get_company_facts(cik: int) -> dict | None:
    """Fetch (and cache) the raw SEC "company facts" JSON for a CIK."""
    cache_file = CACHE_DIR / f"{cik}.json"
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < 24 * 60 * 60:
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


def _best_by_key(entries, key_fn):
    """Dedupe restated / comparative-period facts: keep the value with the
    latest filing date, breaking ties with the later period end (XBRL filings
    sometimes tag a prior-year comparative figure with the same fy/fp label
    as the current period, both filed on the same date)."""
    best = {}
    for e in entries:
        k = key_fn(e)
        rank = (e["filed"], e.get("end", ""))
        if k not in best or rank > (best[k]["filed"], best[k].get("end", "")):
            best[k] = e
    return best


def _discrete_quarters(units: list) -> dict:
    """Turn XBRL duration facts into one value per (fiscal year, quarter).

    Many companies file cash-flow-statement items as year-to-date cumulative
    totals rather than discrete quarters, so besides picking up genuinely
    discrete (~3 month) facts we also pick up YTD facts (~6/9/12 months) and
    difference consecutive quarters within the same fiscal year to recover
    the discrete quarterly value.
    """
    entries = [e for e in units
               if e.get("form") in ("10-Q", "10-K") and "fy" in e and "fp" in e and "start" in e and "end" in e]
    for e in entries:
        e["_start"] = pd.Timestamp(e["start"])
        e["_end"] = pd.Timestamp(e["end"])
        e["_days"] = (e["_end"] - e["_start"]).days

    discrete = _best_by_key(
        [e for e in entries if e["fp"] in ("Q1", "Q2", "Q3", "Q4") and 55 <= e["_days"] <= 100],
        lambda e: (e["fy"], e["fp"]),
    )
    ytd = _best_by_key(
        [e for e in entries if e["fp"] in ("Q2", "Q3") and 150 <= e["_days"] <= 290],
        lambda e: (e["fy"], e["fp"]),
    )
    annual = _best_by_key(
        [e for e in entries if e["fp"] == "FY" and 350 <= e["_days"] <= 380],
        lambda e: e["fy"],
    )

    best = dict(discrete)
    fiscal_years = {fy for fy, _ in best} | {fy for fy, _ in ytd} | set(annual)
    for fy in fiscal_years:
        q1 = best.get((fy, "Q1"))
        # Q2 discrete = YTD(Q2) - Q1
        if (fy, "Q2") not in best and q1 and (fy, "Q2") in ytd:
            e = ytd[(fy, "Q2")]
            best[(fy, "Q2")] = {**e, "val": e["val"] - q1["val"]}
        q2 = best.get((fy, "Q2"))
        # Q3 discrete = YTD(Q3) - YTD(Q2) [i.e. - (Q1+Q2)]
        if (fy, "Q3") not in best and q1 and q2 and (fy, "Q3") in ytd:
            e = ytd[(fy, "Q3")]
            best[(fy, "Q3")] = {**e, "val": e["val"] - q1["val"] - q2["val"]}
        q3 = best.get((fy, "Q3"))
        # Q4 discrete = FY - (Q1+Q2+Q3); there is no 10-Q for Q4 so it's always derived.
        if (fy, "Q4") not in best and q1 and q2 and q3 and fy in annual:
            e = annual[fy]
            best[(fy, "Q4")] = {**e, "val": e["val"] - q1["val"] - q2["val"] - q3["val"], "form": "10-K(derived)"}

    return {v["_end"].normalize(): v["val"] for v in best.values()}


def _instant_values(units: list) -> dict:
    """One balance-sheet snapshot value per period-end date (latest filing wins)."""
    entries = [e for e in units if e.get("form") in ("10-Q", "10-K") and "end" in e]
    best = _best_by_key(entries, lambda e: e["end"])
    return {pd.Timestamp(k).normalize(): v["val"] for k, v in best.items()}


def get_quarterly_fundamentals(cik: int) -> pd.DataFrame | None:
    """Return a DataFrame indexed by fiscal-quarter-end date with raw XBRL line items.

    Companies often switch the XBRL tag they file a concept under over the years
    (e.g. "Revenues" pre-2018 vs "RevenueFromContractWithCustomerExcludingAssessedTax"
    after ASC 606 adoption), so every candidate tag is merged rather than stopping
    at the first one found, keeping whichever tag actually has data for a given quarter.
    """
    facts = get_company_facts(cik)
    if facts is None or "us-gaap" not in facts.get("facts", {}):
        return None
    gaap = facts["facts"]["us-gaap"]

    series = {}
    for field, tags in DURATION_TAGS.items():
        merged = {}
        for tag in tags:
            if tag not in gaap:
                continue
            unit_key = "USD/shares" if "USD/shares" in gaap[tag]["units"] else "USD"
            if unit_key in gaap[tag]["units"]:
                for k, v in _discrete_quarters(gaap[tag]["units"][unit_key]).items():
                    merged.setdefault(k, v)
        if merged:
            series[field] = merged
    for field, tags in INSTANT_TAGS.items():
        merged = {}
        for tag in tags:
            if tag in gaap:
                unit_key = "shares" if "shares" in gaap[tag]["units"] else "USD"
                if unit_key in gaap[tag]["units"]:
                    for k, v in _instant_values(gaap[tag]["units"][unit_key]).items():
                        merged.setdefault(k, v)
        if not merged and field in DEI_FALLBACK_TAGS:
            dei = facts.get("facts", {}).get("dei", {})
            for tag in DEI_FALLBACK_TAGS[field]:
                if tag in dei and "shares" in dei[tag]["units"]:
                    for k, v in _instant_values(dei[tag]["units"]["shares"]).items():
                        merged.setdefault(k, v)
        if merged:
            series[field] = merged

    if not series:
        return None
    df = pd.DataFrame(series)
    df = df.sort_index()
    df.index.name = "period_end"
    return df
