"""Audit the excluded tickers against their original filings (iXBRL) and classify every feature.

Classes per feature (share classes and revenue definitions are never merged):
  PUBLISHED_UNDIMENSIONED  one standard-tag fact without dimensions (fallback candidate, full provenance)
  AMBIGUOUS_DEFINITIONS    several standard tags/values without dimensions (definitions differ; no fallback)
  CLASS_DIMENSION_ONLY     only per-share-class facts (StatementClassOfStockAxis); no consolidated value
  OTHER_DIMENSION_ONLY     only facts with other dimensions (e.g. ProductOrServiceAxis)
  EXTENSION_ONLY           only company-specific extension tags
  NOT_PUBLISHED            no matching fact in the primary document
adapter_status says whether SEC companyfacts carries the same accession for a standard tag
(SUPPORTED) or not (NOT_IN_COMPANYFACTS): a gap in the adapter is not a gap in the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from . import ixbrl, sec_edgar as se
except ImportError:  # pragma: no cover - script execution
    import ixbrl  # type: ignore
    import sec_edgar as se  # type: ignore

FEATURES = {
    "Revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"], False, "revenue"),
    "NetIncome": (["NetIncomeLoss", "ProfitLoss"], False, "netincome"),
    "OperatingIncome": (["OperatingIncomeLoss"], False, "operatingincome"),
    "EPS_Basic": (["EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"], False, "earningspershare"),
    "WeightedShares": (["WeightedAverageNumberOfSharesOutstandingBasic"], False, "weightedaverage"),
    "DepreciationAmortization": (["DepreciationDepletionAndAmortization"], False, "amortization"),
    "Assets": (["Assets"], True, "assets"),
    "StockholdersEquity": (["StockholdersEquity"], True, "stockholdersequity"),
}

# Latest filing per excluded ticker (accession and primary document are cached under cache/filings).
CASES = [
    ("APA", 1841666, "0001841666-26-000053", "apa-20260630.htm", "2026-06-30", (85, 95)),
    ("ARES", 1176948, "0001628280-26-054538", "ares-20260630.htm", "2026-06-30", (85, 95)),
    ("BKR", 1701605, "0001701605-26-000023", "bkr-20260630.htm", "2026-06-30", (85, 95)),
    ("BRK-B", 1067983, "0001193125-26-341032", "brka-20260630.htm", "2026-06-30", (85, 95)),
    ("ERIE", 922621, "0001628280-26-051079", "erie-20260630.htm", "2026-06-30", (85, 95)),
    ("FDXF", 2082247, "0001628280-26-053359", "fdxf-20260531.htm", "2026-05-31", (355, 375)),
    ("GD", 40533, "0000040533-26-000032", "gd-20260705.htm", "2026-07-05", (85, 95)),
    ("SKYD", 2041610, "0002041610-26-000054", "psky-20260630.htm", "2026-06-30", (85, 95)),
]


def _short(fact: dict) -> dict:
    keys = ("tag", "value", "unit", "period_start", "period_end", "dimensions", "context_id", "accession",
            "source_url", "document_sha256")
    return {k: fact[k] for k in keys}


def _prefix_and_name(tag: str) -> tuple[str, str]:
    prefix, _, name = tag.partition(":")
    return prefix, name


def classify(facts: list[dict], tags: list[str], instant: bool, keyword: str, period_end: str,
             days: tuple[int, int]) -> dict:
    def in_period(f: dict) -> bool:
        if f["period_end"] != period_end or f["instant"] != instant:
            return False
        if instant:
            return True
        return days[0] <= (pd.Timestamp(f["period_end"]) - pd.Timestamp(f["period_start"])).days <= days[1]

    scoped = [f for f in facts if in_period(f)]
    standard = [f for f in scoped if _prefix_and_name(f["tag"])[0] == "us-gaap"
                and _prefix_and_name(f["tag"])[1] in tags]
    plain = [f for f in standard if not f["dimensions"] and not f["typed_dimensions"]]
    values = {(f["tag"], f["value"]) for f in plain}
    extension = [f for f in scoped if _prefix_and_name(f["tag"])[0] != "us-gaap"
                 and keyword in f["tag"].lower() and not f["dimensions"]]
    if len(values) == 1:
        label, usable = "PUBLISHED_UNDIMENSIONED", plain
    elif values:
        label, usable = "AMBIGUOUS_DEFINITIONS", plain
    elif standard:
        axes = {d[0] for f in standard for d in f["dimensions"]}
        label = "CLASS_DIMENSION_ONLY" if any("ClassOfStock" in a for a in axes) else "OTHER_DIMENSION_ONLY"
        usable = standard
    elif extension:
        label, usable = "EXTENSION_ONLY", extension
    else:
        label, usable = "NOT_PUBLISHED", []
    seen, evidence = set(), []
    for f in usable:
        key = (f["tag"], f["context_id"], f["value"])
        if key not in seen:
            seen.add(key)
            evidence.append(_short(f))
    return {"classification": label, "fallback_candidate": label == "PUBLISHED_UNDIMENSIONED",
            "evidence": evidence[:8]}


def adapter_status(cik: int, accession: str, tags: list[str]) -> str:
    facts = se.get_company_facts(cik, max_age_seconds=10**9)
    gaap = ((facts or {}).get("facts") or {}).get("us-gaap") or {}
    for tag in tags:
        for entries in (gaap.get(tag, {}).get("units") or {}).values():
            if any(e.get("accn") == accession for e in entries):
                return "SUPPORTED"
    return "NOT_IN_COMPANYFACTS"


def accession_in_companyfacts(cik: int, accession: str) -> bool:
    """False means companyfacts does not yet contain this filing at all (availability lag, not an adapter gap)."""
    facts = se.get_company_facts(cik, max_age_seconds=6 * 3600)
    for taxonomy in ((facts or {}).get("facts") or {}).values():
        for concept in taxonomy.values():
            for entries in (concept.get("units") or {}).values():
                if any(e.get("accn") == accession for e in entries):
                    return True
    return False


def audit() -> dict:
    report: dict = {"method": __doc__, "cases": {}}
    for ticker, cik, accession, document, period_end, days in CASES:
        facts = ixbrl.filing_facts(cik, accession, document)
        rows = {}
        for feature, (tags, instant, keyword) in FEATURES.items():
            row = classify(facts, tags, instant, keyword, period_end, days)
            row["adapter_status"] = adapter_status(cik, accession, tags)
            rows[feature] = row
        report["cases"][ticker] = {
            "cik": cik, "accession": accession, "document": document, "period_end": period_end,
            "source_url": ixbrl.document_url(cik, accession, document),
            "document_sha256": facts[0]["document_sha256"] if facts else None,
            "companyfacts_contains_accession": accession_in_companyfacts(cik, accession),
            "features": rows}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    se.require_user_agent()
    result = audit()
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    for ticker, case in result["cases"].items():
        print(ticker, {f: (r["classification"], r["adapter_status"]) for f, r in case["features"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
