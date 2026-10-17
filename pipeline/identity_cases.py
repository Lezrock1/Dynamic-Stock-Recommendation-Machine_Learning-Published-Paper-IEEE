"""Bounded Q2 identity/SMCI release evidence; never writes production facts or orders.

Run with --output NEW_DIRECTORY --smci-source-run EXISTING_RELEASE_ACQUISITION.
An optional --citigroup-exhibit must be the original exhibit from the same accession,
not a current identity page. Publication dates belong only to the downloaded release's
own rows; missing fields and additional HTML checks are not exclusion rules.
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from .filing_identity_evidence import cover_identity
from .fixed_lag_research import write_json
from .historical_evidence import EvidenceStore, ROOT, now
from .return_comparison import sha256

RELEASE_URL = ("https://ir.supermicro.com/news/news-details/2026/"
               "Supermicro-Announces-Fourth-Quarter-and-Full-Fiscal-Year-2026-Financial-Results/default.aspx")
IDENTITIES = ROOT / "state/research/acquisition/20261016-q2-identity-review-final/security_identities.json"


def referenced_exhibit_identity(main_data: bytes, exhibit_data: bytes,
                               main_provenance: dict, exhibit_provenance: dict) -> dict:
    """Resolve the contextless C exhibit only within its proven original document set."""
    main_url = urlsplit(main_provenance["source_url"])
    exhibit_url = urlsplit(exhibit_provenance["source_url"])
    directory = "/Archives/edgar/data/831001/000083100126000045/"
    if (main_provenance["accession"] != "0000831001-26-000045"
            or exhibit_provenance["accession"] != main_provenance["accession"]
            or main_url.scheme != "https" or exhibit_url.scheme != "https"
            or main_url.netloc != "www.sec.gov" or exhibit_url.netloc != main_url.netloc
            or main_url.path != directory + "c-20260630.htm"
            or exhibit_url.path != directory + "c-20260630_d2.htm"):
        raise ValueError("Exhibit and context donor must be the exact same original accession")
    main = BeautifulSoup(main_data, "lxml")
    exhibit = BeautifulSoup(exhibit_data, "lxml")
    if not main.find("a", href="c-20260630_d2.htm"):
        raise ValueError("Main document does not reference the supplied exhibit")
    if exhibit.find_all("xbrli:context"):
        raise ValueError("Expected original contextless exhibit; do not merge local contexts")
    references = {n.get("contextref") for n in exhibit.find_all("ix:nonnumeric")}
    donor_contexts = {n.get("id"): n for n in main.find_all("xbrli:context")}
    if not references or not references.issubset(donor_contexts):
        raise ValueError("Exhibit context reference missing in original main document")
    # Inline XBRL document sets share contexts, but no unrelated facts are imported.
    for key in sorted(references):
        exhibit.append(donor_contexts[key].extract())
    result = cover_identity(str(exhibit).encode(), exhibit_provenance, "C", 831001)
    result["context_binding"] = {
        "method": "REFERENCED_EXHIBIT_SAME_ACCESSION_INLINE_XBRL_DOCUMENT_SET",
        "context_source": main_provenance, "context_ids": sorted(references),
        "original_exhibit_has_local_contexts": False,
    }
    return result


def release_evidence(data: bytes, provenance: dict) -> dict:
    soup = BeautifulSoup(data, "lxml")
    date = soup.select_one(".evergreen-news-date-text")
    if date is None or date.get_text(strip=True) != "August 11, 2026":
        raise ValueError("Unexpected release publication date")
    tables = soup.find_all("table")
    expected = {0: "CONDENSED CONSOLIDATED BALANCE SHEETS",
                1: "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS",
                3: "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS"}
    for index, heading in expected.items():
        text = tables[index].get_text(" ", strip=True)
        if heading not in text or "2026" not in text or "2025" not in text or "in thousands" not in text:
            raise ValueError("Unexpected financial table layout")
    if "Three Months Ended June 30," not in tables[1].get_text(" ", strip=True):
        raise ValueError("Unexpected operations periods")
    specifications = {
        0: {"Cash and cash equivalents": "CashAndEquivalents", "Inventories": "InventoryNet",
            "Total current assets": "AssetsCurrent", "Total assets": "Assets",
            "Accounts payable": "AccountsPayableCurrent", "Total current liabilities": "LiabilitiesCurrent",
            "Total liabilities": "Liabilities",
            "Total Super Micro Computer, Inc. stockholders’ equity": "StockholdersEquity",
            "Lines of credit and term loans, non-current": "NoncurrentBankLoans",
            "Convertible notes": "ConvertibleNotes"},
        1: {"Net sales": "Revenue", "Cost of sales": "COGS", "Net income": "NetIncome",
            "Income from operations": "OperatingIncome", "Basic": "EPS_Basic"},
        3: {"Depreciation and amortization": "DepreciationAmortization",
            "Net cash (used in) provided by operating activities": "CFO"},
    }
    facts = []
    for index, labels in specifications.items():
        seen = set()
        for row in tables[index].find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"], recursive=False)]
            if not cells or cells[0] not in labels or cells[0] in seen:
                continue
            label = cells[0]
            seen.add(label)
            # Dollar signs and closing negative parentheses occupy separate HTML cells.
            numeric = re.findall(r"\(?\s*\d[\d,]*(?:\.\d+)?\s*\)?|—", " ".join(cells[1:]))
            count = 4 if index == 1 else 2
            if len(numeric) != count:
                raise ValueError(f"Unexpected number of values for {label}")
            for column, raw in enumerate(numeric):
                year = 2026 if column % 2 == 0 else 2025
                quarterly = index == 1 and column < 2
                unit = "USD/share" if labels[label] == "EPS_Basic" else "USD"
                value = Decimal(raw.strip().replace(",", "").replace("(", "-").replace(")", "")
                                .replace(" ", "").replace("—", "0"))
                if unit == "USD":
                    value *= 1000
                facts.append({
                    "field": labels[label], "original_label": label, "value": float(value), "unit": unit,
                    "period_start": f"{year}-04-01" if quarterly else
                                    (f"{year-1}-07-01" if index != 0 else None),
                    "period_end": f"{year}-06-30",
                    "period_kind": "instant" if index == 0 else ("quarter" if quarterly else "annual"),
                    "source_table_index": index, "source_row_cells": cells, "source_column": column,
                    "status": "OBSERVED_UNAUDITED_ORIGINAL_RELEASE_VALUE",
                    **provenance,
                })
        if seen != set(labels):
            raise ValueError(f"Missing requested release rows: {set(labels) - seen}")
    return {
        "ticker": "SMCI", "cik": 1375365, "fiscal_period": "FY2026_Q4_AND_ANNUAL",
        "period_end": "2026-06-30", "publication_date": "2026-08-11",
        "published_at": None, "timestamp_precision": "DATE_ONLY_NO_TIME_OR_ZONE_IN_ISSUER_PAGE",
        "conservative_publication_upper_bound_utc": "2026-08-13T00:00:00Z",
        "upper_bound_note": "Conservative entire local publication day plus timezone margin; not a release timestamp",
        "cutoff_utc": "2026-08-31T20:00:00Z", "before_cutoff": True,
        "status": "EARLIER_ORIGINAL_RELEASE_FOUND_EXACT_LISTED_VALUES_ONLY",
        "facts": facts, "fact_count": len(facts), **provenance, "reviewed_at": now(),
        "missing_required_fields": ["SharesOutstanding", "LongTermDebt"],
        "missing_note": "Weighted-average shares are NOT period-end shares. Bank loans and convertible notes retained separately, not silently mapped to a SEC debt tag.",
        "certification": "No SEC fields backdated; no production certification; historical TTM chain and identity require separate checks",
    }


def run(output: Path, smci_source_run: Path, citigroup_exhibit: Path | None = None) -> dict:
    output = output.resolve()
    smci_source_run = smci_source_run.resolve()
    store = EvidenceStore(output)
    rows = json.loads(IDENTITIES.read_text())
    cases = []
    inputs = {str(IDENTITIES.relative_to(ROOT)): sha256(IDENTITIES)}
    for old in rows:
        if old["ticker"] not in {"C", "MMM", "NDAQ", "RVTY"}:
            continue
        source = ROOT / old["source_file"]
        if sha256(source) != old["sha256"]:
            raise ValueError("Original identity hash mismatch")
        copied = store.copy_cache(source, f"identities/{old['ticker']}.html")
        provenance = {k: old[k] for k in ("accession", "source_url", "sha256", "published_at", "period_end")}
        result = cover_identity(copied.read_bytes(), {
            **provenance, "source_file": str(copied.relative_to(ROOT)),
            "original_source_file": old["source_file"]}, old["ticker"], old["cik"])
        if old["ticker"] == "C" and citigroup_exhibit is not None:
            expected_url = old["source_url"].rsplit("/", 1)[0] + "/c-20260630_d2.htm"
            exhibit_hash = sha256(citigroup_exhibit)
            acquisition_proof = None
            for directory in citigroup_exhibit.resolve().parents:
                if not directory.is_relative_to(ROOT):
                    break
                manifest = directory / "acquisition.json"
                if manifest.is_file():
                    acquisition_proof = next((r for r in json.loads(manifest.read_text())
                                              if r.get("source_url") == expected_url
                                              and r.get("sha256") == exhibit_hash
                                              and r.get("status") == "ACQUIRED"), None)
                    break
            if acquisition_proof is None:
                raise ValueError("Exhibit requires exact URL/hash acquisition proof from the original accession")
            inputs[str(manifest.relative_to(ROOT))] = sha256(manifest)
            exhibit = store.copy_cache(citigroup_exhibit, "identities/C-exhibit99.01.html")
            exhibit_result = referenced_exhibit_identity(copied.read_bytes(), exhibit.read_bytes(),
                {**provenance, "source_file": str(copied.relative_to(ROOT))}, {
                **provenance, "sha256": sha256(exhibit),
                "source_url": expected_url, "observed_at": acquisition_proof["acquired_at"],
                "source_file": str(exhibit.relative_to(ROOT))})
            result = {**exhibit_result, "main_document_evidence": result}
        result.pop("numeric_context_entities", None)
        result["decision_scope"] = "IDENTITY_ONLY_NOT_AUTOMATIC_PAPER_EXCLUSION_OR_FACT_CERTIFICATION"
        cases.append(result)
        old.update(result)
        old.pop("numeric_context_entities", None)
    release_source = smci_source_run / "sources/smci/fy2026-release.html"
    acquisition = json.loads((smci_source_run / "acquisition.json").read_text())
    observed = next(r for r in acquisition if r["source_url"] == RELEASE_URL and r["status"] == "ACQUIRED")
    if sha256(release_source) != observed["sha256"]:
        raise ValueError("Release hash mismatch")
    release = store.copy_cache(release_source, "smci/fy2026-release.html")
    smci = release_evidence(release.read_bytes(), {
        "source_url": RELEASE_URL, "sha256": sha256(release), "observed_at": observed["acquired_at"],
        "source_file": str(release.relative_to(ROOT)),
        "original_acquisition_manifest": str((smci_source_run / "acquisition.json").relative_to(ROOT)),
    })
    smci_identity = next(r for r in rows if r["ticker"] == "SMCI")
    smci_identity["original_filing_status"] = smci_identity["status"]
    smci_identity["status"] = "COVER_IDENTITY_UNRESOLVED_EARLIER_RELEASE_AVAILABLE"
    smci_identity["earlier_release_status"] = smci["status"]
    smci_identity["earlier_release_source_sha256"] = smci["sha256"]
    smci_identity["note"] = "Identity remains unresolved; earlier original report was public before cutoff. Not a NOT_PUBLIC exclusion."
    inputs[str((smci_source_run / "acquisition.json").relative_to(ROOT))] = sha256(smci_source_run / "acquisition.json")
    confirmed = [r for r in rows if r["status"] == "ORIGINAL_COVER_TICKER_ENTITY_MATCH"]
    summary = {
        "securities": len(rows), "distinct_issuers": len({r["cik"] for r in rows}),
        "cover_matches": len(confirmed), "cover_match_issuers": len({r["cik"] for r in confirmed}),
        "cases": [{"ticker": r["ticker"], "status": r["status"]} for r in cases],
        "unresolved": [{"ticker": r["ticker"], "status": r["status"]} for r in rows if r not in confirmed],
        "smci_status": smci["status"], "smci_exact_release_facts": smci["fact_count"],
        "smci_before_cutoff": smci["before_cutoff"], "observed_at": now(),
        "unchanged_other_identities": "Inherited original evidence, not recertified or historically backfilled",
        "scope": "No DB/orders/production writes. Missing additional HTML control is not a fact rejection rule.",
    }
    write_json(output / "identity_cases.json", cases)
    write_json(output / "security_identities.json", rows)
    write_json(output / "smci_release_evidence.json", smci)
    write_json(output / "summary.json", summary)
    write_json(output / "input_manifest.json", {
        **inputs, str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
        "pipeline/filing_identity_evidence.py": sha256(ROOT / "pipeline/filing_identity_evidence.py"),
        "tests/test_identity_cases.py": sha256(ROOT / "tests/test_identity_cases.py")})
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smci-source-run", required=True, type=Path)
    parser.add_argument("--citigroup-exhibit", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.smci_source_run, args.citigroup_exhibit), indent=2))
