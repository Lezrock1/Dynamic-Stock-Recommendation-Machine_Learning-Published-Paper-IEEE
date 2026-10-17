"""Offline, original-release-only review of SMCI end shares, identity and debt."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from .fixed_lag_research import write_json
from .historical_evidence import EvidenceStore, ROOT, now
from .identity_cases import RELEASE_URL, release_evidence
from .return_comparison import sha256


def review(data: bytes) -> dict:
    release = release_evidence(data, {})
    soup = BeautifulSoup(data, "lxml")
    rows = {}
    for row in soup.find_all("table")[0].find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"], recursive=False)]
        if cells:
            rows[cells[0]] = cells
    labels = [
        "Common stock and additional paid-in capital",
        "Lines of credit and term loans, current",
        "Lines of credit and term loans, non-current",
        "Convertible notes",
    ]
    observations = []
    for label in labels:
        if label not in rows:
            raise ValueError(f"Original balance-sheet row missing: {label}")
        cells = rows[label]
        numbers = re.findall(r"\d[\d,]*", " ".join(cells[1:]))
        if len(numbers) != 2:
            raise ValueError(f"Unexpected balance-sheet numeric layout: {label}")
        for number, year in zip(numbers, [2026, 2025]):
            observations.append({
                "original_label": label, "value": int(number.replace(",", "")) * 1000,
                "unit": "USD", "period_end": f"{year}-06-30",
                "original_scale": "in thousands", "source_table_index": 0,
                "source_row_cells": cells, "status": "OBSERVED_ORIGINAL_RELEASE_VALUE",
            })
    for node in soup.find_all(["script", "style", "nav", "header", "footer"]):
        node.decompose()
    text = soup.get_text(" ", strip=True)
    share_terms = [m.group(0) for m in re.finditer(r"\b(?:outstanding|issued)\b", text, re.I)]
    identity = re.search(r"Super Micro Computer, Inc\.\s*\(NASDAQ:\s*SMCI\)", text)
    if identity is None:
        raise ValueError("Original issuer/ticker language not found")
    debt = re.search(r"total bank debt and convertible notes were \$8\.7 billion", text)
    if debt is None:
        raise ValueError("Expected rounded narrative debt disclosure missing")
    return {
        "status": "COMPLETED_ORIGINAL_RELEASE_ONLY_SUPPLEMENT",
        "publication_date": release["publication_date"], "published_at": None,
        "publication_precision": release["timestamp_precision"],
        "before_cutoff": release["before_cutoff"], "reviewed_at": now(),
        "balance_sheet_observations": observations,
        "SharesOutstanding": {
            "status": "NOT_DISCLOSED_IN_THIS_ORIGINAL_RELEASE" if not share_terms else "REQUIRES_REVIEW",
            "value": None, "visible_issued_or_outstanding_word_matches": share_terms,
            "reason": "Common stock/APIC row discloses USD balances only, no issued or outstanding end-share count. EPS denominator rows are weighted averages, not period-end shares.",
            "scope": "This release only; no claim about all publications before cutoff.",
        },
        "identity": {
            "status": "ORIGINAL_RELEASE_ISSUER_TICKER_LANGUAGE_ONLY",
            "exact_text": identity.group(0), "ticker": "SMCI", "issuer": "Super Micro Computer, Inc.",
            "common_stock_row": rows[labels[0]], "share_class_identifier": None,
            "isin": None, "wkn": None,
            "limitation": "Narrative ticker and common-stock/APIC balance are not a registered security-class/ISIN binding; no cover identity upgraded.",
        },
        "debt": {
            "status": "EXACT_COMPONENTS_DISCLOSED_NO_SEC_TAG_MAPPING",
            "narrative_text": debt.group(0),
            "components": [r for r in observations if r["original_label"] != labels[0]],
            "LongTermDebt": None,
            "limitation": "Current bank debt, noncurrent bank debt and convertible notes retained separately. No automatic mapping to LongTermDebt/LongTermDebtNoncurrent; no later SEC facts imported.",
        },
        "no_network": True, "no_production_or_database_writes": True,
    }


def run(source_run: Path, output: Path) -> dict:
    source_run, output = source_run.resolve(), output.resolve()
    source = source_run / "sources/smci/fy2026-release.html"
    journal = source_run / "acquisition.json"
    original = next(r for r in json.loads(journal.read_text())
                    if r.get("source_url") == RELEASE_URL and r.get("status") == "ACQUIRED")
    if sha256(source) != original["sha256"]:
        raise ValueError("Original release acquisition hash mismatch")
    result = review(source.read_bytes())
    store = EvidenceStore(output)
    copied = store.copy_cache(source, "smci/fy2026-release.html")
    result.update({
        "source_url": RELEASE_URL, "source_sha256": sha256(copied),
        "source_file": str(copied.relative_to(ROOT)), "observed_at": original["acquired_at"],
        "original_acquisition_journal": str(journal.relative_to(ROOT)),
        "original_acquisition_journal_sha256": sha256(journal),
    })
    write_json(output / "smci_supplemental_review.json", result)
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.source_run, args.output)
    print(json.dumps({k: result[k] for k in ["status", "SharesOutstanding", "observed_at", "source_sha256"]}, indent=2))
