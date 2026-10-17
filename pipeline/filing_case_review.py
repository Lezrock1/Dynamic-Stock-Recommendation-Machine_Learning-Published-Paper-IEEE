"""Separate historical report exclusions from present-day acquisition and evidence gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .fixed_lag_research import offline_guard, write_csv, write_json
from .historical_evidence import acceptance_map, filings
from .paper_fixed_lag import schedule
from .return_comparison import sha256


def disposition(identity: dict, records: list[dict], expectation: dict, cutoff: str) -> dict:
    period, form = expectation.get("period_end"), expectation.get("filing_form")
    base = {"ticker": identity["ticker"], "cik": identity["cik"],
            "expected_period": period, "expected_form": form, "information_cutoff": cutoff,
            "historical_exclusion_status": None}
    if not period or form not in {"10-Q", "10-K"}:
        return {**base, "category": "TARGETED_REVIEW_REQUIRED",
                "reason": "Expected fiscal report period/form unresolved"}
    matching = [r for r in records if r.get("reportDate") == period and r.get("form") == form]
    accepted = acceptance_map(matching)
    dated = [r for r in matching if r["accessionNumber"] in accepted]
    before = [r for r in dated if accepted[r["accessionNumber"]] <= pd.Timestamp(cutoff)]
    if not before:
        if dated and len(dated) == len(matching):
            first = min(dated, key=lambda r: accepted[r["accessionNumber"]])
            return {**base, "category": "LATE_EXPECTED_FILING_RULE_REVIEW",
                    "earliest_known_accession": first["accessionNumber"],
                    "published_at": accepted[first["accessionNumber"]].isoformat(),
                    "reason": "Expected SEC filing accepted after cutoff. Apply NOT_PUBLIC_BY_CUTOFF only after reviewing the paper rule and possible earlier original report releases; no all-READY gate."}
        return {**base, "category": "TARGETED_REVIEW_REQUIRED",
                "reason": "No sufficiently evidenced acceptance timeline; absence from local submissions is not proof of nonpublication"}
    identity_accession = identity.get("accession")
    known_before = {r["accessionNumber"] for r in before}
    base["pre_cutoff_accessions"] = sorted(known_before)
    if identity_accession in known_before and identity.get("source_file"):
        base["accession"] = identity_accession
        base["published_at"] = accepted[identity_accession].isoformat()
        if identity.get("status") == "ORIGINAL_COVER_TICKER_ENTITY_MATCH":
            return {**base, "category": "FILING_IDENTITY_EVIDENCED",
                    "reason": "Pre-cutoff original identity evidenced; not a complete feature dataset or READY claim"}
        return {**base, "category": "TARGETED_REVIEW_REQUIRED",
                "reason": "Pre-cutoff report acquired, historical security identity/class unresolved; not a historical exclusion"}
    return {**base, "category": "ACQUISITION_REQUIRED",
            "reason": "Expected report published before cutoff, but matching original evidence not acquired; do not exclude as unpublished"}


def review(identities: Path, sources: list[Path], values: list[Path], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    inputs = [identities, Path(__file__)]
    inputs += [p for directory in values + sources for p in sorted(directory.glob("*.json"))]
    input_hashes = {str(p): sha256(p) for p in inputs}
    issuers = {}
    for directory in values:
        for path in sorted(directory.glob("*.json")):
            item = json.loads(path.read_text())
            inputs.append(path)
            if item["cik"] in issuers:
                raise ValueError(f"Duplicate issuer expectation: {item['cik']}")
            issuers[item["cik"]] = item
    rows, attempts = [], []
    with offline_guard(attempts):
        for identity in json.loads(identities.read_text()):
            cik = identity["cik"]
            candidates = [directory / f"{cik}.json" for directory in sources
                          if (directory / f"{cik}.json").is_file()]
            if len({sha256(p) for p in candidates}) > 1:
                raise ValueError(f"Conflicting submissions snapshots for {cik}")
            records = []
            if candidates:
                path = candidates[0]
                inputs.append(path)
                document = json.loads(path.read_text())
                if int(document["cik"]) != cik:
                    raise ValueError(f"Submissions issuer mismatch for {cik}")
                records = filings(document)
            issuer = issuers.get(cik, {})
            expectation = issuer.get("period_expectation", issuer.get("expected", {}))
            rows.append(disposition(identity, records, expectation,
                                    schedule("2026Q2")["information_cutoff"]))
        counts = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        summary = {"cycle_key": "2026Q2", "securities": len(rows), "categories": counts,
                   "all_issuers_ready_required": False, "automatic_exclusions_created": 0,
                   "note": "Acquisition/review gaps are not historical exclusions. Verified paper-rule exclusions may reduce the prediction universe; existing sector, training and optimizer rules still apply.",
                   "model_started": False, "historical_validation": "NOT_VALIDATED",
                   "network_attempts": attempts}
        write_json(output / "cases.json", rows)
        write_csv(output / "cases.csv", rows)
        write_json(output / "summary.json", summary)
        if input_hashes != {path: sha256(Path(path)) for path in input_hashes}:
            raise ValueError("Filing review input changed during classification")
        write_json(output / "input_manifest.json", input_hashes)
        write_json(output / "artifact_manifest.json", {
            str(p.relative_to(output)): sha256(p) for p in output.iterdir() if p.is_file()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, nargs="+", required=True)
    parser.add_argument("--values", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.identities, args.submissions, args.values, args.output), indent=2))


if __name__ == "__main__":
    main()
