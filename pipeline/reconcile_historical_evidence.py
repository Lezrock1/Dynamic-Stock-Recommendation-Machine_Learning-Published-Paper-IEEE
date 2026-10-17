"""Offline cross-check of SEC accession values against acquired original iXBRL facts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pandas as pd

from .fixed_lag_research import offline_guard, write_csv, write_json
from .historical_evidence import ROOT
from .return_comparison import sha256


def normalized_unit(value: str) -> str:
    return "/".join(part.split(":")[-1] for part in value.split("/"))


def match_original(atom: dict, originals: list[dict], context_entities: dict | None = None,
                   cik: int | None = None) -> dict:
    same_accession = [f for f in originals if f["accession"] == atom["accession"]]
    if not same_accession:
        return {"status": "EXACT_COMPONENT_ORIGINAL_NOT_ACQUIRED", "matches": []}
    candidates = [f for f in same_accession if
                  f["tag"] == f"{atom['namespace']}:{atom['tag']}"
                  and normalized_unit(f["unit"]) == atom["unit"]
                  and f["period_start"] == atom["period_start"]
                  and f["period_end"] == atom["period_end"]
                  and not f["dimensions"] and not f["typed_dimensions"]]
    matching = [f for f in candidates if math.isclose(f["value"], atom["value"], rel_tol=1e-10, abs_tol=1e-8)]
    if context_entities is not None:
        matching = [f for f in matching if str(context_entities.get(f["context_id"], "")).isdigit()
                    and int(context_entities[f["context_id"]]) == cik]
        if matching:
            return {"status": "UNDIMENSIONED_ORIGINAL_VALUE_AND_ENTITY_MATCH", "matches": matching}
    if matching:
        return {"status": "UNDIMENSIONED_ORIGINAL_VALUE_MATCH_ENTITY_SCOPE_REVIEW_REQUIRED",
                "matches": matching}
    return {"status": "ORIGINAL_VALUE_OR_CONTEXT_UNRESOLVED_NOT_SUBSTITUTED", "matches": []}


def reconcile(values: Path, originals: Path, identities: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    records, atoms_seen, rejected_ttm = [], set(), []
    files = list((values / "issuers").glob("*.json")) + list((originals / "issuer_values").glob("*.json"))
    inputs = files + list((originals / "numeric_facts").glob("*.json")) + [
        identities / "security_identities.json", Path(__file__)]
    input_hashes = {str(path): sha256(path) for path in inputs}
    rows = json.loads((identities / "security_identities.json").read_text())
    entities = {r["cik"]: r["numeric_context_entities"] for r in rows if r.get("numeric_context_entities")}
    attempts = []
    with offline_guard(attempts):
        for path in sorted(files):
            issuer = json.loads(path.read_text())
            cik = issuer["cik"]
            original_path = originals / "numeric_facts" / f"{cik}.json"
            numeric = json.loads(original_path.read_text()) if original_path.exists() else []
            evidence = issuer.get("period_evidence") or {}
            components = []
            for field, value in evidence.get("period_values", {}).items():
                if value:
                    components.extend((field, atom) for atom in value["components"])
            for field, block in evidence.get("ttm", {}).items():
                if block["status"] != "COMPONENT_VERSIONS_EVIDENCED" or block["value"] is None:
                    rejected_ttm.append({"cik": cik, "field": field, "status": block["status"],
                                         "candidate_periods": block["periods"],
                                         "reason": "No complete consecutive four-quarter chain at expected Q2 period; candidates not used as feature components"})
                    continue
                for quarter in block["components"]:
                    components.extend((f"TTM:{field}", atom) for atom in quarter["components"])
            prior = evidence.get("prior_year_revenue")
            if prior:
                components.extend(("PRIOR_YEAR:Revenue", atom) for atom in prior["components"])
            for dependency, atom in components:
                key = (cik, dependency, atom["value_version"])
                if key in atoms_seen:
                    continue
                atoms_seen.add(key)
                result = match_original(atom, numeric, entities.get(cik), cik)
                records.append({
                    "cik": cik, "feature_dependency": dependency, "tag": atom["tag"], "unit": atom["unit"],
                    "value": atom["value"], "period_start": atom["period_start"], "period_end": atom["period_end"],
                    "accession": atom["accession"], "published_at": atom["published_at"],
                    "value_version": atom["value_version"], "status": result["status"],
                    "original_context_ids": [f["context_id"] for f in result["matches"]],
                    "raw_companyfacts_sha256": atom["raw_companyfacts_sha256"],
                    "matched_original_sha256": result["matches"][0]["document_sha256"] if result["matches"] else None,
                })
        status = {}
        for record in records:
            status[record["status"]] = status.get(record["status"], 0) + 1
        summary = {
            "status": "COMPLETED_COMPONENT_CROSSCHECK_NOT_FEATURE_CERTIFICATION",
            "issuers_with_value_evidence": len({r["cik"] for r in records}),
            "unique_field_component_versions": len(records), "component_results": status,
            "rejected_incomplete_ttm_chains": len(rejected_ttm),
            "original_cover_security_matches": sum(r["status"] == "ORIGINAL_COVER_TICKER_ENTITY_MATCH" for r in rows),
            "all_component_publications_before_cutoff": bool(records) and all(
                pd.Timestamp(r["published_at"]) <= pd.Timestamp("2026-08-31T20:00:00Z") for r in records),
            "remaining_reviews": ["share-class attribution and entity scope of unacquired component originals",
                                  "all unacquired component originals and TTM comparability",
                                  "price/split basis at each cutoff and source-factor overlap",
                                  "historical training/validation windows at their own cutoffs"],
            "historical_validation": "NOT_VALIDATED", "network_attempts": attempts,
            "model_started": False,
        }
        write_csv(output / "component_original_crosscheck.csv", records)
        write_csv(output / "rejected_ttm_chains.csv", rejected_ttm)
        write_json(output / "summary.json", summary)
        if input_hashes != {str(path): sha256(path) for path in inputs}:
            raise ValueError("A component-check input changed during calculation")
        write_json(output / "input_manifest.json", input_hashes)
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.values, args.originals, args.identities, args.output), indent=2))


if __name__ == "__main__":
    main()
