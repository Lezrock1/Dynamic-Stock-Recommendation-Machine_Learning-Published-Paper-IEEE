"""Enumerate exact retained diagnostic rows needing historical qualification, without certifying them."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .fixed_lag_research import write_csv, write_json
from .paper_fixed_lag import schedule
from .return_comparison import sha256


def enumerate_required(diagnostic: Path, output: Path) -> dict:
    manifest = json.loads((diagnostic / "artifact_manifest.json").read_text())
    for name, digest in manifest.items():
        if sha256(diagnostic / name) != digest:
            raise ValueError(f"Diagnostic input changed: {name}")
    document = json.loads((diagnostic / "primary/diagnostic.json").read_text())
    if document.get("status") != "NUMERICAL_DIAGNOSTIC_NOT_HISTORICAL_SIGNAL":
        raise ValueError("Expected explicitly unqualified numerical diagnostic")
    output.mkdir(parents=True, exist_ok=False)
    required, sectors = [], []
    role_counts, source_counts = Counter(), Counter()
    for sector in document["sector_models"]:
        gsector = sector["gsector"]
        for role in ("training", "validation", "prediction"):
            path = diagnostic / f"primary/sector{gsector}_{role}_UNVERIFIED.csv"
            frame = pd.read_csv(path, dtype={"security_id": str, "tic": str})
            if not set(sector["features"]).issubset(frame):
                raise ValueError(f"Retained features missing: {path.name}")
            for _, row in frame.iterrows():
                cycle = str(row["cycle_key"])
                start = schedule(cycle)
                end = schedule(str(pd.Period(cycle, "Q") + 1))
                needs_label = role != "prediction"
                required.append({
                    "security_id": row["security_id"], "gsector": gsector, "role": role,
                    "cycle_key": cycle, "period_end": row["datadate"], "source": row["data_source"],
                    "retained_features": ";".join(sector["features"]),
                    "feature_information_cutoff": start["information_cutoff"],
                    "model_training_information_cutoff": document["information_cutoff"],
                    "required_target_start": start["execution_at"] if needs_label else None,
                    "required_target_end": end["execution_at"] if needs_label else None,
                    "label_requires_recalculation_or_exact_basis_proof": needs_label,
                    "cached_nominal_trade_date": row["tradedate"],
                    "nominal_date_matches_actual_session": str(row["tradedate"])[:10] == start["execution_date"],
                    "label_known_at_latest_allowed": document["information_cutoff"] if needs_label else None,
                    "label_required_for_prediction": False,
                    "qualification": "VALUE_VERSION_MEMBERSHIP_AND_SOURCE_BASIS_NOT_VERIFIED",
                    "source_file": str(path), "source_sha256": manifest[str(path.relative_to(diagnostic))],
                })
                role_counts[role] += 1
                source_counts[(role, str(row["data_source"]))] += 1
            sectors.append({
                "gsector": gsector, "role": role, "rows": len(frame),
                "quarters": sorted(frame["cycle_key"].unique()),
                "features": sector["features"], "source_counts": frame["data_source"].value_counts().to_dict(),
            })
    write_csv(output / "required_rows.csv", required)
    write_json(output / "sectors.json", sectors)
    result = {
        "status": "EXACT_REQUIRED_ROWS_ENUMERATED_NOT_QUALIFIED",
        "diagnostic": str(diagnostic), "diagnostic_manifest_sha256": sha256(diagnostic / "artifact_manifest.json"),
        "rows_by_role": dict(role_counts), "source_rows_by_role": [
            {"role": role, "source": source, "rows": n} for (role, source), n in sorted(source_counts.items())],
        "feature_component_cells_required": sum(len(r["retained_features"].split(";")) for r in required),
        "nominal_trade_date_not_actual_execution_rows": sum(not r["nominal_date_matches_actual_session"] for r in required),
        "target_rule": "Targets mature after their feature cutoff; require end <= Q2 model training cutoff, not end <= own feature cutoff.",
        "historical_validation": "NOT_VALIDATED",
    }
    write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(enumerate_required(args.diagnostic, args.output), indent=2))


if __name__ == "__main__":
    main()
