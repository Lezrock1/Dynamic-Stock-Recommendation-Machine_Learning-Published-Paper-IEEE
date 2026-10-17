"""Real-data numerical diagnostic; never certifies historical inputs or creates handoff packages."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .fixed_lag_research import offline_guard, write_json, write_csv
from .paper_fixed_lag import clean_windows, continuation_cycles, fit_cleaned_sector, rule_manifest, schedule
from .paper_signal_execution import strategy_results
from .return_comparison import sha256
from .run_model import PAPER_FEATURES
from .original_reference import check_rows, load_originals, source_paths


def diagnose(root: Path, output: Path, identities_path: Path | None = None,
             covariance_override: Path | None = None, *,
             original_reference_assumption: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    acquisition = root / "state/research/acquisition"
    identities_path = identities_path or acquisition / "20261016-q2-identity-review-final/security_identities.json"
    sectors_path = acquisition / "20261016-q2-sector-reconciled-final/q2_security_sector_evidence.csv"
    values = (list((acquisition / "20261016-q2-values-01/issuers").glob("*.json")) +
              list((acquisition / "20261016-q2-originals-01/issuer_values").glob("*.json")))
    inputs = [identities_path, sectors_path, *values]
    hybrid = sorted((root / "Data/live").glob("sector*_hybrid.csv"))
    inputs += hybrid
    if original_reference_assumption:
        inputs += source_paths(root)
    price_paths = sorted((acquisition / "20261016-market-sources-02/prices").glob("*.csv"))
    inputs += price_paths
    override = None
    if covariance_override is not None:
        override = pd.read_csv(covariance_override, index_col="date", parse_dates=True)
        if (not isinstance(override.index, pd.DatetimeIndex) or override.index.has_duplicates
                or not override.index.is_monotonic_increasing
                or list(override.columns) != ["PSKY"]
                or override.isna().any().any() or not np.isfinite(override.to_numpy()).all()
                or (override <= 0).any().any()):
            raise ValueError("Covariance override must contain ordered, positive date/PSKY levels only")
        from .market_calendar import nyse_sessions
        required_dates = nyse_sessions("2025-01-01", "2026-08-31")[-253:]
        if not override.index.equals(required_dates):
            raise ValueError("PSKY override must cover exactly the required 253 actual NYSE sessions")
        inputs.append(covariance_override)
    inputs += sorted((root / "pipeline").glob("*.py"))
    inputs += [root / "code/ml_model.py", root / "pipeline/requirements.txt"]
    inputs = sorted(set(inputs))
    before = {str(path): sha256(path) for path in inputs}
    write_json(output / "input_manifest_before.json", before)
    write_json(output / "rules.json", rule_manifest())
    write_json(output / "runtime.json", {
        "python": sys.version, "executable": sys.executable,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    frozen = {}
    for source in inputs:
        relative = source.relative_to(root) if source.is_relative_to(root) else Path("external") / source.name
        destination = output / "frozen_inputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != before[str(source)]:
            raise ValueError(f"Diagnostic input changed during freeze: {source}")
        frozen[str(source)] = destination
    for source in hybrid:
        destination = output / "snapshot" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != before[str(source)]:
            raise ValueError("Hybrid diagnostic source changed during snapshot")
    identities = json.loads(frozen[str(identities_path)].read_text())
    sectors = pd.read_csv(frozen[str(sectors_path)]).set_index("security_id")
    issuers = {}
    for path in values:
        item = json.loads(frozen[str(path)].read_text())
        issuers[item["cik"]] = item
    if covariance_override is not None:
        override = pd.read_csv(frozen[str(covariance_override)], index_col="date", parse_dates=True)
    reference_frames, reference_policy = ({}, None)
    if original_reference_assumption:
        reference_frames, reference_policy = load_originals(output / "frozen_inputs")
        write_json(output / "original_reference_policy.json", reference_policy)
    dates = {}
    for cycle in continuation_cycles("1990Q1", "2026Q2"):
        timing = schedule(cycle)
        for field in ("execution_date", "nominal_execution_date"):
            date = pd.Timestamp(timing[field])
            if date in dates and dates[date] != cycle:
                raise ValueError("Ambiguous nominal/execution cycle label")
            dates[date] = cycle
    eligible = {}
    for row in identities:
        if row["status"] != "ORIGINAL_COVER_TICKER_ENTITY_MATCH":
            continue
        issuer = issuers.get(row["cik"], {})
        expectation = issuer.get("period_expectation", issuer.get("expected", {}))
        if expectation.get("period_end") == row.get("period_end"):
            eligible[row["ticker"]] = row
    attempts, repeat = [], []
    with offline_guard(attempts):
        covariance = pd.concat({
            path.stem: pd.read_csv(frozen[str(path)], index_col="date", parse_dates=True)["provider_close"]
            for path in price_paths}, axis=1).sort_index()
        if override is not None:
            covariance = covariance.drop(columns=["PSKY"], errors="raise").join(override, how="outer").sort_index()
        for pass_name in ("primary", "replay"):
            directory = output / pass_name
            directory.mkdir()
            selected, reports, failures = {}, [], []
            reference_rows = []
            for path in hybrid:
                frame = pd.read_csv(output / "snapshot" / path.name)
                frame["cycle_key"] = pd.to_datetime(frame["tradedate"]).map(dates)
                frame["security_id"] = frame["tic"]
                frame = frame[frame["cycle_key"].notna()].copy()
                frame[PAPER_FEATURES] = frame[PAPER_FEATURES].replace([np.inf, -np.inf], np.nan)
                cycle_values = frame["cycle_key"].map(lambda c: pd.Period(c, "Q"))
                validation_periods = list(pd.period_range("2025Q1", "2025Q4", freq="Q"))
                training_periods = list(pd.period_range("2015Q1", "2024Q4", freq="Q"))
                train = frame[cycle_values.isin(training_periods) & frame["y_return"].notna()].copy()
                validation = frame[cycle_values.isin(validation_periods) & frame["y_return"].notna()].copy()
                sector = int(frame["gsector"].iloc[0])
                trade = frame[frame["tic"].isin(eligible)].copy()
                trade = trade[
                    trade.apply(lambda r: str(r["datadate"])[:10] == eligible[r["tic"]]["period_end"]
                                and int(sectors.loc[r["tic"], "gsector"]) == sector, axis=1)]
                if trade["security_id"].duplicated().any():
                    raise ValueError(f"Ambiguous Q2 source rows in sector {sector}")
                if train["cycle_key"].nunique() < 16 or validation["cycle_key"].nunique() != 4:
                    failures.append({"sector": sector, "error": "Insufficient numerical training/validation quarters"})
                    continue
                try:
                    features, train, validation, trade = clean_windows(train, validation, trade)
                    if reference_policy is not None:
                        source = next(item for item in reference_policy["files"]
                                      if item["path"].endswith(f"/sector{sector}_clean.xlsx"))
                        reference_rows.extend(check_rows(train, reference_frames[sector], source))
                    result = fit_cleaned_sector(features, train, validation, trade)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    failures.append({"sector": sector, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                train.to_csv(directory / f"sector{sector}_training_UNVERIFIED.csv", index=False)
                validation.to_csv(directory / f"sector{sector}_validation_UNVERIFIED.csv", index=False)
                trade.to_csv(directory / f"sector{sector}_prediction_UNVERIFIED.csv", index=False)
                selected.update(result["selected"])
                reports.append({
                    **{k: v for k, v in result.items() if k not in {"train", "validation"}},
                    "gsector": sector, "training_periods": sorted(train["cycle_key"].unique()),
                    "validation_periods": sorted(validation["cycle_key"].unique()),
                    "training_source_counts": train["data_source"].value_counts().to_dict(),
                    "validation_source_counts": validation["data_source"].value_counts().to_dict(),
                    "historical_availability_verified": False,
                    "label_known_at_verified": False,
                })
            strategies = strategy_results(pd.Series(selected, dtype=float), covariance,
                                          schedule("2026Q2")["information_cutoff"])
            result = {
                **schedule("2026Q2"), "status": "NUMERICAL_DIAGNOSTIC_NOT_HISTORICAL_SIGNAL",
                "executable": False, "handoff_eligible": False, "historical_validation": "NOT_VALIDATED",
                "forecast_input_definition": "Today's cached Q2-period features; original cover period/ticker match only. Values not asserted available at historical cutoff.",
                "training_availability": "UNVERIFIED: existing hybrid row versions, label known-at, dated training membership and source overlap unresolved",
                "original_reference_policy": reference_policy,
                "original_reference_rows_checked": len(reference_rows),
                "original_reference_is_not_a_q2_blocker": bool(reference_policy is not None and not failures),
                "legacy_nominal_date_mapping": "Legacy nominal non-trading dates mapped to cycle labels only; existing labels are not certified as actual NYSE execution-to-execution returns",
                "covariance_basis": "Uncertified provider-close split basis; not a reviewed covariance series",
                "psky_covariance_override": str(covariance_override) if covariance_override is not None else None,
                "not_a_qualified_paper_run": True, "sector_models": reports,
                "sector_errors": failures, "strategy_results": strategies,
                "forecast_count": sum(len(r["forecasts"]) for r in reports),
                "selected_count": len(selected),
            }
            if reference_policy is not None:
                result["training_availability"] = (
                    "Identified original rows are accepted reference inputs after exact source comparison. "
                    "SEC value versions, target-session basis, historical membership and source overlap remain unqualified.")
                write_csv(directory / "original_reference_rows.csv", reference_rows)
            write_json(directory / "diagnostic.json", result)
            for strategy, item in strategies.items():
                if item["status"] == "SUCCESS":
                    write_csv(directory / f"{strategy}_DIAGNOSTIC_UNVERIFIED.csv", [
                        {"security_id": security, "diagnostic_weight": weight,
                         "cycle_key": "2026Q2", "executable": False,
                         "historical_validation": "NOT_VALIDATED", "verified_isin": False}
                        for security, weight in sorted(item["weights"].items())])
                else:
                    write_json(directory / f"{strategy}_DIAGNOSTIC_FAILED.json", {
                        "strategy": strategy, "status": "ERROR", "error": item["error"],
                        "weights": None, "executable": False})
            write_csv(directory / "forecasts_UNVERIFIED.csv", [
                {"security_id": security, "gsector": report["gsector"], "prediction": value,
                 "selected": security in report["selected"], "executable": False,
                 "historical_validation": "NOT_VALIDATED"}
                for report in reports for security, value in report["forecasts"].items()])
            repeat.append({str(p.relative_to(directory)): sha256(p)
                           for p in directory.iterdir() if p.is_file()})
    if (repeat[0] != repeat[1] or attempts or before != {str(p): sha256(p) for p in inputs}
            or before != {name: sha256(path) for name, path in frozen.items()}):
        raise ValueError("Numerical diagnostic replay mismatch, network attempt or input change")
    summary = {k: v for k, v in result.items() if k != "sector_models"}
    summary.update({"sector_count": len(result["sector_models"]), "offline_replay_identical": True,
                    "network_attempts": attempts, "computed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identities", type=Path)
    parser.add_argument("--psky-covariance", type=Path,
                        help="Explicit separately reviewed same-class 253-session date/PSKY levels")
    parser.add_argument("--original-reference-assumption", action="store_true",
                        help="Explicitly accept only byte-identical original repository reference files; never certifies SEC or PIT")
    args = parser.parse_args()
    result = diagnose(args.root, args.output, args.identities, args.psky_covariance,
                      original_reference_assumption=args.original_reference_assumption)
    print(json.dumps({k: v for k, v in result.items() if k != "strategy_results"}, indent=2))


if __name__ == "__main__":
    main()
