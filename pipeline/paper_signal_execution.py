"""Frozen causal signal calculation, independent of subsequent portfolio valuation."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .fixed_lag_research import offline_guard, write_csv, write_json
from .paper_fixed_lag import RULES, fit_sector, research_weights, rule_manifest, schedule
from .paper_history_execution import _panel, validate_panel
from .portfolio_weights import OptimizationError
from .return_comparison import sha256
from .publication_provenance import checked_bytes, verify_publication_evidence
from .training_vintage import STRICT, feature_limits

FILES = {"panel.csv", "membership.csv", "lineage.json", "definitions.json", "covariance_levels.csv"}
STRATEGIES = {"equal_weight": "equal_weight", "min_variance": "min_var", "mean_variance": "mean_var"}


def verify_signal_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("review_status") != "SOURCE_REVIEWED"
            or manifest.get("scope") != "SIGNAL" or set(manifest.get("files", {})) != FILES):
        raise ValueError("Signal bundle requires schema1, SOURCE_REVIEWED, scope SIGNAL and exact input files")
    evidence_files = manifest.get("evidence_files", {})
    if not isinstance(evidence_files, dict) or set(evidence_files) & (FILES | {"bundle_manifest.json"}):
        raise ValueError("Invalid additional signal evidence file map")
    inputs = {**manifest["files"], **evidence_files}
    if {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()} != set(inputs) | {"bundle_manifest.json"}:
        raise ValueError("Signal bundle contains unhashed files or missing evidence")
    for name, digest in inputs.items():
        checked_bytes(bundle, {"path": name, "sha256": digest})
    definitions = json.loads((bundle / "definitions.json").read_text())
    for key, expected in {
        "currency": "USD", "historical_membership": "quarter_end_verified",
        "historical_sector": "point_in_time_verified", "identity_scope": "security_share_class_time_versioned",
        "source_transition": "factor_definitions_overlap_reviewed",
    }.items():
        if definitions.get(key) != expected:
            raise ValueError(f"Unreviewed signal definition: {key}")
    pairs = {"forward_execution_split_adjusted_price_log_return": "split_adjusted_price_log",
             "forward_execution_gross_total_return_log": "gross_total_return_ex_date_log"}
    if (definitions.get("target_definition") not in pairs
            or definitions.get("covariance_definition") != pairs[definitions["target_definition"]]):
        raise ValueError("Unreviewed or inconsistent signal target/covariance definitions")
    if (not definitions.get("sources")
            or any(not isinstance(url, str) or not url.startswith("https://")
                   for url in definitions["sources"])):
        raise ValueError("Signal source definitions require reviewed HTTPS references")
    return definitions


def fit_prediction_cycle(panel: pd.DataFrame, membership: pd.DataFrame, cycle: str,
                         output: Path, *, training_vintage_policy: str = STRICT
                         ) -> tuple[pd.Series, list[dict]]:
    trade = panel[panel["cycle_key"] == cycle]
    if trade.empty:
        raise ValueError(f"No verified prediction rows for {cycle}")
    expected = membership[membership["cycle_key"] == cycle]
    if expected.empty or set(trade["gsector"]) != set(expected["gsector"]):
        raise ValueError(f"Prediction panel omits expected historical sectors: {cycle}")
    omitted = expected[~expected["security_id"].isin(trade["security_id"])]
    if not omitted.empty:
        if (not {"exclusion_reason", "exclusion_status"}.issubset(omitted)
                or omitted["exclusion_reason"].isna().any()
                or omitted["exclusion_reason"].astype(str).str.strip().eq("").any()
                or not omitted["exclusion_status"].isin(
                    ["NOT_PUBLIC_BY_CUTOFF", "MODEL_RULE_EXCLUDED", "CORPORATE_ACTION"]).all()):
            raise ValueError(f"Unresolved source gaps cannot be historical exclusions: {cycle}")
        write_csv(output / "source_exclusions.csv", omitted.to_dict("records"))
    selected, sectors = {}, []
    for sector, rows in trade.groupby("gsector"):
        if not isinstance(rows, pd.DataFrame):
            raise ValueError("Prediction sector group must be tabular")
        sector_panel = panel.loc[panel["gsector"] == sector, :]
        result = fit_sector(sector_panel, rows, cycle,
                            training_vintage_policy=training_vintage_policy)
        selected.update(result["selected"])
        directory = output / "training"
        directory.mkdir(exist_ok=True)
        for name in ("train", "validation"):
            result[name].to_csv(directory / f"sector{sector}_{name}.csv", index=False)
        sectors.append({
            **{key: value for key, value in result.items() if key not in {"train", "validation"}},
            "gsector": int(str(sector)),
            "training_periods": sorted(result["train"]["cycle_key"].unique()),
            "validation_periods": sorted(result["validation"]["cycle_key"].unique()),
            "excluded_securities": sorted(set(rows["security_id"]) - set(result["forecasts"])),
        })
    return pd.Series(selected, dtype=float), sectors


def strategy_results(selected: pd.Series, covariance: pd.DataFrame, cutoff: str) -> dict:
    """Missing covariance blocks the two optimizers, not legitimate equal weighting."""
    results = {name: {"status": "ERROR", "weights": {}, "error": None} for name in STRATEGIES}
    if (selected.empty or selected.index.has_duplicates or not np.isfinite(selected).all()
            or len(selected) < round(1 / RULES.weight_cap)):
        for result in results.values():
            result["error"] = "INVALID_SELECTION_OR_5_PERCENT_CAP_INFEASIBLE"
        return results
    equal = {str(security): 1 / len(selected) for security in selected.index}
    results["equal_weight"] = {"status": "SUCCESS", "weights": equal, "error": None}
    try:
        weights, errors = research_weights(selected, covariance, cutoff)
    except OptimizationError as exc:
        for name in ("min_variance", "mean_variance"):
            results[name]["error"] = f"COVARIANCE_OR_OPTIMIZATION_INPUT_ERROR: {exc}"
        return results
    for name, internal in STRATEGIES.items():
        if internal in weights:
            results[name] = {"status": "SUCCESS", "weights": weights[internal], "error": None}
        else:
            results[name]["error"] = errors.get(internal, "OPTIMIZER_DID_NOT_RETURN_RESULT")
    return results


def execute_signal(bundle: Path, output: Path, cycle: str) -> dict:
    """Independently replay the signal only. No cash ledger, orders or network."""
    bundle, output = bundle.resolve(), output.resolve()
    if output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("Signal output and input bundle must be disjoint")
    output.mkdir(parents=True, exist_ok=False)
    started = False
    try:
        definitions = verify_signal_bundle(bundle)
        training_policy = definitions.get("training_vintage_policy", STRICT)
        feature_limits(pd.Series([cycle]), cycle, training_policy)
        publication_evidence = None
        evidence_path = bundle / "publication_evidence.json"
        if evidence_path.is_file():
            publication_evidence = verify_publication_evidence(
                bundle, json.loads(evidence_path.read_text()), schedule(cycle)["information_cutoff"])
        originals = {str(path.relative_to(bundle)): sha256(path)
                     for path in bundle.rglob("*") if path.is_file()}
        shutil.copytree(bundle, output / "snapshot")
        verify_signal_bundle(output / "snapshot")
        if originals != {name: sha256(output / "snapshot" / name) for name in originals}:
            raise ValueError("Signal inputs changed during snapshot")
        write_json(output / "rule_config.json", rule_manifest())
        code = Path(__file__).resolve().parent
        for path in sorted(code.glob("*.py")):
            destination = output / "code/pipeline" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        shutil.copy2(code.parent / "code/ml_model.py", output / "code/ml_model.py")
        code_hashes = {str(p.relative_to(output / "code")): sha256(p)
                       for p in (output / "code").rglob("*.py")}
        write_json(output / "code_manifest.json", code_hashes)
        code_inputs = {str(p): sha256(p) for p in code.glob("*.py")}
        code_inputs[str(code.parent / "code/ml_model.py")] = sha256(code.parent / "code/ml_model.py")
        attempts = []
        comparisons = []
        pass_completed_at = {}
        computed_at = None
        with offline_guard(attempts):
            for pass_name in ("primary", "replay"):
                directory = output / pass_name
                directory.mkdir()
                snapshot = output / "snapshot"
                panel = pd.read_csv(snapshot / "panel.csv", dtype={"security_id": str})
                members = pd.read_csv(snapshot / "membership.csv", dtype={"security_id": str})
                validate_panel(panel, members, json.loads((snapshot / "lineage.json").read_text()),
                               target_cycle=cycle, training_vintage_policy=training_policy)
                started = True
                selected, sectors = fit_prediction_cycle(
                    panel, members, cycle, directory, training_vintage_policy=training_policy)
                strategies = strategy_results(selected, _panel(snapshot / "covariance_levels.csv"),
                                              schedule(cycle)["information_cutoff"])
                pass_completed_at[pass_name] = datetime.now(timezone.utc).isoformat()
                if computed_at is None:
                    computed_at = pass_completed_at[pass_name]
                result = {
                    **schedule(cycle), "sector_models": sectors, "strategy_results": strategies,
                    "computed_at_utc": computed_at,
                    "definitions": definitions, "historical_validation": "NOT_VALIDATED",
                    "training_vintage_policy": training_policy,
                    "valuation_status": "NOT_REQUESTED_SIGNAL_ONLY",
                    "execution_orders": False,
                }
                write_json(directory / "signal.json", result)
                comparisons.append({str(p.relative_to(directory)): sha256(p)
                                    for p in directory.rglob("*") if p.is_file()})
        if (comparisons[0] != comparisons[1] or attempts
                or originals != {name: sha256(bundle / name) for name in originals}
                or originals != {name: sha256(output / "snapshot" / name) for name in originals}
                or code_inputs != {name: sha256(Path(name)) for name in code_inputs}):
            raise ValueError("Signal replay mismatch, network access or changed code/input")
        success = [name for name, item in strategies.items() if item["status"] == "SUCCESS"]
        summary = {
            "status": "COMPLETED_NOT_VALIDATED" if len(success) == 3 else "PARTIAL_STRATEGY_ERRORS",
            "cycle_key": cycle, "model_started": started, "model_replay_identical": True,
            "successful_strategies": success, "strategy_results": strategies,
            "network_attempts": attempts, "historical_validation": "NOT_VALIDATED",
            "valuation_status": "NOT_REQUESTED_SIGNAL_ONLY",
            "data_manifest": originals, "code_manifest": code_hashes,
            "computed_at_utc": computed_at, "pass_completed_at_utc": pass_completed_at,
            "publication_evidence_check": publication_evidence,
            "synthetic_fixture": definitions.get("fixture_classification") == "SYNTHETIC_TEST_ONLY_NOT_HISTORICAL_EVIDENCE",
        }
        write_json(output / "execution_summary.json", summary)
        write_json(output / "artifact_manifest.json", {
            str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()})
        return summary
    except (OSError, ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as exc:
        write_json(output / "execution_summary.json", {
            "status": "FAILED" if started else "NOT_STARTED_SOURCE_PREREQUISITES_FAILED",
            "cycle_key": cycle, "model_started": started, "model_replay_identical": False,
            "historical_validation": "NOT_VALIDATED",
            "error": f"{type(exc).__name__}: {exc}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycle", required=True)
    args = parser.parse_args()
    print(json.dumps(execute_signal(args.bundle, args.output, args.cycle), indent=2))


if __name__ == "__main__":
    main()
