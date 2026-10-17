"""Execute the fixed-lag research strategy only on an explicitly reviewed historical bundle."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, CorporateAction, simulate_portfolio
from .fixed_lag_research import offline_guard, write_json, write_csv
from .paper_fixed_lag import (
    RULES, continuation_cycles, fit_sector, research_weights, rule_manifest, schedule,
)
from .return_comparison import import_series, sha256
from .market_calendar import nyse_sessions, session_close_time
from .paper_live import PERIOD_RULE_V2, period_window
from .portfolio_weights import OptimizationError
from .run_model import PAPER_FEATURES
from .training_vintage import STRICT, feature_limits

REQUIRED_FILES = {"panel.csv", "membership.csv", "raw_prices.csv", "covariance_levels.csv",
                  "corporate_actions.csv", "lineage.json", "definitions.json"}


def verify_bundle(directory: Path) -> dict:
    doc = json.loads((directory / "bundle_manifest.json").read_text())
    if doc.get("schema_version") != 1 or doc.get("review_status") != "SOURCE_REVIEWED":
        raise ValueError("Historical bundle must carry schema_version=1 and SOURCE_REVIEWED")
    if set(doc.get("files", {})) != REQUIRED_FILES:
        raise ValueError(f"Bundle must explicitly hash exactly {sorted(REQUIRED_FILES)}")
    for name, digest in doc["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Historical bundle hash mismatch: {name}")
    definitions = json.loads((directory / "definitions.json").read_text())
    requirements = {
        "currency": "USD",
        "raw_price_basis": "unadjusted_as_traded",
        "action_coverage": "splits_dividends_delistings_verified",
        "historical_membership": "quarter_end_verified",
        "historical_sector": "point_in_time_verified",
        "identity_scope": "security_share_class_time_versioned",
        "source_transition": "factor_definitions_overlap_reviewed",
        "simulation_dividends": "receivable_ex_date_cash_pay_date_until_rebalance",
    }
    for key, expected in requirements.items():
        if definitions.get(key) != expected:
            raise ValueError(f"Missing or incompatible reviewed definition: {key}={expected}")
    pairs = {
        "forward_execution_split_adjusted_price_log_return": "split_adjusted_price_log",
        "forward_execution_gross_total_return_log": "gross_total_return_ex_date_log",
    }
    if (definitions.get("target_definition") not in pairs or
            definitions.get("covariance_definition") != pairs[definitions["target_definition"]]):
        raise ValueError("Target and covariance must have the same reviewed price-return or gross-TR definition")
    if not definitions.get("sources") or any(not str(s).startswith("https://") for s in definitions["sources"]):
        raise ValueError("Bundle needs explicit reviewed source URLs")
    return {"manifest": doc, "definitions": definitions}


def validate_panel(panel: pd.DataFrame, membership: pd.DataFrame, lineage: dict, *,
                   target_cycle: str | None = None,
                   training_vintage_policy: str = STRICT) -> None:
    required = {"security_id", "cycle_key", "gsector", "feature_available_at",
                "reconstruction_status", "y_return", "label_known_at", "label_source_ref",
                "reconstructed_at", "period_end", "expected_period", "period_rule", "filing_form",
                "facts_ready_at", "acquired_at"} | set(PAPER_FEATURES)
    if not required.issubset(panel):
        raise ValueError(f"Panel missing {sorted(required - set(panel.columns))}")
    if panel.duplicated(["security_id", "cycle_key"]).any():
        raise ValueError("Duplicate security/decision quarter")
    if membership.duplicated(["security_id", "cycle_key"]).any():
        raise ValueError("Duplicate historical membership row")
    if not {"security_id", "cycle_key", "gsector"}.issubset(membership):
        raise ValueError("Historical membership needs security_id, cycle_key and gsector")
    limits = feature_limits(panel["cycle_key"], target_cycle, training_vintage_policy)
    available = pd.to_datetime(panel["feature_available_at"], utc=True, errors="raise")
    if available.isna().any() or (available > limits).any():
        raise ValueError("Future information in historical feature row")
    if not panel["reconstruction_status"].eq("VERIFIED").all():
        raise ValueError("Unverified panel rows are not tradable inputs")
    now = pd.Timestamp.now(tz="UTC")
    for field in ("reconstructed_at", "acquired_at"):
        stamps = pd.to_datetime(panel[field], utc=True, errors="raise")
        if stamps.isna().any() or (stamps > now).any():
            raise ValueError(f"Invalid actual {field} timestamp")
    for field in ("feature_available_at", "label_known_at", "reconstructed_at", "acquired_at"):
        if panel[field].dropna().map(lambda value: pd.Timestamp(value).tzinfo is None).any():
            raise ValueError(f"Explicit timezone required for {field}")
    for feature in PAPER_FEATURES + ["y_return"]:
        values = pd.to_numeric(panel[feature], errors="raise")
        if np.isinf(values).any():
            raise ValueError(f"Nonfinite feature/target value: {feature}")
    if panel["gsector"].isna().any() or panel["security_id"].isna().any():
        raise ValueError("Historical identity and sector cannot be missing")
    known = pd.to_datetime(panel["label_known_at"], utc=True, errors="coerce")
    member_sectors = {
        (security, cycle): sector
        for security, cycle, sector in membership[
            ["security_id", "cycle_key", "gsector"]].itertuples(index=False, name=None)
        if pd.notna(security) and pd.notna(cycle)
    }
    period_windows = {}
    label_ends = {}
    for i, row in panel.iterrows():
        key = f"{row['cycle_key']}:{row['security_id']}"
        if row["period_rule"] != PERIOD_RULE_V2 or row["filing_form"] not in {"10-Q", "10-K"}:
            raise ValueError(f"Unreviewed period rule/report form: {key}")
        cycle = row["cycle_key"]
        if cycle not in period_windows:
            period_windows[cycle] = period_window(pd.Period(cycle, "Q").end_time.normalize())
        low, high = period_windows[cycle]
        if row["period_end"] != row["expected_period"] or not low < pd.Timestamp(row["period_end"]) <= high:
            raise ValueError(f"Wrong fiscal report period for decision cycle: {key}")
        evidence = lineage.get(key)
        if not evidence:
            raise ValueError(f"Missing feature lineage: {key}")
        row_available = available.loc[i]
        for feature in PAPER_FEATURES:
            if pd.isna(row[feature]):
                continue
            parts = evidence.get(feature)
            if not isinstance(parts, list) or not parts:
                raise ValueError(f"No source/value-version lineage for {key}:{feature}")
            for part in parts:
                if not all(part.get(k) for k in ("accession", "value_version", "source_url", "published_at")):
                    raise ValueError(f"Incomplete provenance for {key}:{feature}")
                stamp = pd.Timestamp(part["published_at"])
                if stamp.tzinfo is None or stamp > row_available:
                    raise ValueError(f"Feature value published after declared availability: {key}:{feature}")
        member_key = (row["security_id"], cycle)
        if member_key not in member_sectors or int(member_sectors[member_key]) != int(row["gsector"]):
            raise ValueError(f"No matching historical membership/sector: {key}")
        if pd.notna(row["y_return"]):
            if cycle not in label_ends:
                label_ends[cycle] = pd.Timestamp(
                    schedule(str(pd.Period(cycle, "Q") + 1))["execution_at"])
            label_end = label_ends[cycle]
            if (pd.isna(known.loc[i]) or known.loc[i] < label_end
                    or pd.isna(row["label_source_ref"]) or not row["label_source_ref"]):
                raise ValueError(f"Label maturity/source invalid: {key}")


def _panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError(f"Invalid market price dates: {path.name}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"Market dates must be ascending: {path.name}")
    if frame.index.tz is not None or not frame.index.equals(frame.index.normalize()):
        raise ValueError(f"Market dates must be timezone-naive session dates: {path.name}")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if np.isinf(frame.to_numpy()).any() or (frame <= 0).any().any():
        raise ValueError(f"Market values must be positive and finite or explicitly missing: {path.name}")
    return frame


def _execute_once(bundle: Path, output: Path, start: str, end: str) -> dict:
    verified = verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshot"
    shutil.copytree(bundle, snapshot)
    verify_bundle(snapshot)
    write_json(output / "rule_config.json", rule_manifest())
    code = Path(__file__).resolve().parent
    for source in sorted(code.glob("*.py")):
        destination = output / "code" / "pipeline" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(code.parent / "code/ml_model.py", output / "code/ml_model.py")
    write_json(output / "code_manifest.json", {
        str(p.relative_to(output / "code")): sha256(p)
        for p in (output / "code").rglob("*.py")
    })
    panel = pd.read_csv(snapshot / "panel.csv", dtype={"security_id": str})
    membership = pd.read_csv(snapshot / "membership.csv", dtype={"security_id": str})
    lineage = json.loads((snapshot / "lineage.json").read_text())
    validate_panel(panel, membership, lineage)
    raw = _panel(snapshot / "raw_prices.csv")
    covariance_levels = _panel(snapshot / "covariance_levels.csv")
    targets = {strategy: {} for strategy in ("equal_weight", "mean_var", "min_var")}
    quarterly, attempts = [], []
    with offline_guard(attempts):
        for cycle in continuation_cycles(start, end):
            timing = schedule(cycle)
            if pd.Timestamp(timing["execution_at"]) > pd.Timestamp.now(tz="UTC"):
                raise ValueError(f"Future execution period cannot be backtested: {cycle}")
            trade = panel[panel["cycle_key"] == cycle]
            if trade.empty:
                raise ValueError(f"No verified prediction rows for {cycle}; cannot skip quarter")
            expected_sectors = set(membership.loc[membership["cycle_key"] == cycle, "gsector"])
            if not expected_sectors or set(trade["gsector"]) != expected_sectors:
                raise ValueError(f"Prediction panel omits expected historical sectors: {cycle}")
            omitted = membership[(membership["cycle_key"] == cycle) &
                                 ~membership["security_id"].isin(trade["security_id"])]
            if not omitted.empty:
                if not {"exclusion_status", "exclusion_reason"}.issubset(omitted):
                    raise ValueError(f"Unexplained missing prediction securities: {cycle}")
                if (omitted["exclusion_reason"].isna().any() or
                        not omitted["exclusion_status"].isin(
                            ["NOT_PUBLIC_BY_CUTOFF", "MODEL_RULE_EXCLUDED", "CORPORATE_ACTION"]).all()):
                    raise ValueError(f"Unresolved source gaps cannot be historical exclusions: {cycle}")
                write_csv(output / f"{cycle}_source_exclusions.csv", omitted.to_dict("records"))
            selected, sector_info = {}, []
            for sector, rows in trade.groupby("gsector"):
                try:
                    result = fit_sector(panel[panel["gsector"] == sector], rows, cycle)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    raise ValueError(f"{cycle}, sector {sector}: {exc}") from exc
                selected.update(result["selected"])
                directory = output / "training" / cycle
                directory.mkdir(parents=True, exist_ok=True)
                result["train"].to_csv(directory / f"sector{sector}_training.csv", index=False)
                result["validation"].to_csv(directory / f"sector{sector}_validation.csv", index=False)
                sector_info.append({key: value for key, value in result.items()
                                    if key not in {"train", "validation"}})
                sector_info[-1]["gsector"] = int(sector)
                sector_info[-1]["excluded_securities"] = sorted(
                    set(rows["security_id"]) - set(result["forecasts"]))
                sector_info[-1]["training_periods"] = sorted(result["train"]["cycle_key"].unique())
                sector_info[-1]["validation_periods"] = sorted(result["validation"]["cycle_key"].unique())
            if len(selected) < 20:
                quarterly.append({**timing, "sector_models": sector_info,
                                  "status": "OPTIMIZATION_FAILED",
                                  "optimizer_errors": {"all": "Full investment under 5% cap requires >=20 names"}})
                write_json(output / f"{cycle}.json", quarterly[-1])
                continue
            weights, failures = research_weights(pd.Series(selected), covariance_levels, timing["information_cutoff"])
            quarterly.append({**timing, "sector_models": sector_info, "optimizer_errors": failures,
                              "target_definition": verified["definitions"]["target_definition"],
                              "covariance_definition": verified["definitions"]["covariance_definition"],
                              "status": "OPTIMIZATION_FAILED" if failures else "MODEL_COMPLETE_NOT_VALIDATED"})
            write_json(output / f"{cycle}.json", quarterly[-1])
            for strategy in targets:
                if strategy in weights:
                    targets[strategy][timing["execution_date"]] = weights[strategy]
        actions = pd.read_csv(snapshot / "corporate_actions.csv").replace({np.nan: None})
        required_action_columns = {"action_id", "ticker", "ex_date", "kind", "value", "pay_date"}
        if set(actions.columns) != required_action_columns:
            raise ValueError("Corporate action CSV schema invalid")
        corporate_actions = [CorporateAction(**row) for row in actions.to_dict("records")]
        first_execution = schedule(start)["execution_date"]
        completed_end = str(raw.index[-1].date())
        if raw.index[-1].date() >= datetime.now(timezone.utc).date():
            raise ValueError("Last market session must be fully completed before today's UTC date")
        next_execution = schedule(str(pd.Period(end, "Q") + 1))["execution_date"]
        completed_end = min(completed_end, next_execution)
        if datetime.now(timezone.utc).date().isoformat() < next_execution:
            holding_status = "ONGOING_HOLDING_PERIOD"
        elif completed_end < next_execution:
            holding_status = "INCOMPLETE_HISTORICAL_HOLDING_PERIOD"
        else:
            holding_status = "COMPLETED_HOLDING_PERIOD"
        first_valuation = pd.Timestamp(schedule(start)["information_cutoff"]).tz_convert(
            "America/New_York").tz_localize(None).normalize()
        valuation_prices = raw.loc[first_valuation:completed_end]
        if not valuation_prices.index.equals(nyse_sessions(first_valuation, completed_end)):
            raise ValueError("Daily raw market panel has missing NYSE sessions")
        for strategy, weights in targets.items():
            if len(weights) != len(quarterly):
                continue
            for label, cost in (("gross", 0), ("net", 10)):
                portfolio = simulate_portfolio(
                    valuation_prices, weights,
                    config=BacktestConfig(transaction_cost_bps_per_side=cost,
                                          fractional_shares=True, max_weight_per_asset=RULES.weight_cap),
                    corporate_actions=[a for a in corporate_actions
                                       if first_execution <= a.ex_date <= completed_end],
                )
                directory = output / strategy / label
                directory.mkdir(parents=True)
                portfolio.nav.rename("nav").to_csv(directory / "daily_nav.csv")
                portfolio.positions.to_csv(directory / "daily_positions.csv")
                portfolio.cash.rename("cash").to_csv(directory / "cash.csv")
                portfolio.dividend_receivables.rename("dividend_receivables").to_csv(directory / "receivables.csv")
                portfolio.trades.to_csv(directory / "trades.csv", index=False)
                pd.DataFrame.from_dict(weights, orient="index").fillna(0).rename_axis("date").to_csv(
                    directory / "target_weights.csv")
                actual = portfolio.positions.mul(valuation_prices.reindex(columns=portfolio.positions.columns))
                actual = actual.mask(portfolio.positions.eq(0), 0)
                actual = actual.div(portfolio.nav, axis=0)
                actual.to_csv(directory / "daily_actual_weights.csv")
                history = []
                for day, actual_weights in actual.iterrows():
                    if day.date().isoformat() < first_execution:
                        history.append({"date": day.date().isoformat(), "positions": []})
                        continue
                    effective = max(d for d in weights if d <= day.date().isoformat())
                    cycle = next(q["cycle_key"] for q in quarterly if q["execution_date"] == effective)
                    universe = membership[membership["cycle_key"] == cycle]
                    history.append({"date": day.date().isoformat(), "positions": [
                        {"security_id": r["security_id"], "sector": str(r["gsector"]),
                         "target_weight": weights[effective].get(r["security_id"], 0),
                         "actual_weight": float(actual_weights.get(r["security_id"], 0)),
                         "benchmark_weight": None, "benchmark_source": None}
                        for r in universe.to_dict("records")
                    ]})
                write_json(directory / "weight_history.json", history)
                write_csv(directory / "comparison.csv", [
                    {"date": day.date().isoformat(), "value": float(nav),
                     "valued_at": session_close_time(day).tz_convert("UTC").isoformat()}
                    for day, nav in portfolio.nav.items()
                ])
                write_json(directory / "definition.json", {
                    "currency": "USD", "return_type": "strategy_total_return",
                    "cost_basis": f"{cost}bps per side on actual turnover",
                    "dividend_tax_basis": "before_personal_tax",
                    "reinvestment_rule": "cash_until_rebalance",
                    "historical_validation": "NOT_VALIDATED",
                    "period": holding_status, "next_required_execution": next_execution,
                    "initial_cash_valuation": first_valuation.date().isoformat(),
                    "initial_cost_note": "Prior-close cash baseline retains the first execution's transaction costs",
                    "sources": verified["definitions"]["sources"],
                })
                write_json(directory / "comparison_metadata.json", {
                    "instrument_id": f"paper-fixed-lag-{output.parent.name}-{strategy}-{label}",
                    "name": f"paper_fixed_lag {strategy} {'vor' if label == 'gross' else 'nach'} Handelskosten",
                    "instrument_type": "strategy", "identifiers": {"strategy_id": [strategy + "-" + label]},
                    "share_class": "research_fractional_shares", "exchange": "NYSE_calendar",
                    "currency": "USD", "return_type": "strategy_total_return",
                    "cost_basis": f"{cost}bps per side on actual turnover",
                    "dividend_tax_basis": "before_personal_tax", "reinvestment_rule": "cash_until_rebalance",
                    "source": "reviewed bundle; offline research simulation, NOT_VALIDATED",
                    "source_url": verified["definitions"]["sources"][0],
                    "valuation_convention": "NYSE_close_UTC", "calendar": "NYSE",
                    "review_status": "SOURCE_REVIEWED", "observed_at": panel["reconstructed_at"].max(),
                    "weight_history_file": "weight_history.json",
                    "weight_history_sha256": sha256(directory / "weight_history.json"),
                    "evidence_files": [{"path": "../../snapshot/definitions.json",
                                        "sha256": sha256(snapshot / "definitions.json")}],
                })
    write_json(output / "execution_summary.json", {
        "quarters": quarterly, "network_attempts": attempts,
        "historical_validation": "NOT_VALIDATED",
        "simulation_exclusions": {strategy: "Missing quarterly optimizer result; no continuous NAV generated"
                                  for strategy in targets if len(targets[strategy]) != len(quarterly)},
    })
    return {"quarters": len(quarterly), "strategies": targets, "network_attempts": attempts}


def execute(bundle: Path, output: Path, start: str, end: str) -> dict:
    continuation_cycles(start, end)
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Execution output must be outside the immutable input bundle")
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(p): sha256(p) for p in bundle.rglob("*") if p.is_file()}
    attempts = []
    try:
        with offline_guard(attempts):
            first = _execute_once(bundle, output / "primary", start, end)
            second = _execute_once(output / "primary/snapshot", output / "replay", start, end)
        artifacts = []
        for directory in (output / "primary", output / "replay"):
            artifacts.append({str(p.relative_to(directory)): sha256(p) for p in directory.rglob("*")
                              if p.is_file() and "snapshot" not in p.relative_to(directory).parts})
        changed = [p for p, digest in sources.items() if sha256(Path(p)) != digest]
        if first != second or artifacts[0] != artifacts[1] or changed or attempts:
            raise RuntimeError(f"Independent offline model replay differs or inputs/network changed: {changed}")
        simulated = [s for s in first["strategies"]
                     if len(first["strategies"][s]) == first["quarters"]]
        result = {"status": "COMPLETED_NOT_VALIDATED" if len(simulated) == 3 else "COMPLETED_WITH_OPTIMIZER_ERRORS",
                  "quarters": first["quarters"], "simulated_strategies": simulated,
                  "model_replay_identical": True, "network_attempts": attempts,
                  "inputs_changed": changed, "historical_validation": "NOT_VALIDATED",
                  "artifact_hashes": artifacts[0]}
        write_json(output / "execution_summary.json", result)
        return result
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OptimizationError, np.linalg.LinAlgError) as exc:
        write_json(output / "execution_summary.json", {
            "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
            "historical_validation": "NOT_VALIDATED", "network_attempts": attempts,
            "model_replay_identical": False,
        })
        raise


def publish_execution(root: Path, output: Path) -> list[Path]:
    summary = json.loads((output / "execution_summary.json").read_text())
    if not summary.get("model_replay_identical") or summary["status"] == "FAILED":
        raise ValueError("Only independently replayed executions can be published")
    for relative, digest in summary["artifact_hashes"].items():
        if sha256(output / "primary" / relative) != digest:
            raise ValueError(f"Execution artifact changed before publication: {relative}")
    return [import_series(root, metadata.with_name("comparison.csv"), metadata)
            for metadata in sorted((output / "primary").glob("*/*/comparison_metadata.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2017Q2")
    parser.add_argument("--end", default="2026Q2")
    parser.add_argument("--publish", action="store_true", help="Import replayed research series into local dashboard registry")
    args = parser.parse_args()
    result = execute(args.bundle, args.output, args.start, args.end)
    print({k: v for k, v in result.items() if k != "artifact_hashes"})
    if args.publish:
        print(publish_execution(Path(__file__).resolve().parent.parent, args.output))


if __name__ == "__main__":
    main()
