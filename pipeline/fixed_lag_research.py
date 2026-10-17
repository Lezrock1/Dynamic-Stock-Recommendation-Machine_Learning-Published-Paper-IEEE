"""Isolated historical evidence audit; blocked inputs never become historical signals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from .paper_fixed_lag import continuation_cycles, rule_manifest, schedule
from .return_comparison import sha256

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("state/research/paper_fixed_lag")
OLD_Q2 = Path("state/diagnostics/runs/20261015T214633Z-q2-f73ec4")
Q2_MANIFEST = Path("state/expected_filings_q2_2026Q2/2026Q2.json")

RULE_COMPARISON = [
    ("Fixed two-month lag", "Current-cycle alignment after readiness", "paper_fixed_lag-v1 fixed NYSE date",
     "Prior-close cutoff and close execution are explicit causal assumptions, not proven paper intraday rules"),
    ("16–40 training quarters then 4 test quarters", "40 train; some sectors have 3 nonempty validation quarters",
     "Require 16–40 contiguous train and 4 complete validation quarters after filters",
     "One maturity gap necessary with prior-close cutoff"),
    ("Five families; MSE selection", "OLS/RF/Ridge/GBM/stepwise AIC",
     "Five explicit families with sector MSE selection", "Python defaults not exact original R/caret hyperparameter search"),
    ("20 ratios; >5% missing factor removal and stock/row deletion",
     "50% coverage and median imputation", "Training-only 5% factor filter and complete cases",
     "Paper stock deletion heuristic underspecified; NOT_VALIDATED"),
    ("Forward quarterly log return", "Live labels based on cached adjusted closes",
     "Require independently defined target-return source and label maturity",
     "Cached adj_close has no verified TR definition"),
    ("Top 20% per sector", "prediction >= quantile(0.8)",
     "ceil(20% eligible), prediction descending then security ID",
     "Deterministic tie/rounding change; legacy helper actually uses 70th percentile"),
    ("Quarter prediction and year daily covariance", "4*quarter expected, 252*daily simple covariance",
     "Annual log prediction, annual log covariance, log1p RF",
     "Different from original mixed/unspecified units; fixed 1.5% RF is an assumption"),
    ("0–5%, full investment, no leverage", "Equal variant may leave cash; optimized full investment",
     "Full investment only with >=20 feasible names; explicit optimizer error",
     "No substitute weights"),
    ("0.1% on changed shares * price", "Existing self-financing share/cash simulator",
     "Reuse simulate_portfolio with 10bps on each actual trade side",
     "Need verified raw-price share basis and action coverage"),
    ("Historical S&P500 member pool (~1142)", "Current mapping plus dated membership reconstruction",
     "Dated membership; historical class identity and sector required",
     "Historical mapping/sector evidence incomplete; not equivalent to original pool"),
]


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


@contextmanager
def offline_guard(attempts: list):
    def reject(*args, **kwargs):
        attempts.append("network_connection_blocked")
        raise RuntimeError("Historical offline research forbids network access")
    with patch.object(socket.socket, "connect", reject), patch.object(socket.socket, "connect_ex", reject), \
            patch.object(socket.socket, "sendto", reject), patch.object(socket, "getaddrinfo", reject), \
            patch.object(socket, "create_connection", reject):
        yield


def input_files(root: Path) -> list[Path]:
    files = [Path("cache/sp500_historical_snapshots.csv"), Path("cache/sp500_universe.csv"), Q2_MANIFEST]
    files += [p.relative_to(root) for p in (root / "cache/secsubmissions").glob("*.json")]
    files += [p.relative_to(root) for p in (root / "cache/secfacts").glob("*.json")]
    files += [p.relative_to(root) for p in (root / "cache/filings").rglob("*") if p.is_file()]
    files += [Path("Data/stocks_weight_table.xlsx")]
    files += [p.relative_to(root) for p in (root / "Data/2-portfolio_data").glob("*.xlsx")]
    files += [p.relative_to(root) for p in (root / "Data/1-focasting_data").glob("*.xlsx")]
    files += [p.relative_to(root) for p in (root / "Data/live").glob("*.csv")]
    files += [p.relative_to(root) for p in (root / "cache/prices").glob("*.csv")]
    for name in ("q2_gap_report.json", "forecast_records.json", "model_run_t50_enforced.json"):
        files.append(OLD_Q2 / name)
    files.append(OLD_Q2 / "training_data_summary.json")
    files += [p.relative_to(root) for p in (root / "pipeline").glob("*.py")]
    files += [Path("code/ml_model.py")]
    return sorted(set(files))


def snapshot_inputs(root: Path, destination: Path) -> dict:
    manifest = {}
    for relative in input_files(root):
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required research input missing: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = sha256(source)
        if sha256(target) != digest:
            raise RuntimeError(f"Source changed during snapshot creation: {relative}")
        manifest[str(relative)] = {"sha256": digest, "size": target.stat().st_size}
    return manifest


def _json(root: Path, relative: Path | str):
    return json.loads((root / relative).read_text())


def _filings(doc: dict) -> list[dict]:
    recent = doc.get("filings", {}).get("recent", {})
    names = ("form", "reportDate", "accessionNumber", "acceptanceDateTime", "primaryDocument")
    return [dict(zip(names, values)) for values in zip(*(recent.get(name, []) for name in names))]


def q2_crosswalk(root: Path, observed_at: str) -> tuple[list[dict], list[dict]]:
    manifest = _json(root, Q2_MANIFEST)
    gap = {row["ticker"]: row for row in _json(root, OLD_Q2 / "q2_gap_report.json")["rows"]}
    forecasts = {row["security_id"]: row for row in _json(root, OLD_Q2 / "forecast_records.json")}
    weights = _json(root, OLD_Q2 / "model_run_t50_enforced.json")["target_weights"]
    expected = {str(int(row["company_id"])): row for row in manifest["expected"]}
    cutoff = pd.Timestamp(schedule("2026Q2")["information_cutoff"])
    rows, fact_versions = [], []
    issuer_cache = {}
    for security in manifest["securities"]:
        ticker, cik = security["ticker"], security.get("cik")
        cik_key = str(int(cik)) if cik is not None else None
        entry = expected.get(cik_key, {})
        old_gap, old_forecast = gap.get(ticker, {}), forecasts.get(ticker, {})
        row = {
            "ticker": ticker, "cik": cik, "gsector": security.get("gsector"),
            "cycle_key": "2026Q2", "expected_period": entry.get("expected_period"),
            "information_cutoff": cutoff.isoformat(), "execution_date": "2026-09-01",
            "old_october_gap_status": old_gap.get("status"),
            "old_october_missing_metrics": old_gap.get("missing_metrics"),
            "old_october_feature_coverage": old_forecast.get("feature_coverage"),
            "old_october_forecast_kind": old_forecast.get("forecast_kind"),
            "old_october_eligible": old_forecast.get("eligible_for_current_cycle", False),
            "old_october_prediction": old_forecast.get("expected_return"),
            "old_october_weights": {key: value.get(ticker, 0) for key, value in weights.items()},
            "explanation": (
                "Gap readiness requires complete configured facts/features. ELIGIBLE in the old model "
                "means expected-period match plus the 50% model feature threshold, not full readiness. "
                "October forecasts/weights are not historical September decisions."
            ),
            "historical_source_status": "TODAY_INCOMPLETE_OR_UNRESOLVED",
            "historical_model_status": "NOT_RUN_UNVERIFIED_TRAINING_AND_RETURN_DEFINITION",
            "historical_forecast": None, "historical_weights": None,
            "local_reconstruction_checked_at": observed_at, "operational_facts_ready_at": None,
        }
        if cik_key is None:
            row["blocker"] = "UNRESOLVED_HISTORICAL_CIK"
            rows.append(row)
            continue
        if cik_key not in issuer_cache:
            path = root / "cache/secsubmissions" / f"{cik_key}.json"
            issuer_cache[cik_key] = _filings(json.loads(path.read_text())) if path.exists() else []
        candidates = [f for f in issuer_cache[cik_key] if
                      f["reportDate"] == entry.get("expected_period") and
                      f["form"] == entry.get("filing_type") and f["acceptanceDateTime"]]
        candidates.sort(key=lambda f: pd.Timestamp(f["acceptanceDateTime"]))
        public = [f for f in candidates if pd.Timestamp(f["acceptanceDateTime"]) <= cutoff]
        if not public:
            row["blocker"] = "OBSERVED_FIRST_REPORT_AFTER_CUTOFF" if candidates else "NO_LOCAL_PERIOD_ACCESSION_EVIDENCE"
            if candidates:
                row["historical_source_status"] = "LOCALLY_OBSERVED_REPORT_AFTER_CUTOFF_NOT_PROOF_NO_EARLIER_RELEASE"
                row["first_cached_report_at"] = candidates[0]["acceptanceDateTime"]
            rows.append(row)
            continue
        filing = public[0]
        accession = filing["accessionNumber"]
        document = root / "cache/filings" / cik_key / accession / filing["primaryDocument"]
        if not document.is_file():
            document = root / "cache/filings" / cik_key / accession.replace("-", "") / filing["primaryDocument"]
        row.update({
            "historical_source_status": "ORIGINAL_FILING_ACCEPTANCE_BEFORE_CUTOFF",
            "accession": accession, "published_at": filing["acceptanceDateTime"],
            "original_document_cached": document.is_file(),
            "original_document_sha256": sha256(document) if document.is_file() else None,
            "source_url": f"https://www.sec.gov/Archives/edgar/data/{cik_key}/"
            f"{accession.replace('-', '')}/{filing['primaryDocument']}",
            "blocker": "FEATURE_VERSION_LINEAGE_TRAINING_LABELS_AND_MARKET_ACTIONS_NOT_VERIFIED",
        })
        facts_path = root / "cache/secfacts" / f"{cik_key}.json"
        if facts_path.is_file() and not any(f["cik"] == cik for f in fact_versions):
            doc = json.loads(facts_path.read_text())
            accepted = {f["accessionNumber"]: f["acceptanceDateTime"]
                        for f in issuer_cache[cik_key] if f["acceptanceDateTime"]}
            version_counts, unknown_acceptance, later_versions = {}, 0, 0
            for namespace, concepts in doc.get("facts", {}).items():
                for tag, concept in concepts.items():
                    for unit, entries in concept.get("units", {}).items():
                        for value in entries:
                            if value.get("end") != entry["expected_period"]:
                                continue
                            stamp = accepted.get(value.get("accn"))
                            if stamp is None:
                                unknown_acceptance += 1
                            elif pd.Timestamp(stamp) > cutoff:
                                later_versions += 1
                            else:
                                key = f"{namespace}:{tag}:{unit}"
                                version_counts[key] = version_counts.get(key, 0) + 1
            fact_versions.append({
                "cik": cik, "expected_period": entry["expected_period"],
                "source_file_sha256": sha256(facts_path),
                "pre_cutoff_concept_unit_version_counts": version_counts,
                "unknown_accession_acceptance_versions": unknown_acceptance,
                "post_cutoff_versions_excluded": later_versions,
                "downloaded_at": None, "downloaded_at_note": "Cache fetch timestamp not proven; not inferred from mtime",
                "local_read_at": observed_at, "operational_facts_ready_at": None,
                "reconstruction_check": "INVENTORIED_NOT_FEATURE_LINEAGE_VERIFIED",
            })
        rows.append(row)
    return rows, fact_versions


def compute_audit(root: Path, observed_at: str) -> dict:
    original = pd.read_excel(root / "Data/stocks_weight_table.xlsx")
    dates = sorted(original["trade_date"].unique())
    archived_dates = {pd.to_datetime(str(int(d)), format="%Y%m%d").date().isoformat() for d in dates}
    schedule_checks = []
    for value in dates:
        day = pd.to_datetime(str(int(value)), format="%Y%m%d")
        cycle = str(pd.Period(day - pd.DateOffset(months=3), freq="Q"))
        expected = schedule(cycle)["execution_date"]
        schedule_checks.append({"archived": day.date().isoformat(), "cycle_key": cycle,
                                "next_nyse_rule": expected, "matches": expected == day.date().isoformat()})
    universe = pd.read_csv(root / "cache/sp500_universe.csv")
    mapping = {str(r.tic).replace(".", "-"): str(int(r.cik))
               for r in universe.dropna(subset=["cik"]).itertuples()}
    memberships = pd.read_csv(root / "cache/sp500_historical_snapshots.csv")
    memberships["date"] = pd.to_datetime(memberships["date"])
    sectors_historical = {}
    for path in sorted((root / "Data/1-focasting_data").glob("*.xlsx")):
        frame = pd.read_excel(path)
        for ticker in frame["tic"].unique():
            sectors_historical.setdefault(str(ticker), []).append(path.stem)
    cycles = []
    for cycle in continuation_cycles():
        timing = schedule(cycle)
        available = memberships[memberships["date"] <= timing["quarter_end"]]
        latest = available.sort_values("date").iloc[-1] if not available.empty else None
        tickers = sorted({t.replace(".", "-") for t in latest["tickers"].split(",")}) if latest is not None else []
        unmapped = [t for t in tickers if t not in mapping]
        cycles.append({
            **timing, "status": "BLOCKED_BEFORE_MODEL", "model_started": False,
            "model_completed": False, "forecast_count": 0, "weights_count": 0,
            "membership_snapshot": latest["date"].date().isoformat() if latest is not None else None,
            "security_count": len(tickers), "unmapped_historical_tickers": unmapped,
            "historical_sectors_status": "LOCAL_CURRENT_OR_ORIGINAL_POOL_MAPPING_NOT_TIME_VERSIONED",
            "blockers": [
                "No source-verified historical feature panel preserving per-signal accession/value-version lineage",
                "Historical sectors, identities and successor/class corporate actions not fully evidenced",
                "No verified forward-target/price total-return definition and complete delisting/action history",
                "No train-only factor-definition overlap reconciliation between original Compustat and SEC",
            ],
            "next_step": "Supply/verify period-specific source lineage and historical market/actions coverage; "
                         "run paper_fixed_lag.window_frames/fit_sector only after these prerequisites pass",
            "validation": "NOT_VALIDATED",
        })
    rows, facts = q2_crosswalk(root, observed_at)
    panels = []
    for path in sorted((root / "Data/live").glob("sector*_hybrid.csv")):
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame["tradedate"], errors="coerce")
        panels.append({
            "file": str(path.relative_to(root)), "sha256": sha256(path),
            "status": "PRESENT_INSPECTED_NOT_HISTORICAL_LINEAGE_VERIFIED",
            "rows": len(frame), "first_available_trade_date": str(dates.min().date()),
            "last_available_trade_date": str(dates.max().date()),
            "unparseable_dates": int(dates.isna().sum()),
            "available_quarters": int(dates.dt.to_period("Q").nunique()),
            "rows_with_target": int(frame["y_return"].notna().sum()),
            "source_counts": frame["data_source"].fillna("UNSPECIFIED").value_counts().to_dict(),
            "feature_missing_counts": frame.filter(regex=r"^X\d+_").isna().sum().to_dict(),
            "label_known_at_present": "label_known_at" in frame,
            "historical_model_rows_used": 0,
            "note": "Available hybrid inventory is not an executed historical training-window export.",
        })
    source_counts = pd.Series([r["historical_source_status"] for r in rows]).value_counts().to_dict()
    return {
        "schema_version": 1, "strategy_variant": "paper_fixed_lag", "rule": rule_manifest(),
        "observed_at": observed_at, "audit_status": "COMPLETED_WITH_BLOCKED_MODEL_RUNS",
        "historical_validation": "NOT_VALIDATED", "executable": False,
        "scheduler_enabled": False, "orders_created": 0, "production_database_opened": False,
        "handoff": {
            "last_archived_model_date": max(archived_dates),
            "first_new_execution": schedule("2017Q2")["execution_date"],
            "new_decision_count": len(cycles),
            "archived_schedule_checks": schedule_checks,
            "archived_schedule_mismatch_count": sum(not c["matches"] for c in schedule_checks),
            "nav_handoff_status": "NOT_VERIFIED_NO_PAPER_NAV_SPLICING",
            "paper_reference": "Model ends June 2017; final holding period extends to September 2017",
        },
        "rule_comparison": [dict(zip(("paper", "current", "continuation", "deviation"), r))
                            for r in RULE_COMPARISON],
        "quarters": cycles, "q2_crosswalk": rows, "q2_fact_inventory": facts,
        "q2_summary": {
            "securities": len(rows), "source_status_by_security": source_counts,
            "issuers_with_original_filing_before_cutoff": len({
                r["cik"] for r in rows if r["historical_source_status"] == "ORIGINAL_FILING_ACCEPTANCE_BEFORE_CUTOFF"}),
            "verified_historical_factor_panels": 0, "historical_forecasts": 0,
            "completeness": "NOT_PROVEN",
        },
        "available_hybrid_panel_inventory": panels,
        "prior_october_diagnostic_training": _json(root, OLD_Q2 / "training_data_summary.json"),
        "model_run_status": "NOT_STARTED_SOURCE_PREREQUISITES_FAILED",
        "performance_status": "NOT_COMPUTED_NO_VERIFIED_CONTINUOUS_SIMULATION",
        "external_requirements": [
            "Real configured DYNAMIC_STOCK_USER_AGENT for original filing/submissions retrieval",
            "Historical identifier/sector/action/delisting sources and definition-evidenced prices/targets",
            "Source-reviewed total-return comparison series or CSV + reviewed source metadata",
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def run(root: Path = ROOT) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = root / RUNS / run_id
    snapshot = out / "snapshot"
    out.mkdir(parents=True, exist_ok=False)
    protected_paths = [Path("state/real_paper.sqlite3"), Path("state/real_paper_q2_2026Q2.sqlite3"),
                       Path("state/expected_filings/2026Q3.json"), Q2_MANIFEST]
    protected_paths += [p.relative_to(root) for p in (root / OLD_Q2).iterdir() if p.is_file()]
    protected = {str(p): sha256(root / p) for p in protected_paths if (root / p).is_file()}
    inputs = snapshot_inputs(root, snapshot)
    write_json(out / "input_manifest.json", inputs)
    write_json(out / "config.json", rule_manifest())
    attempts = []
    with offline_guard(attempts):
        first = compute_audit(snapshot, now)
        second = compute_audit(snapshot, now)
    replay = first == second
    changed = [name for name, item in inputs.items() if sha256(snapshot / name) != item["sha256"]]
    source_changed = [name for name, item in inputs.items() if sha256(root / name) != item["sha256"]]
    protected_after = {name: sha256(root / name) for name in protected}
    if not replay or changed or source_changed or attempts or protected != protected_after:
        raise RuntimeError(f"Offline audit replay failed: {changed}, {source_changed}, {attempts}")
    write_json(out / "protected_inputs.json", {
        "before": protected, "after": protected_after, "unchanged": protected == protected_after})
    write_json(out / "research_report.json", first)
    write_json(out / "research_report_replay.json", second)
    write_csv(out / "quarter_status.csv", first["quarters"])
    write_csv(out / "q2_crosswalk.csv", first["q2_crosswalk"])
    write_csv(out / "rule_comparison.csv", first["rule_comparison"])
    write_json(out / "q2_fact_inventory.json", first["q2_fact_inventory"])
    write_csv(out / "available_hybrid_panels.csv", first["available_hybrid_panel_inventory"])
    write_json(out / "prior_october_diagnostic_training.json", first["prior_october_diagnostic_training"])
    with sqlite3.connect(out / "research.sqlite3") as con:
        con.execute("PRAGMA user_version=1")
        con.execute("CREATE TABLE quarter_audits(cycle_key TEXT PRIMARY KEY,status TEXT,payload_json TEXT)")
        con.execute("CREATE TABLE security_crosswalk(ticker TEXT PRIMARY KEY,payload_json TEXT)")
        con.executemany("INSERT INTO quarter_audits VALUES(?,?,?)",
                        [(r["cycle_key"], r["status"], json.dumps(r)) for r in first["quarters"]])
        con.executemany("INSERT INTO security_crosswalk VALUES(?,?)",
                        [(r["ticker"], json.dumps(r)) for r in first["q2_crosswalk"]])
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    write_json(out / "replay.json", {"audit_outputs_identical": replay, "network_attempts": attempts,
                                    "snapshot_changed": changed, "source_changed": source_changed,
                                    "sqlite_integrity": integrity, "model_replay": "NOT_RUN_MODEL_BLOCKED"})
    artifact_manifest = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "artifact_manifest.json", artifact_manifest)
    write_json(root / RUNS / "latest.json", {"run_id": run_id,
                                           "report_sha256": sha256(out / "research_report.json")})
    return out


def load_summary(root: Path) -> dict:
    pointer = root / RUNS / "latest.json"
    if not pointer.exists():
        return {"available": False, "error": "No fixed-lag research audit published"}
    try:
        latest = json.loads(pointer.read_text())
        if Path(latest["run_id"]).name != latest["run_id"]:
            raise ValueError("Invalid research run ID")
        path = root / RUNS / latest["run_id"] / "research_report.json"
        if sha256(path) != latest["report_sha256"]:
            raise ValueError("Research report hash mismatch")
        doc = json.loads(path.read_text())
        return {"available": True, "run_id": latest["run_id"],
                **{k: v for k, v in doc.items() if k not in {
                    "q2_crosswalk", "q2_fact_inventory", "available_hybrid_panel_inventory",
                    "prior_october_diagnostic_training"}}}
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {"available": False, "error": f"Fixed-lag research unavailable: {exc}"}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("SEC_USER_AGENT_CONFIGURED", bool(os.environ.get("DYNAMIC_STOCK_USER_AGENT")))
    out = run()
    print("AUDIT_COMPLETE", out)
    print("MODEL_NOT_STARTED: unverified historical training/market prerequisites; NOT_VALIDATED")


if __name__ == "__main__":
    main()
