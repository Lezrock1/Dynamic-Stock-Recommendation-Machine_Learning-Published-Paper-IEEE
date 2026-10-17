"""Read-only model-coverage diagnosis and readiness breakdown for one reporting cycle.

Nothing here writes to the paper database, creates a signal or relaxes a readiness gate.
Per ticker it separates data import, feature construction, training, forecast and optimisation
and distinguishes MAPPING_ERROR / FETCH_ERROR / NOT_APPLICABLE / MISSING_DATA.

    python model_coverage.py diagnose --cycle-key 2026Q3 [--model-run FILE]
    python model_coverage.py readiness --cycle-key 2026Q3
    python model_coverage.py model-run --cycle-key 2026Q3     # slow (~30 min); writes FILE, no DB access
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

try:
    from . import freeze as fz, paper_live as pl, prices as pr, ratios as ra, sec_edgar as se
    from . import universe_asof as ua
    from .paper_timeline import cycle_timeline
except ImportError:  # pragma: no cover - script execution
    import freeze as fz  # type: ignore
    import paper_live as pl  # type: ignore
    import prices as pr  # type: ignore
    import ratios as ra  # type: ignore
    import sec_edgar as se  # type: ignore
    import universe_asof as ua  # type: ignore
    from paper_timeline import cycle_timeline  # type: ignore

# Feature -> raw inputs. A feature is NOT_APPLICABLE only if the company never filed the concept.
FEATURE_INPUTS = {
    "X1_REVGH": ["Revenue"], "X6_PS": ["Revenue", "SharesOutstanding"], "X7_NPM": ["Revenue"],
    "X9_OM": ["Revenue", "OperatingIncome"], "X8_GPM": ["Revenue", "COGS"],
    "X12_CR": ["AssetsCurrent", "LiabilitiesCurrent"], "X16_WCR": ["AssetsCurrent", "LiabilitiesCurrent"],
    "X18_QR": ["AssetsCurrent", "LiabilitiesCurrent", "InventoryNet"],
    "X19_DSI": ["InventoryNet", "COGS"], "X20_DPO": ["AccountsPayableCurrent", "COGS"],
    "X5_PE": ["SharesOutstanding"], "X10_PB": ["SharesOutstanding"], "X13_EM": ["SharesOutstanding"],
    "X11_PCFO": ["CFO", "SharesOutstanding"], "X14_EVCFO": ["CFO", "SharesOutstanding"],
}
SHARE_REASON = "no share count / EPS in companyfacts (multi-class listings report dimensioned values only)"


def _tag_evidence(facts: Mapping | None) -> dict[str, dict]:
    """Per raw field: tags with data, unit and last period end among 10-Q/10-K facts."""
    gaap = ((facts or {}).get("facts") or {}).get("us-gaap") or {}
    dei = ((facts or {}).get("facts") or {}).get("dei") or {}
    out: dict[str, dict] = {}
    for field, tags in {**se.DURATION_TAGS, **se.INSTANT_TAGS}.items():
        found = []
        sources = [(t, gaap) for t in tags] + [(t, dei) for t in se.DEI_FALLBACK_TAGS.get(field, [])]
        for tag, namespace in sources:
            for unit, entries in ((namespace.get(tag) or {}).get("units") or {}).items():
                ends = [e["end"] for e in entries if e.get("form") in ("10-Q", "10-K")]
                if ends:
                    quarterly = sum(1 for e in entries if e.get("form") == "10-Q")
                    found.append({"tag": tag, "unit": unit, "values": len(ends), "quarterly_values": quarterly,
                                  "last_period": max(ends)})
        out[field] = {"tags": found, "present": bool(found)}
    return out


def diagnose_ticker(config: Mapping, tic: str, cik: int, gsector: int, snapshot: pd.Timestamp,
                    price_session: pd.Timestamp, sector_features: list[str]) -> dict:
    required = pl.required_facts_for(config, gsector, cik)
    record: dict = {"ticker": tic, "cik": cik, "gsector": gsector, "required_facts": required}
    facts = se.get_company_facts(cik)
    if facts is None or "us-gaap" not in (facts.get("facts") or {}):
        return {**record, "stage": "DATA_IMPORT", "cause": "FETCH_ERROR",
                "detail": "no companyfacts us-gaap data for this CIK"}
    evidence = _tag_evidence(facts)
    record["tags"] = {k: v["tags"] for k, v in evidence.items() if v["present"]}
    acceptance = se.acceptance_times(se.get_submissions(cik, max_age_seconds=6 * 3600))
    with se.point_in_time(snapshot, acceptance):
        fund = se.get_quarterly_fundamentals(cik)
    history = pr.get_price_history(tic)
    record["price_days"] = 0 if history is None else int(len(history))
    if fund is not None and not fund.empty:
        record["ixbrl_fallback_fields"] = sorted(
            c[:-len("_ixbrl_provenance")] for c in fund.columns
            if c.endswith("_ixbrl_provenance") and fund[c].notna().any())
    if fund is None or fund.empty:
        return {**record, "stage": "DATA_IMPORT", "cause": "MISSING_DATA", "detail": "no point-in-time quarters"}
    if history is None or history.empty:
        return {**record, "stage": "DATA_IMPORT", "cause": "FETCH_ERROR", "detail": "no price history"}
    missing_cols = [c for c in required if c not in fund.columns]
    if missing_cols:
        # A tag that exists only in annual filings (no 10-Q yet, e.g. a fresh spin-off) is missing
        # data, not a mapping error; a mapping error needs quarterly facts that were not picked up.
        mapping_gap = [c for c in missing_cols
                       if any(t["quarterly_values"] for t in evidence[c]["tags"])]
        return {**record, "stage": "FEATURES", "missing_required": missing_cols,
                "cause": "MAPPING_ERROR" if mapping_gap else "MISSING_DATA",
                "detail": f"required facts without point-in-time values: {missing_cols}"}
    complete = fund.dropna(subset=required)
    record["company_quarters_total"] = int(len(fund))
    record["company_quarters_with_required_facts"] = int(len(complete))
    if complete.empty:
        return {**record, "stage": "FEATURES", "cause": "MISSING_DATA", "detail": "no quarter has all required facts"}
    try:
        ratios = ra.compute_ratios(complete, history, tic, price_session=price_session)
    except ValueError as exc:
        return {**record, "stage": "FEATURES", "cause": "MISSING_DATA", "detail": str(exc)}
    latest = ratios.sort_values("datadate").iloc[-1]
    record["latest_period"] = pd.Timestamp(latest["datadate"]).date().isoformat()
    record["quarters_with_rows"] = int(len(ratios))
    status: dict[str, dict] = {}
    for feature in sector_features:
        value = latest[feature]
        if np.isfinite(value) if isinstance(value, (int, float, np.floating)) else False:
            status[feature] = {"state": "OK"}
            continue
        inputs = FEATURE_INPUTS.get(feature, [])
        absent = [i for i in inputs if not evidence.get(i, {}).get("present")]
        if feature in ("X2_EPS", "X3_ROA", "X4_ROE", "X15_LTDTA", "X17_DE", "X5_PE") and "SharesOutstanding" in absent:
            status[feature] = {"state": "MISSING", "reason": SHARE_REASON}
        elif absent and "SharesOutstanding" in absent:
            status[feature] = {"state": "MISSING", "reason": SHARE_REASON}
        elif absent:
            status[feature] = {"state": "NOT_APPLICABLE",
                               "reason": f"company never filed {absent} (concept absent from its statements)"}
        else:
            status[feature] = {"state": "MISSING", "reason": "inputs exist but value unavailable for the latest period "
                               "(e.g. fewer than 4 consecutive quarters for TTM, or no price on that date)"}
    record["feature_status"] = status
    ok = sum(1 for v in status.values() if v["state"] == "OK")
    coverage = ok / len(sector_features)
    record["feature_coverage"] = round(coverage, 3)
    minimum = float(config["model"].get("min_feature_coverage", 0.5))
    if coverage < minimum:
        return {**record, "stage": "FEATURES", "cause": "MISSING_DATA",
                "detail": f"feature coverage {coverage:.2f} < {minimum}"}
    return {**record, "stage": "FEATURES_OK", "cause": None, "detail": "usable features"}


def fallback_trace(config: Mapping, cycle_key: str, evaluation_time: pd.Timestamp) -> dict:
    """Original filing -> extracted fact -> computed feature -> model usability, with and without the fallback."""
    try:
        from . import filing_audit as fa
    except ImportError:  # pragma: no cover
        import filing_audit as fa  # type: ignore
    manifest = json.loads((pl.resolve_path(config, "manifest_dir") / f"{cycle_key}.json").read_text())
    members, _ = ua.members_as_of(pd.Timestamp(manifest["universe_as_of"]))
    members = members.drop_duplicates("tic").set_index("tic")
    sessions = pl.nyse_sessions(evaluation_time.normalize() - pd.Timedelta(days=10), evaluation_time.normalize())
    price_session = [d for d in sessions if pl.session_close_time(d) <= evaluation_time][-1]
    configured = dict(se.IXBRL_FALLBACK)
    cases = {}
    for ticker, cik, accession, document, period_end, _days in fa.CASES:
        sector = int(members.loc[ticker, "gsector"])
        features = pl.sector_feature_set(sector)
        outcome = {}
        for label, enabled in (("without_fallback", False), ("with_fallback", True)):
            se.register_ixbrl_fallback({**configured, "enabled": enabled})
            diag = diagnose_ticker(config, ticker, cik, sector, evaluation_time, price_session, features)
            outcome[label] = {k: diag.get(k) for k in ("stage", "cause", "detail", "feature_coverage",
                                                       "missing_required", "ixbrl_fallback_fields")}
            outcome[label]["features_ok"] = sorted(f for f, v in (diag.get("feature_status") or {}).items()
                                                   if v["state"] == "OK")
            outcome[label]["features_missing"] = {f: v.get("reason", v["state"]) for f, v in
                                                  (diag.get("feature_status") or {}).items() if v["state"] != "OK"}
        se.register_ixbrl_fallback(configured)
        acceptance = se.acceptance_times(se.get_submissions(cik, max_age_seconds=10**9))
        with se.point_in_time(evaluation_time, acceptance):
            fund = se.get_quarterly_fundamentals(cik)
        extracted = []
        for col in [c for c in fund.columns if c.endswith("_ixbrl_provenance")]:
            for period, value in fund[col].dropna().items():
                field = col[:-len("_ixbrl_provenance")]
                extracted.append({"field": field, "period": period.date().isoformat(),
                                  "value": float(fund.loc[period, field]), **json.loads(value)})
        cases[ticker] = {"cik": cik, "gsector": sector, "audited_filing": accession,
                         "extracted_by_fallback": extracted, **outcome,
                         "model_usable_after": outcome["with_fallback"]["stage"] == "FEATURES_OK"}
    return {"rule_version": configured.get("rule_version"), "evaluation_time": evaluation_time.isoformat(),
            "cases": cases}


def _db_filings(db: Path, cycle_key: str) -> dict[str, str]:
    """Read ML-cycle report status only; SPY's separate portfolio cannot change readiness."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT e.company_id, e.status FROM expected_filings e JOIN reporting_cycles c "
            "ON c.cycle_id = e.cycle_id WHERE c.cycle_key = ? AND c.strategy != 'spy_buy_hold_v1' "
            "ORDER BY c.cycle_id", (cycle_key,)).fetchall()
    finally:
        connection.close()
    seen: dict[str, str] = {}
    for company, status in rows:
        company = str(company)
        if company in seen and seen[company] != status:
            raise ValueError(f"ML cycles disagree on filing status for issuer {company}: "
                             f"{seen[company]} vs {status}")
        seen[company] = status
    return seen


def readiness_breakdown(config: Mapping, db_status: Mapping[str, str], manifest: Mapping) -> dict:
    """Received / evidenced special cases / open - a resolved case is never READY."""
    resolutions = (config.get("readiness_rule") or {}).get("corporate_action_resolutions") or {}
    received, special, open_cases = [], [], []
    for item in manifest["expected"]:
        company, key = item["company_id"], item["filing_key"]
        status = db_status.get(company, "NOT_IN_DB")
        resolution = resolutions.get(company)
        if resolution and key in resolution.get("applies_to_filing_keys", []) \
                and resolution.get("final_trading_end") and resolution.get("evidence"):
            special.append({"ticker": item["security_id"], "status": resolution["status"],
                            "db_status": status, "event": resolution["event"],
                            "evidence": resolution["evidence"], "counts_as_ready": False, "satisfies_gate": True,
                            "buy_candidate": False})
        elif status == "READY":
            received.append(item["security_id"])
        else:
            open_cases.append({"ticker": item["security_id"], "status": status})
    waiting_facts = [c for c in open_cases if c["status"] == "WAITING_FACTS"]
    invalid_data = [c for c in open_cases if c["status"] == "INVALID_DATA"]
    groups = {
        "READY": len(received),
        "WAITING_FACTS": len(waiting_facts),
        "CORPORATE_ACTION_RESOLVED": len(special),
        "WAITING_FILING": sum(1 for c in open_cases if c["status"] in ("WAITING_FILING", "NOT_IN_DB")),
        "INVALID_DATA": len(invalid_data),
    }
    assert sum(groups.values()) == len(manifest["expected"])
    return {
        "readiness_rule_version": (config.get("readiness_rule") or {}).get("version"),
        "status_groups": groups, "status_groups_sum": sum(groups.values()),
        "open_total": groups["WAITING_FACTS"] + groups["WAITING_FILING"] + groups["INVALID_DATA"],
        "expected": len(manifest["expected"]), "received_ready": len(received),
        "evidenced_special_cases_not_ready": special,
        "open": len(open_cases), "open_cases_by_status": dict(Counter(c["status"] for c in open_cases)),
        "gate_note": "rule v3: every expected filing must be READY or an evidenced CORPORATE_ACTION_RESOLVED case; "
                     "special cases are never counted as READY or buy candidates; no automatic exception exists",
        "gate_satisfied": not open_cases,
        "buy_blocks": {k: v["ticker"] for k, v in ((config.get("readiness_rule") or {}).get("buy_blocks") or {}).items()},
        "open_waiting_facts": [c["ticker"] for c in open_cases if c["status"] == "WAITING_FACTS"],
    }


def run_diagnosis(config: Mapping, cycle_key: str, model_run: Mapping | None,
                  snapshot: pd.Timestamp | None = None) -> dict:
    manifest = json.loads((pl.resolve_path(config, "manifest_dir") / f"{cycle_key}.json").read_text())
    as_of = pd.Timestamp(manifest["universe_as_of"])
    members, _ = ua.members_as_of(as_of)
    members = members.drop_duplicates("tic").assign(cik=lambda f: f["cik"].astype(int))
    db = pl.resolve_path(config, "database")
    db_status = _db_filings(db, cycle_key)
    snapshot = snapshot if snapshot is not None else pd.Timestamp.now(tz="UTC")
    sessions = pl.nyse_sessions(snapshot.normalize() - pd.Timedelta(days=10), snapshot.normalize())
    price_session = [d for d in sessions if pl.session_close_time(d) <= snapshot][-1]
    feature_sets: dict[int, list[str]] = {}
    for sector in sorted(members["gsector"].astype(int).unique()):
        feature_sets[sector] = pl.sector_feature_set(int(sector))
    tickers = []
    for row in members.sort_values("tic").itertuples():
        diag = diagnose_ticker(config, str(row.tic), int(row.cik), int(row.gsector), snapshot, price_session,
                               feature_sets[int(row.gsector)])
        diag["member_at_universe_as_of"] = True
        diag["filing_status"] = db_status.get(str(int(row.cik)), "NOT_IN_DB")
        tickers.append(diag)
    by_tic = {t["ticker"]: t for t in tickers}
    snapshot_data = (model_run or {}).get("data_snapshot", {})
    reasons = snapshot_data.get("skip_reasons") or {}
    forecasts = (model_run or {}).get("forecasts", {})
    weights = {t for w in (model_run or {}).get("target_weights", {}).values() for t in w}
    for tic, diag in by_tic.items():
        if model_run:
            diag["predicted"] = tic in forecasts
            diag["optimisable"] = tic in weights
            if tic in reasons:
                diag["model_run_skip"] = reasons[tic]
            elif diag["stage"] == "FEATURES_OK" and tic not in forecasts:
                diag["not_predicted_reason"] = ("no row in the pending trade-date slot (company has no newly "
                                                 "available report in the cycle window) or sector not trained")
    sectors = {}
    for sector in sorted(feature_sets):
        rows = [t for t in tickers if t["gsector"] == sector]
        sectors[str(sector)] = {
            "members": len(rows),
            "report_ready": sum(1 for t in rows if t["filing_status"] == "READY"),
            "usable_features": sum(1 for t in rows if t["stage"] == "FEATURES_OK"),
            "predicted": sum(1 for t in rows if t.get("predicted")) if model_run else None,
            "optimisable": sum(1 for t in rows if t.get("optimisable")) if model_run else None,
            "sector_feature_set": feature_sets[sector],
            "excluded": {t["ticker"]: f'{t["stage"]}/{t["cause"]}' for t in rows if t["stage"] != "FEATURES_OK"},
        }
    outside = [t for t in reasons if t not in by_tic]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "cycle_key": cycle_key,
        "universe_as_of": manifest["universe_as_of"], "readiness": readiness_breakdown(config, db_status, manifest),
        "model_run_present": bool(model_run),
        "model_run_universe_as_of": snapshot_data.get("universe_as_of"),
        "skipped_outside_cycle_universe": outside,
        "sector_coverage": sectors,
        "tickers": tickers,
        "note": "Training data are the pooled historical sector panels (Compustat-era files); the current "
                "candidates are the live rows of the cycle universe. Historical validation stays NOT_VALIDATED.",
    }


def _forecast_comparison(diagnostic: Mapping, enforced: Mapping) -> dict:
    """Record-level difference between the diagnostic run and the eligibility-enforced run (same threshold)."""
    a = {r["security_id"]: r for r in diagnostic["data_snapshot"]["forecast_records"]}
    b = {r["security_id"]: r for r in enforced["data_snapshot"]["forecast_records"]}
    keep = ("gsector", "expected_period", "used_period", "eligible_for_current_cycle", "eligibility_reason",
            "forecast_kind", "expected_return")
    only_a = {t: {k: a[t].get(k) for k in keep} for t in sorted(set(a) - set(b))}
    only_b = {t: {k: b[t].get(k) for k in keep} for t in sorted(set(b) - set(a))}
    return {"diagnostic_count": len(a), "enforced_count": len(b), "in_both": len(set(a) & set(b)),
            "only_in_diagnostic": only_a, "only_in_enforced": only_b,
            "skipped_in_enforced_for_only_in_diagnostic": {
                t: (enforced["data_snapshot"].get("skip_reasons") or {}).get(t) for t in only_a},
            "expected_return_differs_for_common": {
                t: [a[t]["expected_return"], b[t]["expected_return"]] for t in sorted(set(a) & set(b))
                if a[t]["expected_return"] != b[t]["expected_return"]}}


def _model_rules(config: Mapping, report: Mapping, runs: Mapping) -> dict:
    """Features actually used per sector model plus the rules applied to them."""
    sectors = {}
    for sector, body in report["sector_coverage"].items():
        used = {label: (run["data_snapshot"].get("sector_models") or {}).get(sector)
                for label, run in runs.items()}
        sectors[sector] = {"features": body["sector_feature_set"], "feature_count": len(body["sector_feature_set"]),
                           "model_by_run": used, "members": body["members"], "usable_features": body["usable_features"],
                           "predicted": body["predicted"], "optimisable": body["optimisable"]}
    return {
        "feature_selection": "features with at least one non-NaN value in the sector's historical training "
                             "panel (Compustat-era files); identical for training and live rows",
        "coverage_definition": "share of the sector's selected features that are finite (not NaN/inf) in a row",
        "min_feature_coverage_by_run": {k: v["threshold"] for k, v in runs.items()},
        "required_fields": "none individually; only the coverage share is enforced",
        "missing_values": "rows below the threshold are skipped (LOW_FEATURE_COVERAGE); remaining NaN are "
                          "imputed by SimpleImputer(median) inside each model pipeline, fitted on the training "
                          "window only; missing is never turned into zero",
        "sectors": sectors}


def _code_hash(
    config_path: str | None,
    repo_root: str | Path | None = None,
    git_root: str | Path | None = None,
) -> dict:
    import hashlib
    root = Path(repo_root or pl.REPO_ROOT).resolve()
    source_files = sorted((root / "pipeline").glob("*.py")) + sorted((root / "code").glob("*.py"))
    requirements = root / "pipeline" / "requirements.txt"
    if requirements.is_file():
        source_files.append(requirements)
    cfg = Path(config_path) if config_path else root / "config" / "real_paper.json"
    if cfg.is_file():
        source_files.append(cfg)
    parts = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    git_cwd = Path(git_root or root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=git_cwd, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "pipeline", "code", "config"],
        cwd=git_cwd, text=True, capture_output=True, check=False
    )
    return {
        "combined_sha256": hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest(),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "files": parts,
    }


def _sqlite_content_hash(path: Path) -> str:
    import hashlib
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        digest = hashlib.sha256()
        for statement in connection.iterdump():
            digest.update(statement.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
    finally:
        connection.close()


def _cycle_timeline_rows(connection: sqlite3.Connection, cycle_key: str) -> list[Mapping[str, object]]:
    connection.row_factory = sqlite3.Row
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(expected_filings)")
    }
    if "facts_ready_at" not in columns:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        raise ValueError(
            f"cycle {cycle_key} database schema {version} lacks expected_filings.facts_ready_at; "
            "migrate a secured database copy and recheck existing READY filings to record a verified "
            "availability time. Do not infer it from filed_at, first_seen_at, or updated_at."
        )
    rows = connection.execute(
        "SELECT e.filing_key,e.filed_at,e.first_seen_at,e.facts_ready_at,r.recorded_at "
        "FROM expected_filings e JOIN reporting_cycles c ON c.cycle_id=e.cycle_id "
        "LEFT JOIN filing_resolutions r ON r.cycle_id=e.cycle_id AND r.filing_key=e.filing_key "
        "WHERE c.cycle_key=? AND c.strategy!='spy_buy_hold_v1'",
        (cycle_key,),
    ).fetchall()
    missing = [
        row["filing_key"] for row in rows
        if not row["recorded_at"] and not all(
            row[field] for field in ("filed_at", "first_seen_at", "facts_ready_at")
        )
    ]
    if missing:
        raise ValueError(
            f"cycle {cycle_key} has {len(missing)} ordinary filing rows without complete persisted "
            "publication/observation/facts-readiness timestamps; recheck them before a point-in-time run"
        )
    return [dict(row) for row in rows]


def _data_fingerprint(config: Mapping) -> dict:
    import hashlib
    manifest = fz.build_file_manifest(config)
    files = manifest["files"]
    groups = {}
    for category, prefix in (
        ("historical_training", "Data/1-focasting_data/"),
        ("live_panels", "Data/live/"),
        ("prices", "cache/prices/"),
        ("secfacts", "cache/secfacts/"),
        ("submissions", "cache/secsubmissions/"),
        ("filings", "cache/filings/"),
    ):
        entries = {path: item for path, item in files.items() if path.startswith(prefix)}
        groups[category] = {
            "file_count": len(entries),
            "combined_sha256": hashlib.sha256(
                json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    groups["all_inputs"] = {"file_count": manifest["file_count"],
                            "combined_sha256": manifest["combined_sha256"]}
    return groups


THRESHOLDS = (0.5, 0.75, 1.0)


def suite(
    config: Mapping,
    cycle_key: str,
    config_path: str | None,
    compare_run_id: str | None = None,
    snapshot_from: str | Path | None = None,
) -> Path:
    """Build or reuse one immutable input snapshot, then run all thresholds offline against it."""
    import copy
    import uuid

    source_root = pl.resolve_path(config, "database").parent.parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = pl.resolve_path(config, "database").parent / "diagnostics" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    snapshot_root = out / "snapshot"
    source_manifest_before = fz.build_file_manifest(config)
    (out / "source_manifest_before.json").write_text(
        json.dumps(source_manifest_before, indent=1, sort_keys=True)
    )
    if snapshot_from is None:
        snapshot_db = fz.create_input_snapshot(config, snapshot_root)
    else:
        snapshot_root = Path(snapshot_from).expanduser().resolve()
        if not snapshot_root.is_dir():
            raise FileNotFoundError(f"shared input snapshot not found: {snapshot_root}")
        snapshot_db = snapshot_root / pl.resolve_path(config, "database").relative_to(source_root)
        if not snapshot_db.is_file():
            raise FileNotFoundError(f"shared snapshot database not found: {snapshot_db}")
        (out / "shared_snapshot.json").write_text(json.dumps({
            "snapshot_root": str(snapshot_root),
            "snapshot_database": str(snapshot_db),
        }, indent=2, sort_keys=True))
    snapshot_config = fz.config_for_snapshot(config, snapshot_root)
    source_db = pl.resolve_path(config, "database")
    source_db_relative = source_db.relative_to(source_root).as_posix()
    source_manifest = source_manifest_before["files"]
    snapshot_manifest_before = fz.build_file_manifest(snapshot_config)
    snapshot_files = snapshot_manifest_before["files"]
    skipped_db_files = {source_db_relative, source_db_relative + "-wal", source_db_relative + "-journal"}
    file_copy_mismatches = sorted(
        path for path, value in source_manifest.items()
        if path not in skipped_db_files and snapshot_files.get(path) != value
    )
    unaccounted_snapshot_files = sorted(
        path for path in set(snapshot_files) - set(source_manifest)
        if path not in skipped_db_files
    )
    source_db_hash = _sqlite_content_hash(source_db)
    snapshot_db_hash = _sqlite_content_hash(snapshot_db)
    if source_db_hash != snapshot_db_hash:
        raise sqlite3.DatabaseError("SQLite snapshot content differs from the source database")
    if file_copy_mismatches or unaccounted_snapshot_files:
        raise OSError(
            "input snapshot copy mismatch: "
            f"changed_or_missing={file_copy_mismatches}, unaccounted={unaccounted_snapshot_files}"
        )

    db = pl.resolve_path(snapshot_config, "database")
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        timeline_rows = _cycle_timeline_rows(connection, cycle_key)
    finally:
        connection.close()
    timeline = cycle_timeline(timeline_rows)
    snapshot = pd.Timestamp(timeline["snapshot_as_of"])
    evaluation_time = pd.Timestamp.now(tz="UTC")
    header = {
        "run_id": run_id,
        "evaluation_time": evaluation_time.isoformat(),
        "signal_snapshot": snapshot.isoformat(),
        "cycle_key": cycle_key,
        "code": _code_hash(snapshot_config["_path"], snapshot_root, source_root),
        "data": _data_fingerprint(snapshot_config),
        "derivation_rule": se.DERIVATION_RULE_VERSION,
        "historical_validation": "NOT_VALIDATED",
        "scheduler_enabled": False,
        "orders_created": 0,
        "source_input_manifest": {
            "file_count": source_manifest_before["file_count"],
            "combined_sha256": source_manifest_before["combined_sha256"],
        },
        "snapshot_input_manifest": {
            "file_count": snapshot_manifest_before["file_count"],
            "combined_sha256": snapshot_manifest_before["combined_sha256"],
        },
        "snapshot_copy_verification": {
            "non_database_file_copy_matches": True,
            "sqlite_integrity_check": "ok",
            "source_sqlite_content_sha256": source_db_hash,
            "snapshot_sqlite_content_sha256": snapshot_db_hash,
        },
    }
    header["source_git_commit"] = header["code"]["git_commit"]
    (out / "input_manifest.json").write_text(
        json.dumps(snapshot_manifest_before, indent=1, sort_keys=True)
    )
    header["status"] = "RUNNING_OFFLINE"
    (out / "run_header.json").write_text(json.dumps(header, indent=1, sort_keys=True, default=str))

    guard = fz.network_guard()
    attempts: list[str] = []
    guard_entered = False
    historical_cache: dict = {}
    original_historical = pl._historical_sector
    runs: dict[str, dict] = {}
    report: dict = {}
    try:
        try:
            with fz.snapshot_context(snapshot_root):
                attempts = guard.__enter__()
                guard_entered = True

                def cached_historical(sector, cutoff):
                    key = (sector, pd.Timestamp(cutoff))
                    if key not in historical_cache:
                        historical_cache[key] = original_historical(sector, cutoff)
                    return historical_cache[key].copy()

                pl._historical_sector = cached_historical
                live_cache: dict = {}
                for label, threshold, enforce in (
                    ("t50_diagnostic", 0.5, False),
                    ("t50_enforced", 0.5, True),
                    ("t75_diagnostic", 0.75, False),
                    ("t100_diagnostic", 1.0, False),
                ):
                    cfg = copy.deepcopy(dict(snapshot_config))
                    cfg["model"] = {**cfg.get("model", {}), "min_feature_coverage": threshold}
                    provider = pl.ModelSignalProvider(cfg)
                    provider.prepare_cycle(cycle_key)
                    provider.enforce_eligibility = enforce
                    provider.allow_diagnostic_optimizer_failures = True
                    original_live = provider._live_rows

                    def cached_live(universe, snap, price_session, _o=original_live):
                        key = (snap, price_session)
                        if key not in live_cache:
                            live_cache[key] = _o(universe, snap, price_session)
                        live, prices_by_tic, skipped = live_cache[key]
                        return live.copy(), prices_by_tic, dict(skipped)

                    provider._live_rows = cached_live
                    result = provider.predict_as_of(snapshot.isoformat())
                    runs[label] = {
                        "threshold": threshold,
                        "enforce_eligibility": enforce,
                        "signal_as_of": result.signal_as_of,
                        "data_snapshot": result.data_snapshot,
                        "forecasts": result.forecasts,
                        "target_weights": result.target_weights,
                    }
                    (out / f"model_run_{label}.json").write_text(
                        json.dumps(runs[label], indent=1, sort_keys=True, default=str)
                    )
                    (out / f"sector_feature_sets_{label}.json").write_text(
                        json.dumps(result.data_snapshot["sector_features"], indent=1, sort_keys=True)
                    )
                    print("done", label, flush=True)
                base = runs["t50_diagnostic"]
                report = run_diagnosis(snapshot_config, cycle_key, base, snapshot=snapshot)
        except Exception as exc:
            header["status"] = "FAILED"
            header["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            header["network_attempts"] = list(attempts)
            (out / "run_header.json").write_text(json.dumps(header, indent=1, sort_keys=True, default=str))
            raise
    finally:
        pl._historical_sector = original_historical
        if guard_entered:
            guard.__exit__(None, None, None)

    snapshot_manifest_after = fz.build_file_manifest(snapshot_config)
    source_manifest_after = fz.build_file_manifest(config)
    header["network_attempts"] = list(attempts)
    header["snapshot_unchanged_after_run"] = fz.compare_manifests(
        snapshot_manifest_before, snapshot_manifest_after
    )
    header["source_unchanged_after_run"] = fz.compare_manifests(
        source_manifest_before, source_manifest_after
    )
    header["status"] = (
        "COMPLETE_OFFLINE"
        if not attempts
        and header["snapshot_unchanged_after_run"]["unchanged"]
        and header["source_unchanged_after_run"]["unchanged"]
        else "INCOMPLETE_OR_INPUTS_CHANGED"
    )
    header["snapshot_manifest_after"] = {
        "file_count": snapshot_manifest_after["file_count"],
        "combined_sha256": snapshot_manifest_after["combined_sha256"],
    }
    (out / "source_manifest_after.json").write_text(
        json.dumps(source_manifest_after, indent=1, sort_keys=True)
    )
    (out / "snapshot_manifest_after.json").write_text(
        json.dumps(snapshot_manifest_after, indent=1, sort_keys=True)
    )
    (out / "coverage.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    records = base["data_snapshot"]["forecast_records"]
    (out / "forecast_records.json").write_text(json.dumps(records, indent=1, sort_keys=True, default=str))
    (out / "readiness.json").write_text(json.dumps(report["readiness"], indent=1, sort_keys=True, default=str))

    manifest = json.loads(
        (pl.resolve_path(snapshot_config, "manifest_dir") / f"{cycle_key}.json").read_text()
    )
    members, _ = ua.members_as_of(pd.Timestamp(manifest["universe_as_of"]))
    securities = len(members)
    issuers = members["cik"].astype(int).nunique()
    expected = len(manifest["expected"])
    unique_expected_issuers = len({str(item["company_id"]) for item in manifest["expected"]})
    by_kind = Counter(r["forecast_kind"] for r in records)
    sector35 = [r for r in records if r["gsector"] == 35]
    enforced = runs["t50_enforced"]
    blocked = {r["security_id"] for r in records if not r["eligible_for_current_cycle"]}
    enforced_tickers = {t for w in enforced["target_weights"].values() for t in w}
    sensitivity = {}
    for label, run in runs.items():
        if not label.endswith("diagnostic"):
            continue
        recs = run["data_snapshot"]["forecast_records"]
        sensitivity[label] = {
            "threshold": run["threshold"],
            "forecasts": len(recs),
            "eligible": sum(1 for r in recs if r["eligible_for_current_cycle"]),
            "per_sector_forecasts": dict(sorted(Counter(r["gsector"] for r in recs).items())),
            "per_sector_eligible": dict(sorted(Counter(
                r["gsector"] for r in recs if r["eligible_for_current_cycle"]
            ).items())),
            "skipped": len(run["data_snapshot"]["skipped_tickers"]),
            "weights": {k: {"positions": len(v), "sum": round(sum(v.values()), 6)}
                        for k, v in run["target_weights"].items()},
            "weight_tickers_by_strategy": {k: sorted(v) for k, v in run["target_weights"].items()},
            "portfolio_optimisation": run["data_snapshot"].get("portfolio_optimisation"),
            "note": "Sensitivity of coverage/forecasts/weights only; no return-quality claim, "
                    "final test window not used.",
        }
    reconciliation = {
        **header,
        "securities_in_universe": securities,
        "distinct_issuer_ciks": int(issuers),
        "expected_reports_in_manifest": expected,
        "unique_expected_issuer_ids": unique_expected_issuers,
        "universe_manifest_reconciled": int(issuers) == unique_expected_issuers == expected,
        "multi_class_issuers": {
            str(c): sorted(g["tic"]) for c, g in
            members.assign(cik=members["cik"].astype(int)).groupby("cik") if len(g) > 1
        },
        "readiness": report["readiness"],
        "forecast_records": len(records),
        "forecast_kind_counts": dict(by_kind),
        "eligibility_reasons": dict(Counter(r["eligibility_reason"] for r in records)),
        "sector_35": [{
            k: r[k] for k in ("security_id", "issuer_cik", "used_period", "expected_period",
                              "accession", "forecast_kind", "eligible_for_current_cycle", "eligibility_reason")
        } for r in sector35],
        "diagnostic_only_in_enforced_weights": sorted(blocked & enforced_tickers),
        "enforced_run_weight_tickers": len(enforced_tickers),
        "enforced_run_forecast_kinds": dict(Counter(
            r["forecast_kind"] for r in enforced["data_snapshot"]["forecast_records"]
        )),
        "threshold_sensitivity": sensitivity,
        "historical_validation": "NOT_VALIDATED",
    }
    eligibility_compare = _forecast_comparison(runs["t50_diagnostic"], runs["t50_enforced"])
    (out / "forecast_eligibility_comparison.json").write_text(
        json.dumps(eligibility_compare, indent=1, sort_keys=True, default=str)
    )
    if compare_run_id:
        if Path(compare_run_id).name != compare_run_id:
            raise ValueError("comparison run id must be a single explicit directory name")
        old_records_path = out.parent / compare_run_id / "forecast_records.json"
        if not old_records_path.is_file():
            raise FileNotFoundError(f"comparison forecast records not found: {old_records_path}")
        old_records = json.loads(old_records_path.read_text())
        old_by_ticker = {item["security_id"]: item for item in old_records}
        current_by_ticker = {item["security_id"]: item for item in records}
        all_tickers = sorted(set(old_by_ticker) | set(current_by_ticker))
        changes = {
            ticker: {
                "previous": old_by_ticker.get(ticker),
                "current": current_by_ticker.get(ticker),
            }
            for ticker in all_tickers if old_by_ticker.get(ticker) != current_by_ticker.get(ticker)
        }
        reconciliation["baseline_run_comparison"] = {
            "baseline_run_id": compare_run_id,
            "baseline_forecast_count": len(old_records),
            "current_forecast_count": len(records),
            "shared_security_count": len(set(old_by_ticker) & set(current_by_ticker)),
            "only_in_baseline": sorted(set(old_by_ticker) - set(current_by_ticker)),
            "only_in_current": sorted(set(current_by_ticker) - set(old_by_ticker)),
            "changed_records": changes,
            "count_difference_explained": len(records) - len(old_records),
        }
        old_named_comparison = out.parent / compare_run_id / "forecast_comparison_35_vs_32.json"
        if old_named_comparison.is_file():
            reconciliation["baseline_run_comparison"]["legacy_named_artifact"] = {
                "filename": old_named_comparison.name,
                "contents": json.loads(old_named_comparison.read_text()),
                "conclusion": "the artifact's actual counts, not its filename, are the evidence",
            }
        (out / "forecast_baseline_comparison.json").write_text(
            json.dumps(reconciliation["baseline_run_comparison"], indent=1, sort_keys=True, default=str)
        )
    reconciliation["forecast_eligibility_comparison_summary"] = {
        key: eligibility_compare[key] for key in
        ("diagnostic_count", "enforced_count", "in_both", "only_in_diagnostic", "only_in_enforced")
    }
    (out / "model_rules.json").write_text(
        json.dumps(_model_rules(snapshot_config, report, runs), indent=1, sort_keys=True, default=str)
    )
    (out / "reconciliation.json").write_text(
        json.dumps(reconciliation, indent=1, sort_keys=True, default=str)
    )
    (out / "run_header.json").write_text(json.dumps(header, indent=1, sort_keys=True, default=str))
    import hashlib
    artifact_names = sorted(
        path.name for path in out.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "run_id": run_id,
        "evaluation_time": evaluation_time.isoformat(),
        "signal_snapshot": snapshot.isoformat(),
        "code_sha256": header["code"]["combined_sha256"],
        "artifacts": {
            name: {
                "size": (out / name).stat().st_size,
                "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
            }
            for name in artifact_names
        },
    }
    (out / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=1, sort_keys=True)
    )
    print(json.dumps({
        k: reconciliation[k] for k in (
            "run_id", "status", "securities_in_universe", "distinct_issuer_ciks",
            "expected_reports_in_manifest", "forecast_records", "forecast_kind_counts",
            "diagnostic_only_in_enforced_weights",
        )
    }, indent=1))
    if header["status"] != "COMPLETE_OFFLINE":
        raise RuntimeError(f"diagnostic suite did not complete reproducibly: {header['status']}")
    if not reconciliation["universe_manifest_reconciled"]:
        raise ValueError(
            "universe securities, issuer count and unique expected report issuers do not reconcile"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["diagnose", "readiness", "model-run", "fallback-trace", "suite", "acquire"])
    parser.add_argument("--cycle-key", required=True)
    parser.add_argument("--config")
    parser.add_argument("--model-run", help="JSON written by `model-run` to include forecast/optimisation stages")
    parser.add_argument("--out", help="output file (default under state/diagnostics/)")
    parser.add_argument("--compare-run-id", help="explicit prior run directory for record-level forecast comparison")
    parser.add_argument("--snapshot-from", help="reuse an exact prior run snapshot for independent offline replay")
    args = parser.parse_args(argv)
    se.require_user_agent()
    config = pl.load_config(args.config)
    out_dir = pl.resolve_path(config, "database").parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "acquire":
        manifest = json.loads((pl.resolve_path(config, "manifest_dir") / f"{args.cycle_key}.json").read_text())
        members, _ = ua.members_as_of(pd.Timestamp(manifest["universe_as_of"]))
        log = {"started_at": datetime.now(timezone.utc).isoformat(), "prices": {}}
        ua.fetch_changes_wikitext(max_age_seconds=0)
        ua.members_as_of(pd.Timestamp(manifest["universe_as_of"]))  # fills the sector-lookup cache
        for tic in sorted(set(members["tic"].astype(str)) | {"SPY"}):
            try:
                frame = pr.get_price_history(tic)
                log["prices"][tic] = (None if frame is None else
                                      {"rows": len(frame), "last": str(frame.index.max().date())})
            except Exception as exc:  # recorded, never hidden
                log["prices"][tic] = {"error": repr(exc)}
        log["finished_at"] = datetime.now(timezone.utc).isoformat()
        log["missing_or_failed"] = sorted(t for t, v in log["prices"].items() if not v or "error" in v)
        path = Path(args.out or out_dir / f"acquisition_{args.cycle_key}_{log['finished_at'][:19].replace(':', '')}.json")
        path.write_text(json.dumps(log, indent=1, sort_keys=True))
        print(path, "failed:", log["missing_or_failed"])
        return 0
    if args.command == "suite":
        print("suite written to", suite(
            config, args.cycle_key, args.config, args.compare_run_id, args.snapshot_from
        ))
        return 0
    if args.command == "readiness":
        manifest = json.loads((pl.resolve_path(config, "manifest_dir") / f"{args.cycle_key}.json").read_text())
        report = readiness_breakdown(config, _db_filings(pl.resolve_path(config, "database"), args.cycle_key),
                                     manifest)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "fallback-trace":
        report = fallback_trace(config, args.cycle_key, pd.Timestamp.now(tz="UTC"))
        path = Path(args.out or out_dir / f"ixbrl_fallback_trace_{args.cycle_key}.json")
        path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
        for t, c in report["cases"].items():
            print(t, c["without_fallback"]["stage"], c["without_fallback"].get("feature_coverage"), "->",
                  c["with_fallback"]["stage"], c["with_fallback"].get("feature_coverage"),
                  [e["field"] for e in c["extracted_by_fallback"]][:6])
        return 0
    if args.command == "model-run":
        provider = pl.ModelSignalProvider(config)
        provider.prepare_cycle(args.cycle_key)
        provider.enforce_eligibility = False  # diagnostic ranking; eligibility is recorded per forecast
        db = pl.resolve_path(config, "database")
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            timeline_rows = _cycle_timeline_rows(connection, args.cycle_key)
        finally:
            connection.close()
        snapshot = cycle_timeline(timeline_rows)["snapshot_as_of"]
        result = provider.predict_as_of(snapshot)
        path = Path(args.out or out_dir / f"model_run_{args.cycle_key}.json")
        path.write_text(json.dumps({"signal_as_of": result.signal_as_of, "data_snapshot": result.data_snapshot,
                                    "forecasts": result.forecasts, "target_weights": result.target_weights},
                                   indent=1, sort_keys=True, default=str))
        print(f"model run written to {path} (diagnostic only; nothing stored in the database)")
        return 0
    model_run = json.loads(Path(args.model_run).read_text()) if args.model_run else None
    report = run_diagnosis(config, args.cycle_key, model_run)
    path = Path(args.out or out_dir / f"coverage_{args.cycle_key}.json")
    path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    summary = {s: {k: v for k, v in d.items() if k not in ("sector_feature_set",)} for s, d in
               report["sector_coverage"].items()}
    print(json.dumps({"report": str(path), "readiness": report["readiness"], "sectors": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
