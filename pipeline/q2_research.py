"""Build and run an isolated, non-executable 2026Q2 research snapshot."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sqlite3
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "pipeline"))

import freeze as fz  # noqa: E402
import model_coverage as mc  # noqa: E402
import paper_live as pl  # noqa: E402
import paper_store as ps  # noqa: E402
import sec_edgar as se  # noqa: E402

CYCLE_KEY = "2026Q2"
QUARTER_END = pd.Timestamp("2026-06-30")
UNIVERSE_AS_OF = "2026-06-30"
INFO_CUTOFF = pd.Timestamp("2026-10-14T16:00:00Z")
HISTORICAL_MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
Q2_ROOT = ROOT / "state" / "diagnostics" / "q2_2026Q2"
CONFIG_PATH = Q2_ROOT / "real_paper_q2.json"
DATABASE_PATH = ROOT / "state" / "real_paper_q2_2026Q2.sqlite3"
MANIFEST_DIR = ROOT / "state" / "expected_filings_q2_2026Q2"
GAP_JSON = Q2_ROOT / "q2_gap_report.json"
GAP_CSV = Q2_ROOT / "q2_gap_report.csv"
GAP_FIELDS = [
    "ticker", "cik", "expected_period", "expected_form", "used_period", "accession",
    "published_at", "source", "missing_metrics", "missing_prices", "feature_coverage",
    "status", "exclusion_reason", "local_checked_at", "source_document_cached",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _normalize_ticker(value: object) -> str:
    return str(value).strip().upper().replace(".", "-")


def _timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _q2_report(submissions: dict | None) -> dict | None:
    """Prefer the actual report period within the versioned Q2 window (including 52/53-week ends)."""
    filings = pl._filings(submissions)
    low, high = pl.period_window(QUARTER_END, pl.PERIOD_RULE_V2)
    in_window = [row for row in filings if low < pd.Timestamp(row["reportDate"]) <= high]
    periods = {(str(row["reportDate"]), str(row["form"])) for row in in_window}
    if len(periods) == 1:
        report_date, form = next(iter(periods))
        return {"period_end": report_date, "form": form, "derivation": "observed_report_in_q2_window"}
    if len(periods) > 1:
        return None
    if not filings:
        return None
    recent = sorted(filings, key=lambda row: row["reportDate"], reverse=True)[:8]
    grid_months = {
        pl._nominal_month_end(row["reportDate"]).month % 3 for row in recent
    }
    if len(grid_months) != 1:
        anniversary = pl._anniversary_report(filings, QUARTER_END, pl.PERIOD_RULE_V2)
        return ({**anniversary, "derivation": "period_rule_v2_anniversary"}
                if anniversary else None)
    year_end_month = next(
        (pl._nominal_month_end(row["reportDate"]).month for row in recent if row["form"] == "10-K"),
        None,
    )
    if year_end_month is None:
        return None
    reference = pl._nominal_month_end(recent[0]["reportDate"])
    for offset in range(-24, 25):
        candidate = pl._month_end(reference.year, reference.month) + pd.DateOffset(months=offset)
        candidate = candidate + pd.offsets.MonthEnd(0)
        if candidate.month % 3 not in grid_months or not low < candidate <= high:
            continue
        actual = next(
            (
                pd.Timestamp(row["reportDate"]) for row in filings
                if abs((pd.Timestamp(row["reportDate"]) - candidate).days)
                <= pl.PERIOD_RULE_TOLERANCE_DAYS
            ),
            None,
        )
        period_end = actual if actual is not None else candidate
        form = "10-K" if candidate.month == year_end_month else "10-Q"
        return {
            "period_end": period_end.date().isoformat(),
            "form": form,
            "derivation": "period_rule_v2_grid",
        }
    return None


def _latest_q2_filing(submissions: dict | None, expected: dict) -> dict | None:
    recent = (((submissions or {}).get("filings") or {}).get("recent") or {})
    names = ("form", "accessionNumber", "reportDate", "filingDate",
             "acceptanceDateTime", "primaryDocument")
    cutoff = INFO_CUTOFF
    found = []
    for values in zip(*(recent.get(name, []) for name in names)):
        form, accession, report_date, filed, accepted, primary = values
        if form != expected["form"] or str(report_date) != expected["period_end"] or not accession:
            continue
        accepted_at = _timestamp(str(accepted) if accepted else None)
        if accepted_at is None or accepted_at > cutoff:
            continue
        found.append({
            "form": form, "accession": str(accession), "report_date": str(report_date),
            "filing_date": str(filed or ""), "published_at": accepted_at.isoformat(),
            "primary_document": str(primary or ""),
        })
    return max(found, key=lambda row: row["published_at"]) if found else None


def _read_source_config() -> dict:
    base = json.loads((ROOT / "config" / "real_paper.json").read_text(encoding="utf-8"))
    config = copy.deepcopy(base)
    config["database"] = DATABASE_PATH.relative_to(ROOT).as_posix()
    config["manifest_dir"] = MANIFEST_DIR.relative_to(ROOT).as_posix()
    config["universe"]["source"] = (
        "dated local S&P 500 membership snapshot; source URL and SHA-256 recorded in 2026Q2 manifest"
    )
    config["universe"]["as_of_policy"] = "explicit fixed universe_as_of=2026-06-30 for cycle 2026Q2"
    config["universe"]["period_overrides"] = {}
    config["universe"]["cik_successors"] = {}
    config["universe"]["exclusions"] = {}
    old_readiness = config.get("readiness_rule", {})
    config["readiness_rule"] = {
        key: old_readiness[key]
        for key in ("version", "required_facts_by_gsector", "required_facts_by_cik")
        if key in old_readiness
    }
    config["q2_research"] = {
        "cycle_key": CYCLE_KEY,
        "universe_as_of": UNIVERSE_AS_OF,
        "information_cutoff": INFO_CUTOFF.isoformat(),
        "classification": "CURRENT_Q2_PERIOD_DIAGNOSTIC_NOT_HISTORICAL_SIGNAL",
        "scheduler_enabled": False,
        "orders_created": 0,
    }
    return config


def _membership_and_expected(base_config: dict, checked_at: str) -> dict:
    source_membership = ROOT / "cache" / "sp500_historical_snapshots.csv"
    snapshots = pd.read_csv(source_membership, dtype={"date": str, "tickers": str})
    selected = snapshots.loc[snapshots["date"].eq(UNIVERSE_AS_OF), "tickers"]
    if len(selected) != 1 or not selected.iloc[0]:
        raise ValueError(f"the dated membership snapshot for {UNIVERSE_AS_OF} is absent or ambiguous")
    tickers = sorted({_normalize_ticker(item) for item in selected.iloc[0].split(",") if item.strip()})

    universe = pd.read_csv(ROOT / "cache" / "sp500_universe.csv")
    universe["tic"] = universe["tic"].map(_normalize_ticker)
    universe_rows = {
        row["tic"]: row for row in universe.drop_duplicates("tic", keep="last").to_dict("records")
    }
    sec_tickers = json.loads((ROOT / "cache" / "sec_company_tickers.json").read_text(encoding="utf-8"))
    sec_by_ticker = {_normalize_ticker(row.get("ticker")): row for row in sec_tickers.values()}
    live_path = ROOT / "Data" / "live" / "fundamental_hybrid_table.csv"
    live = pd.read_csv(live_path, usecols=["tic", "gsector", "conm"], low_memory=False)
    live["tic"] = live["tic"].map(_normalize_ticker)
    live_sector = live.dropna(subset=["gsector"]).groupby("tic")["gsector"].agg(
        lambda values: int(values.mode().iloc[0])
    ).to_dict()
    live_name = live.dropna(subset=["conm"]).drop_duplicates("tic", keep="last").set_index("tic")["conm"].to_dict()
    historical_path = ROOT / "Data" / "fundamental_final_table.xlsx"
    historical = pd.read_excel(historical_path, usecols=["tic", "gsector", "conm"])
    historical["tic"] = historical["tic"].map(_normalize_ticker)
    historical_sector = historical.dropna(subset=["gsector"]).groupby("tic")["gsector"].agg(
        lambda values: int(values.mode().iloc[0])
    ).to_dict()
    historical_name = historical.dropna(subset=["conm"]).drop_duplicates("tic", keep="last").set_index("tic")["conm"].to_dict()

    successor_history = base_config.get("universe", {}).get("cik_successors", {})
    securities = []
    for ticker in tickers:
        saved = universe_rows.get(ticker, {})
        sec = sec_by_ticker.get(ticker, {})
        cik = int(saved["cik"]) if pd.notna(saved.get("cik")) else (
            int(sec["cik_str"]) if sec.get("cik_str") is not None else None
        )
        sector_value = saved.get("gsector")
        sector = int(sector_value) if pd.notna(sector_value) else (
            live_sector.get(ticker, historical_sector.get(ticker))
        )
        candidates = (saved.get("conm"), live_name.get(ticker), historical_name.get(ticker), sec.get("title"))
        name = next((str(value) for value in candidates if value is not None and pd.notna(value)), "")
        mapping_source = (
            "cache/sp500_universe.csv"
            if saved and cik is not None and pd.notna(sector_value)
            else "SEC ticker cache + dated local sector panel"
            if sec and sector is not None
            else "dated local Compustat sector panel; SEC CIK unresolved"
            if sector is not None
            else "dated membership only; CIK/GICS mapping unresolved"
        )
        succession = None
        if ticker == "XOM" and cik is not None:
            for successor, item in successor_history.items():
                if int(successor) == cik and pd.Timestamp(item["effective"]) > QUARTER_END:
                    predecessors = [int(value) for value in item.get("predecessors", [])]
                    if len(predecessors) == 1:
                        cik = predecessors[0]
                        succession = {
                            "current_cik": int(successor), "q2_cik": cik,
                            "effective": item["effective"], "source": item.get("source"),
                            "rule": "Q2 ended before succession effective date; predecessor retained for Q2",
                        }
        securities.append({
            "ticker": ticker, "cik": cik, "gsector": int(sector) if sector is not None else None,
            "company_name": name, "mapping_source": mapping_source,
            "is_member_as_of": True, "succession_resolution": succession,
        })

    by_cik: dict[int, list[dict]] = {}
    unresolved: list[dict] = []
    for security in securities:
        if security["cik"] is None:
            unresolved.append({
                "ticker": security["ticker"], "reason": "UNRESOLVED_CIK",
                "detail": "No historical CIK mapping exists in the local SEC ticker/submission caches.",
            })
            continue
        by_cik.setdefault(int(security["cik"]), []).append(security)
        if security["gsector"] is None:
            unresolved.append({
                "ticker": security["ticker"], "cik": security["cik"], "reason": "UNRESOLVED_GICS_SECTOR",
                "detail": "The dated membership list is known, but no sourced local sector mapping is present.",
            })

    expected, issuer_rows = [], []
    submissions_by_cik: dict[int, dict | None] = {}
    for cik, issuer_securities in sorted(by_cik.items()):
        cache_path = ROOT / "cache" / "secsubmissions" / f"{cik}.json"
        submissions = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else None
        submissions_by_cik[cik] = submissions
        ticker_names = sorted({row["ticker"] for row in issuer_securities})
        report = _q2_report(submissions)
        if report is None:
            unresolved.append({
                "cik": cik, "tickers": ticker_names, "reason": "UNRESOLVED_Q2_PERIOD",
                "detail": "No local SEC submissions establish a unique report period inside the Q2 v2 window.",
            })
            continue
        filing = _latest_q2_filing(submissions, report)
        key = f"{cik}-{report['period_end']}-{report['form']}"
        entry = {
            "filing_key": key, "company_id": str(cik), "security_id": ticker_names[0],
            "expected_period": report["period_end"], "filing_type": report["form"],
            "period_derivation": report["derivation"], "issuer_tickers": ticker_names,
            "report_status": "AVAILABLE" if filing else "MISSING_Q2_REPORT",
            "accession": filing["accession"] if filing else None,
            "filed_at": filing["published_at"] if filing else None,
            "primary_document": filing["primary_document"] if filing else None,
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{filing['accession'].replace('-', '')}/{filing['primary_document']}"
                if filing and filing["primary_document"] else None
            ),
            "source": "SEC submissions cache" if filing else "No matching SEC submission in local cache",
            "local_checked_at": checked_at,
        }
        if filing:
            accession_path = filing["accession"].replace("-", "")
            filing_path = ROOT / "cache" / "filings" / str(cik) / accession_path / filing["primary_document"]
            entry["source_document_cached"] = filing_path.is_file()
            entry["source_document_sha256"] = _sha256(filing_path) if filing_path.is_file() else None
        else:
            entry["source_document_cached"] = False
            entry["source_document_sha256"] = None
        expected.append(entry)
        issuer_rows.append({
            "filing_key": key, "company_id": str(cik), "security_id": ticker_names[0],
            "expected_period": report["period_end"], "filing_type": report["form"],
            "source": entry["source"],
        })

    evidence = {
        "source_url": HISTORICAL_MEMBERSHIP_URL,
        "source_file": str(source_membership.relative_to(ROOT)),
        "source_sha256": _sha256(source_membership),
        "snapshot_date": UNIVERSE_AS_OF,
        "snapshot_count": len(tickers),
        "normalization": "ticker symbols normalized from dot to dash for class-share files",
    }
    q2_config = _read_source_config()
    issuer_keys = {str(cik) for cik in by_cik}
    issuer_keys.update(
        f"unresolved-ticker:{row['ticker']}" for row in unresolved
        if row["reason"] == "UNRESOLVED_CIK"
    )
    manifest = {
        "cycle_key": CYCLE_KEY,
        "quarter_end": QUARTER_END.date().isoformat(),
        "universe_as_of": UNIVERSE_AS_OF,
        "information_cutoff": INFO_CUTOFF.isoformat(),
        "information_cutoff_classification": "current Q2-period analysis; not the historical Q2 information set",
        "availability_assumption": (
            "SEC acceptance timestamps establish original public filing availability. Local Company Facts "
            "observation/readiness is recorded at the actual Q2 research check time, not backdated."
        ),
        "universe_source": evidence,
        "period_rule": pl.PERIOD_RULE_V2,
        "period_rule_note": (
            "Actual SEC report periods inside the versioned Q2 window are preferred, including 52/53-week "
            "period ends; otherwise the existing versioned grid rule is used. Q3 overrides are not copied."
        ),
        "q3_configuration_not_applied": [
            "period overrides", "post-Q2 CIK successors as Q2 identity", "corporate-action resolutions",
            "buy blocks",
        ],
        "expected": expected,
        "unresolved": unresolved,
        "securities": securities,
        "generated_at": checked_at,
        "expected_issuer_count": len(issuer_keys),
        "mapped_issuer_count": len(by_cik),
        "unresolved_issuer_mapping_count": sum(
            row["reason"] == "UNRESOLVED_CIK" for row in unresolved
        ),
        "security_count": len(securities),
        "unresolved_security_count": len({r.get("ticker") for r in unresolved if r.get("ticker")}),
    }
    feature_sets = {
        str(sector): pl.sector_feature_set(int(sector))
        for sector in sorted({row["gsector"] for row in securities if row["gsector"] is not None})
    }
    manifest["sector_feature_sets"] = feature_sets
    return {"config": q2_config, "manifest": manifest, "issuer_rows": issuer_rows,
            "submissions_by_cik": submissions_by_cik, "universe_evidence": evidence}


def prepare_workspace() -> tuple[dict, dict]:
    if CONFIG_PATH.exists() or DATABASE_PATH.exists() or (MANIFEST_DIR / f"{CYCLE_KEY}.json").exists():
        raise FileExistsError("Q2 workspace already exists; refusing to overwrite its immutable inputs.")
    checked_at = datetime.now(timezone.utc).isoformat()
    base = json.loads((ROOT / "config" / "real_paper.json").read_text(encoding="utf-8"))
    built = _membership_and_expected(base, checked_at)
    Q2_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    built["config"]["_path"] = str(CONFIG_PATH)
    _write_json(CONFIG_PATH, {key: value for key, value in built["config"].items() if key != "_path"})
    manifest_path = pl.write_manifest(built["config"], CYCLE_KEY, built["manifest"])
    store = ps.PaperStore.create(DATABASE_PATH, environment="real")
    store.create_portfolio(
        "q2_2026_research", "2026Q2 research only (non-executable)", "equal_weight_v1",
        float(built["config"]["initial_capital"]),
    )
    cycle_id = store.create_cycle(
        "q2_2026_research", CYCLE_KEY, "equal_weight_v1",
        rebalance_policy="fixed_quarterly", started_at=checked_at,
    )
    if built["issuer_rows"]:
        store.register_expected_filings(
            cycle_id,
            [str(row["filing_key"]) for row in built["issuer_rows"]],
            filing_metadata={
                str(row["filing_key"]): {
                    key: value for key, value in row.items() if key != "filing_key"
                }
                for row in built["issuer_rows"]
            },
            updated_at=checked_at,
        )
    store.set_cycle_model_status(
        cycle_id, "BLOCKED",
        reason="Research-only Q2 snapshot; unresolved expected issuers and no production authorization.",
    )
    with sqlite3.connect(DATABASE_PATH) as connection:
        schema = int(connection.execute("PRAGMA user_version").fetchone()[0])
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    provenance = {
        "cycle_key": CYCLE_KEY,
        "created_at": checked_at,
        "base_config_sha256": _sha256(ROOT / "config" / "real_paper.json"),
        "q2_config_sha256": _sha256(CONFIG_PATH),
        "q2_manifest_sha256": _sha256(manifest_path),
        "database_path": str(DATABASE_PATH.relative_to(ROOT)),
        "database_sha256": mc._sqlite_content_hash(DATABASE_PATH),
        "database_schema_version": schema,
        "database_integrity_check": integrity,
        "database_environment": "real, isolated research copy",
        "production_database_modified": False,
        "scheduler_enabled": False,
        "orders_created": 0,
    }
    _write_json(Q2_ROOT / "workspace_provenance.json", provenance)
    _write_json(Q2_ROOT / "q2_manifest.json", built["manifest"])
    return built["config"], built["manifest"]


def _strict_local_readers(expected_by_cik: dict[int, dict]):
    def submissions(cik: int, max_age_seconds: float = 0):
        path = se.SUBMISSIONS_DIR / f"{int(cik)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def facts(cik: int, max_age_seconds: float = 0):
        path = se.CACHE_DIR / f"{int(cik)}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = expected_by_cik.get(int(cik))
        period = expected.get("expected_period") if expected else None
        accession = expected.get("accession") if expected else None
        us_gaap = (((payload.get("facts") or {}).get("us-gaap")) or {})
        for concept in us_gaap.values():
            for unit, entries in (concept.get("units") or {}).items():
                concept["units"][unit] = [
                    entry for entry in entries
                    if str(entry.get("end")) != str(period)
                    or (accession is not None and str(entry.get("accn")) == str(accession))
                ]
        return payload

    return submissions, facts


def _q2_members_function(manifest: dict):
    rows = [row for row in manifest["securities"] if row["cik"] is not None and row["gsector"] is not None]
    members = pd.DataFrame(rows).rename(columns={"ticker": "tic", "company_name": "conm"})
    members["cik"] = members["cik"].astype(int)
    members["gsector"] = members["gsector"].astype(int)
    members["is_current"] = True
    evidence = {
        "as_of": UNIVERSE_AS_OF,
        "source_sha256": manifest["universe_source"]["source_sha256"],
        "member_count": len(manifest["securities"]),
        "model_mappable_security_count": len(members),
        "source": "Q2 manifest backed by dated historical membership snapshot",
    }

    def exact_members(as_of):
        day = pd.Timestamp(as_of).normalize()
        if day != QUARTER_END:
            raise ValueError(f"Q2 frozen universe is not valid for {day.date()}")
        return members.copy(), dict(evidence)

    return exact_members, members


def _sector_for_model_tickers(tickers: list[str], sector_map: dict[str, int]) -> int | None:
    return next((sector_map[str(ticker)] for ticker in tickers if str(ticker) in sector_map), None)


def _capture_training(ml_model):
    original = {
        "run": ml_model.run_4model,
        "train": ml_model.prepare_rolling_train,
        "test": ml_model.prepare_rolling_test,
    }
    context: dict[str, Any] = {"sector": None, "frame": None, "enabled": False}
    captured: dict[int, dict[str, pd.DataFrame]] = {}

    def train(*args, **kwargs):
        result = original["train"](*args, **kwargs)
        if context["enabled"] and context["sector"] is not None:
            captured.setdefault(context["sector"], {})["training"] = context["frame"].loc[result[0].index].copy()
        return result

    def test(*args, **kwargs):
        result = original["test"](*args, **kwargs)
        if context["enabled"] and context["sector"] is not None:
            captured.setdefault(context["sector"], {})["validation"] = context["frame"].loc[result[0].index].copy()
        return result

    def run(*args, **kwargs):
        frame = args[0]
        tickers = args[5]
        sector_map = context["sector_map"]
        sector = _sector_for_model_tickers(tickers, sector_map)
        context["sector"], context["frame"] = sector, frame
        return original["run"](*args, **kwargs)

    ml_model.prepare_rolling_train = train
    ml_model.prepare_rolling_test = test
    ml_model.run_4model = run
    return captured, context, original


def _write_training_exports(run_dir: Path, captures: dict, feature_sets: dict, snapshot: str) -> dict:
    output = {}
    directory = run_dir / "training"
    directory.mkdir(parents=True, exist_ok=True)
    for sector, feature_names in sorted(feature_sets.items(), key=lambda item: int(item[0])):
        samples = captures.get(int(sector), {})
        output[sector] = {}
        for kind in ("training", "validation"):
            frame = samples.get(kind, pd.DataFrame())
            export = frame.copy()
            if not export.empty:
                export["row_source"] = np.where(
                    pd.to_datetime(export["tradedate"]) < pl.HISTORICAL_CUTOFF,
                    "Compustat historical sector file", "SEC-derived live panel",
                )
                export["label_known_by_snapshot"] = (
                    pd.to_datetime(export["label_known_at"], utc=True, errors="coerce")
                    <= pd.Timestamp(snapshot)
                )
                export.to_csv(directory / f"sector{sector}_{kind}.csv", index=False)
            output[sector][kind] = {
                "rows": int(len(frame)),
                "quarters": int(frame["tradedate"].nunique()) if not frame.empty else 0,
                "date_start": str(frame["tradedate"].min()) if not frame.empty else None,
                "date_end": str(frame["tradedate"].max()) if not frame.empty else None,
                "source_rows": dict(Counter(
                    "Compustat historical sector file"
                    if pd.Timestamp(day) < pl.HISTORICAL_CUTOFF else "SEC-derived live panel"
                    for day in frame["tradedate"]
                )) if not frame.empty else {},
                "known_labels": int(frame["y_return"].notna().sum()) if not frame.empty else 0,
                "max_label_known_at": str(frame["label_known_at"].max()) if not frame.empty else None,
                "missing_feature_values": {
                    feature: int(frame[feature].isna().sum())
                    for feature in feature_names if feature in frame
                } if not frame.empty else {},
            }
    return {
        "snapshot_as_of": snapshot,
        "historical_cutoff": pl.HISTORICAL_CUTOFF.date().isoformat(),
        "features_by_sector": feature_sets,
        "sector_rows": output,
        "minimum_rules": {
            "first_trade_date_index": 21,
            "testing_windows": 4,
            "max_rolling_window_index": 45,
            "nonempty_training_and_validation_required": True,
            "quarter_aware_cv": "3 expanding folds; at least 5 unique quarters before fold purging",
            "label_rule": "y_return present and label_known_at <= signal_time; future target rows are not training labels",
        },
        "historical_validation": "NOT_VALIDATED",
    }


def _gap_rows(manifest: dict, live: pd.DataFrame, prices: dict, fundamentals: dict,
              skipped: dict, price_session: pd.Timestamp, checked_at: str) -> list[dict]:
    expected_by_cik = {int(row["company_id"]): row for row in manifest["expected"]}
    rows = []
    features_by_sector = {int(k): list(v) for k, v in manifest["sector_feature_sets"].items()}
    live = live.copy()
    live["tic"] = live["tic"].map(_normalize_ticker)
    for security in manifest["securities"]:
        ticker = security["ticker"]
        cik = int(security["cik"]) if security["cik"] is not None else None
        sector = int(security["gsector"]) if security["gsector"] is not None else None
        expected = expected_by_cik.get(cik) if cik is not None else None
        expected_period = expected.get("expected_period") if expected else None
        report_status = expected.get("report_status") if expected else None
        reason = None
        if cik is None:
            status, reason = "UNRESOLVED_CIK", "No locally evidenced CIK for the Q2 member."
        elif sector is None:
            status, reason = "UNRESOLVED_GICS_SECTOR", "No locally evidenced historical GICS sector."
        elif expected is None:
            unresolved = next((item for item in manifest["unresolved"]
                               if item.get("cik") == cik or item.get("ticker") == ticker
                               or ticker in item.get("tickers", [])), None)
            status = unresolved["reason"] if unresolved else "UNRESOLVED_Q2_PERIOD"
            reason = unresolved["detail"] if unresolved else "No Q2 report period was derivable."
        elif report_status != "AVAILABLE":
            status, reason = "MISSING_Q2_REPORT", "No matching Q2 accession is available by the information cutoff."
        else:
            status, reason = "WAITING_FACTS", "Q2 filing facts, prices, or model-feature coverage are incomplete."

        raw = fundamentals.get(cik) if cik is not None else None
        fact_row = None
        if raw is not None and expected_period is not None and pd.Timestamp(expected_period) in raw.index:
            fact_row = raw.loc[pd.Timestamp(expected_period)]
            if isinstance(fact_row, pd.DataFrame):
                fact_row = fact_row.iloc[-1]
        required = pl.required_facts_for(
            json.loads(CONFIG_PATH.read_text(encoding="utf-8")), sector, cik
        ) if sector is not None and cik is not None else []
        missing_metrics = [name for name in required if fact_row is None or name not in fact_row
                           or pd.isna(fact_row[name])]
        expected_accession = expected.get("accession") if expected else None
        accession_fields = {
            str(fact_row.get(f"{name}_accn", "")) for name in required
            if fact_row is not None and fact_row.get(f"{name}_accn") is not None
        }
        if expected_accession and accession_fields and expected_accession not in accession_fields:
            missing_metrics.append("Q2_REQUIRED_FACTS_NOT_FROM_EXPECTED_ACCESSION")
        match = live[
            live["tic"].eq(ticker)
            & (pd.to_datetime(live["datadate"], errors="coerce").dt.normalize()
               == pd.Timestamp(expected_period).normalize() if expected_period else False)
        ] if expected_period else pd.DataFrame()
        live_row = match.iloc[-1] if not match.empty else None
        feature_coverage = None
        if live_row is not None and sector is not None:
            selected_features = features_by_sector.get(sector, [])
            values = pd.to_numeric(live_row[selected_features], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            feature_coverage = float(values.notna().mean()) if selected_features else None
            missing_metrics.extend(name for name in selected_features if pd.isna(values[name]))
        history = prices.get(ticker)
        missing_prices = []
        if history is None or history.empty:
            missing_prices.append("no cached price history")
        else:
            history.index = pd.to_datetime(history.index)
            if price_session not in history.index.normalize():
                missing_prices.append(f"no price on model session {price_session.date()}")
            else:
                day = history.loc[history.index.normalize() == price_session.normalize()].iloc[-1]
                missing_prices.extend(name for name in ("close", "adj_close")
                                       if name not in day or pd.isna(day[name]))
        if status == "WAITING_FACTS" and not missing_metrics and not missing_prices and live_row is not None:
            if feature_coverage is not None and feature_coverage >= 0.5:
                status, reason = "READY_FOR_NON_EXECUTABLE_DIAGNOSTIC", None
            else:
                status, reason = "LOW_FEATURE_COVERAGE", "Q2 row is below the unchanged 50% feature-coverage rule."
        if missing_metrics or missing_prices:
            status = status if status not in {"WAITING_FACTS"} else "WAITING_FACTS"
        rows.append({
            "ticker": ticker, "cik": cik, "expected_period": expected_period,
            "expected_form": expected.get("filing_type") if expected else None,
            "used_period": str(live_row["datadate"]) if live_row is not None else None,
            "accession": expected_accession,
            "published_at": expected.get("filed_at") if expected else None,
            "source": expected.get("source") if expected else security["mapping_source"],
            "missing_metrics": sorted(set(missing_metrics)),
            "missing_prices": missing_prices,
            "feature_coverage": feature_coverage,
            "status": status, "exclusion_reason": reason,
            "local_checked_at": checked_at,
            "source_document_cached": expected.get("source_document_cached") if expected else False,
        })
    return rows


def _write_gap_report(rows: list[dict], run_dir: Path | None = None) -> None:
    _write_json(GAP_JSON, {
        "cycle_key": CYCLE_KEY,
        "information_cutoff": INFO_CUTOFF.isoformat(),
        "classification": "current Q2-report-period diagnostic; not historical Q2 signal or backtest",
        "rows": rows,
    })
    with GAP_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=GAP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if run_dir is not None:
        _write_json(run_dir / "q2_gap_report.json", json.loads(GAP_JSON.read_text(encoding="utf-8")))
        (run_dir / "q2_gap_report.csv").write_bytes(GAP_CSV.read_bytes())


def run_diagnostic(config: dict, manifest: dict) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-q2-" + uuid.uuid4().hex[:6]
    run_dir = ROOT / "state" / "diagnostics" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot_root = run_dir / "snapshot"
    source_manifest_before = fz.build_file_manifest(config)
    _write_json(run_dir / "source_manifest_before.json", source_manifest_before)
    snapshot_db = fz.create_input_snapshot(config, snapshot_root)
    snapshot_config = fz.config_for_snapshot(config, snapshot_root)
    snapshot_manifest_before = fz.build_file_manifest(snapshot_config)
    database_relative = DATABASE_PATH.relative_to(ROOT).as_posix()
    skipped_database_files = {
        database_relative, database_relative + "-wal", database_relative + "-journal",
    }
    source_files = {
        path: value for path, value in source_manifest_before["files"].items()
        if path not in skipped_database_files
    }
    frozen_files = {
        path: value for path, value in snapshot_manifest_before["files"].items()
        if path not in skipped_database_files
    }
    if source_files != frozen_files:
        raise OSError("Q2 input snapshot differs from its source manifest.")
    if mc._sqlite_content_hash(DATABASE_PATH) != mc._sqlite_content_hash(snapshot_db):
        raise sqlite3.DatabaseError("Q2 work database snapshot content mismatch.")
    header = {
        "run_id": run_id, "status": "RUNNING_OFFLINE", "cycle_key": CYCLE_KEY,
        "evaluation_time": datetime.now(timezone.utc).isoformat(),
        "signal_snapshot": INFO_CUTOFF.isoformat(),
        "information_cutoff_classification": "current Q2-period analysis, not a historical Q2 information set",
        "historical_validation": "NOT_VALIDATED", "scheduler_enabled": False,
        "orders_created": 0, "executable": False,
        "code": mc._code_hash(snapshot_config["_path"], snapshot_root, ROOT),
        "input_manifest": {
            "file_count": snapshot_manifest_before["file_count"],
            "combined_sha256": snapshot_manifest_before["combined_sha256"],
        },
        "source_git_commit": None,
        "snapshot_copy_verification": {"manifest_matches": True, "sqlite_integrity_check": "ok"},
    }
    _write_json(run_dir / "run_header.json", header)
    _write_json(run_dir / "input_manifest.json", snapshot_manifest_before)
    expected_by_cik = {int(row["company_id"]): row for row in manifest["expected"]}
    exact_members, model_members = _q2_members_function(manifest)
    result_runs: dict[str, dict] = {}
    attempts: list[str] = []
    gap_rows: list[dict] = []
    training_summary = None
    snapshot_after = None
    original_members = pl.ua.members_as_of
    original_fact_reader = se._fetch_company_facts
    original_submission_reader = se._fetch_submissions
    original_get_fundamentals = None
    ml_model = None
    original_model = None
    fund_cache: dict[int, pd.DataFrame | None] = {}
    try:
        with fz.snapshot_context(snapshot_root):
            with fz.network_guard() as attempts:
                local_submissions, local_facts = _strict_local_readers(expected_by_cik)
                se._fetch_submissions = local_submissions
                se._fetch_company_facts = local_facts
                pl.ua.members_as_of = exact_members
                se.register_ixbrl_fallback(snapshot_config.get("ixbrl_fallback"))
                original_get_fundamentals = se.get_quarterly_fundamentals
                def cached_fundamentals(cik: int, as_of=None):
                    cik = int(cik)
                    if cik not in fund_cache:
                        fund_cache[cik] = original_get_fundamentals(cik, as_of)
                    value = fund_cache[cik]
                    return value.copy() if value is not None else None
                se.get_quarterly_fundamentals = cached_fundamentals

                import ml_model as loaded_ml_model
                ml_model = loaded_ml_model
                captures, capture_context, original_model = _capture_training(ml_model)
                capture_context["sector_map"] = {
                    str(row["ticker"]): int(row["gsector"])
                    for row in manifest["securities"] if row["gsector"] is not None
                }
                live_cache: dict[str, Any] = {}
                first_gap_written = False
                for label, threshold, enforce in (
                    ("t50_diagnostic", 0.5, False),
                    ("t50_enforced", 0.5, True),
                    ("t75_diagnostic", 0.75, False),
                    ("t100_diagnostic", 1.0, False),
                    ("t50_diagnostic_replay", 0.5, False),
                ):
                    model_config = copy.deepcopy(snapshot_config)
                    model_config["model"] = {
                        **model_config.get("model", {}), "min_feature_coverage": threshold,
                    }
                    provider = pl.ModelSignalProvider(model_config)
                    provider.prepare_cycle(CYCLE_KEY)
                    provider.enforce_eligibility = enforce
                    provider.allow_diagnostic_optimizer_failures = True
                    original_live_rows = provider._live_rows

                    def cached_live(universe, snapshot, price_session, _original=original_live_rows):
                        key = f"{pd.Timestamp(snapshot).isoformat()}|{pd.Timestamp(price_session).isoformat()}"
                        if key not in live_cache:
                            live_cache[key] = _original(universe, snapshot, price_session)
                        live, price_map, skipped = live_cache[key]
                        return live.copy(), price_map, dict(skipped)

                    provider._live_rows = cached_live
                    capture_context["enabled"] = label == "t50_diagnostic"
                    result = provider.predict_as_of(INFO_CUTOFF.isoformat())
                    result_runs[label] = {
                        "threshold": threshold, "enforce_eligibility": enforce,
                        "signal_as_of": result.signal_as_of, "data_snapshot": result.data_snapshot,
                        "forecasts": result.forecasts, "target_weights": result.target_weights,
                        "executable": False,
                    }
                    if not first_gap_written:
                        live, prices_by_tic, skipped = next(iter(live_cache.values()))
                        price_session = pd.Timestamp(result.data_snapshot["price_session"])
                        gap_rows = _gap_rows(
                            manifest, live, prices_by_tic, fund_cache, skipped, price_session,
                            datetime.now(timezone.utc).isoformat(),
                        )
                        _write_gap_report(gap_rows, run_dir)
                        first_gap_written = True
                    if label == "t50_diagnostic":
                        training_summary = _write_training_exports(
                            run_dir, captures, manifest["sector_feature_sets"], INFO_CUTOFF.isoformat()
                        )
                        capture_context["enabled"] = False
                    filename = (
                        "model_run_t50_diagnostic.json" if label == "t50_diagnostic"
                        else "model_run_t50_enforced.json" if label == "t50_enforced"
                        else "model_run_t75_diagnostic.json" if label == "t75_diagnostic"
                        else "model_run_t100_diagnostic.json" if label == "t100_diagnostic"
                        else "model_run_t50_diagnostic_replay.json"
                    )
                    _write_json(run_dir / filename, result_runs[label])
                    print(f"done {label}", flush=True)
                snapshot_after = fz.build_file_manifest(snapshot_config)
    except Exception as exc:
        header["status"] = "FAILED"
        header["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        header["network_attempts"] = list(attempts)
        _write_json(run_dir / "run_header.json", header)
        raise
    finally:
        pl.ua.members_as_of = original_members
        se._fetch_company_facts = original_fact_reader
        se._fetch_submissions = original_submission_reader
        if original_get_fundamentals is not None:
            se.get_quarterly_fundamentals = original_get_fundamentals
        if ml_model is not None and original_model is not None:
            ml_model.run_4model = original_model["run"]
            ml_model.prepare_rolling_train = original_model["train"]
            ml_model.prepare_rolling_test = original_model["test"]

    baseline = result_runs["t50_diagnostic"]
    replay = result_runs["t50_diagnostic_replay"]
    output_fields = ("signal_as_of", "data_snapshot", "forecasts", "target_weights")
    equivalent = all(
        json.dumps(baseline[field], sort_keys=True, default=str, allow_nan=False)
        == json.dumps(replay[field], sort_keys=True, default=str, allow_nan=False)
        for field in output_fields
    )
    _write_json(run_dir / "replay_comparison.json", {
        "same_frozen_snapshot": True, "outputs_identical": equivalent,
        "compared_fields": list(output_fields), "network_attempts": attempts,
        "snapshot_unchanged": fz.compare_manifests(snapshot_manifest_before, snapshot_after),
    })
    _write_json(run_dir / "training_data_summary.json", training_summary or {})
    _write_json(run_dir / "readiness.json", {
        "cycle_key": CYCLE_KEY, "expected_issuer_count": manifest["expected_issuer_count"],
        "available_period_evidence": sum(row["report_status"] == "AVAILABLE" for row in manifest["expected"]),
        "unresolved_issuers_and_securities": manifest["unresolved"],
        "q2_gap_status_counts": dict(Counter(row["status"] for row in gap_rows)),
        "full_coverage": False, "executable": False, "historical_validation": "NOT_VALIDATED",
    })
    _write_json(run_dir / "coverage.json", {
        "cycle_key": CYCLE_KEY,
        "tickers": [{"cik": row["cik"], "ticker": row["ticker"]} for row in manifest["securities"]],
        "q2_gap_status_counts": dict(Counter(row["status"] for row in gap_rows)),
        "historical_validation": "NOT_VALIDATED",
    })
    (run_dir / "forecast_records.json").write_text(
        json.dumps(baseline["data_snapshot"].get("forecast_records", []), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_after = fz.build_file_manifest(config)
    unchanged_snapshot = fz.compare_manifests(snapshot_manifest_before, snapshot_after)
    unchanged_source = fz.compare_manifests(source_manifest_before, source_after)
    header.update({
        "network_attempts": attempts,
        "snapshot_unchanged_after_run": unchanged_snapshot,
        "source_unchanged_after_run": unchanged_source,
        "snapshot_manifest_after": {
            "file_count": snapshot_after["file_count"],
            "combined_sha256": snapshot_after["combined_sha256"],
        },
        "status": "COMPLETE_OFFLINE" if not attempts and unchanged_snapshot["unchanged"]
        and unchanged_source["unchanged"] and equivalent else "INCOMPLETE_OR_INPUTS_CHANGED",
        "replay_outputs_identical": equivalent,
        "q2_gap_status_counts": dict(Counter(row["status"] for row in gap_rows)),
        "historical_validation": "NOT_VALIDATED",
    })
    _write_json(run_dir / "source_manifest_after.json", source_after)
    _write_json(run_dir / "snapshot_manifest_after.json", snapshot_after)
    _write_json(run_dir / "run_header.json", header)
    artifact_names = [
        "model_run_t50_diagnostic.json", "model_run_t50_enforced.json",
        "model_run_t75_diagnostic.json", "model_run_t100_diagnostic.json",
        "model_run_t50_diagnostic_replay.json", "q2_gap_report.json", "q2_gap_report.csv",
        "readiness.json", "coverage.json", "forecast_records.json", "training_data_summary.json",
        "replay_comparison.json",
    ]
    _write_json(run_dir / "artifact_manifest.json", {
        "artifacts": {
            name: {"sha256": _sha256(run_dir / name), "size": (run_dir / name).stat().st_size}
            for name in artifact_names
        }
    })
    return run_dir


def main() -> None:
    if not CONFIG_PATH.exists():
        config, manifest = prepare_workspace()
        print(f"prepared {CYCLE_KEY}: {len(manifest['securities'])} securities, "
              f"{len(manifest['expected'])} issuer report expectations")
    else:
        config = pl.load_config(CONFIG_PATH)
        manifest_path = MANIFEST_DIR / f"{CYCLE_KEY}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = pl.load_config(CONFIG_PATH)
    run_dir = run_diagnostic(config, manifest)
    print(f"Q2 diagnostic archive: {run_dir.relative_to(ROOT)}")
    header = json.loads((run_dir / "run_header.json").read_text(encoding="utf-8"))
    print(f"status={header['status']} replay_identical={header.get('replay_outputs_identical')}")
    print(f"historical_validation={header['historical_validation']} executable={header['executable']}")


if __name__ == "__main__":
    main()
