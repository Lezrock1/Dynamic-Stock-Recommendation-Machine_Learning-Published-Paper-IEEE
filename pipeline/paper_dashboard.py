"""Read-only, loopback-first dashboard for the persistent paper portfolio."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import parse_qs, urlsplit

try:
    from .paper_store import PaperStore
    from . import benchmark_weights
except ImportError:
    from paper_store import PaperStore  # type: ignore[no-redef]
    import benchmark_weights  # type: ignore[no-redef]

GICS_SECTORS = {
    10: "Energie", 15: "Roh-, Hilfs- & Betriebsstoffe", 20: "Industrie", 25: "Zyklischer Konsum",
    30: "Basiskonsumgüter", 35: "Gesundheitswesen", 40: "Finanzwesen", 45: "Informationstechnologie",
    50: "Kommunikationsdienste", 55: "Versorger", 60: "Immobilien",
}
SYMBOL_ALIASES = {"BRK.B": "BRK-B", "BF.B": "BF-B"}


def _canonical_symbol(symbol: str) -> str:
    value = str(symbol).strip().upper()
    return SYMBOL_ALIASES.get(value, value)


def _universe_metadata(universe_as_of: str | None) -> tuple[dict[str, int], dict[str, str]]:
    if not universe_as_of:
        return {}, {}
    import pandas as pd
    try:
        from . import universe_asof as ua
    except ImportError:
        import universe_asof as ua  # type: ignore[no-redef]
    members, _ = ua.members_as_of(pd.Timestamp(universe_as_of))
    sectors = {_canonical_symbol(str(row.tic)): int(row.gsector) for row in members.itertuples()}
    names = {
        _canonical_symbol(str(row.tic)): str(row.conm)
        for row in members.itertuples()
        if getattr(row, "conm", None) and str(row.conm).strip()
    }
    return sectors, names


def _sector_lookup(universe_as_of: str | None) -> dict[str, int]:
    return _universe_metadata(universe_as_of)[0]


def _report_status(connection: sqlite3.Connection, cycle_id: str) -> dict[str, Any]:
    """READY, evidenced special cases and open reports are kept apart; special cases are not READY."""
    rows = connection.execute(
        "SELECT f.filing_key, f.company_id, f.expected_period, f.status, r.resolution, r.evidence_json, r.recorded_at "
        "FROM expected_filings f LEFT JOIN filing_resolutions r ON r.cycle_id = f.cycle_id "
        "AND r.filing_key = f.filing_key WHERE f.cycle_id = ? ORDER BY f.company_id", (cycle_id,)).fetchall()
    ready = [r["company_id"] for r in rows if r["status"] == "READY" and not r["resolution"]]
    special = [{"company_id": r["company_id"], "filing_key": r["filing_key"], "resolution": r["resolution"],
                "evidence": json.loads(r["evidence_json"]), "recorded_at": r["recorded_at"]}
               for r in rows if r["resolution"]]
    special_keys = {x["filing_key"] for x in special}
    open_cases = [{"company_id": r["company_id"], "expected_period": r["expected_period"], "status": r["status"]}
                  for r in rows if r["status"] != "READY" and r["filing_key"] not in special_keys]
    groups = {
        "READY": len(ready),
        "WAITING_FACTS": sum(1 for c in open_cases if c["status"] == "WAITING_FACTS"),
        "CORPORATE_ACTION_RESOLVED": len(special),
        "WAITING_FILING": sum(1 for c in open_cases if c["status"] == "WAITING_FILING"),
        "INVALID_DATA": sum(1 for c in open_cases if c["status"] == "INVALID_DATA"),
    }
    return {"expected": len(rows), "ready": len(ready), "special_cases": special, "open": len(open_cases),
            "status_groups": groups, "status_groups_sum": sum(groups.values()),
            "open_cases": open_cases, "gate_satisfied": not open_cases,
            "note": "CORPORATE_ACTION_RESOLVED is evidence-based and not READY; open cases include ordinary data gaps."}


def _weight_comparison(connection: sqlite3.Connection, environment: str, report_cycle_id: str | None,
                       portfolios: list[dict[str, Any]]) -> dict[str, Any]:
    signal = connection.execute(
        "SELECT snapshot_id AS signal_id, captured_at AS created_at, payload_json FROM data_snapshots "
        "WHERE snapshot_id LIKE 'model:%' ORDER BY captured_at DESC LIMIT 1").fetchone()
    benchmark = benchmark_weights.latest() if environment == "real" else None
    result: dict[str, Any] = {
        "benchmark": ({
            "source": benchmark["source_url"],
            "as_of": benchmark["as_of"],
            "sha256": benchmark["sha256"],
            "kind": benchmark["kind"],
            "fetched_at": benchmark.get("fetched_at"),
            "total_weight": benchmark.get("sum"),
            "equity_weight_sum": benchmark.get("equity_weight_sum"),
            "non_equity_weight_sum": benchmark.get("non_equity_weight_sum"),
            "non_equity_positions": benchmark.get("non_equity_positions", []),
            "symbol_aliases": {},
        } if benchmark else None),
        "signal": None, "rows": [], "sectors": [], "diagnostic": False, "actual_marked_at": {},
        "banner": "Kein gültiges ML-Signal gespeichert: keine Zielgewichte vorhanden." if not signal else None}
    payload = json.loads(signal["payload_json"]) if signal else {}
    data = payload.get("data", {})
    if signal:
        diagnostic_symbols = {
            str(record.get("security_id"))
            for record in data.get("forecast_records", [])
            if record.get("forecast_kind") == "DIAGNOSTIC_ONLY"
        }
        stored_targets = payload.get("target_weights", {})
        targeted_diagnostics = any(
            diagnostic_symbols.intersection(weights)
            for weights in stored_targets.values()
        )
        result["diagnostic"] = data.get("eligibility_enforced") is not True or targeted_diagnostics
        result["signal"] = {"signal_id": signal["signal_id"], "signal_as_of": payload.get("signal_as_of"),
                            "created_at": signal["created_at"], "universe_as_of": data.get("universe_as_of"),
                            "price_session": data.get("price_session"), "panel_fingerprint": data.get("panel_fingerprint"),
                            "origin": data.get("origin")}
        if result["diagnostic"]:
            result["banner"] = "DIAGNOSELAUF - kein gültiges Portfolio"
    ml = [p for p in portfolios if "spy" not in p["strategy"]]
    names = [p["portfolio_id"] for p in ml]
    result["portfolios"] = names
    payload_targets = payload.get("target_weights", {}) if signal and not result["diagnostic"] else {}
    targets: dict[str, dict[str, float]] = {}
    for portfolio in ml:
        weights = payload_targets.get(portfolio["strategy"], {})
        normalized: dict[str, float] = {}
        for symbol, weight in weights.items():
            canonical = _canonical_symbol(symbol)
            if canonical in normalized:
                raise ValueError(f"duplicate ticker aliases in target weights for {portfolio['strategy']}: {canonical}")
            normalized[canonical] = float(weight)
        targets[portfolio["portfolio_id"]] = normalized
    actual: dict[str, dict[str, float] | None] = {}
    for p in ml:
        latest = p["latest_mark"] or {}
        nav = latest.get("nav")
        raw_values = latest.get("position_values_json")
        if raw_values is None or not nav:
            actual[p["portfolio_id"]] = None
            continue
        marked_values = json.loads(raw_values)
        normalized_values: dict[str, float] = {}
        for symbol, item in marked_values.items():
            canonical = _canonical_symbol(symbol)
            if canonical in normalized_values:
                raise ValueError(f"duplicate ticker aliases in daily mark for {p['portfolio_id']}: {canonical}")
            normalized_values[canonical] = float(item["market_value"]) / float(nav)
        actual[p["portfolio_id"]] = normalized_values
        result["actual_marked_at"][p["portfolio_id"]] = latest["marked_at"]
    sectors, company_names = _universe_metadata(data.get("universe_as_of") or (benchmark or {}).get("as_of"))
    benchmark_values: dict[str, float] = {}
    aliases_applied: dict[str, str] = {}
    for raw_symbol, weight in (benchmark or {}).get("weights", {}).items():
        symbol = _canonical_symbol(raw_symbol)
        if symbol in benchmark_values:
            raise ValueError(f"duplicate ticker aliases in benchmark holdings: {symbol}")
        benchmark_values[symbol] = float(weight)
        if symbol != raw_symbol:
            aliases_applied[raw_symbol] = symbol
    if benchmark:
        result["benchmark"]["symbol_aliases"] = aliases_applied
    symbols = set(benchmark_values) | {t for weights in targets.values() for t in weights}
    symbols |= {t for weights in actual.values() if weights is not None for t in weights}
    rows = []
    for symbol in symbols:
        row = {"symbol": symbol, "name": company_names.get(symbol),
               "sector": GICS_SECTORS.get(sectors.get(symbol), "unbekannt"),
               "benchmark_weight": benchmark_values.get(symbol)}
        for name in names:
            row[f"target_{name}"] = targets[name].get(symbol, 0.0) if targets.get(name) else None
            actual_weights = actual.get(name)
            row[f"actual_{name}"] = actual_weights.get(symbol, 0.0) if actual_weights is not None else None
            target_weight = row[f"target_{name}"]
            actual_weight = row[f"actual_{name}"]
            row[f"target_active_{name}"] = (
                target_weight - row["benchmark_weight"]
                if target_weight is not None and row["benchmark_weight"] is not None else None
            )
            row[f"actual_active_{name}"] = (
                actual_weight - row["benchmark_weight"]
                if actual_weight is not None and row["benchmark_weight"] is not None else None
            )
        rows.append(row)
    rows.sort(key=lambda r: (-(r["benchmark_weight"] or 0.0), r["symbol"]))
    result["rows"] = rows
    totals: dict[str, dict[str, float | None]] = {}
    for r in rows:
        t = totals.setdefault(r["sector"], {
            "benchmark_weight": 0.0 if benchmark else None,
            **{f"{kind}_{n}": (0.0 if kind == "target" and targets.get(n) else
                               0.0 if kind == "actual" and actual.get(n) is not None else None)
               for kind in ("target", "actual") for n in names},
        })
        if t["benchmark_weight"] is not None and r["benchmark_weight"] is not None:
            t["benchmark_weight"] += r["benchmark_weight"]
        for name in names:
            for kind in ("target", "actual"):
                key = f"{kind}_{name}"
                value = r[f"{kind}_{name}"]
                if t[key] is not None and value is not None:
                    t[key] += value
    result["sectors"] = sorted(
        ({"sector": k, **v} for k, v in totals.items()),
        key=lambda x: -(x["benchmark_weight"] or 0.0),
    )
    result["note"] = ("Benchmarkgewichte: SPY-ETF-Bestand laut Fondsdatei zum angegebenen Stichtag (nicht S&P-Index, "
                      "nicht zwingend der Depotstart). Tatsächliche Gewichte: gespeicherter Marktwert je Aktie / "
                      "NAV aus demselben Bewertungszeitpunkt; ohne Positionspreis-Snapshot werden sie nicht angezeigt.")
    result["benchmark_label"] = "SPY-ETF-Bestände als S&P-500-Proxy; kein S&P-500-Indexgewicht"
    return result

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
DEFAULT_DB = STATE_DIR / "real_paper.sqlite3"
SYNTHETIC_DB = STATE_DIR / "synthetic_demo.sqlite3"
REQUIRED_EVIDENCE = (
    "raw_price_basis", "corporate_actions", "historical_membership",
    "point_in_time_signal_and_labels", "benchmark_provenance",
)
REQUIRED_HASHES = ("raw_prices", "targets", "corporate_actions", "benchmark_values", "benchmark_metadata")

try:
    from .dashboard_view import (
        HTML as _HTML, enhance_snapshot as _enhance_dashboard_snapshot, diagnostic_file,
        target_package_download,
    )
except ImportError:
    from dashboard_view import (
        HTML as _HTML, enhance_snapshot as _enhance_dashboard_snapshot, diagnostic_file,
        target_package_download,
    )



def evaluate_backtest_validation(manifest: str | Path | None) -> dict[str, str]:
    """Historical validation is recomputed from evidence; a bare status field never counts.

    VALIDATED requires a historical-backtest run manifest with all input hashes and, for every
    evidence category, ``verified`` plus an evidence file whose SHA-256 matches the recorded one.
    """
    if manifest is None:
        return {"status": "NOT_VALIDATED", "detail": "No historical backtest manifest configured."}
    path = Path(manifest)
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "NOT_VALIDATED", "detail": f"Manifest missing or unreadable: {path}"}
    problems: list[str] = []
    if recorded.get("mode") != "historical_backtest":
        problems.append("not a historical_backtest run manifest")
    hashes = recorded.get("input_sha256") or {}
    problems += [f"missing input hash {k}" for k in REQUIRED_HASHES if not hashes.get(k)]
    evidence = recorded.get("validation_evidence") or {}
    for key in REQUIRED_EVIDENCE:
        item = evidence.get(key)
        if not isinstance(item, dict) or item.get("verified") is not True:
            problems.append(f"evidence {key} not verified")
            continue
        uri = str(item.get("evidence_uri", "")).strip()
        target = (path.parent / uri) if uri and "://" not in uri else None
        if target is None or not target.is_file():
            problems.append(f"evidence {key}: no local evidence file")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != item.get("sha256"):
            problems.append(f"evidence {key}: SHA-256 mismatch")
    if problems:
        return {"status": "NOT_VALIDATED", "detail": "; ".join(problems), "source": str(path)}
    return {"status": "VALIDATED", "detail": "All evidence categories verified by file hash.",
            "source": str(path)}


def dashboard_snapshot(database: str | Path, validation_manifest: str | Path | None = None) -> dict[str, Any]:
    """Return read-only portfolio, report-cycle, valuation and job status."""
    store = PaperStore(database)
    environment = store.environment()
    uri = store.path.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        portfolios = [
            dict(row) for row in connection.execute(
                "SELECT portfolio_id, name, strategy, initial_cash, created_at "
                "FROM portfolios ORDER BY portfolio_id"
            )
        ]
        marks_by_portfolio: dict[str, list[dict[str, Any]]] = {}
        for row in connection.execute(
            """SELECT portfolio_id, marked_at, cash, positions_value,
                      dividend_receivables, nav, benchmark_value, position_values_json
               FROM daily_marks ORDER BY portfolio_id, marked_at"""
        ):
            marks_by_portfolio.setdefault(row["portfolio_id"], []).append(dict(row))
        for portfolio in portfolios:
            marks = marks_by_portfolio.get(portfolio["portfolio_id"], [])
            positions = connection.execute(
                """SELECT symbol, SUM(quantity_delta) AS quantity
                   FROM position_ledger WHERE portfolio_id = ?
                   GROUP BY symbol HAVING ABS(SUM(quantity_delta)) > 1e-9
                   ORDER BY symbol""",
                (portfolio["portfolio_id"],),
            ).fetchall()
            peak = None
            for mark in marks:
                peak = mark["nav"] if peak is None else max(peak, mark["nav"])
            latest = marks[-1] if marks else None
            portfolio["latest_mark"] = latest
            portfolio["daily_marks"] = marks
            portfolio["positions"] = [dict(row) for row in positions]
            portfolio["drawdown"] = (
                latest["nav"] / peak - 1.0 if latest is not None and peak else None
            )
        tracked = [p for p in portfolios if p["strategy"] == "spy_buy_hold_v1"
                   or p["strategy"] in {"equal_weight_v1", "mean_var_v1", "min_var_v1"}]
        date_sets = [
            {mark["marked_at"] for mark in p["daily_marks"]}
            for p in tracked
        ]
        shared_dates = sorted(set.intersection(*date_sets)) if date_sets and all(date_sets) else []
        nav_series: list[dict[str, Any]] = []
        if shared_dates:
            first_date = shared_dates[0]
            base_marks = {
                p["portfolio_id"]: next(mark for mark in p["daily_marks"] if mark["marked_at"] == first_date)
                for p in tracked
            }
            for marked_at in shared_dates:
                point: dict[str, Any] = {"marked_at": marked_at, "portfolios": {}}
                for portfolio in tracked:
                    mark = next(mark for mark in portfolio["daily_marks"] if mark["marked_at"] == marked_at)
                    base_mark = base_marks[portfolio["portfolio_id"]]
                    point["portfolios"][portfolio["portfolio_id"]] = mark["nav"] / base_mark["nav"]
                proxy_source = next(
                    (mark for portfolio in tracked for mark in portfolio["daily_marks"]
                     if mark["marked_at"] == marked_at and mark["benchmark_value"] is not None),
                    None,
                )
                base_proxy = next(
                    (mark["benchmark_value"] for portfolio in tracked for mark in portfolio["daily_marks"]
                     if mark["marked_at"] == first_date and mark["benchmark_value"] is not None),
                    None,
                )
                point["spy_adj_close_proxy"] = (
                    proxy_source["benchmark_value"] / base_proxy
                    if proxy_source is not None and base_proxy else None
                )
                nav_series.append(point)
        portfolio_accounts = [
            {
                "portfolio_id": portfolio["portfolio_id"],
                "marked_at": (portfolio["latest_mark"] or {}).get("marked_at"),
                "nav": (portfolio["latest_mark"] or {}).get("nav"),
                "cash": (portfolio["latest_mark"] or {}).get("cash"),
                "dividend_receivables": (portfolio["latest_mark"] or {}).get("dividend_receivables"),
                "positions_value": (portfolio["latest_mark"] or {}).get("positions_value"),
            }
            for portfolio in portfolios
        ]

        cycles = [
            dict(row) for row in connection.execute(
                """SELECT c.cycle_id, c.portfolio_id, c.cycle_key, c.strategy,
                          c.started_at, c.rebalance_policy, c.deadline_at, c.status,
                          c.blocked_reason, c.filing_status, c.model_status, c.model_reason,
                          COUNT(f.filing_key) AS expected_count,
                          SUM(CASE WHEN f.status = 'READY' THEN 1 ELSE 0 END) AS ready_count,
                          SUM(CASE WHEN f.status = 'WAITING_FILING' THEN 1 ELSE 0 END) AS waiting_filing_count,
                          SUM(CASE WHEN f.status = 'WAITING_FACTS' THEN 1 ELSE 0 END) AS waiting_facts_count,
                          SUM(CASE WHEN f.status = 'INVALID_DATA' THEN 1 ELSE 0 END) AS invalid_count,
                          GROUP_CONCAT(CASE WHEN f.status != 'READY' THEN
                              f.company_id || ':' || f.expected_period || ':' || f.status END) AS missing_filings
                   FROM reporting_cycles c
                   LEFT JOIN expected_filings f ON f.cycle_id = c.cycle_id
                   GROUP BY c.cycle_id
                   ORDER BY c.started_at DESC"""
            )
        ]
        timelines = [
            dict(row) for row in connection.execute(
                "SELECT cycle_id, published_at, first_seen_at, snapshot_as_of, model_completed_at "
                "FROM cycle_timeline ORDER BY model_completed_at DESC LIMIT 20"
            )
        ]
        orders = [
            dict(row) for row in connection.execute(
                """SELECT c.portfolio_id, b.cycle_id, b.strategy, o.client_order_id,
                          o.symbol, o.side, o.quantity, o.filled_quantity, o.status
                   FROM simulated_orders o
                   JOIN execution_batches b ON b.batch_id = o.batch_id
                   JOIN reporting_cycles c ON c.cycle_id = b.cycle_id
                   ORDER BY b.created_at DESC LIMIT 100"""
            )
        ]
        jobs = [
            dict(row) for row in connection.execute(
                "SELECT run_id, job_type, status, started_at, finished_at "
                "FROM job_runs ORDER BY started_at DESC LIMIT 25"
            )
        ]
        errors = [
            dict(row) for row in connection.execute(
                "SELECT run_id, message, details_json, created_at "
                "FROM job_errors ORDER BY created_at DESC LIMIT 25"
            )
        ]
        data_origins = sorted({
            origin
            for row in connection.execute("SELECT payload_json FROM job_events WHERE event_type = 'started'")
            if (origin := json.loads(row["payload_json"]).get("data_origin"))
        })
        latest_signal = connection.execute(
            "SELECT created_at, payload_json FROM signal_snapshots ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        rebalances = [
            dict(row) for row in connection.execute(
                """SELECT b.cycle_id, c.portfolio_id, b.created_at, s.scheduled_for, c.status
                   FROM execution_batches b JOIN reporting_cycles c ON c.cycle_id = b.cycle_id
                   LEFT JOIN execution_schedule s ON s.batch_id = b.batch_id
                   ORDER BY b.created_at DESC LIMIT 10"""
            )
        ]
        primary_cycle = next((c["cycle_id"] for c in cycles if c["portfolio_id"] == "equal_weight"), None) or (
            cycles[0]["cycle_id"] if cycles else None)
        report_status = _report_status(connection, primary_cycle) if primary_cycle else None
        weight_comparison = _weight_comparison(connection, environment, primary_cycle, portfolios)
        traded = connection.execute(
            "SELECT COALESCE(SUM(quantity * price), 0) AS notional, "
            "COALESCE(SUM(transaction_cost), 0) AS costs FROM simulated_fills"
        ).fetchone()
        failed_jobs = connection.execute(
            "SELECT COUNT(*) FROM job_runs WHERE status = 'FAILED'"
        ).fetchone()[0]
        last_activity = connection.execute("SELECT MAX(started_at) FROM job_runs").fetchone()[0]
    statuses = {c["status"] for c in cycles}
    if "BLOCKED" in statuses:
        live_state = "BLOCKED"
    elif "OPEN" in statuses:
        live_state = "WAITING_FOR_REPORTS"
    elif "SIGNALLED" in statuses:
        live_state = "AWAITING_EXECUTION"
    elif statuses:
        live_state = "RUNNING"
    else:
        live_state = "NOT_STARTED"
    if environment == "real":
        validation = evaluate_backtest_validation(validation_manifest)
    else:
        validation = {"status": "NOT_VALIDATED",
                      "detail": f"{environment} data never counts as historical validation."}
    return {
        "environment": environment,
        "data_banner": ("REAL PAPER TRACKING (real SEC and market data, virtual bookings)"
                        if environment == "real" else
                        "SYNTHETIC DEMO / TEST - NOT REAL DATA, NOT A TRACK RECORD"),
        "cycle_timelines": timelines,
        "historical_backtest_validation": validation,
        "live_paper_depot": {
            "status": live_state,
            "data_origins": data_origins,
            "latest_signal_at": latest_signal["created_at"] if latest_signal else None,
            "latest_signal_as_of": (
                json.loads(latest_signal["payload_json"]).get("signal_as_of") if latest_signal else None
            ),
            "rebalances": rebalances,
            "traded_notional": traded["notional"],
            "transaction_costs": traded["costs"],
            "failed_job_attempts": failed_jobs,
            "last_scheduler_activity": last_activity,
        },
        "benchmark_label": ("Primärer Vergleich: paralleles SPY-Musterdepot 'spy_benchmark' (gleiches Startkapital, "
                            "Start, USD, Kosten- und Dividendenregeln; Rohkurs + gebuchte Dividenden; kein S&P-500-TR-Index). "
                            "Spalte benchmark_value = SPY adj_close, nur Total-Return-Näherung."),
        "portfolios": portfolios,
        "report_status": report_status,
        "weight_comparison": weight_comparison,
        "portfolio_accounts": portfolio_accounts,
        "nav_comparison": {
            "actual_common_start": nav_series[0]["marked_at"] if nav_series else None,
            "series": nav_series,
            "benchmark_label": "SPY-ETF-Proxy (adj_close), zusätzliche Referenz; primär ist das parallele SPY-Depot",
        },
        "automatic_refresh_seconds": 60,
        "cycles": cycles,
        "latest_orders": orders,
        "latest_jobs": jobs,
        "latest_errors": errors,
        "valuation_quality": (
            "NO_DAILY_MARKS" if not any(marks_by_portfolio.values())
            else "LATEST_DAILY_MARKS_RECORDED"
        ),
    }


class _DashboardHandler(BaseHTTPRequestHandler):
    database: Path
    validation_manifest: Path | None = None

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path == "/":
            body = _HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif request.path == "/api/status":
            query = parse_qs(request.query, keep_blank_values=True)
            cycle_values = query.get("cycle_key", [])
            if len(cycle_values) > 1 or (
                cycle_values and cycle_values[0] and not re.fullmatch(r"\d{4}Q[1-4]", cycle_values[0])
            ):
                self.send_error(400, "invalid cycle_key")
                return
            cycle_key = cycle_values[0] if cycle_values and cycle_values[0] else None
            body = json.dumps(
                _enhance_dashboard_snapshot(
                    dashboard_snapshot(self.database, self.validation_manifest),
                    Path(__file__).resolve().parent.parent,
                    cycle_key,
                ),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif request.path == "/api/health":
            PaperStore(self.database)
            body = b'{"status":"ok"}'
            content_type = "application/json; charset=utf-8"
        elif request.path == "/api/diagnostic-download":
            query = parse_qs(request.query)
            names = query.get("name", [])
            if len(names) != 1 or set(query) != {"name"}:
                self.send_error(400, "exactly one artifact name required")
                return
            try:
                path = diagnostic_file(Path(__file__).resolve().parent.parent, names[0])
                body = path.read_bytes()
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(404, "verified diagnostic artifact unavailable")
                return
            content_type = "text/csv; charset=utf-8" if path.suffix == ".csv" else "application/json; charset=utf-8"
        elif request.path == "/api/target-download":
            query = parse_qs(request.query)
            if set(query) != {"package", "name"} or any(len(values) != 1 for values in query.values()):
                self.send_error(400, "exactly one package and artifact name required")
                return
            name = query["name"][0]
            try:
                body = target_package_download(
                    Path(__file__).resolve().parent.parent, query["package"][0], name)
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(404, "verified target artifact unavailable")
                return
            content_type = ("text/csv; charset=utf-8" if name.endswith(".csv") else
                            "text/plain; charset=utf-8" if name.endswith(".md") else
                            "application/json; charset=utf-8")
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def serve(database: str | Path = DEFAULT_DB, host: str = "127.0.0.1", port: int = 8765,
          validation_manifest: str | Path | None = None) -> None:
    store = PaperStore(database)
    handler = type("PaperDashboardHandler", (_DashboardHandler,),
                   {"database": store.path, "validation_manifest": validation_manifest})
    server = HTTPServer((host, port), handler)
    print(f"Paper dashboard listening at http://{host}:{port}/ (read-only)")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local read-only paper portfolio dashboard.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.db, args.host, args.port)


if __name__ == "__main__":
    main()
