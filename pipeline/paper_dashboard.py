"""Read-only, loopback-first dashboard for the persistent paper portfolio."""
from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sqlite3
from typing import Any

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


def _sector_lookup(universe_as_of: str | None) -> dict[str, int]:
    if not universe_as_of:
        return {}
    try:
        import pandas as pd
        try:
            from . import universe_asof as ua
        except ImportError:
            import universe_asof as ua  # type: ignore[no-redef]
        members, _ = ua.members_as_of(pd.Timestamp(universe_as_of))
        return {str(t): int(g) for t, g in zip(members["tic"], members["gsector"])}
    except Exception:
        return {}


def _report_status(connection: sqlite3.Connection, cycle_id: str) -> dict[str, Any]:
    """READY, evidenced special cases and open reports are kept apart; special cases are not READY."""
    rows = connection.execute(
        "SELECT f.filing_key, f.company_id, f.expected_period, f.status, r.resolution, r.evidence_json, r.recorded_at "
        "FROM expected_filings f LEFT JOIN filing_resolutions r ON r.cycle_id = f.cycle_id "
        "AND r.filing_key = f.filing_key WHERE f.cycle_id = ? ORDER BY f.company_id", (cycle_id,)).fetchall()
    ready = [r["company_id"] for r in rows if r["status"] == "READY"]
    special = [{"company_id": r["company_id"], "filing_key": r["filing_key"], "resolution": r["resolution"],
                "evidence": json.loads(r["evidence_json"]), "recorded_at": r["recorded_at"]}
               for r in rows if r["resolution"] and r["status"] != "READY"]
    special_keys = {x["filing_key"] for x in special}
    open_cases = [{"company_id": r["company_id"], "expected_period": r["expected_period"], "status": r["status"]}
                  for r in rows if r["status"] != "READY" and r["filing_key"] not in special_keys]
    return {"expected": len(rows), "ready": len(ready), "special_cases": special, "open": len(open_cases),
            "open_cases": open_cases, "gate_satisfied": not open_cases,
            "note": "CORPORATE_ACTION_RESOLVED is evidence-based and not READY; open cases include ordinary data gaps."}


def _weight_comparison(connection: sqlite3.Connection, environment: str, report_cycle_id: str | None,
                       portfolios: list[dict[str, Any]]) -> dict[str, Any]:
    signal = connection.execute(
        "SELECT snapshot_id AS signal_id, captured_at AS created_at, payload_json FROM data_snapshots "
        "WHERE snapshot_id LIKE 'model:%' ORDER BY captured_at DESC LIMIT 1").fetchone()
    benchmark = benchmark_weights.latest() if environment == "real" else None
    result: dict[str, Any] = {
        "benchmark": ({"source": benchmark["source_url"], "as_of": benchmark["as_of"], "sha256": benchmark["sha256"],
                       "kind": benchmark["kind"]} if benchmark else None),
        "signal": None, "rows": [], "sectors": [], "diagnostic": False,
        "banner": "Kein gültiges ML-Signal gespeichert: keine Zielgewichte vorhanden." if not signal else None}
    payload = json.loads(signal["payload_json"]) if signal else {}
    data = payload.get("data", {})
    if signal:
        result["diagnostic"] = data.get("eligibility_enforced") is False
        result["signal"] = {"signal_id": signal["signal_id"], "signal_as_of": payload.get("signal_as_of"),
                            "created_at": signal["created_at"], "universe_as_of": data.get("universe_as_of"),
                            "price_session": data.get("price_session"), "panel_fingerprint": data.get("panel_fingerprint")}
        if result["diagnostic"]:
            result["banner"] = "DIAGNOSELAUF - kein gültiges Portfolio"
    ml = [p for p in portfolios if "spy" not in p["strategy"]]
    names = [p["portfolio_id"] for p in ml]
    result["portfolios"] = names
    payload_targets = payload.get("target_weights", {}) if signal and not result["diagnostic"] else {}
    targets = {p["portfolio_id"]: payload_targets.get(p["strategy"], {}) for p in ml}
    actual: dict[str, dict[str, float]] = {}
    for p in ml:
        prices = {r["symbol"]: r["price"] for r in connection.execute(
            "SELECT o.symbol, f.price FROM simulated_fills f JOIN simulated_orders o ON o.order_id = f.order_id "
            "JOIN execution_batches b ON b.batch_id = o.batch_id JOIN reporting_cycles c ON c.cycle_id = b.cycle_id "
            "WHERE c.portfolio_id = ? ORDER BY f.filled_at", (p["portfolio_id"],))}
        nav = (p["latest_mark"] or {}).get("nav")
        if nav:
            actual[p["portfolio_id"]] = {x["symbol"]: x["quantity"] * prices[x["symbol"]] / nav
                                         for x in p["positions"] if x["symbol"] in prices}
    sectors = _sector_lookup(data.get("universe_as_of") or (benchmark or {}).get("as_of"))
    symbols = set((benchmark or {}).get("weights", {})) | {t for w in targets.values() for t in w}
    symbols |= {t for w in actual.values() for t in w}
    rows = []
    for symbol in symbols:
        row = {"symbol": symbol, "sector": GICS_SECTORS.get(sectors.get(symbol), "unbekannt"),
               "benchmark_weight": (benchmark or {}).get("weights", {}).get(symbol)}
        for name in names:
            row[f"target_{name}"] = targets.get(name, {}).get(symbol)
            row[f"actual_{name}"] = actual.get(name, {}).get(symbol)
        rows.append(row)
    rows.sort(key=lambda r: (-(r["benchmark_weight"] or 0.0), r["symbol"]))
    result["rows"] = rows
    totals: dict[str, dict[str, float]] = {}
    for r in rows:
        t = totals.setdefault(r["sector"], {"benchmark_weight": 0.0, **{f"{k}_{n}": 0.0 for k in ("target", "actual")
                                                                          for n in names}})
        for key in t:
            t[key] += r.get(key) or 0.0
    result["sectors"] = sorted(({"sector": k, **v} for k, v in totals.items()), key=lambda x: -x["benchmark_weight"])
    result["note"] = ("Benchmarkgewichte: SPY-ETF-Bestand laut Fondsdatei zum angegebenen Stichtag (nicht S&P-Index, "
                      "nicht zwingend der Depotstart). Tatsächliche Gewichte: Stück x letzter Fill-Preis / NAV.")
    return result

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
DEFAULT_DB = STATE_DIR / "real_paper.sqlite3"
SYNTHETIC_DB = STATE_DIR / "synthetic_demo.sqlite3"
REQUIRED_EVIDENCE = (
    "raw_price_basis", "corporate_actions", "historical_membership",
    "point_in_time_signal_and_labels", "benchmark_provenance",
)
REQUIRED_HASHES = ("raw_prices", "targets", "corporate_actions", "benchmark_values", "benchmark_metadata")

_HTML = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper-Depot</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f4f6f8;color:#15202b}
h1{margin-bottom:.25rem}.muted{color:#52606d}.card{background:white;border:1px solid #d8e0e8;border-radius:10px;padding:1rem;margin:1rem 0}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #d8e0e8;padding:2px 6px;text-align:right}td:first-child,th:first-child,td.l{text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0}.warn{color:#8a3b00;font-weight:600}
</style>
</head>
<body>
<h1>Virtuelles Paper-Depot</h1>
<p id="banner" class="warn"></p>
<p class="muted">Nur simulierte Buchungen. Keine Brokerorders. Aktualisierung: beim Neuladen.</p>
<section class="card"><h2>Status</h2><pre id="status">Lade lokalen Depotstatus …</pre></section>
<section class="card"><h2>Berichtsbereitschaft</h2><div id="reports">Lade …</div></section>
<section class="card"><h2>Gewichtsvergleich je Aktie</h2><p id="wbanner" class="warn"></p><p id="wnote" class="muted"></p><div id="weights">Lade …</div></section>
<section class="card"><h2>Sektorvergleich</h2><div id="sectors">Lade …</div></section>
<section class="card"><h2>Berichte & Zyklen</h2><pre id="cycles">Lade …</pre></section>
<section class="card"><h2>Aufträge & letzte Läufe</h2><pre id="operations">Lade …</pre></section>
<script>
async function load() {
  const response = await fetch('/api/status', {cache: 'no-store'});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  document.querySelector('#banner').textContent = data.environment + ': ' + data.data_banner;
  document.querySelector('#status').textContent = JSON.stringify({
    cycle_timelines: data.cycle_timelines,
    historical_backtest_validation: data.historical_backtest_validation,
    live_paper_depot: data.live_paper_depot,
    portfolios: data.portfolios,
    valuation_quality: data.valuation_quality,
    benchmark_and_model_data: 'Not asserted by the local ledger; inspect stored manifests.'
  }, null, 2);
  const pct = v => v == null ? '–' : (100 * v).toFixed(2) + ' %';
  const esc = t => String(t).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
  const rs = data.report_status;
  document.querySelector('#reports').innerHTML = rs ? `<p>Erwartet: ${rs.expected} | <b>READY: ${rs.ready}</b> | belegte Sonderfälle (nicht READY): ${rs.special_cases.length} | offen: ${rs.open} | Gate erfüllt: ${rs.gate_satisfied ? 'ja' : 'nein'}</p>` +
    '<h3>Belegte Sonderfälle</h3><pre>' + esc(JSON.stringify(rs.special_cases, null, 1)) + '</pre><h3>Offene Berichte</h3><pre>' + esc(rs.open_cases.map(c => c.company_id + ' ' + c.expected_period + ' ' + c.status).join('\n')) + '</pre>' : 'Kein Zyklus.';
  const w = data.weight_comparison;
  document.querySelector('#wbanner').textContent = w.banner || '';
  document.querySelector('#wnote').textContent = (w.benchmark ? 'Benchmark-Quelle: ' + w.benchmark.source + ' (Stand ' + w.benchmark.as_of + ', SHA-256 ' + w.benchmark.sha256.slice(0, 16) + '…). ' : 'Keine Benchmarkgewichte. ') + w.note + (w.signal ? ' Signal: ' + w.signal.signal_as_of + ', Kursstand ' + w.signal.price_session : '');
  const head = '<tr><th>Aktie</th><th>Sektor</th><th>Benchmark</th>' + w.portfolios.map(n => `<th>Ziel ${n}</th><th>Ist ${n}</th>`).join('') + '</tr>';
  document.querySelector('#weights').innerHTML = '<table>' + head + w.rows.map(r => `<tr><td>${esc(r.symbol)}</td><td class="l">${esc(r.sector)}</td><td>${pct(r.benchmark_weight)}</td>` + w.portfolios.map(n => `<td>${pct(r['target_' + n])}</td><td>${pct(r['actual_' + n])}</td>`).join('') + '</tr>').join('') + '</table>';
  document.querySelector('#sectors').innerHTML = '<table>' + head.replace('Aktie', 'Sektor').replace('<th>Sektor</th>', '') + w.sectors.map(r => `<tr><td>${esc(r.sector)}</td><td>${pct(r.benchmark_weight)}</td>` + w.portfolios.map(n => `<td>${pct(r['target_' + n])}</td><td>${pct(r['actual_' + n])}</td>`).join('') + '</tr>').join('') + '</table>';
  document.querySelector('#cycles').textContent = JSON.stringify(data.cycles, null, 2);
  document.querySelector('#operations').textContent = JSON.stringify({
    latest_orders: data.latest_orders,
    latest_jobs: data.latest_jobs,
    latest_errors: data.latest_errors
  }, null, 2);
}
load().catch(error => {
  for (const id of ['status', 'cycles', 'operations', 'reports', 'weights', 'sectors']) {
    document.querySelector(`#${id}`).textContent = `Dashboard-Fehler: ${error.message}`;
  }
});
</script>
</body>
</html>
"""


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
                      dividend_receivables, nav, benchmark_value
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
        if self.path == "/":
            body = _HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif self.path == "/api/status":
            body = json.dumps(
                dashboard_snapshot(self.database, self.validation_manifest), ensure_ascii=False, default=str
            ).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif self.path == "/api/health":
            PaperStore(self.database)
            body = b'{"status":"ok"}'
            content_type = "application/json; charset=utf-8"
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
