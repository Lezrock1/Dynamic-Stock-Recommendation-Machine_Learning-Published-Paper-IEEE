"""Command-line administration for the local SQLite paper-depot state."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import uuid

try:
    from .paper_dashboard import DEFAULT_DB, SYNTHETIC_DB, dashboard_snapshot, serve
    from .paper_runner import PaperRunner, RunnerConfig, exclusive_lock, load_adapters
    from .paper_store import PaperStore
except ImportError:
    from paper_dashboard import DEFAULT_DB, SYNTHETIC_DB, dashboard_snapshot, serve
    from paper_runner import PaperRunner, RunnerConfig, exclusive_lock, load_adapters
    from paper_store import PaperStore

STRATEGIES = {
    "equal_weight": ("Equal Weight", "equal_weight_v1"),
    "mean_var": ("Mean-Var", "mean_var_v1"),
    "min_var": ("Min-Var", "min_var_v1"),
    "spy_benchmark": ("SPY Benchmark Portfolio", "spy_buy_hold_v1"),
}


def _load_expected_filings(path: Path) -> tuple[list[str], dict[str, dict[str, object]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "filing_key", "company_id", "security_id", "expected_period",
            "filing_type", "source",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "expected-filings CSV must contain filing_key,company_id,security_id,"
                "expected_period,filing_type,source"
            )
        rows = list(reader)
    keys: list[str] = []
    metadata: dict[str, dict[str, object]] = {}
    for row in rows:
        key = row["filing_key"]
        if key in metadata:
            raise ValueError(f"duplicate filing_key in expected directory: {key}")
        keys.append(key)
        metadata[key] = {
            "company_id": row["company_id"],
            "security_id": row["security_id"],
            "expected_period": row["expected_period"],
            "filing_type": row["filing_type"],
            "source": row["source"],
        }
    return keys, metadata


def _database_portfolios(database: Path) -> dict[str, tuple[str, str, float]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT portfolio_id, name, strategy, initial_cash FROM portfolios"
        ).fetchall()
    return {row[0]: (row[1], row[2], float(row[3])) for row in rows}


def _validation_manifest(config: dict | None) -> Path | None:
    value = (config or {}).get("validation_manifest")
    return (Path(__file__).resolve().parent.parent / value) if value else None


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage the local, paper-only portfolio database.")
    parser.add_argument("--env", choices=("real", "synthetic"), default="real",
                        help="real paper tracking (default) or synthetic demo; they never share a database")
    parser.add_argument("--db", type=Path, default=None,
                        help="default: state/real_paper.sqlite3 (real) or state/synthetic_demo.sqlite3")
    parser.add_argument("--config", type=Path, default=None, help="real configuration (config/real_paper.json)")
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="explicitly create the database and three isolated portfolios")
    initialize.add_argument("--initial-capital", type=float, default=1_000_000.0)

    migrate = commands.add_parser("migrate", help="versioned schema migration with verified backup first")
    migrate.add_argument("--backup-dir", type=Path, default=None)

    generate = commands.add_parser(
        "generate-cycle", help="derive expected reports from the universe and open one cycle per strategy")
    generate.add_argument("--cycle-key", required=True)
    generate.add_argument("--quarter-end", required=True, help="calendar quarter end, e.g. 2026-09-30")
    generate.add_argument("--deadline-at")

    correct = commands.add_parser(
        "correct-periods", help="re-derive expected periods of a not yet used cycle with the current period rule")
    correct.add_argument("--cycle-key", required=True)
    correct.add_argument("--quarter-end", required=True)
    correct.add_argument("--apply", action="store_true", help="without it only the audit is printed")

    backup = commands.add_parser("backup", help="create a consistent SQLite backup")
    backup.add_argument("--output", type=Path, required=True)

    dashboard = commands.add_parser("dashboard", help="serve the read-only local dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)

    create_cycle = commands.add_parser("create-cycle", help="freeze a cycle and its expected-filing directory")
    create_cycle.add_argument("--portfolio", required=True, choices=tuple(STRATEGIES))
    create_cycle.add_argument("--cycle-key", required=True)
    create_cycle.add_argument("--expected-filings", type=Path, required=True)
    create_cycle.add_argument(
        "--policy", choices=("all_expected_filings", "fixed_quarterly"),
        default="all_expected_filings",
    )
    create_cycle.add_argument("--deadline-at")

    update_filing = commands.add_parser("update-filing", help="record a filing/facts validation result")
    update_filing.add_argument("--cycle-id", required=True)
    update_filing.add_argument("--filing-key", required=True)
    update_filing.add_argument("--received", action="store_true")
    update_filing.add_argument("--facts-ready", action="store_true")
    update_filing.add_argument("--invalid", action="store_true")
    update_filing.add_argument("--filed-at")
    update_filing.add_argument("--first-seen-at")
    update_filing.add_argument("--source")
    update_filing.add_argument("--error-reason")

    set_model = commands.add_parser("set-model-status", help="record a separate model-readiness outcome")
    set_model.add_argument("--cycle-id", required=True)
    set_model.add_argument("--status", choices=("WAITING", "BLOCKED", "READY", "FAILED"), required=True)
    set_model.add_argument("--reason")

    for name, text in (
        ("check-filings", "check expected reports once and log their availability"),
        ("run-signal", "run the single predict_as_of model run for a ready cycle"),
        ("tick", "one locked scheduler pass (for cron/launchd)"),
        ("scheduler", "locked scheduler loop with persistent retries"),
    ):
        runner_cmd = commands.add_parser(name, help=text)
        runner_cmd.add_argument("--adapters", default=None,
                                help="package.module:factory (default for --env real: paper_live:real_adapters)")
        if name in {"check-filings", "run-signal"}:
            runner_cmd.add_argument("--cycle-key", required=True)
        if name == "scheduler":
            runner_cmd.add_argument("--interval-seconds", type=float, default=900.0)

    demo = commands.add_parser("demo", help="SYNTHETIC offline demo on a NEW database (no real data)")
    demo.add_argument("--symbols", type=int, default=4)

    commands.add_parser("status", help="print a read-only JSON view of the persistent paper state")
    args = parser.parse_args(argv)
    database = (args.db or (DEFAULT_DB if args.env == "real" else SYNTHETIC_DB)).expanduser().resolve()
    if args.env == "synthetic" and database == DEFAULT_DB.resolve():
        parser.error("the synthetic environment must not use the real database")
    real_config = None
    if args.env == "real":
        try:
            from .paper_live import load_config, resolve_path
        except ImportError:
            from paper_live import load_config, resolve_path
        real_config = load_config(args.config)
        if args.db is None:
            database = resolve_path(real_config, "database")

    if args.command == "init":
        if args.initial_capital <= 0:
            parser.error("--initial-capital must be positive")
        store = PaperStore.create(database, args.env)
        existing = _database_portfolios(database)
        unexpected = set(existing) - set(STRATEGIES)
        if unexpected:
            raise ValueError(
                f"database has unexpected portfolio IDs; refusing to change them: {sorted(unexpected)}"
            )
        for portfolio_id, (name, strategy) in STRATEGIES.items():
            expected = (name, strategy, args.initial_capital)
            if portfolio_id in existing:
                if existing[portfolio_id] != expected:
                    raise ValueError(
                        f"existing portfolio {portfolio_id} conflicts with requested initialization"
                    )
            else:
                store.create_portfolio(portfolio_id, name, strategy, args.initial_capital)
        print(f"Initialized paper-only SQLite database: {database}")
        print(f"Created independent portfolios (three strategies plus the SPY benchmark portfolio) with ${args.initial_capital:,.2f} each.")
        return

    if args.command == "migrate":
        backup_dir = args.backup_dir or (resolve_path(real_config, "backup_dir") if real_config
                                         else database.parent / "backups")
        print(json.dumps(PaperStore.migrate(database, args.env, backup_dir), default=str))
        return
    if args.command == "demo":
        if args.env != "synthetic":
            parser.error("the demo only runs with --env synthetic")
        try:
            from .paper_synthetic import run_demo
        except ImportError:
            from paper_synthetic import run_demo
        print(json.dumps(run_demo(database, args.symbols), indent=2, default=str))
        return
    store = PaperStore(database)
    if store.environment() != args.env:
        parser.error(f"{database} is a {store.environment()!r} database; pass --env {store.environment()}")
    if args.command in {"check-filings", "run-signal", "tick", "scheduler"}:
        spec = args.adapters or ("paper_live:real_adapters" if args.env == "real" else None)
        if spec is None:
            parser.error("--adapters is required for the synthetic environment")
        cycle_preparer = None
        data_refresher = None
        if args.env == "real" and args.command in {"tick", "scheduler"}:
            try:
                from . import benchmark_weights
                from .paper_live import prepare_next_real_cycle
            except ImportError:
                import benchmark_weights  # type: ignore[no-redef]
                from paper_live import prepare_next_real_cycle  # type: ignore[no-redef]
            cycle_preparer = lambda: prepare_next_real_cycle(store, real_config)
            data_refresher = lambda: benchmark_weights.fetch_if_stale()
        runner = PaperRunner(
            store,
            load_adapters(spec),
            RunnerConfig(),
            cycle_preparer=cycle_preparer,
            data_refresher=data_refresher,
        )
        if args.command == "check-filings":
            with exclusive_lock(store.path):
                print(json.dumps(runner.check_filings(args.cycle_key), default=str))
        elif args.command == "run-signal":
            with exclusive_lock(store.path):
                print(json.dumps({"result": runner.run_signal(args.cycle_key)}))
        elif args.command == "tick":
            print(json.dumps(runner.tick(), default=str))
        else:
            runner.serve_forever(args.interval_seconds)
        return
    if args.command == "backup":
        destination = args.output.expanduser().resolve()
        if destination == store.path:
            raise ValueError("backup output must differ from the source database")
        if destination.exists():
            raise FileExistsError(f"backup output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with sqlite3.connect(store.path.as_uri() + "?mode=ro", uri=True) as source:
                with sqlite3.connect(temporary) as target:
                    source.backup(target)
            PaperStore(temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"Created and verified backup: {destination}")
        return
    if args.command == "dashboard":
        serve(database, host=args.host, port=args.port, validation_manifest=_validation_manifest(real_config))
        return
    if args.command == "generate-cycle":
        if args.env != "real":
            parser.error("generate-cycle derives reports from the real universe; use --env real")
        try:
            from .paper_live import create_real_cycle
        except ImportError:
            from paper_live import create_real_cycle
        print(json.dumps(create_real_cycle(store, real_config, args.cycle_key, args.quarter_end,
                                           args.deadline_at), indent=2))
        return
    if args.command == "correct-periods":
        if args.env != "real":
            parser.error("correct-periods needs --env real")
        try:
            from .paper_live import correct_expected_periods
        except ImportError:
            from paper_live import correct_expected_periods
        print(json.dumps(correct_expected_periods(store, real_config, args.cycle_key, args.quarter_end, args.apply),
                         indent=2))
        return
    if args.command == "create-cycle":
        keys, metadata = _load_expected_filings(args.expected_filings)
        strategy_id = STRATEGIES[args.portfolio][1]
        cycle_id = store.create_cycle(
            args.portfolio,
            args.cycle_key,
            strategy_id,
            rebalance_policy=args.policy,
            deadline_at=args.deadline_at,
        )
        store.register_expected_filings(cycle_id, keys, filing_metadata=metadata)
        print(json.dumps({"cycle_id": cycle_id, "filing_status": store.filing_status(cycle_id).value}))
        return
    if args.command == "update-filing":
        status = store.set_filing_result(
            args.cycle_id,
            args.filing_key,
            received=args.received,
            facts_ready=args.facts_ready,
            valid=not args.invalid,
            filed_at=args.filed_at,
            first_seen_at=args.first_seen_at,
            source=args.source,
            error_reason=args.error_reason,
        )
        print(json.dumps({"filing_status": status.value}))
        return
    if args.command == "set-model-status":
        store.set_cycle_model_status(args.cycle_id, args.status, reason=args.reason)
        print(json.dumps({"model_status": args.status, "cycle_id": args.cycle_id}))
        return
    if args.command == "status":
        print(json.dumps(dashboard_snapshot(database, _validation_manifest(real_config)), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    run_cli()
