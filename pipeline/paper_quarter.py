"""Manual CLI for paper_fixed_lag target-portfolio packages (no network, no orders, no scheduler).

  python -m pipeline.paper_quarter schedule [--cycle 2026Q3]
  python -m pipeline.paper_quarter research --signal-result S --instrument-map M --output DIR
  python -m pipeline.paper_quarter regular  --signal-result S --instrument-map M --output DIR
  python -m pipeline.paper_quarter validate --package P [--strategy NAME] [--ledger L]
  python -m pipeline.paper_quarter claim    --package P --strategy NAME --ledger L --consumer ID
  python -m pipeline.paper_quarter from-history --execution-dir D --cycle 2026Q2 --output S
  python -m pipeline.paper_quarter from-signal  --execution-dir D --output S

Exit codes: 0 ok/handoff allowed, 2 invalid input/package or timing refusal, 3 package intact
but not eligible for handoff.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from .target_portfolio import (
    MODE_HISTORICAL, MODE_REGULAR, STRATEGIES, ProcessedLedger, TargetPortfolioError, TimingError,
    claim_target_portfolio, cycle_timing, export_package, inspect_package, iso_utc, list_packages,
    load_target_portfolio, parse_utc, signal_result_from_history_execution,
    signal_result_from_signal_execution, utc_now, _tz,
)

RECOMMENDED_COMPUTE_BUFFER = timedelta(hours=4)


def _local(moment: datetime) -> dict:
    return {"utc": iso_utc(moment),
            "america_new_york": moment.astimezone(_tz("America/New_York")).isoformat(),
            "europe_berlin": moment.astimezone(_tz("Europe/Berlin")).isoformat()}


def run_window(cycle_key: str) -> dict:
    timing = cycle_timing(cycle_key)
    cutoff = parse_utc(timing["information_cutoff_utc"], "cutoff")
    execution = parse_utc(timing["scheduled_execution_at_utc"], "execution")
    return {
        "cycle_key": cycle_key, "execution_date": timing["execution_date"],
        "information_cutoff": _local(cutoff), "scheduled_execution": _local(execution),
        "start_not_before": _local(cutoff),
        "recommended_start_by": _local(execution - RECOMMENDED_COMPUTE_BUFFER),
        "valid_until": _local(execution),
        "rule": "Start after the information cutoff and finish before scheduled execution; "
                "a missed run is not backdated.",
    }


def _candidates(now: datetime) -> list[str]:
    current = pd.Period(pd.Timestamp(now).tz_convert("UTC").tz_localize(None), freq="Q-DEC")
    return [str(p) for p in pd.period_range(current - 4, current + 1, freq="Q-DEC")]


def active_cycle(now: datetime) -> str | None:
    for cycle in _candidates(now):
        timing = cycle_timing(cycle)
        if (parse_utc(timing["information_cutoff_utc"], "c") <= now
                < parse_utc(timing["scheduled_execution_at_utc"], "e")):
            return cycle
    return None


def next_cycle(now: datetime) -> str:
    for cycle in _candidates(now):
        if parse_utc(cycle_timing(cycle)["information_cutoff_utc"], "c") > now:
            return cycle
    raise TargetPortfolioError("No upcoming cycle found")


def _print(doc) -> None:
    print(json.dumps(doc, indent=2, sort_keys=True, default=str))


def _signal(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TargetPortfolioError(f"Cannot read signal result {path}: {exc}") from exc


def _report_export(package: Path, now: datetime, **extra) -> int:
    report = inspect_package(package, now=now)
    _print({"published": str(package), **report.as_dict(), **extra})
    return 0


def main(argv: list[str] | None = None, *, now: Callable[[], datetime] = utc_now) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("schedule", help="show cutoff, execution and run window")
    show.add_argument("--cycle")
    for name in ("research", "regular"):
        command = commands.add_parser(name)
        command.add_argument("--signal-result", type=Path, required=True)
        command.add_argument("--instrument-map", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    commands.choices["regular"].add_argument(
        "--allow-stale-diagnostic", action="store_true",
        help="publish a late run as STALE (never a handoff package) instead of refusing")
    for name in ("validate", "claim"):
        command = commands.add_parser(name)
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--strategy", choices=STRATEGIES, required=(name == "claim"))
        command.add_argument("--ledger", type=Path, required=(name == "claim"))
        command.add_argument("--accept-unvalidated-model", action="store_true")
    commands.choices["claim"].add_argument("--consumer", required=True)
    history = commands.add_parser("from-history", help="adapt a history-executor cycle (historical only)")
    history.add_argument("--execution-dir", type=Path, required=True,
                         help="run root containing execution_summary.json, primary/ and replay/")
    history.add_argument("--cycle", required=True)
    history.add_argument("--output", type=Path, required=True)
    listing = commands.add_parser("list", help="list published packages (read-only, for the dashboard)")
    listing.add_argument("--root", type=Path, required=True)
    listing.add_argument("--ledger", type=Path)
    adapted = commands.add_parser("from-signal", help="adapt paper_signal_execution output")
    adapted.add_argument("--execution-dir", type=Path, required=True,
                         help="sealed run root containing artifact_manifest.json")
    adapted.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    moment = now()
    try:
        if args.command == "schedule":
            cycle = args.cycle or active_cycle(moment) or next_cycle(moment)
            _print({"now": _local(moment), "active_cycle": active_cycle(moment), **run_window(cycle)})
            return 0
        if args.command == "research":
            package = export_package(_signal(args.signal_result), args.instrument_map, args.output,
                                     mode=MODE_HISTORICAL, now=lambda: moment)
            return _report_export(package, moment)
        if args.command == "regular":
            signal = _signal(args.signal_result)
            active = active_cycle(moment)
            requested = signal.get("cycle_key")
            if active != requested and not args.allow_stale_diagnostic:
                window = run_window(requested) if isinstance(requested, str) else None
                _print({"refused": "Regular run outside its window", "now": _local(moment),
                        "active_cycle": active, "signal_cycle": requested, "signal_window": window})
                return 2
            package = export_package(signal, args.instrument_map, args.output, mode=MODE_REGULAR,
                                     now=lambda: moment, allow_stale=args.allow_stale_diagnostic)
            return _report_export(package, moment, window=run_window(str(requested)))
        if args.command in ("from-history", "from-signal"):
            doc = (signal_result_from_history_execution(args.execution_dir, args.cycle)
                   if args.command == "from-history"
                   else signal_result_from_signal_execution(args.execution_dir))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "x", encoding="utf-8") as handle:
                json.dump(doc, handle, indent=2, sort_keys=True)
            _print({"written": str(args.output), "historical_only": doc["historical_only"]})
            return 0
        ledger = ProcessedLedger(args.ledger) if args.ledger else None
        if args.command == "list":
            _print(list_packages(args.root, now=moment, ledger=ledger))
            return 0
        options = {"now": moment, "ledger": ledger, "accept_unvalidated_model": args.accept_unvalidated_model}
        if args.command == "validate":
            report = inspect_package(args.package, **options)
            result = report.as_dict()
            if args.strategy:
                try:
                    load_target_portfolio(args.package, args.strategy, **options)
                    result["strategy_handoff_allowed"] = {args.strategy: True}
                except TargetPortfolioError as exc:
                    result["strategy_handoff_allowed"] = {args.strategy: False, "reason": str(exc)}
                    _print(result)
                    return 3
            _print(result)
            return 0 if result["handoff_allowed"] or args.strategy else 3
        options.pop("ledger")
        assert ledger is not None
        portfolio, receipt = claim_target_portfolio(args.package, args.strategy, ledger,
                                                    consumer=args.consumer, **options)
        _print({"claimed": str(receipt), "strategy": portfolio.strategy, "securities": len(portfolio.positions)})
        return 0
    except TimingError as exc:
        _print({"refused": str(exc), "now": _local(moment)})
        return 2
    except TargetPortfolioError as exc:
        _print({"error": str(exc)})
        return 2 if args.command != "claim" else 3


if __name__ == "__main__":
    sys.exit(main())
