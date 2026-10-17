"""Manual acquire/qualify/freeze/calculate/export/import entry point; never places orders."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .fixed_lag_research import write_json
from .historical_evidence import EvidenceStore
from .paper_fixed_lag import schedule
from .paper_quarter import active_cycle, next_cycle
from .paper_signal_execution import FILES, execute_signal, verify_signal_bundle
from .publication_provenance import checked_bytes, timestamp, verify_signal_availability
from .return_comparison import sha256
from .target_portfolio import (
    MODE_HISTORICAL, MODE_REGULAR, TargetPortfolioError, export_package, inspect_package,
    signal_result_from_signal_execution,
)


def run(plan_path: Path, output: Path, source_root: Path, *, cycle: str | None = None,
        historical: bool = False) -> dict:
    """Consume a reviewed data-source plan, never a manually authored signal JSON.

    Source plans locate versioned data tables and raw evidence; they are not assertions that
    arbitrary SEC aggregates are a qualified training panel. Unsupported source transformations
    remain explicit qualification failures, not successful empty portfolio packages.
    """
    plan = json.loads(plan_path.read_text())
    if plan.get("format_version") != "manual-paper-source-plan-v1":
        raise ValueError("Expected manual-paper-source-plan-v1")
    started = datetime.now(timezone.utc)
    selected = cycle or active_cycle(started) or next_cycle(started)
    timing = schedule(selected)
    if plan.get("cycle_key") != selected:
        raise ValueError("Source plan does not belong to the selected cycle")
    if not historical and not (timestamp(timing["information_cutoff"]) <= started
                               < timestamp(timing["execution_at"])):
        raise ValueError("Regular manual run is outside its actual execution window; no acquisition started")
    if historical and timestamp(timing["execution_at"]) > started:
        raise ValueError("Cannot reconstruct a future historical execution")
    output = output.resolve()
    source_root = source_root.resolve()
    if output == source_root or source_root.is_relative_to(output):
        raise ValueError("Output must not contain the source root")
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "cycle_key": selected, "started_at_utc": started.isoformat(),
        "historical_only": historical, "status": "ACQUIRING",
        "model_started": False, "historical_validation": "NOT_VALIDATED", "orders_enabled": False,
        "source_plan_sha256": sha256(plan_path),
    }
    try:
        write_json(output / "source_plan.json", plan)
        write_json(output / "workflow.json", state)
        source_files = plan.get("files")
        if (not isinstance(source_files, dict)
                or not (FILES | {"publication_evidence.json", "instrument_map.json"}).issubset(source_files)):
            raise ValueError("Source plan requires all SIGNAL tables, publication_evidence.json and instrument_map.json")
        store = EvidenceStore(output / "acquisition")
        bundle = output / "bundle"
        bundle.mkdir()
        for name, source in source_files.items():
            if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                    or name == "bundle_manifest.json" or not isinstance(source, dict)):
                raise ValueError(f"Invalid source-plan destination: {name}")
            if set(source) == {"cache_path", "sha256"}:
                raw = checked_bytes(source_root, {"path": source["cache_path"], "sha256": source["sha256"]})
                path = store.copy_cache(source_root / source["cache_path"], name)
                if path.read_bytes() != raw:
                    raise ValueError(f"Cache changed during acquisition: {name}")
            elif set(source) == {"url", "sha256"}:
                path = store.fetch(name, source["url"])
                if sha256(path) != source["sha256"]:
                    raise ValueError(f"Downloaded source differs from reviewed version: {name}")
            else:
                raise ValueError(f"Source {name} needs cache_path or HTTPS url plus reviewed SHA-256")
            destination = output / "instrument_map.json" if name == "instrument_map.json" else bundle / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        evidence_files = {str(p.relative_to(bundle)): sha256(p) for p in bundle.rglob("*")
                          if p.is_file() and str(p.relative_to(bundle)) not in FILES}
        write_json(bundle / "bundle_manifest.json", {
            "schema_version": 1, "scope": "SIGNAL", "review_status": "SOURCE_REVIEWED",
            "files": {name: sha256(bundle / name) for name in FILES}, "evidence_files": evidence_files})
        state["status"] = "QUALIFYING"
        write_json(output / "workflow.json", state)
        verify_signal_bundle(bundle)
        qualification = verify_signal_availability(bundle, timing["information_cutoff"])
        write_json(output / "publication_check.json", qualification)
        state.update(status="CALCULATING", model_started=True)
        write_json(output / "workflow.json", state)
        execution = execute_signal(bundle, output / "signal_run", selected)
        signal = signal_result_from_signal_execution(output / "signal_run")
        write_json(output / "signal_result.json", signal)
        package = export_package(
            signal, output / "instrument_map.json", output / "packages",
            mode=MODE_HISTORICAL if historical else MODE_REGULAR)
        report = inspect_package(package)
        complete = all(item.status == "COMPLETE" for item in report.strategies.values())
        state.update(
            status="COMPLETED" if complete else "PARTIAL_STRATEGY_ERRORS",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            model_replay_identical=execution["model_replay_identical"], package=str(package),
            independent_import=report.as_dict(), data_qualification="PUBLICATION_EVIDENCE_CHECKED",
            current_handoff_allowed=report.as_dict()["handoff_allowed"],
            strategy_results={name: {"status": result.status, "error": result.error}
                              for name, result in report.strategies.items()})
        write_json(output / "workflow.json", state)
        return state
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        state.update(status="FAILED" if state["model_started"] else "BLOCKED_BEFORE_MODEL",
                     failed_stage=state["status"], error=f"{type(exc).__name__}: {exc}",
                     completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "workflow.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycle")
    parser.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.source_plan, args.output, args.source_root,
                     cycle=args.cycle, historical=args.historical)
    except (OSError, ValueError, RuntimeError, KeyError, TargetPortfolioError) as exc:
        print(json.dumps({"status": "REFUSED_OR_BLOCKED", "error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))
    if result["status"] != "COMPLETED":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
