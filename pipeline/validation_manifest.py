"""Build a historical-validation manifest from what is actually reproducible on disk.

    python validation_manifest.py build [--out FILE]

The manifest records code version, configuration, data provenance with hashes, periods, and every evidence
category with an explicit verified flag. Categories that are not proven stay ``verified: false``; the dashboard
recomputes the status and a manifest alone never yields VALIDATED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from . import model_coverage as mc
except ImportError:  # pragma: no cover
    import model_coverage as mc  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
DATA_FILES = {
    "raw_prices": "cache/prices/SPY.csv", "targets": "Data/live/fundamental_live_table.csv",
    "corporate_actions": "cache/sp500_changes.json", "benchmark_values": "cache/prices/SPY.csv",
    "benchmark_metadata": "config/real_paper.json",
}
UNVERIFIED = {
    "raw_price_basis": "Price cache stores close/adj_close only; raw (split-only) basis and corporate-action factors are not archived.",
    "corporate_actions": "No archived, hash-verified corporate-action history for the validation period (e.g. CTVA/VYLR spin-off factor missing).",
    "historical_membership": "Membership snapshots exist (cache/sp500_historical_snapshots.csv) but completeness against the "
                             "index provider was not verified.",
    "point_in_time_signal_and_labels": "Historical sector panels originate from the paper's fundamentals tables; "
                                       "per-row availability timestamps were not independently verified, and a walk-forward run "
                                       "with an untouched final test window has not been executed in this repository.",
    "benchmark_provenance": "SPY adj_close is a total-return approximation, not an index total-return series with provenance.",
}


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def build(out: Path) -> dict:
    config = json.loads((ROOT / "config" / "real_paper.json").read_text())
    spy = pd.read_csv(ROOT / "cache/prices/SPY.csv", parse_dates=["Date"])
    live = ROOT / "Data/live/fundamental_live_table.csv"
    periods = {"benchmark_price_history": [spy["Date"].min().date().isoformat(), spy["Date"].max().date().isoformat()]}
    if live.is_file():
        dates = pd.read_csv(live, usecols=["datadate", "tradedate"], parse_dates=["datadate", "tradedate"])
        periods["feature_rows_datadate"] = [dates["datadate"].min().date().isoformat(),
                                            dates["datadate"].max().date().isoformat()]
    manifest = {
        "mode": "historical_backtest", "generated_at": datetime.now(timezone.utc).isoformat(),
        "code": mc._code_hash(None), "configuration": {"model": config["model"], "readiness_rule": config["readiness_rule"],
                                                       "benchmark": config["benchmark"]},
        "input_sha256": {k: _sha(ROOT / v) for k, v in DATA_FILES.items()},
        "data_origin": {k: v for k, v in DATA_FILES.items()}, "periods": periods,
        "temporal_availability": "Rows carry datadate/tradedate; SEC-derived live rows use acceptance timestamps "
                                 "(q-derive-v2). Historical Compustat-era rows: availability not verified.",
        "results": None, "results_note": "No walk-forward validation result exists; nothing is reported.",
        "validation_evidence": {k: {"verified": False, "reason": r} for k, r in UNVERIFIED.items()},
        "limitations": list(UNVERIFIED.values()) + ["The 50% feature-coverage threshold is provisional.",
                                                   "Final test window must not be used for threshold selection."],
        "status_note": "A manifest does not set VALIDATED; the dashboard recomputes the status from evidence files.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--out", default=str(ROOT / "state" / "validation" / "validation_manifest.json"))
    args = parser.parse_args()
    build(Path(args.out))
    try:
        from .paper_dashboard import evaluate_backtest_validation
    except ImportError:
        from paper_dashboard import evaluate_backtest_validation  # type: ignore
    print(json.dumps(evaluate_backtest_validation(args.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
