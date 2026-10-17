"""Sourced instrument registry and strict, offline total-return comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .market_calendar import nyse_sessions, session_close_time

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = Path("state/research/comparison")
SCHEMA_VERSION = 1
REQUIRED = {
    "instrument_id", "name", "instrument_type", "identifiers", "share_class", "exchange",
    "currency", "return_type", "dividend_tax_basis", "reinvestment_rule", "cost_basis",
    "source", "source_url", "valuation_convention", "calendar", "review_status",
    "evidence_files", "observed_at",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_isin(value: str) -> bool:
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", value):
        return False
    expanded = "".join(str(ord(c) - 55) if c.isalpha() else c for c in value)
    total = 0
    for i, digit in enumerate(reversed(expanded)):
        number = int(digit) * (2 if i % 2 else 1)
        total += number // 10 + number % 10
    return total % 10 == 0


def validate_metadata(doc: dict) -> None:
    missing = REQUIRED - set(doc)
    if missing:
        raise ValueError(f"Missing comparison metadata: {sorted(missing)}")
    for key in REQUIRED - {"identifiers", "evidence_files"}:
        if not isinstance(doc[key], str) or not doc[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", doc["instrument_id"]):
        raise ValueError("instrument_id must be a safe, unique instrument/class identifier")
    if doc["instrument_type"] not in {"stock", "etf", "index", "strategy", "paper_reference"}:
        raise ValueError("Unsupported instrument_type")
    if doc["currency"] != "USD":
        raise ValueError("Only USD imports supported; EUR requires independently sourced historical FX conversion")
    if doc["return_type"] not in {"gross_total_return", "etf_total_return", "strategy_total_return"}:
        raise ValueError("Price series and undocumented adj_close are not verified total returns")
    if doc["dividend_tax_basis"] != "before_personal_tax":
        raise ValueError("Comparison requires before_personal_tax; internal fund taxes must be documented in cost_basis")
    if doc["reinvestment_rule"] not in {"ex_date", "pay_date", "cash_until_rebalance"}:
        raise ValueError("reinvestment_rule must be ex_date, pay_date or cash_until_rebalance")
    if doc["review_status"] != "SOURCE_REVIEWED":
        raise ValueError("A reviewed source definition is required; data import is not a historical strategy validation")
    if doc["calendar"] not in {"NYSE", "EXPLICIT"}:
        raise ValueError("calendar must be NYSE or EXPLICIT")
    if not doc["source_url"].startswith("https://"):
        raise ValueError("Source URL must identify the actual HTTPS data/definition source")
    stamp = pd.Timestamp(doc["observed_at"])
    if stamp.tzinfo is None or stamp > pd.Timestamp.now(tz="UTC"):
        raise ValueError("observed_at must be a genuine, timezone-aware nonfuture observation time")
    identifiers = doc["identifiers"]
    if not isinstance(identifiers, dict) or not identifiers:
        raise ValueError("Explicit instrument identifiers required; no heuristic identifier conversion")
    for kind, values in identifiers.items():
        if kind not in {"ticker", "isin", "wkn", "index_id", "strategy_id"}:
            raise ValueError(f"Unsupported identifier type {kind}")
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"Invalid identifier list for {kind}")
        for value in values:
            if kind == "isin" and not valid_isin(value):
                raise ValueError(f"Invalid ISIN checksum: {value}")
            if kind == "wkn" and not re.fullmatch(r"[A-Z0-9]{6}", value):
                raise ValueError("WKN must be six explicitly sourced alphanumeric characters")
    if not isinstance(doc["evidence_files"], list) or not doc["evidence_files"]:
        raise ValueError("Source definition/review evidence files with SHA256 are required")


def read_points(path: Path, metadata: dict) -> list[dict]:
    frame = pd.read_csv(path, dtype={"date": str, "valued_at": str})
    if set(frame.columns) != {"date", "value", "valued_at"} or frame.empty:
        raise ValueError("CSV must have exactly date,value,valued_at and nonempty rows")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    stamps = pd.to_datetime(frame["valued_at"], utc=True, errors="raise")
    if frame["valued_at"].map(lambda value: pd.Timestamp(value).tzinfo is None).any():
        raise ValueError("Every valuation timestamp must have an explicit timezone")
    if dates.duplicated().any() or not dates.is_monotonic_increasing or stamps.duplicated().any():
        raise ValueError("Dates and valuation timestamps must be unique and strictly ascending")
    if not stamps.is_monotonic_increasing:
        raise ValueError("Valuation timestamps must be ascending")
    if (stamps.dt.date < dates.dt.date).any() or (stamps.dt.date > (dates + pd.Timedelta(days=1)).dt.date).any():
        raise ValueError("Valuation timestamp does not match its valuation date")
    if (stamps > pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("Future valuations are not allowed")
    values = pd.to_numeric(frame["value"], errors="raise")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Series values must be finite and strictly positive")
    if metadata["calendar"] == "NYSE":
        expected = nyse_sessions(dates.iloc[0], dates.iloc[-1])
        if list(dates) != list(expected):
            missing = expected.difference(dates)
            extra = pd.DatetimeIndex(dates).difference(expected)
            raise ValueError(f"Missing/non-session NYSE valuations; missing={list(missing[:5])}, extra={list(extra[:5])}")
        if metadata["valuation_convention"] == "NYSE_close_UTC":
            closes = pd.DatetimeIndex([session_close_time(day).tz_convert("UTC") for day in dates])
            if not pd.DatetimeIndex(stamps).equals(closes):
                raise ValueError("Declared NYSE close timestamps disagree with actual session closes")
    else:
        declared = metadata.get("expected_sessions")
        if declared != frame["date"].tolist():
            raise ValueError("EXPLICIT calendar needs expected_sessions matching every date; gaps may not be hidden")
    return [{"date": d, "value": float(v), "valued_at": stamp.isoformat()}
            for d, v, stamp in zip(frame["date"], values, stamps, strict=True)]


def import_series(root: Path, csv_path: Path, metadata_path: Path) -> Path:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    evidence = {}
    for item in metadata["evidence_files"]:
        path = (metadata_path.parent / item["path"]).resolve()
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Source evidence hash mismatch: {path.name}")
        evidence[path.name] = digest
    points = read_points(csv_path, metadata)
    history = []
    if metadata.get("weight_history_file"):
        path = (metadata_path.parent / metadata["weight_history_file"]).resolve()
        if sha256(path) != metadata.get("weight_history_sha256"):
            raise ValueError("Weight history hash mismatch")
        history = json.loads(path.read_text())
        validate_weight_history(history, points)
    directory = root / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{metadata['instrument_id']}.json"
    if output.exists():
        raise FileExistsError(f"Instrument already imported; use an explicitly versioned instrument_id: {output.name}")
    record = {
        "schema_version": SCHEMA_VERSION, "metadata": metadata, "points": points,
        "weight_history": history,
        "csv_sha256": sha256(csv_path), "metadata_sha256": sha256(metadata_path),
        "evidence_sha256": evidence, "imported_at": datetime.now(timezone.utc).isoformat(),
        "historical_strategy_validation": "NOT_VALIDATED",
    }
    raw = json.dumps(record, ensure_ascii=True, sort_keys=True, allow_nan=False).encode()
    with output.open("xb") as stream:
        stream.write(raw)
    with output.with_suffix(".sha256").open("x") as stream:
        stream.write(hashlib.sha256(raw).hexdigest() + "\n")
    return output


def validate_weight_history(history: list, points: list[dict]) -> None:
    if not isinstance(history, list):
        raise ValueError("Historical weights must be a dated list")
    dates = {p["date"] for p in points}
    observed = set()
    for snapshot in history:
        if snapshot["date"] not in dates or snapshot["date"] in observed:
            raise ValueError("Weight date must be unique and match a portfolio valuation")
        observed.add(snapshot["date"])
        positions = snapshot["positions"]
        if len({p["security_id"] for p in positions}) != len(positions):
            raise ValueError("Duplicate security in historical weight snapshot")
        for position in positions:
            for field in ("target_weight", "actual_weight", "benchmark_weight"):
                value = position.get(field)
                if value is not None and (not isinstance(value, (float, int))
                                          or not math.isfinite(value) or not 0 <= value <= 1 + 1e-7):
                    raise ValueError(f"Invalid historical {field}")
            if position.get("benchmark_weight") is not None and not position.get("benchmark_source"):
                raise ValueError("Historical benchmark weight needs its own dated source")
        for field in ("actual_weight", "target_weight"):
            if sum(p.get(field) or 0 for p in positions) > 1 + 1e-7:
                raise ValueError(f"Historical {field} exceeds full investment")


def load_registry(root: Path) -> dict:
    result = {"instruments": [], "errors": [], "provider": "local_source_reviewed_registry",
              "external_search_available": False}
    for path in sorted((root / DIRECTORY).glob("*.json")):
        try:
            if sha256(path) != path.with_suffix(".sha256").read_text().strip():
                raise ValueError("Imported record hash mismatch")
            doc = json.loads(path.read_text())
            if doc["schema_version"] != SCHEMA_VERSION:
                raise ValueError("Unsupported imported series schema")
            validate_metadata(doc["metadata"])
            validate_weight_history(doc.get("weight_history", []), doc["points"])
            meta = doc["metadata"]
            result["instruments"].append({
                **meta, "points": doc["points"], "start": doc["points"][0]["date"],
                "weight_history": doc.get("weight_history", []),
                "end": doc["points"][-1]["date"], "csv_sha256": doc["csv_sha256"],
                "imported_at": doc["imported_at"],
                "historical_strategy_validation": doc["historical_strategy_validation"],
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["errors"].append(f"{path.name}: {exc}")
    if not result["instruments"]:
        result["errors"].append(
            "No source-reviewed USD total-return series imported. Local adj_close files have no verified "
            "dividend/tax/reinvestment definition and are deliberately not promoted to total-return comparisons."
        )
    return result


def search_instruments(registry: dict, query: str) -> list[dict]:
    query = query.strip().casefold()
    if not query:
        return registry["instruments"]
    return [item for item in registry["instruments"] if
            query in (item["name"] + " " + item["instrument_id"]).casefold()
            or any(query == value.casefold() for values in item["identifiers"].values() for value in values)]


def compare(instruments: list[dict], start: str, end: str) -> dict:
    if not instruments:
        raise ValueError("Select at least one instrument")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Start must not exceed end")
    keys = ("currency", "dividend_tax_basis", "reinvestment_rule", "valuation_convention")
    for key in keys:
        if len({instrument[key] for instrument in instruments}) != 1:
            raise ValueError(f"Incompatible comparison definition: {key}")
    maps = [{p["date"]: p for p in i["points"] if start <= p["date"] <= end} for i in instruments]
    dates = sorted(set.intersection(*(set(m) for m in maps)))
    if len(dates) < 2:
        raise ValueError("Fewer than two common, evidenced valuation dates")
    if any(len({m[day]["valued_at"] for m in maps}) != 1 for day in dates):
        raise ValueError("Different valuation timestamps; use a sourced common-close conversion")
    series = []
    for instrument, mapping in zip(instruments, maps, strict=True):
        values = np.array([mapping[d]["value"] for d in dates], dtype=float)
        normalized = values / values[0] * 100
        drawdown = values / np.maximum.accumulate(values) - 1
        returns = values[1:] / values[:-1] - 1
        years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
        series.append({
            "instrument_id": instrument["instrument_id"],
            "points": [{"date": d, "index100": float(n), "return": float(n / 100 - 1),
                        "drawdown": float(dd), "value": float(v)}
                       for d, n, dd, v in zip(dates, normalized, drawdown, values, strict=True)],
            "metrics": {"return": float(values[-1] / values[0] - 1),
                        "cagr": float((values[-1] / values[0]) ** (1 / years) - 1),
                        "maximum_drawdown": float(drawdown.min()),
                        "observations": len(dates),
                        "observation_return_std": float(returns.std(ddof=1)) if len(returns) > 1 else None},
        })
    return {"start": dates[0], "end": dates[-1], "dates": dates, "series": series,
            "requested_start": start, "requested_end": end,
            "excluded_noncommon_dates": len(set.union(*(set(m) for m in maps)) - set(dates)),
            "normalization": "all series start at 100 on the same first common valuation",
            "missing_price_fill": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--search")
    args = parser.parse_args()
    if args.csv or args.metadata:
        if not args.csv or not args.metadata:
            parser.error("--csv and --metadata must be supplied together")
        print(import_series(ROOT, args.csv, args.metadata))
    else:
        registry = load_registry(ROOT)
        if args.search:
            print(json.dumps([{k: v for k, v in item.items() if k != "points"}
                              for item in search_instruments(registry, args.search)], indent=2))
        else:
            print(json.dumps({"count": len(registry["instruments"]), "errors": registry["errors"]}, indent=2))


if __name__ == "__main__":
    main()
