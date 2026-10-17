"""Versioned target-portfolio packages for the paper_fixed_lag strategy.

A package holds three separate *complete target portfolios* (equal_weight, min_variance,
mean_variance) for one decision cycle. It is never a list of buy/sell orders and contains no
broker or platform logic. See EXECUTION_INTERFACE.md for the contract.

Inputs:
* a signal result (``paper-signal-result-v1``) produced by a signal executor or adapted from
  the historical executor; valuation is not required to export signals;
* a frozen, verified instrument map (``paper-instrument-map-v1``); identities are never guessed
  from tickers.

Packages are published atomically (temporary directory + rename). Import validation is strict
and read-only; marking a package as processed is a separate durable claim (ProcessedLedger).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from .paper_fixed_lag import OPTIMIZATION_VERSION, RULE_VERSION, RULES, optimization_definition, schedule
from .publication_provenance import verify_signal_availability
from .training_vintage import STRICT

PACKAGE_SCHEMA_VERSION = "paper-target-portfolio-package-v1"
CSV_SCHEMA_VERSION = "paper-target-portfolio-csv-v1"
SIGNAL_RESULT_VERSION = "paper-signal-result-v1"
INSTRUMENT_MAP_VERSION = "paper-instrument-map-v1"
PROTOCOL_VERSION = "paper-fixed-lag-handoff-v1"

STRATEGIES = ("equal_weight", "min_variance", "mean_variance")
INTERNAL_STRATEGY_NAMES = {"equal_weight": "equal_weight", "min_var": "min_variance",
                           "mean_var": "mean_variance"}
BASE_CURRENCY = "USD"
WEIGHT_DECIMALS = 12
WEIGHT_QUANTUM = Decimal(1).scaleb(-WEIGHT_DECIMALS)
POSITION_CAP = Decimal(str(RULES.weight_cap))
BOUND_TOLERANCE = Decimal("1e-8")
SUM_TOLERANCE = Decimal("1e-6")
MAX_FILE_BYTES = 20_000_000

KIND_CURRENT = "CURRENT_HANDOFF"
KIND_HISTORICAL = "HISTORICAL_REPLAY"
KIND_STALE = "STALE"
KIND_DIAGNOSTIC = "DIAGNOSTIC"
PACKAGE_KINDS = (KIND_CURRENT, KIND_HISTORICAL, KIND_STALE, KIND_DIAGNOSTIC)
MODE_HISTORICAL = "historical_replay"
MODE_REGULAR = "regular_quarter"

STATUS_COMPLETE = "COMPLETE"
STATUS_FAILED = "FAILED"
STATUS_BLOCKED_IDENTITY = "BLOCKED_IDENTITY"

ROW_SECURITY = "SECURITY"
ROW_CASH = "CASH"
CSV_COLUMNS = (
    "schema_version", "run_id", "allocation_id", "package_kind", "strategy", "cycle_key",
    "signal_as_of_utc", "scheduled_execution_at_utc", "row_type", "security_id", "isin", "wkn",
    "ticker", "name", "asset_type", "share_class", "sector_code", "sector_name", "currency",
    "target_weight",
)
DATA_CHECK_PASSED = "PASSED"

TARGET_SEMANTICS = [
    "Each CSV is the COMPLETE target portfolio of one strategy, not a list of orders.",
    "target_weight is a fraction of that strategy portfolio's total value: 0.025 means 2.5 percent.",
    "Exactly one CASH row per CSV holds the uninvested remainder in the base currency; "
    "security rows plus CASH sum to 1 within sum_tolerance.",
    "For an account dedicated exclusively to one strategy, a previously held position that is "
    "absent from that strategy's CSV has target weight zero.",
    "That rule must never be applied to positions that belong to other strategies or are unrelated.",
    "equal_weight, min_variance and mean_variance are three separate portfolios, never three "
    "consecutive rebalancings of the same account.",
    "No silent renormalization: missing or untradeable securities are not removed or replaced; "
    "a platform-adapted variant would be a separately documented strategy.",
    "A strategy with status other than COMPLETE has no CSV; its <strategy>.FAILED.json file "
    "states the reason. Weights of another strategy are never substituted.",
    "Reference prices are not part of v1; share quantities must not be derived from research data.",
]

PAPER_REPLICATION_STATUS = "NOT_EXACT_PAPER_REPLICATION"
# Frozen definition parameters; a change requires a new definition version, never silent tuning.
FROZEN_DEFINITION_PARAMETERS = {"risk_free_annual": 0.015, "weight_cap": 0.05}
RISK_FREE_ANNUAL_LOG = math.log1p(FROZEN_DEFINITION_PARAMETERS["risk_free_annual"])
ANNUAL_LOG_UNITS = {
    "expected_return": "annual log return = 4 x predicted quarterly log return",
    "covariance": "252 x covariance of daily log returns over the last 253 NYSE sessions through cutoff",
    "risk_free": f"log1p(0.015) = {RISK_FREE_ANNUAL_LOG!r} (fixed assumption, no historical series)",
}
STRATEGY_DEFINITIONS = {
    "equal_weight": {
        "label": "Gleichgewichtet",
        "definition_version": "equal_weight-1_over_n-v1",
        "definition": "1/n over the selected securities (n >= 20, so each weight <= 5 %); any "
                      "remainder is cash.",
        "covariance_required": False,
    },
    "min_variance": {
        "label": "Minimale Varianz",
        "definition_version": f"min_variance-{OPTIMIZATION_VERSION}",
        "units": {"covariance": ANNUAL_LOG_UNITS["covariance"]},
        "definition": "Long-only 0-5 %, fully invested, minimum annualized variance of daily log "
                      "returns over the last 253 NYSE sessions through the information cutoff (SLSQP).",
        "covariance_required": True,
        "deviation": "The original Python notebook computes min_var via max_sharpe on zero expected "
                     "returns (a bug); this implementation minimizes variance directly.",
    },
    "mean_variance": {
        "label": "Rendite/Risiko – Max-Sharpe",
        "definition_version": f"mean_variance-{OPTIMIZATION_VERSION}",
        "units": ANNUAL_LOG_UNITS,
        "definition": "Long-only 0-5 %, fully invested, maximum Sharpe ratio (tangency) with annual "
                      "log expected returns (4 x predicted quarterly log return), annualized daily "
                      "log covariance and risk-free log(1.015) (SLSQP).",
        "covariance_required": True,
        "deviation": "The paper specifies expected quarterly returns, 1-year daily covariance, 0-5 % "
                     "bounds, full investment and maximization of the Sharpe ratio; the exact original "
                     "parameters are not established. The original notebook also uses max_sharpe. The fixed "
                     "1.5 % risk-free rate is an assumption, not a historical series. Not an exact "
                     "reproduction of the original MATLAB/notebook optimizer; parameters are frozen "
                     "(no tuning on later data).",
    },
}
PERFORMANCE_METRIC_NOTE = (
    "The optimizer objective is the annual-log Max-Sharpe approximation of the optimization definition; "
    "any conventional simple-return Sharpe ratio reported for realized portfolio performance is a "
    "different quantity and must not be presented as the objective value.")
MODEL_DEVIATIONS = [
    "Deterministic scikit-learn estimators with defaults, not the original R caret tuning.",
    "Annual log-return units for expected returns, covariance and risk-free rate.",
    "Fixed 1.5 % annual risk-free rate assumption.",
    "Selection: ceiling(20 % of eligible securities) with security-ID tie-break instead of quantile >= 0.8.",
    "Fixed-lag paper schedule with cutoff at the previous NYSE close and execution at the next fixed "
    "session close is an implementation assumption; the paper does not fix intraday times.",
]

_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_WKN = re.compile(r"^[A-Z0-9]{6}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_ALLOCATION = re.compile(r"^alloc-[0-9a-f]{32}$")
_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")
_WEIGHT_TEXT = re.compile(r"^[0-9]+\.[0-9]{%d}$" % WEIGHT_DECIMALS)


class TargetPortfolioError(ValueError):
    """Input, package or import contract violated; no target portfolio may be used."""


class TimingError(TargetPortfolioError):
    """A regular quarter run was started outside its information/execution window."""


class AlreadyClaimed(TargetPortfolioError):
    """The allocation/strategy pair was already claimed by an execution consumer."""


# ---------------------------------------------------------------- helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise TargetPortfolioError(f"{field_name} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TargetPortfolioError(f"{field_name} is not ISO-8601: {value!r}") from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise TargetPortfolioError(f"{field_name} must carry a UTC offset: {value!r}")
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise TargetPortfolioError("Naive datetimes are not accepted")
    return value.astimezone(timezone.utc).isoformat()


def isin_is_valid(isin: str) -> bool:
    """ISO 6166 format and check digit."""
    if not isinstance(isin, str) or not _ISIN.match(isin):
        return False
    digits = "".join(str(int(char, 36)) for char in isin)
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            value = value - 9 if value > 9 else value
        total += value
    return total % 10 == 0


def format_weight(value: Decimal) -> str:
    return f"{value.quantize(WEIGHT_QUANTUM):.{WEIGHT_DECIMALS}f}"


def _require_text(doc: Mapping, key: str, where: str, *, allow_empty: bool = False) -> str:
    value = doc.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise TargetPortfolioError(f"{where}: {key} must be a{'' if allow_empty else ' nonempty'} string")
    if any(char in value for char in "\r\n\x00"):
        raise TargetPortfolioError(f"{where}: {key} contains control characters")
    return value


def cycle_timing(cycle_key: str) -> dict:
    """Schedule facts derived from the NYSE calendar (America/New_York, DST-aware)."""
    try:
        period = pd.Period(cycle_key, freq="Q-DEC")
    except (ValueError, TypeError) as exc:
        raise TargetPortfolioError(f"Invalid cycle key {cycle_key!r}") from exc
    if str(period) != cycle_key:
        raise TargetPortfolioError(f"Cycle key must be canonical like 2026Q2, got {cycle_key!r}")
    timing = schedule(cycle_key)
    cutoff = parse_utc(timing["information_cutoff"], "information_cutoff")
    execution = parse_utc(timing["execution_at"], "execution_at")
    return {
        "cycle_key": cycle_key, "rule_version": timing["rule_version"],
        "nominal_execution_date": timing["nominal_execution_date"],
        "execution_date": timing["execution_date"],
        "information_cutoff_utc": iso_utc(cutoff), "scheduled_execution_at_utc": iso_utc(execution),
        "information_cutoff_new_york": cutoff.astimezone(_tz("America/New_York")).isoformat(),
        "scheduled_execution_new_york": execution.astimezone(_tz("America/New_York")).isoformat(),
        "earliest_start_utc": iso_utc(cutoff),
        "valid_until_utc": iso_utc(execution),
        "cutoff_assumption": timing["cutoff_assumption"],
        "execution_assumption": timing["execution_assumption"],
    }


def _tz(name: str):
    from zoneinfo import ZoneInfo
    return ZoneInfo(name)


# ---------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Instrument:
    security_id: str
    isin: str
    wkn: str
    ticker: str
    name: str
    asset_type: str
    share_class: str
    sector_code: str
    sector_name: str
    currency: str


@dataclass(frozen=True)
class InstrumentMap:
    raw: bytes
    sha256: str
    frozen_at_utc: str
    instruments: Mapping[str, Instrument]


def load_instrument_map(path: Path) -> InstrumentMap:
    """Frozen verified map security_id -> ISIN/name/class; ticker-only rows are refused."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise TargetPortfolioError(f"Instrument map must be a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise TargetPortfolioError("Instrument map too large")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetPortfolioError(f"Instrument map is not UTF-8 JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("format_version") != INSTRUMENT_MAP_VERSION:
        raise TargetPortfolioError(f"Instrument map format_version must be {INSTRUMENT_MAP_VERSION}")
    frozen = iso_utc(parse_utc(doc.get("frozen_at_utc"), "frozen_at_utc"))
    rows = doc.get("instruments")
    if not isinstance(rows, list) or not rows:
        raise TargetPortfolioError("Instrument map has no instruments")
    instruments, isins = {}, set()
    for index, row in enumerate(rows):
        where = f"instrument[{index}]"
        if not isinstance(row, dict):
            raise TargetPortfolioError(f"{where} must be an object")
        security_id = _require_text(row, "security_id", where)
        isin = _require_text(row, "isin", where)
        if not isin_is_valid(isin):
            raise TargetPortfolioError(f"{where}: invalid ISIN {isin!r}")
        wkn = _require_text(row, "wkn", where, allow_empty=True)
        if wkn and not _WKN.match(wkn):
            raise TargetPortfolioError(f"{where}: invalid WKN {wkn!r}")
        if row.get("verification_status") != "VERIFIED":
            raise TargetPortfolioError(f"{where}: verification_status must be VERIFIED")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(
                isinstance(item, dict) and isinstance(item.get("source"), str) and item["source"].strip()
                and isinstance(item.get("reference"), str) and item["reference"].strip()
                and parse_utc(item.get("retrieved_at_utc"), f"{where}.evidence.retrieved_at_utc")
                for item in evidence):
            raise TargetPortfolioError(f"{where}: evidence with source, reference and retrieved_at_utc required")
        currency = _require_text(row, "currency", where)
        if currency != BASE_CURRENCY:
            raise TargetPortfolioError(f"{where}: only {BASE_CURRENCY} research instruments supported")
        instrument = Instrument(
            security_id=security_id, isin=isin, wkn=wkn,
            ticker=_require_text(row, "ticker", where), name=_require_text(row, "name", where),
            asset_type=_require_text(row, "asset_type", where),
            share_class=_require_text(row, "share_class", where, allow_empty=True),
            sector_code=_require_text(row, "sector_code", where),
            sector_name=_require_text(row, "sector_name", where), currency=currency,
        )
        if security_id in instruments or isin in isins:
            raise TargetPortfolioError(f"{where}: duplicate security_id or ISIN")
        instruments[security_id] = instrument
        isins.add(isin)
    return InstrumentMap(raw=raw, sha256=sha256_bytes(raw), frozen_at_utc=frozen, instruments=instruments)


def _finite_weight(value, where: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetPortfolioError(f"{where}: weight must be a number")
    if not math.isfinite(float(value)):
        raise TargetPortfolioError(f"{where}: weight must be finite")
    return Decimal(repr(float(value)))


def validate_signal_result(doc: Mapping) -> dict:
    """Validate the signal-result contract; returns a normalized copy."""
    if not isinstance(doc, Mapping) or doc.get("format_version") != SIGNAL_RESULT_VERSION:
        raise TargetPortfolioError(f"Signal result format_version must be {SIGNAL_RESULT_VERSION}")
    cycle_key = _require_text(doc, "cycle_key", "signal")
    timing = cycle_timing(cycle_key)
    if doc.get("rule_version") != RULE_VERSION:
        raise TargetPortfolioError(f"Signal rule_version must be {RULE_VERSION}")
    for key, expected in (("information_cutoff_utc", timing["information_cutoff_utc"]),
                          ("scheduled_execution_at_utc", timing["scheduled_execution_at_utc"])):
        if iso_utc(parse_utc(doc.get(key), key)) != expected:
            raise TargetPortfolioError(f"Signal {key} does not match the {cycle_key} schedule ({expected})")
    computed = parse_utc(doc.get("signal_computed_at_utc"), "signal_computed_at_utc")
    if computed < parse_utc(timing["information_cutoff_utc"], "cutoff"):
        raise TargetPortfolioError("Signal computed before its information cutoff")
    if doc.get("base_currency") != BASE_CURRENCY:
        raise TargetPortfolioError(f"Signal base_currency must be {BASE_CURRENCY}")
    for key in ("synthetic_fixture", "historical_only"):
        if not isinstance(doc.get(key), bool):
            raise TargetPortfolioError(f"Signal {key} must be boolean")
    # Rules are required and pinned: results from older/unversioned optimizers are never joined
    # to the current definition by falling back to the code's rule manifest.
    model_rules = doc.get("model_rules")
    if not isinstance(model_rules, Mapping) or model_rules.get("version") != RULE_VERSION:
        raise TargetPortfolioError(f"model_rules must be the {RULE_VERSION} rule manifest of the producing run")
    rules_source = "signal_result"
    if model_rules.get("optimization_version") != OPTIMIZATION_VERSION:
        raise TargetPortfolioError(f"model_rules optimization_version {model_rules.get('optimization_version')!r} "
                                   f"is not {OPTIMIZATION_VERSION!r}; older optimizer results are not joined")
    if model_rules.get("optimization_definition") != optimization_definition():
        raise TargetPortfolioError(f"model_rules optimization_definition differs from {OPTIMIZATION_VERSION}")
    for key, expected in FROZEN_DEFINITION_PARAMETERS.items():
        if model_rules.get(key) != expected:
            raise TargetPortfolioError(f"model_rules {key}={model_rules.get(key)!r} differs from the frozen "
                                       f"strategy definition ({expected}); a new definition version is required")
    restrictions = doc.get("restrictions", [])
    if not isinstance(restrictions, list) or not all(isinstance(r, str) for r in restrictions):
        raise TargetPortfolioError("Signal restrictions must be a list of strings")
    strategies = doc.get("strategies")
    if not isinstance(strategies, Mapping) or set(strategies) != set(STRATEGIES):
        raise TargetPortfolioError(f"Signal strategies must be exactly {list(STRATEGIES)}")
    normalized = {}
    for name in STRATEGIES:
        item = strategies[name]
        if not isinstance(item, Mapping):
            raise TargetPortfolioError(f"Strategy {name} must be an object")
        status = item.get("status")
        if status == STATUS_COMPLETE:
            weights = item.get("weights")
            if not isinstance(weights, Mapping) or not weights or item.get("error") not in (None, ""):
                raise TargetPortfolioError(f"Strategy {name}: COMPLETE needs weights and no error")
            parsed = {}
            for security_id, value in weights.items():
                if not isinstance(security_id, str) or not security_id.strip():
                    raise TargetPortfolioError(f"Strategy {name}: invalid security_id")
                parsed[security_id] = _finite_weight(value, f"{name}.{security_id}")
            normalized[name] = {"status": status, "weights": parsed, "error": None}
        elif status == STATUS_FAILED:
            error = item.get("error")
            if not isinstance(error, str) or not error.strip() or item.get("weights") is not None:
                raise TargetPortfolioError(f"Strategy {name}: FAILED needs an error and weights=null")
            normalized[name] = {"status": status, "weights": None, "error": error}
        else:
            raise TargetPortfolioError(f"Strategy {name}: status must be COMPLETE or FAILED")
    computed_source = doc.get("signal_computed_at_source", "declared")
    if computed_source not in ("declared", "signal.json", "file_mtime"):
        raise TargetPortfolioError("signal_computed_at_source must be declared, signal.json or file_mtime")
    return {
        "cycle_key": cycle_key, "timing": timing,
        "signal_computed_at_utc": iso_utc(computed), "signal_computed_at_source": computed_source,
        "code_id": _require_text(doc, "code_id", "signal"),
        "data_id": _require_text(doc, "data_id", "signal"),
        "config_id": _require_text(doc, "config_id", "signal"),
        "data_check_status": _require_text(doc, "data_check_status", "signal"),
        "historical_validation_status": _require_text(doc, "historical_validation_status", "signal"),
        "synthetic_fixture": doc["synthetic_fixture"], "historical_only": doc["historical_only"],
        "restrictions": list(restrictions), "strategies": normalized,
        "model_rules": dict(model_rules), "model_rules_source": rules_source,
    }


# ---------------------------------------------------------------- building

def _strategy_rows(weights: Mapping[str, Decimal], instruments: InstrumentMap) -> dict:
    missing = sorted(set(weights) - set(instruments.instruments))
    if missing:
        return {"status": STATUS_BLOCKED_IDENTITY, "rows": None,
                "error": f"No frozen verified instrument identity for security_id(s): {missing}"}
    rows, total = [], Decimal(0)
    for security_id in sorted(weights):
        weight = weights[security_id]
        if weight < -BOUND_TOLERANCE or weight > POSITION_CAP + BOUND_TOLERANCE:
            return {"status": STATUS_FAILED, "rows": None,
                    "error": f"Weight {weight} of {security_id} outside [0, {POSITION_CAP}]"}
        weight = min(max(weight, Decimal(0)), POSITION_CAP).quantize(WEIGHT_QUANTUM)
        total += weight
        rows.append((instruments.instruments[security_id], weight))
    cash = Decimal(1) - total
    if abs(cash) <= SUM_TOLERANCE and cash < 0:
        cash = Decimal(0)
    if cash < 0:
        return {"status": STATUS_FAILED, "rows": None,
                "error": f"Security weights sum to {total} > 1; no renormalization applied"}
    if rows and all(weight == 0 for _, weight in rows):
        return {"status": STATUS_FAILED, "rows": None, "error": "All security weights are zero"}
    return {"status": STATUS_COMPLETE, "rows": rows, "cash": cash.quantize(WEIGHT_QUANTUM), "error": None}


def _allocation_payload(cycle_key: str, timing: dict, ids: Mapping[str, str], instrument_sha: str,
                        strategies: Mapping[str, dict]) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION, "csv_schema_version": CSV_SCHEMA_VERSION,
        "rule_version": RULE_VERSION, "cycle_key": cycle_key,
        "signal_as_of_utc": timing["information_cutoff_utc"],
        "scheduled_execution_at_utc": timing["scheduled_execution_at_utc"],
        "code_id": ids["code_id"], "data_id": ids["data_id"], "config_id": ids["config_id"],
        "instrument_map_sha256": instrument_sha,
        "optimization_version": OPTIMIZATION_VERSION,
        "optimization_definition_sha256": sha256_bytes(_canonical(optimization_definition())),
        "strategy_definition_versions": {name: STRATEGY_DEFINITIONS[name]["definition_version"]
                                         for name in STRATEGIES},
        "strategies": {name: strategies[name] for name in STRATEGIES},
    }


def allocation_id_for(payload: Mapping) -> str:
    return "alloc-" + sha256_bytes(_canonical(payload))[:32]


def classify(mode: str, generated_at: datetime, timing: dict, signal: dict, *, allow_stale: bool) -> tuple[str, list[str]]:
    reasons = []
    cutoff = parse_utc(timing["information_cutoff_utc"], "cutoff")
    execution = parse_utc(timing["scheduled_execution_at_utc"], "execution")
    if mode == MODE_HISTORICAL:
        return KIND_HISTORICAL, ["Historical research/replay package; never a current trade signal."]
    if mode != MODE_REGULAR:
        raise TargetPortfolioError(f"Unknown mode {mode!r}")
    if generated_at < cutoff:
        raise TimingError(f"Too early: {timing['cycle_key']} information cutoff is "
                          f"{timing['information_cutoff_utc']}; start after it")
    if generated_at >= execution:
        if not allow_stale:
            raise TimingError(f"Too late: {timing['cycle_key']} execution was "
                              f"{timing['scheduled_execution_at_utc']}; missed runs are not backdated")
        return KIND_STALE, ["Generated after scheduled execution; stale, not a paper signal."]
    if signal["historical_only"]:
        reasons.append("Signal source is historical-only (e.g. history executor adapter).")
    if signal["data_check_status"] != DATA_CHECK_PASSED:
        reasons.append(f"Data check status {signal['data_check_status']!r} is not {DATA_CHECK_PASSED}.")
    if parse_utc(signal["signal_computed_at_utc"], "computed") >= execution:
        reasons.append("Signal computed after scheduled execution.")
    return (KIND_DIAGNOSTIC, reasons) if reasons else (KIND_CURRENT, [])


def _csv_text(name: str, rows: list, cash: Decimal, header: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    base = [CSV_SCHEMA_VERSION, header["run_id"], header["allocation_id"], header["package_kind"], name,
            header["cycle_key"], header["signal_as_of_utc"], header["scheduled_execution_at_utc"]]
    for instrument, weight in rows:
        writer.writerow(base + [ROW_SECURITY, instrument.security_id, instrument.isin, instrument.wkn,
                                instrument.ticker, instrument.name, instrument.asset_type,
                                instrument.share_class, instrument.sector_code, instrument.sector_name,
                                instrument.currency, format_weight(weight)])
    writer.writerow(base + [ROW_CASH, f"CASH:{BASE_CURRENCY}", "", "", "", f"Cash {BASE_CURRENCY}", "CASH",
                            "", "", "", BASE_CURRENCY, format_weight(cash)])
    return buffer.getvalue()


def _strategy_fingerprint(result: dict) -> dict:
    if result["status"] != STATUS_COMPLETE:
        return {"status": result["status"], "error": result["error"]}
    return {"status": STATUS_COMPLETE, "cash": format_weight(result["cash"]),
            "rows": [[i.security_id, i.isin, format_weight(w)] for i, w in result["rows"]]}


def _definition_ref(name: str) -> dict:
    return {"label": STRATEGY_DEFINITIONS[name]["label"],
            "definition_version": STRATEGY_DEFINITIONS[name]["definition_version"]}


def _summary_markdown(manifest: dict) -> str:
    berlin, new_york = _tz("Europe/Berlin"), _tz("America/New_York")

    def when(value: str) -> str:
        moment = parse_utc(value, "time")
        return (f"{moment.isoformat()} (New York {moment.astimezone(new_york):%Y-%m-%d %H:%M %Z}, "
                f"Berlin {moment.astimezone(berlin):%Y-%m-%d %H:%M %Z})")

    lines = [
        f"# Zielportfolio-Paket {manifest['cycle_key']}", "",
        f"- Paketart: **{manifest['package_kind']}** (Übergabe zulässig: {manifest['handoff_eligible']})",
        f"- Modus: {manifest['mode']}; synthetische Testdaten: {manifest['synthetic_fixture']}",
        f"- Allokations-ID: `{manifest['allocation_id']}`; Ausgabe-ID: `{manifest['run_id']}`",
        f"- Informationsschluss: {when(manifest['information_cutoff_utc'])}",
        f"- Vorgesehene Ausführung: {when(manifest['scheduled_execution_at_utc'])}",
        f"- Gültig bis: {when(manifest['valid_until_utc'])}",
        f"- Erzeugt: {when(manifest['generated_at_utc'])}",
        f"- Basiswährung: {manifest['base_currency']}; Gewichtseinheit: Anteil (0.025 = 2,5 %)",
        f"- Datenprüfung: {manifest['data_check_status']}; historische Validierung: "
        f"{manifest['historical_validation_status']}", "",
        "## Strategien (drei getrennte Portfolios)", "",
        "| Strategie | Bezeichnung | Status | Titel | Cash | Datei / Grund |", "|---|---|---|---|---|---|",
    ]
    for name in STRATEGIES:
        item = manifest["strategies"][name]
        detail = item["file"] if item["status"] == STATUS_COMPLETE else item["error"].replace("|", "/")
        lines.append(f"| {name} | {item['label']} | {item['status']} | {item['security_count']} | "
                     f"{item['cash_weight'] or '-'} | {detail} |")
    lines += ["", "## Modell", "",
              f"- Regelversion {manifest['model']['strategy_rule_version']}, "
              f"{manifest['model']['paper_replication']}, {manifest['model']['validation_status']}",
              f"- Optimierungsdefinition {manifest['model']['optimization_version']}: "
              f"{manifest['model']['optimization_definition']['objective']}",
              f"- MATLAB-Äquivalenz: {manifest['model']['optimization_definition']['matlab_equivalence']}",
              f"- Hinweis: {manifest['model']['performance_metric_note']}",
              *[f"- Abweichung: {item}" for item in manifest["model"]["deviations"]],
              *[f"- {name}: {item.get('deviation', item['definition'])}"
                for name, item in manifest["model"]["strategy_definitions"].items()]]
    lines += ["", "## Einschränkungen", ""]
    lines += [f"- {reason}" for reason in manifest["restrictions"]] or ["- keine"]
    lines += ["", "Dies ist ein vollständiges Zielportfolio je Strategie, **keine Handelsanweisung**. "
              "Modellergebnisse sind nicht validiert, sofern oben nicht anders angegeben.", ""]
    return "\n".join(lines)


def _fsync_write(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def new_run_id(now: datetime) -> str:
    return f"{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"


def _existing_package(output_root: Path, allocation_id: str, kind: str) -> Path | None:
    for candidate in sorted(output_root.iterdir()):
        if candidate.name.startswith(".") or candidate.is_symlink() or not candidate.is_dir():
            continue
        manifest = candidate / "manifest.json"
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if doc.get("allocation_id") == allocation_id and doc.get("package_kind") == kind:
            return candidate
    return None


def export_package(signal_doc: Mapping, instrument_map_path: Path, output_root: Path, *, mode: str,
                   now: Callable[[], datetime] = utc_now, allow_stale: bool = False) -> Path:
    """Build and atomically publish a package; returns the published (or identical existing) directory."""
    signal = validate_signal_result(signal_doc)
    if not signal["synthetic_fixture"] and not signal["historical_only"]:
        source_run = signal_doc.get("source_execution_dir")
        if not isinstance(source_run, str) or not source_run:
            raise TargetPortfolioError("Current real signal requires its sealed source execution, not a declared status")
        checked = signal_result_from_signal_execution(Path(source_run))
        if checked["historical_only"] or _canonical(checked) != _canonical(dict(signal_doc)):
            raise TargetPortfolioError("Current signal differs from its evidence-checked source execution")
    instruments = load_instrument_map(instrument_map_path)
    generated_at = now()
    if parse_utc(instruments.frozen_at_utc, "frozen") > generated_at:
        raise TargetPortfolioError("Instrument map frozen_at_utc lies in the future")
    timing = signal["timing"]
    kind, kind_reasons = classify(mode, generated_at, timing, signal, allow_stale=allow_stale)
    results = {}
    for name in STRATEGIES:
        item = signal["strategies"][name]
        results[name] = (_strategy_rows(item["weights"], instruments)
                         if item["status"] == STATUS_COMPLETE
                         else {"status": STATUS_FAILED, "rows": None, "error": item["error"]})
    payload = _allocation_payload(signal["cycle_key"], timing, signal, instruments.sha256,
                                  {name: _strategy_fingerprint(results[name]) for name in STRATEGIES})
    allocation_id = allocation_id_for(payload)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    existing = _existing_package(output_root, allocation_id, kind)
    if existing is not None:
        return existing
    run_id = new_run_id(generated_at)
    restrictions = list(signal["restrictions"]) + kind_reasons
    if signal["synthetic_fixture"]:
        restrictions.append("SYNTHETIC FIXTURE: test data only, never a real signal.")
    restrictions.append("Model is NOT an exact paper replication; see manifest model.deviations.")
    if signal["historical_validation_status"] != "VALIDATED":
        restrictions.append(f"Model historical validation status: {signal['historical_validation_status']}.")
    for name in STRATEGIES:
        if results[name]["status"] != STATUS_COMPLETE:
            restrictions.append(f"{name}: {results[name]['status']}: {results[name]['error']}")
    header = {"run_id": run_id, "allocation_id": allocation_id, "package_kind": kind,
              "cycle_key": signal["cycle_key"], "signal_as_of_utc": timing["information_cutoff_utc"],
              "scheduled_execution_at_utc": timing["scheduled_execution_at_utc"]}
    files: dict[str, bytes] = {"instrument_map.json": instruments.raw,
                               "signal_result.json": _canonical(dict(signal_doc))}
    strategy_manifest = {}
    for name in STRATEGIES:
        result = results[name]
        if result["status"] == STATUS_COMPLETE:
            data = _csv_text(name, result["rows"], result["cash"], header).encode("utf-8")
            files[f"{name}.csv"] = data
            strategy_manifest[name] = {
                **_definition_ref(name),
                "status": STATUS_COMPLETE, "file": f"{name}.csv", "failure_file": None, "error": None,
                "security_count": len(result["rows"]), "row_count": len(result["rows"]) + 1,
                "cash_weight": format_weight(result["cash"]),
                "weight_sum": format_weight(result["cash"] + sum((w for _, w in result["rows"]), Decimal(0))),
            }
        else:
            failure = {"schema_version": PACKAGE_SCHEMA_VERSION, "strategy": name, "status": result["status"],
                       "error": result["error"], "no_target_portfolio": True, **header}
            files[f"{name}.FAILED.json"] = json.dumps(failure, indent=2, sort_keys=True).encode("utf-8")
            strategy_manifest[name] = {
                **_definition_ref(name),
                "status": result["status"], "file": None, "failure_file": f"{name}.FAILED.json",
                "error": result["error"], "security_count": 0, "row_count": 0,
                "cash_weight": None, "weight_sum": None,
            }
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION, "csv_schema_version": CSV_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION, "strategy_rule_version": RULE_VERSION,
        "package_kind": kind, "handoff_eligible": kind == KIND_CURRENT, "mode": mode,
        "synthetic_fixture": signal["synthetic_fixture"], "historical_only": signal["historical_only"],
        "not_trade_instructions": True,
        **header, "information_cutoff_utc": timing["information_cutoff_utc"],
        "scheduled_execution_date": timing["execution_date"], "execution_timezone": "America/New_York",
        "signal_computed_at_utc": signal["signal_computed_at_utc"],
        "signal_computed_at_source": signal["signal_computed_at_source"], "generated_at_utc": iso_utc(generated_at),
        "valid_from_utc": iso_utc(generated_at), "valid_until_utc": timing["valid_until_utc"],
        "base_currency": BASE_CURRENCY,
        "code_id": signal["code_id"], "data_id": signal["data_id"], "config_id": signal["config_id"],
        "instrument_map": {"file": "instrument_map.json", "sha256": instruments.sha256,
                           "format_version": INSTRUMENT_MAP_VERSION, "frozen_at_utc": instruments.frozen_at_utc},
        "data_check_status": signal["data_check_status"],
        "historical_validation_status": signal["historical_validation_status"],
        "model": {
            "strategy_rule_version": RULE_VERSION, "paper_replication": PAPER_REPLICATION_STATUS,
            "validation_status": "NOT_VALIDATED", "strategy_definitions": STRATEGY_DEFINITIONS,
            "deviations": MODEL_DEVIATIONS,
            "optimization_version": OPTIMIZATION_VERSION,
            "optimization_definition": signal["model_rules"]["optimization_definition"],
            "performance_metric_note": PERFORMANCE_METRIC_NOTE,
            "rules_source": signal["model_rules_source"], "rules": signal["model_rules"],
        },
        "strategies": strategy_manifest, "restrictions": restrictions,
        "weights": {"unit": "fraction_of_strategy_portfolio_value", "example": "0.025 = 2.5 percent",
                    "decimal_places": WEIGHT_DECIMALS, "position_cap": format_weight(POSITION_CAP),
                    "bound_tolerance": str(BOUND_TOLERANCE), "sum_tolerance": str(SUM_TOLERANCE),
                    "cash": "explicit CASH row per CSV; security weights + cash = 1"},
        "csv_format": {"encoding": "utf-8", "bom": False, "delimiter": ",", "quotechar": "\"",
                       "decimal_separator": ".", "line_terminator": "\\n", "timestamps": "ISO-8601 UTC",
                       "columns": list(CSV_COLUMNS)},
        "target_portfolio_semantics": TARGET_SEMANTICS,
        "files": {name: {"sha256": sha256_bytes(data), "bytes": len(data)} for name, data in files.items()},
    }
    files["zusammenfassung.md"] = _summary_markdown(manifest).encode("utf-8")
    manifest["files"]["zusammenfassung.md"] = {"sha256": sha256_bytes(files["zusammenfassung.md"]),
                                               "bytes": len(files["zusammenfassung.md"])}
    temporary = output_root / f".tmp-{run_id}"
    final = output_root / f"{signal['cycle_key']}_{kind}_{allocation_id[6:22]}_{run_id}"
    temporary.mkdir()
    try:
        for name, data in files.items():
            _fsync_write(temporary / name, data)
        _fsync_write(temporary / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        _fsync_dir(temporary)
        os.rename(temporary, final)
        _fsync_dir(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


# ---------------------------------------------------------------- import / validation

@dataclass(frozen=True)
class TargetPosition:
    security_id: str
    isin: str
    wkn: str
    ticker: str
    name: str
    asset_type: str
    share_class: str
    sector_code: str
    sector_name: str
    currency: str
    target_weight: Decimal


@dataclass(frozen=True)
class StrategyPortfolio:
    strategy: str
    status: str
    error: str | None
    positions: tuple[TargetPosition, ...] = ()
    cash_weight: Decimal | None = None


@dataclass
class PackageReport:
    path: Path
    manifest: dict
    strategies: dict[str, StrategyPortfolio]
    handoff_blockers: list[str] = field(default_factory=list)

    @property
    def allocation_id(self) -> str:
        return self.manifest["allocation_id"]

    def as_dict(self) -> dict:
        return {
            "path": str(self.path), "allocation_id": self.allocation_id, "run_id": self.manifest["run_id"],
            "cycle_key": self.manifest["cycle_key"], "package_kind": self.manifest["package_kind"],
            "integrity": "VERIFIED", "handoff_allowed": not self.handoff_blockers,
            "handoff_blockers": self.handoff_blockers,
            "valid_until_utc": self.manifest["valid_until_utc"],
            "strategies": {name: {"status": item.status, "error": item.error,
                                  "securities": len(item.positions),
                                  "cash_weight": None if item.cash_weight is None else format_weight(item.cash_weight)}
                           for name, item in self.strategies.items()},
        }


def _read_member(directory: Path, name: str) -> bytes:
    if not isinstance(name, str) or not _SAFE_NAME.match(name) or name in {".", ".."}:
        raise TargetPortfolioError(f"Unsafe package member name {name!r}")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise TargetPortfolioError(f"Package member is not a regular file: {name}")
    if path.resolve().parent != directory.resolve():
        raise TargetPortfolioError(f"Package member escapes package: {name}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise TargetPortfolioError(f"Package member too large: {name}")
    return path.read_bytes()


def _parse_weight(text: str, where: str) -> Decimal:
    if not _WEIGHT_TEXT.match(text):
        raise TargetPortfolioError(f"{where}: weight {text!r} is not a nonnegative {WEIGHT_DECIMALS}-decimal fraction")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise TargetPortfolioError(f"{where}: invalid weight {text!r}") from exc
    if not value.is_finite() or value < 0:
        raise TargetPortfolioError(f"{where}: weight must be finite and nonnegative")
    return value


def _parse_csv(name: str, data: bytes, manifest: dict, instruments: InstrumentMap,
               expected: dict) -> tuple[list[TargetPosition], Decimal, dict]:
    if data.startswith(b"\xef\xbb\xbf"):
        raise TargetPortfolioError(f"{name}.csv: BOM not allowed")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TargetPortfolioError(f"{name}.csv is not UTF-8") from exc
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=",", quotechar='"', strict=True)
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise TargetPortfolioError(f"{name}.csv malformed: {exc}") from exc
    if not rows or tuple(rows[0]) != CSV_COLUMNS:
        raise TargetPortfolioError(f"{name}.csv header must be exactly {list(CSV_COLUMNS)}")
    fixed = {"schema_version": CSV_SCHEMA_VERSION, "run_id": manifest["run_id"],
             "allocation_id": manifest["allocation_id"], "package_kind": manifest["package_kind"],
             "strategy": name, "cycle_key": manifest["cycle_key"],
             "signal_as_of_utc": manifest["information_cutoff_utc"],
             "scheduled_execution_at_utc": manifest["scheduled_execution_at_utc"]}
    positions, cash_rows, ids, isins, total = [], [], set(), set(), Decimal(0)
    for number, values in enumerate(rows[1:], start=2):
        where = f"{name}.csv line {number}"
        if len(values) != len(CSV_COLUMNS):
            raise TargetPortfolioError(f"{where}: expected {len(CSV_COLUMNS)} fields")
        row: dict[str, str] = dict(zip(CSV_COLUMNS, values))
        for key, value in fixed.items():
            if row[key] != value:
                raise TargetPortfolioError(f"{where}: {key}={row[key]!r} does not match package ({value!r})")
        weight = _parse_weight(row["target_weight"], where)
        total += weight
        if row["row_type"] == ROW_CASH:
            if row["security_id"] != f"CASH:{BASE_CURRENCY}" or row["currency"] != BASE_CURRENCY or row["isin"]:
                raise TargetPortfolioError(f"{where}: invalid cash row")
            cash_rows.append(weight)
            continue
        if row["row_type"] != ROW_SECURITY:
            raise TargetPortfolioError(f"{where}: unknown row_type {row['row_type']!r}")
        if weight > POSITION_CAP + BOUND_TOLERANCE:
            raise TargetPortfolioError(f"{where}: weight {weight} exceeds position cap {POSITION_CAP} "
                                       "(wrong unit, e.g. percent instead of fraction?)")
        if not isin_is_valid(row["isin"]):
            raise TargetPortfolioError(f"{where}: invalid ISIN")
        if row["security_id"] in ids or row["isin"] in isins:
            raise TargetPortfolioError(f"{where}: duplicate security_id or ISIN")
        ids.add(row["security_id"])
        isins.add(row["isin"])
        reference = instruments.instruments.get(row["security_id"])
        position = TargetPosition(**{key: row[key] for key in (
            "security_id", "isin", "wkn", "ticker", "name", "asset_type", "share_class", "sector_code",
            "sector_name", "currency")}, target_weight=weight)
        if reference is None or any(getattr(reference, key) != getattr(position, key)
                                    for key in Instrument.__dataclass_fields__):
            raise TargetPortfolioError(f"{where}: identity does not match the frozen instrument map")
        positions.append(position)
    if len(cash_rows) != 1:
        raise TargetPortfolioError(f"{name}.csv must contain exactly one CASH row")
    if not positions:
        raise TargetPortfolioError(f"{name}.csv has no security rows")
    if abs(total - 1) > SUM_TOLERANCE:
        raise TargetPortfolioError(f"{name}.csv weights incl. cash sum to {total}, not 1")
    if len(rows) - 1 != expected["row_count"] or format_weight(cash_rows[0]) != expected["cash_weight"]:
        raise TargetPortfolioError(f"{name}.csv rows/cash do not match manifest")
    fingerprint = {"status": STATUS_COMPLETE, "cash": format_weight(cash_rows[0]),
                   "rows": [[p.security_id, p.isin, format_weight(p.target_weight)]
                            for p in sorted(positions, key=lambda p: p.security_id)]}
    return positions, cash_rows[0], fingerprint


def inspect_package(path: Path, *, now: datetime | None = None,
                    ledger: "ProcessedLedger | None" = None,
                    accept_unvalidated_model: bool = False,
                    allow_synthetic_fixture: bool = False) -> PackageReport:
    """Verify integrity (raises TargetPortfolioError) and list handoff blockers. Read-only."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise TargetPortfolioError(f"Package must be a real directory: {path}")
    if path.name.startswith("."):
        raise TargetPortfolioError("Unpublished/temporary package directory")
    try:
        manifest = json.loads(_read_member(path, "manifest.json").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetPortfolioError(f"manifest.json invalid: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TargetPortfolioError("manifest.json must be an object")
    for key, expected in (("schema_version", PACKAGE_SCHEMA_VERSION), ("csv_schema_version", CSV_SCHEMA_VERSION),
                          ("protocol_version", PROTOCOL_VERSION), ("strategy_rule_version", RULE_VERSION),
                          ("base_currency", BASE_CURRENCY), ("not_trade_instructions", True)):
        if manifest.get(key) != expected:
            raise TargetPortfolioError(f"manifest {key} must be {expected!r}")
    if manifest.get("package_kind") not in PACKAGE_KINDS:
        raise TargetPortfolioError("manifest package_kind unknown")
    if not _ALLOCATION.match(str(manifest.get("allocation_id"))) or not _RUN_ID.match(str(manifest.get("run_id"))):
        raise TargetPortfolioError("manifest allocation_id/run_id malformed")
    timing = cycle_timing(str(manifest.get("cycle_key")))
    for key in ("information_cutoff_utc", "scheduled_execution_at_utc", "valid_until_utc"):
        if manifest.get(key) != timing[key]:
            raise TargetPortfolioError(f"manifest {key} does not match the {timing['cycle_key']} schedule")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TargetPortfolioError("manifest files missing")
    present = {entry.name for entry in path.iterdir()}
    if present != set(files) | {"manifest.json"}:
        raise TargetPortfolioError(f"Package files differ from manifest: extra={sorted(present - set(files) - {'manifest.json'})}, "
                                   f"missing={sorted(set(files) - present)}")
    contents = {}
    for name, info in files.items():
        data = _read_member(path, name)
        if not isinstance(info, dict) or sha256_bytes(data) != info.get("sha256") or len(data) != info.get("bytes"):
            raise TargetPortfolioError(f"Hash/size mismatch for {name}")
        contents[name] = data
    map_info = manifest.get("instrument_map", {})
    if map_info.get("file") != "instrument_map.json" or map_info.get("sha256") != sha256_bytes(contents.get("instrument_map.json", b"")):
        raise TargetPortfolioError("Instrument map hash mismatch")
    map_copy = path / "instrument_map.json"
    instruments = load_instrument_map(map_copy)
    strategies_doc = manifest.get("strategies")
    if not isinstance(strategies_doc, dict) or set(strategies_doc) != set(STRATEGIES):
        raise TargetPortfolioError(f"manifest strategies must be exactly {list(STRATEGIES)}")
    portfolios, fingerprints = {}, {}
    for name in STRATEGIES:
        item = strategies_doc[name]
        status = item.get("status")
        if item.get("definition_version") != STRATEGY_DEFINITIONS[name]["definition_version"]:
            raise TargetPortfolioError(f"{name}: unknown strategy definition version {item.get('definition_version')!r}")
        if status == STATUS_COMPLETE:
            if item.get("file") != f"{name}.csv" or f"{name}.FAILED.json" in contents:
                raise TargetPortfolioError(f"{name}: COMPLETE strategy file layout invalid")
            positions, cash, fingerprints[name] = _parse_csv(name, contents[f"{name}.csv"], manifest, instruments, item)
            portfolios[name] = StrategyPortfolio(name, status, None, tuple(positions), cash)
        elif status in (STATUS_FAILED, STATUS_BLOCKED_IDENTITY):
            if f"{name}.csv" in contents or item.get("failure_file") != f"{name}.FAILED.json":
                raise TargetPortfolioError(f"{name}: failed strategy must have only a FAILED file")
            failure = json.loads(contents[f"{name}.FAILED.json"].decode("utf-8"))
            if (failure.get("status") != status or failure.get("error") != item.get("error")
                    or failure.get("allocation_id") != manifest["allocation_id"]):
                raise TargetPortfolioError(f"{name}: FAILED file inconsistent with manifest")
            fingerprints[name] = {"status": status, "error": item.get("error")}
            portfolios[name] = StrategyPortfolio(name, status, item.get("error"))
        else:
            raise TargetPortfolioError(f"{name}: unknown status {status!r}")
    payload = _allocation_payload(manifest["cycle_key"], timing, manifest, map_info["sha256"], fingerprints)
    if allocation_id_for(payload) != manifest["allocation_id"]:
        raise TargetPortfolioError("allocation_id does not match package content")
    report = PackageReport(path, manifest, portfolios)
    blockers = report.handoff_blockers
    if manifest["package_kind"] != KIND_CURRENT or manifest.get("handoff_eligible") is not True:
        blockers.append(f"Package kind {manifest['package_kind']} is not a current handoff package")
    if manifest.get("mode") != MODE_REGULAR or manifest.get("historical_only") is not False:
        blockers.append("Not a regular quarter run")
    if manifest.get("synthetic_fixture") is not False and not allow_synthetic_fixture:
        blockers.append("Synthetic fixture package")
    if manifest.get("data_check_status") != DATA_CHECK_PASSED:
        blockers.append(f"Data check status {manifest.get('data_check_status')!r}")
    if manifest.get("historical_validation_status") != "VALIDATED" and not accept_unvalidated_model:
        blockers.append(f"Model historical validation status {manifest.get('historical_validation_status')!r}; "
                        "explicit acceptance required")
    moment = now or utc_now()
    generated = parse_utc(manifest.get("generated_at_utc"), "generated_at_utc")
    if generated < parse_utc(timing["information_cutoff_utc"], "cutoff") and manifest["package_kind"] == KIND_CURRENT:
        raise TargetPortfolioError("Current package generated before its information cutoff")
    if moment < generated:
        blockers.append("Current time precedes package generation (clock error?)")
    if moment >= parse_utc(timing["valid_until_utc"], "valid_until"):
        blockers.append(f"Expired: valid until {timing['valid_until_utc']}")
    if ledger is not None:
        for name in STRATEGIES:
            if ledger.is_claimed(manifest["allocation_id"], name):
                blockers.append(f"{name}: allocation {manifest['allocation_id']} already processed")
    return report


def load_target_portfolio(path: Path, strategy: str, **kwargs) -> tuple[PackageReport, StrategyPortfolio]:
    """Strict import for an execution consumer; raises unless this strategy may be handed off now."""
    if strategy not in STRATEGIES:
        raise TargetPortfolioError(f"Unknown strategy {strategy!r}")
    report = inspect_package(path, **kwargs)
    prefixes = tuple(f"{other}:" for other in STRATEGIES if other != strategy)
    blockers = [b for b in report.handoff_blockers if not b.startswith(prefixes)]
    portfolio = report.strategies[strategy]
    if portfolio.status != STATUS_COMPLETE:
        blockers.append(f"{strategy} status {portfolio.status}: {portfolio.error}")
    if blockers:
        raise TargetPortfolioError("Handoff refused: " + "; ".join(blockers))
    return report, portfolio


DEFAULT_PACKAGE_ROOT = Path("state/research/target_portfolios")


def list_packages(root: Path = DEFAULT_PACKAGE_ROOT, *, now: datetime | None = None,
                  ledger: "ProcessedLedger | None" = None, **kwargs) -> list[dict]:
    """Read-only dashboard listing: one entry per published package, newest first.

    Temporary (dot) directories are skipped; damaged packages are listed with integrity INVALID
    instead of being hidden, and are never handoff-allowed."""
    root = Path(root)
    if not root.is_dir():
        return []
    entries = []
    for path in root.iterdir():
        if path.name.startswith(".") or not path.is_dir() or path.is_symlink():
            continue
        try:
            report = inspect_package(path, now=now, ledger=ledger, **kwargs)
            manifest = report.manifest
            entry = {**report.as_dict(), "generated_at_utc": manifest.get("generated_at_utc"),
                     "mode": manifest.get("mode"), "historical_only": manifest.get("historical_only"),
                     "data_check_status": manifest.get("data_check_status"),
                     "synthetic_fixture": manifest.get("synthetic_fixture"),
                     "model_validation_status": manifest.get("model", {}).get("validation_status"),
                     "paper_replication": manifest.get("model", {}).get("paper_replication")}
        except (TargetPortfolioError, OSError, ValueError, KeyError) as exc:
            entry = {"path": str(path), "integrity": "INVALID", "error": str(exc),
                     "handoff_allowed": False, "generated_at_utc": None}
        entries.append(entry)
    return sorted(entries, key=lambda item: (item.get("generated_at_utc") or "", item["path"]), reverse=True)


class ProcessedLedger:
    """Durable claim store: one exclusive file per (allocation_id, strategy). Validation never claims."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, allocation_id: str, strategy: str) -> Path:
        if not _ALLOCATION.match(allocation_id) or strategy not in STRATEGIES:
            raise TargetPortfolioError("Invalid allocation_id or strategy for ledger")
        return self.directory / f"{allocation_id}__{strategy}.claim.json"

    def is_claimed(self, allocation_id: str, strategy: str) -> bool:
        return self._path(allocation_id, strategy).exists()

    def claim(self, allocation_id: str, strategy: str, *, consumer: str, run_id: str,
              now: datetime | None = None) -> Path:
        target = self._path(allocation_id, strategy)
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"allocation_id": allocation_id, "strategy": strategy, "run_id": run_id,
                  "consumer": consumer, "claimed_at_utc": iso_utc(now or utc_now())}
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise AlreadyClaimed(f"{allocation_id}/{strategy} already claimed") from exc
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(record, sort_keys=True).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.directory)
        return target


def claim_target_portfolio(path: Path, strategy: str, ledger: ProcessedLedger, *, consumer: str,
                           **kwargs) -> tuple[StrategyPortfolio, Path]:
    """Validate, then durably claim; a second claim of the same allocation/strategy fails."""
    report, portfolio = load_target_portfolio(path, strategy, ledger=ledger, **kwargs)
    receipt = ledger.claim(report.allocation_id, strategy, consumer=consumer, run_id=report.manifest["run_id"],
                           now=kwargs.get("now"))
    return portfolio, receipt


# ---------------------------------------------------------------- adapters

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SIGNAL_RUN_REQUIRED = ("primary/signal.json", "replay/signal.json", "execution_summary.json",
                       "rule_config.json", "code_manifest.json", "snapshot/bundle_manifest.json")
HISTORY_RUN_STATUSES = ("COMPLETED_NOT_VALIDATED", "COMPLETED_WITH_OPTIMIZER_ERRORS")
SIGNAL_RUN_STATUSES = ("COMPLETED_NOT_VALIDATED", "PARTIAL_STRATEGY_ERRORS")
MTIME_COMPUTED_NOTE = ("signal_computed_at_utc is the file modification time of {file}: the actual generation "
                       "time of this run, not a historical decision time.")


def _relative_files(root: Path) -> set[str]:
    """All entries below root as POSIX relative paths; symlinks anywhere are refused."""
    found = set()
    for item in root.rglob("*"):
        if item.is_symlink():
            raise TargetPortfolioError(f"Symlink in executor output: {item.relative_to(root)}")
        if item.is_file():
            found.add(item.relative_to(root).as_posix())
    return found


def _hash_map(doc, where: str) -> dict[str, str]:
    if not isinstance(doc, Mapping) or not doc:
        raise TargetPortfolioError(f"{where} must be a nonempty mapping of relative path to sha256")
    result = {}
    for name, digest in doc.items():
        parts = name.split("/") if isinstance(name, str) else []
        if (not parts or name.startswith("/") or "\\" in name
                or any(part in ("", ".", "..") for part in parts)):
            raise TargetPortfolioError(f"{where}: unsafe path {name!r}")
        if not isinstance(digest, str) or not _HEX_SHA256.match(digest):
            raise TargetPortfolioError(f"{where}: malformed sha256 for {name}")
        result[name] = digest
    return result


def _verify_tree(root: Path, hashes: Mapping[str, str], where: str, *, exclude=lambda name: False) -> None:
    actual = {name for name in _relative_files(root) if not exclude(name)}
    if actual != set(hashes):
        raise TargetPortfolioError(f"{where}: files differ from recorded hashes: "
                                   f"unrecorded={sorted(actual - set(hashes))[:5]}, "
                                   f"missing={sorted(set(hashes) - actual)[:5]}")
    for name, digest in hashes.items():
        if sha256_bytes((root / name).read_bytes()) != digest:
            raise TargetPortfolioError(f"{where}: hash mismatch for {name} (artifact mutated)")


def _verified_bytes(root: Path, name: str, hashes: Mapping[str, str], where: str) -> bytes:
    if name not in hashes:
        raise TargetPortfolioError(f"{where}: required artifact {name} is not recorded")
    data = (root / name).read_bytes()
    if sha256_bytes(data) != hashes[name]:
        raise TargetPortfolioError(f"{where}: hash mismatch for {name} (artifact mutated)")
    return data


def _json_bytes(data: bytes, name: str):
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TargetPortfolioError(f"{name} is not valid JSON: {exc}") from exc


def _prefixed(hashes: Mapping[str, str], prefix: str) -> dict[str, str]:
    return {name[len(prefix):]: digest for name, digest in hashes.items() if name.startswith(prefix)}


def signal_result_from_history_execution(run_dir: Path, cycle_key: str) -> dict:
    """Adapt one cycle of a ``paper_history_execution.execute`` run directory. Always historical_only.

    ``run_dir`` is the executor's output root (``execution_summary.json``, ``primary/``, ``replay/``).
    Every non-snapshot primary artifact must match ``artifact_hashes`` and the independent replay
    copy; both snapshots must be identical. The summary's replay flag alone is not trusted."""
    run_dir = Path(run_dir)
    timing = cycle_timing(cycle_key)
    where = f"history run {run_dir.name}"
    summary_path = run_dir / "execution_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise TargetPortfolioError(f"{where}: execution_summary.json missing (pass the run root, not primary/)")
    summary = _json_bytes(summary_path.read_bytes(), "execution_summary.json")
    if not isinstance(summary, dict) or summary.get("status") not in HISTORY_RUN_STATUSES:
        raise TargetPortfolioError(f"{where}: status {summary.get('status') if isinstance(summary, dict) else None!r} "
                                   "is not exportable")
    if summary.get("model_replay_identical") is not True or summary.get("network_attempts") != [] \
            or summary.get("inputs_changed") != []:
        raise TargetPortfolioError(f"{where}: not an independently replayed offline run")
    hashes = _hash_map(summary.get("artifact_hashes"), f"{where} artifact_hashes")
    primary, replay = run_dir / "primary", run_dir / "replay"
    in_snapshot = lambda name: name.split("/", 1)[0] == "snapshot"
    for directory in (primary, replay):
        if directory.is_symlink() or not directory.is_dir():
            raise TargetPortfolioError(f"{where}: {directory.name}/ missing")
        _verify_tree(directory, hashes, f"{where} {directory.name}", exclude=in_snapshot)
    snapshots = [{name: sha256_bytes((d / "snapshot" / name).read_bytes())
                  for name in _relative_files(d / "snapshot")} for d in (primary, replay)]
    if not snapshots[0] or snapshots[0] != snapshots[1]:
        raise TargetPortfolioError(f"{where}: primary and replay input snapshots differ")
    bundle = _json_bytes((primary / "snapshot" / "bundle_manifest.json").read_bytes(), "bundle_manifest.json")
    bundle_files = _hash_map(bundle.get("files") if isinstance(bundle, dict) else None, "bundle_manifest files")
    if bundle.get("review_status") != "SOURCE_REVIEWED" or any(snapshots[0].get(n) != d for n, d in bundle_files.items()):
        raise TargetPortfolioError(f"{where}: snapshot does not match its SOURCE_REVIEWED bundle manifest")
    quarter_name = f"{cycle_key}.json"
    quarter = _json_bytes(_verified_bytes(primary, quarter_name, hashes, where), quarter_name)
    rule_bytes = _verified_bytes(primary, "rule_config.json", hashes, where)
    code_bytes = _verified_bytes(primary, "code_manifest.json", hashes, where)
    if _json_bytes(code_bytes, "code_manifest.json") != _prefixed(hashes, "code/"):
        raise TargetPortfolioError(f"{where}: code_manifest.json does not match the frozen code copy")
    selected = set()
    for sector in quarter.get("sector_models", []):
        selected.update(str(key) for key in sector.get("selected", {}))
    errors = quarter.get("optimizer_errors") or {}
    strategies = {}
    for internal, name in INTERNAL_STRATEGY_NAMES.items():
        error = errors.get(internal) or errors.get("all")
        weights_name = f"{internal}/gross/target_weights.csv"
        if error:
            strategies[name] = {"status": STATUS_FAILED, "weights": None, "error": str(error)}
            continue
        if weights_name not in hashes:
            strategies[name] = {"status": STATUS_FAILED, "weights": None,
                                "error": "History executor did not publish target weights for this strategy"}
            continue
        frame = pd.read_csv(io.StringIO(_verified_bytes(primary, weights_name, hashes, where).decode("utf-8")),
                            index_col="date")
        if timing["execution_date"] not in frame.index:
            strategies[name] = {"status": STATUS_FAILED, "weights": None,
                                "error": f"No weights for execution date {timing['execution_date']}"}
            continue
        row = frame.loc[timing["execution_date"]].astype(float)
        stray = sorted(set(row[row != 0].index) - selected)
        if stray:
            raise TargetPortfolioError(f"Weights for unselected securities: {stray}")
        strategies[name] = {"status": STATUS_COMPLETE, "error": None,
                            "weights": {sid: float(row.get(sid, 0.0)) for sid in sorted(selected)}}
    data_digest = hashlib.sha256(_canonical(snapshots[0])).hexdigest()
    return {
        "format_version": SIGNAL_RESULT_VERSION, "cycle_key": cycle_key, "rule_version": RULE_VERSION,
        "information_cutoff_utc": timing["information_cutoff_utc"],
        "scheduled_execution_at_utc": timing["scheduled_execution_at_utc"],
        "signal_computed_at_utc": iso_utc(datetime.fromtimestamp((primary / quarter_name).stat().st_mtime,
                                                                 timezone.utc)),
        "signal_computed_at_source": "file_mtime",
        "code_id": "sha256:" + sha256_bytes(code_bytes),
        "data_id": "sha256:" + data_digest,
        "config_id": "sha256:" + sha256_bytes(rule_bytes),
        "data_check_status": "HISTORY_EXECUTOR_REPLAY_VERIFIED",
        "historical_validation_status": "NOT_VALIDATED", "base_currency": BASE_CURRENCY,
        "synthetic_fixture": False, "historical_only": True,
        "restrictions": [f"Adapted from history executor run {run_dir.name}; status {quarter.get('status')}.",
                         MTIME_COMPUTED_NOTE.format(file=f"primary/{quarter_name}")],
        "strategies": strategies,
        "model_rules": _json_bytes(rule_bytes, "rule_config.json"),
    }


SIGNAL_EXECUTION_STATUS = {"SUCCESS": STATUS_COMPLETE, "ERROR": STATUS_FAILED}
LIVE_PROVENANCE = "OBSERVED_LIVE_BEFORE_CUTOFF"


def signal_result_from_signal_execution(output_dir: Path) -> dict:
    """Adapt ``paper_signal_execution.execute_signal`` output.

    Every file below the run (except ``artifact_manifest.json`` itself) must be recorded there with a
    matching hash and nothing else may exist; primary/ and replay/ artifacts must be identical, and the
    summary's data/code manifests must match the frozen snapshot/code copies. The replay flag alone is
    not trusted. SOURCE_REVIEWED reconstruction inputs stay ``historical_only`` unless the summary
    declares ``input_provenance == "OBSERVED_LIVE_BEFORE_CUTOFF"`` with nonempty
    ``input_observation_evidence``.
    """
    output_dir = Path(output_dir)
    where = f"signal run {output_dir.name}"
    manifest_path = output_dir / "artifact_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TargetPortfolioError(f"{where}: artifact_manifest.json missing; unsealed output is not exportable")
    hashes = _hash_map(_json_bytes(manifest_path.read_bytes(), "artifact_manifest.json"),
                       f"{where} artifact_manifest")
    _verify_tree(output_dir, hashes, where, exclude=lambda name: name == "artifact_manifest.json")
    for name in SIGNAL_RUN_REQUIRED:
        if name not in hashes:
            raise TargetPortfolioError(f"{where}: required artifact {name} is not recorded")
    primary, replayed = _prefixed(hashes, "primary/"), _prefixed(hashes, "replay/")
    replay_identical = bool(primary) and primary == replayed
    signal_raw = _verified_bytes(output_dir, "primary/signal.json", hashes, where)
    summary_raw = _verified_bytes(output_dir, "execution_summary.json", hashes, where)
    rule_bytes = _verified_bytes(output_dir, "rule_config.json", hashes, where)
    code_bytes = _verified_bytes(output_dir, "code_manifest.json", hashes, where)
    doc, summary = _json_bytes(signal_raw, "signal.json"), _json_bytes(summary_raw, "execution_summary.json")
    if not isinstance(doc, dict) or not isinstance(summary, dict):
        raise TargetPortfolioError("Signal execution documents must be objects")
    replay = summary.get("model_replay_identical")
    if not isinstance(replay, bool):
        raise TargetPortfolioError("execution_summary.model_replay_identical must be boolean")
    if replay is not replay_identical:
        raise TargetPortfolioError(f"{where}: model_replay_identical={replay} contradicts the recorded "
                                   f"primary/replay artifacts (identical={replay_identical})")
    if not replay_identical:
        raise TargetPortfolioError(f"{where}: primary and replay signal artifacts differ; not exportable")
    snapshot = _prefixed(hashes, "snapshot/")
    data_manifest = _hash_map(summary.get("data_manifest"), "execution_summary data_manifest")
    if data_manifest != snapshot:
        raise TargetPortfolioError(f"{where}: summary data_manifest does not match the frozen input snapshot")
    bundle = _json_bytes(_verified_bytes(output_dir, "snapshot/bundle_manifest.json", hashes, where),
                         "bundle_manifest.json")
    bundle_files = _hash_map(bundle.get("files") if isinstance(bundle, dict) else None, "bundle_manifest files")
    if (bundle.get("review_status") != "SOURCE_REVIEWED" or bundle.get("scope") != "SIGNAL"
            or any(snapshot.get(name) != digest for name, digest in bundle_files.items())):
        raise TargetPortfolioError(f"{where}: snapshot does not match its SOURCE_REVIEWED SIGNAL bundle manifest")
    code_manifest = _json_bytes(code_bytes, "code_manifest.json")
    if code_manifest != _prefixed(hashes, "code/") or summary.get("code_manifest") != code_manifest:
        raise TargetPortfolioError(f"{where}: code_manifest does not match the frozen code copy")
    cycle_key = _require_text(doc, "cycle_key", "signal.json")
    timing = cycle_timing(cycle_key)
    if iso_utc(parse_utc(doc.get("information_cutoff"), "information_cutoff")) != timing["information_cutoff_utc"]:
        raise TargetPortfolioError("signal.json information_cutoff does not match the schedule")
    if iso_utc(parse_utc(doc.get("execution_at"), "execution_at")) != timing["scheduled_execution_at_utc"]:
        raise TargetPortfolioError("signal.json execution_at does not match the schedule")
    results = doc.get("strategy_results")
    if not isinstance(results, dict) or set(results) != set(STRATEGIES):
        raise TargetPortfolioError(f"strategy_results must be exactly {list(STRATEGIES)}")
    strategies = {}
    for name in STRATEGIES:
        item = results[name]
        if not isinstance(item, dict) or str(item.get("status")) not in SIGNAL_EXECUTION_STATUS:
            raise TargetPortfolioError(f"{name}: status must be SUCCESS or ERROR")
        status = SIGNAL_EXECUTION_STATUS[str(item["status"])]
        if status == STATUS_COMPLETE:
            if item.get("error") not in (None, ""):
                raise TargetPortfolioError(f"{name}: SUCCESS with error")
            strategies[name] = {"status": status, "weights": item.get("weights"), "error": None}
        else:
            # Any weights accompanying an ERROR are discarded; they are never a fallback.
            strategies[name] = {"status": status, "weights": None,
                                "error": str(item.get("error") or "ERROR without message")}
    if summary.get("status") not in SIGNAL_RUN_STATUSES:
        raise TargetPortfolioError(f"Signal execution status {summary.get('status')!r} is not exportable")
    if summary.get("cycle_key") != cycle_key:
        raise TargetPortfolioError("execution_summary cycle_key differs from signal.json")
    if summary.get("execution_orders") or doc.get("execution_orders") is not False:
        raise TargetPortfolioError("Signal execution output must declare execution_orders false; refused")
    availability = None
    if (output_dir / "snapshot/publication_evidence.json").is_file():
        availability = verify_signal_availability(
            output_dir / "snapshot", doc["information_cutoff"],
            training_vintage_policy=doc.get("training_vintage_policy", STRICT))
    publication_qualified = availability is not None
    restrictions = [f"Adapted from signal execution {output_dir.name}."]
    if doc.get("computed_at_utc") is not None:
        computed, computed_source = doc["computed_at_utc"], "signal.json"
        if summary.get("computed_at_utc") is not None and summary["computed_at_utc"] != computed:
            raise TargetPortfolioError("Signal computation timestamp differs from the sealed executor summary")
    else:
        computed = iso_utc(datetime.fromtimestamp((output_dir / "primary" / "signal.json").stat().st_mtime,
                                                  timezone.utc))
        computed_source = "file_mtime"
        restrictions.append(MTIME_COMPUTED_NOTE.format(file="primary/signal.json"))
    if computed_source == "file_mtime":
        publication_qualified = False
        restrictions.append("Legacy file mtime cannot qualify a new current handoff.")
    if not publication_qualified:
        restrictions.append("No complete byte-checked publication/lineage coverage; historical only. "
                            "A provenance marker or nonempty evidence list does not qualify current handoff.")
    return {
        "format_version": SIGNAL_RESULT_VERSION, "cycle_key": cycle_key, "rule_version": RULE_VERSION,
        "information_cutoff_utc": timing["information_cutoff_utc"],
        "scheduled_execution_at_utc": timing["scheduled_execution_at_utc"],
        "signal_computed_at_utc": iso_utc(parse_utc(computed, "computed_at_utc")),
        "signal_computed_at_source": computed_source,
        "code_id": "sha256:" + sha256_bytes(code_bytes),
        "data_id": "sha256:" + sha256_bytes(_canonical(data_manifest)),
        "config_id": "sha256:" + sha256_bytes(rule_bytes),
        "data_check_status": DATA_CHECK_PASSED,
        "historical_validation_status": "NOT_VALIDATED", "base_currency": BASE_CURRENCY,
        "synthetic_fixture": bool(summary.get("synthetic_fixture", False)),
        "historical_only": not publication_qualified, "restrictions": restrictions, "strategies": strategies,
        "publication_availability": availability,
        "source_execution_dir": str(output_dir.resolve()),
        "model_rules": _json_bytes(rule_bytes, "rule_config.json"),
    }
