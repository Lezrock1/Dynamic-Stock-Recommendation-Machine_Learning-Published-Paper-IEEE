"""Real-data adapters for the paper depot: SEC filings, the sector models, Yahoo prices.

Nothing here simulates data. Missing inputs raise ``MissingDataError`` (the runner retries)
or are reported as not received. ``environment="real"`` adapters only run on a ``real`` store.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

try:
    from . import ixbrl_fallback as ixf, prices as pr, ratios as ra, sec_edgar as se, universe as uv, universe_asof as ua
    from .market_calendar import (
        first_session_on_or_after, nyse_sessions, previous_nyse_session, session_close_time,
    )
    from .paper_runner import (
        Adapters, CorporateAction, FilingObservation, MissingDataError, SignalResult,
    )
    from .paper_store import PaperStore, ValidationError
    from .portfolio_weights import OptimizationError, capped_equal_weights, optimize_weights
except ImportError:  # pragma: no cover - script execution
    import ixbrl_fallback as ixf, prices as pr, ratios as ra, sec_edgar as se, universe as uv, universe_asof as ua  # type: ignore
    from market_calendar import (  # type: ignore
        first_session_on_or_after, nyse_sessions, previous_nyse_session, session_close_time,
    )
    from paper_runner import (  # type: ignore
        Adapters, CorporateAction, FilingObservation, MissingDataError, SignalResult,
    )
    from paper_store import PaperStore, ValidationError  # type: ignore
    from portfolio_weights import OptimizationError, capped_equal_weights, optimize_weights  # type: ignore

REAL_ORIGIN = "REAL_SEC_YAHOO"
DEFAULT_CONFIG = REPO_ROOT / "config" / "real_paper.json"
STRATEGY_IDS = {"equal_weight": "equal_weight_v1", "mean_var": "mean_var_v1", "min_var": "min_var_v1",
    "spy_benchmark": "spy_buy_hold_v1"}
FEATURES = [
    "X1_REVGH", "X2_EPS", "X3_ROA", "X4_ROE", "X5_PE", "X6_PS", "X7_NPM", "X8_GPM", "X9_OM", "X10_PB",
    "X11_PCFO", "X12_CR", "X13_EM", "X14_EVCFO", "X15_LTDTA", "X16_WCR", "X17_DE", "X18_QR",
    "X19_DSI", "X20_DPO",
]
HISTORICAL_CUTOFF = pd.Timestamp("2017-09-01")
ANNUAL_PERIOD_TYPES = ("10-K", "10-Q")


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.getenv("PAPER_REAL_CONFIG") or DEFAULT_CONFIG)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("environment") != "real":
        raise ValidationError(f"{path} is not a real-environment configuration")
    config["_path"] = str(path)
    se.register_ixbrl_fallback(config.get("ixbrl_fallback"))
    se.register_successors({
        int(cik): tuple(int(c) for c in item["predecessors"])
        for cik, item in config.get("universe", {}).get("cik_successors", {}).items()
    })
    return config


def resolve_path(config: Mapping[str, object], key: str) -> Path:
    path = Path(str(config[key])).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


# ---------------------------------------------------------------- expected reports
def _filings(submissions: Mapping | None) -> list[dict]:
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    keys = ("form", "accessionNumber", "reportDate", "filingDate", "acceptanceDateTime")
    columns = [recent.get(k, []) for k in keys]
    rows = [dict(zip(keys, values)) for values in zip(*columns)]
    return [r for r in rows if r["form"] in ANNUAL_PERIOD_TYPES and r.get("reportDate")]


def _month_end(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _nominal_month_end(value: str) -> pd.Timestamp:
    """52/53-week fiscal periods end up to a few days off the calendar month end."""
    date = pd.Timestamp(value)
    if date.day <= 7:
        date = date - pd.DateOffset(months=1)
    return date + pd.offsets.MonthEnd(0)


def expected_report(submissions: Mapping | None, quarter_end: pd.Timestamp) -> dict | None:
    """Reporting period of one company that belongs to the calendar quarter ``quarter_end``.

    The company's own quarter grid is derived from its filed reports (period-end months);
    the period within (quarter_end-45d, quarter_end+45d] is expected, as 10-K for the fiscal
    year-end month and as 10-Q otherwise. Returns None if no grid can be derived.
    """
    filings = _filings(submissions)
    if not filings:
        return None
    recent = sorted(filings, key=lambda r: r["reportDate"], reverse=True)[:8]
    grid_months = {_nominal_month_end(r["reportDate"]).month % 3 for r in recent}
    if len(grid_months) != 1:
        return _anniversary_report(filings, quarter_end)
    year_end_month = next((_nominal_month_end(r["reportDate"]).month for r in recent if r["form"] == "10-K"), None)
    if year_end_month is None:
        return None
    low, high = quarter_end - pd.Timedelta(days=45), quarter_end + pd.Timedelta(days=45)
    reference = _nominal_month_end(recent[0]["reportDate"])
    for offset in range(-24, 25):
        candidate = _month_end(reference.year, reference.month) + pd.DateOffset(months=offset)
        candidate = candidate + pd.offsets.MonthEnd(0)
        if int(candidate.month) % 3 not in grid_months or not low < candidate <= high:
            continue
        actual = next((pd.Timestamp(r["reportDate"]) for r in filings
                       if abs((pd.Timestamp(r["reportDate"]) - candidate).days) <= 10), None)
        period_end = actual if actual is not None else candidate
        form = "10-K" if candidate.month == year_end_month else "10-Q"
        return {"period_end": period_end.date().isoformat(), "form": form}
    return None


def _anniversary_report(filings: list[dict], quarter_end: pd.Timestamp) -> dict | None:
    """Off-grid fiscal calendars (e.g. 16-week first quarter): same period one year later.

    Needs a full prior fiscal year (one 10-K); otherwise the period is not derivable.
    """
    low, high = quarter_end - pd.Timedelta(days=45), quarter_end + pd.Timedelta(days=45)
    if not any(r["form"] == "10-K" for r in filings):
        return None
    ordered = sorted(filings, key=lambda r: r["reportDate"], reverse=True)
    for r in ordered:
        date = pd.Timestamp(r["reportDate"])
        if low < date <= high:
            return {"period_end": date.date().isoformat(), "form": r["form"]}
    for r in ordered:
        date = pd.Timestamp(r["reportDate"]) + pd.Timedelta(days=364)
        if low < date <= high:
            return {"period_end": date.date().isoformat(), "form": r["form"]}
    return None


def verify_succession(successor: int, item: Mapping[str, object], required: Sequence[str]) -> dict:
    """Continuity of the reported financial data across a documented CIK succession."""
    effective = pd.Timestamp(str(item["effective"]))
    old_ciks = [int(c) for c in item["predecessors"]]
    new_facts = ((se._fetch_company_facts(successor, 6 * 3600) or {}).get("facts") or {}).get("us-gaap") or {}
    old_facts = {c: ((se._fetch_company_facts(c, 6 * 3600) or {}).get("facts") or {}).get("us-gaap") or {}
                 for c in old_ciks}
    tags = {**se.DURATION_TAGS, **se.INSTANT_TAGS}
    compared, mismatches = 0, []
    for name in required:
        for tag in tags[name]:
            for unit, entries in (new_facts.get(tag, {}).get("units") or {}).items():
                for entry in entries:
                    for cik, facts in old_facts.items():
                        for old in (facts.get(tag, {}).get("units") or {}).get(unit, []):
                            if (old.get("end"), old.get("start"), old.get("fp")) == (
                                    entry.get("end"), entry.get("start"), entry.get("fp")) \
                                    and old.get("form") == entry.get("form") and old.get("accn") != entry.get("accn"):
                                compared += 1
                                if old.get("val") != entry.get("val"):
                                    mismatches.append({"fact": name, "tag": tag, "end": entry.get("end"),
                                                       "old": old.get("val"), "new": entry.get("val")})
    submissions_new = se._fetch_submissions(successor, 6 * 3600) or {}
    forms = (((submissions_new.get("filings") or {}).get("recent")) or {})
    reorganisation = [a for f, a, d in zip(forms.get("form", []), forms.get("accessionNumber", []),
                                           forms.get("filingDate", []))
                      if f == "8-K12B" and pd.Timestamp(d) == effective]
    result = {"successor": successor, "predecessors": old_ciks, "effective": effective.date().isoformat(),
              "source": item.get("source"), "sec_8k12b_accessions": reorganisation,
              "overlapping_values_compared": compared, "mismatches": mismatches}
    if mismatches:
        raise ValidationError(f"financial data discontinuity across CIK succession {successor}: {mismatches[:3]}")
    if not reorganisation:
        raise MissingDataError(f"no 8-K12B dated {effective.date()} found for successor CIK {successor}")
    return result


def classify_unresolved(cik: int, tickers: list[str], submissions: Mapping | None) -> dict:
    """Distinguish the reason a reporting period cannot be derived."""
    filings = _filings(submissions)
    reason = "no_filing_history" if not filings else (
        "no_annual_report_in_history" if not any(f["form"] == "10-K" for f in filings)
        else "irregular_period_calendar")
    return {"cik": cik, "tickers": tickers, "category": "unknown_reporting_period", "reason": reason,
            "filings_seen": len(filings)}


def generate_expected_filings(config: Mapping[str, object], quarter_end: str,
                              universe_as_of: str | None = None) -> dict[str, object]:
    """Deterministic list of expected reports for every universe company and one cycle."""
    se.require_user_agent()
    quarter = pd.Timestamp(quarter_end)
    spec = config.get("universe", {})
    as_of = pd.Timestamp(universe_as_of or quarter_end)
    members, universe_evidence = ua.members_as_of(as_of)
    exclusions = dict(spec.get("exclusions", {}))
    overrides = {int(k): v for k, v in spec.get("period_overrides", {}).items()}
    successors = {int(k): v for k, v in spec.get("cik_successors", {}).items()}
    companies: dict[int, list[str]] = {}
    for row in members.sort_values("tic").itertuples():
        companies.setdefault(int(row.cik), []).append(str(row.tic))
    expected, unresolved, succession_checks, applied_overrides = [], [], [], []
    for cik, tickers in sorted(companies.items()):
        if any(t in exclusions for t in tickers):
            continue
        submissions = se.get_submissions(cik, max_age_seconds=6 * 3600)
        report, evidence_source = None, None
        override = overrides.get(cik)
        if override and pd.Timestamp(override["quarter_end"]) == quarter:
            report = {"period_end": override["period_end"], "form": override["form"]}
            evidence_source = f"override:{override['source']}"
            applied_overrides.append({"cik": cik, **override})
        else:
            if cik in successors:
                succession_checks.append(verify_succession(cik, successors[cik], config["required_facts"]))
            report = expected_report(submissions, quarter)
        if report is None:
            unresolved.append(classify_unresolved(cik, tickers, submissions))
            continue
        entry = {
            "filing_key": f"{cik}-{report['period_end']}-{report['form']}",
            "company_id": str(cik), "security_id": tickers[0],
            "expected_period": report["period_end"], "filing_type": report["form"],
        }
        if evidence_source:
            entry["evidence"] = evidence_source
        expected.append(entry)
    if unresolved:
        path = resolve_path(config, "manifest_dir") / f"unresolved_{quarter.date().isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"unresolved": unresolved}, indent=2, sort_keys=True))
        raise MissingDataError(
            f"{len(unresolved)} companies lack a derivable reporting period (logged to {path}); the cycle is "
            f"not created and no rebalancing can happen until each is resolved by evidence: {unresolved[:10]}"
        )
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    return {
        "quarter_end": quarter.date().isoformat(), "universe_as_of": as_of.date().isoformat(),
        "universe_evidence": universe_evidence, "universe_source": spec["source"],
        "universe_hash": digest, "rule": "company quarter grid from filed 10-Q/10-K period ends "
        "(52/53-week calendars normalised); period within +-45 days of the calendar quarter end",
        "exclusions": exclusions, "period_overrides_applied": applied_overrides,
        "succession_checks": succession_checks,
        "generated_at": datetime.now(timezone.utc).isoformat(), "expected": expected,
    }


def write_manifest(config: Mapping[str, object], cycle_key: str, manifest: Mapping[str, object]) -> Path:
    directory = resolve_path(config, "manifest_dir")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cycle_key}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("universe_as_of") != manifest["universe_as_of"]:
            raise ValidationError(
                f"{path} is frozen with universe_as_of={existing.get('universe_as_of')}; "
                f"refusing silent change to {manifest['universe_as_of']}")
        if existing.get("universe_hash") != manifest["universe_hash"]:
            raise ValidationError(f"{path} exists with a different expected-report set")
        return path
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def create_real_cycle(store: PaperStore, config: Mapping[str, object], cycle_key: str,
                      quarter_end: str, deadline_at: str | None = None) -> dict[str, object]:
    """Freeze the expected reports and open one cycle per strategy portfolio."""
    if store.environment() != "real":
        raise ValidationError("real cycles can only be created in a real store")
    path = resolve_path(config, "manifest_dir") / f"{cycle_key}.json"
    if path.exists():  # a frozen manifest is never regenerated
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = generate_expected_filings(config, quarter_end)
        path = write_manifest(config, cycle_key, manifest)
    keys = [e["filing_key"] for e in manifest["expected"]]
    metadata = {e["filing_key"]: {**{k: e[k] for k in ("company_id", "security_id", "expected_period",
                                                       "filing_type")},
                                  "source": f"manifest:{manifest['universe_hash'][:16]}"
                                            + (f";{e['evidence']}" if e.get("evidence") else "")}
                for e in manifest.get("expected", [])}
    cycles = {}
    for portfolio_id, strategy in STRATEGY_IDS.items():
        cycle_id = store.create_cycle(portfolio_id, cycle_key, strategy, deadline_at=deadline_at)
        if not store.query("SELECT 1 FROM expected_filings WHERE cycle_id = ? LIMIT 1", (cycle_id,)):
            store.register_expected_filings(cycle_id, keys, filing_metadata=metadata)
        cycles[portfolio_id] = cycle_id
    return {"cycle_key": cycle_key, "expected_reports": len(keys), "manifest": str(path), "cycles": cycles}


def required_facts_for(config: Mapping[str, object], gsector: int | None, cik: int | None = None) -> list[str]:
    """Readiness rule: facts needed per company. Per-CIK overrides beat sector rules beat the default."""
    rule = config.get("readiness_rule") or {}
    by_cik = (rule.get("required_facts_by_cik") or {})
    if cik is not None and str(cik) in by_cik:
        return list(by_cik[str(cik)])
    by_sector = (rule.get("required_facts_by_gsector") or {})
    if gsector is not None and str(int(gsector)) in by_sector:
        return list(by_sector[str(int(gsector))])
    return list(config["required_facts"])


def _sector_by_cik() -> dict[int, int]:
    members = _universe_members()
    return {int(c): int(g) for c, g in zip(members["cik"], members["gsector"])}


# ---------------------------------------------------------------- filings
class SecFilingSource:
    """Observes SEC submissions (acceptance time) and companyfacts (data availability)."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.tolerance = int(config.get("report_matching_tolerance_days", 10))
        self.config = config
        self.required = list(config["required_facts"])
        self._sectors: dict[int, int] | None = None

    def _required_for(self, cik: int) -> list[str]:
        if self._sectors is None:
            self._sectors = _sector_by_cik()
        return required_facts_for(self.config, self._sectors.get(cik), cik)

    def resolutions(self) -> dict[str, dict[str, object]]:
        """Evidenced CORPORATE_ACTION_RESOLVED cases keyed by filing_key (final trading end only)."""
        rule = self.config.get("readiness_rule") or {}
        out: dict[str, dict[str, object]] = {}
        for cik, item in (rule.get("corporate_action_resolutions") or {}).items():
            if item.get("status") != "CORPORATE_ACTION_RESOLVED" or not item.get("final_trading_end") \
                    or not item.get("evidence"):
                continue
            for key in item.get("applies_to_filing_keys", []):
                out[key] = {"cik": cik, "ticker": item["ticker"], "event": item["event"],
                            "evidence": item["evidence"]}
        return out

    def _match(self, filings: list[dict], period: pd.Timestamp, form: str) -> dict | None:
        hits = [f for f in filings if f["form"] == form
                and abs((pd.Timestamp(f["reportDate"]) - period).days) <= self.tolerance]
        return min(hits, key=lambda f: f["acceptanceDateTime"]) if hits else None

    def _facts_ready(self, cik: int, accession: str, required: Sequence[str] | None = None) -> bool:
        facts = se.get_company_facts(cik, max_age_seconds=300)
        gaap = ((facts or {}).get("facts") or {}).get("us-gaap") or {}
        tags = {**se.DURATION_TAGS, **se.INSTANT_TAGS}
        names = self.required if required is None else required
        fallback = None
        for name in names:
            if any(e.get("accn") == accession
                   for tag in tags[name] for units in (gaap.get(tag, {}).get("units") or {}).values()
                   for e in units):
                continue
            # The original filing may already carry the fact although companyfacts does not (yet).
            if fallback is None:
                fallback = ixf.fallback_entries(
                    cik, {**{f: (t, False) for f, t in se.DURATION_TAGS.items()},
                          **{f: (t, True) for f, t in se.INSTANT_TAGS.items()}}) if ixf.enabled_for(cik) else {}
            if not any(e["accn"] == accession for e in fallback.get(name, [])):
                return False
        return True

    def observe(self, expected: Sequence[Mapping[str, str]]) -> dict[str, FilingObservation]:
        result: dict[str, FilingObservation] = {}
        for item in expected:
            cik = int(item["company_id"])
            submissions = se.get_submissions(cik, max_age_seconds=300)
            match = self._match(_filings(submissions), pd.Timestamp(item["expected_period"]),
                                item["filing_type"])
            if match is None:
                continue
            accepted = pd.Timestamp(match["acceptanceDateTime"]).tz_convert("UTC").isoformat()
            ready = self._facts_ready(cik, match["accessionNumber"], self._required_for(cik))
            result[item["filing_key"]] = FilingObservation(
                True, ready, True, accepted, f"sec:{match['accessionNumber']}",
                None if ready else "report accepted; required XBRL facts not yet in companyfacts",
            )
        return result


# ---------------------------------------------------------------- market data
class YahooMarket:
    """Valuation: raw close for holdings (dividends booked separately), dividend-adjusted close for the
    SPY comparison only - so a distribution is never counted twice for either side."""

    PAY_DATE_LOOKBACK_DAYS = 45

    def __init__(self, config: Mapping[str, object]) -> None:
        self.benchmark_symbol = str(config["benchmark"]["symbol"])
        self.dividend_policy = str(config["dividend_cash_policy"])
        if self.dividend_policy != "receivable_on_ex_date_cash_on_pay_date":
            raise ValidationError("unsupported dividend_cash_policy")

    @staticmethod
    def _value(symbol: str, session: str, column: str) -> float | None:
        history = pr.get_price_history(symbol)
        if history is None or column not in history.columns:
            return None
        rows = history.loc[history.index.normalize() == pd.Timestamp(session), column].dropna()
        return float(rows.iloc[-1]) if len(rows) else None

    def close(self, symbol: str, session: str) -> float | None:
        return self._value(symbol, session, "close")

    def benchmark(self, session: str) -> float | None:
        return self._value(self.benchmark_symbol, session, "adj_close")

    @staticmethod
    def _nasdaq_calendar(ex_day: str) -> list[dict] | None:
        """Nasdaq dividend calendar for one ex-date (cached); carries the payment date."""
        import requests

        cache = REPO_ROOT / "cache" / "nasdaq_dividends" / f"{ex_day}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        try:
            response = requests.get("https://api.nasdaq.com/api/calendar/dividends", params={"date": ex_day},
                                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=20)
            response.raise_for_status()
            rows = (((response.json().get("data") or {}).get("calendar") or {}).get("rows"))
        except (requests.RequestException, ValueError):
            return None
        if rows is None:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
        return rows

    def pay_date(self, symbol: str, ex_day: str, amount: float) -> str | None:
        """Documented payment date of a cash dividend, or None (never guessed)."""
        rows = self._nasdaq_calendar(ex_day)
        for row in rows or []:
            if str(row.get("symbol", "")).replace(".", "-").upper() != symbol.upper():
                continue
            rate, paid = row.get("dividend_Rate"), row.get("payment_Date")
            if rate is None or not paid or abs(float(rate) - amount) > max(0.011, 0.02 * amount):
                continue
            try:
                day = pd.to_datetime(paid, format="%m/%d/%Y")
            except ValueError:
                continue
            if day >= pd.Timestamp(ex_day):
                return day.date().isoformat()
        return None

    def corporate_actions(self, session: str, symbols: Sequence[str]) -> list[CorporateAction]:
        import yfinance as yf

        day = pd.Timestamp(session)
        actions: list[CorporateAction] = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            if splits is not None and not splits.empty:
                hit = splits[pd.DatetimeIndex(splits.index).tz_localize(None).normalize() == day]
                for value in hit:
                    if float(value) > 0:
                        actions.append(CorporateAction(f"yahoo:{symbol}:split:{session}", symbol, "split",
                                                       float(value), "yfinance:split"))
            dividends = ticker.dividends
            if dividends is None or dividends.empty:
                continue
            index = pd.DatetimeIndex(dividends.index).tz_localize(None).normalize()
            window = (index <= day) & (index >= day - pd.Timedelta(days=self.PAY_DATE_LOOKBACK_DAYS))
            for ex_ts, value in zip(index[window], dividends[window]):
                if float(value) <= 0:
                    continue
                ex_day = ex_ts.date().isoformat()
                # An unknown pay date still yields the ex-date claim (receivable, no purchasing power);
                # the date is attached later once the calendar documents it.
                paid = self.pay_date(symbol, ex_day, float(value))
                actions.append(CorporateAction(
                    f"yahoo:{symbol}:dividend:{ex_day}", symbol, "dividend", float(value),
                    "yfinance:dividend amount; nasdaq:payment_Date" if paid else "yfinance:dividend amount",
                    pay_date=paid, ex_date=ex_day))
        return actions


# ---------------------------------------------------------------- model
def _universe_members() -> pd.DataFrame:
    universe = uv.get_sp500_universe().copy()
    universe["cik"] = universe["cik"].astype(int)
    return universe.drop_duplicates("tic")


def _historical_sector(sector: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    path = REPO_ROOT / "Data" / "1-focasting_data" / f"sector{sector}_clean.xlsx"
    frame = pd.read_excel(path)
    frame["tradedate"] = pd.to_datetime(frame["tradedate"].astype("Int64").astype(str), format="%Y%m%d",
                                         errors="coerce")
    frame = frame[frame["tradedate"] < cutoff].copy()
    for name in FEATURES:
        if name not in frame.columns:
            frame[name] = np.nan
    return frame[["tic", "tradedate", "y_return"] + FEATURES]


def sector_feature_set(sector: int, historical: pd.DataFrame | None = None) -> list[str]:
    """Features the sector's historical training data actually carries (a model-design fact, not live data).

    Compustat-era sector files do not contain every ratio for every industry (e.g. no current-asset
    ratios for financials). The same set is used for training and for the live rows.
    """
    cache = REPO_ROOT / "cache" / "sector_feature_sets.json"
    path = REPO_ROOT / "Data" / "1-focasting_data" / f"sector{sector}_clean.xlsx"
    stamp = f"{path.stat().st_size}:{int(path.stat().st_mtime)}"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    entry = store.get(str(sector))
    if entry and entry["stamp"] == stamp:
        return list(entry["features"])
    frame = historical if historical is not None else _historical_sector(sector, HISTORICAL_CUTOFF)
    features = [f for f in FEATURES if frame[f].notna().any()]
    store[str(sector)] = {"stamp": stamp, "features": features}
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(store, indent=1, sort_keys=True))
    return features


def _add_historical_timing(frame: pd.DataFrame, following: pd.Timestamp) -> pd.DataFrame:
    """Compustat rows: signal at the prior close of the trade date; label known at the next one."""
    dates = sorted(frame["tradedate"].dropna().unique())
    nxt = {d: (dates[i + 1] if i + 1 < len(dates) else following) for i, d in enumerate(dates)}
    signal = {d: session_close_time(previous_nyse_session(first_session_on_or_after(d))) for d in dates}
    known = {d: session_close_time(first_session_on_or_after(nxt[d])) for d in dates}
    frame = frame.copy()
    frame["signal_time"] = frame["tradedate"].map(signal)
    frame["label_known_at"] = frame["tradedate"].map(known)
    return frame


class ModelSignalProvider:
    """``predict_as_of``: point-in-time panel -> sector models -> top quantile -> three weightings."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = config
        self.model_cfg = dict(config["model"])
        self.required = list(config["required_facts"])
        self.universe_as_of: pd.Timestamp | None = None
        self.expected_periods: dict[int, pd.Timestamp] = {}
        # Diagnostic runs may rank every ticker with a pending row; production only trades eligible ones.
        self.enforce_eligibility = True

    def prepare_cycle(self, cycle_key: str) -> None:
        """Use the universe frozen in the cycle manifest, not the (later) snapshot date."""
        path = resolve_path(self.config, "manifest_dir") / f"{cycle_key}.json"
        if not path.exists():
            raise MissingDataError(f"manifest {path} not found; cannot fix the cycle universe")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.universe_as_of = pd.Timestamp(manifest["universe_as_of"])
        self.expected_periods = {int(e["company_id"]): pd.Timestamp(e["expected_period"])
                                 for e in manifest.get("expected", [])}

    def _eligibility(self, tic: str, cik: int, used: Mapping[str, object] | None) -> tuple[bool, str]:
        """A forecast may enter the cycle only when it rests on the report the cycle expects."""
        rule = self.config.get("readiness_rule") or {}
        if str(cik) in (rule.get("corporate_action_resolutions") or {}):
            return False, "CORPORATE_ACTION_RESOLVED"
        if str(cik) in (rule.get("buy_blocks") or {}):
            return False, "BUY_BLOCK"
        if not self.expected_periods:
            return False, "NO_CYCLE_MANIFEST"
        expected = self.expected_periods.get(cik)
        if expected is None:
            return False, "NOT_EXPECTED_IN_CYCLE"
        if used is None:
            return False, "NO_REPORT_ROW"
        tolerance = int(self.config.get("report_matching_tolerance_days", 10))
        if abs((pd.Timestamp(used["datadate"]) - expected).days) > tolerance:
            return False, "PERIOD_NOT_CURRENT_CYCLE"
        return True, "OK"

    def _live_rows(self, universe: pd.DataFrame, snapshot: pd.Timestamp,
                   price_session: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
        rows, prices_by_tic, skipped = [], {}, {}
        for row in universe.itertuples():
            tic, cik = str(row.tic), int(row.cik)
            acceptance = se.acceptance_times(se.get_submissions(cik, max_age_seconds=6 * 3600))
            with se.point_in_time(snapshot, acceptance):
                fund = se.get_quarterly_fundamentals(cik)
            history = pr.get_price_history(tic)
            if fund is None or fund.empty:
                skipped[tic] = "NO_SEC_FUNDAMENTALS"
                continue
            if history is None or history.empty:
                skipped[tic] = "NO_PRICE_HISTORY"
                continue
            required = required_facts_for(self.config, int(row.gsector), cik)
            missing_cols = [c for c in required if c not in fund.columns]
            if missing_cols:
                skipped[tic] = f"MISSING_REQUIRED_FACTS:{','.join(missing_cols)}"
                continue
            fund = fund.dropna(subset=required)
            if fund.empty:
                skipped[tic] = "NO_ROW_WITH_REQUIRED_FACTS"
                continue
            try:
                ratios = ra.compute_ratios(fund, history, tic, price_session=price_session)
            except ValueError as exc:
                skipped[tic] = f"RATIO_ERROR:{exc}"
                continue
            ratios["gsector"] = int(row.gsector)
            ratios["is_current"] = bool(row.is_current)
            ratios["cik"] = cik
            accession = fund["NetIncome_accn"] if "NetIncome_accn" in fund.columns else None
            ratios["accession"] = [str(accession.get(pd.Timestamp(d), "")) if accession is not None else ""
                                   for d in ratios["datadate"]]
            ratios["published_at"] = [
                acceptance[a].isoformat() if a in acceptance else "" for a in ratios["accession"]]
            rows.append(ratios)
            prices_by_tic[tic] = history
        if not rows:
            raise MissingDataError("no SEC fundamentals could be assembled")
        return pd.concat(rows, ignore_index=True), prices_by_tic, skipped

    def predict_as_of(self, signal_as_of: str) -> SignalResult:
        import ml_model

        snapshot = pd.Timestamp(signal_as_of)
        if snapshot.tzinfo is None:
            raise ValidationError("signal_as_of must be timezone-aware")
        snapshot = snapshot.tz_convert("UTC")
        sessions = nyse_sessions(snapshot.normalize() - pd.Timedelta(days=10), snapshot.normalize())
        closed = [d for d in sessions if session_close_time(d) <= snapshot]
        if not closed:
            raise MissingDataError("no NYSE session closed before the snapshot")
        price_session = closed[-1]
        as_of_day = self.universe_as_of or snapshot.normalize().tz_localize(None)
        universe, _evidence = ua.members_as_of(as_of_day)
        universe = universe.drop_duplicates("tic").assign(cik=lambda f: f["cik"].astype(int))
        live, prices_by_tic, skipped = self._live_rows(universe, snapshot, price_session)

        pending_candidates = live.loc[live["tradedate"] > price_session, "tradedate"]
        if pending_candidates.empty:
            raise MissingDataError("no pending trade date can be derived from the available reports")
        pending = pending_candidates.min()
        live = live[live["tradedate"] <= pending]

        forecasts: dict[str, dict[str, object]] = {}
        forecast_records: list[dict[str, object]] = []
        cik_by_tic = {str(t): int(c) for t, c in zip(universe["tic"], universe["cik"])}
        selected_parts, sector_models, sector_counts, sector_feature_sets = [], {}, {}, {}
        for sector, sector_live in live.groupby("gsector"):
            sector = int(sector)
            historical = _historical_sector(sector, HISTORICAL_CUTOFF)
            features = sector_feature_set(sector, historical)
            live_part = sector_live[sector_live["tradedate"] >= HISTORICAL_CUTOFF].copy()
            live_part = live_part.sort_values(["tic", "tradedate", "datadate"]).drop_duplicates(
                ["tic", "tradedate"], keep="last")
            historical = _add_historical_timing(historical, live_part["tradedate"].min())
            cols = ["tic", "tradedate", "y_return", "signal_time", "label_known_at"] + FEATURES
            data = pd.concat([historical[cols], live_part[cols]], ignore_index=True)
            data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
            min_cov = float(self.model_cfg.get("min_feature_coverage", 0.5))
            coverage = data[features].notna().mean(axis=1)
            weak = data["tradedate"].eq(pending) & coverage.lt(min_cov)
            for tic in data.loc[weak, "tic"]:
                skipped[str(tic)] = f"LOW_FEATURE_COVERAGE<{min_cov}"
            data = data[~weak]
            live_cov = live_part[features].replace([np.inf, -np.inf], np.nan).notna().mean(axis=1)
            used_rows = {
                str(r.tic): {"datadate": r.datadate, "accession": r.accession, "published_at": r.published_at,
                             "coverage": float(c)}
                for r, c in zip(live_part.itertuples(), live_cov)
                if r.tradedate == pending and c >= min_cov
            }
            trade_rows = data[data["tradedate"] == pending]
            if trade_rows.empty:
                continue
            dates = sorted(data["tradedate"].unique())
            tickers = sorted(data["tic"].unique())
            result = ml_model.run_4model(
                data, features, "y_return", "tradedate", "tic", tickers, dates, [pending],
                self.model_cfg["first_trade_date_index"], self.model_cfg["testing_windows"],
                self.model_cfg["max_rolling_window_index"], trade_start_date=pending)
            predicted = result[4].loc[pending].dropna().astype(float)
            members = set(universe.loc[universe["is_current"].astype(bool), "tic"])
            predicted = predicted[[t for t in predicted.index if t in members]]
            if predicted.empty:
                continue
            eligibility = {t: self._eligibility(t, cik_by_tic[t], used_rows.get(t)) for t in predicted.index}
            pool = predicted[[t for t in predicted.index
                              if eligibility[t][0] or (not self.enforce_eligibility and eligibility[t][1]
                                                       not in ("CORPORATE_ACTION_RESOLVED", "BUY_BLOCK"))]]
            if pool.empty:
                continue
            threshold = pool.quantile(self.model_cfg["selection_quantile"])
            chosen = pool[pool >= threshold]
            model_name = str(result[5].loc[pending, "model_name"])
            sector_models[str(sector)] = model_name
            sector_feature_sets[str(sector)] = features
            sector_counts[str(sector)] = {"predicted": int(len(predicted)), "selected": int(len(chosen))}
            for tic, value in predicted.items():
                ok, reason = eligibility[tic]
                forecasts[tic] = {"expected_return": float(value), "sector": sector, "model": model_name,
                                  "selected": bool(tic in chosen.index), "eligible": ok,
                                  "eligibility_reason": reason}
                used = used_rows.get(tic, {})
                expected = self.expected_periods.get(cik_by_tic[tic])
                forecast_records.append({
                    "security_id": tic, "issuer_cik": cik_by_tic[tic], "universe_member": True,
                    "gsector": sector,
                    "expected_period": expected.date().isoformat() if expected is not None else None,
                    "used_period": pd.Timestamp(used["datadate"]).date().isoformat() if used else None,
                    "accession": used.get("accession") or None, "published_at": used.get("published_at") or None,
                    "feature_coverage": used.get("coverage"), "expected_return": float(value),
                    "eligible_for_current_cycle": ok, "eligibility_reason": reason,
                    "forecast_kind": "ELIGIBLE" if ok else "DIAGNOSTIC_ONLY"})
            selected_parts.append(chosen)
        if not selected_parts:
            raise MissingDataError("the sector models produced no forecasts")
        selected = pd.concat(selected_parts)
        selected = selected[~selected.index.duplicated()]
        panel = pd.concat({t: prices_by_tic[t]["adj_close"] for t in selected.index if t in prices_by_tic},
                          axis=1).sort_index()
        panel = panel.loc[:price_session]
        try:
            weights = {
                "equal_weight_v1": capped_equal_weights(selected.index.tolist()),
                "mean_var_v1": optimize_weights(selected, panel, price_session, fallback_to_equal=False),
                "min_var_v1": optimize_weights(selected, panel, price_session, minimum_variance=True,
                                               fallback_to_equal=False),
            }
        except OptimizationError as exc:
            raise MissingDataError(f"portfolio optimisation failed: {exc}") from exc
        for name, series in weights.items():
            if series.empty:
                raise MissingDataError(f"{name} produced no weights (insufficient price history)")
        fingerprint = hashlib.sha256(
            live[["tic", "datadate", "available_at", "tradedate"]].astype(str).to_csv(index=False).encode()
        ).hexdigest()
        iso = snapshot.isoformat()
        return SignalResult(
            signal_as_of=iso,
            sector_as_of={f"sector{s}": iso for s in sector_models},
            data_snapshot={
                "origin": REAL_ORIGIN, "price_session": price_session.date().isoformat(),
                "pending_tradedate": pending.date().isoformat(), "panel_fingerprint": fingerprint,
                "skipped_tickers": sorted(skipped), "skip_reasons": dict(sorted(skipped.items())),
                "universe_as_of": as_of_day.date().isoformat(),
                "sector_features": sector_feature_sets, "sector_models": sector_models,
                "sector_counts": sector_counts, "forecast_records": forecast_records,
                "eligibility_enforced": self.enforce_eligibility,
            },
            forecasts=forecasts,
            target_weights={n: {t: float(w) for t, w in s.items() if w > 1e-9} for n, s in weights.items()},
        )


def real_adapters(config_path: str | None = None) -> Adapters:
    se.require_user_agent()
    config = load_config(config_path)
    return Adapters(SecFilingSource(config), ModelSignalProvider(config), YahooMarket(config),
                    REAL_ORIGIN, environment="real")
