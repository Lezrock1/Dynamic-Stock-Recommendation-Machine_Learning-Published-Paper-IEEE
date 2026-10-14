"""Orchestration of the virtual paper portfolio: filings -> signal -> orders -> marks.

All state lives in :class:`PaperStore`. Every step is idempotent (deterministic
ids plus store uniqueness rules), so a restart repeats no booking. External data
comes exclusively from injected adapters; nothing is invented when data is missing.
"""
from __future__ import annotations

import fcntl
import importlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Mapping, Protocol, Sequence

import pandas as pd

try:
    from .market_calendar import nyse_schedule, nyse_sessions, previous_nyse_session, session_close_time
    from .paper_store import (
        DuplicateRecordError, JobStatus, NotFoundError, OrderRequest, OrderSide, PaperStore,
        StateError, ValidationError,
    )
except ImportError:  # pragma: no cover - script execution
    from market_calendar import nyse_schedule, nyse_sessions, previous_nyse_session, session_close_time  # type: ignore
    from paper_store import (  # type: ignore
        DuplicateRecordError, JobStatus, NotFoundError, OrderRequest, OrderSide, PaperStore,
        StateError, ValidationError,
    )


BENCHMARK_STRATEGY = "spy_buy_hold_v1"
BENCHMARK_SYMBOL = "SPY"


class RunnerError(Exception):
    pass


class MissingDataError(RunnerError):
    """Required external data is not available; the step is retried, never guessed."""


class LockBusyError(RunnerError):
    pass


@dataclass(frozen=True)
class FilingObservation:
    received: bool
    facts_ready: bool = False
    valid: bool = True
    filed_at: str | None = None  # SEC acceptance timestamp (publication time), timezone-aware
    source: str = ""
    error_reason: str | None = None


@dataclass(frozen=True)
class SignalResult:
    """Output of one ``predict_as_of`` run; all sectors share ``signal_as_of``."""

    signal_as_of: str  # full ISO timestamp (snapshot cut-off), never a bare date
    sector_as_of: Mapping[str, str]
    data_snapshot: Mapping[str, object]
    forecasts: Mapping[str, Mapping[str, object]]
    target_weights: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class CorporateAction:
    action_id: str
    symbol: str
    action_type: str
    value: float
    source: str
    pay_date: str | None = None
    ex_date: str | None = None  # defaults to the processed session; set for late-resolved pay dates


class FilingSource(Protocol):
    def observe(self, expected: Sequence[Mapping[str, str]]) -> Mapping[str, FilingObservation]: ...


class SignalProvider(Protocol):
    def predict_as_of(self, signal_as_of: str) -> SignalResult: ...


class MarketData(Protocol):
    def close(self, symbol: str, session: str) -> float | None: ...
    def benchmark(self, session: str) -> float | None: ...
    def corporate_actions(self, session: str, symbols: Sequence[str]) -> Sequence[CorporateAction]: ...


@dataclass(frozen=True)
class Adapters:
    filings: FilingSource
    signals: SignalProvider
    market: MarketData
    data_origin: str
    environment: str = "synthetic"  # 'real' adapters may only write to a 'real' store


@dataclass(frozen=True)
class RunnerConfig:
    max_weight: float = 0.05
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 0.0
    cash_buffer_bps: float = 50.0
    backoff_base_seconds: float = 60.0
    backoff_max_seconds: float = 6 * 3600.0


def load_adapters(spec: str) -> Adapters:
    """Load ``package.module:factory``; the factory must return :class:`Adapters`."""
    module_name, _, factory = spec.partition(":")
    if not module_name or not factory:
        raise ValidationError("adapter spec must look like 'package.module:factory'")
    adapters = getattr(importlib.import_module(module_name), factory)()
    if not isinstance(adapters, Adapters):
        raise ValidationError("adapter factory must return paper_runner.Adapters")
    return adapters


@contextmanager
def exclusive_lock(database: str | Path) -> Iterator[None]:
    path = Path(str(database) + ".lock")
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockBusyError(f"another scheduler holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _aware_utc(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(str(value))
    return stamp.tz_convert("UTC") if stamp.tzinfo is not None else None


def _iso(value: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(value).isoformat()


class PaperRunner:
    def __init__(
        self,
        store: PaperStore,
        adapters: Adapters,
        config: RunnerConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.adapters = adapters
        self.config = config or RunnerConfig()
        store_env = store.environment()
        if (adapters.environment == "real") != (store_env == "real"):
            raise ValidationError(
                f"{adapters.environment} adapters cannot run on a {store_env!r} store; "
                "real paper tracking and synthetic demos must stay separate"
            )
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self._clock()).tz_convert("UTC")

    # ----- job wrapper: persistent attempts, backoff, crash recovery -----
    def recover_interrupted(self) -> int:
        """Mark RUNNING jobs left by a crashed process as FAILED (lock is held)."""
        rows = self.store.query("SELECT run_id FROM job_runs WHERE status = 'RUNNING'")
        for row in rows:
            self.store.record_job_error(
                str(row["run_id"]), "interrupted by process exit", {"recovered_at": _iso(self.now())}
            )
        return len(rows)

    def _job(self, base: str, job_type: str, payload: Mapping[str, object],
             fn: Callable[[str], object]) -> tuple[str, object]:
        now = self.now()
        rows = [
            r for r in self.store.query("SELECT run_id, status, finished_at FROM job_runs")
            if r["run_id"] == base or str(r["run_id"]).startswith(base + ":a")
        ]
        if any(r["status"] == JobStatus.SUCCEEDED.value for r in rows):
            return "DONE", None
        if rows:
            last = max(pd.Timestamp(r["finished_at"]) for r in rows if r["finished_at"])
            delay = min(
                self.config.backoff_base_seconds * 2 ** (len(rows) - 1),
                self.config.backoff_max_seconds,
            )
            if now < last + timedelta(seconds=delay):
                return "BACKOFF", None
        run_id = f"{base}:a{len(rows) + 1}"
        self.store.start_job(
            run_id, job_type, "started",
            {**payload, "data_origin": self.adapters.data_origin}, started_at=_iso(now),
        )
        try:
            result = fn(run_id)
        except Exception as exc:  # noqa: BLE001 - persisted, then retried with backoff
            self.store.record_job_error(
                run_id, f"{type(exc).__name__}: {exc}", {"job_type": job_type},
                created_at=_iso(self.now()),
            )
            return "FAILED", exc
        self.store.finish_job(run_id, finished_at=_iso(self.now()))
        return "SUCCEEDED", result

    # ----- helpers -----
    def _cycles(self, cycle_key: str) -> list[dict[str, object]]:
        return self.store.query(
            "SELECT * FROM reporting_cycles WHERE cycle_key = ? ORDER BY portfolio_id", (cycle_key,)
        )

    def open_cycle_keys(self) -> list[str]:
        rows = self.store.query(
            "SELECT DISTINCT cycle_key FROM reporting_cycles WHERE status = 'OPEN' ORDER BY cycle_key"
        )
        return [str(r["cycle_key"]) for r in rows]

    def latest_closed_session(self) -> pd.Timestamp:
        now = self.now()
        days = nyse_sessions(now.normalize() - pd.Timedelta(days=10), now.normalize())
        closed = [d for d in days if session_close_time(d) <= now]
        if not closed:
            raise MissingDataError("no closed NYSE session found")
        return closed[-1]

    def _price(self, symbol: str, session: str) -> float:
        price = self.adapters.market.close(symbol, session)
        if price is None or not math.isfinite(price) or price <= 0:
            raise MissingDataError(f"no close price for {symbol} on {session}")
        return float(price)

    # ----- 1. filing availability -----
    def check_filings(self, cycle_key: str) -> dict[str, object]:
        now = self.now()
        cycles = [c for c in self._cycles(cycle_key) if c["status"] == "OPEN"]
        if not cycles:
            return {"cycle_key": cycle_key, "open_cycles": 0}
        expected = self.store.query(
            "SELECT filing_key, company_id, security_id, expected_period, filing_type "
            "FROM expected_filings WHERE cycle_id = ? ORDER BY filing_key",
            (cycles[0]["cycle_id"],),
        )
        summary: dict[str, object] = {}

        def work(run_id: str) -> None:
            pending = {str(r["filing_key"]) for r in self.store.query(
                "SELECT filing_key FROM expected_filings WHERE cycle_id = ? "
                "AND (received = 0 OR facts_ready = 0)", (cycles[0]["cycle_id"],))}
            to_observe = [{k: str(v) for k, v in e.items()} for e in expected
                          if str(e["filing_key"]) in pending]
            observed = self.adapters.filings.observe(to_observe) if to_observe else {}
            resolutions = getattr(self.adapters.filings, "resolutions", lambda: {})()
            changes: list[dict[str, object]] = []
            for cycle in cycles:
                cid = str(cycle["cycle_id"])
                for row in self.store.query(
                    "SELECT * FROM expected_filings WHERE cycle_id = ?", (cid,)
                ):
                    obs = observed.get(str(row["filing_key"]))
                    if obs is None:
                        continue
                    received = bool(row["received"]) or obs.received
                    facts = bool(row["facts_ready"]) or obs.facts_ready
                    filed_at = row["filed_at"] or obs.filed_at
                    valid = obs.valid
                    reason = obs.error_reason
                    if received and not filed_at:
                        valid, reason = False, "received without filed_at timestamp"
                    elif received and not _aware_utc(filed_at):
                        received, facts, filed_at = False, False, None
                        valid, reason = False, "filed_at lacks a timezone"
                    elif received and _aware_utc(filed_at) > now:
                        received, facts, filed_at = False, False, None
                        valid, reason = False, "filed_at lies in the future"
                    new = (received, facts, valid, filed_at)
                    if new == (bool(row["received"]), bool(row["facts_ready"]),
                               bool(row["valid"]), row["filed_at"]):
                        continue
                    self.store.set_filing_result(
                        cid, str(row["filing_key"]), received=received, facts_ready=facts,
                        valid=valid, filed_at=filed_at,
                        first_seen_at=row["first_seen_at"] or _iso(now),
                        source=obs.source or None, error_reason=reason, updated_at=_iso(now),
                    )
                    if cid == cycles[0]["cycle_id"]:
                        changes.append({"filing_key": row["filing_key"], "received": received,
                                        "facts_ready": facts, "valid": valid, "filed_at": filed_at})
                for key, evidence in resolutions.items():
                    if self.store.query("SELECT 1 FROM expected_filings WHERE cycle_id = ? AND filing_key = ?",
                                        (cid, key)):
                        self.store.resolve_filing(cid, key, evidence, recorded_at=_iso(now))
                self._apply_deadline(cycle, now)
            counts = self.store.query(
                "SELECT status, COUNT(*) AS n FROM expected_filings WHERE cycle_id = ? GROUP BY status",
                (cycles[0]["cycle_id"],),
            )
            summary.update({"cycle_key": cycle_key, "status_counts": {r["status"]: r["n"] for r in counts},
                            "changes": changes})
            self.store.log_job_event(run_id, "filing_availability", summary, created_at=_iso(now))

        base = f"check:{cycle_key}:{now.strftime('%Y%m%dT%H%M%S%f')}"
        status, _ = self._job(base, "filing_check", {"cycle_key": cycle_key}, work)
        return {"status": status, **summary}

    def _apply_deadline(self, cycle: Mapping[str, object], now: pd.Timestamp) -> None:
        deadline = cycle["deadline_at"]
        if not deadline or now <= pd.Timestamp(str(deadline)):
            return
        if self.store.filing_status(str(cycle["cycle_id"])).value != "READY":
            self.store.set_cycle_model_status(
                str(cycle["cycle_id"]), "BLOCKED",
                reason="deadline passed with incomplete expected filings; quota not lowered",
            )

    # ----- 2./3./4. signal, snapshots, orders -----
    def run_signal(self, cycle_key: str) -> str:
        cycles = [c for c in self._cycles(cycle_key) if c["status"] == "OPEN"]
        if not cycles:
            return "NO_OPEN_CYCLE"
        if any(c["model_status"] == "BLOCKED" for c in cycles):
            return "BLOCKED"
        if any(self.store.filing_status(str(c["cycle_id"])).value != "READY" for c in cycles):
            return "WAITING_FOR_FILINGS"
        status, error = self._job(
            f"signal:{cycle_key}", "model_signal", {"cycle_key": cycle_key},
            lambda run_id: self._signal_work(cycle_key, cycles),
        )
        if status == "FAILED":
            for cycle in cycles:
                self.store.set_cycle_model_status(str(cycle["cycle_id"]), "FAILED", reason=str(error)[:500])
        return status

    def _timeline_inputs(self, cycle: Mapping[str, object]) -> dict[str, str]:
        """Use actual filing timestamps plus recorded special-case evidence, never invented filing times."""
        rows = self.store.query(
            "SELECT e.filing_key, e.filed_at, e.first_seen_at, r.recorded_at "
            "FROM expected_filings e LEFT JOIN filing_resolutions r "
            "ON r.cycle_id = e.cycle_id AND r.filing_key = e.filing_key "
            "WHERE e.cycle_id = ?",
            (cycle["cycle_id"],),
        )
        published, seen = [], []
        for r in rows:
            p, f = _aware_utc(r["filed_at"]), _aware_utc(r["first_seen_at"])
            resolved_at = _aware_utc(r["recorded_at"])
            if resolved_at is not None:
                seen.append(resolved_at)
                continue
            if p is None or f is None:
                raise ValidationError(f"filing {r['filing_key']} needs timezone-aware publication and first-seen times")
            published.append(p)
            seen.append(f)
        if not published:
            raise ValidationError("cycle has no published filing timestamp; resolutions cannot supply one")
        latest_seen = max(seen)
        latest_published = max(published)
        if latest_seen < latest_published:
            raise ValidationError("first-seen evidence time precedes a filing publication time")
        return {"published_at": _iso(max(published)), "first_seen_at": _iso(latest_seen),
                "snapshot_as_of": _iso(latest_seen)}

    def _signal_work(self, cycle_key: str, cycles: list[dict[str, object]]) -> None:
        now = self.now()
        stored = self.store.query(
            "SELECT payload_json FROM data_snapshots WHERE instr(snapshot_id, ?) = 1 LIMIT 1",
            (f"model:{cycle_key}:",),
        )
        if stored:
            payload = json.loads(str(stored[0]["payload_json"]))
        else:
            timeline = self._timeline_inputs(cycles[0])
            snapshot = timeline["snapshot_as_of"]
            if pd.Timestamp(snapshot) > now:
                raise RunnerError("snapshot cut-off lies in the future")
            prepare = getattr(self.adapters.signals, "prepare_cycle", None)
            if prepare is not None:
                prepare(cycle_key)
            result = self.adapters.signals.predict_as_of(snapshot)
            if any(c["strategy"] == BENCHMARK_STRATEGY for c in cycles):
                # Same snapshot, execution session and cost rules as the model portfolios; no model input.
                result.target_weights.setdefault(BENCHMARK_STRATEGY, {BENCHMARK_SYMBOL: 1.0})
            payload = self._validated_payload(result, snapshot, cycles)
            payload["timeline"] = {**timeline, "model_completed_at": _iso(self.now())}
        timeline = payload["timeline"]
        completed = pd.Timestamp(timeline["model_completed_at"])
        for cycle in cycles:
            cid = str(cycle["cycle_id"])
            try:
                self.store.record_data_snapshot(
                    cid, f"model:{cycle_key}:{cid}", payload, captured_at=_iso(completed))
            except DuplicateRecordError:
                pass
            self.store.record_cycle_timeline(cid, **timeline)
            weights = payload["target_weights"][cycle["strategy"]]
            for symbol, weight in sorted(weights.items()):
                try:
                    self.store.record_signal_snapshot(
                        cid, str(cycle["strategy"]), symbol,
                        {"signal_as_of": payload["signal_as_of"], "weight": weight,
                         "forecast": payload["forecasts"].get(symbol)},
                        signal_id=f"sig:{cid}:{symbol}", created_at=_iso(completed))
                except DuplicateRecordError:
                    pass
            self.store.set_cycle_model_status(cid, "READY", reason="predict_as_of completed")
            self._plan_orders(cycle, weights, completed)
            self.store.set_cycle_status(cid, "SIGNALLED")

    def _validated_payload(self, result: SignalResult, signal_as_of: str,
                           cycles: list[dict[str, object]]) -> dict[str, object]:
        if result.signal_as_of != signal_as_of:
            raise ValidationError("signal_as_of differs from the requested snapshot timestamp")
        if not result.sector_as_of or any(v != signal_as_of for v in result.sector_as_of.values()):
            raise ValidationError("all sectors must share the common snapshot timestamp")
        data = result.data_snapshot
        if data.get("eligibility_enforced") is False:
            raise ValidationError("diagnostic-only model run (eligibility not enforced) cannot create a signal")
        blocked = {r["security_id"] for r in data.get("forecast_records", [])
                   if not r.get("eligible_for_current_cycle", True)}
        for cycle in cycles:
            weights = result.target_weights.get(str(cycle["strategy"]))
            if blocked & set(weights or {}):
                raise ValidationError("target weights contain DIAGNOSTIC_ONLY forecasts: "
                                      + ", ".join(sorted(blocked & set(weights))))
            if not weights:
                raise ValidationError(f"no target weights for {cycle['strategy']}")
            values = list(weights.values())
            cap = 1.0 if cycle["strategy"] == BENCHMARK_STRATEGY else self.config.max_weight
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                   or v > cap + 1e-12 for v in values):
                raise ValidationError(f"invalid weight for {cycle['strategy']}")
            if math.fsum(values) > 1.0 + 1e-9:
                raise ValidationError(f"weights exceed 100% for {cycle['strategy']}")
        return json.loads(json.dumps({
            "signal_as_of": signal_as_of, "sector_as_of": dict(result.sector_as_of),
            "data": dict(result.data_snapshot), "forecasts": {k: dict(v) for k, v in result.forecasts.items()},
            "target_weights": {k: dict(v) for k, v in result.target_weights.items()},
        }, allow_nan=False))

    def _plan_orders(self, cycle: Mapping[str, object], weights: Mapping[str, float],
                     now: pd.Timestamp) -> None:
        cid, pid = str(cycle["cycle_id"]), str(cycle["portfolio_id"])
        existing = self.store.query(
            "SELECT b.batch_id, s.scheduled_for FROM execution_batches b "
            "LEFT JOIN execution_schedule s ON s.batch_id = b.batch_id WHERE b.cycle_id = ?", (cid,))
        if existing and existing[0]["scheduled_for"]:
            return
        ref = self.latest_closed_session().date().isoformat()
        held = {r["symbol"]: float(r["q"]) for r in self.store.query(
            "SELECT symbol, SUM(quantity_delta) AS q FROM position_ledger WHERE portfolio_id = ? "
            "GROUP BY symbol HAVING ABS(SUM(quantity_delta)) > 1e-9", (pid,))}
        prices = {s: self._price(s, ref) for s in set(held) | set(weights)}
        budget = (self.store.cash_balance(pid) + sum(q * prices[s] for s, q in held.items())) * (
            1 - self.config.cash_buffer_bps / 10_000)
        unit_cost = 1 + (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000
        orders: list[OrderRequest] = []
        if cycle["strategy"] == BENCHMARK_STRATEGY:
            prior_spy_order = self.store.query(
                "SELECT 1 FROM simulated_orders o JOIN execution_batches b ON b.batch_id = o.batch_id "
                "JOIN reporting_cycles c ON c.cycle_id = b.cycle_id "
                "WHERE c.portfolio_id = ? AND o.symbol = ? LIMIT 1",
                (pid, BENCHMARK_SYMBOL),
            )
            if not prior_spy_order and weights.get(BENCHMARK_SYMBOL, 0.0) > 0:
                target = math.floor(
                    self.store.cash_balance(pid) * (1 - self.config.cash_buffer_bps / 10_000) /
                    (prices[BENCHMARK_SYMBOL] * unit_cost)
                )
                if target > 0:
                    orders.append(OrderRequest(
                        f"{cid}:{BENCHMARK_SYMBOL}:BUY", BENCHMARK_SYMBOL, OrderSide.BUY, target
                    ))
        else:
            for symbol in sorted(set(held) | set(weights)):
                target = math.floor(weights.get(symbol, 0.0) * budget / (prices[symbol] * unit_cost))
                delta = target - held.get(symbol, 0.0)
                if abs(delta) < 1e-9:
                    continue
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                orders.append(OrderRequest(f"{cid}:{symbol}:{side.value}", symbol, side, abs(delta)))
        scheduled = self._first_session_opening_after(now)
        batch = f"batch:{cid}"
        if not existing:
            self.store.rebalance(
                cid, str(cycle["strategy"]), batch, weights, orders, created_at=_iso(now))
        self.store.schedule_batch(
            batch, scheduled, planned_at=_iso(now), note=f"reference close {ref}")

    @staticmethod
    def _first_session_opening_after(moment: pd.Timestamp) -> str:
        """Execution is only allowed in a session that opens after the signal was completed."""
        start = moment.tz_convert("America/New_York").normalize().tz_localize(None)
        schedule = nyse_schedule(start, start + pd.Timedelta(days=14))
        for day, row in schedule.iterrows():
            if pd.Timestamp(row["market_open"]) > moment:
                return pd.Timestamp(day).date().isoformat()
        raise MissingDataError("no NYSE session opens after model completion")

    # ----- 4./5. execution and daily valuation -----
    def _execute_batch(self, batch: Mapping[str, object], session: str) -> None:
        pid, cid = str(batch["portfolio_id"]), str(batch["cycle_id"])
        orders = self.store.orders_for_batch(str(batch["batch_id"]))
        stamp = _iso(session_close_time(batch["scheduled_for"]))
        sells = [o for o in orders if o.side == OrderSide.SELL]
        buys = [o for o in orders if o.side == OrderSide.BUY]
        for order in sells + buys:
            if order.status != "OPEN":
                continue
            price = self._price(order.symbol, str(batch["scheduled_for"]))
            quantity = order.quantity
            if order.side == OrderSide.BUY:
                unit = price * (1 + (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000)
                quantity = min(quantity, math.floor(max(self.store.cash_balance(pid), 0.0) / unit))
            if quantity <= 0:
                continue
            try:
                self.store.record_fill(
                    f"exec:{order.order_id}", order.order_id, quantity, price,
                    transaction_cost_bps_per_side=self.config.transaction_cost_bps,
                    spread_slippage_bps_per_side=self.config.slippage_bps, filled_at=stamp)
            except (DuplicateRecordError, StateError):
                if self.store.query(
                    "SELECT 1 FROM simulated_fills WHERE execution_id = ?", (f"exec:{order.order_id}",)
                ):
                    continue
                raise
        self.store.set_cycle_status(cid, "EXECUTED")

    def _apply_actions(self, session: str) -> None:
        stamp = _iso(session_close_time(session))
        portfolios = [str(r["portfolio_id"]) for r in self.store.query("SELECT portfolio_id FROM portfolios")]
        known_symbols = sorted(str(r["symbol"]) for r in self.store.query(
            "SELECT DISTINCT symbol FROM position_ledger"))
        if known_symbols:
            for action in self.adapters.market.corporate_actions(session, known_symbols):
                ex_day = action.ex_date or session
                for pid in portfolios:
                    entitled_quantity = None
                    if action.action_type == "dividend":
                        entitlement_session = previous_nyse_session(pd.Timestamp(ex_day)).date().isoformat()
                        entitled_quantity = self._held_at(pid, action.symbol, entitlement_session)
                        if entitled_quantity <= 1e-9:
                            continue
                    else:
                        current_quantity = self.store.position_balance(pid, action.symbol)
                        if current_quantity <= 1e-9:
                            continue
                        if ex_day != session and self._held_at(pid, action.symbol, ex_day) != current_quantity:
                            continue
                    try:
                        self.store.record_corporate_action(
                            f"{action.action_id}:{pid}", pid, action.symbol, action.action_type,
                            ex_day, action.value, action.source, pay_date=action.pay_date,
                            entitled_quantity=entitled_quantity,
                            created_at=stamp)
                    except DuplicateRecordError:
                        if action.action_type == "dividend" and action.pay_date:
                            try:
                                self.store.set_dividend_pay_date(
                                    f"{action.action_id}:{pid}", action.pay_date, action.source, recorded_at=stamp)
                            except (NotFoundError, StateError):
                                pass
        lookup = getattr(self.adapters.market, "pay_date", None)
        if lookup is not None:
            # Open claims are resolved independently of current holdings.
            for row in self.store.query(
                "SELECT d.action_id, a.symbol, a.ex_date, a.value FROM dividend_receivables d "
                "JOIN corporate_actions a ON a.action_id = d.action_id "
                "WHERE d.status = 'RECEIVABLE' AND d.pay_date IS NULL"):
                try:
                    found = lookup(str(row["symbol"]), str(row["ex_date"])[:10], float(row["value"]))
                except Exception:  # noqa: BLE001 - an unavailable source must not touch recorded claims
                    found = None
                if found:
                    try:
                        self.store.set_dividend_pay_date(str(row["action_id"]), found, "nasdaq:payment_Date",
                                                         recorded_at=stamp)
                    except (NotFoundError, StateError):
                        pass
        for row in self.store.query(
            "SELECT action_id, pay_date FROM dividend_receivables WHERE status = 'RECEIVABLE' "
            "AND pay_date IS NOT NULL"
        ):
            if str(row["pay_date"])[:10] <= session:
                self.store.pay_dividend(str(row["action_id"]), paid_at=stamp)

    def _held_at(self, portfolio_id: str, symbol: str, session: str) -> float:
        rows = self.store.query(
            "SELECT COALESCE(SUM(quantity_delta), 0) AS q FROM position_ledger "
            "WHERE portfolio_id = ? AND symbol = ? AND julianday(created_at) <= julianday(?)",
            (portfolio_id, symbol, _iso(session_close_time(session))))
        return float(rows[0]["q"])

    def _mark(self, portfolio: Mapping[str, object], session: str) -> None:
        pid = str(portfolio["portfolio_id"])
        stamp = _iso(session_close_time(session))
        if self.store.query("SELECT 1 FROM daily_marks WHERE portfolio_id = ? AND marked_at = ?", (pid, stamp)):
            return
        positions = self.store.query(
            "SELECT symbol, SUM(quantity_delta) AS q FROM position_ledger WHERE portfolio_id = ? "
            "AND julianday(created_at) <= julianday(?) "
            "GROUP BY symbol HAVING ABS(SUM(quantity_delta)) > 1e-9", (pid, stamp))
        position_values = {
            str(row["symbol"]): (float(row["q"]), self._price(str(row["symbol"]), session))
            for row in positions
        }
        value = sum(quantity * price for quantity, price in position_values.values())
        created = pd.Timestamp(str(portfolio["created_at"])).tz_convert("UTC")
        base_session = previous_nyse_session(self._first_session(created)).date().isoformat()
        base, level = self.adapters.market.benchmark(base_session), self.adapters.market.benchmark(session)
        if not base or not level or base <= 0 or level <= 0:
            raise MissingDataError(f"benchmark value missing for {base_session} or {session}")
        cash = float(self.store.query(
            "SELECT COALESCE(SUM(amount), 0) AS balance FROM cash_ledger "
            "WHERE portfolio_id = ? AND julianday(created_at) <= julianday(?)",
            (pid, stamp),
        )[0]["balance"])
        recv = float(self.store.query(
            """SELECT COALESCE(SUM(r.amount), 0) AS amount
               FROM dividend_receivables r
               JOIN corporate_actions a ON a.action_id = r.action_id
               WHERE r.portfolio_id = ? AND julianday(a.created_at) <= julianday(?)
                 AND (r.status = 'RECEIVABLE' OR julianday(r.paid_at) > julianday(?))""",
            (pid, stamp, stamp),
        )[0]["amount"])
        self.store.record_daily_mark(
            pid, stamp, cash, value, cash + value + recv,
            float(portfolio["initial_cash"]) * level / base, recv, position_values)

    @staticmethod
    def _first_session(created: pd.Timestamp) -> pd.Timestamp:
        for day in nyse_sessions(created.normalize(), created.normalize() + pd.Timedelta(days=14)):
            if session_close_time(day) > created:
                return day
        raise MissingDataError("no session after portfolio creation")

    def run_days(self) -> list[str]:
        """Process every closed session without a complete valuation, oldest first."""
        portfolios = self.store.query("SELECT * FROM portfolios ORDER BY portfolio_id")
        if not portfolios:
            return []
        latest = self.latest_closed_session()
        start = min(self._first_session(pd.Timestamp(str(p["created_at"])).tz_convert("UTC")) for p in portfolios)
        done: list[str] = []
        if start > latest:
            return done
        for day in nyse_sessions(start, latest):
            session = day.date().isoformat()
            stamp = _iso(session_close_time(session))
            marked = {r["portfolio_id"] for r in self.store.query(
                "SELECT portfolio_id FROM daily_marks WHERE marked_at = ?", (stamp,))}
            if all(p["portfolio_id"] in marked or self._first_session(
                    pd.Timestamp(str(p["created_at"])).tz_convert("UTC")) > day for p in portfolios):
                continue
            status, _ = self._job(f"day:{session}", "daily_valuation", {"session": session},
                                  lambda run_id, d=day, s=session: self._day_work(s, [
                                      p for p in portfolios
                                      if self._first_session(pd.Timestamp(str(p["created_at"])).tz_convert("UTC")) <= d]))
            if status in {"FAILED", "BACKOFF"}:
                break
            if status == "SUCCEEDED":
                done.append(session)
        return done

    def _day_work(self, session: str, portfolios: list[dict[str, object]]) -> None:
        self._apply_actions(session)
        for batch in self.store.query(
            """SELECT b.batch_id, b.cycle_id, c.portfolio_id, s.scheduled_for
               FROM execution_batches b JOIN execution_schedule s ON s.batch_id = b.batch_id
               JOIN reporting_cycles c ON c.cycle_id = b.cycle_id
               WHERE c.status = 'SIGNALLED' AND s.scheduled_for <= ? ORDER BY s.scheduled_for""",
            (session,)):
            self._execute_batch(batch, session)
        for portfolio in portfolios:
            self._mark(portfolio, session)

    # ----- scheduler -----
    def tick(self) -> dict[str, object]:
        """One locked scheduler pass; safe to call from cron/launchd or the loop."""
        with exclusive_lock(self.store.path):
            recovered = self.recover_interrupted()
            outcome: dict[str, object] = {"recovered_jobs": recovered, "signals": {}}
            for key in self.open_cycle_keys():
                self.check_filings(key)
                outcome["signals"][key] = self.run_signal(key)  # type: ignore[index]
            outcome["days_processed"] = self.run_days()
            return outcome

    def serve_forever(self, interval_seconds: float = 900.0) -> None:
        while True:
            try:
                print(json.dumps(self.tick(), default=str), flush=True)
            except LockBusyError as exc:
                print(f"skipped: {exc}", flush=True)
            time.sleep(interval_seconds)
