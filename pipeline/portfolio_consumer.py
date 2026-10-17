"""Local, resumable target-portfolio consumption; no broker adapter or quantity model.

The injected provider must return fresh, strategy/account-scoped positions, open orders
and confirmed fills. Query absence must be authoritative, not eventual-consistency lag.
Only an injected planner may size orders, using current execution data, never research CSV
prices. A committed intent precedes every external submission. This is local idempotency,
NOT a guarantee of external exactly-once execution.

SQLite commits use FULL synchronization; a POSIX advisory lock serializes cooperating
processes across external calls. Use a private local filesystem and the same ledger path
for all consumers of an account. Neither this ledger nor ProcessedLedger is a broker.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Mapping, Protocol

from . import target_portfolio as tp


class ConsumerError(ValueError):
    """Invalid input or unsafe local transition."""


@dataclass(frozen=True)
class OrderIntent:
    security_id: str
    side: str
    quantity: Decimal


@dataclass(frozen=True)
class OrderReport:
    client_key: str
    order_id: str
    status: str  # accepted, partially_filled, filled, cancelled, rejected
    filled_quantity: Decimal = Decimal(0)


@dataclass(frozen=True)
class ConfirmedFill:
    fill_id: str
    client_key: str
    order_id: str
    quantity: Decimal


@dataclass(frozen=True)
class ExecutionSnapshot:
    account: str
    strategy: str
    positions: Mapping[str, Decimal]
    open_orders: tuple[OrderReport, ...] = ()
    confirmed_fills: tuple[ConfirmedFill, ...] = ()


class ExecutionProvider(Protocol):
    """Implementations must scope every operation to the supplied account AND strategy.

    submit must honor client_key if supports_idempotency is true. lookup must search
    terminal orders too, and None must prove nonacceptance if supports_query is true.
    snapshot must include all outstanding orders and confirmed fills for this scope.
    No implementation, credentials, scheduler, or real order route is supplied here.
    """

    supports_query: bool
    supports_idempotency: bool

    def snapshot(self, account: str, strategy: str) -> ExecutionSnapshot: ...
    def lookup(self, account: str, strategy: str, client_key: str) -> OrderReport | None: ...
    def submit(self, account: str, strategy: str, client_key: str,
               intent: OrderIntent) -> OrderReport: ...
    def cancel(self, account: str, strategy: str, order_id: str) -> None: ...


Planner = Callable[[dict, ExecutionSnapshot], list[OrderIntent]]
_TERMINAL = {"filled", "cancelled", "rejected"}
_REPORT_STATES = _TERMINAL | {"accepted", "partially_filled"}


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConsumerError("Nonempty identifiers/reasons required")
    return value


def _number(value: Decimal, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0 or (positive and value == 0):
        raise ConsumerError("Quantities must be finite nonnegative Decimals (orders/fills positive)")
    return value


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class ConsumerLedger:
    """Durable state keyed by (account/depot, strategy, allocation_id).

    accept_package validates before accepting; resume uses the frozen accepted target,
    not a newly loaded/expired package or a second ProcessedLedger claim. Do not use the
    one-shot ProcessedLedger as a completion receipt: it cannot describe partial execution.
    """

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS allocations (
                    account TEXT NOT NULL, strategy TEXT NOT NULL, allocation_id TEXT NOT NULL,
                    document TEXT NOT NULL, PRIMARY KEY(account, strategy, allocation_id));
                CREATE TABLE IF NOT EXISTS bindings (
                    account TEXT NOT NULL, strategy TEXT NOT NULL, allocation_id TEXT NOT NULL,
                    PRIMARY KEY(account, strategy));
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, at_utc TEXT NOT NULL,
                    account TEXT NOT NULL, strategy TEXT NOT NULL, allocation_id TEXT NOT NULL,
                    event TEXT NOT NULL, details TEXT NOT NULL);
            """)
            db.commit()

    @contextmanager
    def _session(self):
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = sqlite3.connect(self.path, timeout=30)
            try:
                db.execute("PRAGMA synchronous=FULL")
                yield db
            finally:
                db.close()
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _get(db, account, strategy, allocation_id) -> dict:
        row = db.execute("SELECT document FROM allocations WHERE account=? AND strategy=? "
                         "AND allocation_id=?", (account, strategy, allocation_id)).fetchone()
        if row is None:
            raise ConsumerError("Unknown allocation/account/strategy binding")
        return json.loads(row[0])

    @staticmethod
    def _save(db, record: dict, event: str, details=None, *, commit=True):
        key = (record["account"], record["strategy"], record["allocation_id"])
        db.execute("INSERT INTO allocations VALUES(?,?,?,?) ON CONFLICT(account,strategy,allocation_id) "
                   "DO UPDATE SET document=excluded.document", (*key, _json(record)))
        db.execute("INSERT INTO audit(at_utc,account,strategy,allocation_id,event,details) VALUES(?,?,?,?,?,?)",
                   (datetime.now(timezone.utc).isoformat(), *key, event, _json(details or {})))
        if commit:
            db.commit()

    def get(self, account: str, strategy: str, allocation_id: str) -> dict:
        with self._session() as db:
            return self._get(db, account, strategy, allocation_id)

    def audit(self, account: str, strategy: str, allocation_id: str) -> list[dict]:
        with self._session() as db:
            rows = db.execute("SELECT sequence,at_utc,event,details FROM audit WHERE account=? "
                              "AND strategy=? AND allocation_id=? ORDER BY sequence",
                              (account, strategy, allocation_id)).fetchall()
            return [dict(sequence=r[0], at_utc=r[1], event=r[2], details=json.loads(r[3])) for r in rows]

    def accept_package(self, path: Path, strategy: str, account: str, *,
                       now: datetime | None = None, research_simulation: bool = False,
                       replaces: str | None = None, replacement_reason: str | None = None) -> dict:
        """Accept a strict handoff, or explicitly opt into NON-production research simulation.

        Research mode still verifies package integrity/identity and COMPLETE strategy.
        It retains every handoff blocker in the receipt, without declaring the model validated.
        Replacements are explicit even when the preceding allocation completed.
        """
        _text(account)
        if strategy not in tp.STRATEGIES:
            raise ConsumerError("Unknown strategy")
        if research_simulation:
            report = tp.inspect_package(path, now=now)
            portfolio = report.strategies[strategy]
            if portfolio.status != tp.STATUS_COMPLETE:
                raise ConsumerError("Research simulation still requires COMPLETE strategy")
        else:
            report, portfolio = tp.load_target_portfolio(path, strategy, now=now)
        target = {"positions": [asdict(p) for p in portfolio.positions],
                  "cash_weight": portfolio.cash_weight}
        target = json.loads(_json(target))
        with self._session() as db:
            existing = db.execute("SELECT document FROM allocations WHERE account=? AND strategy=? "
                                  "AND allocation_id=?", (account, strategy, report.allocation_id)).fetchone()
            if existing:
                record = json.loads(existing[0])
                if record["target"] != target or record["research_simulation"] != research_simulation:
                    raise ConsumerError("Duplicate allocation has different target or execution mode")
                if replaces is not None and (record["replaces"] != replaces or
                                            record["replacement_reason"] != replacement_reason):
                    raise ConsumerError("Duplicate replacement metadata differs")
                return record
            prior = db.execute("SELECT allocation_id FROM bindings WHERE account=? AND strategy=?",
                               (account, strategy)).fetchone()
            if prior:
                if replaces != prior[0] or not replacement_reason or not replacement_reason.strip():
                    raise ConsumerError("New allocation requires explicit replacement ID and reason")
                old = self._get(db, account, strategy, replaces)
                if (old["status"] == "replacing" or
                        (old["replaces"] and self._get(db, account, strategy, old["replaces"])["status"]
                         != "replaced")):
                    raise ConsumerError("Finish the outstanding replacement before replacing again")
                if old["research_simulation"] != research_simulation:
                    raise ConsumerError("Research and strict handoff must use separate account bindings")
                old["status"] = "replacing"
                self._save(db, old, "replacement_requested",
                           {"old": replaces, "new": report.allocation_id, "reason": replacement_reason},
                           commit=False)
            elif replaces is not None or replacement_reason is not None:
                raise ConsumerError("Replacement requires an existing bound allocation")
            record = dict(account=account, strategy=strategy, allocation_id=report.allocation_id,
                          run_id=report.manifest["run_id"], status="accepted", target=target,
                          research_simulation=research_simulation,
                          handoff_blockers=report.handoff_blockers, replaces=replaces,
                          replacement_reason=replacement_reason, orders=None, blocked_reason=None)
            db.execute("INSERT INTO bindings VALUES(?,?,?) ON CONFLICT(account,strategy) DO UPDATE "
                       "SET allocation_id=excluded.allocation_id", (account, strategy, report.allocation_id))
            self._save(db, record, "accepted", {"replaces": replaces, "reason": replacement_reason})
            return record

    def _snapshot(self, db, record, provider) -> ExecutionSnapshot:
        snap = provider.snapshot(record["account"], record["strategy"])
        if (snap.account, snap.strategy) != (record["account"], record["strategy"]):
            raise ConsumerError("Provider snapshot account/strategy mismatch")
        for security, quantity in snap.positions.items():
            _text(security)
            _number(quantity)
        keys = set()
        for order in snap.open_orders:
            _text(order.client_key)
            _text(order.order_id)
            if order.status not in {"accepted", "partially_filled"} or order.client_key in keys:
                raise ConsumerError("Invalid/duplicate open order")
            keys.add(order.client_key)
            _number(order.filled_quantity)
        fills = set()
        for fill in snap.confirmed_fills:
            _text(fill.fill_id)
            _text(fill.client_key)
            _text(fill.order_id)
            _number(fill.quantity, positive=True)
            if fill.fill_id in fills:
                raise ConsumerError("Duplicate confirmed fill ID")
            fills.add(fill.fill_id)
        self._save(db, record, "reconciled_snapshot", asdict(snap))
        return snap

    def _block(self, db, record, reason):
        record["status"], record["blocked_reason"] = "blocked", reason
        self._save(db, record, "blocked", {"reason": reason})
        return record

    @staticmethod
    def _ack(order, report: OrderReport):
        if (report.client_key != order["client_key"] or report.status not in _REPORT_STATES
                or not report.order_id):
            raise ConsumerError("Provider report identity/status mismatch")
        filled = _number(report.filled_quantity)
        quantity = Decimal(order["quantity"])
        if filled > quantity or filled < Decimal(order["filled_quantity"]):
            raise ConsumerError("Provider fill quantity regressed or exceeds intent")
        if ((report.status == "filled" and filled != quantity) or
                (report.status == "partially_filled" and not 0 < filled < quantity) or
                (report.status in {"accepted", "rejected"} and filled != 0)):
            raise ConsumerError("Provider report status/fill quantity inconsistent")
        if order["order_id"] and order["order_id"] != report.order_id:
            raise ConsumerError("Provider changed external order ID")
        order.update(status=report.status, filled_quantity=str(filled), order_id=report.order_id,
                     retry_uncertain=False)

    def _reconcile_orders(self, db, record, provider, snap, *, replacing=False) -> bool:
        for order in record["orders"] or []:
            if ((order["status"] == "prepared" and not order.get("retry_uncertain")) or
                    order["status"] in _TERMINAL):
                continue
            report = (provider.lookup(record["account"], record["strategy"], order["client_key"])
                      if provider.supports_query else None)
            if report is None:
                report = next((o for o in snap.open_orders if o.client_key == order["client_key"]), None)
            fills = [f for f in snap.confirmed_fills if f.client_key == order["client_key"]]
            if report is None and fills:
                total = sum((f.quantity for f in fills), Decimal(0))
                if total == Decimal(order["quantity"]) and len({f.order_id for f in fills}) == 1:
                    report = OrderReport(order["client_key"], fills[0].order_id, "filled", total)
            if report is not None:
                if fills and (sum((f.quantity for f in fills), Decimal(0)) > report.filled_quantity or
                              any(f.order_id != report.order_id for f in fills)):
                    raise ConsumerError("Confirmed fills contradict provider order report")
                self._ack(order, report)
                self._save(db, record, "order_reconciled", order)
            elif ((order["status"] == "submitting" or order.get("retry_uncertain")) and not fills and
                  (provider.supports_query or (provider.supports_idempotency and not replacing))):
                order["status"] = "prepared"
                # Idempotency permits a same-key retry, but does not prove the prior order absent.
                order["retry_uncertain"] = not provider.supports_query
                self._save(db, record, "retry_authorized_after_reconciliation", order)
            else:
                self._block(db, record, "Ambiguous external acceptance; query/idempotency evidence required")
                return False
        return True

    def _replace(self, db, record, provider) -> bool:
        old = self._get(db, record["account"], record["strategy"], record["replaces"])
        if old["status"] == "replaced":
            return True
        snap = self._snapshot(db, old, provider)
        if not self._reconcile_orders(db, old, provider, snap, replacing=True):
            self._block(db, record, "Replacement predecessor has ambiguous external state")
            return False
        for order in old["orders"] or []:
            if order["status"] == "prepared":
                order["status"] = "cancelled"
                self._save(db, old, "unsubmitted_intent_cancelled", order)
            elif order["status"] not in _TERMINAL:
                self._save(db, old, "cancel_requested", order)
                provider.cancel(old["account"], old["strategy"], order["order_id"])
        # A cancellation request is not cancellation confirmation.
        snap = self._snapshot(db, old, provider)
        if not self._reconcile_orders(db, old, provider, snap, replacing=True):
            self._block(db, record, "Replacement cancellation is ambiguous")
            return False
        if snap.open_orders or any(o["status"] not in _TERMINAL for o in old["orders"] or []):
            self._block(db, record, "Replacement waiting for confirmed residual-order cancellation")
            return False
        old["status"], old["blocked_reason"] = "replaced", None
        self._save(db, old, "replaced", {"old": old["allocation_id"],
                                       "new": record["allocation_id"],
                                       "reason": record["replacement_reason"]})
        return True

    def process(self, account: str, strategy: str, allocation_id: str,
                provider: ExecutionProvider, planner: Planner) -> dict:
        """Resume one frozen allocation; never plan twice or blindly retry an uncertain submit.

        Each invocation reconciles before any external action. Existing pending orders
        are queried, not submitted again. Blocked records may recover through subsequent
        authoritative reconciliation, but failed/cancelled plans require explicit replacement.
        """
        with self._session() as db:
            record = self._get(db, account, strategy, allocation_id)
            current = db.execute("SELECT allocation_id FROM bindings WHERE account=? AND strategy=?",
                                 (account, strategy)).fetchone()
            if record["status"] in {"completed", "replaced"}:
                return record
            if current is None or current[0] != allocation_id:
                raise ConsumerError("Superseded allocation cannot submit orders")
            if record["replaces"] and not self._replace(db, record, provider):
                return record
            snap = self._snapshot(db, record, provider)
            if not self._reconcile_orders(db, record, provider, snap):
                return record
            own_keys = {o["client_key"] for o in record["orders"] or []}
            if any(o.client_key not in own_keys for o in snap.open_orders):
                return self._block(db, record, "Untracked open orders require reconciliation")
            if record["orders"] is None:
                intents = list(planner(json.loads(_json(record["target"])), snap))
                orders = []
                securities = set()
                buyable = {p["security_id"] for p in record["target"]["positions"]
                           if Decimal(p["target_weight"]) > 0}
                for index, intent in enumerate(intents):
                    _text(intent.security_id)
                    _number(intent.quantity, positive=True)
                    if intent.side not in {"buy", "sell"} or intent.security_id in securities:
                        raise ConsumerError("Plan requires one buy/sell intent per security")
                    if intent.side == "buy" and intent.security_id not in buyable:
                        raise ConsumerError("Buy intent is not part of the accepted target")
                    if (intent.side == "sell" and
                            intent.quantity > snap.positions.get(intent.security_id, Decimal(0))):
                        raise ConsumerError("Sell intent exceeds reconciled scoped position")
                    securities.add(intent.security_id)
                    key = "pc-" + hashlib.sha256(_json(
                        [account, strategy, allocation_id, index, asdict(intent)]).encode()).hexdigest()
                    orders.append(dict(client_key=key, security_id=intent.security_id, side=intent.side,
                                       quantity=str(intent.quantity), filled_quantity="0",
                                       status="prepared", order_id=None, retry_uncertain=False))
                record["orders"] = orders
                self._save(db, record, "plan_frozen", {"orders": orders})
            record["blocked_reason"] = None
            for order in record["orders"]:
                if order["status"] in {"cancelled", "rejected"}:
                    return self._block(db, record, "Terminal unfilled intent requires explicit replacement")
            for order in record["orders"]:
                if order["status"] != "prepared":
                    continue
                record["status"] = "in_progress"
                order["status"] = "submitting"
                self._save(db, record, "submission_intent", order)
                # If the provider raises or this process dies, the committed intent survives.
                report = provider.submit(account, strategy, order["client_key"],
                                         OrderIntent(order["security_id"], order["side"],
                                                     Decimal(order["quantity"])))
                self._ack(order, report)
                self._save(db, record, "submission_acknowledged", order)
                if order["status"] in {"cancelled", "rejected"}:
                    return self._block(db, record, "Terminal unfilled intent requires explicit replacement")
            if all(o["status"] == "filled" for o in record["orders"]):
                record["status"] = "completed"
            elif any(Decimal(o["filled_quantity"]) > 0 for o in record["orders"]):
                record["status"] = "partially_executed"
            else:
                record["status"] = "in_progress"
            self._save(db, record, "processing_state", {"status": record["status"]})
            return record
