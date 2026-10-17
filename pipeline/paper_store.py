"""Persistent SQLite storage for a local paper-trading workflow.

This module stores paper-trading state only. It does not run a scheduler,
connect to a broker, or execute live trades.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Mapping, Sequence


_APPLICATION_ID = 0x50535452
_SCHEMA_VERSION = 6
ENVIRONMENTS = ("real", "synthetic", "test")
_V3_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS store_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TRIGGER IF NOT EXISTS store_meta_no_update
        BEFORE UPDATE ON store_meta
        BEGIN SELECT RAISE(ABORT, 'store metadata is immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS store_meta_no_delete
        BEFORE DELETE ON store_meta
        BEGIN SELECT RAISE(ABORT, 'store metadata is immutable'); END""",
    """CREATE TABLE IF NOT EXISTS cycle_timeline (
        cycle_id TEXT PRIMARY KEY REFERENCES reporting_cycles(cycle_id),
        published_at TEXT NOT NULL,
        first_seen_at TEXT NOT NULL,
        snapshot_as_of TEXT NOT NULL,
        model_completed_at TEXT NOT NULL
    )""",
    """CREATE TRIGGER IF NOT EXISTS cycle_timeline_no_update
        BEFORE UPDATE ON cycle_timeline
        BEGIN SELECT RAISE(ABORT, 'cycle timeline is immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS cycle_timeline_no_delete
        BEFORE DELETE ON cycle_timeline
        BEGIN SELECT RAISE(ABORT, 'cycle timeline is immutable'); END""",
)
_V4_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS filing_resolutions (
        cycle_id TEXT NOT NULL REFERENCES reporting_cycles(cycle_id),
        filing_key TEXT NOT NULL,
        resolution TEXT NOT NULL CHECK(resolution = 'CORPORATE_ACTION_RESOLVED'),
        evidence_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY(cycle_id, filing_key),
        FOREIGN KEY(cycle_id, filing_key) REFERENCES expected_filings(cycle_id, filing_key)
    )""",
    """CREATE TRIGGER IF NOT EXISTS filing_resolutions_no_update
        BEFORE UPDATE ON filing_resolutions
        BEGIN SELECT RAISE(ABORT, 'filing resolutions are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS filing_resolutions_no_delete
        BEFORE DELETE ON filing_resolutions
        BEGIN SELECT RAISE(ABORT, 'filing resolutions are immutable'); END""",
    """CREATE TABLE IF NOT EXISTS dividend_pay_dates (
        action_id TEXT PRIMARY KEY REFERENCES dividend_receivables(action_id),
        pay_date TEXT NOT NULL,
        source TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )""",
)
_V5_STATEMENTS = (
    """CREATE TRIGGER IF NOT EXISTS daily_marks_no_update
        BEFORE UPDATE ON daily_marks
        BEGIN SELECT RAISE(ABORT, 'daily marks are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS daily_marks_no_delete
        BEFORE DELETE ON daily_marks
        BEGIN SELECT RAISE(ABORT, 'daily marks are immutable'); END""",
)
_V6_STATEMENTS = ()
_SCHEMA_TABLES = {
    "filing_resolutions",
    "dividend_pay_dates",
    "portfolios",
    "reporting_cycles",
    "expected_filings",
    "data_snapshots",
    "signal_snapshots",
    "execution_batches",
    "execution_schedule",
    "store_meta",
    "cycle_timeline",
    "target_weights",
    "simulated_orders",
    "simulated_fills",
    "cash_ledger",
    "position_ledger",
    "corporate_actions",
    "dividend_receivables",
    "daily_marks",
    "job_runs",
    "job_events",
    "job_errors",
}


class PaperStoreError(Exception):
    """Base error for storage operations."""


class StoreNotFoundError(PaperStoreError):
    """Raised when opening a database that has not been explicitly created."""


class SchemaError(PaperStoreError):
    """Raised when a database does not contain the supported store schema."""


class ValidationError(PaperStoreError, ValueError):
    """Raised when a supplied identifier or value is invalid."""


class DuplicateRecordError(PaperStoreError):
    """Raised when an immutable record or idempotency key conflicts."""


class NotFoundError(PaperStoreError):
    """Raised when a referenced record does not exist."""


class StateError(PaperStoreError):
    """Raised when an operation conflicts with the current record state."""


class FilingStatus(str, Enum):
    WAITING_FILING = "WAITING_FILING"
    WAITING_FACTS = "WAITING_FACTS"
    INVALID_DATA = "INVALID_DATA"
    READY = "READY"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class JobStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class OrderRequest:
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: float


@dataclass(frozen=True)
class SimulatedOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    status: str


class PaperStore:
    """SQLite-backed paper-trading state store.

    Use :meth:`create` once to explicitly initialize a database. Constructing
    ``PaperStore(path)`` only opens and validates an existing database.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = _database_path(path)
        if not self.path.exists():
            raise StoreNotFoundError(
                f"Paper-trading database does not exist: {self.path}; "
                "use PaperStore.create(path) to initialize it"
            )
        if not self.path.is_file():
            raise ValidationError(f"Database path is not a file: {self.path}")
        with self._connection() as connection:
            self._verify_schema(connection)

    @classmethod
    def create(cls, path: str | Path, environment: str = "test") -> PaperStore:
        """Create an empty database, or validate an already initialized one.

        ``environment`` ('real', 'synthetic' or 'test') is stored immutably in a new
        database; for an existing one it must match the stored value.

        Existing non-store databases are never overwritten or initialized.
        Repeating this call for an initialized store leaves its data untouched.
        """
        if environment not in ENVIRONMENTS:
            raise ValidationError(f"environment must be one of {ENVIRONMENTS}")
        database_path = _database_path(path)
        if database_path.exists() and not database_path.is_file():
            raise ValidationError(f"Database path is not a file: {database_path}")
        try:
            database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                _database_uri(database_path, "rwc"),
                uri=True,
                timeout=10,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise PaperStoreError(f"Could not create database: {exc}") from exc
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            cls._initialize_or_verify(connection, environment)
        finally:
            connection.close()
        return cls(database_path)

    @staticmethod
    def _initialize_or_verify(connection: sqlite3.Connection, environment: str = "test") -> None:
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if application_id == _APPLICATION_ID and schema_version == _SCHEMA_VERSION:
            PaperStore._verify_schema(connection)
            stored = connection.execute(
                "SELECT value FROM store_meta WHERE key = 'environment'"
            ).fetchone()
            if stored is None or stored[0] != environment:
                raise SchemaError(
                    f"database environment is {stored[0] if stored else 'unset'!r}, "
                    f"not {environment!r}; real and synthetic stores must stay separate"
                )
            return
        if application_id == _APPLICATION_ID and schema_version < _SCHEMA_VERSION:
            raise SchemaError(
                f"database schema v{schema_version} is outdated; run "
                "`paper_portfolio.py migrate --environment ...` (creates a backup first)"
            )
        if application_id != 0 or schema_version != 0:
            raise SchemaError("Database has an unsupported paper-store schema version")
        objects = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'view', 'trigger') AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if objects:
            raise SchemaError("Refusing to initialize a non-empty, unrecognized database")

        connection.execute("BEGIN IMMEDIATE")
        try:
            statements = (
                """CREATE TABLE portfolios (
                    portfolio_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    initial_cash REAL NOT NULL CHECK(initial_cash >= 0),
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'paper' CHECK(mode = 'paper'),
                    currency TEXT NOT NULL DEFAULT 'USD' CHECK(currency = 'USD'),
                    strategy_version TEXT NOT NULL DEFAULT '1'
                )""",
                """CREATE TABLE reporting_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                    cycle_key TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    rebalance_policy TEXT NOT NULL DEFAULT 'all_expected_filings'
                        CHECK(rebalance_policy IN ('all_expected_filings', 'fixed_quarterly')),
                    deadline_at TEXT,
                    blocked_reason TEXT,
                    status TEXT NOT NULL DEFAULT 'OPEN'
                        CHECK(status IN ('OPEN', 'BLOCKED', 'SIGNALLED', 'EXECUTED')),
                    model_status TEXT NOT NULL DEFAULT 'WAITING'
                        CHECK(model_status IN ('WAITING', 'BLOCKED', 'READY', 'FAILED')),
                    model_reason TEXT,
                    filing_status TEXT NOT NULL DEFAULT 'WAITING_FILING'
                        CHECK(filing_status IN
                            ('WAITING_FILING', 'WAITING_FACTS', 'INVALID_DATA', 'READY')),
                    UNIQUE(portfolio_id, cycle_key, strategy),
                    UNIQUE(cycle_id, strategy)
                )""",
                """CREATE TABLE expected_filings (
                    cycle_id TEXT NOT NULL REFERENCES reporting_cycles(cycle_id),
                    filing_key TEXT NOT NULL,
                    received INTEGER NOT NULL DEFAULT 0 CHECK(received IN (0, 1)),
                    facts_ready INTEGER NOT NULL DEFAULT 0 CHECK(facts_ready IN (0, 1)),
                    valid INTEGER NOT NULL DEFAULT 1 CHECK(valid IN (0, 1)),
                    company_id TEXT NOT NULL DEFAULT '',
                    security_id TEXT NOT NULL DEFAULT '',
                    expected_period TEXT NOT NULL DEFAULT '',
                    filing_type TEXT NOT NULL DEFAULT '',
                    filed_at TEXT,
                    first_seen_at TEXT,
                    source TEXT NOT NULL DEFAULT '',
                    error_reason TEXT NOT NULL DEFAULT '',
                    facts_ready_at TEXT,
                    status TEXT NOT NULL DEFAULT 'WAITING_FILING'
                        CHECK(status IN
                            ('WAITING_FILING', 'WAITING_FACTS', 'INVALID_DATA', 'READY')),
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(cycle_id, filing_key)
                )""",
                """CREATE TABLE data_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL REFERENCES reporting_cycles(cycle_id),
                    captured_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    UNIQUE(cycle_id, snapshot_id)
                )""",
                """CREATE TABLE signal_snapshots (
                    signal_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL REFERENCES reporting_cycles(cycle_id),
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    FOREIGN KEY(cycle_id, strategy)
                        REFERENCES reporting_cycles(cycle_id, strategy)
                )""",
                """CREATE TABLE execution_batches (
                    batch_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(cycle_id, strategy),
                    FOREIGN KEY(cycle_id, strategy)
                        REFERENCES reporting_cycles(cycle_id, strategy)
                )""",
                """CREATE TABLE execution_schedule (
                    batch_id TEXT PRIMARY KEY REFERENCES execution_batches(batch_id),
                    scheduled_for TEXT NOT NULL,
                    planned_at TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT ''
                )""",
                """CREATE TRIGGER execution_schedule_no_update
                    BEFORE UPDATE ON execution_schedule
                    BEGIN SELECT RAISE(ABORT, 'execution schedule is immutable'); END""",
                """CREATE TRIGGER execution_schedule_no_delete
                    BEFORE DELETE ON execution_schedule
                    BEGIN SELECT RAISE(ABORT, 'execution schedule is immutable'); END""",
                """CREATE TABLE target_weights (
                    batch_id TEXT NOT NULL REFERENCES execution_batches(batch_id),
                    symbol TEXT NOT NULL,
                    weight REAL NOT NULL CHECK(weight >= 0 AND weight <= 1),
                    PRIMARY KEY(batch_id, symbol)
                )""",
                """CREATE TABLE simulated_orders (
                    order_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL REFERENCES execution_batches(batch_id),
                    client_order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                    quantity REAL NOT NULL CHECK(quantity > 0),
                    filled_quantity REAL NOT NULL DEFAULT 0
                        CHECK(filled_quantity >= 0 AND filled_quantity <= quantity),
                    status TEXT NOT NULL DEFAULT 'OPEN'
                        CHECK(status IN ('OPEN', 'PARTIAL', 'FILLED')),
                    UNIQUE(batch_id, client_order_id)
                )""",
                """CREATE TABLE simulated_fills (
                    fill_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL UNIQUE,
                    order_id TEXT NOT NULL REFERENCES simulated_orders(order_id),
                    quantity REAL NOT NULL CHECK(quantity > 0),
                    price REAL NOT NULL CHECK(price >= 0),
                    transaction_cost REAL NOT NULL DEFAULT 0 CHECK(transaction_cost >= 0),
                    filled_at TEXT NOT NULL
                )""",
                """CREATE TABLE cash_ledger (
                    entry_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                    entry_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    execution_id TEXT UNIQUE
                )""",
                """CREATE TABLE position_ledger (
                    entry_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                    symbol TEXT NOT NULL,
                    quantity_delta REAL NOT NULL,
                    entry_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    execution_id TEXT UNIQUE
                )""",
                """CREATE TABLE corporate_actions (
                    action_id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                    symbol TEXT NOT NULL,
                    action_type TEXT NOT NULL
                        CHECK(action_type IN ('split', 'dividend', 'merger', 'cash_delisting')),
                    ex_date TEXT NOT NULL,
                    pay_date TEXT,
                    value REAL NOT NULL CHECK(value > 0),
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )""",
                """CREATE TABLE dividend_receivables (
                    action_id TEXT PRIMARY KEY REFERENCES corporate_actions(action_id),
                    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                    symbol TEXT NOT NULL,
                    amount REAL NOT NULL CHECK(amount >= 0),
                    pay_date TEXT,
                    status TEXT NOT NULL CHECK(status IN ('RECEIVABLE', 'PAID')),
                    paid_at TEXT
                )""",
                """CREATE TABLE daily_marks (
                    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                    marked_at TEXT NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    dividend_receivables REAL NOT NULL DEFAULT 0,
                    nav REAL NOT NULL,
                    benchmark_value REAL NOT NULL,
                    position_values_json TEXT,
                    PRIMARY KEY(portfolio_id, marked_at)
                )""",
                """CREATE TABLE job_runs (
                    run_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('RUNNING', 'SUCCEEDED', 'FAILED')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                )""",
                """CREATE TABLE job_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES job_runs(run_id),
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )""",
                """CREATE TABLE job_errors (
                    error_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES job_runs(run_id),
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )""",
                """CREATE TRIGGER data_snapshots_no_update
                    BEFORE UPDATE ON data_snapshots
                    BEGIN SELECT RAISE(ABORT, 'data snapshots are immutable'); END""",
                """CREATE TRIGGER data_snapshots_no_delete
                    BEFORE DELETE ON data_snapshots
                    BEGIN SELECT RAISE(ABORT, 'data snapshots are immutable'); END""",
                """CREATE TRIGGER signal_snapshots_no_update
                    BEFORE UPDATE ON signal_snapshots
                    BEGIN SELECT RAISE(ABORT, 'signal snapshots are immutable'); END""",
                """CREATE TRIGGER signal_snapshots_no_delete
                    BEFORE DELETE ON signal_snapshots
                    BEGIN SELECT RAISE(ABORT, 'signal snapshots are immutable'); END""",
                """CREATE TRIGGER simulated_fills_no_update
                    BEFORE UPDATE ON simulated_fills
                    BEGIN SELECT RAISE(ABORT, 'simulated fills are immutable'); END""",
                """CREATE TRIGGER simulated_fills_no_delete
                    BEFORE DELETE ON simulated_fills
                    BEGIN SELECT RAISE(ABORT, 'simulated fills are immutable'); END""",
                """CREATE TRIGGER cash_ledger_no_update
                    BEFORE UPDATE ON cash_ledger
                    BEGIN SELECT RAISE(ABORT, 'cash ledger is append-only'); END""",
                """CREATE TRIGGER cash_ledger_no_delete
                    BEFORE DELETE ON cash_ledger
                    BEGIN SELECT RAISE(ABORT, 'cash ledger is append-only'); END""",
                """CREATE TRIGGER position_ledger_no_update
                    BEFORE UPDATE ON position_ledger
                    BEGIN SELECT RAISE(ABORT, 'position ledger is append-only'); END""",
                """CREATE TRIGGER position_ledger_no_delete
                    BEFORE DELETE ON position_ledger
                    BEGIN SELECT RAISE(ABORT, 'position ledger is append-only'); END""",
                """CREATE TRIGGER corporate_actions_no_update
                    BEFORE UPDATE ON corporate_actions
                    BEGIN SELECT RAISE(ABORT, 'corporate action records are immutable'); END""",
                """CREATE TRIGGER corporate_actions_no_delete
                    BEFORE DELETE ON corporate_actions
                    BEGIN SELECT RAISE(ABORT, 'corporate action records are immutable'); END""",
                """CREATE TRIGGER job_events_no_update
                    BEFORE UPDATE ON job_events
                    BEGIN SELECT RAISE(ABORT, 'job events are append-only'); END""",
                """CREATE TRIGGER job_events_no_delete
                    BEFORE DELETE ON job_events
                    BEGIN SELECT RAISE(ABORT, 'job events are append-only'); END""",
                """CREATE TRIGGER job_errors_no_update
                    BEFORE UPDATE ON job_errors
                    BEGIN SELECT RAISE(ABORT, 'job errors are append-only'); END""",
                """CREATE TRIGGER job_errors_no_delete
                    BEFORE DELETE ON job_errors
                    BEGIN SELECT RAISE(ABORT, 'job errors are append-only'); END""",
            )
            for statement in statements + _V3_STATEMENTS + _V4_STATEMENTS + _V5_STATEMENTS + _V6_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO store_meta (key, value) VALUES ('environment', ?)", (environment,)
            )
            connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _verify_schema(connection: sqlite3.Connection) -> None:
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if application_id != _APPLICATION_ID or schema_version != _SCHEMA_VERSION:
            raise SchemaError("Database is not an initialized paper-trading store")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not _SCHEMA_TABLES.issubset(tables):
            raise SchemaError("Paper-trading database is missing required tables")

    @classmethod
    def migrate(cls, path: str | Path, environment: str, backup_dir: str | Path) -> dict[str, object]:
        """Upgrade an older store in place, after a verified backup; data is never replaced.

        An already current store is returned unchanged. ``environment`` must be stated
        explicitly because older stores carry no marker.
        """
        if environment not in ENVIRONMENTS:
            raise ValidationError(f"environment must be one of {ENVIRONMENTS}")
        database_path = _database_path(path)
        if not database_path.is_file():
            raise StoreNotFoundError(f"Database does not exist: {database_path}")
        probe = sqlite3.connect(_database_uri(database_path, "rw"), uri=True, timeout=10, isolation_level=None)
        try:
            app_id = int(probe.execute("PRAGMA application_id").fetchone()[0])
            version = int(probe.execute("PRAGMA user_version").fetchone()[0])
        finally:
            probe.close()
        if app_id != _APPLICATION_ID or not 1 <= version <= _SCHEMA_VERSION:
            raise SchemaError("not a migratable paper-store database")
        if version == _SCHEMA_VERSION:
            cls(database_path)
            return {"migrated": False, "from_version": version, "to_version": version, "backup": None}
        backup_path = Path(backup_dir) / (
            f"{database_path.stem}.v{version}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.sqlite3"
        )
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(_database_uri(database_path, "ro"), uri=True, timeout=10)
        target = sqlite3.connect(str(backup_path))
        try:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise PaperStoreError("backup failed its integrity check; migration aborted")
        finally:
            target.close()
            source.close()
        connection = sqlite3.connect(
            _database_uri(database_path, "rw"), uri=True, timeout=10, isolation_level=None
        )
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            try:
                if version < 3:
                    # v2 stores written before execution_schedule existed lack that table.
                    connection.execute(
                        """CREATE TABLE IF NOT EXISTS execution_schedule (
                            batch_id TEXT PRIMARY KEY REFERENCES execution_batches(batch_id),
                            scheduled_for TEXT NOT NULL,
                            planned_at TEXT NOT NULL,
                            note TEXT NOT NULL DEFAULT ''
                        )""")
                    for name, verb in (("update", "UPDATE"), ("delete", "DELETE")):
                        connection.execute(
                            f"""CREATE TRIGGER IF NOT EXISTS execution_schedule_no_{name}
                                BEFORE {verb} ON execution_schedule
                                BEGIN SELECT RAISE(ABORT, 'execution schedule is immutable'); END""")
                    for statement in _V3_STATEMENTS:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT OR IGNORE INTO store_meta (key, value) VALUES ('environment', ?)",
                        (environment,),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO store_meta (key, value) VALUES ('migrated_from', ?)",
                        (f"v{version}:{backup_path.name}",),
                    )
                if version < 4:
                    # SQLite cannot drop NOT NULL in place: rebuild the (unreferenced) receivables table.
                    connection.execute("ALTER TABLE dividend_receivables RENAME TO dividend_receivables_v3")
                    connection.execute(
                        """CREATE TABLE dividend_receivables (
                            action_id TEXT PRIMARY KEY REFERENCES corporate_actions(action_id),
                            portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                            symbol TEXT NOT NULL,
                            amount REAL NOT NULL CHECK(amount >= 0),
                            pay_date TEXT,
                            status TEXT NOT NULL CHECK(status IN ('RECEIVABLE', 'PAID')),
                            paid_at TEXT
                        )""")
                    connection.execute(
                        "INSERT INTO dividend_receivables SELECT action_id, portfolio_id, symbol, amount, "
                        "pay_date, status, paid_at FROM dividend_receivables_v3")
                    connection.execute("DROP TABLE dividend_receivables_v3")
                    for statement in _V4_STATEMENTS:
                        connection.execute(statement)
                if version < 5:
                    columns = {
                        row[1] for row in connection.execute("PRAGMA table_info(daily_marks)")
                    }
                    if "position_values_json" not in columns:
                        connection.execute("ALTER TABLE daily_marks ADD COLUMN position_values_json TEXT")
                    for statement in _V5_STATEMENTS:
                        connection.execute(statement)
                if version < 6:
                    columns = {
                        row[1] for row in connection.execute("PRAGMA table_info(expected_filings)")
                    }
                    if "facts_ready_at" not in columns:
                        connection.execute("ALTER TABLE expected_filings ADD COLUMN facts_ready_at TEXT")
                    for statement in _V6_STATEMENTS:
                        connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            connection.execute("PRAGMA foreign_key_check")
        finally:
            connection.close()
        cls(database_path)
        return {"migrated": True, "from_version": version, "to_version": _SCHEMA_VERSION,
                "backup": str(backup_path)}

    def environment(self) -> str:
        rows = self.query("SELECT value FROM store_meta WHERE key = 'environment'")
        return str(rows[0]["value"])

    def record_cycle_timeline(
        self, cycle_id: str, *, published_at: str, first_seen_at: str,
        snapshot_as_of: str, model_completed_at: str,
    ) -> None:
        """Store the four information timestamps of a cycle once; identical retries are no-ops."""
        cycle_id = _identifier(cycle_id, "cycle_id")
        values = [_aware(_timestamp(v), n) for n, v in (
            ("published_at", published_at), ("first_seen_at", first_seen_at),
            ("snapshot_as_of", snapshot_as_of), ("model_completed_at", model_completed_at))]
        published, first_seen, snapshot, completed = values
        if not (first_seen >= published and snapshot >= first_seen and completed >= snapshot):
            raise ValidationError(
                "timeline must satisfy published_at <= first_seen_at <= snapshot_as_of <= model_completed_at"
            )
        iso = [v.isoformat() for v in values]
        with self._transaction() as connection:
            self._require_row(
                connection, "SELECT 1 FROM reporting_cycles WHERE cycle_id = ?",
                (cycle_id,), f"Cycle not found: {cycle_id}",
            )
            existing = connection.execute(
                "SELECT * FROM cycle_timeline WHERE cycle_id = ?", (cycle_id,)).fetchone()
            if existing:
                if [existing[k] for k in ("published_at", "first_seen_at", "snapshot_as_of",
                                          "model_completed_at")] != iso:
                    raise DuplicateRecordError(f"Timeline already recorded: {cycle_id}")
                return
            connection.execute(
                "INSERT INTO cycle_timeline VALUES (?, ?, ?, ?, ?)", (cycle_id, *iso))

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = sqlite3.connect(
                _database_uri(self.path, "rw"),
                uri=True,
                timeout=10,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            if not self.path.exists():
                raise StoreNotFoundError(f"Database does not exist: {self.path}") from exc
            raise PaperStoreError(f"Could not open database: {exc}") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                if "UNIQUE constraint failed" in str(exc):
                    raise DuplicateRecordError(str(exc)) from exc
                raise PaperStoreError(f"Database constraint failed: {exc}") from exc
            except Exception:
                connection.rollback()
                raise

    def create_portfolio(
        self,
        portfolio_id: str,
        name: str,
        strategy: str,
        initial_cash: float,
        *,
        created_at: str | None = None,
    ) -> None:
        portfolio_id = _identifier(portfolio_id, "portfolio_id")
        name = _text(name, "name")
        strategy = _identifier(strategy, "strategy")
        initial_cash = _number(initial_cash, "initial_cash", minimum=0)
        created_at = _timestamp(created_at)
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)
            ).fetchone():
                raise DuplicateRecordError(f"Portfolio already exists: {portfolio_id}")
            connection.execute(
                """INSERT INTO portfolios
                   (portfolio_id, name, strategy, initial_cash, created_at, mode, currency, strategy_version)
                   VALUES (?, ?, ?, ?, ?, 'paper', 'USD', '1')""",
                (portfolio_id, name, strategy, initial_cash, created_at),
            )
            connection.execute(
                """INSERT INTO cash_ledger
                   (entry_id, idempotency_key, portfolio_id, entry_type, amount,
                    created_at, execution_id)
                   VALUES (?, ?, ?, 'OPENING', ?, ?, NULL)""",
                (str(uuid.uuid4()), f"opening:{portfolio_id}", portfolio_id,
                 initial_cash, created_at),
            )

    def create_cycle(
        self,
        portfolio_id: str,
        cycle_key: str,
        strategy: str,
        *,
        rebalance_policy: str = "all_expected_filings",
        deadline_at: str | None = None,
        started_at: str | None = None,
    ) -> str:
        portfolio_id = _identifier(portfolio_id, "portfolio_id")
        cycle_key = _identifier(cycle_key, "cycle_key")
        strategy = _identifier(strategy, "strategy")
        if rebalance_policy not in {"all_expected_filings", "fixed_quarterly"}:
            raise ValidationError("rebalance_policy must be all_expected_filings or fixed_quarterly")
        deadline_at = _timestamp(deadline_at) if deadline_at is not None else None
        started_at = _timestamp(started_at)
        with self._transaction() as connection:
            existing = connection.execute(
                """SELECT cycle_id, rebalance_policy, deadline_at FROM reporting_cycles
                   WHERE portfolio_id = ? AND cycle_key = ? AND strategy = ?""",
                (portfolio_id, cycle_key, strategy),
            ).fetchone()
            if existing:
                if (
                    existing["rebalance_policy"] != rebalance_policy
                    or existing["deadline_at"] != deadline_at
                ):
                    raise DuplicateRecordError(
                        f"Cycle already exists with different policy/deadline: {cycle_key}"
                    )
                return str(existing["cycle_id"])
            self._require_row(
                connection, "SELECT 1 FROM portfolios WHERE portfolio_id = ?",
                (portfolio_id,), f"Portfolio not found: {portfolio_id}",
            )
            cycle_id = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO reporting_cycles
                   (cycle_id, portfolio_id, cycle_key, strategy, started_at,
                    rebalance_policy, deadline_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    cycle_id, portfolio_id, cycle_key, strategy, started_at,
                    rebalance_policy, deadline_at,
                ),
            )
            return cycle_id

    def set_cycle_model_status(
        self,
        cycle_id: str,
        status: str,
        *,
        reason: str | None = None,
    ) -> None:
        cycle_id = _identifier(cycle_id, "cycle_id")
        if status not in {"WAITING", "BLOCKED", "READY", "FAILED"}:
            raise ValidationError("unsupported model status")
        reason = _text(reason, "reason") if reason is not None else None
        with self._transaction() as connection:
            cycle = self._require_row(
                connection,
                "SELECT filing_status, status FROM reporting_cycles WHERE cycle_id = ?",
                (cycle_id,),
                f"Cycle not found: {cycle_id}",
            )
            if status == "READY" and cycle["filing_status"] != FilingStatus.READY.value:
                raise StateError("model cannot be marked READY before all expected filings are READY")
            if cycle["status"] in {"SIGNALLED", "EXECUTED"} and status != "READY":
                raise StateError("completed cycle model status is immutable")
            connection.execute(
                """UPDATE reporting_cycles SET model_status = ?, model_reason = ?,
                       status = CASE WHEN ? = 'BLOCKED' THEN 'BLOCKED' ELSE status END,
                       blocked_reason = CASE WHEN ? = 'BLOCKED' THEN ? ELSE blocked_reason END
                   WHERE cycle_id = ?""",
                (status, reason, status, status, reason, cycle_id),
            )

    def register_expected_filings(
        self,
        cycle_id: str,
        filing_keys: Sequence[str],
        *,
        filing_metadata: Mapping[str, Mapping[str, object]] | None = None,
        updated_at: str | None = None,
    ) -> None:
        cycle_id = _identifier(cycle_id, "cycle_id")
        keys = [_identifier(key, "filing_key") for key in filing_keys]
        if not keys:
            raise ValidationError("filing_keys must contain at least one expected filing")
        if len(set(keys)) != len(keys):
            raise ValidationError("filing_keys must not contain duplicates")
        if filing_metadata is None or set(filing_metadata) != set(keys):
            raise ValidationError("metadata is required for every expected filing")
        unknown_metadata = set(filing_metadata) - set(keys)
        if unknown_metadata:
            raise ValidationError(
                f"filing metadata references unknown filing keys: {sorted(unknown_metadata)}"
            )
        allowed_metadata = {
            "company_id", "security_id", "expected_period", "filing_type", "source"
        }
        metadata_rows = []
        company_periods: set[tuple[str, str]] = set()
        for key in keys:
            fields = dict(filing_metadata.get(key, {}))
            unknown_fields = set(fields) - allowed_metadata
            if unknown_fields:
                raise ValidationError(
                    f"unsupported expected-filing metadata fields for {key}: {sorted(unknown_fields)}"
                )
            company_id = _identifier(str(fields.get("company_id", "")), "company_id")
            expected_period = _text(str(fields.get("expected_period", "")), "expected_period")
            filing_type = _text(str(fields.get("filing_type", "")), "filing_type")
            company_period = (company_id, expected_period)
            if company_period in company_periods:
                raise ValidationError(
                    "expected filing directory must contain one report per company and period"
                )
            company_periods.add(company_period)
            metadata_rows.append((
                company_id,
                _text(str(fields.get("security_id", "")), "security_id") if fields.get("security_id") else "",
                expected_period,
                filing_type,
                _text(str(fields.get("source", "")), "source") if fields.get("source") else "",
            ))
        updated_at = _timestamp(updated_at)
        with self._transaction() as connection:
            self._require_row(
                connection, "SELECT 1 FROM reporting_cycles WHERE cycle_id = ?",
                (cycle_id,), f"Cycle not found: {cycle_id}",
            )
            if connection.execute(
                "SELECT 1 FROM expected_filings WHERE cycle_id = ? LIMIT 1", (cycle_id,)
            ).fetchone():
                raise DuplicateRecordError(f"Expected filings already registered for {cycle_id}")
            connection.executemany(
                """INSERT INTO expected_filings
                   (cycle_id, filing_key, updated_at, company_id, security_id,
                    expected_period, filing_type, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    (cycle_id, key, updated_at, *metadata)
                    for key, metadata in zip(keys, metadata_rows, strict=True)
                ),
            )
            self._refresh_filing_status(connection, cycle_id)

    def set_filing_result(
        self,
        cycle_id: str,
        filing_key: str,
        *,
        received: bool,
        facts_ready: bool,
        valid: bool,
        filed_at: str | None = None,
        first_seen_at: str | None = None,
        source: str | None = None,
        error_reason: str | None = None,
        facts_ready_at: str | None = None,
        updated_at: str | None = None,
    ) -> FilingStatus:
        cycle_id = _identifier(cycle_id, "cycle_id")
        filing_key = _identifier(filing_key, "filing_key")
        for field_name, value in (
            ("received", received), ("facts_ready", facts_ready), ("valid", valid)
        ):
            if not isinstance(value, bool):
                raise ValidationError(f"{field_name} must be a bool")
        if received and (filed_at is None or first_seen_at is None or source is None):
            raise ValidationError(
                "received filings require filed_at, first_seen_at, and source metadata"
            )
        if not valid and error_reason is None:
            raise ValidationError("invalid filings require an explicit error_reason")
        if facts_ready and not received:
            raise ValidationError("structured facts cannot be ready before a filing is received")
        updated_at = _timestamp(updated_at)
        facts_ready_at = _timestamp(facts_ready_at) if facts_ready_at is not None else (
            updated_at if facts_ready else None
        )
        status = _filing_status(received, facts_ready, valid)
        filed_at = _timestamp(filed_at) if filed_at is not None else None
        first_seen_at = _timestamp(first_seen_at) if first_seen_at is not None else None
        source = _text(source, "source") if source is not None else None
        error_reason = _text(error_reason, "error_reason") if error_reason is not None else None
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE expected_filings
                   SET received = ?, facts_ready = ?, valid = ?, status = ?, updated_at = ?,
                       filed_at = COALESCE(?, filed_at),
                       first_seen_at = COALESCE(?, first_seen_at),
                       facts_ready_at = CASE WHEN ? THEN COALESCE(facts_ready_at, ?) ELSE facts_ready_at END,
                       source = COALESCE(?, source),
                       error_reason = COALESCE(?, error_reason)
                   WHERE cycle_id = ? AND filing_key = ?""",
                (
                    received, facts_ready, valid, status.value, updated_at,
                    filed_at, first_seen_at, facts_ready, facts_ready_at,
                    source, error_reason, cycle_id, filing_key,
                ),
            )
            if cursor.rowcount != 1:
                raise NotFoundError(
                    f"Expected filing not found: {cycle_id}/{filing_key}"
                )
            self._refresh_filing_status(connection, cycle_id)
        return status

    def filing_status(self, cycle_id: str) -> FilingStatus:
        cycle_id = _identifier(cycle_id, "cycle_id")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT filing_status FROM reporting_cycles WHERE cycle_id = ?",
                (cycle_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Cycle not found: {cycle_id}")
        return FilingStatus(row["filing_status"])

    def record_data_snapshot(
        self,
        cycle_id: str,
        snapshot_id: str,
        payload: Mapping[str, object],
        *,
        captured_at: str | None = None,
    ) -> str:
        cycle_id = _identifier(cycle_id, "cycle_id")
        snapshot_id = _identifier(snapshot_id, "snapshot_id")
        payload_json, payload_hash = _canonical_payload(payload)
        captured_at = _timestamp(captured_at)
        with self._transaction() as connection:
            self._require_row(
                connection, "SELECT 1 FROM reporting_cycles WHERE cycle_id = ?",
                (cycle_id,), f"Cycle not found: {cycle_id}",
            )
            if connection.execute(
                "SELECT 1 FROM data_snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone():
                raise DuplicateRecordError(f"Snapshot already exists: {snapshot_id}")
            connection.execute(
                """INSERT INTO data_snapshots
                   (snapshot_id, cycle_id, captured_at, payload_json, payload_sha256)
                   VALUES (?, ?, ?, ?, ?)""",
                (snapshot_id, cycle_id, captured_at, payload_json, payload_hash),
            )
        return snapshot_id

    def record_signal_snapshot(
        self,
        cycle_id: str,
        strategy: str,
        symbol: str,
        payload: Mapping[str, object],
        *,
        signal_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        cycle_id = _identifier(cycle_id, "cycle_id")
        strategy = _identifier(strategy, "strategy")
        symbol = _identifier(symbol, "symbol")
        signal_id = _identifier(signal_id or str(uuid.uuid4()), "signal_id")
        payload_json, payload_hash = _canonical_payload(payload)
        created_at = _timestamp(created_at)
        with self._transaction() as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM reporting_cycles WHERE cycle_id = ? AND strategy = ?",
                (cycle_id, strategy), f"Cycle/strategy not found: {cycle_id}/{strategy}",
            )
            if connection.execute(
                "SELECT 1 FROM signal_snapshots WHERE signal_id = ?", (signal_id,)
            ).fetchone():
                raise DuplicateRecordError(f"Signal snapshot already exists: {signal_id}")
            connection.execute(
                """INSERT INTO signal_snapshots
                   (signal_id, cycle_id, strategy, symbol, created_at,
                    payload_json, payload_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (signal_id, cycle_id, strategy, symbol, created_at, payload_json, payload_hash),
            )
        return signal_id

    def rebalance(
        self,
        cycle_id: str,
        strategy: str,
        batch_id: str,
        target_weights: Mapping[str, float],
        orders: Sequence[OrderRequest],
        *,
        created_at: str | None = None,
    ) -> str:
        """Persist one cycle/strategy execution batch; identical retries are idempotent."""
        cycle_id = _identifier(cycle_id, "cycle_id")
        strategy = _identifier(strategy, "strategy")
        batch_id = _identifier(batch_id, "batch_id")
        created_at = _timestamp(created_at)
        if not isinstance(target_weights, Mapping) or not target_weights:
            raise ValidationError("target_weights must be a non-empty mapping")
        weights = {
            _identifier(symbol, "symbol"): _number(weight, f"weight[{symbol}]", minimum=0, maximum=1)
            for symbol, weight in target_weights.items()
        }
        if math.fsum(weights.values()) > 1.0 + 1e-9:
            raise ValidationError("target weights must sum to no more than 1")
        order_rows: list[dict[str, object]] = []
        client_ids: set[str] = set()
        for order in orders:
            if not isinstance(order, OrderRequest):
                raise ValidationError("orders must contain OrderRequest values")
            client_id = _identifier(order.client_order_id, "client_order_id")
            symbol = _identifier(order.symbol, "symbol")
            try:
                side = OrderSide(order.side).value
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"Invalid order side: {order.side!r}") from exc
            quantity = _number(order.quantity, "order quantity", minimum=0, minimum_inclusive=False)
            if client_id in client_ids:
                raise ValidationError(f"Duplicate client_order_id: {client_id}")
            client_ids.add(client_id)
            order_rows.append(
                {"client_order_id": client_id, "symbol": symbol,
                 "side": side, "quantity": quantity}
            )
        payload = {
            "cycle_id": cycle_id,
            "strategy": strategy,
            "target_weights": weights,
            "orders": order_rows,
        }
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self._transaction() as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM reporting_cycles WHERE cycle_id = ? AND strategy = ?",
                (cycle_id, strategy), f"Cycle/strategy not found: {cycle_id}/{strategy}",
            )
            existing = connection.execute(
                """SELECT batch_id, payload_sha256 FROM execution_batches
                   WHERE cycle_id = ? AND strategy = ?""",
                (cycle_id, strategy),
            ).fetchone()
            if existing:
                if existing["batch_id"] == batch_id and existing["payload_sha256"] == payload_hash:
                    return str(batch_id)
                raise DuplicateRecordError(
                    f"A different execution batch already exists for {cycle_id}/{strategy}"
                )
            if connection.execute(
                "SELECT 1 FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone():
                raise DuplicateRecordError(f"Execution batch id already exists: {batch_id}")
            connection.execute(
                """INSERT INTO execution_batches
                   (batch_id, cycle_id, strategy, payload_sha256, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (batch_id, cycle_id, strategy, payload_hash, created_at),
            )
            connection.executemany(
                "INSERT INTO target_weights (batch_id, symbol, weight) VALUES (?, ?, ?)",
                ((batch_id, symbol, weight) for symbol, weight in sorted(weights.items())),
            )
            connection.executemany(
                """INSERT INTO simulated_orders
                   (order_id, batch_id, client_order_id, symbol, side, quantity)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    (str(uuid.uuid4()), batch_id, order["client_order_id"], order["symbol"],
                     order["side"], order["quantity"])
                    for order in order_rows
                ),
            )
        return batch_id

    def schedule_batch(
        self, batch_id: str, scheduled_for: str, *, planned_at: str | None = None, note: str = ""
    ) -> str:
        """Fix the (future) session date of a batch; identical retries are idempotent."""
        batch_id = _identifier(batch_id, "batch_id")
        scheduled_for = _identifier(scheduled_for, "scheduled_for")
        planned_at = _timestamp(planned_at)
        with self._transaction() as connection:
            self._require_row(
                connection, "SELECT 1 FROM execution_batches WHERE batch_id = ?",
                (batch_id,), f"Execution batch not found: {batch_id}",
            )
            existing = connection.execute(
                "SELECT scheduled_for FROM execution_schedule WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if existing:
                if existing["scheduled_for"] != scheduled_for:
                    raise DuplicateRecordError(f"Batch already scheduled: {batch_id}")
                return str(existing["scheduled_for"])
            connection.execute(
                """INSERT INTO execution_schedule (batch_id, scheduled_for, planned_at, note)
                   VALUES (?, ?, ?, ?)""",
                (batch_id, scheduled_for, planned_at, note),
            )
        return scheduled_for

    def set_cycle_status(self, cycle_id: str, status: str) -> None:
        """Advance a cycle OPEN -> SIGNALLED -> EXECUTED; repeating a status is a no-op."""
        cycle_id = _identifier(cycle_id, "cycle_id")
        order = {"OPEN": 0, "SIGNALLED": 1, "EXECUTED": 2}
        if status not in {"SIGNALLED", "EXECUTED"}:
            raise ValidationError("status must be SIGNALLED or EXECUTED")
        with self._transaction() as connection:
            cycle = self._require_row(
                connection, "SELECT status FROM reporting_cycles WHERE cycle_id = ?",
                (cycle_id,), f"Cycle not found: {cycle_id}",
            )
            current = cycle["status"]
            if current == "BLOCKED":
                raise StateError("blocked cycle cannot advance")
            if order[current] > order[status]:
                return
            connection.execute(
                "UPDATE reporting_cycles SET status = ? WHERE cycle_id = ?", (status, cycle_id)
            )

    def query(self, sql: str, params: Sequence[object] = ()) -> list[dict[str, object]]:
        """Run one read-only SELECT against the store."""
        if not sql.lstrip().upper().startswith("SELECT"):
            raise ValidationError("query() accepts SELECT statements only")
        with self._connection() as connection:
            connection.execute("PRAGMA query_only = ON")
            return [dict(row) for row in connection.execute(sql, tuple(params)).fetchall()]

    def orders_for_batch(self, batch_id: str) -> tuple[SimulatedOrder, ...]:
        batch_id = _identifier(batch_id, "batch_id")
        with self._connection() as connection:
            if connection.execute(
                "SELECT 1 FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone() is None:
                raise NotFoundError(f"Execution batch not found: {batch_id}")
            rows = connection.execute(
                """SELECT order_id, client_order_id, symbol, side, quantity,
                          filled_quantity, status
                   FROM simulated_orders WHERE batch_id = ? ORDER BY client_order_id""",
                (batch_id,),
            ).fetchall()
        return tuple(
            SimulatedOrder(
                order_id=str(row["order_id"]),
                client_order_id=str(row["client_order_id"]),
                symbol=str(row["symbol"]),
                side=OrderSide(row["side"]),
                quantity=float(row["quantity"]),
                filled_quantity=float(row["filled_quantity"]),
                status=str(row["status"]),
            )
            for row in rows
        )

    def record_fill(
        self,
        execution_id: str,
        order_id: str,
        quantity: float,
        price: float,
        *,
        transaction_cost_bps_per_side: float = 10.0,
        spread_slippage_bps_per_side: float = 0.0,
        filled_at: str | None = None,
    ) -> str:
        execution_id = _identifier(execution_id, "execution_id")
        order_id = _identifier(order_id, "order_id")
        quantity = _number(quantity, "fill quantity", minimum=0, minimum_inclusive=False)
        price = _number(price, "fill price", minimum=0)
        transaction_cost_bps_per_side = _number(
            transaction_cost_bps_per_side, "transaction_cost_bps_per_side", minimum=0
        )
        spread_slippage_bps_per_side = _number(
            spread_slippage_bps_per_side, "spread_slippage_bps_per_side", minimum=0
        )
        filled_at = _timestamp(filled_at)
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM simulated_fills WHERE execution_id = ?", (execution_id,)
            ).fetchone():
                raise DuplicateRecordError(f"Execution already recorded: {execution_id}")
            order = connection.execute(
                """SELECT o.quantity, o.filled_quantity, o.side, o.symbol,
                          b.cycle_id, c.portfolio_id
                   FROM simulated_orders AS o
                   JOIN execution_batches AS b ON b.batch_id = o.batch_id
                   JOIN reporting_cycles AS c ON c.cycle_id = b.cycle_id
                   WHERE o.order_id = ?""",
                (order_id,),
            ).fetchone()
            if order is None:
                raise NotFoundError(f"Order not found: {order_id}")
            if order["filled_quantity"] + quantity > order["quantity"] + 1e-9:
                raise StateError("Fill quantity exceeds the order's remaining quantity")
            portfolio_id = str(order["portfolio_id"])
            position = connection.execute(
                """SELECT COALESCE(SUM(quantity_delta), 0) AS quantity
                   FROM position_ledger WHERE portfolio_id = ? AND symbol = ?""",
                (portfolio_id, order["symbol"]),
            ).fetchone()
            signed_quantity = quantity if order["side"] == OrderSide.BUY.value else -quantity
            if float(position["quantity"]) + signed_quantity < -1e-9:
                raise StateError(f"Fill would create a short position in {order['symbol']}")
            notional = quantity * price
            transaction_cost = notional * (
                transaction_cost_bps_per_side + spread_slippage_bps_per_side
            ) / 10_000.0
            current_cash = float(connection.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM cash_ledger WHERE portfolio_id = ?",
                (portfolio_id,),
            ).fetchone()[0])
            cash_delta = notional if signed_quantity < 0 else -notional
            cash_delta -= transaction_cost
            if current_cash + cash_delta < -1e-8:
                raise StateError("Fill would make available cash negative")
            connection.execute(
                """INSERT INTO simulated_fills
                   (fill_id, execution_id, order_id, quantity, price, transaction_cost, filled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()), execution_id, order_id, quantity, price,
                    transaction_cost, filled_at,
                ),
            )
            new_filled = float(order["filled_quantity"]) + quantity
            status = "FILLED" if math.isclose(new_filled, order["quantity"], abs_tol=1e-9) else "PARTIAL"
            connection.execute(
                "UPDATE simulated_orders SET filled_quantity = ?, status = ? WHERE order_id = ?",
                (new_filled, status, order_id),
            )
            connection.execute(
                """INSERT INTO cash_ledger
                   (entry_id, idempotency_key, portfolio_id, entry_type, amount,
                    created_at, execution_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()), f"fill:{execution_id}", portfolio_id,
                    f"FILL_{order['side']}", cash_delta, filled_at, execution_id,
                ),
            )
            connection.execute(
                """INSERT INTO position_ledger
                   (entry_id, idempotency_key, portfolio_id, symbol, quantity_delta,
                    entry_type, created_at, execution_id)
                   VALUES (?, ?, ?, ?, ?, 'FILL', ?, ?)""",
                (
                    str(uuid.uuid4()), f"position:{execution_id}", portfolio_id,
                    order["symbol"], signed_quantity, filled_at, execution_id,
                ),
            )
        return execution_id

    def record_daily_mark(
        self,
        portfolio_id: str,
        marked_at: str,
        cash: float,
        positions_value: float,
        nav: float,
        benchmark_value: float,
        dividend_receivables: float = 0.0,
        position_values: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        portfolio_id = _identifier(portfolio_id, "portfolio_id")
        marked_at = _timestamp(marked_at)
        cash = _number(cash, "cash", minimum=0)
        positions_value = _number(positions_value, "positions_value", minimum=0)
        dividend_receivables = _number(dividend_receivables, "dividend_receivables", minimum=0)
        nav = _number(nav, "nav", minimum=0, minimum_inclusive=False)
        benchmark_value = _number(benchmark_value, "benchmark_value", minimum=0, minimum_inclusive=False)
        serialized_positions = None
        if position_values is not None:
            marked_positions: dict[str, dict[str, float]] = {}
            for symbol, (quantity, price) in position_values.items():
                symbol = _identifier(symbol, "symbol")
                quantity = _number(quantity, f"quantity[{symbol}]", minimum=0, minimum_inclusive=False)
                price = _number(price, f"price[{symbol}]", minimum=0, minimum_inclusive=False)
                marked_positions[symbol] = {
                    "quantity": quantity,
                    "price": price,
                    "market_value": quantity * price,
                }
            marked_value = math.fsum(item["market_value"] for item in marked_positions.values())
            if not math.isclose(marked_value, positions_value, rel_tol=1e-9, abs_tol=1e-8):
                raise ValidationError("position marks must sum to positions_value")
            serialized_positions = json.dumps(
                marked_positions, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        if not math.isclose(
            nav, cash + positions_value + dividend_receivables, rel_tol=1e-9, abs_tol=1e-8
        ):
            raise ValidationError("nav must equal cash + positions_value + dividend_receivables")
        with self._transaction() as connection:
            self._require_row(
                connection, "SELECT 1 FROM portfolios WHERE portfolio_id = ?",
                (portfolio_id,), f"Portfolio not found: {portfolio_id}",
            )
            if connection.execute(
                "SELECT 1 FROM daily_marks WHERE portfolio_id = ? AND marked_at = ?",
                (portfolio_id, marked_at),
            ).fetchone():
                raise DuplicateRecordError(f"Daily mark already exists for {portfolio_id}/{marked_at}")
            connection.execute(
                """INSERT INTO daily_marks
                   (portfolio_id, marked_at, cash, positions_value, dividend_receivables,
                    nav, benchmark_value, position_values_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    portfolio_id, marked_at, cash, positions_value,
                    dividend_receivables, nav, benchmark_value, serialized_positions,
                ),
            )

    def start_job(
        self,
        run_id: str,
        job_type: str,
        event_type: str,
        event_payload: Mapping[str, object],
        *,
        started_at: str | None = None,
    ) -> str:
        """Atomically insert a running job and its initial event."""
        run_id = _identifier(run_id, "run_id")
        job_type = _identifier(job_type, "job_type")
        event_type = _identifier(event_type, "event_type")
        payload_json, _ = _canonical_payload(event_payload)
        started_at = _timestamp(started_at)
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM job_runs WHERE run_id = ?", (run_id,)
            ).fetchone():
                raise DuplicateRecordError(f"Job run already exists: {run_id}")
            connection.execute(
                """INSERT INTO job_runs (run_id, job_type, status, started_at)
                   VALUES (?, ?, 'RUNNING', ?)""",
                (run_id, job_type, started_at),
            )
            connection.execute(
                """INSERT INTO job_events
                   (event_id, run_id, event_type, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), run_id, event_type, payload_json, started_at),
            )
        return run_id

    def log_job_event(
        self, run_id: str, event_type: str, payload: Mapping[str, object],
        *, created_at: str | None = None,
    ) -> None:
        run_id = _identifier(run_id, "run_id")
        event_type = _identifier(event_type, "event_type")
        payload_json, _ = _canonical_payload(payload)
        created_at = _timestamp(created_at)
        with self._transaction() as connection:
            self._require_row(
                connection, "SELECT 1 FROM job_runs WHERE run_id = ?",
                (run_id,), f"Job run not found: {run_id}",
            )
            connection.execute(
                """INSERT INTO job_events (event_id, run_id, event_type, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), run_id, event_type, payload_json, created_at),
            )

    def record_job_error(
        self,
        run_id: str,
        message: str,
        details: Mapping[str, object],
        *,
        created_at: str | None = None,
    ) -> str:
        run_id = _identifier(run_id, "run_id")
        message = _text(message, "message")
        details_json, _ = _canonical_payload(details)
        created_at = _timestamp(created_at)
        error_id = str(uuid.uuid4())
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT status FROM job_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise NotFoundError(f"Job run not found: {run_id}")
            if run["status"] != JobStatus.RUNNING.value:
                raise StateError(f"Job run is not running: {run_id}")
            connection.execute(
                """INSERT INTO job_errors
                   (error_id, run_id, message, details_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (error_id, run_id, message, details_json, created_at),
            )
            connection.execute(
                "UPDATE job_runs SET status = 'FAILED', finished_at = ? WHERE run_id = ?",
                (created_at, run_id),
            )
        return error_id

    def finish_job(
        self,
        run_id: str,
        *,
        finished_at: str | None = None,
    ) -> None:
        run_id = _identifier(run_id, "run_id")
        finished_at = _timestamp(finished_at)
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE job_runs SET status = 'SUCCEEDED', finished_at = ?
                   WHERE run_id = ? AND status = 'RUNNING'""",
                (finished_at, run_id),
            )
            if cursor.rowcount != 1:
                if connection.execute(
                    "SELECT 1 FROM job_runs WHERE run_id = ?", (run_id,)
                ).fetchone():
                    raise StateError(f"Job run is not running: {run_id}")
                raise NotFoundError(f"Job run not found: {run_id}")

    def cash_balance(self, portfolio_id: str) -> float:
        portfolio_id = _identifier(portfolio_id, "portfolio_id")
        with self._connection() as connection:
            row = connection.execute(
                """SELECT COALESCE(SUM(amount), 0) AS balance,
                          (SELECT COUNT(*) FROM portfolios WHERE portfolio_id = ?)
                              AS portfolio_count
                   FROM cash_ledger WHERE portfolio_id = ?""",
                (portfolio_id, portfolio_id),
            ).fetchone()
        if row is None or row["portfolio_count"] == 0:
            raise NotFoundError(f"Portfolio not found: {portfolio_id}")
        return float(row["balance"])

    def position_balance(self, portfolio_id: str, symbol: str) -> float:
        portfolio_id = _identifier(portfolio_id, "portfolio_id")
        symbol = _identifier(symbol, "symbol")
        with self._connection() as connection:
            portfolio = connection.execute(
                "SELECT 1 FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)
            ).fetchone()
            if portfolio is None:
                raise NotFoundError(f"Portfolio not found: {portfolio_id}")
            row = connection.execute(
                """SELECT COALESCE(SUM(quantity_delta), 0) AS quantity
                   FROM position_ledger WHERE portfolio_id = ? AND symbol = ?""",
                (portfolio_id, symbol),
            ).fetchone()
        return float(row["quantity"])

    def receivable_balance(self, portfolio_id: str) -> float:
        portfolio_id = _identifier(portfolio_id, "portfolio_id")
        with self._connection() as connection:
            portfolio = connection.execute(
                "SELECT 1 FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)
            ).fetchone()
            if portfolio is None:
                raise NotFoundError(f"Portfolio not found: {portfolio_id}")
            row = connection.execute(
                """SELECT COALESCE(SUM(amount), 0) AS amount FROM dividend_receivables
                   WHERE portfolio_id = ? AND status = 'RECEIVABLE'""",
                (portfolio_id,),
            ).fetchone()
        return float(row["amount"])

    def record_corporate_action(
        self,
        action_id: str,
        portfolio_id: str,
        symbol: str,
        action_type: str,
        ex_date: str,
        value: float,
        source: str,
        *,
        pay_date: str | None = None,
        created_at: str | None = None,
        entitled_quantity: float | None = None,
    ) -> str:
        """Book a split, cash dividend receivable, or cash-settled delisting once."""
        action_id = _identifier(action_id, "action_id")
        portfolio_id = _identifier(portfolio_id, "portfolio_id")
        symbol = _identifier(symbol, "symbol")
        action_type = _identifier(action_type, "action_type")
        if action_type not in {"split", "dividend", "cash_delisting"}:
            raise ValidationError(
                "action_type must be split, dividend, or cash_delisting; "
                "stock-for-stock mergers need explicit conversion terms"
            )
        value = _number(value, "value", minimum=0, minimum_inclusive=False)
        source = _text(source, "source")
        ex_date = _timestamp(ex_date)
        if action_type == "dividend":
            if pay_date is not None:
                pay_date = _timestamp(pay_date)
                parsed_pay = datetime.fromisoformat(pay_date.replace("Z", "+00:00"))
                parsed_ex = datetime.fromisoformat(ex_date.replace("Z", "+00:00"))
                if parsed_pay < parsed_ex:
                    raise ValidationError("dividend pay_date cannot precede ex_date")
        elif pay_date is not None:
            raise ValidationError("pay_date is only valid for cash dividends")
        created_at = _timestamp(created_at)
        if entitled_quantity is not None and action_type != "dividend":
            raise ValidationError("entitled_quantity is only valid for dividend actions")
        if entitled_quantity is not None:
            entitled_quantity = _number(
                entitled_quantity, "entitled_quantity", minimum=0
            )
        with self._transaction() as connection:
            self._require_row(
                connection, "SELECT 1 FROM portfolios WHERE portfolio_id = ?",
                (portfolio_id,), f"Portfolio not found: {portfolio_id}",
            )
            if connection.execute(
                "SELECT 1 FROM corporate_actions WHERE action_id = ?", (action_id,)
            ).fetchone():
                raise DuplicateRecordError(f"Corporate action already recorded: {action_id}")
            held = float(connection.execute(
                """SELECT COALESCE(SUM(quantity_delta), 0)
                   FROM position_ledger WHERE portfolio_id = ? AND symbol = ?""",
                (portfolio_id, symbol),
            ).fetchone()[0])
            if held < -1e-9:
                raise StateError(f"Position ledger is short {symbol} before action")
            dividend_quantity = held if entitled_quantity is None else entitled_quantity
            status = (
                "NO_POSITION" if (dividend_quantity if action_type == "dividend" else held) <= 1e-9 else
                "RECEIVABLE" if action_type == "dividend" else
                "SETTLED" if action_type == "cash_delisting" else
                "APPLIED"
            )
            connection.execute(
                """INSERT INTO corporate_actions
                   (action_id, portfolio_id, symbol, action_type, ex_date, pay_date,
                    value, source, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    action_id, portfolio_id, symbol, action_type, ex_date, pay_date,
                    value, source, status, created_at,
                ),
            )
            if held > 1e-9 and action_type == "split":
                connection.execute(
                    """INSERT INTO position_ledger
                       (entry_id, idempotency_key, portfolio_id, symbol, quantity_delta,
                        entry_type, created_at, execution_id)
                       VALUES (?, ?, ?, ?, ?, 'SPLIT', ?, ?)""",
                    (
                        str(uuid.uuid4()), f"split:{action_id}", portfolio_id,
                        symbol, held * (value - 1.0), created_at, f"split:{action_id}",
                    ),
                )
            elif dividend_quantity > 1e-9 and action_type == "dividend":
                connection.execute(
                    """INSERT INTO dividend_receivables
                       (action_id, portfolio_id, symbol, amount, pay_date, status)
                       VALUES (?, ?, ?, ?, ?, 'RECEIVABLE')""",
                    (action_id, portfolio_id, symbol, dividend_quantity * value, pay_date),
                )
            elif held > 1e-9 and action_type == "cash_delisting":
                connection.execute(
                    """INSERT INTO position_ledger
                       (entry_id, idempotency_key, portfolio_id, symbol, quantity_delta,
                        entry_type, created_at, execution_id)
                       VALUES (?, ?, ?, ?, ?, 'CASH_DELISTING', ?, ?)""",
                    (
                        str(uuid.uuid4()), f"delisting:{action_id}", portfolio_id,
                        symbol, -held, created_at, f"delisting:{action_id}",
                    ),
                )
                connection.execute(
                    """INSERT INTO cash_ledger
                       (entry_id, idempotency_key, portfolio_id, entry_type, amount,
                        created_at, execution_id)
                       VALUES (?, ?, ?, 'CASH_DELISTING', ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), f"delisting:{action_id}", portfolio_id,
                        held * value, created_at, f"delisting:{action_id}",
                    ),
                )
        return action_id

    def set_dividend_pay_date(self, action_id: str, pay_date: str, source: str,
                              *, recorded_at: str | None = None) -> bool:
        """Attach an evidenced pay date to an open receivable; repeating it never books anything twice.

        Returns True when the date was newly stored. A different date for an already dated claim is
        refused instead of silently replaced.
        """
        action_id = _identifier(action_id, "action_id")
        pay_date = _timestamp(pay_date)
        source = _text(source, "source")
        recorded_at = _timestamp(recorded_at)
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT r.pay_date, r.status, a.ex_date FROM dividend_receivables r
                   JOIN corporate_actions a ON a.action_id = r.action_id WHERE r.action_id = ?""",
                (action_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Dividend receivable not found: {action_id}")
            if row["pay_date"] is not None:
                if row["pay_date"] == pay_date:
                    return False
                raise StateError(f"Dividend already has pay date {row['pay_date']}: {action_id}")
            if pay_date < _timestamp(row["ex_date"]):
                raise ValidationError("dividend pay_date cannot precede ex_date")
            connection.execute(
                "UPDATE dividend_receivables SET pay_date = ? WHERE action_id = ?", (pay_date, action_id))
            connection.execute(
                "INSERT INTO dividend_pay_dates (action_id, pay_date, source, recorded_at) VALUES (?, ?, ?, ?)",
                (action_id, pay_date, source, recorded_at))
        return True

    def resolve_filing(self, cycle_id: str, filing_key: str, evidence: Mapping[str, object],
                       *, recorded_at: str | None = None) -> None:
        """Record an evidenced CORPORATE_ACTION_RESOLVED case; it never changes the filing's own status."""
        cycle_id = _identifier(cycle_id, "cycle_id")
        filing_key = _identifier(filing_key, "filing_key")
        if not isinstance(evidence, Mapping) or not evidence.get("evidence"):
            raise ValidationError("a resolution needs primary-source evidence")
        recorded_at = _timestamp(recorded_at)
        with self._transaction() as connection:
            self._require_row(
                connection, "SELECT 1 FROM expected_filings WHERE cycle_id = ? AND filing_key = ?",
                (cycle_id, filing_key), f"Expected filing not found: {cycle_id}/{filing_key}")
            connection.execute(
                """INSERT OR IGNORE INTO filing_resolutions
                   (cycle_id, filing_key, resolution, evidence_json, recorded_at)
                   VALUES (?, ?, 'CORPORATE_ACTION_RESOLVED', ?, ?)""",
                (cycle_id, filing_key, json.dumps(dict(evidence), sort_keys=True), recorded_at))
            self._refresh_filing_status(connection, cycle_id)

    def pay_dividend(
        self,
        action_id: str,
        *,
        paid_at: str | None = None,
    ) -> None:
        action_id = _identifier(action_id, "action_id")
        paid_at = _timestamp(paid_at)
        with self._transaction() as connection:
            receivable = connection.execute(
                """SELECT portfolio_id, amount, status, pay_date FROM dividend_receivables
                   WHERE action_id = ?""",
                (action_id,),
            ).fetchone()
            if receivable is None:
                raise NotFoundError(f"Dividend receivable not found: {action_id}")
            if receivable["status"] != "RECEIVABLE":
                raise StateError(f"Dividend has already been paid: {action_id}")
            if receivable["pay_date"] is None:
                raise StateError(f"Dividend pay date is not yet evidenced: {action_id}")
            paid_datetime = datetime.fromisoformat(paid_at.replace("Z", "+00:00"))
            pay_datetime = datetime.fromisoformat(receivable["pay_date"].replace("Z", "+00:00"))
            if paid_datetime < pay_datetime:
                raise StateError(f"Dividend cannot be paid before its pay_date: {action_id}")
            connection.execute(
                """INSERT INTO cash_ledger
                   (entry_id, idempotency_key, portfolio_id, entry_type, amount,
                    created_at, execution_id)
                   VALUES (?, ?, ?, 'DIVIDEND', ?, ?, ?)""",
                (
                    str(uuid.uuid4()), f"dividend:{action_id}", receivable["portfolio_id"],
                    receivable["amount"], paid_at, f"dividend:{action_id}",
                ),
            )
            connection.execute(
                """UPDATE dividend_receivables SET status = 'PAID', paid_at = ?
                   WHERE action_id = ? AND status = 'RECEIVABLE'""",
                (paid_at, action_id),
            )

    def job_status(self, run_id: str) -> JobStatus:
        run_id = _identifier(run_id, "run_id")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM job_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Job run not found: {run_id}")
        return JobStatus(row["status"])

    @staticmethod
    def _require_row(
        connection: sqlite3.Connection,
        query: str,
        parameters: tuple[object, ...],
        message: str,
    ) -> sqlite3.Row:
        row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise NotFoundError(message)
        return row

    @staticmethod
    def _refresh_filing_status(connection: sqlite3.Connection, cycle_id: str) -> None:
        filings = connection.execute(
            """SELECT e.status FROM expected_filings e
               LEFT JOIN filing_resolutions r ON r.cycle_id = e.cycle_id AND r.filing_key = e.filing_key
               WHERE e.cycle_id = ? AND (r.filing_key IS NULL OR e.status = 'INVALID_DATA')""",
            (cycle_id,)).fetchall()
        total = connection.execute(
            "SELECT COUNT(*) FROM expected_filings WHERE cycle_id = ?", (cycle_id,)).fetchone()[0]
        if not total:
            status = FilingStatus.WAITING_FILING
        else:
            values = {row["status"] for row in filings}
            if FilingStatus.INVALID_DATA.value in values:
                status = FilingStatus.INVALID_DATA
            elif FilingStatus.WAITING_FILING.value in values:
                status = FilingStatus.WAITING_FILING
            elif FilingStatus.WAITING_FACTS.value in values:
                status = FilingStatus.WAITING_FACTS
            else:
                status = FilingStatus.READY
        connection.execute(
            "UPDATE reporting_cycles SET filing_status = ? WHERE cycle_id = ?",
            (status.value, cycle_id),
        )


def _database_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValidationError("path must be a non-empty string or Path")
    return Path(path).expanduser().resolve()


def _database_uri(path: Path, mode: str) -> str:
    return f"{path.as_uri()}?mode={mode}"


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValidationError(f"{name} must be a non-empty trimmed string")
    if "\x00" in value:
        raise ValidationError(f"{name} must not contain NUL")
    return value


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value


def _number(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be a finite number")
    if minimum is not None and (
        result < minimum or (not minimum_inclusive and result == minimum)
    ):
        relation = "greater than" if not minimum_inclusive else "at least"
        raise ValidationError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValidationError(f"{name} must be no more than {maximum}")
    return result


def _timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("timestamp must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("timestamp must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        if len(value) == 10:
            return parsed.replace(tzinfo=timezone.utc).isoformat()
        raise ValidationError("timestamp must include a timezone")
    return value


def _aware(value: str, name: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValidationError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_payload(payload: Mapping[str, object]) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        raise ValidationError("payload must be a mapping")
    try:
        serialized = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"payload must contain JSON-compatible values: {exc}") from exc
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return serialized, digest


def _filing_status(received: bool, facts_ready: bool, valid: bool) -> FilingStatus:
    if not valid:
        return FilingStatus.INVALID_DATA
    if not received:
        return FilingStatus.WAITING_FILING
    if not facts_ready:
        return FilingStatus.WAITING_FACTS
    return FilingStatus.READY
