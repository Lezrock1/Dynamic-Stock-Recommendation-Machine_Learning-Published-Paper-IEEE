import json
import sqlite3
import unittest
import uuid
from pathlib import Path

from pipeline.paper_store import (
    DuplicateRecordError,
    FilingStatus,
    NotFoundError,
    OrderRequest,
    OrderSide,
    PaperStore,
    PaperStoreError,
    StateError,
    StoreNotFoundError,
)
from pipeline.paper_dashboard import dashboard_snapshot


class PaperStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = Path(__file__).parent / f".paper-store-test-{uuid.uuid4().hex}.sqlite3"

    def tearDown(self) -> None:
        for path in (
            self.database,
            Path(f"{self.database}-wal"),
            Path(f"{self.database}-shm"),
        ):
            path.unlink(missing_ok=True)

    def new_store(self) -> PaperStore:
        return PaperStore.create(self.database)

    def add_portfolio(self, store: PaperStore) -> None:
        store.create_portfolio("paper", "Paper Portfolio", "strategy-a", 10_000.0)

    def test_missing_database_requires_explicit_create_and_create_is_idempotent(self) -> None:
        with self.assertRaises(StoreNotFoundError):
            PaperStore(self.database)

        store = PaperStore.create(self.database)
        self.add_portfolio(store)
        reopened = PaperStore.create(self.database)
        self.assertEqual(reopened.cash_balance("paper"), 10_000.0)
        self.assertEqual(PaperStore(self.database).cash_balance("paper"), 10_000.0)

        with self.assertRaises(DuplicateRecordError):
            reopened.create_portfolio("paper", "Other", "strategy-a", 1.0)

    def test_portfolio_and_cycle_survive_reopen_and_cycle_creation_is_idempotent(self) -> None:
        store = self.new_store()
        self.add_portfolio(store)
        first_cycle = store.create_cycle("paper", "2026-Q1", "strategy-a")
        reopened = PaperStore(self.database)
        second_cycle = reopened.create_cycle("paper", "2026-Q1", "strategy-a")

        self.assertEqual(first_cycle, second_cycle)
        self.assertEqual(reopened.cash_balance("paper"), 10_000.0)
        self.assertEqual(reopened.filing_status(first_cycle), FilingStatus.WAITING_FILING)
        reopened.record_daily_mark("paper", "2026-06-30", 9_000, 1_000, 10_000, 125)
        with sqlite3.connect(self.database) as connection:
            mark = connection.execute(
                "SELECT nav, benchmark_value FROM daily_marks WHERE portfolio_id = 'paper'"
            ).fetchone()
        self.assertEqual(mark, (10_000.0, 125.0))

    def test_dashboard_reads_cycle_and_valuation_status(self) -> None:
        store = self.new_store()
        self.add_portfolio(store)
        cycle_id = store.create_cycle("paper", "cycle-dashboard", "strategy-a")
        store.register_expected_filings(
            cycle_id,
            ["filing-a"],
            filing_metadata={
                "filing-a": {
                    "company_id": "cik-0000000001",
                    "expected_period": "FY2025-Q4-ending-2025-12-31",
                    "filing_type": "10-K",
                    "source": "synthetic directory",
                }
            },
        )
        store.record_daily_mark("paper", "2026-01-02", 10_000, 0, 10_000, 125)
        snapshot = dashboard_snapshot(self.database)
        self.assertEqual(snapshot["valuation_quality"], "LATEST_DAILY_MARKS_RECORDED")
        self.assertEqual(snapshot["portfolios"][0]["latest_mark"]["nav"], 10_000.0)
        self.assertEqual(snapshot["cycles"][0]["expected_count"], 1)
        self.assertEqual(snapshot["cycles"][0]["ready_count"], 0)

    def test_daily_mark_persists_immutable_per_symbol_valuation(self) -> None:
        store = self.new_store()
        self.add_portfolio(store)
        store.record_daily_mark(
            "paper", "2026-01-02T21:00:00+00:00", 9_800, 200, 10_000, 125,
            position_values={"AAA": (2, 100)},
        )
        with sqlite3.connect(self.database) as connection:
            mark = connection.execute(
                "SELECT position_values_json FROM daily_marks WHERE portfolio_id = 'paper'"
            ).fetchone()
            self.assertEqual(
                json.loads(mark[0]),
                {"AAA": {"market_value": 200.0, "price": 100.0, "quantity": 2.0}},
            )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE daily_marks SET cash = 1 WHERE portfolio_id = 'paper'"
                )
        with self.assertRaisesRegex(PaperStoreError, "position marks must sum"):
            store.record_daily_mark(
                "paper", "2026-01-03T21:00:00+00:00", 9_800, 200, 10_000, 125,
                position_values={"AAA": (1, 100)},
            )

    def test_rebalance_is_idempotent_and_conflicting_batches_are_rejected(self) -> None:
        store = self.new_store()
        self.add_portfolio(store)
        cycle_id = store.create_cycle("paper", "cycle-1", "strategy-a")
        weights = {"AAA": 0.6, "BBB": 0.2}
        orders = [
            OrderRequest("order-1", "AAA", OrderSide.BUY, 10),
            OrderRequest("order-2", "BBB", OrderSide.BUY, 5),
        ]

        self.assertEqual(store.rebalance(cycle_id, "strategy-a", "batch-1", weights, orders), "batch-1")
        self.assertEqual(store.rebalance(cycle_id, "strategy-a", "batch-1", weights, orders), "batch-1")
        with self.assertRaises(DuplicateRecordError):
            store.rebalance(
                cycle_id, "strategy-a", "batch-2", {"AAA": 0.5}, []
            )

        with sqlite3.connect(self.database) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM execution_batches").fetchone()[0], 1
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM target_weights").fetchone()[0], 2
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM simulated_orders").fetchone()[0], 2
            )

    def test_filing_readiness_requires_all_500_filings_and_complete_facts(self) -> None:
        store = self.new_store()
        self.add_portfolio(store)
        cycle_id = store.create_cycle("paper", "cycle-500", "strategy-a")
        filing_keys = [f"filing-{index:03}" for index in range(500)]
        store.register_expected_filings(
            cycle_id,
            filing_keys,
            filing_metadata={
                key: {
                    "company_id": f"cik-{index:010}",
                    "security_id": f"security-{index:03}",
                    "expected_period": "2026-Q1-ending-2026-03-31",
                    "filing_type": "10-Q",
                    "source": "predeclared synthetic directory",
                }
                for index, key in enumerate(filing_keys)
            },
        )

        for filing_key in filing_keys[:-1]:
            store.set_filing_result(
                cycle_id, filing_key, received=True, facts_ready=True, valid=True,
                filed_at="2026-05-01T12:00:00Z",
                first_seen_at="2026-05-04T14:00:00Z",
                source="synthetic SEC filing/facts fixture",
            )
        self.assertEqual(store.filing_status(cycle_id), FilingStatus.WAITING_FILING)
        with self.assertRaises(StateError):
            store.set_cycle_model_status(cycle_id, "READY")

        store.set_filing_result(
            cycle_id, filing_keys[-1], received=True, facts_ready=False, valid=True,
            filed_at="2026-05-01T12:00:00Z",
            first_seen_at="2026-05-04T14:00:00Z",
            source="synthetic SEC filing without required facts",
        )
        self.assertEqual(store.filing_status(cycle_id), FilingStatus.WAITING_FACTS)
        store.set_filing_result(
            cycle_id, filing_keys[-1], received=True, facts_ready=True, valid=True,
            filed_at="2026-05-01T12:00:00Z",
            first_seen_at="2026-05-04T14:00:00Z",
            source="synthetic SEC filing/facts fixture",
        )
        self.assertEqual(store.filing_status(cycle_id), FilingStatus.READY)
        store.set_cycle_model_status(cycle_id, "READY")

        store.set_filing_result(
            cycle_id, filing_keys[0], received=True, facts_ready=True, valid=False,
            filed_at="2026-05-01T12:00:00Z",
            first_seen_at="2026-05-04T14:00:00Z",
            source="synthetic SEC filing/facts fixture",
            error_reason="synthetic validation failure",
        )
        self.assertEqual(store.filing_status(cycle_id), FilingStatus.INVALID_DATA)

    def test_split_dividend_and_cash_delisting_are_persisted_once(self) -> None:
        store = self.new_store()
        store.create_portfolio("paper", "Paper", "strategy-a", 1_000)
        cycle_id = store.create_cycle("paper", "cycle-actions", "strategy-a")
        store.rebalance(
            cycle_id,
            "strategy-a",
            "batch-actions",
            {"AAA": 1.0},
            [OrderRequest("buy-aaa", "AAA", OrderSide.BUY, 5)],
        )
        order = store.orders_for_batch("batch-actions")[0]
        store.record_fill("buy-aaa-fill", order.order_id, 5, 100)
        self.assertAlmostEqual(store.cash_balance("paper"), 499.5)
        store.record_corporate_action(
            "split-1", "paper", "AAA", "split", "2026-01-05", 2.0, "fixture"
        )
        self.assertEqual(store.position_balance("paper", "AAA"), 10.0)
        store.record_corporate_action(
            "dividend-1", "paper", "AAA", "dividend", "2026-01-06", 2.0,
            "fixture", pay_date="2026-01-12",
        )
        self.assertEqual(store.receivable_balance("paper"), 20.0)
        self.assertAlmostEqual(store.cash_balance("paper"), 499.5)
        with self.assertRaises(DuplicateRecordError):
            store.record_corporate_action(
                "dividend-1", "paper", "AAA", "dividend", "2026-01-06", 2.0,
                "fixture", pay_date="2026-01-12",
            )
        with self.assertRaises(StateError):
            store.pay_dividend("dividend-1", paid_at="2026-01-10")
        store.pay_dividend("dividend-1", paid_at="2026-01-12")
        self.assertEqual(store.receivable_balance("paper"), 0.0)
        self.assertAlmostEqual(store.cash_balance("paper"), 519.5)
        store.record_corporate_action(
            "delist-1", "paper", "AAA", "cash_delisting", "2026-01-13",
            50.0, "fixture",
        )
        self.assertEqual(store.position_balance("paper", "AAA"), 0.0)
        self.assertAlmostEqual(store.cash_balance("paper"), 1_019.5)

    def test_job_start_and_initial_event_are_atomic_and_duplicate_start_rejected(self) -> None:
        store = self.new_store()
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """CREATE TRIGGER reject_initial_event BEFORE INSERT ON job_events
                   BEGIN SELECT RAISE(ABORT, 'injected event failure'); END"""
            )
        with self.assertRaises(PaperStoreError):
            store.start_job("run-atomic", "rebalance", "started", {"cycle": "cycle-1"})
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0], 0)
            connection.execute("DROP TRIGGER reject_initial_event")

        store.start_job("run-1", "rebalance", "started", {"cycle": "cycle-1"})

        with self.assertRaises(DuplicateRecordError):
            store.start_job("run-1", "rebalance", "started-again", {"cycle": "cycle-2"})

        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM job_events").fetchone()[0], 1)

        store.record_job_error("run-1", "failed", {"reason": "test"})
        self.assertEqual(store.job_status("run-1").value, "FAILED")
        with self.assertRaises(StateError):
            store.finish_job("run-1")

    def test_data_and_signal_snapshots_are_append_only(self) -> None:
        store = self.new_store()
        self.add_portfolio(store)
        cycle_id = store.create_cycle("paper", "cycle-snapshots", "strategy-a")
        store.record_data_snapshot(cycle_id, "snapshot-1", {"source": "test"})
        store.record_signal_snapshot(
            cycle_id, "strategy-a", "AAA", {"score": 0.75}, signal_id="signal-1"
        )
        with sqlite3.connect(self.database) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE data_snapshots SET payload_json = '{}' WHERE snapshot_id = 'snapshot-1'"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "DELETE FROM signal_snapshots WHERE signal_id = 'signal-1'"
                )

    def test_fill_is_idempotency_guarded_and_updates_the_cash_ledger(self) -> None:
        store = self.new_store()
        self.add_portfolio(store)
        cycle_id = store.create_cycle("paper", "cycle-fill", "strategy-a")
        store.rebalance(
            cycle_id,
            "strategy-a",
            "batch-fill",
            {"AAA": 1.0},
            [OrderRequest("buy-aaa", "AAA", OrderSide.BUY, 10)],
        )
        order = store.orders_for_batch("batch-fill")[0]
        order_id = order.order_id
        self.assertEqual(order.status, "OPEN")

        store.record_fill("execution-1", order_id, 4, 12.5)
        self.assertAlmostEqual(store.cash_balance("paper"), 9_949.95)
        self.assertEqual(store.position_balance("paper", "AAA"), 4.0)
        self.assertEqual(store.orders_for_batch("batch-fill")[0].filled_quantity, 4.0)
        with self.assertRaises(DuplicateRecordError):
            store.record_fill("execution-1", order_id, 4, 12.5)
        with self.assertRaises(StateError):
            store.record_fill("execution-2", order_id, 7, 12.5)
        self.assertAlmostEqual(store.cash_balance("paper"), 9_949.95)

    def test_unknown_records_raise_explicit_errors(self) -> None:
        store = self.new_store()
        with self.assertRaises(NotFoundError):
            store.filing_status("missing-cycle")
        with self.assertRaises(NotFoundError):
            store.cash_balance("missing-portfolio")


if __name__ == "__main__":
    unittest.main()
