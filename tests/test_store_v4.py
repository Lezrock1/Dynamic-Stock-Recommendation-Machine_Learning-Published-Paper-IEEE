import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline.paper_store import (
    FilingStatus, OrderRequest, OrderSide, PaperStore, StateError,
)

REAL_DB = Path(__file__).parents[1] / "state" / "real_paper.sqlite3"


class StoreMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.store = PaperStore.create(self.dir / "s.sqlite3")
        self.store.create_portfolio("paper", "Paper", "strategy-a", 1_000)
        cycle = self.store.create_cycle("paper", "c", "strategy-a")
        self.store.rebalance(cycle, "strategy-a", "b", {"AAA": 1.0},
                             [OrderRequest("o", "AAA", OrderSide.BUY, 5)])
        self.store.record_fill("f", self.store.orders_for_batch("b")[0].order_id, 5, 100)

    def test_dividend_without_pay_date_is_receivable_and_not_cash(self) -> None:
        cash = self.store.cash_balance("paper")
        self.store.record_corporate_action("d", "paper", "AAA", "dividend", "2026-01-06", 2.0, "fx")
        self.assertEqual(self.store.receivable_balance("paper"), 10.0)
        self.assertEqual(self.store.cash_balance("paper"), cash)
        with self.assertRaises(StateError):
            self.store.pay_dividend("d", paid_at="2026-02-01")
        self.assertEqual(self.store.cash_balance("paper"), cash)

    def test_late_pay_date_never_double_books(self) -> None:
        cash = self.store.cash_balance("paper")
        self.store.record_corporate_action("d", "paper", "AAA", "dividend", "2026-01-06", 2.0, "fx")
        self.assertTrue(self.store.set_dividend_pay_date("d", "2026-01-12", "nasdaq"))
        self.assertFalse(self.store.set_dividend_pay_date("d", "2026-01-12", "nasdaq"))
        with self.assertRaises(StateError):
            self.store.set_dividend_pay_date("d", "2026-01-13", "other")
        self.store.pay_dividend("d", paid_at="2026-01-12")
        self.assertEqual(self.store.cash_balance("paper"), cash + 10.0)
        self.assertEqual(self.store.receivable_balance("paper"), 0.0)

    def test_resolution_satisfies_gate_but_filing_stays_unready(self) -> None:
        cycle = self.store.create_cycle("paper", "q", "strategy-a")
        self.store.register_expected_filings(cycle, ["k1", "k2"], filing_metadata={
            k: {"company_id": k, "security_id": k, "expected_period": "2026-09-30",
                "filing_type": "10-Q", "source": "t"} for k in ("k1", "k2")})
        self.store.set_filing_result(cycle, "k1", received=True, facts_ready=True, valid=True,
                                     filed_at="2026-10-01T12:00:00Z", first_seen_at="2026-10-01T13:00:00Z",
                                     source="t")
        self.assertEqual(self.store.filing_status(cycle), FilingStatus.WAITING_FILING)
        with self.assertRaises(Exception):
            self.store.resolve_filing(cycle, "k2", {})
        self.store.resolve_filing(cycle, "k2", {"evidence": [{"form": "25"}]})
        self.store.resolve_filing(cycle, "k2", {"evidence": [{"form": "25"}]})
        self.assertEqual(self.store.filing_status(cycle), FilingStatus.READY)
        status = self.store.query("SELECT status FROM expected_filings WHERE filing_key='k2'")[0]["status"]
        self.assertNotEqual(status, "READY")

    @unittest.skipUnless(REAL_DB.exists(), "real database not present")
    def test_migration_of_a_copy_of_the_real_database_keeps_history(self) -> None:
        copy = self.dir / "real.sqlite3"
        shutil.copy(REAL_DB, copy)
        con = sqlite3.connect(copy)
        before = con.execute("SELECT portfolio_id, created_at FROM portfolios ORDER BY 1").fetchall()
        version = con.execute("PRAGMA user_version").fetchone()[0]
        con.close()
        result = PaperStore.migrate(copy, "real", self.dir / "bk")
        self.assertEqual(result["to_version"], 6)
        if version < 6:
            self.assertTrue(Path(result["backup"]).exists())
        con = sqlite3.connect(copy)
        after = con.execute("SELECT portfolio_id, created_at FROM portfolios ORDER BY 1").fetchall()
        con.close()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
