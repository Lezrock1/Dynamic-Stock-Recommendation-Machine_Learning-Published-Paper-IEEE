import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pipeline.paper_dashboard import dashboard_snapshot
from pipeline.paper_runner import (
    CorporateAction, LockBusyError, PaperRunner, RunnerConfig, exclusive_lock,
)
from pipeline.paper_store import PaperStore
from pipeline.paper_synthetic import STRATEGIES, synthetic_adapters

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
KEYS = [f"co{i}-2025Q4" for i in range(20)]
TABLES = ("simulated_fills", "cash_ledger", "position_ledger", "signal_snapshots",
          "execution_batches", "simulated_orders", "daily_marks", "corporate_actions")


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class RunnerFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = Path(__file__).parent / f".paper-runner-test-{uuid.uuid4().hex}.sqlite3"
        self.now = utc("2026-01-05T16:00:00")
        self.adapters = synthetic_adapters(SYMBOLS)
        store = PaperStore.create(self.database)
        for name in STRATEGIES:
            store.create_portfolio(name, name, name, 1_000_000.0, created_at="2026-01-05T15:00:00+00:00")
            self.cycle_id = store.create_cycle(
                name, "2025Q4", name, started_at="2026-01-05T15:00:00+00:00",
                deadline_at="2026-02-15T00:00:00+00:00")
            store.register_expected_filings(
                self.cycle_id,
                KEYS,
                filing_metadata={k: {"company_id": k.split("-")[0], "expected_period": "2025Q4",
                                     "filing_type": "10-K"} for k in KEYS})

    def tearDown(self) -> None:
        for suffix in ("", "-wal", "-shm", ".lock"):
            Path(f"{self.database}{suffix}").unlink(missing_ok=True)

    def runner(self, config: RunnerConfig | None = None) -> PaperRunner:
        return PaperRunner(PaperStore(self.database), self.adapters, config, clock=lambda: self.now)

    def counts(self) -> dict[str, int]:
        store = PaperStore(self.database)
        return {t: store.query(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] for t in TABLES}

    def test_last_report_to_dashboard_and_idempotent_restart(self) -> None:
        runner = self.runner()
        for key in KEYS[:-1]:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        runner.tick()
        self.assertEqual(self.adapters.signals.calls, [])
        self.assertEqual(self.counts()["signal_snapshots"], 0)

        self.adapters.filings.arrive(KEYS[-1], "2026-01-06T12:00:00+00:00")
        self.now = utc("2026-01-06T15:00:00")
        runner.tick()
        self.assertEqual(self.adapters.signals.calls, ["2026-01-06T15:00:00+00:00"])
        counts = self.counts()
        self.assertEqual(counts["execution_batches"], 3)
        self.assertEqual(counts["signal_snapshots"], 3 * len(SYMBOLS))
        self.assertEqual(counts["simulated_fills"], 0)
        store = PaperStore(self.database)
        self.assertEqual({r["scheduled_for"] for r in store.query("SELECT scheduled_for FROM execution_schedule")},
                         {"2026-01-07"})
        timelines = store.query("SELECT * FROM cycle_timeline")
        self.assertEqual(len(timelines), 3)
        self.assertEqual({(t["published_at"], t["first_seen_at"], t["snapshot_as_of"], t["model_completed_at"])
                          for t in timelines},
                         {("2026-01-06T12:00:00+00:00", "2026-01-06T15:00:00+00:00",
                           "2026-01-06T15:00:00+00:00", "2026-01-06T15:00:00+00:00")})

        self.now = utc("2026-01-07T22:00:00")
        runner.tick()
        self.assertEqual(self.counts()["simulated_fills"], 3 * len(SYMBOLS))
        self.now = utc("2026-01-08T22:00:00")
        runner.tick()
        snapshot = dashboard_snapshot(self.database)
        for portfolio in snapshot["portfolios"]:
            self.assertTrue(portfolio["latest_mark"]["marked_at"].startswith("2026-01-08"))
            self.assertGreater(portfolio["latest_mark"]["positions_value"], 0)
            self.assertEqual(len(portfolio["positions"]), len(SYMBOLS))
        self.assertEqual({c["status"] for c in snapshot["cycles"]}, {"EXECUTED"})
        self.assertEqual(snapshot["historical_backtest_validation"]["status"], "NOT_VALIDATED")
        self.assertEqual(snapshot["live_paper_depot"]["data_origins"], ["SYNTHETIC_DEMO"])

        before = self.counts()
        restarted = self.runner()
        restarted.tick()
        self.assertEqual(self.counts(), before)
        self.assertEqual(self.adapters.signals.calls, ["2026-01-06T15:00:00+00:00"])
        self.now = utc("2026-01-09T22:00:00")
        restarted.tick()
        after = self.counts()
        self.assertEqual(after["daily_marks"], before["daily_marks"] + 3)
        for table in TABLES:
            if table != "daily_marks":
                self.assertEqual(after[table], before[table], table)

    def test_sector_with_different_information_date_is_rejected(self) -> None:
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        original = self.adapters.signals.predict_as_of

        def bad(as_of):
            result = original(as_of)
            return type(result)(result.signal_as_of, {**result.sector_as_of, "sector_0": "2025-12-31T00:00:00+00:00"},
                                result.data_snapshot, result.forecasts, result.target_weights)

        self.adapters.signals.predict_as_of = bad
        self.now = utc("2026-01-06T15:00:00")
        self.runner().tick()
        self.assertEqual(self.counts()["signal_snapshots"], 0)
        self.assertEqual(self.counts()["execution_batches"], 0)
        self.assertTrue(PaperStore(self.database).query("SELECT 1 FROM job_errors"))

    def test_execution_waits_for_session_open_after_model_completion(self) -> None:
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.now = utc("2026-01-06T13:00:00")  # 08:00 New York, before the open
        self.runner().tick()
        scheduled = {r["scheduled_for"] for r in PaperStore(self.database).query(
            "SELECT scheduled_for FROM execution_schedule")}
        self.assertEqual(scheduled, {"2026-01-06"})

    def test_future_or_naive_filing_time_is_invalid(self) -> None:
        for key in KEYS[:-2]:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.adapters.filings.arrive(KEYS[-2], "2026-01-05T12:00:00")
        self.adapters.filings.arrive(KEYS[-1], "2026-01-09T12:00:00+00:00")
        self.runner().tick()
        rows = PaperStore(self.database).query(
            "SELECT DISTINCT status FROM expected_filings WHERE filing_key IN (?, ?)", (KEYS[-2], KEYS[-1]))
        self.assertEqual({r["status"] for r in rows}, {"INVALID_DATA"})
        self.assertEqual(self.adapters.signals.calls, [])

    def test_real_adapters_are_refused_on_non_real_store(self) -> None:
        from dataclasses import replace
        from pipeline.paper_store import ValidationError
        with self.assertRaises(ValidationError):
            PaperRunner(PaperStore(self.database), replace(self.adapters, environment="real"))

    def test_deadline_blocks_without_lowering_completeness(self) -> None:
        for key in KEYS[:-1]:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.now = utc("2026-02-16T00:00:00")
        self.runner().tick()
        rows = PaperStore(self.database).query("SELECT status, filing_status FROM reporting_cycles")
        self.assertEqual({(r["status"], r["filing_status"]) for r in rows}, {("BLOCKED", "WAITING_FILING")})
        self.assertEqual(self.adapters.signals.calls, [])

    def test_failed_step_backs_off_and_resumes_without_duplicates(self) -> None:
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        original = self.adapters.market.close
        self.adapters.market.close = lambda s, d: None
        self.now = utc("2026-01-06T15:00:00")
        runner = self.runner()
        runner.tick()
        self.assertEqual(self.counts()["execution_batches"], 0)
        self.adapters.market.close = original
        runner.tick()  # still inside the backoff window
        self.assertEqual(self.counts()["execution_batches"], 0)
        self.now = utc("2026-01-06T15:05:00")
        runner.tick()
        self.assertEqual(self.counts()["execution_batches"], 3)
        self.assertEqual(self.adapters.signals.calls, ["2026-01-06T15:00:00+00:00"])  # model ran once; the retry reuses the stored result

    def test_dividend_and_split_are_booked_once(self) -> None:
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.adapters.market.actions = {
            "2026-01-08": [CorporateAction("AAA-div", "AAA", "dividend", 0.5, "synthetic", "2026-01-09")],
            "2026-01-09": [CorporateAction("AAA-split", "AAA", "split", 2.0, "synthetic")],
        }
        self.now = utc("2026-01-06T15:00:00")
        self.runner().tick()
        self.now = utc("2026-01-09T22:00:00")
        self.runner().tick()
        before = self.counts()
        self.runner().tick()
        self.assertEqual(self.counts(), before)
        store = PaperStore(self.database)
        self.assertEqual(len(store.query("SELECT 1 FROM dividend_receivables WHERE status = 'PAID'")), 3)
        self.assertEqual(before["corporate_actions"], 6)

    def test_lock_prevents_concurrent_schedulers(self) -> None:
        with exclusive_lock(self.database):
            with self.assertRaises(LockBusyError):
                self.runner().tick()


if __name__ == "__main__":
    unittest.main()
