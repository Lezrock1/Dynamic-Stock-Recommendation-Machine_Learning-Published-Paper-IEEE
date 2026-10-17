import json
import unittest
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from pipeline import paper_live
from pipeline.paper_dashboard import dashboard_snapshot
from pipeline.paper_runner import (
    CorporateAction, FilingObservation, LockBusyError, OrderRequest, OrderSide, PaperRunner, RunnerConfig,
    exclusive_lock,
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
            mark_values = json.loads(portfolio["latest_mark"]["position_values_json"])
            self.assertAlmostEqual(
                sum(item["market_value"] for item in mark_values.values()),
                portfolio["latest_mark"]["positions_value"],
            )
        equal_portfolio = next(p for p in snapshot["portfolios"] if p["portfolio_id"] == "equal_weight_v1")
        self.assertEqual(
            snapshot["weight_comparison"]["actual_marked_at"]["equal_weight_v1"],
            equal_portfolio["latest_mark"]["marked_at"],
        )
        marked_values = json.loads(equal_portfolio["latest_mark"]["position_values_json"])
        actual_rows = {row["symbol"]: row["actual_equal_weight_v1"]
                       for row in snapshot["weight_comparison"]["rows"]}
        for symbol, item in marked_values.items():
            self.assertAlmostEqual(actual_rows[symbol], item["market_value"] / equal_portfolio["latest_mark"]["nav"])
        self.assertEqual({c["status"] for c in snapshot["cycles"]}, {"EXECUTED"})
        self.assertEqual(snapshot["historical_backtest_validation"]["status"], "NOT_VALIDATED")
        self.assertEqual(snapshot["live_paper_depot"]["data_origins"], ["SYNTHETIC_DEMO"])
        self.assertEqual(
            snapshot["nav_comparison"]["actual_common_start"],
            equal_portfolio["daily_marks"][0]["marked_at"],
        )
        self.assertTrue(snapshot["nav_comparison"]["series"])
        self.assertEqual(
            snapshot["portfolio_accounts"][0]["marked_at"],
            snapshot["portfolios"][0]["latest_mark"]["marked_at"],
        )

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

    def test_signal_snapshot_waits_until_last_facts_become_ready(self) -> None:
        delayed_key = KEYS[-1]
        self.adapters.filings.arrive(delayed_key, "2026-01-06T12:00:00+00:00")
        for key in KEYS[:-1]:
            self.adapters.filings.arrive(key, "2026-01-06T12:00:00+00:00")
        ready = {"delayed": False}

        def observe(expected):
            return {
                item["filing_key"]: FilingObservation(
                    True, item["filing_key"] != delayed_key or ready["delayed"], True,
                    "2026-01-06T12:00:00+00:00", "fixture",
                )
                for item in expected if item["filing_key"] in self.adapters.filings.arrived
            }

        self.adapters.filings.observe = observe
        self.now = utc("2026-01-06T15:00:00")
        runner = self.runner()
        cycle_id = str(runner._cycles("2025Q4")[0]["cycle_id"])
        first_tick = runner.tick()
        self.assertEqual(self.adapters.signals.calls, [])
        delayed = PaperStore(self.database).query(
            "SELECT filed_at, first_seen_at, facts_ready_at FROM expected_filings "
            "WHERE cycle_id = ? AND filing_key = ?", (cycle_id, delayed_key),
        )[0]
        self.assertEqual(delayed["filed_at"], "2026-01-06T12:00:00+00:00", first_tick)
        self.assertEqual(delayed["first_seen_at"], "2026-01-06T15:00:00+00:00")
        self.assertIsNone(delayed["facts_ready_at"])

        ready["delayed"] = True
        self.now = utc("2026-01-08T15:00:00")
        runner.tick()
        timeline = PaperStore(self.database).query(
            "SELECT snapshot_as_of FROM cycle_timeline WHERE cycle_id = ?", (cycle_id,)
        )[0]["snapshot_as_of"]
        self.assertEqual(timeline, "2026-01-08T15:00:00+00:00")
        self.assertEqual(self.adapters.signals.calls, ["2026-01-08T15:00:00+00:00"])
        delayed = PaperStore(self.database).query(
            "SELECT facts_ready_at FROM expected_filings WHERE cycle_id = ? AND filing_key = ?",
            (cycle_id, delayed_key),
        )[0]
        self.assertEqual(delayed["facts_ready_at"], "2026-01-08T15:00:00+00:00")

    def test_legacy_ready_filing_without_facts_timestamp_is_rechecked_not_backfilled(self) -> None:
        key = KEYS[-1]
        self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        store = PaperStore(self.database)
        cycles = store.query("SELECT cycle_id FROM reporting_cycles")
        for cycle in cycles:
            store.set_filing_result(
                cycle["cycle_id"],
                key,
                received=True,
                facts_ready=True,
                valid=True,
                filed_at="2026-01-05T12:00:00+00:00",
                first_seen_at="2026-01-05T13:00:00+00:00",
                source="synthetic:legacy-fixture",
                facts_ready_at="2026-01-05T13:00:00+00:00",
                updated_at="2026-01-05T13:00:00+00:00",
            )
        with store._connection() as connection:
            connection.execute(
                "UPDATE expected_filings SET facts_ready_at = NULL WHERE filing_key = ?", (key,)
            )

        result = self.runner().check_filings("2025Q4")

        self.assertEqual(result["status"], "SUCCEEDED")
        rows = store.query(
            "SELECT filed_at, first_seen_at, facts_ready_at FROM expected_filings WHERE filing_key = ?",
            (key,),
        )
        self.assertEqual(len(rows), len(cycles))
        self.assertEqual(
            {(r["filed_at"], r["first_seen_at"], r["facts_ready_at"]) for r in rows},
            {("2026-01-05T12:00:00+00:00", "2026-01-05T13:00:00+00:00",
              "2026-01-05T16:00:00+00:00")},
        )

    def test_resolved_special_case_needs_no_fabricated_filing_timestamp(self) -> None:
        for key in KEYS[:-1]:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.adapters.filings.resolutions = lambda: {
            KEYS[-1]: {
                "status": "CORPORATE_ACTION_RESOLVED",
                "evidence": [{"source": "SEC Form 25", "effective_date": "2026-01-06"}],
            }
        }
        self.now = utc("2026-01-06T15:00:00")
        runner = self.runner()
        runner.tick()

        cycles = PaperStore(self.database).query(
            "SELECT cycle_id FROM reporting_cycles WHERE cycle_key = '2025Q4'"
        )
        cycle_id = str(cycles[0]["cycle_id"])
        special = PaperStore(self.database).query(
            "SELECT filed_at, first_seen_at FROM expected_filings WHERE cycle_id = ? AND filing_key = ?",
            (cycle_id, KEYS[-1]),
        )[0]
        self.assertIsNone(special["filed_at"])
        self.assertIsNone(special["first_seen_at"])
        timeline = PaperStore(self.database).query(
            "SELECT published_at, first_seen_at, snapshot_as_of FROM cycle_timeline WHERE cycle_id = ?",
            (cycle_id,),
        )[0]
        self.assertEqual(timeline["published_at"], "2026-01-05T12:00:00+00:00")
        self.assertEqual(timeline["first_seen_at"], "2026-01-06T15:00:00+00:00")
        self.assertEqual(timeline["snapshot_as_of"], "2026-01-06T15:00:00+00:00")
        self.assertEqual(self.counts()["signal_snapshots"], 3 * len(SYMBOLS))

    def test_current_reports_persist_one_shared_cycle_signal(self) -> None:
        snapshot = "2026-12-15T20:00:00+00:00"
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-12-15T19:00:00+00:00")

        config = {
            "model": {"min_feature_coverage": 0.5, "first_trade_date_index": 0,
                      "testing_windows": 1, "max_rolling_window_index": 10, "selection_quantile": 0.5},
            "required_facts": [],
            "report_matching_tolerance_days": 10,
        }
        provider = paper_live.ModelSignalProvider(config)
        provider.prepare_cycle = lambda _: setattr(
            provider, "expected_periods", {1: pd.Timestamp("2026-09-30"), 2: pd.Timestamp("2026-09-30")}
        )
        features = list(paper_live.FEATURES)
        live_rows = []
        for ticker, cik, available, tradedate in (
            ("AAA", 1, "2026-10-20T20:00:00Z", "2026-12-01"),
            ("BBB", 2, "2026-12-10T20:00:00Z", "2027-03-01"),
        ):
            row = {feature: 1.0 for feature in features}
            row.update({
                "tic": ticker, "cik": cik, "gsector": 10, "is_current": True,
                "datadate": pd.Timestamp("2026-09-30"), "available_at": available,
                "tradedate": pd.Timestamp(tradedate), "y_return": np.nan,
                "signal_as_of": pd.Timestamp("2026-12-14T21:00:00Z"),
                "signal_time": pd.Timestamp("2026-12-14T21:00:00Z"),
                "label_known_at": pd.Timestamp("2026-12-14T21:00:00Z"),
                "accession": f"acc-{ticker}", "published_at": available,
            })
            live_rows.append(row)
        history = pd.DataFrame(live_rows)
        historical = pd.DataFrame([{
            "tic": "OLD", "tradedate": pd.Timestamp("2016-12-01"), "y_return": 0.01,
            **{feature: 1.0 for feature in features},
        }])
        price_history = pd.DataFrame(
            {"close": [100.0], "adj_close": [100.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-12-14")]),
        )
        weights = pd.Series({"AAA": 0.05, "BBB": 0.05})
        universe = pd.DataFrame([
            {"tic": "AAA", "cik": 1, "gsector": 10, "is_current": True},
            {"tic": "BBB", "cik": 2, "gsector": 10, "is_current": True},
        ])

        def fake_model_run(data, feature_names, *args, **kwargs):
            pending = pd.Timestamp(args[5][0])
            self.assertEqual(
                set(data.loc[data["tradedate"] == pending, "tic"]), {"AAA", "BBB"}
            )
            predictions = pd.DataFrame([[0.02, 0.03]], index=[pending], columns=["AAA", "BBB"])
            model_names = pd.DataFrame({"model_name": ["fixture"]}, index=[pending])
            return (None, None, None, None, predictions, model_names)

        with (
            mock.patch.object(paper_live.ua, "members_as_of", return_value=(universe, {})),
            mock.patch.object(provider, "_live_rows", return_value=(history, {"AAA": price_history,
                                                                             "BBB": price_history}, {})),
            mock.patch.object(paper_live, "_historical_sector", return_value=historical),
            mock.patch.object(paper_live, "sector_feature_set", return_value=["X1_REVGH"]),
            mock.patch("ml_model.run_4model", side_effect=fake_model_run),
            mock.patch.object(paper_live, "capped_equal_weights", return_value=weights),
            mock.patch.object(paper_live, "optimize_weights", return_value=weights),
        ):
            self.adapters = replace(self.adapters, signals=provider)
            self.now = utc(snapshot)
            runner = self.runner()
            runner.tick()

        records = PaperStore(self.database).query(
            "SELECT DISTINCT payload_json FROM signal_snapshots WHERE cycle_id = ?", (self.cycle_id,)
        )
        self.assertTrue(records)
        persisted = [__import__("json").loads(str(row["payload_json"])) for row in records]
        self.assertEqual({item["forecast"]["expected_return"] for item in persisted
                         if item["forecast"] is not None}, {0.02, 0.03})
        timeline = PaperStore(self.database).query(
            "SELECT snapshot_as_of FROM cycle_timeline WHERE cycle_id = ?", (self.cycle_id,)
        )[0]["snapshot_as_of"]
        self.assertEqual(timeline, snapshot)

    def test_dividend_pay_date_is_processed_after_all_positions_are_sold(self) -> None:
        store = PaperStore(self.database)
        store.create_portfolio("single", "Single", "single", 10_000.0)
        buy_cycle = store.create_cycle("single", "buy", "single")
        store.rebalance(
            buy_cycle, "single", "buy-batch", {"AAA": 1.0},
            [OrderRequest("buy-aaa", "AAA", OrderSide.BUY, 10)],
        )
        buy_order = store.orders_for_batch("buy-batch")[0]
        store.record_fill("buy-fill", buy_order.order_id, 10, 100.0,
                          transaction_cost_bps_per_side=0.0)
        store.record_corporate_action(
            "div-aaa", "single", "AAA", "dividend", "2026-01-08", 1.0, "fixture",
            pay_date="2026-01-12", created_at="2026-01-08T21:00:00+00:00",
        )
        self.assertEqual(store.receivable_balance("single"), 10.0)

        sell_cycle = store.create_cycle("single", "sell", "single")
        store.rebalance(
            sell_cycle, "single", "sell-batch", {"AAA": 0.0},
            [OrderRequest("sell-aaa", "AAA", OrderSide.SELL, 10)],
        )
        sell_order = store.orders_for_batch("sell-batch")[0]
        store.record_fill("sell-fill", sell_order.order_id, 10, 100.0,
                          transaction_cost_bps_per_side=0.0)
        self.assertEqual(store.position_balance("single", "AAA"), 0.0)
        cash_before = store.cash_balance("single")

        runner = PaperRunner(
            store, synthetic_adapters(SYMBOLS), clock=lambda: utc("2026-01-12T22:00:00")
        )
        runner._apply_actions("2026-01-12")

        self.assertEqual(store.receivable_balance("single"), 0.0)
        self.assertEqual(store.cash_balance("single"), cash_before + 10.0)
        self.assertEqual(store.query("SELECT status FROM dividend_receivables")[0]["status"], "PAID")

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

    def test_tick_runs_injected_maintenance_and_persists_refresh_attempt_once_per_day(self) -> None:
        refresh_calls = []
        prepare_calls = []
        runner = PaperRunner(
            PaperStore(self.database),
            self.adapters,
            clock=lambda: self.now,
            data_refresher=lambda: (refresh_calls.append(self.now.date().isoformat()) or {"status": "FRESH"}),
            cycle_preparer=lambda: (prepare_calls.append(self.now.date().isoformat())
                                    or {"status": "WAITING_FOR_PREDECESSOR"}),
        )

        first = runner.tick()
        second = runner.tick()

        self.assertEqual(first["data_refresh"]["job_status"], "SUCCEEDED")
        self.assertEqual(first["data_refresh"]["result"]["status"], "FRESH")
        self.assertEqual(second["data_refresh"]["job_status"], "DONE")
        self.assertEqual(refresh_calls, ["2026-01-05"])
        self.assertEqual(prepare_calls, ["2026-01-05", "2026-01-05"])
        refresh_jobs = PaperStore(self.database).query(
            "SELECT run_id FROM job_runs WHERE job_type = 'data_refresh' AND status = 'SUCCEEDED'"
        )
        self.assertEqual(len(refresh_jobs), 1)


if __name__ == "__main__":
    unittest.main()
