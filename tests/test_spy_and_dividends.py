import unittest

from pipeline.paper_runner import CorporateAction
from pipeline.paper_store import PaperStore
from tests import test_paper_runner as base
from tests.test_paper_runner import KEYS, utc


class SpyAndLateDividend(base.RunnerFlowTests):
    def setUp(self) -> None:
        super().setUp()
        store = PaperStore(self.database)
        store.create_portfolio("spy_benchmark", "SPY", "spy_buy_hold_v1", 1_000_000.0,
                               created_at="2026-01-05T15:00:00+00:00")
        cid = store.create_cycle("spy_benchmark", "2025Q4", "spy_buy_hold_v1",
                                 started_at="2026-01-05T15:00:00+00:00",
                                 deadline_at="2026-02-15T00:00:00+00:00")
        store.register_expected_filings(cid, KEYS, filing_metadata={
            k: {"company_id": k.split("-")[0], "expected_period": "2025Q4", "filing_type": "10-K"}
            for k in KEYS})

    def test_spy_portfolio_buys_spy_with_same_schedule_and_receives_dividend_only_on_pay_date(self) -> None:
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        pay = {}
        self.adapters.market.pay_date = lambda symbol, ex, amount: pay.get((symbol, ex))
        self.adapters.market.actions = {
            "2026-01-08": [CorporateAction("SPY-div", "SPY", "dividend", 1.0, "synthetic", None, "2026-01-08")],
        }
        self.now = utc("2026-01-06T15:00:00")
        self.runner().tick()
        store = PaperStore(self.database)
        batches = store.query("SELECT b.batch_id, s.scheduled_for FROM execution_batches b "
                              "JOIN execution_schedule s ON s.batch_id = b.batch_id "
                              "JOIN reporting_cycles c ON c.cycle_id = b.cycle_id "
                              "WHERE c.portfolio_id IN ('spy_benchmark','equal_weight')")
        self.assertEqual(len({b["scheduled_for"] for b in batches}), 1)
        self.now = utc("2026-01-09T22:00:00")
        self.runner().tick()
        shares = store.position_balance("spy_benchmark", "SPY")
        self.assertGreater(shares, 0)
        self.assertAlmostEqual(store.receivable_balance("spy_benchmark"), shares)
        cash = store.cash_balance("spy_benchmark")
        pay[("SPY", "2026-01-08")] = "2026-01-09"
        self.now = utc("2026-01-12T22:00:00")
        self.runner().tick()
        self.assertAlmostEqual(store.cash_balance("spy_benchmark"), cash + shares)
        again = store.cash_balance("spy_benchmark")
        self.runner().tick()
        self.assertEqual(store.cash_balance("spy_benchmark"), again)

    def test_spy_depot_starts_with_ml_depots_and_pays_fees_only_on_its_own_fills(self) -> None:
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.now = utc("2026-01-06T15:00:00")
        self.runner().tick()
        self.now = utc("2026-01-12T22:00:00")
        self.runner().tick()
        store = PaperStore(self.database)
        fills = store.query(
            "SELECT p.portfolio_id, f.filled_at, f.quantity, f.price, f.transaction_cost, o.symbol FROM simulated_fills f "
            "JOIN simulated_orders o ON o.order_id = f.order_id JOIN execution_batches b ON b.batch_id = o.batch_id "
            "JOIN reporting_cycles c ON c.cycle_id = b.cycle_id JOIN portfolios p ON p.portfolio_id = c.portfolio_id")
        spy = [f for f in fills if f["portfolio_id"] == "spy_benchmark"]
        ml = [f for f in fills if f["portfolio_id"] != "spy_benchmark"]
        self.assertTrue(spy and ml)
        self.assertEqual({f["symbol"] for f in spy}, {"SPY"})
        self.assertEqual(len(spy), 1)  # buy-and-hold: one purchase, no re-balancing trades
        self.assertEqual({f["filled_at"] for f in spy}, {f["filled_at"] for f in ml}, ml[:2])
        notional = spy[0]["quantity"] * spy[0]["price"]
        bps = self.runner().config.transaction_cost_bps + self.runner().config.slippage_bps
        self.assertLessEqual(spy[0]["transaction_cost"], notional * bps / 10_000 * 1.0001 + 1e-9)

    def test_spy_buy_and_hold_has_no_followup_order_across_cycle_and_restart(self) -> None:
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.now = utc("2026-01-06T15:00:00")
        runner = self.runner()
        runner.tick()
        self.now = utc("2026-01-07T22:00:00")
        runner.tick()
        store = PaperStore(self.database)
        first_fills = store.query(
            "SELECT f.execution_id, f.transaction_cost FROM simulated_fills f "
            "JOIN simulated_orders o ON o.order_id = f.order_id "
            "JOIN execution_batches b ON b.batch_id = o.batch_id "
            "JOIN reporting_cycles c ON c.cycle_id = b.cycle_id "
            "WHERE c.portfolio_id = 'spy_benchmark'"
        )
        self.assertEqual(len(first_fills), 1)
        initial_cost = first_fills[0]["transaction_cost"]

        for portfolio_id in (*base.STRATEGIES, "spy_benchmark"):
            cycle_id = store.create_cycle(
                portfolio_id, "2026Q1", portfolio_id,
                started_at="2026-01-08T20:00:00+00:00",
            )
            store.register_expected_filings(cycle_id, KEYS, filing_metadata={
                key: {"company_id": key.split("-")[0], "expected_period": "2026Q1", "filing_type": "10-Q"}
                for key in KEYS
            })
        self.now = utc("2026-01-08T20:00:00")
        runner.tick()
        self.assertEqual(store.query(
            "SELECT COUNT(*) AS n FROM simulated_orders o "
            "JOIN execution_batches b ON b.batch_id = o.batch_id "
            "JOIN reporting_cycles c ON c.cycle_id = b.cycle_id "
            "WHERE c.portfolio_id = 'spy_benchmark'"
        )[0]["n"], 1)

        self.now = utc("2026-01-09T22:00:00")
        restarted = self.runner()
        restarted.tick()
        later_fills = store.query(
            "SELECT f.execution_id, f.transaction_cost FROM simulated_fills f "
            "JOIN simulated_orders o ON o.order_id = f.order_id "
            "JOIN execution_batches b ON b.batch_id = o.batch_id "
            "JOIN reporting_cycles c ON c.cycle_id = b.cycle_id "
            "WHERE c.portfolio_id = 'spy_benchmark'"
        )
        self.assertEqual(len(later_fills), 1)
        self.assertEqual(later_fills[0]["execution_id"], first_fills[0]["execution_id"])
        self.assertEqual(later_fills[0]["transaction_cost"], initial_cost)

    def test_dashboard_separates_report_states_and_compares_weights(self) -> None:
        from pipeline.paper_dashboard import dashboard_snapshot
        for key in KEYS[:-1]:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.now = utc("2026-01-06T15:00:00")
        self.runner().tick()
        before = dashboard_snapshot(self.database)
        self.assertEqual(before["report_status"]["open"], 1)
        self.assertFalse(before["report_status"]["gate_satisfied"])
        self.assertIn("Kein gültiges", before["weight_comparison"]["banner"])
        self.adapters.filings.arrive(KEYS[-1], "2026-01-06T15:30:00+00:00")
        self.now = utc("2026-01-07T15:00:00")
        self.runner().tick()
        self.now = utc("2026-01-12T22:00:00")
        self.runner().tick()
        after = dashboard_snapshot(self.database)
        self.assertEqual(after["report_status"]["open"], 0)
        comparison = after["weight_comparison"]
        self.assertFalse(comparison["diagnostic"])
        self.assertTrue(comparison["rows"])
        name = comparison["portfolios"][0]
        self.assertTrue(any(r[f"target_{name}"] for r in comparison["rows"]))
        self.assertTrue(any(r[f"actual_{name}"] for r in comparison["rows"]))

    def test_diagnostic_signal_cannot_create_weights(self) -> None:
        from pipeline.paper_runner import SignalResult
        from pipeline.paper_store import ValidationError
        runner = self.runner()
        snapshot = "2026-01-06T15:00:00+00:00"
        cycles = [{"strategy": "equal_weight"}]

        def result(enforced, kind_ok):
            return SignalResult(snapshot, {"sector10": snapshot}, {
                "eligibility_enforced": enforced,
                "forecast_records": [{"security_id": "AAA", "eligible_for_current_cycle": kind_ok}]},
                {}, {"equal_weight": {"AAA": 0.02}})
        with self.assertRaises(Exception):
            runner._validated_payload(result(False, True), snapshot, cycles)
        with self.assertRaises(Exception):
            runner._validated_payload(result(True, False), snapshot, cycles)
        runner._validated_payload(result(True, True), snapshot, cycles)

    def test_late_discovered_dividend_uses_pre_ex_date_holdings_after_sale(self) -> None:
        store = PaperStore(self.database)
        store.create_portfolio("late_dividend", "Late dividend", "late_dividend_v1", 5_000.0,
                               created_at="2026-01-05T15:00:00+00:00")
        buy_cycle = store.create_cycle("late_dividend", "buy", "late_dividend_v1")
        store.rebalance(
            buy_cycle, "late_dividend_v1", "buy-batch", {"AAA": 1.0},
            [base.OrderRequest("buy-aaa", "AAA", base.OrderSide.BUY, 10)],
        )
        buy = store.orders_for_batch("buy-batch")[0]
        store.record_fill("buy-fill", buy.order_id, 10, 100.0, filled_at="2026-01-07T21:00:00+00:00")

        sell_cycle = store.create_cycle("late_dividend", "sell", "late_dividend_v1")
        store.rebalance(
            sell_cycle, "late_dividend_v1", "sell-batch", {"AAA": 0.0},
            [base.OrderRequest("sell-aaa", "AAA", base.OrderSide.SELL, 10)],
        )
        sell = store.orders_for_batch("sell-batch")[0]
        store.record_fill("sell-fill", sell.order_id, 10, 101.0, filled_at="2026-01-09T21:00:00+00:00")
        self.assertEqual(store.position_balance("late_dividend", "AAA"), 0.0)

        self.adapters.market.actions = {
            "2026-01-12": [
                CorporateAction("late-div", "AAA", "dividend", 2.0, "synthetic",
                                "2026-01-12", "2026-01-08")
            ]
        }
        runner = self.runner()
        cash_before_dividend = store.cash_balance("late_dividend")
        runner._apply_actions("2026-01-12")

        self.assertEqual(store.receivable_balance("late_dividend"), 0.0)
        self.assertAlmostEqual(store.cash_balance("late_dividend"), cash_before_dividend + 20.0)
        self.assertEqual(store.query(
            "SELECT amount, status FROM dividend_receivables WHERE action_id = 'late-div:late_dividend'"
        ), [{"amount": 20.0, "status": "PAID"}])
        runner._apply_actions("2026-01-12")
        self.assertAlmostEqual(store.cash_balance("late_dividend"), cash_before_dividend + 20.0)

    def test_failed_calendar_keeps_known_claim(self) -> None:
        for key in KEYS:
            self.adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
        self.adapters.market.pay_date = lambda *a: None
        self.adapters.market.actions = {
            "2026-01-08": [CorporateAction("SPY-div", "SPY", "dividend", 1.0, "synthetic", None, "2026-01-08")]}
        self.now = utc("2026-01-06T15:00:00")
        self.runner().tick()
        self.now = utc("2026-01-20T22:00:00")
        self.runner().tick()
        store = PaperStore(self.database)
        self.assertGreater(store.receivable_balance("spy_benchmark"), 0)


for _name in [n for n in dir(base.RunnerFlowTests) if n.startswith("test_")]:
    if _name not in SpyAndLateDividend.__dict__:
        setattr(SpyAndLateDividend, _name, None)  # inherited scenarios are covered in test_paper_runner


if __name__ == "__main__":
    unittest.main()
