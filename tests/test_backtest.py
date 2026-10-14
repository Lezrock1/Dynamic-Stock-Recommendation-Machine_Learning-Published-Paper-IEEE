import unittest

import pandas as pd

from pipeline.backtest import (
    BacktestConfig,
    BacktestDataError,
    CorporateAction,
    simulate_portfolio,
    total_return_metrics,
)


def prices(rows):
    return pd.DataFrame(
        rows,
        index=pd.date_range("2025-01-02", periods=len(next(iter(rows.values()))), freq="B"),
        dtype=float,
    )


class BacktestTests(unittest.TestCase):
    def test_costs_persist_and_initial_cost_is_measured_from_initial_capital(self):
        result = simulate_portfolio(
            prices({"AAA": [100, 100, 100]}),
            {"2025-01-02": {"AAA": 1.0}},
            config=BacktestConfig(
                initial_capital=1_000,
                fractional_shares=True,
                max_weight_per_asset=None,
            ),
        )
        self.assertLess(result.nav.iloc[0], 1_000)
        self.assertAlmostEqual(result.nav.iloc[1], result.nav.iloc[0])
        self.assertAlmostEqual(
            result.nav.iloc[0],
            1_000 - result.trades["transaction_cost"].sum(),
            places=7,
        )
        metrics = total_return_metrics(result.nav, initial_capital=1_000)
        self.assertAlmostEqual(metrics["total_return"], result.nav.iloc[-1] / 1_000 - 1)

    def test_weights_drift_and_rebalance_uses_real_positions(self):
        result = simulate_portfolio(
            prices({"AAA": [100, 200, 200], "BBB": [100, 100, 100]}),
            {
                "2025-01-02": {"AAA": 0.5, "BBB": 0.5},
                "2025-01-06": {"AAA": 0.5, "BBB": 0.5},
            },
            config=BacktestConfig(
                initial_capital=2_000,
                transaction_cost_bps_per_side=0,
                fractional_shares=True,
                max_weight_per_asset=None,
            ),
        )
        self.assertAlmostEqual(result.positions.loc["2025-01-03", "AAA"], 10.0)
        self.assertAlmostEqual(result.positions.loc["2025-01-03", "BBB"], 10.0)
        rebalance = result.trades[result.trades["date"] == pd.Timestamp("2025-01-06")]
        self.assertAlmostEqual(rebalance["notional"].sum(), 1_000.0)
        self.assertEqual(set(rebalance["side"]), {"BUY", "SELL"})

    def test_split_preserves_nav(self):
        result = simulate_portfolio(
            prices({"AAA": [100, 50]}),
            {"2025-01-02": {"AAA": 0.5}},
            config=BacktestConfig(
                initial_capital=1_000,
                transaction_cost_bps_per_side=0,
                fractional_shares=True,
                max_weight_per_asset=None,
            ),
            corporate_actions=[
                CorporateAction("split-1", "AAA", "2025-01-03", "split", 2.0)
            ],
        )
        self.assertAlmostEqual(result.nav.iloc[0], result.nav.iloc[1])
        self.assertAlmostEqual(result.positions["AAA"].iloc[1], 10.0)

    def test_dividend_receivable_prevents_ex_date_nav_drop_and_pays_once(self):
        result = simulate_portfolio(
            prices({"AAA": [100, 90, 90]}),
            {"2025-01-02": {"AAA": 1.0}},
            config=BacktestConfig(
                initial_capital=1_000,
                transaction_cost_bps_per_side=0,
                fractional_shares=True,
                max_weight_per_asset=None,
            ),
            corporate_actions=[
                CorporateAction(
                    "dividend-1", "AAA", "2025-01-03", "dividend", 10.0, "2025-01-06"
                )
            ],
        )
        self.assertAlmostEqual(result.nav.iloc[1], 1_000.0)
        self.assertAlmostEqual(result.dividend_receivables.iloc[1], 100.0)
        self.assertAlmostEqual(result.dividend_receivables.iloc[2], 0.0)
        self.assertAlmostEqual(result.nav.iloc[2], 1_000.0)

    def test_whole_share_rounding_leaves_cash_and_no_borrowing(self):
        result = simulate_portfolio(
            prices({"AAA": [300, 300]}),
            {"2025-01-02": {"AAA": 1.0}},
            config=BacktestConfig(initial_capital=1_000, max_weight_per_asset=None),
        )
        self.assertEqual(result.positions.loc["2025-01-02", "AAA"], 3)
        self.assertGreaterEqual(result.cash.iloc[0], 0)
        self.assertAlmostEqual(result.nav.iloc[0], result.cash.iloc[0] + 900)

    def test_held_security_missing_price_is_a_data_error(self):
        with self.assertRaisesRegex(BacktestDataError, "missing/invalid valuation price"):
            simulate_portfolio(
                prices({"AAA": [100, float("nan")]}),
                {"2025-01-02": {"AAA": 0.5}},
                config=BacktestConfig(
                    initial_capital=1_000,
                    transaction_cost_bps_per_side=0,
                    fractional_shares=True,
                    max_weight_per_asset=None,
                ),
            )


if __name__ == "__main__":
    unittest.main()
