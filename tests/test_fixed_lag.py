from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.paper_fixed_lag import (
    clean_windows, continuation_cycles, research_weights, schedule,
    select_top_fraction, window_frames,
    optimization_inputs, optimization_definition,
)
from pipeline.market_calendar import nyse_sessions
from pipeline.portfolio_weights import OptimizationError
from pipeline.run_model import PAPER_FEATURES


def historical_panel(target: str = "2026Q2") -> pd.DataFrame:
    periods = pd.period_range(end=pd.Period(target, "Q") - 2, periods=44, freq="Q")
    return pd.DataFrame([{
        "cycle_key": str(p), "y_return": 0.01,
        "label_known_at": schedule(str(p + 1))["execution_at"],
        "feature_available_at": schedule(str(p))["information_cutoff"],
        "reconstruction_status": "VERIFIED", **{f: 1.0 for f in PAPER_FEATURES},
    } for p in periods])


class FixedLagTests(unittest.TestCase):
    def test_optimizer_inputs_share_annual_log_basis(self) -> None:
        days = nyse_sessions("2025-01-01", "2026-08-31")[-253:]
        log_returns = np.column_stack([np.sin(np.arange(252) / 7) * .02,
                                       np.cos(np.arange(252) / 11) * .01])
        levels = pd.DataFrame(np.exp(np.vstack([np.zeros(2), log_returns.cumsum(axis=0)])) * 100,
                              index=days, columns=["A", "B"])
        predicted = pd.Series([.10, -.02], index=["A", "B"])
        expected, covariance, risk_free = optimization_inputs(predicted, levels, "2026-08-31T20:00:00Z")
        np.testing.assert_allclose(expected, [.4, -.08])
        np.testing.assert_allclose(covariance, np.cov(log_returns, rowvar=False) * 252, atol=1e-12)
        self.assertAlmostEqual(risk_free, np.log1p(.015))
        simple_covariance = levels.pct_change(fill_method=None).iloc[1:].cov().to_numpy() * 252
        self.assertGreater(float(np.max(np.abs(covariance - simple_covariance))), 1e-6)
        self.assertIn("Not exact expected simple-return Sharpe",
                      optimization_definition()["portfolio_approximation"])

    def test_zero_covariance_cannot_manufacture_max_sharpe(self) -> None:
        dates = nyse_sessions("2025-01-01", "2026-08-31")[-253:]
        forecasts = pd.Series(.02, index=[f"T{i}" for i in range(25)])
        prices = pd.DataFrame(100., index=dates, columns=forecasts.index)
        weights, errors = research_weights(forecasts, prices, "2026-08-31T20:00:00Z")
        self.assertIn("equal_weight", weights)
        self.assertIn("min_var", weights)
        self.assertNotIn("mean_var", weights)
        self.assertIn("undefined", errors["mean_var"])

    def test_fixed_dates_and_prior_close(self) -> None:
        q2 = schedule("2026Q2")
        self.assertEqual(q2["execution_date"], "2026-09-01")
        self.assertEqual(q2["information_cutoff"], "2026-08-31T20:00:00+00:00")
        self.assertEqual(schedule("2026Q3")["execution_date"], "2026-12-01")
        self.assertEqual(schedule("2024Q4")["execution_date"], "2025-03-03")
        self.assertEqual(schedule("1990Q2")["execution_date"], "1990-09-04")
        self.assertLess(pd.Timestamp(q2["information_cutoff"]), pd.Timestamp(q2["execution_at"]))

    def test_nullspace_portfolio_cannot_manufacture_max_sharpe(self) -> None:
        dates = nyse_sessions("2025-01-01", "2026-08-31")[-253:]
        daily = np.sin(np.arange(252) / 7) * .01
        path = np.r_[0., np.cumsum(daily)]
        prices = pd.DataFrame({
            f"T{i}": 100 * np.exp(path if i < 10 else -path) for i in range(20)
        }, index=dates)
        forecasts = pd.Series(.02, index=prices.columns)
        weights, errors = research_weights(forecasts, prices, "2026-08-31T20:00:00Z")
        self.assertIn("min_var", weights)
        self.assertNotIn("mean_var", weights)
        self.assertIn("portfolio log variance", errors["mean_var"])

    def test_37_decisions_and_original_handoff(self) -> None:
        cycles = continuation_cycles()
        self.assertEqual(len(cycles), 37)
        self.assertEqual(schedule(cycles[0])["execution_date"], "2017-09-01")
        self.assertEqual(cycles[-1], "2026Q2")

    def test_four_matured_validation_and_maximum_40_train(self) -> None:
        train, validation = window_frames(historical_panel(), "2026Q2")
        self.assertEqual(train["cycle_key"].nunique(), 40)
        self.assertEqual(validation["cycle_key"].nunique(), 4)
        self.assertEqual(validation["cycle_key"].iloc[-1], "2025Q4")

    def test_incomplete_validation_and_future_feature_rejected(self) -> None:
        panel = historical_panel()
        panel.loc[panel.index[-1], "feature_available_at"] = "2026-09-01T20:00:00Z"
        with self.assertRaisesRegex(ValueError, "Latest completed label quarter"):
            window_frames(panel, "2026Q2")
        panel = historical_panel()
        panel.loc[panel.index[-2], "label_known_at"] = "2026-09-01T20:00:00Z"
        with self.assertRaisesRegex(ValueError, "four required validation"):
            window_frames(panel, "2026Q2")

    def test_missing_training_quarter_not_hidden(self) -> None:
        panel = historical_panel().drop(index=5)
        with self.assertRaisesRegex(ValueError, "Gap in training"):
            window_frames(panel, "2026Q2")

    def test_top_twenty_ties_do_not_select_entire_sector(self) -> None:
        forecasts = pd.Series(0.01, index=[f"T{i:03}" for i in reversed(range(100))])
        chosen, info = select_top_fraction(forecasts)
        self.assertEqual(list(chosen.index), [f"T{i:03}" for i in range(20)])
        self.assertEqual(info["original_quantile_ge_selected"], 100)
        self.assertTrue(info["constant_prediction"])
        tiny, _ = select_top_fraction(pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"]))
        self.assertEqual(list(tiny.index), ["C"])

    def test_training_only_cleaning_and_no_median_imputation(self) -> None:
        train, validation = window_frames(historical_panel(), "2026Q2")
        trade = validation.iloc[:1].copy()
        train.loc[train.index[:3], PAPER_FEATURES[0]] = np.nan
        validation[PAPER_FEATURES[1]] = np.nan
        with self.assertRaisesRegex(ValueError, "validation has only 0"):
            clean_windows(train, validation, trade)
        validation[PAPER_FEATURES[1]] = 5.0
        features, clean_train, _, _ = clean_windows(train, validation, trade)
        self.assertNotIn(PAPER_FEATURES[0], features)
        self.assertEqual(len(clean_train), len(train))

    def test_optimizer_fails_infeasible_without_fallback(self) -> None:
        with self.assertRaisesRegex(OptimizationError, "at least 20"):
            research_weights(pd.Series([0.01], index=["A"]), pd.DataFrame(), "2026-08-31T20:00:00Z")

    def test_optimizer_requires_full_year_and_does_not_fill_gaps(self) -> None:
        predicted = pd.Series(np.arange(20) / 1000, index=[f"T{i}" for i in range(20)])
        prices = pd.DataFrame(100.0, index=pd.bdate_range("2025-09-01", periods=253),
                              columns=predicted.index)
        prices.iloc[1, 0] = np.nan
        with self.assertRaisesRegex(OptimizationError, "253 complete"):
            research_weights(predicted, prices, "2026-08-31T20:00:00Z")

    def test_missing_quarter_after_complete_case_filter_is_rejected(self) -> None:
        train, validation = window_frames(historical_panel(), "2026Q2")
        train.loc[train.index[20], PAPER_FEATURES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "quarter gap after complete-case"):
            clean_windows(train, validation, validation.iloc[:1])

    def test_weekday_covariance_is_not_a_nyse_session_year(self) -> None:
        predicted = pd.Series(0.01, index=[f"T{i}" for i in range(20)])
        history = pd.DataFrame(100.0, index=pd.bdate_range(end="2026-08-31", periods=253),
                               columns=predicted.index)
        with self.assertRaisesRegex(OptimizationError, "actual NYSE sessions"):
            research_weights(predicted, history, "2026-08-31T20:00:00Z")


if __name__ == "__main__":
    unittest.main()
