import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pipeline.market_calendar import (  # noqa: E402
    first_available_session_after_filing,
    is_nyse_session,
    session_close_time,
)
from pipeline.ratios import _next_rebalance_date  # noqa: E402
from pipeline.sec_edgar import _best_by_key  # noqa: E402
import ml_model  # noqa: E402


class PointInTimeTests(unittest.TestCase):
    def test_date_only_sec_filing_is_available_no_earlier_than_next_session(self):
        self.assertEqual(
            first_available_session_after_filing("2025-01-02"),
            pd.Timestamp("2025-01-03"),
        )
        self.assertEqual(
            first_available_session_after_filing("2025-01-03"),
            pd.Timestamp("2025-01-06"),
        )
        self.assertFalse(is_nyse_session("2025-01-01"))

    def test_holiday_shifted_quarter_trade_date_and_prior_close_cutoff(self):
        # March 1, 2025 is Saturday, so the execution is Monday March 3.
        self.assertEqual(_next_rebalance_date(pd.Timestamp("2025-02-28")), pd.Timestamp("2025-03-03"))
        # Facts first available on March 3 cannot enter a signal frozen Friday
        # after the close for Monday's execution.
        self.assertEqual(_next_rebalance_date(pd.Timestamp("2025-03-03")), pd.Timestamp("2025-06-02"))

    def test_later_restatement_does_not_change_earlier_information_set(self):
        facts = [
            {"period": "2024-Q1", "filed": "2024-05-01", "accn": "a", "val": 10},
            {"period": "2024-Q1", "filed": "2024-08-01", "accn": "b", "val": 12},
        ]
        before_restatement = _best_by_key(
            facts, lambda item: item["period"], as_of="2024-06-01"
        )
        after_restatement = _best_by_key(
            facts, lambda item: item["period"], as_of="2024-08-05"
        )
        self.assertEqual(before_restatement["2024-Q1"]["val"], 10)
        self.assertEqual(after_restatement["2024-Q1"]["val"], 12)

    def test_quarter_cv_keeps_quarters_together_and_purges_boundary_label(self):
        dates = pd.date_range("2022-03-01", periods=12, freq="QS")
        row_dates = pd.Series(np.repeat(dates, 2))
        splits = ml_model.quarter_group_splits(row_dates, n_splits=3)
        for train_indices, validation_indices in splits:
            train_dates = set(row_dates.iloc[train_indices])
            validation_dates = set(row_dates.iloc[validation_indices])
            self.assertEqual(len(train_dates), len(train_indices) // 2)
            self.assertEqual(len(validation_dates), len(validation_indices) // 2)
            self.assertLess(max(train_dates), min(validation_dates))
            prior_to_validation = max(date for date in dates if date < min(validation_dates))
            self.assertNotIn(prior_to_validation, train_dates)
        for dates_train, _ in splits:
            self.assertEqual(len(dates_train) % 2, 0)

    def test_market_calendar_exposes_shortened_session_close(self):
        close = session_close_time("2025-11-28")
        self.assertEqual(close.tz_convert("America/New_York").strftime("%H:%M"), "13:00")


if __name__ == "__main__":
    unittest.main()
