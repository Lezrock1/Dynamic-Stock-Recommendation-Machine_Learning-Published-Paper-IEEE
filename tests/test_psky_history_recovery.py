import copy
from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

from pipeline.market_calendar import nyse_sessions
from pipeline.psky_history_recovery import exact_levels, parse_chart


class PskyHistoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        sessions = nyse_sessions("2025-08-07", "2026-08-31")
        self.days = [day.date().isoformat() for day in sessions]
        self.payload = {"chart": {"error": None, "result": [{
            "meta": {
                "symbol": "SKYD", "currency": "USD", "dataGranularity": "1d",
                "instrumentType": "EQUITY",
            },
            "timestamp": [
                int(datetime.fromisoformat(day + "T09:30:00").replace(
                    tzinfo=ZoneInfo("America/New_York")
                ).timestamp()) for day in self.days
            ],
            "indicators": {
                "quote": [{"close": [10.0] * len(self.days)}],
                "adjclose": [{"adjclose": [8.0] * len(self.days)}],
            },
        }]}}

    def test_exact_calendar_and_quote_not_adjusted(self):
        prices, _ = parse_chart(self.payload)
        levels = exact_levels(prices)
        self.assertEqual(len(levels), 253)
        self.assertEqual(levels[0], ("2025-08-28", 10.0))
        self.assertEqual(levels[-1], ("2026-08-31", 10.0))

    def test_reject_silent_monthly_fallback(self):
        self.payload["chart"]["result"][0]["meta"]["dataGranularity"] = "1mo"
        with self.assertRaisesRegex(ValueError, "daily"):
            parse_chart(self.payload)

    def test_reject_wrong_reused_ticker(self):
        self.payload["chart"]["result"][0]["meta"]["symbol"] = "PARA"
        with self.assertRaisesRegex(ValueError, "symbol"):
            parse_chart(self.payload)

    def test_reject_missing_invalid_and_duplicate_data(self):
        for value in [None, float("nan"), float("inf"), 0, -1, True]:
            with self.subTest(value=value):
                payload = copy.deepcopy(self.payload)
                payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][30] = value
                with self.assertRaises(ValueError):
                    parse_chart(payload)
        self.payload["chart"]["result"][0]["timestamp"][30] = (
            self.payload["chart"]["result"][0]["timestamp"][29]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_chart(self.payload)

    def test_reject_split_events_without_adjustment_rule(self):
        self.payload["chart"]["result"][0]["events"] = {"splits": {"1": {"splitRatio": "2:1"}}}
        with self.assertRaisesRegex(ValueError, "split"):
            parse_chart(self.payload)

    def test_reject_missing_session_and_predecessor(self):
        prices, _ = parse_chart(self.payload)
        missing = dict(prices)
        del missing["2026-08-28"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            exact_levels(missing)
        prices["2025-08-06"] = 10
        with self.assertRaisesRegex(ValueError, "pre-listing"):
            exact_levels(prices)

    def test_reject_unequal_arrays(self):
        self.payload["chart"]["result"][0]["timestamp"].pop()
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            parse_chart(self.payload)


if __name__ == "__main__":
    unittest.main()
