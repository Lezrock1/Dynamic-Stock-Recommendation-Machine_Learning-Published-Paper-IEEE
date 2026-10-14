import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.benchmark import BenchmarkDataError, BenchmarkSeries


class BenchmarkTests(unittest.TestCase):
    def test_total_return_series_uses_previous_value_and_normalizes_nav(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
            pd.DataFrame({
                "date": dates,
                "total_return_index": [100.0, 110.0, 99.0],
            }).to_csv(root / "benchmark.csv", index=False)
            metadata = {
                "type": "sp500_gross_total_return",
                "source": "Synthetic fixture",
                "index_identity": "S&P 500 Gross Total Return (USD)",
                "currency": "USD",
                "dividend_treatment": "gross_dividends_reinvested_before_withholding",
                "start_date": dates[0],
                "end_date": dates[-1],
                "retrieved_at": "2025-01-07T00:00:00Z",
                "verified": True,
                "verification_method": "Synthetic unit-test fixture",
                "evidence_uri": "test://synthetic",
            }
            (root / "benchmark.json").write_text(json.dumps(metadata), encoding="utf-8")
            benchmark = BenchmarkSeries.from_files(
                root / "benchmark.csv",
                root / "benchmark.json",
                expected_trading_dates=dates,
            )
            returns = benchmark.returns()
            np.testing.assert_allclose(returns.iloc[1:].to_numpy(), [0.1, -0.1])
            self.assertTrue(pd.isna(returns.iloc[0]))
            nav = benchmark.nav(1_000, dates)
            np.testing.assert_allclose(nav.to_numpy(), [1_000, 1_100, 990])

    def test_expected_benchmark_day_cannot_disappear(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = ["2025-01-02", "2025-01-06"]
            pd.DataFrame({
                "date": dates,
                "total_return_index": [100.0, 101.0],
            }).to_csv(root / "benchmark.csv", index=False)
            metadata = {
                "type": "sp500_gross_total_return",
                "source": "Synthetic fixture",
                "index_identity": "S&P 500 Gross Total Return (USD)",
                "currency": "USD",
                "dividend_treatment": "gross_dividends_reinvested_before_withholding",
                "start_date": dates[0],
                "end_date": dates[-1],
                "retrieved_at": "2025-01-07T00:00:00Z",
                "verified": True,
                "verification_method": "Synthetic unit-test fixture",
                "evidence_uri": "test://synthetic",
            }
            (root / "benchmark.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkDataError, "missing 1 expected trading dates"):
                BenchmarkSeries.from_files(
                    root / "benchmark.csv",
                    root / "benchmark.json",
                    expected_trading_dates=["2025-01-02", "2025-01-03", "2025-01-06"],
                )


if __name__ == "__main__":
    unittest.main()
