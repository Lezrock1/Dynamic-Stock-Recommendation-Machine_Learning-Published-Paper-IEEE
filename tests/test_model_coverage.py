"""Regression tests for the model-coverage fixes (synthetic inputs only, no network)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import model_coverage as mc  # noqa: E402
import paper_live as pl  # noqa: E402
import ratios as ra  # noqa: E402
import sec_edgar as se  # noqa: E402

CONFIG = json.loads((ROOT / "config" / "real_paper.json").read_text())


class TagMapping(unittest.TestCase):
    def test_revenue_tags_cover_observed_filers(self):
        for tag in ("RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"):
            self.assertIn(tag, se.DURATION_TAGS["Revenue"])


class RatiosWithoutIndustrialInputs(unittest.TestCase):
    def test_missing_revenue_and_shares_stay_nan_not_zero(self):
        index = pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"])
        fund = pd.DataFrame({"NetIncome": 10.0, "Assets": 1000.0, "StockholdersEquity": 100.0,
                             "available_at": (index + pd.Timedelta(days=30)).strftime("%Y-%m-%dT12:00:00Z")},
                            index=index)
        days = pd.bdate_range("2025-01-01", "2026-12-31")
        prices = pd.DataFrame({"close": 50.0, "adj_close": 50.0}, index=days)
        out = ra.compute_ratios(fund, prices, "BANK")
        self.assertTrue(out["X1_REVGH"].isna().all())
        self.assertTrue(out["X7_NPM"].isna().all())
        self.assertTrue(out["X6_PS"].isna().all())
        self.assertTrue((out["X3_ROA"] == 0.01).all())


class DiagnosticOptimizerFailures(unittest.TestCase):
    def test_insufficient_universe_is_reported_without_fabricated_optimized_weights(self):
        tickers = [f"T{i}" for i in range(6)]
        selected = pd.Series(np.linspace(0.01, 0.06, len(tickers)), index=tickers)
        days = pd.bdate_range("2025-01-01", periods=253)
        values = np.arange(len(days), dtype=float)
        prices = pd.DataFrame({
            ticker: 100 + values * (index + 1) / 100
            for index, ticker in enumerate(tickers)
        }, index=days)

        weights, failures = pl._portfolio_weights(
            selected, prices, days[-1], allow_optimizer_failures=True
        )

        self.assertEqual(set(weights), {"equal_weight_v1"})
        self.assertAlmostEqual(float(weights["equal_weight_v1"].sum()), 0.30)
        self.assertEqual(set(failures), {"mean_var_v1", "min_var_v1"})
        self.assertTrue(all("requires at least 20 usable names" in reason
                            for reason in failures.values()))
        with self.assertRaisesRegex(pl.OptimizationError, "requires at least 20 usable names"):
            pl._portfolio_weights(selected, prices, days[-1])


class ReadinessRule(unittest.TestCase):
    def test_bank_sector_does_not_require_revenue_but_visa_does(self):
        self.assertNotIn("Revenue", pl.required_facts_for(CONFIG, 40, 35527))
        self.assertIn("Revenue", pl.required_facts_for(CONFIG, 40, 1403161))
        self.assertIn("Revenue", pl.required_facts_for(CONFIG, 45, 1))
        self.assertEqual(CONFIG["readiness_rule"]["version"], "3")

    def test_resolved_case_is_reported_separately_and_never_ready(self):
        manifest = {"expected": [
            {"company_id": "1437107", "security_id": "WBD", "filing_key": "1437107-2026-09-30-10-Q"},
            {"company_id": "1", "security_id": "AAA", "filing_key": "1-2026-09-30-10-Q"},
            {"company_id": "2", "security_id": "BBB", "filing_key": "2-2026-09-30-10-Q"}]}
        report = mc.readiness_breakdown(CONFIG, {"1437107": "WAITING_FILING", "1": "READY", "2": "WAITING_FILING"},
                                        manifest)
        self.assertEqual(report["received_ready"], 1)
        self.assertEqual(report["open"], 1)
        special = report["evidenced_special_cases_not_ready"]
        self.assertEqual([s["ticker"] for s in special], ["WBD"])
        self.assertFalse(special[0]["counts_as_ready"] or special[0]["buy_candidate"])

    def test_timeout_or_missing_facts_never_create_a_special_case(self):
        manifest = {"expected": [{"company_id": "9", "security_id": "ZZZ", "filing_key": "9-2026-09-30-10-Q"}]}
        report = mc.readiness_breakdown(CONFIG, {"9": "WAITING_FACTS"}, manifest)
        self.assertEqual(report["evidenced_special_cases_not_ready"], [])
        self.assertEqual(report["open"], 1)


class ManifestUniverse(unittest.TestCase):
    def test_prepare_cycle_uses_frozen_universe_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "X.json").write_text(json.dumps({"universe_as_of": "2026-09-30"}))
            provider = pl.ModelSignalProvider({**CONFIG, "manifest_dir": tmp})
            provider.prepare_cycle("X")
            self.assertEqual(provider.universe_as_of, pd.Timestamp("2026-09-30"))
            with self.assertRaises(pl.MissingDataError):
                provider.prepare_cycle("missing")


class DividendPayDate(unittest.TestCase):
    def setUp(self):
        self.market = pl.YahooMarket(CONFIG)
        self.rows = [{"symbol": "BRK.B", "dividend_Rate": 0.5, "payment_Date": "8/14/2026"},
                     {"symbol": "EARLY", "dividend_Rate": 1.0, "payment_Date": "8/01/2026"}]
        self.market._nasdaq_calendar = lambda day: self.rows

    def test_known_pay_date_and_symbol_normalisation(self):
        self.assertEqual(self.market.pay_date("BRK-B", "2026-08-07", 0.5), "2026-08-14")

    def test_unknown_or_inconsistent_pay_date_is_none(self):
        self.assertIsNone(self.market.pay_date("NONE", "2026-08-07", 0.5))
        self.assertIsNone(self.market.pay_date("BRK-B", "2026-08-07", 0.9))
        self.assertIsNone(self.market.pay_date("EARLY", "2026-08-07", 1.0))
        self.market._nasdaq_calendar = lambda day: None
        self.assertIsNone(self.market.pay_date("BRK-B", "2026-08-07", 0.5))


class ValuationRule(unittest.TestCase):
    def test_holdings_use_raw_close_and_benchmark_adjusted(self):
        history = pd.DataFrame({"close": [10.0], "adj_close": [9.0]}, index=pd.to_datetime(["2026-08-07"]))
        original = pl.pr.get_price_history
        pl.pr.get_price_history = lambda symbol: history
        try:
            market = pl.YahooMarket(CONFIG)
            self.assertEqual(market.close("AAA", "2026-08-07"), 10.0)
            self.assertEqual(market.benchmark("2026-08-07"), 9.0)
        finally:
            pl.pr.get_price_history = original
        self.assertEqual(CONFIG["benchmark"]["label"], "SPY-Vergleich")


class QuarterDerivation(unittest.TestCase):
    def test_q4_is_derived_despite_comparative_facts_with_later_fy_labels(self):
        def fact(start, end, val, fy, fp, form, filed):
            return {"start": start, "end": end, "val": val, "fy": fy, "fp": fp, "form": form, "filed": filed,
                    "accn": f"{form}-{filed}"}
        units = [
            fact("2024-01-01", "2024-03-31", 100, 2024, "Q1", "10-Q", "2024-05-01"),
            fact("2024-01-01", "2024-06-30", 220, 2024, "Q2", "10-Q", "2024-08-01"),
            fact("2024-04-01", "2024-06-30", 120, 2024, "Q2", "10-Q", "2024-08-01"),
            fact("2024-01-01", "2024-09-30", 360, 2024, "Q3", "10-Q", "2024-11-01"),
            fact("2024-01-01", "2024-12-31", 520, 2024, "FY", "10-K", "2025-02-20"),
            # the same facts repeat as comparatives under fy=2025 and must not corrupt the result
            fact("2024-01-01", "2024-03-31", 100, 2025, "Q1", "10-Q", "2025-05-01"),
            fact("2024-01-01", "2024-12-31", 520, 2025, "FY", "10-K", "2026-02-20"),
            fact("2023-01-01", "2023-12-31", 400, 2025, "FY", "10-K", "2026-02-20"),
        ]
        quarters = se._discrete_quarters(units)
        values = {k.date().isoformat(): v["val"] for k, v in quarters.items()}
        self.assertEqual(values["2024-03-31"], 100)
        self.assertEqual(values["2024-06-30"], 120)
        self.assertEqual(values["2024-09-30"], 140)
        self.assertEqual(values["2024-12-31"], 160)
        self.assertNotIn("2023-12-31", values)

    def test_non_additive_field_is_not_derived(self):
        def fact(start, end, val, form, filed):
            return {"start": start, "end": end, "val": val, "form": form, "filed": filed, "accn": f"{form}-{filed}"}
        units = [fact("2024-01-01", "2024-09-30", 2.27, "10-Q", "2024-11-01"),
                 fact("2024-01-01", "2024-12-31", 3.54, "10-K", "2025-02-20")]
        self.assertEqual(se._discrete_quarters(units, derive=False), {})
        self.assertEqual(se.ADDITIVE_FIELDS & {"EPS_Basic", "SharesOutstanding", "Assets"}, set())

    def test_real_stored_filings_do_not_leak_later_comparatives(self):
        path = Path(__file__).resolve().parent.parent / "cache" / "secfacts" / "1326160.json"
        if not path.exists():
            self.skipTest("stored DUK companyfacts not available")
        facts = json.loads(path.read_text())["facts"]["us-gaap"]
        units = facts["NetIncomeLoss"]["units"]["USD"]
        before = se._discrete_quarters(units, as_of="2024-01-15")
        after = se._discrete_quarters(units, as_of="2024-03-01")
        self.assertNotIn(pd.Timestamp("2023-12-31"), before)
        q4 = after[pd.Timestamp("2023-12-31")]
        self.assertEqual(q4["accn"].split("-")[1], "24")
        self.assertEqual(q4["filed"], "2024-02-23")
        # the value known on 2024-03-01 equals the one derived from filings available then, not later restatements
        self.assertEqual(se._discrete_quarters(units, as_of="2024-03-01")[pd.Timestamp("2023-12-31")]["val"], q4["val"])

    def test_real_eps_q4_is_not_derived_when_absent(self):
        path = Path(__file__).resolve().parent.parent / "cache" / "secfacts" / "1326160.json"
        if not path.exists():
            self.skipTest("stored DUK companyfacts not available")
        units = [e for e in json.loads(path.read_text())["facts"]["us-gaap"]["EarningsPerShareBasic"]["units"]["USD/shares"]
                 if not (e["start"] == "2023-10-01" and e["end"] == "2023-12-31")]
        quarters = se._discrete_quarters(units, derive=False)
        self.assertNotIn(pd.Timestamp("2023-12-31"), quarters)


class TagEvidence(unittest.TestCase):
    def test_annual_only_tag_has_no_quarterly_values(self):
        facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
            {"form": "10-K", "start": "2025-06-01", "end": "2026-05-31", "val": 1}]}}}}}
        evidence = mc._tag_evidence(facts)["NetIncome"]
        self.assertTrue(evidence["present"])
        self.assertEqual(evidence["tags"][0]["quarterly_values"], 0)


class PriceCacheRepair(unittest.TestCase):
    def test_stale_nan_close_cache_is_replaced_by_fresh_rows(self):
        import prices as prices_module
        days = pd.bdate_range("2026-01-05", periods=40)
        stale = pd.DataFrame({"adj_close": 9.0, "close": np.nan}, index=days)
        fresh_raw = pd.DataFrame({("Close", "AAA"): 10.0, ("Adj Close", "AAA"): 9.0}, index=days)
        fresh_raw.columns = pd.MultiIndex.from_tuples(fresh_raw.columns)
        with tempfile.TemporaryDirectory() as tmp:
            original = (prices_module.CACHE_DIR, prices_module.yf.download)
            prices_module.CACHE_DIR = Path(tmp)
            stale.to_csv(Path(tmp) / "AAA.csv")
            prices_module.yf.download = lambda *a, **k: fresh_raw
            try:
                history = prices_module.get_price_history("AAA")
            finally:
                prices_module.CACHE_DIR, prices_module.yf.download = original
        self.assertEqual(float(history["close"].notna().mean()), 1.0)
        self.assertEqual(float(history["close"].iloc[0]), 10.0)


if __name__ == "__main__":
    unittest.main()
