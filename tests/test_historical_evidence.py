import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.historical_evidence import acceptance_map, evidence_for_period, fact_catalog, filings, period_expectation
from pipeline.spy_primary_comparison import reconstruct
from pipeline.filing_identity_evidence import cover_identity
from pipeline.historical_universe_sources import parse_constituents
from pipeline.market_source_evidence import parse_chart, collect as collect_market
from pipeline.market_calendar import nyse_sessions
from pipeline.reconcile_historical_evidence import match_original
from pipeline.primary_index_reconciliation import changes
from pipeline.dashboard_view import load_source_progress


def entry(value, accession, start=None, end="2026-06-30"):
    result = {"val": value, "accn": accession, "end": end, "form": "10-Q", "filed": "2026-08-01"}
    if start:
        result["start"] = start
    return result


class HistoricalEvidenceTests(unittest.TestCase):
    def test_market_coverage_summary_uses_completed_bars_not_incomplete_source_bar(self):
        today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
        prior = nyse_sessions(today - pd.Timedelta(days=14), today - pd.Timedelta(days=1))[-1]
        timestamps = [int((day.tz_localize("UTC") + pd.Timedelta(hours=15)).timestamp())
                      for day in (prior, today)]
        chart = {"chart": {"result": [{"meta": {"symbol": "TEST", "currency": "USD"},
                                     "timestamp": timestamps,
                                     "indicators": {"quote": [{"close": [100, None]}]}}]}}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "chart.json"
            source.write_text(json.dumps(chart))
            output = Path(temp) / "output"
            class Store:
                def __init__(self, directory):
                    directory.mkdir()
                def fetch(self, name, url):
                    return source
            with patch("pipeline.market_source_evidence.EvidenceStore", Store):
                collect_market(output, ["TEST"])
            row = json.loads((output / "security_market_coverage.json").read_text())[0]
            self.assertEqual(row["source_observations_including_incomplete"], 2)
            self.assertEqual(row["observations"], 1)
            self.assertEqual(row["null_closes"], 0)
            self.assertEqual(row["last_date"], prior.date().isoformat())

    def test_stale_quarters_never_form_current_ttm_value(self):
        catalog = {field: {} for field in ("Revenue", "CFO", "EPS_Basic", "OperatingIncome",
                                          "DepreciationAmortization")}
        catalog["Revenue"] = {date: {"value": 10, "components": []}
                              for date in ("2014-03-31", "2014-06-30", "2014-09-30", "2014-12-31")}
        result = evidence_for_period(catalog, "2026-06-30")
        self.assertIsNone(result["period_values"]["Revenue"])
        self.assertIsNone(result["ttm"]["Revenue"]["value"])
        self.assertEqual(result["ttm"]["Revenue"]["status"], "MISSING_CONSECUTIVE_QUARTERS")

    def test_primary_index_change_keeps_index_and_effective_date_distinct(self):
        def row(index):
            return (f'<tr><td>June 29, 2026</td><td>{index}</td><td>Addition</td>'
                    '<td>Honeywell Aerospace</td><td>HONA</td><td>Industrials</td></tr>')
        parsed = changes(('<table>' + row('S&amp;P 100') + row('S&amp;P 500') + '</table>').encode())
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["effective_date"], "2026-06-29")
        self.assertEqual(parsed[0]["sector"], "Industrials")

    def test_primary_sector_must_not_be_guessed_from_subindustry(self):
        document = (b'<table><tr><td>June 29, 2026</td><td>S&amp;P 500</td><td>Addition</td>'
                    b'<td>Company</td><td>TEST</td><td>Industrial Conglomerates</td></tr></table>')
        with self.assertRaisesRegex(ValueError, "Unsupported primary index change"):
            changes(document)

    def test_source_progress_reports_integrity_errors_and_rejects_outside_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "state/research/acquisition"
            directory.mkdir(parents=True)
            summary = directory / "summary.json"
            summary.write_text('{}')
            pointer = directory / "latest.json"
            pointer.write_text(json.dumps({"path": str(summary.relative_to(root)), "sha256": "wrong"}))
            self.assertIn("hash mismatch", load_source_progress(root)["error"])
            pointer.write_text(json.dumps({"path": str(summary.relative_to(root)),
                                           "sha256": hashlib.sha256(summary.read_bytes()).hexdigest()}))
            self.assertIn("schema invalid", load_source_progress(root)["error"])
            pointer.write_text(json.dumps({"path": "../outside.json", "sha256": "wrong"}))
            self.assertIn("outside acquisition", load_source_progress(root)["error"])

    def test_numeric_original_of_other_registrant_does_not_prove_issuer_value(self):
        atom = {"accession": "abc", "namespace": "us-gaap", "tag": "Assets", "unit": "USD",
                "period_start": None, "period_end": "2026-06-30", "value": 12.0}
        original = {"accession": "abc", "tag": "us-gaap:Assets", "unit": "iso4217:USD",
                    "period_start": None, "period_end": "2026-06-30", "value": 12.0,
                    "dimensions": [], "typed_dimensions": [], "context_id": "c1"}
        self.assertEqual(match_original(atom, [original], {"c1": "0000000002"}, 1)["matches"], [])
        self.assertEqual(match_original(atom, [original], {"c1": "0000000001"}, 1)["status"],
                         "UNDIMENSIONED_ORIGINAL_VALUE_AND_ENTITY_MATCH")

    def test_symbol_display_quotes_and_exchange_suffix_do_not_change_identity(self):
        for symbol in ('\u201cTEST\u201d', 'TEST (NYSE)'):
            body = ('<xbrli:context id="c"><xbrli:identifier>1</xbrli:identifier></xbrli:context>'
                    f'<ix:nonnumeric name="dei:TradingSymbol" contextref="c">{symbol}</ix:nonnumeric>'
                    '<ix:nonnumeric name="dei:Security12bTitle" contextref="c">Registered Shares</ix:nonnumeric>')
            self.assertEqual(cover_identity(body.encode(), {}, "TEST", 1)["status"],
                             "ORIGINAL_COVER_TICKER_ENTITY_MATCH")

    def test_cover_debt_ticker_is_not_stock_and_exchange_duplicates_are_preserved(self):
        body = '<html><body>'
        for context, title, member in (
            ("nyse", "Comm<span>on</span> Stock", "us-gaap:CommonStockMember"),
            ("texas", "Common Stock", "us-gaap:CommonStockMember"),
            ("debt", "Common Stock Linked Notes due 2028", "issuer:DebtMember"),
        ):
            body += (f'<xbrli:context id="{context}"><xbrli:entity>'
                     '<xbrli:identifier>0000000001</xbrli:identifier>'
                     f'<xbrldi:explicitmember dimension="us-gaap:StatementClassOfStockAxis">{member}</xbrldi:explicitmember>'
                     f'</xbrli:entity></xbrli:context>'
                     f'<ix:nonnumeric name="dei:TradingSymbol" contextref="{context}">TEST</ix:nonnumeric>'
                     f'<ix:nonnumeric name="dei:Security12bTitle" contextref="{context}">{title}</ix:nonnumeric>')
        result = cover_identity((body + '</body></html>').encode(), {}, "TEST", 1)
        self.assertEqual(result["status"], "ORIGINAL_COVER_TICKER_ENTITY_MATCH")
        self.assertEqual(len(result["equity_matches"]), 2)
        self.assertEqual(len(result["matches"]), 3)

    def test_subindustry_mislabeled_sector_is_rejected_not_guessed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "constituents.csv"
            path.write_text("Symbol,Name,Sector\nA,A company,Industrial Conglomerates\n")
            with self.assertRaisesRegex(ValueError, "Unknown dated sector"):
                parse_constituents(path)

    def test_stock_chart_does_not_certify_actions_and_keeps_unknown_pay_dates(self):
        timestamp = int(pd.Timestamp("2026-09-01T13:30:00Z").timestamp())
        chart = {"chart": {"result": [{
            "meta": {"symbol": "TEST", "currency": "USD"}, "timestamp": [timestamp],
            "indicators": {"quote": [{"close": [100.0]}]},
            "events": {"dividends": {"a": {"date": timestamp, "amount": 1.0}}},
        }]}}
        frame, actions, summary = parse_chart(chart, "TEST")
        self.assertEqual(len(frame), 1)
        self.assertIsNone(actions[0]["pay_date"])
        self.assertEqual(summary["source_review_status"], "ACQUIRED_NOT_RAW_AS_TRADED_OR_ACTION_COVERAGE_CERTIFIED")

    def test_exact_version_no_future_comparatives_or_filed_date_fallback(self):
        doc = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
            entry(100, "original"), entry(999, "restated"), entry(777, "unknown"),
        ]}}}}}
        accepted = {"original": pd.Timestamp("2026-07-31T20:01:00Z"),
                    "restated": pd.Timestamp("2026-09-01T12:00:00Z")}
        catalog, counts = fact_catalog(doc, accepted, "2026-08-31T20:00:00Z", 1, "source-hash")
        value = catalog["Assets"]["2026-06-30"]
        self.assertEqual(value["value"], 100)
        self.assertEqual(value["components"][0]["accession"], "original")
        self.assertEqual(value["components"][0]["raw_entry"]["val"], 100)
        self.assertEqual(counts["later_versions_excluded"], 1)
        self.assertEqual(counts["unknown_acceptance_excluded"], 1)

    def test_additive_lineage_preserves_both_components_and_no_eps_subtraction(self):
        values = [entry(250, "six", "2026-01-01"), entry(100, "three", "2026-01-01", "2026-03-31")]
        doc = {"facts": {"us-gaap": {
            "NetIncomeLoss": {"units": {"USD": values}},
            "EarningsPerShareBasic": {"units": {"USD/shares": values}},
        }}}
        accepted = {"six": pd.Timestamp("2026-07-31T20:01:00Z"),
                    "three": pd.Timestamp("2026-05-01T20:01:00Z")}
        catalog, _ = fact_catalog(doc, accepted, "2026-08-31T20:00:00Z", 1, "hash")
        derived = catalog["NetIncome"]["2026-06-30"]
        self.assertEqual(derived["value"], 150)
        self.assertEqual({p["accession"] for p in derived["components"]}, {"six", "three"})
        self.assertNotIn("2026-06-30", catalog["EPS_Basic"])

    def test_inconsistent_same_accession_value_is_unknown_not_last_wins(self):
        doc = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [entry(100, "a"), entry(101, "a")]}}}}}
        catalog, counts = fact_catalog(doc, {"a": pd.Timestamp("2026-07-31T20:00:00Z")},
                                       "2026-08-31T20:00:00Z", 1, "hash")
        self.assertNotIn("2026-06-30", catalog["Assets"])
        self.assertEqual(counts["ambiguous_values_excluded"], 1)

    def test_annual_and_off_grid_actual_period_not_fiscal_quarter_guess(self):
        rows = [{"form": "10-K", "reportDate": "2026-05-31"}]
        result = period_expectation(rows, "2026Q2")
        self.assertEqual(result["filing_form"], "10-K")
        self.assertEqual(result["period_end"], "2026-05-31")
        rows.append({"form": "10-Q", "reportDate": "2026-07-05"})
        self.assertEqual(period_expectation(rows, "2026Q2")["status"], "MISSING_OR_AMBIGUOUS_PERIOD")

    def test_missing_acceptance_not_fabricated_and_missing_parallel_arrays_retained(self):
        rows = filings({"accessionNumber": ["a", "b"], "form": ["10-K", "10-Q"]})
        self.assertEqual(len(rows), 2)
        self.assertEqual(acceptance_map(rows), {})
        with self.assertRaisesRegex(ValueError, "lacks timezone"):
            acceptance_map([{"accessionNumber": "a", "acceptanceDateTime": "2026-08-01"}])

    def test_real_return_arithmetic_no_adjusted_close_double_dividend(self):
        nav = pd.Series([100.0, 99.0, 100.0], index=pd.to_datetime(["2026-09-16", "2026-09-17", "2026-09-18"]))
        events = pd.DataFrame({"EX-DATE": pd.to_datetime(["2026-09-17"]), "cash_distribution": [1.0]})
        points, review = reconstruct(nav, events, "2026-09-16")
        self.assertAlmostEqual(points["value"].iloc[1], 100)
        self.assertAlmostEqual(points["value"].iloc[-1], 100 * 100 / 99)
        self.assertTrue(review["not_adjusted_close"])
        self.assertTrue(review["not_exchange_execution_price"])

    def test_primary_nav_gap_never_forward_filled(self):
        nav = pd.Series([100.0, 100.0], index=pd.to_datetime(["2026-09-16", "2026-09-18"]))
        events = pd.DataFrame({"EX-DATE": pd.to_datetime([]), "cash_distribution": []})
        with self.assertRaisesRegex(ValueError, "session gaps"):
            reconstruct(nav, events, "2026-09-16")


if __name__ == "__main__":
    unittest.main()
