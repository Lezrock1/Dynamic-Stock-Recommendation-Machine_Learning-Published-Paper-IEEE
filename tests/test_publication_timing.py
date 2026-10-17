from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pipeline import publication_timing as timing
from pipeline.dashboard_view import HTML


class PublicationTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.now = datetime(2026, 10, 16, tzinfo=timezone.utc)
        self.write_json("state/expected_filings_q2_2026Q2/2026Q2.json", {
            "period_rule": "calendar-quarter-v2",
            "universe_source": {"source": "test"},
            "expected": [{
                "company_id": "1", "expected_period": "2026-06-30",
                "filing_type": "10-Q", "security_id": "AAA",
                "issuer_tickers": ["AAA", "AAA-B"],
            }],
            "securities": [{"ticker": "AAA", "cik": 1}, {"ticker": "AAA-B", "cik": 1},
                           {"ticker": "BBB", "cik": 2}, {"ticker": "UNKNOWN", "cik": None}],
        })
        self.write_csv("cache/sp500_universe.csv", ["tic", "cik"],
                       [{"tic": "AAA", "cik": 1}, {"tic": "AAA-B", "cik": 1},
                        {"tic": "BBB", "cik": 2}])
        self.write_csv("cache/sp500_historical_snapshots.csv", ["date", "tickers"], [
            {"date": "2023-01-01", "tickers": "AAA,AAA-B,BBB,UNKNOWN"},
        ])
        self.submissions("1", [
            ("10-Q", "2026-06-30", "2026-07-08T01:30:00Z", "first"),
            ("10-Q", "2026-06-30", "2026-07-20T12:00:00Z", "second"),
            ("10-Q/A", "2026-06-30", "2026-07-02T12:00:00Z", "amendment"),
            ("8-K", "2026-07-01", "2026-07-01T12:00:00Z", "earnings-candidate"),
        ])

    def write_json(self, name: str, doc: dict) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding="utf-8")

    def write_csv(self, name: str, fields: list[str], rows: list[dict]) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def submissions(self, cik: str, rows: list[tuple]) -> None:
        self.write_json(f"cache/secsubmissions/{cik}.json", {
            "filings": {"recent": {
                "form": [r[0] for r in rows], "reportDate": [r[1] for r in rows],
                "acceptanceDateTime": [r[2] for r in rows],
                "accessionNumber": [r[3] for r in rows],
                "primaryDocument": ["report.htm"] * len(rows),
            }},
        })

    def evidence(self, period: str = "2026-06-30") -> None:
        self.write_json(str(timing.EVIDENCE_PATH), {
            "schema_version": 1,
            "earnings": [{
                "cycle_key": "2026Q2", "company_id": "1", "period_end": period,
                "published_on": "2026-07-01", "first_publication_verified": True,
                "observed_at": "2026-10-15T12:00:00Z", "source_url": "https://example.org/release",
            }],
            "market_caps": [{
                "cycle_key": "2026Q2", "company_id": "1", "as_of": "2026-06-30",
                "scope": "issuer_all_share_classes", "value_usd": 100_000_000_000,
                "observed_at": "2026-10-15T12:00:00Z", "source_url": "https://example.org/cap",
            }],
        })

    def test_buckets_include_negative_boundaries_and_long_delays(self) -> None:
        for days, expected in [(-20, 0), (-1, 0), (0, 1), (6, 1), (7, 2),
                               (90, 13), (91, 14), (600, 14)]:
            with self.subTest(days=days):
                self.assertEqual(timing.bucket_index(days), expected)

    def test_issuer_dedup_first_original_and_new_york_date(self) -> None:
        with patch("socket.create_connection", side_effect=AssertionError("network forbidden")) as network:
            report = timing.build_report(self.root, self.now)
            network.assert_not_called()
        quarter = report["quarters"]["2026Q2"]
        self.assertEqual(quarter["sec"]["expected"], 3)
        self.assertEqual(quarter["sec"]["known_dates"], 1)
        self.assertEqual(quarter["sec"]["missing_dates"], 2)
        self.assertEqual(sum(quarter["sec"]["counts"]), 1)
        row = next(r for r in quarter["rows"] if r["company_id"] == "1")
        self.assertEqual(row["sec_days"], 7)
        self.assertEqual(row["accession"], "first")
        self.assertEqual(row["tickers"], ["AAA", "AAA-B"])
        self.assertEqual(quarter["earnings"]["known_dates"], 0)
        self.assertEqual(quarter["sec"]["known_caps"], 0)
        self.assertIsNone(row["market_cap_usd"])

    def test_separate_earnings_and_sourced_cap_no_renormalization(self) -> None:
        self.evidence()
        quarter = timing.build_report(self.root, self.now)["quarters"]["2026Q2"]
        self.assertEqual(quarter["earnings"]["known_dates"], 1)
        self.assertEqual(quarter["earnings"]["counts"][timing.bucket_index(1)], 1)
        self.assertEqual(quarter["sec"]["counts"][timing.bucket_index(7)], 1)
        self.assertEqual(quarter["sec"]["market_cap_usd"][timing.bucket_index(7)], 100e9)
        self.assertEqual(quarter["sec"]["joint_observations"], 1)
        self.assertFalse(quarter["sec"]["full_cap_coverage"])
        self.assertEqual(quarter["sec"]["known_cap_usd"], 100e9)

    def test_wrong_period_earnings_not_used(self) -> None:
        self.evidence("2026-03-31")
        quarter = timing.build_report(self.root, self.now)["quarters"]["2026Q2"]
        self.assertEqual(quarter["earnings"]["known_dates"], 0)
        self.assertEqual(quarter["rows"][0]["earnings_status"], "EVIDENCE_PERIOD_MISMATCH")

    def test_ambiguous_fiscal_period_and_future_filing_not_counted(self) -> None:
        self.submissions("2", [
            ("10-Q", "2026-04-30", "2026-05-15T12:00:00Z", "a"),
            ("10-Q", "2026-05-31", "2026-06-15T12:00:00Z", "b"),
        ])
        doc = timing.build_report(self.root, self.now)
        row = next(r for r in doc["quarters"]["2026Q2"]["rows"] if r["company_id"] == "2")
        self.assertEqual(row["sec_status"], "AMBIGUOUS_REPORT_PERIOD")
        self.submissions("2", [("10-Q", "2026-06-30", "2026-11-01T12:00:00Z", "future")])
        doc = timing.build_report(self.root, self.now)
        self.assertEqual(doc["quarters"]["2026Q2"]["sec"]["known_dates"], 1)

    def test_history_counts_company_quarters_and_excludes_selected(self) -> None:
        doc = timing.build_report(self.root, self.now, 4)
        history = doc["history"]["2026Q2"]
        self.assertEqual(history["cycles"], ["2025Q2", "2025Q3", "2025Q4", "2026Q1"])
        self.assertEqual(history["sec"]["expected"], 12)
        self.assertTrue(history["sec"]["historical"])

    def test_fiscal_early_release_retained_and_annual_report_included(self) -> None:
        self.submissions("2", [
            ("10-Q", "2026-04-30", "2026-05-15T12:00:00Z", "early"),
            ("10-K", "2025-12-31", "2026-02-01T12:00:00Z", "annual"),
        ])
        doc = timing.build_report(self.root, self.now)
        row = next(r for r in doc["quarters"]["2026Q2"]["rows"] if r["company_id"] == "2")
        self.assertEqual(row["sec_days"], -46)
        self.assertEqual(doc["quarters"]["2026Q2"]["sec"]["counts"][0], 1)
        annual = next(r for r in doc["quarters"]["2025Q4"]["rows"] if r["company_id"] == "2")
        self.assertEqual(annual["form"], "10-K")
        self.assertEqual(annual["sec_days"], 32)

    def test_unverified_earnings_and_future_observation_fail_explicitly(self) -> None:
        self.evidence()
        path = self.root / timing.EVIDENCE_PATH
        evidence = json.loads(path.read_text())
        evidence["earnings"][0]["first_publication_verified"] = False
        path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "first publication is not verified"):
            timing.build_report(self.root, self.now)
        evidence["earnings"][0]["first_publication_verified"] = True
        evidence["earnings"][0]["observed_at"] = "2026-11-01T12:00:00Z"
        path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "observation time is in the future"):
            timing.build_report(self.root, self.now)

    def test_invalid_cap_evidence_fails_explicitly(self) -> None:
        self.evidence()
        path = self.root / timing.EVIDENCE_PATH
        evidence = json.loads(path.read_text())
        evidence["market_caps"][0]["as_of"] = "2026-10-15"
        path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "quarter-end date required"):
            timing.build_report(self.root, self.now)

    def test_load_missing_corrupt_and_selected_quarter(self) -> None:
        self.assertFalse(timing.load_report(self.root, "2026Q2")["available"])
        doc = timing.build_report(self.root, self.now)
        self.write_json(str(timing.REPORT_PATH), doc)
        loaded = timing.load_report(self.root, "2026Q2")
        self.assertTrue(loaded["available"])
        self.assertNotIn("rows", loaded["quarter"])
        self.assertFalse(timing.load_report(self.root, "2027Q1")["available"])
        self.write_json(str(timing.REPORT_PATH), {"schema_version": 999})
        self.assertIn("ungültig", timing.load_report(self.root, "2026Q2")["error"])

    def test_dashboard_contains_both_metrics_events_and_history(self) -> None:
        self.assertIn('data-tab="publications"', HTML)
        self.assertIn('id="panel-publications"', HTML)
        self.assertIn('id="publication-scope"', HTML)
        self.assertIn("Erste Earnings-Veröffentlichung", HTML)
        self.assertIn("SEC-Berichtseinreichung", HTML)
        self.assertIn("publicationChart(group[key],true", HTML)
        self.assertIn("publicationChart(group[key],false", HTML)
        self.assertNotIn("['reports','operations','publications'].includes(tab)", HTML)
        self.assertIn("if(location.hash==='#publications')go('publications');", HTML)


if __name__ == "__main__":
    unittest.main()
