from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from pipeline import paper_live, portfolio_weights, sec_edgar, universe_asof
from pipeline.paper_dashboard import dashboard_snapshot, evaluate_backtest_validation
from pipeline.paper_portfolio import run_cli
from pipeline.paper_store import PaperStore, ValidationError


def filings(*rows):
    keys = ("form", "reportDate")
    return {"filings": {"recent": {"form": [r[0] for r in rows], "reportDate": [r[1] for r in rows],
                                   "accessionNumber": [f"a{i}" for i in range(len(rows))],
                                   "filingDate": [r[1] for r in rows],
                                   "acceptanceDateTime": [r[1] + "T20:00:00.000Z" for r in rows]}}}


class MigrationTests(unittest.TestCase):
    def test_v2_store_is_backed_up_migrated_once_and_keeps_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "real.sqlite3"
            store = PaperStore.create(database, "real")
            store.create_portfolio("equal_weight", "Equal Weight", "equal_weight_v1", 1000.0)
            before = store.cash_balance("equal_weight")
            connection = sqlite3.connect(database)
            connection.executescript("DROP TABLE cycle_timeline; DROP TABLE store_meta; PRAGMA user_version = 2;")
            connection.close()
            result = PaperStore.migrate(database, "real", root / "backups")
            self.assertTrue(result["migrated"])
            self.assertTrue(Path(result["backup"]).is_file())
            migrated = PaperStore(database)
            self.assertEqual(migrated.environment(), "real")
            self.assertEqual(migrated.cash_balance("equal_weight"), before)
            again = PaperStore.migrate(database, "real", root / "backups")
            self.assertFalse(again["migrated"])
            self.assertEqual(len(list((root / "backups").iterdir())), 1)


class SeparationTests(unittest.TestCase):
    def test_environments_never_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real, synthetic = root / "real.sqlite3", root / "synthetic.sqlite3"
            with redirect_stdout(io.StringIO()):
                run_cli(["--env", "real", "--db", str(real), "init"])
                run_cli(["--env", "synthetic", "--db", str(synthetic), "demo"])
            self.assertEqual(PaperStore(real).environment(), "real")
            self.assertEqual(PaperStore(synthetic).environment(), "synthetic")
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                run_cli(["--env", "real", "--db", str(synthetic), "status"])
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                run_cli(["--env", "synthetic", "--db", str(real), "status"])
            self.assertEqual(dashboard_snapshot(synthetic)["historical_backtest_validation"]["status"],
                             "NOT_VALIDATED")
            self.assertIn("SYNTHETIC", dashboard_snapshot(synthetic)["data_banner"])
            self.assertIn("REAL", dashboard_snapshot(real)["data_banner"])


class BacktestValidationTests(unittest.TestCase):
    def manifest(self, root: Path, tamper: bool = False) -> Path:
        evidence = {}
        for key in ("raw_price_basis", "corporate_actions", "historical_membership",
                    "point_in_time_signal_and_labels", "benchmark_provenance"):
            (root / f"{key}.txt").write_text(key)
            digest = hashlib.sha256(b"x" if tamper else key.encode()).hexdigest()
            evidence[key] = {"verified": True, "evidence_uri": f"{key}.txt", "sha256": digest}
        path = root / "run_manifest.json"
        path.write_text(json.dumps({
            "mode": "historical_backtest", "historical_data_validation": "VALIDATED",
            "input_sha256": {k: "h" for k in paper_live_required()}, "validation_evidence": evidence}))
        return path

    def test_status_field_alone_never_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            path.write_text(json.dumps({"mode": "historical_backtest",
                                        "historical_data_validation": "VALIDATED"}))
            self.assertEqual(evaluate_backtest_validation(path)["status"], "NOT_VALIDATED")
            (Path(directory) / "backtest_validation.json").write_text('{"status": "VALIDATED"}')
            store = PaperStore.create(Path(directory) / "real.sqlite3", "real")
            self.assertEqual(dashboard_snapshot(store.path)["historical_backtest_validation"]["status"],
                             "NOT_VALIDATED")

    def test_hash_verified_evidence_validates_and_tampering_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(evaluate_backtest_validation(self.manifest(root))["status"], "VALIDATED")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(evaluate_backtest_validation(self.manifest(Path(directory), True))["status"],
                             "NOT_VALIDATED")


def paper_live_required():
    return ("raw_prices", "targets", "corporate_actions", "benchmark_values", "benchmark_metadata")


class WeightTests(unittest.TestCase):
    def test_three_strategies_use_their_own_calculation_and_cap(self) -> None:
        rng = np.random.default_rng(7)
        names = [f"S{i}" for i in range(30)]
        vol = np.linspace(0.005, 0.03, 30)
        panel = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, vol, (300, 30)), axis=0), columns=names,
                             index=pd.bdate_range("2025-01-01", periods=300))
        selected = pd.Series(np.linspace(0.01, 0.06, 30), index=names)
        date = panel.index[-1]
        equal = portfolio_weights.capped_equal_weights(list(selected.index))
        mean_var = portfolio_weights.optimize_weights(selected, panel, date, fallback_to_equal=False)
        min_var = portfolio_weights.optimize_weights(selected, panel, date, minimum_variance=True,
                                                     fallback_to_equal=False)
        for weights in (equal, mean_var, min_var):
            self.assertLessEqual(weights.max(), portfolio_weights.WEIGHT_CAP + 1e-9)
            self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        self.assertGreater((mean_var - equal).abs().max(), 1e-3)
        self.assertGreater((min_var - mean_var).abs().max(), 1e-3)
        self.assertGreater(min_var[names[0]], min_var[names[-1]])


class ExpectedReportTests(unittest.TestCase):
    q = pd.Timestamp("2026-09-30")

    def test_52_53_week_calendar_maps_to_nominal_quarters(self) -> None:
        sub = filings(("10-Q", "2026-07-03"), ("10-Q", "2026-04-03"), ("10-K", "2025-12-31"),
                      ("10-Q", "2025-09-26"), ("10-Q", "2025-06-27"))
        self.assertEqual(paper_live.expected_report(sub, self.q), {"period_end": "2026-09-30", "form": "10-Q"})

    def test_fiscal_year_end_quarter_expects_10k(self) -> None:
        sub = filings(("10-Q", "2026-06-30"), ("10-Q", "2026-03-31"), ("10-K", "2025-12-31"),
                      ("10-Q", "2025-09-30"))
        self.assertEqual(paper_live.expected_report(sub, self.q)["form"], "10-Q")
        sub = filings(("10-Q", "2026-06-30"), ("10-Q", "2026-03-31"), ("10-K", "2025-09-30"),
                      ("10-Q", "2025-06-30"), ("10-Q", "2025-03-31"))
        self.assertEqual(paper_live.expected_report(sub, self.q), {"period_end": "2026-09-30", "form": "10-K"})

    def test_off_grid_calendar_uses_anniversary_and_unknown_history_is_unresolved(self) -> None:
        sub = filings(("10-Q", "2026-05-23"), ("10-K", "2026-01-31"), ("10-Q", "2025-11-08"),
                      ("10-Q", "2025-08-16"), ("10-Q", "2025-05-24"))
        self.assertEqual(paper_live.expected_report(sub, self.q),
                         {"period_end": "2026-11-07", "form": "10-Q"})
        self.assertIsNone(paper_live.expected_report(filings(("10-Q", "2026-06-30")), self.q))
        self.assertIsNone(paper_live.expected_report(None, self.q))
        self.assertEqual(paper_live.classify_unresolved(1, ["X"], filings(("10-Q", "2026-06-30")))["reason"],
                         "no_annual_report_in_history")


class SuccessionTests(unittest.TestCase):
    def tearDown(self) -> None:
        sec_edgar.register_successors({})

    def test_histories_are_merged_and_identical_publications_deduplicated(self) -> None:
        old = filings(("10-K", "2025-12-31"), ("10-Q", "2026-06-30"))
        new = filings(("10-Q", "2026-06-30"))
        new["filings"]["recent"]["accessionNumber"] = [old["filings"]["recent"]["accessionNumber"][1]]
        sec_edgar.register_successors({2: (1,)})
        with mock.patch.object(sec_edgar, "_fetch_submissions", side_effect=lambda c, a=0: {1: old, 2: new}[c]):
            merged = sec_edgar.get_submissions(2)
        recent = merged["filings"]["recent"]
        self.assertEqual(len(recent["accessionNumber"]), 2)
        self.assertEqual(sorted(recent["form"]), ["10-K", "10-Q"])
        self.assertEqual(paper_live.expected_report(merged, pd.Timestamp("2026-09-30")),
                         {"period_end": "2026-09-30", "form": "10-Q"})

    def test_unregistered_cik_is_not_merged(self) -> None:
        with mock.patch.object(sec_edgar, "_fetch_submissions", return_value={"filings": {"recent": {}}}) as fetch:
            sec_edgar.get_submissions(2)
        fetch.assert_called_once()


class UniverseAsOfTests(unittest.TestCase):
    def test_changes_after_reference_date_are_reversed(self) -> None:
        text = ("{|\n|-\n|October 6, 2026\n|TWLO\n|[[Twilio]]\n|WBD\n|[[Warner]]\n|reason\n|-\n"
                "|October 1, 2026\n|VYLR\n|[[Vylor]]\n|\n|\n|spin-off\n|-\n")
        changes = universe_asof.parse_changes(text)
        self.assertEqual(changes.to_dict("records")[0]["removed"], "WBD")
        universe = pd.DataFrame([
            {"tic": "AAA", "cik": 1, "gsector": 10, "date_added": "2000-01-01", "is_current": True},
            {"tic": "TWLO", "cik": 2, "gsector": 45, "date_added": "2026-10-06", "is_current": True},
            {"tic": "VYLR", "cik": 3, "gsector": 30, "date_added": "2026-10-01", "is_current": True},
            {"tic": "WBD", "cik": 4, "gsector": 50, "date_added": None, "is_current": False},
        ])
        with mock.patch.object(universe_asof.uv, "get_sp500_universe", return_value=universe), \
                mock.patch.object(universe_asof, "fetch_changes_wikitext", return_value=text), \
                mock.patch.object(universe_asof.uv, "_get_sec_ticker_map", return_value={}):
            members, evidence = universe_asof.members_as_of("2026-09-30")
        self.assertEqual(sorted(members["tic"]), ["AAA", "WBD"])
        self.assertEqual(evidence["reversed_additions"], ["TWLO", "VYLR"])

    def test_frozen_manifest_refuses_a_different_universe_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {"manifest_dir": directory}
            manifest = {"universe_as_of": "2026-09-30", "universe_hash": "h"}
            paper_live.write_manifest(config, "2026Q3", manifest)
            with self.assertRaises(ValidationError):
                paper_live.write_manifest(config, "2026Q3", {**manifest, "universe_as_of": "2026-10-13"})


class PointInTimeTests(unittest.TestCase):
    def test_same_day_fact_needs_acceptance_before_cutoff(self) -> None:
        entries = [{"filed": "2026-10-13", "accn": "late", "end": "2026-09-30"},
                   {"filed": "2026-10-13", "accn": "early", "end": "2026-06-30"},
                   {"filed": "2026-10-12", "accn": "old", "end": "2026-03-31"},
                   {"filed": "2026-10-13", "accn": "unproven", "end": "2025-12-31"}]
        acceptance = {"late": pd.Timestamp("2026-10-13T18:00:00Z"), "early": pd.Timestamp("2026-10-13T08:00:00Z"),
                      "old": pd.Timestamp("2026-10-12T20:00:00Z")}
        with sec_edgar.point_in_time(pd.Timestamp("2026-10-13T12:00:00Z"), acceptance):
            kept = sec_edgar._best_by_key(entries, lambda e: e["end"])
        self.assertEqual(sorted(kept), ["2026-03-31", "2026-06-30"])


@unittest.skipUnless("@" in os.getenv("DYNAMIC_STOCK_USER_AGENT", ""),
                     "set DYNAMIC_STOCK_USER_AGENT to run the real-data integration test")
class RealDataIntegrationTests(unittest.TestCase):
    """Network test against SEC and Yahoo; never writes to a depot database."""

    def test_sec_filing_prices_and_weights_with_current_data(self) -> None:
        config = paper_live.load_config()
        submissions = sec_edgar.get_submissions(320193, max_age_seconds=300)
        report = paper_live.expected_report(submissions, pd.Timestamp("2026-06-30"))
        self.assertEqual(report["form"], "10-Q")
        observed = paper_live.SecFilingSource(config).observe([{
            "filing_key": "aapl", "company_id": "320193", "expected_period": report["period_end"],
            "filing_type": "10-Q"}])
        self.assertTrue(observed["aapl"].received and observed["aapl"].facts_ready)
        self.assertIsNotNone(observed["aapl"].filed_at)
        self.assertIsNone(paper_live.SecFilingSource(config).observe([{
            "filing_key": "future", "company_id": "320193", "expected_period": "2099-03-31",
            "filing_type": "10-Q"}]).get("future"))
        market = paper_live.YahooMarket(config)
        self.assertGreater(market.close("AAPL", "2026-10-09"), 0)
        self.assertGreater(market.benchmark("2026-10-09"), 0)
        sec_edgar.register_successors({})
        xom = sec_edgar.get_submissions(2115436, max_age_seconds=300)
        self.assertTrue(xom)


if __name__ == "__main__":
    unittest.main()
