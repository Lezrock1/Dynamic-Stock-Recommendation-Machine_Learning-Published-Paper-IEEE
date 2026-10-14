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

from pipeline import paper_dashboard, paper_live, portfolio_weights, sec_edgar, universe_asof
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
    def test_load_config_resolves_relative_config_path(self) -> None:
        config = paper_live.load_config("config/real_paper.json")
        self.assertEqual(Path(config["_path"]), Path("config/real_paper.json").resolve())

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
            with sqlite3.connect(database) as migrated_connection:
                self.assertEqual(migrated_connection.execute("PRAGMA user_version").fetchone()[0], 5)
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


class DashboardProjectionTests(unittest.TestCase):
    def test_dashboard_report_groups_are_disjoint_and_include_invalid_data(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """CREATE TABLE expected_filings (
                   filing_key TEXT, company_id TEXT, expected_period TEXT, status TEXT, cycle_id TEXT);
               CREATE TABLE filing_resolutions (
                   cycle_id TEXT, filing_key TEXT, resolution TEXT, evidence_json TEXT, recorded_at TEXT);"""
        )
        rows = [
            ("ready", "1", "READY"),
            ("facts", "2", "WAITING_FACTS"),
            ("filing", "3", "WAITING_FILING"),
            ("invalid", "4", "INVALID_DATA"),
            ("special", "5", "WAITING_FILING"),
        ]
        connection.executemany(
            "INSERT INTO expected_filings VALUES (?, ?, '2026-09-30', ?, 'cycle')", rows
        )
        connection.execute(
            "INSERT INTO filing_resolutions VALUES (?, ?, ?, ?, ?)",
            ("cycle", "special", "CORPORATE_ACTION_RESOLVED", '{"evidence":[{"id":"e"}]}',
             "2026-10-06T12:00:00+00:00"),
        )
        result = paper_dashboard._report_status(connection, "cycle")
        self.assertEqual(result["status_groups"], {
            "READY": 1, "WAITING_FACTS": 1, "CORPORATE_ACTION_RESOLVED": 1,
            "WAITING_FILING": 1, "INVALID_DATA": 1,
        })
        self.assertEqual(result["status_groups_sum"], 5)
        self.assertEqual(result["open"], 3)
        self.assertEqual(result["special_cases"][0]["company_id"], "5")
        connection.close()

    def test_benchmark_share_class_alias_maps_to_model_ticker(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute(
            "CREATE TABLE data_snapshots (snapshot_id TEXT, captured_at TEXT, payload_json TEXT)"
        )
        connection.execute(
            "INSERT INTO data_snapshots VALUES (?, ?, ?)",
            ("model:test", "2026-10-14T12:00:00+00:00",
             json.dumps({"signal_as_of": "2026-10-14T12:00:00+00:00",
                         "data": {"eligibility_enforced": True, "universe_as_of": "2026-09-30"}})),
        )
        holdings = {"source_url": "https://example.test/spy.xlsx", "as_of": "2026-10-13",
                    "sha256": "abc", "kind": "SPY holdings", "weights": {"BRK.B": 0.07}}
        with mock.patch.object(paper_dashboard.benchmark_weights, "latest", return_value=holdings), \
                mock.patch.object(paper_dashboard, "_universe_metadata",
                                  return_value=({"BRK-B": 45}, {"BRK-B": "Berkshire Hathaway Inc. B"})):
            result = paper_dashboard._weight_comparison(connection, "real", None, [])
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["symbol"], "BRK-B")
        self.assertEqual(result["rows"][0]["name"], "Berkshire Hathaway Inc. B")
        self.assertEqual(result["rows"][0]["sector"], "Informationstechnologie")
        self.assertEqual(result["rows"][0]["benchmark_weight"], 0.07)
        self.assertEqual(result["benchmark"]["symbol_aliases"], {"BRK.B": "BRK-B"})
        connection.close()

    def test_diagnostic_signal_target_weights_are_hidden(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute(
            "CREATE TABLE data_snapshots (snapshot_id TEXT, captured_at TEXT, payload_json TEXT)"
        )
        payload = {
            "signal_as_of": "2026-10-14T12:00:00+00:00",
            "data": {"eligibility_enforced": False, "universe_as_of": "2026-09-30"},
            "target_weights": {"equal_weight": {"AAA": 1.0}},
        }
        connection.execute("INSERT INTO data_snapshots VALUES (?, ?, ?)",
                           ("model:test", "2026-10-14T12:00:00+00:00", json.dumps(payload)))
        with mock.patch.object(paper_dashboard.benchmark_weights, "latest", return_value=None):
            result = paper_dashboard._weight_comparison(
                connection, "synthetic", None,
                [{"portfolio_id": "equal_weight", "strategy": "equal_weight",
                  "latest_mark": None, "positions": []}],
            )
        self.assertTrue(result["diagnostic"])
        self.assertIn("DIAGNOSELAUF", result["banner"])
        self.assertEqual(result["rows"], [])
        connection.close()


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

    def test_optimized_strategy_aborts_when_cap_prevents_full_investment(self) -> None:
        rng = np.random.default_rng(19)
        dates = pd.bdate_range("2025-01-01", periods=300)
        prices = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (300, 19)), axis=0),
            index=dates, columns=[f"S{i}" for i in range(19)],
        )
        selected = pd.Series(0.02, index=prices.columns)
        with self.assertRaisesRegex(portfolio_weights.OptimizationError, "requires at least 20"):
            portfolio_weights.optimize_weights(
                selected, prices, dates[-1], minimum_variance=True, fallback_to_equal=False
            )


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
                         {"period_end": "2026-08-15", "form": "10-Q"})
        # legacy +-45 day window reproduced only under the archived rule
        self.assertEqual(paper_live.expected_report(sub, self.q, paper_live.PERIOD_RULE_V1)["period_end"],
                         "2026-11-07")
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
    def test_reinstatement_and_removal_are_reversed_in_chronological_order(self) -> None:
        changes = (
            "{|\n|-\n|December 1, 2026\n|XYZ\n|[[XYZ]]\n|\n|\n|reinstatement\n|-\n"
            "|October 1, 2026\n|\n|\n|XYZ\n|[[XYZ]]\n|removal\n|- \n"
        )
        universe = pd.DataFrame([{
            "tic": "XYZ", "cik": 99, "gsector": 10, "date_added": "2026-12-01",
            "is_current": True, "data_source": "fixture",
        }])
        with mock.patch.object(universe_asof.uv, "get_sp500_universe", return_value=universe), \
                mock.patch.object(universe_asof, "fetch_changes_wikitext", return_value=changes), \
                mock.patch.object(universe_asof.uv, "_get_sec_ticker_map", return_value={}):
            before_removal, evidence = universe_asof.members_as_of("2026-09-30")
            between_events, _ = universe_asof.members_as_of("2026-11-01")
            after_reinstatement, _ = universe_asof.members_as_of("2026-12-01")
        self.assertEqual(before_removal["tic"].tolist(), ["XYZ"])
        self.assertEqual(evidence["reversed_additions"], [])
        self.assertTrue(between_events.empty)
        self.assertEqual(after_reinstatement["tic"].tolist(), ["XYZ"])

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


class PeriodCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "real.sqlite3"
        self.store = PaperStore.create(self.database, "real")
        self.store.create_portfolio("equal_weight", "Equal Weight", "equal_weight_v1", 1000.0)
        self.cycle_id = self.store.create_cycle("equal_weight", "2026Q3", "equal_weight_v1")
        self.config = {"manifest_dir": str(self.root), "database": str(self.database)}
        self.old_manifest = {
            "quarter_end": "2026-09-30",
            "universe_as_of": "2026-09-30",
            "universe_hash": "old-hash",
            "rule": "legacy",
            "period_rule": paper_live.PERIOD_RULE_V1,
            "expected": [{
                "filing_key": "1-2026-10-31-10-Q",
                "company_id": "1",
                "security_id": "AAA",
                "expected_period": "2026-10-31",
                "filing_type": "10-Q",
            }],
        }
        self.new_manifest = {
            **self.old_manifest,
            "universe_hash": "new-hash",
            "rule": "new-rule",
            "period_rule": paper_live.PERIOD_RULE_V2,
            "expected": [{
                "filing_key": "1-2026-07-31-10-Q",
                "company_id": "1",
                "security_id": "AAA",
                "expected_period": "2026-07-31",
                "filing_type": "10-Q",
            }],
        }
        (self.root / "2026Q3.json").write_text(json.dumps(self.old_manifest))
        self.store.register_expected_filings(
            self.cycle_id,
            ["1-2026-10-31-10-Q"],
            filing_metadata={"1-2026-10-31-10-Q": {
                "company_id": "1", "security_id": "AAA", "expected_period": "2026-10-31",
                "filing_type": "10-Q", "source": "manifest:old",
            }},
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _mark_ready(self) -> None:
        self.store.set_filing_result(
            self.cycle_id,
            "1-2026-10-31-10-Q",
            received=True,
            facts_ready=True,
            valid=True,
            filed_at="2026-11-20T21:00:00+00:00",
            first_seen_at="2026-11-20T21:00:00+00:00",
            source="sec:test",
        )
        self.store.set_cycle_model_status(self.cycle_id, "READY", reason="before correction")

    def test_period_correction_resets_ready_cycle_to_waiting(self) -> None:
        self._mark_ready()
        with mock.patch.object(paper_live, "generate_expected_filings", return_value=self.new_manifest):
            audit = paper_live.correct_expected_periods(
                self.store, self.config, "2026Q3", "2026-09-30", apply=True
            )
        self.assertTrue(audit["applied"])
        rows = self.store.query(
            "SELECT filing_key, expected_period, status FROM expected_filings WHERE cycle_id = ?",
            (self.cycle_id,),
        )
        self.assertEqual(rows, [{
            "filing_key": "1-2026-07-31-10-Q",
            "expected_period": "2026-07-31",
            "status": "WAITING_FILING",
        }])
        cycle = self.store.query(
            "SELECT status, filing_status, model_status, blocked_reason FROM reporting_cycles WHERE cycle_id = ?",
            (self.cycle_id,),
        )[0]
        self.assertEqual(
            {k: cycle[k] for k in ("status", "filing_status", "model_status", "blocked_reason")},
            {"status": "OPEN", "filing_status": "WAITING_FILING", "model_status": "WAITING",
             "blocked_reason": None},
        )

    def test_period_correction_resume_heals_db_manifest_mismatch(self) -> None:
        with mock.patch.object(paper_live, "generate_expected_filings", return_value=self.new_manifest), \
                mock.patch.object(paper_live, "_publish_corrected_manifest", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                paper_live.correct_expected_periods(self.store, self.config, "2026Q3", "2026-09-30", apply=True)
        pending = self.root / "2026Q3.period_correction.pending.json"
        self.assertTrue(pending.is_file())
        self.assertEqual(json.loads((self.root / "2026Q3.json").read_text())["period_rule"], paper_live.PERIOD_RULE_V1)
        self.assertEqual(
            self.store.query("SELECT filing_key FROM expected_filings WHERE cycle_id = ?", (self.cycle_id,))[0]["filing_key"],
            "1-2026-07-31-10-Q",
        )
        with mock.patch.object(paper_live, "generate_expected_filings", return_value=self.new_manifest):
            resumed = paper_live.correct_expected_periods(self.store, self.config, "2026Q3", "2026-09-30", apply=True)
        self.assertTrue(resumed["applied"])
        self.assertFalse(pending.exists())
        self.assertEqual(json.loads((self.root / "2026Q3.json").read_text())["period_rule"], paper_live.PERIOD_RULE_V2)
        again = paper_live.correct_expected_periods(self.store, self.config, "2026Q3", "2026-09-30")
        self.assertIn("archive, and audit agree", again["note"])
        self.audit_path = self.root / "2026Q3.period_correction.json"
        self.audit_path.unlink()
        with self.assertRaisesRegex(ValidationError, "publication is incomplete"):
            paper_live.correct_expected_periods(self.store, self.config, "2026Q3", "2026-09-30")

    def test_period_correction_rolls_back_before_commit_and_resumes(self) -> None:
        with mock.patch.object(PaperStore, "_refresh_filing_status", side_effect=RuntimeError("commit injection")):
            with mock.patch.object(paper_live, "generate_expected_filings", return_value=self.new_manifest):
                with self.assertRaisesRegex(RuntimeError, "commit injection"):
                    paper_live.correct_expected_periods(
                        self.store, self.config, "2026Q3", "2026-09-30", apply=True
                    )
        self.assertTrue((self.root / "2026Q3.period_correction.pending.json").is_file())
        self.assertEqual(
            self.store.query("SELECT filing_key FROM expected_filings WHERE cycle_id = ?", (self.cycle_id,))[0]["filing_key"],
            "1-2026-10-31-10-Q",
        )
        with mock.patch.object(paper_live, "generate_expected_filings", return_value=self.new_manifest):
            resumed = paper_live.correct_expected_periods(
                self.store, self.config, "2026Q3", "2026-09-30", apply=True
            )
        self.assertTrue(resumed["applied"])
        self.assertEqual(
            self.store.query("SELECT filing_key FROM expected_filings WHERE cycle_id = ?", (self.cycle_id,))[0]["filing_key"],
            "1-2026-07-31-10-Q",
        )

    def test_period_correction_respects_scheduler_lock(self) -> None:
        from pipeline.paper_runner import LockBusyError, exclusive_lock
        (self.root / "2026Q3.json").write_text(json.dumps(self.old_manifest))
        with exclusive_lock(self.database):
            with self.assertRaises(LockBusyError):
                paper_live.correct_expected_periods(
                    self.store, self.config, "2026Q3", "2026-09-30", apply=True
                )
        self.assertFalse((self.root / "2026Q3.period_correction.pending.json").exists())

    def test_period_correction_preserves_unrelated_model_block(self) -> None:
        self.store.set_cycle_model_status(self.cycle_id, "BLOCKED", reason="manual review")
        with mock.patch.object(paper_live, "generate_expected_filings", return_value=self.new_manifest):
            with self.assertRaisesRegex(ValidationError, "is blocked for another reason"):
                paper_live.correct_expected_periods(
                    self.store, self.config, "2026Q3", "2026-09-30", apply=True
                )
        row = self.store.query(
            "SELECT status, blocked_reason, model_status, model_reason "
            "FROM reporting_cycles WHERE cycle_id = ?", (self.cycle_id,)
        )[0]
        self.assertEqual(
            row,
            {
                "status": "BLOCKED",
                "blocked_reason": "manual review",
                "model_status": "BLOCKED",
                "model_reason": "manual review",
            },
        )
        self.assertFalse((self.root / "2026Q3.period_correction.pending.json").exists())


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

    def test_acceptance_after_utc_cutoff_is_rejected_even_when_filed_date_is_previous_day(self) -> None:
        accepted = pd.Timestamp("2026-10-14T01:00:00Z")
        entries = [{"filed": "2026-10-13", "accn": "overnight", "end": "2026-09-30"}]
        with sec_edgar.point_in_time(pd.Timestamp("2026-10-14T00:30:00Z"), {"overnight": accepted}):
            kept = sec_edgar._best_by_key(entries, lambda entry: entry["end"])
        self.assertEqual(kept, {})

        cutoff = pd.Timestamp("2026-10-14T01:30:00Z")
        with sec_edgar.point_in_time(cutoff, {"overnight": accepted}):
            kept = sec_edgar._best_by_key(entries, lambda entry: entry["end"])
        self.assertEqual(kept["2026-09-30"]["_available_at"], accepted)

    def test_facts_ready_requires_current_numeric_value_and_unit(self) -> None:
        names = ["Revenue", "NetIncome", "Assets", "StockholdersEquity"]
        accession, period = "acc-current", "2026-09-30"

        def make_facts(revenue_value: float | None, revenue_period: str = period) -> dict:
            gaap = {}
            for name in names:
                tag = (sec_edgar.DURATION_TAGS if name in sec_edgar.DURATION_TAGS
                       else sec_edgar.INSTANT_TAGS)[name][0]
                end = revenue_period if name == "Revenue" else period
                value = revenue_value if name == "Revenue" else 100.0
                gaap[tag] = {"units": {"USD": [{"accn": accession, "end": end, "val": value}]}}
            return {"facts": {"us-gaap": gaap}}

        source = paper_live.SecFilingSource({"required_facts": names})
        with mock.patch.object(sec_edgar, "get_company_facts", return_value=make_facts(None)), \
                mock.patch.object(paper_live.ixf, "enabled_for", return_value=False):
            self.assertFalse(source._facts_ready(123, accession, names, period))

        with mock.patch.object(sec_edgar, "get_company_facts", return_value=make_facts(10.0, "2026-06-30")), \
                mock.patch.object(paper_live.ixf, "enabled_for", return_value=False):
            self.assertFalse(source._facts_ready(123, accession, names, period))

        with mock.patch.object(sec_edgar, "get_company_facts", return_value=make_facts(None)), \
                mock.patch.object(paper_live.ixf, "enabled_for", return_value=True), \
                mock.patch.object(paper_live.ixf, "fallback_entries", return_value={
                    "Revenue": [{
                        "accn": accession, "end": period, "val": 10.0,
                        "_prov": {"unit": "USD", "context_id": "ctx-current", "period_end": period},
                    }]
                }):
            self.assertTrue(source._facts_ready(123, accession, names, period))

        with mock.patch.object(sec_edgar, "get_company_facts", return_value=make_facts(None)), \
                mock.patch.object(paper_live.ixf, "enabled_for", return_value=True), \
                mock.patch.object(paper_live.ixf, "fallback_entries", return_value={
                    "Revenue": [{
                        "accn": accession, "end": "2026-06-30", "val": 10.0,
                        "_prov": {"unit": "USD", "context_id": "ctx-comparative",
                                  "period_end": "2026-06-30"},
                    }]
                }):
            self.assertFalse(source._facts_ready(123, accession, names, period))

    def test_current_cycle_reports_share_future_slot_even_if_one_old_slot_passed(self) -> None:
        provider = paper_live.ModelSignalProvider({"model": {}, "required_facts": [],
                                                   "report_matching_tolerance_days": 10})
        provider.expected_periods = {1: pd.Timestamp("2026-09-30"), 2: pd.Timestamp("2026-09-30"),
                                     3: pd.Timestamp("2026-09-30")}
        snapshot = pd.Timestamp("2026-12-15T20:00:00Z")
        price_session = pd.Timestamp("2026-12-14")
        rows = pd.DataFrame([
            {"tic": "AAA", "datadate": "2026-09-30", "available_at": "2026-10-20T20:00:00Z",
             "tradedate": pd.Timestamp("2026-12-01")},
            {"tic": "BBB", "datadate": "2026-09-30", "available_at": "2026-12-10T20:00:00Z",
             "tradedate": pd.Timestamp("2027-03-01")},
            {"tic": "CCC", "datadate": "2026-09-30", "available_at": "2026-12-16T12:00:00Z",
             "tradedate": pd.Timestamp("2027-06-01")},
        ])
        aligned, pending = provider._align_cycle_reports(
            rows, {"AAA": 1, "BBB": 2, "CCC": 3}, price_session, snapshot
        )
        self.assertEqual(pending, pd.Timestamp("2027-03-01"))
        self.assertEqual(aligned.loc[0, "tradedate"], pending)
        self.assertEqual(aligned.loc[1, "tradedate"], pending)
        self.assertEqual(aligned.loc[2, "tradedate"], pd.Timestamp("2027-06-01"))

    def test_only_past_current_cycle_rows_get_a_new_future_slot(self) -> None:
        provider = paper_live.ModelSignalProvider({"model": {}, "required_facts": [],
                                                   "report_matching_tolerance_days": 10})
        provider.expected_periods = {1: pd.Timestamp("2026-09-30")}
        snapshot = pd.Timestamp("2026-12-15T20:00:00Z")
        rows = pd.DataFrame([{
            "tic": "AAA", "datadate": "2026-09-30", "available_at": "2026-10-20T20:00:00Z",
            "tradedate": pd.Timestamp("2026-12-01"),
        }])
        aligned, pending = provider._align_cycle_reports(
            rows, {"AAA": 1}, pd.Timestamp("2026-12-14"), snapshot
        )
        self.assertGreater(pending, pd.Timestamp("2026-12-14"))
        self.assertEqual(aligned.loc[0, "tradedate"], pending)


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
