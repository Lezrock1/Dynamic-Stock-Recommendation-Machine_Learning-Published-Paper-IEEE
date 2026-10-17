"""Offline guard, manifest comparison and status-group arithmetic (synthetic, no network)."""
import sys
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import freeze as fz  # noqa: E402
import model_coverage as mc  # noqa: E402


class Guard(unittest.TestCase):
    def test_requests_are_blocked_counted_and_restored(self):
        original = requests.sessions.Session.request
        with fz.network_guard() as attempts:
            with self.assertRaises(requests.RequestException):
                requests.get("https://example.invalid/x")
        self.assertEqual(len(attempts), 1)
        self.assertIs(requests.sessions.Session.request, original)

    def test_manifest_comparison_detects_change(self):
        a = {"files": {"x": {"sha256": "1", "size": 1}, "y": {"sha256": "2", "size": 1}}}
        b = {"files": {"x": {"sha256": "9", "size": 1}, "z": {"sha256": "3", "size": 1}}}
        diff = fz.compare_manifests(a, b)
        self.assertFalse(diff["unchanged"])
        self.assertEqual((diff["added"], diff["removed"], diff["modified"]), (["z"], ["y"], ["x"]))

    def test_snapshot_copies_files_and_consistent_sqlite_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "state" / "real_paper.sqlite3"
            database.parent.mkdir(parents=True)
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE ledger (id INTEGER PRIMARY KEY, amount REAL)")
                connection.execute("INSERT INTO ledger(amount) VALUES (12.5)")
            manifest_dir = root / "state" / "filings"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "2026Q3.json").write_text('{"expected":[]}')
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "real_paper.json"
            config_path.write_text(json.dumps({"database": str(database)}))
            bytecode_dir = root / "pipeline" / "__pycache__"
            bytecode_dir.mkdir(parents=True)
            (bytecode_dir / "ignored.cpython-312.pyc").write_bytes(b"not an input")
            (root / "pipeline" / "source.py").write_text("value = 1\n")
            training_dir = root / "Data" / "1-focasting_data"
            training_dir.mkdir(parents=True)
            (training_dir / "sector10.csv").write_text("date,value\n2026-01-01,1\n")
            (root / "Data" / "fundamental_final_table.xlsx").write_bytes(b"frozen-fundamentals")
            config = {
                "database": str(database),
                "manifest_dir": str(manifest_dir),
                "_path": str(config_path),
            }
            source_manifest = fz.build_file_manifest(config)
            self.assertNotIn("pipeline/__pycache__/ignored.cpython-312.pyc", source_manifest["files"])
            self.assertIn("pipeline/source.py", source_manifest["files"])
            snapshot_root = root / "snapshot"
            snapshot_db = fz.create_input_snapshot(config, snapshot_root)
            snapshot_config = fz.config_for_snapshot(config, snapshot_root)
            frozen_manifest = fz.build_file_manifest(snapshot_config)
            db_relative = database.relative_to(root).as_posix()
            for relative_path, entry in source_manifest["files"].items():
                if relative_path != db_relative:
                    self.assertEqual(frozen_manifest["files"][relative_path], entry)
            self.assertEqual(mc._sqlite_content_hash(database), mc._sqlite_content_hash(snapshot_db))
            self.assertEqual(
                (snapshot_root / "Data" / "1-focasting_data" / "sector10.csv").read_text(),
                "date,value\n2026-01-01,1\n",
            )
            self.assertEqual(
                (snapshot_root / "Data" / "fundamental_final_table.xlsx").read_bytes(),
                b"frozen-fundamentals",
            )
            self.assertEqual(fz.compare_manifests(frozen_manifest, fz.build_file_manifest(snapshot_config))["unchanged"],
                             True)

    def test_offline_universe_readers_ignore_cache_age_without_network_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            universe = root / "universe.csv"
            memberships = root / "membership.csv"
            sec_tickers = root / "sec-tickers.json"
            changes = root / "changes.json"
            universe.write_text("tic,conm,gsector,cik,date_added,is_current,data_source\nAAA,A Corp,10,1,2000-01-01,True,fixture\n")
            memberships.write_text("date,tickers\n2026-01-01,AAA\n")
            sec_tickers.write_text(json.dumps({"0": {"ticker": "AAA", "title": "A Corp"}}))
            changes.write_text(json.dumps({"wikitext": "fixture changes"}))
            for path in (universe, memberships, sec_tickers, changes):
                os.utime(path, (1, 1))
            with (
                mock.patch.object(fz.uv, "CACHE_FILE", universe),
                mock.patch.object(fz.uv, "HISTORICAL_MEMBERSHIP_CACHE", memberships),
                mock.patch.object(fz.uv, "SEC_TICKERS_CACHE", sec_tickers),
                mock.patch.object(fz.ua, "CACHE", changes),
                fz.network_guard() as attempts,
            ):
                self.assertEqual(fz.uv.get_sp500_universe().iloc[0]["tic"], "AAA")
                self.assertEqual(fz.uv.get_historical_membership_snapshots().iloc[0]["tickers"], "AAA")
                self.assertEqual(fz.uv._get_sec_ticker_map()["AAA"]["title"], "A Corp")
                self.assertEqual(fz.ua.fetch_changes_wikitext(), "fixture changes")
            self.assertEqual(attempts, [])

    def test_snapshot_context_redirects_code_and_cache_roots(self):
        previous_root = __import__("paper_live").REPO_ROOT
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with fz.snapshot_context(root):
                self.assertEqual(__import__("paper_live").REPO_ROOT, root.resolve())
                self.assertEqual(__import__("prices").CACHE_DIR, root.resolve() / "cache" / "prices")
        self.assertEqual(__import__("paper_live").REPO_ROOT, previous_root)

    def test_ixbrl_reads_snapshot_and_clears_loaded_filing_cache(self):
        import ixbrl
        import ixbrl_fallback

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_cache = root / "source" / "cache" / "filings"
            snapshot_cache = root / "snapshot" / "cache" / "filings"
            relative = Path("123") / "0000000123-26-000001" / "report.htm"
            source_path, snapshot_path = source_cache / relative, snapshot_cache / relative
            source_path.parent.mkdir(parents=True)
            snapshot_path.parent.mkdir(parents=True)

            def write_cached(path: Path, content: bytes) -> None:
                path.write_bytes(content)
                path.with_name(path.name + ".meta.json").write_text(json.dumps({
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "accession": "0000000123-26-000001",
                    "cik": 123,
                }))

            write_cached(source_path, b"SOURCE")
            write_cached(snapshot_path, b"SNAPSHOT")
            original_cache = ixbrl.FILING_CACHE
            ixbrl_fallback._facts.cache_clear()
            try:
                ixbrl.FILING_CACHE = source_cache
                with mock.patch.object(
                    ixbrl, "filing_facts",
                    side_effect=lambda cik, accession, document:
                    [ixbrl.fetch_document(cik, accession, document)[0].decode()],
                ):
                    self.assertEqual(
                        ixbrl_fallback._facts(123, "0000000123-26-000001", "report.htm"),
                        ("SOURCE",),
                    )
                    disabled_source = source_cache.with_name("filings-disabled")
                    source_cache.rename(disabled_source)
                    try:
                        with fz.snapshot_context(root / "snapshot"):
                            self.assertEqual(
                                ixbrl.FILING_CACHE, snapshot_cache.resolve(),
                            )
                            self.assertEqual(
                                ixbrl_fallback._facts(
                                    123, "0000000123-26-000001", "report.htm"
                                ),
                                ("SNAPSHOT",),
                            )
                    finally:
                        disabled_source.rename(source_cache)
            finally:
                ixbrl.FILING_CACHE = original_cache
                ixbrl_fallback._facts.cache_clear()


class DatabaseFilings(unittest.TestCase):
    def test_spy_status_is_excluded_and_ml_status_must_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            db = Path(temporary) / "paper.sqlite3"
            with sqlite3.connect(db) as connection:
                connection.executescript(
                    "CREATE TABLE reporting_cycles (cycle_id TEXT, cycle_key TEXT, strategy TEXT);"
                    "CREATE TABLE expected_filings (cycle_id TEXT, company_id TEXT, status TEXT);"
                    "INSERT INTO reporting_cycles VALUES "
                    "('eq','2026Q3','equal_weight_v1'),"
                    "('mv','2026Q3','mean_var_v1'),"
                    "('spy','2026Q3','spy_buy_hold_v1');"
                    "INSERT INTO expected_filings VALUES "
                    "('eq','issuer','READY'),('mv','issuer','READY'),('spy','issuer','WAITING_FILING');"
                )
            self.assertEqual(mc._db_filings(db, "2026Q3"), {"issuer": "READY"})
            with sqlite3.connect(db) as connection:
                connection.execute(
                    "UPDATE expected_filings SET status='WAITING_FACTS' WHERE cycle_id='mv'"
                )
            with self.assertRaisesRegex(ValueError, "disagree on filing status"):
                mc._db_filings(db, "2026Q3")


class StatusGroups(unittest.TestCase):
    def test_groups_are_disjoint_and_sum_to_expected(self):
        manifest = {"expected": [{"company_id": str(i), "filing_key": f"k{i}", "security_id": f"T{i}"} for i in range(6)]}
        db = {"0": "READY", "1": "WAITING_FACTS", "2": "WAITING_FILING", "3": "WAITING_FILING",
              "4": "READY", "5": "INVALID_DATA"}
        config = {"readiness_rule": {"version": 3, "corporate_action_resolutions": {"4": {
            "applies_to_filing_keys": ["k4"], "final_trading_end": "2026-01-01", "evidence": {"x": 1},
            "status": "CORPORATE_ACTION_RESOLVED", "event": "e"}}}}
        report = mc.readiness_breakdown(config, db, manifest)
        self.assertEqual(report["status_groups"], {"READY": 1, "WAITING_FACTS": 1,
                                                   "CORPORATE_ACTION_RESOLVED": 1, "WAITING_FILING": 2,
                                                   "INVALID_DATA": 1})
        self.assertEqual(report["open_total"], 4)


if __name__ == "__main__":
    unittest.main()
