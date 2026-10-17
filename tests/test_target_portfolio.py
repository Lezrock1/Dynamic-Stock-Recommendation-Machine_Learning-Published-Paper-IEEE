"""Target-portfolio export/import contract tests.

All securities, ISINs and weights below are SYNTHETIC FIXTURES (prefix "SYNTH"); they are not
model results and say nothing about real performance. Only the NYSE schedule is real.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

from pipeline import paper_quarter
from pipeline import target_portfolio as tp
from pipeline.paper_fixed_lag import OPTIMIZATION_VERSION, optimization_definition, rule_manifest

CYCLE = "2026Q2"
CUTOFF = datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc)
EXECUTION = datetime(2026, 9, 1, 20, 0, tzinfo=timezone.utc)
IN_WINDOW = CUTOFF + timedelta(hours=3)


def synthetic_isin(body: str) -> str:
    for digit in "0123456789":
        if tp.isin_is_valid(f"XX{body}{digit}"):
            return f"XX{body}{digit}"
    raise AssertionError(body)


IDS = [f"SYNTH{index:02d}" for index in range(25)]


def instrument_map(ids=IDS, **override) -> dict:
    rows = []
    for index, security_id in enumerate(ids):
        rows.append({
            "security_id": security_id, "isin": synthetic_isin(f"SYN{index:06d}"), "wkn": "",
            "ticker": f"SYN{index}", "name": f"Synthetic, \"Fixture\" {index}", "asset_type": "COMMON_STOCK",
            "share_class": "", "sector_code": "45", "sector_name": "Synthetic sector", "currency": "USD",
            "verification_status": "VERIFIED",
            "evidence": [{"source": "SYNTHETIC_TEST_FIXTURE", "reference": f"fixture:{security_id}",
                          "retrieved_at_utc": "2026-08-01T00:00:00+00:00", "sha256": "0" * 64}],
        })
    doc = {"format_version": tp.INSTRUMENT_MAP_VERSION, "frozen_at_utc": "2026-08-15T00:00:00+00:00",
           "instruments": rows}
    doc.update(override)
    return doc


def signal(**override) -> dict:
    mean = {sid: 0.05 for sid in IDS[:10]}
    mean.update({sid: 0.5 / 15 for sid in IDS[10:]})
    doc = {
        "format_version": tp.SIGNAL_RESULT_VERSION, "cycle_key": CYCLE, "rule_version": tp.RULE_VERSION,
        "information_cutoff_utc": "2026-08-31T20:00:00+00:00",
        "scheduled_execution_at_utc": "2026-09-01T20:00:00Z",
        "signal_computed_at_utc": "2026-08-31T22:00:00+00:00",
        "code_id": "synthetic-code", "data_id": "synthetic-data", "config_id": "synthetic-config",
        "data_check_status": "PASSED", "historical_validation_status": "NOT_VALIDATED",
        "base_currency": "USD", "synthetic_fixture": True, "historical_only": False,
        "restrictions": ["SYNTHETIC TEST FIXTURE"], "model_rules": rule_manifest(),
        "strategies": {
            "equal_weight": {"status": "COMPLETE", "weights": {sid: 1 / 25 for sid in IDS}, "error": None},
            "min_variance": {"status": "COMPLETE", "error": None,
                             "weights": {**{sid: 0.05 for sid in IDS[:20]}, **{sid: 0.0 for sid in IDS[20:]}}},
            "mean_variance": {"status": "COMPLETE", "weights": mean, "error": None},
        },
    }
    doc.update(override)
    return doc


ACCEPT: dict[str, Any] = {"accept_unvalidated_model": True, "allow_synthetic_fixture": True}


class Workspace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.map_path = self.write_map(instrument_map())
        self.out = self.root / "packages"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_map(self, doc: dict, name: str = "map.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def export(self, doc=None, *, mode=tp.MODE_REGULAR, at=IN_WINDOW, out=None, **kwargs) -> Path:
        return tp.export_package(doc or signal(), self.map_path, out or self.out, mode=mode,
                                 now=lambda: at, **kwargs)

    @staticmethod
    def rewrite(package: Path, name: str, data: bytes) -> None:
        (package / name).write_bytes(data)
        manifest = json.loads((package / "manifest.json").read_text())
        manifest["files"][name] = {"sha256": tp.sha256_bytes(data), "bytes": len(data)}
        (package / "manifest.json").write_text(json.dumps(manifest))


class StrategyDefinitionTests(Workspace):
    def test_max_sharpe_definition_is_versioned_and_matches_rules(self) -> None:
        from pipeline.paper_fixed_lag import RULES
        for key, value in tp.FROZEN_DEFINITION_PARAMETERS.items():
            self.assertEqual(getattr(RULES, key), value, key)
        self.assertAlmostEqual(tp.RISK_FREE_ANNUAL_LOG, 0.014888612493750654, places=15)
        package = self.export()
        manifest = json.loads((package / "manifest.json").read_text())
        item = manifest["strategies"]["mean_variance"]
        self.assertEqual(item["label"], "Rendite/Risiko – Max-Sharpe")
        self.assertEqual(item["definition_version"], "mean_variance-annual-log-max-sharpe-v1")
        self.assertEqual(manifest["model"]["optimization_version"], OPTIMIZATION_VERSION)
        self.assertEqual(manifest["model"]["optimization_definition"], optimization_definition())
        self.assertIn("simple-return Sharpe", manifest["model"]["performance_metric_note"])
        units = manifest["model"]["strategy_definitions"]["mean_variance"]["units"]
        self.assertIn("log1p(0.015)", units["risk_free"])
        self.assertIn("252 x", units["covariance"])
        self.assertIn("4 x predicted quarterly log return", units["expected_return"])
        self.assertIn("Rendite/Risiko – Max-Sharpe", (package / "zusammenfassung.md").read_text())
        historical = self.export(mode=tp.MODE_HISTORICAL, out=self.root / "hist")
        self.assertEqual(json.loads((historical / "manifest.json").read_text())["strategies"]["mean_variance"]
                         ["definition_version"], item["definition_version"])

    def test_tuned_parameters_rejected(self) -> None:
        for key, value in (("risk_free_annual", 0.02), ("weight_cap", 0.06)):
            rules = {**rule_manifest(), key: value}
            with self.subTest(key=key), self.assertRaisesRegex(tp.TargetPortfolioError, "frozen strategy definition"):
                tp.validate_signal_result(signal(model_rules=rules))

    def test_unpinned_or_older_optimizer_results_not_joined(self) -> None:
        legacy = {k: v for k, v in rule_manifest().items() if not k.startswith("optimization")}
        changed = {**rule_manifest(), "optimization_definition": {**optimization_definition(), "risk_free": "x"}}
        for rules, pattern in ((None, "rule manifest of the producing run"),
                               (legacy, "older optimizer results are not joined"),
                               (changed, "optimization_definition differs")):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(tp.TargetPortfolioError, pattern):
                tp.validate_signal_result(signal(model_rules=rules))

    def test_unknown_definition_version_rejected_on_import(self) -> None:
        package = self.export()
        manifest = json.loads((package / "manifest.json").read_text())
        manifest["strategies"]["mean_variance"]["definition_version"] = "mean_variance-max_sharpe-v2"
        (package / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(tp.TargetPortfolioError, "unknown strategy definition version"):
            tp.inspect_package(package, now=IN_WINDOW, **ACCEPT)


class ExportTests(Workspace):
    def test_current_package_layout_and_csv_contract(self) -> None:
        package = self.export()
        self.assertEqual(sorted(p.name for p in package.iterdir()), sorted([
            "equal_weight.csv", "min_variance.csv", "mean_variance.csv", "manifest.json",
            "zusammenfassung.md", "instrument_map.json", "signal_result.json"]))
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual(manifest["package_kind"], tp.KIND_CURRENT)
        self.assertTrue(manifest["handoff_eligible"])
        self.assertTrue(manifest["not_trade_instructions"])
        self.assertEqual(manifest["valid_until_utc"], "2026-09-01T20:00:00+00:00")
        self.assertEqual(manifest["information_cutoff_utc"], "2026-08-31T20:00:00+00:00")
        with open(package / "equal_weight.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(tuple(rows[0]), tp.CSV_COLUMNS)
        self.assertEqual(len(rows), 26)
        self.assertEqual(rows[0]["target_weight"], "0.040000000000")
        self.assertEqual(rows[-1]["row_type"], "CASH")
        self.assertEqual(sum(Decimal(r["target_weight"]) for r in rows), Decimal(1))
        self.assertIn('"Synthetic, ""Fixture"" 0"', (package / "equal_weight.csv").read_text())
        self.assertIn("keine Handelsanweisung", (package / "zusammenfassung.md").read_text())
        self.assertEqual(manifest["model"]["paper_replication"], "NOT_EXACT_PAPER_REPLICATION")
        self.assertEqual(manifest["model"]["validation_status"], "NOT_VALIDATED")
        self.assertIn("max_sharpe", manifest["model"]["strategy_definitions"]["mean_variance"]["deviation"])
        self.assertIn("NOT_EXACT_PAPER_REPLICATION", (package / "zusammenfassung.md").read_text())

    def test_import_succeeds_only_with_explicit_acceptance(self) -> None:
        package = self.export()
        report, portfolio = tp.load_target_portfolio(package, "mean_variance", now=IN_WINDOW, **ACCEPT)
        self.assertEqual(len(portfolio.positions), 25)
        self.assertEqual(report.manifest["package_kind"], tp.KIND_CURRENT)
        with self.assertRaisesRegex(tp.TargetPortfolioError, "Synthetic fixture"):
            tp.load_target_portfolio(package, "mean_variance", now=IN_WINDOW, accept_unvalidated_model=True)
        with self.assertRaisesRegex(tp.TargetPortfolioError, "validation status"):
            tp.load_target_portfolio(package, "mean_variance", now=IN_WINDOW, allow_synthetic_fixture=True)

    def test_equal_weight_cash_is_explicit_without_renormalization(self) -> None:
        doc = signal()
        doc["strategies"]["equal_weight"]["weights"] = {sid: 0.03 for sid in IDS}
        package = self.export(doc)
        _, portfolio = tp.load_target_portfolio(package, "equal_weight", now=IN_WINDOW, **ACCEPT)
        self.assertEqual(portfolio.cash_weight, Decimal("0.25"))
        self.assertTrue(all(p.target_weight == Decimal("0.03") for p in portfolio.positions))

    def test_allocation_id_stable_and_issuance_distinct(self) -> None:
        first = self.export()
        self.assertEqual(self.export(at=IN_WINDOW + timedelta(minutes=5)), first)
        other = self.export(out=self.root / "elsewhere", at=IN_WINDOW + timedelta(minutes=7))
        a, b = (json.loads((p / "manifest.json").read_text()) for p in (first, other))
        self.assertEqual(a["allocation_id"], b["allocation_id"])
        self.assertNotEqual(a["run_id"], b["run_id"])
        changed = signal()
        changed["strategies"]["equal_weight"]["weights"] = {sid: 0.03 for sid in IDS}
        c = json.loads((self.export(changed) / "manifest.json").read_text())
        self.assertNotEqual(c["allocation_id"], a["allocation_id"])

    def test_dst_schedule_and_cli_window(self) -> None:
        q3 = tp.cycle_timing("2026Q3")
        self.assertEqual(q3["information_cutoff_utc"], "2026-11-30T21:00:00+00:00")
        self.assertEqual(q3["scheduled_execution_at_utc"], "2026-12-01T21:00:00+00:00")
        self.assertEqual(paper_quarter.active_cycle(IN_WINDOW), CYCLE)
        self.assertIsNone(paper_quarter.active_cycle(CUTOFF - timedelta(seconds=1)))
        self.assertIsNone(paper_quarter.active_cycle(EXECUTION))
        self.assertEqual(paper_quarter.next_cycle(EXECUTION), "2026Q3")


class TimingTests(Workspace):
    def test_too_early_and_too_late_publish_nothing(self) -> None:
        with self.assertRaises(tp.TimingError):
            self.export(at=CUTOFF - timedelta(minutes=1))
        with self.assertRaises(tp.TimingError):
            self.export(at=EXECUTION)
        self.assertEqual(list(self.out.iterdir()) if self.out.exists() else [], [])

    def test_stale_and_historical_are_machine_distinguishable(self) -> None:
        stale = self.export(at=EXECUTION + timedelta(days=1), allow_stale=True)
        replay = self.export(mode=tp.MODE_HISTORICAL, at=datetime(2026, 10, 16, tzinfo=timezone.utc))
        for package, kind in ((stale, tp.KIND_STALE), (replay, tp.KIND_HISTORICAL)):
            report = tp.inspect_package(package, now=IN_WINDOW, **ACCEPT)
            self.assertEqual(report.manifest["package_kind"], kind)
            self.assertFalse(report.manifest["handoff_eligible"])
            with self.assertRaisesRegex(tp.TargetPortfolioError, "not a current handoff"):
                tp.load_target_portfolio(package, "equal_weight", now=IN_WINDOW, **ACCEPT)

    def test_expiry_uses_actual_time(self) -> None:
        package = self.export()
        with self.assertRaisesRegex(tp.TargetPortfolioError, "Expired"):
            tp.load_target_portfolio(package, "equal_weight", now=EXECUTION, **ACCEPT)
        with self.assertRaisesRegex(tp.TargetPortfolioError, "Expired"):
            tp.load_target_portfolio(package, "equal_weight", **ACCEPT)  # real clock: 2026Q2 is over

    def test_failed_data_check_or_historical_signal_is_diagnostic(self) -> None:
        for doc in (signal(data_check_status="BLOCKED"), signal(historical_only=True)):
            package = self.export(doc, out=self.root / doc["data_check_status"] / str(doc["historical_only"]))
            self.assertEqual(json.loads((package / "manifest.json").read_text())["package_kind"], tp.KIND_DIAGNOSTIC)


class FailureTests(Workspace):
    def test_failed_strategy_has_no_csv_and_no_substitute(self) -> None:
        doc = signal()
        doc["strategies"]["min_variance"] = {"status": "FAILED", "weights": None,
                                             "error": "OPTIMIZATION_FAILED: synthetic"}
        package = self.export(doc)
        self.assertFalse((package / "min_variance.csv").exists())
        failure = json.loads((package / "min_variance.FAILED.json").read_text())
        self.assertTrue(failure["no_target_portfolio"])
        with self.assertRaisesRegex(tp.TargetPortfolioError, "min_variance status FAILED"):
            tp.load_target_portfolio(package, "min_variance", now=IN_WINDOW, **ACCEPT)
        tp.load_target_portfolio(package, "equal_weight", now=IN_WINDOW, **ACCEPT)

    def test_missing_identity_blocks_strategy_not_guessed(self) -> None:
        self.map_path = self.write_map(instrument_map(IDS[:-1]))
        package = self.export()
        manifest = json.loads((package / "manifest.json").read_text())
        for name in tp.STRATEGIES:
            self.assertEqual(manifest["strategies"][name]["status"], tp.STATUS_BLOCKED_IDENTITY)
        self.assertIn("SYNTH24", manifest["strategies"]["equal_weight"]["error"])

    def test_weights_above_cap_or_sum_above_one_fail(self) -> None:
        doc = signal()
        doc["strategies"]["equal_weight"]["weights"] = {sid: 4.0 for sid in IDS}  # percent units
        doc["strategies"]["min_variance"]["weights"] = {sid: 0.05 for sid in IDS}  # sums to 1.25
        manifest = json.loads((self.export(doc) / "manifest.json").read_text())
        self.assertEqual(manifest["strategies"]["equal_weight"]["status"], tp.STATUS_FAILED)
        self.assertIn("no renormalization", manifest["strategies"]["min_variance"]["error"])

    def test_invalid_signal_results(self) -> None:
        cases = [
            signal(information_cutoff_utc="2026-08-31T21:00:00+00:00"),
            signal(signal_computed_at_utc="2026-08-31T19:00:00+00:00"),
            signal(cycle_key="2026q2"),
            signal(base_currency="EUR"),
            signal(synthetic_fixture="no"),
        ]
        extra = signal()
        extra["strategies"]["fallback"] = extra["strategies"]["equal_weight"]
        nan = signal()
        nan["strategies"]["equal_weight"]["weights"]["SYNTH00"] = float("nan")
        boolean = signal()
        boolean["strategies"]["equal_weight"]["weights"]["SYNTH00"] = True
        fallback = signal()
        fallback["strategies"]["mean_variance"] = {"status": "FAILED", "error": "x",
                                                   "weights": fallback["strategies"]["equal_weight"]["weights"]}
        for doc in cases + [extra, nan, boolean, fallback]:
            with self.subTest(doc=str(doc)[:80]), self.assertRaises(tp.TargetPortfolioError):
                self.export(doc)

    def test_instrument_map_requires_verified_evidence(self) -> None:
        bad_isin = instrument_map()
        bad_isin["instruments"][0]["isin"] = "XXSYN0000001"
        if tp.isin_is_valid(bad_isin["instruments"][0]["isin"]):
            bad_isin["instruments"][0]["isin"] = "XXSYN0000002"
        unverified = instrument_map()
        unverified["instruments"][0]["verification_status"] = "TICKER_MATCH"
        no_evidence = instrument_map()
        no_evidence["instruments"][0]["evidence"] = []
        duplicate = instrument_map()
        duplicate["instruments"][1]["isin"] = duplicate["instruments"][0]["isin"]
        future = instrument_map(frozen_at_utc="2027-01-01T00:00:00+00:00")
        for index, doc in enumerate([bad_isin, unverified, no_evidence, duplicate, future]):
            self.map_path = self.write_map(doc, f"bad{index}.json")
            with self.subTest(index=index), self.assertRaises(tp.TargetPortfolioError):
                self.export()

    def test_atomic_publish_leaves_nothing_on_failure(self) -> None:
        real = tp._fsync_write
        calls = []

        def failing(path, data):
            calls.append(path)
            if len(calls) == 3:
                raise OSError("disk full (simulated)")
            real(path, data)

        with mock.patch.object(tp, "_fsync_write", failing), self.assertRaises(OSError):
            self.export()
        self.assertEqual(list(self.out.iterdir()), [])


class IntegrityTests(Workspace):
    def setUp(self) -> None:
        super().setUp()
        self.package = self.export()

    def assertRejected(self, pattern: str) -> None:
        with self.assertRaisesRegex(tp.TargetPortfolioError, pattern):
            tp.inspect_package(self.package, now=IN_WINDOW, **ACCEPT)

    def test_hash_mismatch(self) -> None:
        path = self.package / "equal_weight.csv"
        path.write_bytes(path.read_bytes().replace(b"0.040000000000", b"0.040000000001", 1))
        self.assertRejected("Hash/size mismatch")

    def test_consistent_rehash_still_detected_by_allocation_id(self) -> None:
        text = (self.package / "equal_weight.csv").read_text()
        rows = text.splitlines()
        rows[1] = rows[1].replace("0.040000000000", "0.050000000000")
        rows[-1] = rows[-1].replace("0.000000000000", "-0.010000000000")
        self.rewrite(self.package, "equal_weight.csv", ("\n".join(rows) + "\n").encode())
        self.assertRejected("nonnegative")
        rows[-1] = rows[-1].replace("-0.010000000000", "0.000000000000")
        rows[2] = rows[2].replace("0.040000000000", "0.030000000000")
        self.rewrite(self.package, "equal_weight.csv", ("\n".join(rows) + "\n").encode())
        self.assertRejected("allocation_id does not match")

    def test_listing_shows_invalid_and_skips_temporary(self) -> None:
        root = self.package.parent
        (root / ".tmp-unpublished").mkdir()
        listed = tp.list_packages(root, now=IN_WINDOW, **ACCEPT)
        self.assertEqual([item["integrity"] for item in listed], ["VERIFIED"])
        self.assertEqual(listed[0]["paper_replication"], "NOT_EXACT_PAPER_REPLICATION")
        (self.package / "orders.csv").write_text("x")
        listed = tp.list_packages(root, now=IN_WINDOW, **ACCEPT)
        self.assertEqual((listed[0]["integrity"], listed[0]["handoff_allowed"]), ("INVALID", False))
        self.assertEqual(tp.list_packages(self.root / "missing"), [])

    def test_percent_unit_detected(self) -> None:
        buffer = io.StringIO()
        reader = list(csv.reader(io.StringIO((self.package / "equal_weight.csv").read_text())))
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(reader[0])
        for row in reader[1:]:
            row[-1] = f"{Decimal(row[-1]) * 100:.12f}"
            writer.writerow(row)
        self.rewrite(self.package, "equal_weight.csv", buffer.getvalue().encode())
        self.assertRejected("exceeds position cap")

    def test_extra_missing_and_symlinked_files(self) -> None:
        (self.package / "orders.csv").write_text("x")
        self.assertRejected("extra=\\['orders.csv'\\]")
        (self.package / "orders.csv").unlink()
        (self.package / "mean_variance.csv").rename(self.root / "moved.csv")
        self.assertRejected("missing=\\['mean_variance.csv'\\]")
        os.symlink(self.root / "moved.csv", self.package / "mean_variance.csv")
        self.assertRejected("not a regular file")

    def test_path_traversal_in_manifest(self) -> None:
        manifest = json.loads((self.package / "manifest.json").read_text())
        manifest["files"]["../map.json"] = manifest["files"].pop("signal_result.json")
        (self.package / "manifest.json").write_text(json.dumps(manifest))
        (self.package / "signal_result.json").unlink()
        self.assertRejected("differ from manifest|Unsafe")
        with self.assertRaisesRegex(tp.TargetPortfolioError, "Unsafe"):
            tp._read_member(self.package, "../map.json")

    def test_manipulated_kind_or_timing(self) -> None:
        for key, value, pattern in (("valid_until_utc", "2027-01-01T00:00:00+00:00", "schedule"),
                                    ("package_kind", "LIVE", "package_kind")):
            manifest = json.loads((self.package / "manifest.json").read_text())
            original = manifest[key]
            manifest[key] = value
            (self.package / "manifest.json").write_text(json.dumps(manifest))
            self.assertRejected(pattern)
            manifest[key] = original
            (self.package / "manifest.json").write_text(json.dumps(manifest))

    def test_unpublished_temp_directory_rejected(self) -> None:
        hidden = self.out / ".tmp-x"
        self.package.rename(hidden)
        with self.assertRaisesRegex(tp.TargetPortfolioError, "temporary"):
            tp.inspect_package(hidden, now=IN_WINDOW, **ACCEPT)


class LedgerTests(Workspace):
    def test_validation_does_not_claim_and_duplicate_claim_fails(self) -> None:
        package = self.export()
        ledger = tp.ProcessedLedger(self.root / "ledger")
        for _ in range(2):
            tp.load_target_portfolio(package, "equal_weight", now=IN_WINDOW, ledger=ledger, **ACCEPT)
        self.assertFalse((self.root / "ledger").exists())
        tp.claim_target_portfolio(package, "equal_weight", ledger, consumer="synthetic-test", now=IN_WINDOW, **ACCEPT)
        with self.assertRaisesRegex(tp.TargetPortfolioError, "already processed"):
            tp.load_target_portfolio(package, "equal_weight", now=IN_WINDOW, ledger=ledger, **ACCEPT)
        reissued = self.export(out=self.root / "reissue")
        with self.assertRaisesRegex(tp.TargetPortfolioError, "already processed"):
            tp.claim_target_portfolio(reissued, "equal_weight", ledger, consumer="x", now=IN_WINDOW, **ACCEPT)
        report = tp.inspect_package(package, now=IN_WINDOW, ledger=ledger, **ACCEPT)
        self.assertIn("allocation", " ".join(report.handoff_blockers))
        tp.load_target_portfolio(package, "min_variance", now=IN_WINDOW, ledger=ledger, **ACCEPT)
        with self.assertRaises(tp.AlreadyClaimed):
            ledger.claim(report.allocation_id, "equal_weight", consumer="x", run_id="r")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path, skip=lambda name: False) -> dict:
    return {p.relative_to(root).as_posix(): file_sha(p) for p in root.rglob("*")
            if p.is_file() and not skip(p.relative_to(root).as_posix())}


def write_frozen_inputs(directory: Path, scope: str | None) -> None:
    """SYNTHETIC frozen snapshot + code copy, hashed the way the executors hash them."""
    (directory / "snapshot").mkdir(parents=True)
    (directory / "snapshot" / "panel.csv").write_text("synthetic\n")
    (directory / "snapshot" / "bundle_manifest.json").write_text(json.dumps({
        "schema_version": 1, "review_status": "SOURCE_REVIEWED", **({"scope": scope} if scope else {}),
        "files": {"panel.csv": file_sha(directory / "snapshot" / "panel.csv")}}))
    (directory / "code" / "pipeline").mkdir(parents=True)
    (directory / "code" / "pipeline" / "synthetic.py").write_text("# synthetic\n")
    (directory / "code_manifest.json").write_text(json.dumps(tree_hashes(directory / "code")))
    (directory / "rule_config.json").write_text(json.dumps(rule_manifest()))


class HistoryAdapterTests(Workspace):
    """SYNTHETIC imitation of a paper_history_execution.execute run root (primary/, replay/, summary)."""

    def write_history_run(self, **summary) -> Path:
        run = self.root / f"history_run_{len(list(self.root.iterdir()))}"
        for name in ("primary", "replay"):
            directory = run / name
            write_frozen_inputs(directory, None)
            (directory / f"{CYCLE}.json").write_text(json.dumps({
                "status": "OPTIMIZATION_FAILED", "optimizer_errors": {"min_var": "OPTIMIZATION_FAILED: synthetic"},
                "sector_models": [{"selected": {sid: 0.01 for sid in IDS}}]}))
            for internal in ("equal_weight", "mean_var"):
                weights = {"equal_weight": {sid: 0.04 for sid in IDS},
                           "mean_var": signal()["strategies"]["mean_variance"]["weights"]}[internal]
                (directory / internal / "gross").mkdir(parents=True)
                (directory / internal / "gross" / "target_weights.csv").write_text(
                    "date," + ",".join(IDS + ["SYNTHOLD"]) + "\n2026-09-01,"
                    + ",".join(str(weights[s]) for s in IDS) + ",0\n")
        self.seal_history(run, **summary)
        return run

    @staticmethod
    def seal_history(run: Path, **summary) -> None:
        (run / "execution_summary.json").write_text(json.dumps({
            "status": "COMPLETED_WITH_OPTIMIZER_ERRORS", "model_replay_identical": True,
            "network_attempts": [], "inputs_changed": [], "historical_validation": "NOT_VALIDATED",
            "artifact_hashes": tree_hashes(run / "primary", lambda n: n.startswith("snapshot/")), **summary}))

    def test_history_executor_cycle_is_historical_only(self) -> None:
        run = self.write_history_run()
        doc = tp.signal_result_from_history_execution(run, CYCLE)
        self.assertTrue(doc["historical_only"])
        self.assertEqual(doc["signal_computed_at_source"], "file_mtime")
        self.assertEqual(doc["strategies"]["min_variance"]["status"], "FAILED")
        package = self.export(doc, mode=tp.MODE_HISTORICAL)
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual(manifest["package_kind"], tp.KIND_HISTORICAL)
        self.assertEqual(manifest["signal_computed_at_source"], "file_mtime")
        self.assertTrue(any("not a historical decision time" in r for r in manifest["restrictions"]))
        self.assertEqual(manifest["strategies"]["mean_variance"]["status"], tp.STATUS_COMPLETE)
        regular = self.export(doc, out=self.root / "regular")
        self.assertEqual(json.loads((regular / "manifest.json").read_text())["package_kind"], tp.KIND_DIAGNOSTIC)

    def test_tampered_history_runs_rejected(self) -> None:
        cases = {
            "primary quarter": (lambda run: (run / "primary" / f"{CYCLE}.json").write_text("{}"), "primary: hash mismatch"),
            "replay weights": (lambda run: (run / "replay" / "mean_var/gross/target_weights.csv").write_text("x"),
                               "replay: hash mismatch"),
            "primary rules": (lambda run: (run / "primary" / "rule_config.json").write_text("{}"), "hash mismatch"),
            "extra file": (lambda run: (run / "primary" / "orders.csv").write_text("x"), "unrecorded"),
            "replay snapshot": (lambda run: (run / "replay" / "snapshot" / "panel.csv").write_text("y"),
                                "snapshots differ"),
            "snapshot vs bundle": (lambda run: [(run / d / "snapshot" / "panel.csv").write_text("y")
                                                for d in ("primary", "replay")], "bundle manifest"),
            "replay flag": (lambda run: self.seal_history(run, model_replay_identical=False), "independently replayed"),
            "summary status": (lambda run: self.seal_history(run, status="FAILED"), "not exportable"),
            "primary dir passed": (None, "pass the run root"),
        }
        for label, (mutate, pattern) in cases.items():
            run = self.write_history_run()
            target = run / "primary" if mutate is None else run
            if mutate is not None:
                mutate(run)
            with self.subTest(label), self.assertRaisesRegex(tp.TargetPortfolioError, pattern):
                tp.signal_result_from_history_execution(target, CYCLE)

    def test_unpinned_rules_not_exportable(self) -> None:
        run = self.write_history_run()
        for name in ("primary", "replay"):
            (run / name / "rule_config.json").write_text("{}")
        self.seal_history(run)
        with self.assertRaisesRegex(tp.TargetPortfolioError, "rule manifest of the producing run"):
            self.export(tp.signal_result_from_history_execution(run, CYCLE), mode=tp.MODE_HISTORICAL)

    def test_real_history_executor_output_on_synthetic_bundle(self) -> None:
        from pipeline.paper_history_execution import execute
        from tests.test_paper_history_execution import synthetic_bundle

        synthetic_bundle(self.root / "bundle")  # SYNTHETIC bundle from the history-execution tests
        run = self.root / "history_run"
        execute(self.root / "bundle", run, CYCLE, CYCLE)
        doc = tp.signal_result_from_history_execution(run, CYCLE)
        self.assertTrue(doc["historical_only"])
        self.assertEqual(doc["data_check_status"], "HISTORY_EXECUTOR_REPLAY_VERIFIED")
        ids = sorted({sid for item in doc["strategies"].values() if item["weights"] for sid in item["weights"]})
        self.map_path = self.write_map(instrument_map(ids), "synthetic_bundle_map.json")
        package = self.export(doc, mode=tp.MODE_HISTORICAL, at=datetime(2026, 10, 16, tzinfo=timezone.utc))
        self.assertEqual(tp.inspect_package(package, now=IN_WINDOW, **ACCEPT).manifest["package_kind"],
                         tp.KIND_HISTORICAL)
        with open(run / "replay" / f"{CYCLE}.json", "a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(tp.TargetPortfolioError, "replay: hash mismatch"):
            tp.signal_result_from_history_execution(run, CYCLE)


class SignalExecutionAdapterTests(Workspace):
    """SYNTHETIC imitation of paper_signal_execution.execute_signal output, sealed like the executor."""

    def write_run(self, summary: dict, mean_status: str = "ERROR", *, replay_signal: dict | None = None) -> Path:
        run = self.root / f"signal_run_{len(list(self.root.iterdir()))}"
        write_frozen_inputs(run, "SIGNAL")
        source = signal()["strategies"]
        results = {
            "equal_weight": {"status": "SUCCESS", "weights": source["equal_weight"]["weights"], "error": None},
            "min_variance": {"status": "SUCCESS", "weights": source["min_variance"]["weights"], "error": None},
            "mean_variance": ({"status": "ERROR", "weights": source["equal_weight"]["weights"],
                               "error": "Need 253 complete covariance levels (synthetic)"}
                              if mean_status == "ERROR" else
                              {"status": "SUCCESS", "weights": source["mean_variance"]["weights"], "error": None}),
        }
        primary = {"cycle_key": CYCLE, "information_cutoff": "2026-08-31T20:00:00+00:00",
                   "execution_at": "2026-09-01T20:00:00+00:00", "computed_at_utc": "2026-08-31T22:00:00+00:00",
                   "sector_models": [], "strategy_results": results, "execution_orders": False}
        for name, doc in (("primary", primary), ("replay", replay_signal or primary)):
            (run / name).mkdir()
            (run / name / "signal.json").write_text(json.dumps(doc))
        self.seal(run, summary)
        return run

    @staticmethod
    def seal(run: Path, summary: dict | None = None) -> None:
        if summary is not None:
            (run / "execution_summary.json").write_text(json.dumps({
                "synthetic_fixture": True, "status": "PARTIAL_STRATEGY_ERRORS", "cycle_key": CYCLE,
                "data_manifest": tree_hashes(run / "snapshot"),
                "code_manifest": json.loads((run / "code_manifest.json").read_text()), **summary}))
        (run / "artifact_manifest.json").unlink(missing_ok=True)
        (run / "artifact_manifest.json").write_text(json.dumps(tree_hashes(run)))

    def test_reviewed_reconstruction_is_historical_and_error_has_no_fallback(self) -> None:
        run = self.write_run({"model_replay_identical": True, "review_status": "SOURCE_REVIEWED"})
        doc = tp.signal_result_from_signal_execution(run)
        self.assertTrue(doc["historical_only"])
        self.assertEqual(doc["signal_computed_at_source"], "signal.json")
        self.assertEqual(doc["strategies"]["mean_variance"], {
            "status": "FAILED", "weights": None, "error": "Need 253 complete covariance levels (synthetic)"})
        package = self.export(doc)
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual(manifest["package_kind"], tp.KIND_DIAGNOSTIC)
        self.assertFalse((package / "mean_variance.csv").exists())

    def test_live_marker_and_nonempty_list_are_not_actual_provenance(self) -> None:
        claimed = self.write_run({"model_replay_identical": True,
                                  "input_provenance": tp.LIVE_PROVENANCE}, "SUCCESS")
        self.assertTrue(tp.signal_result_from_signal_execution(claimed)["historical_only"])
        live = self.write_run({"model_replay_identical": True, "input_provenance": tp.LIVE_PROVENANCE,
                               "input_observation_evidence": ["synthetic capture log"]}, "SUCCESS")
        doc = tp.signal_result_from_signal_execution(live)
        self.assertTrue(doc["historical_only"])
        self.assertEqual(json.loads((self.export(doc) / "manifest.json").read_text())["package_kind"],
                         tp.KIND_DIAGNOSTIC)
        for bad in ({"model_replay_identical": "yes"}, {"model_replay_identical": True, "status": "FAILED"},
                    {"model_replay_identical": True, "cycle_key": "2026Q3"}):
            with self.subTest(bad=bad), self.assertRaises(tp.TargetPortfolioError):
                tp.signal_result_from_signal_execution(self.write_run(bad))

    def test_replay_flag_checked_against_actual_signal_hashes(self) -> None:
        different = {"cycle_key": CYCLE, "strategy_results": {}, "execution_orders": False}
        for summary, replay_signal, pattern in (
                ({"model_replay_identical": False}, None, "contradicts"),
                ({"model_replay_identical": True}, different, "contradicts"),
                ({"model_replay_identical": False}, different, "differ; not exportable")):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(tp.TargetPortfolioError, pattern):
                tp.signal_result_from_signal_execution(self.write_run(summary, replay_signal=replay_signal))

    def test_tampered_signal_runs_rejected(self) -> None:
        ok = {"model_replay_identical": True}

        def symlink(run: Path) -> None:
            (run / "primary" / "link.json").symlink_to(run / "rule_config.json")

        def traversal(run: Path) -> None:
            hashes = json.loads((run / "artifact_manifest.json").read_text())
            hashes["../outside.json"] = "0" * 64
            (run / "artifact_manifest.json").write_text(json.dumps(hashes))

        def resealed_code(run: Path) -> None:
            (run / "code" / "pipeline" / "synthetic.py").write_text("# changed\n")
            self.seal(run)

        def resealed_snapshot(run: Path) -> None:
            (run / "snapshot" / "panel.csv").write_text("changed\n")
            self.seal(run)

        def resealed_both_signals(run: Path) -> None:
            for name in ("primary", "replay"):
                doc = json.loads((run / name / "signal.json").read_text())
                doc["execution_orders"] = True
                (run / name / "signal.json").write_text(json.dumps(doc))
            self.seal(run)

        cases = {
            "signal": (lambda run: (run / "primary" / "signal.json").write_text("{}"), "hash mismatch for primary/signal.json"),
            "rules": (lambda run: (run / "rule_config.json").write_text("{}"), "hash mismatch for rule_config.json"),
            "summary": (lambda run: (run / "execution_summary.json").write_text("{}"), "hash mismatch for execution_summary"),
            "extra": (lambda run: (run / "orders.csv").write_text("x"), "unrecorded"),
            "unsealed": (lambda run: (run / "artifact_manifest.json").unlink(), "artifact_manifest.json missing"),
            "symlink": (symlink, "Symlink"),
            "traversal": (traversal, "unsafe path"),
            "code": (resealed_code, "code_manifest does not match"),
            "snapshot": (resealed_snapshot, "data_manifest does not match"),
            "orders": (resealed_both_signals, "execution_orders"),
        }
        for label, (mutate, pattern) in cases.items():
            run = self.write_run(ok)
            mutate(run)
            with self.subTest(label), self.assertRaisesRegex(tp.TargetPortfolioError, pattern):
                tp.signal_result_from_signal_execution(run)

    def test_mtime_fallback_is_labelled_as_generation_time(self) -> None:
        run = self.write_run({"model_replay_identical": True})
        for name in ("primary", "replay"):
            doc = json.loads((run / name / "signal.json").read_text())
            del doc["computed_at_utc"]
            (run / name / "signal.json").write_text(json.dumps(doc))
        self.seal(run)
        doc = tp.signal_result_from_signal_execution(run)
        self.assertEqual(doc["signal_computed_at_source"], "file_mtime")
        self.assertTrue(any("not a historical decision time" in r for r in doc["restrictions"]))

    def test_real_execute_signal_output_on_synthetic_bundle(self) -> None:
        from pipeline.fixed_lag_research import write_json
        from pipeline.paper_signal_execution import FILES, execute_signal
        from pipeline.return_comparison import sha256
        from tests.test_paper_history_execution import synthetic_bundle

        bundle = self.root / "bundle"  # SYNTHETIC bundle from the history-execution tests
        synthetic_bundle(bundle)
        (bundle / "raw_prices.csv").unlink()
        (bundle / "corporate_actions.csv").unlink()
        definitions = json.loads((bundle / "definitions.json").read_text())
        for name in ("action_coverage", "raw_price_basis", "simulation_dividends"):
            del definitions[name]
        write_json(bundle / "definitions.json", definitions)
        write_json(bundle / "bundle_manifest.json", {
            "schema_version": 1, "review_status": "SOURCE_REVIEWED", "scope": "SIGNAL",
            "files": {name: sha256(bundle / name) for name in FILES}})
        execute_signal(bundle, self.root / "run", CYCLE)
        doc = tp.signal_result_from_signal_execution(self.root / "run")
        self.assertTrue(doc["historical_only"])
        self.assertTrue(doc["code_id"].startswith("sha256:") and doc["data_id"].startswith("sha256:"))
        self.assertEqual(doc["model_rules"]["status"], "NOT_VALIDATED")
        ids = sorted({sid for item in doc["strategies"].values() if item["weights"] for sid in item["weights"]})
        self.map_path = self.write_map(instrument_map(ids), "synthetic_bundle_map.json")
        package = self.export(doc, mode=tp.MODE_HISTORICAL, at=datetime(2026, 10, 16, tzinfo=timezone.utc))
        report = tp.inspect_package(package, now=IN_WINDOW, **ACCEPT)
        self.assertEqual(report.manifest["package_kind"], tp.KIND_HISTORICAL)
        self.assertEqual(report.manifest["model"]["rules_source"], "signal_result")
        for name in tp.STRATEGIES:
            self.assertEqual(report.strategies[name].status, tp.STATUS_COMPLETE, name)
        replay = self.root / "run" / "replay" / "signal.json"
        replay.write_text(replay.read_text().replace("SUCCESS", "SUCCES5", 1))
        with self.assertRaisesRegex(tp.TargetPortfolioError, "hash mismatch for replay/signal.json"):
            tp.signal_result_from_signal_execution(self.root / "run")


class CliTests(Workspace):
    def run_cli(self, *argv, at=IN_WINDOW) -> tuple[int, dict]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = paper_quarter.main(list(map(str, argv)), now=lambda: at)
        text = buffer.getvalue()
        return code, json.loads(text)

    def test_regular_research_validate_claim(self) -> None:
        signal_path = self.root / "signal.json"
        signal_path.write_text(json.dumps(signal()))
        common = ["--signal-result", signal_path, "--instrument-map", self.map_path, "--output", self.out]
        code, doc = self.run_cli("regular", *common, at=CUTOFF - timedelta(hours=1))
        self.assertEqual((code, doc["active_cycle"]), (2, None))
        code, doc = self.run_cli("regular", *common, at=EXECUTION + timedelta(hours=1))
        self.assertEqual(code, 2)
        code, doc = self.run_cli("regular", *common)
        self.assertEqual((code, doc["package_kind"]), (0, tp.KIND_CURRENT))
        package = doc["published"]
        code, doc = self.run_cli("validate", "--package", package)
        self.assertEqual(code, 3)
        self.assertIn("Synthetic fixture package", doc["handoff_blockers"])
        code, doc = self.run_cli("research", *common[:4], "--output", self.root / "replay")
        self.assertEqual((code, doc["package_kind"]), (0, tp.KIND_HISTORICAL))
        code, doc = self.run_cli("schedule", "--cycle", "2026Q2")
        self.assertEqual(doc["information_cutoff"]["europe_berlin"], "2026-08-31T22:00:00+02:00")
        code, doc = self.run_cli("claim", "--package", package, "--strategy", "equal_weight",
                                 "--ledger", self.root / "ledger", "--consumer", "x", "--accept-unvalidated-model")
        self.assertEqual(code, 3)  # synthetic fixtures are never claimable through the CLI
        self.assertFalse((self.root / "ledger").exists())


if __name__ == "__main__":
    unittest.main()
