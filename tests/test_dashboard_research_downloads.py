import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.dashboard_view import (
    diagnostic_file, load_numeric_diagnostic, load_target_packages, target_package_download,
)
from pipeline import target_portfolio as tp
from tests import test_target_portfolio as fixtures


class ResearchDownloadTests(unittest.TestCase):
    def test_package_listing_and_download_are_read_only_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapping = root / "map.json"
            mapping.write_text(json.dumps(fixtures.instrument_map()))
            package = tp.export_package(
                fixtures.signal(), mapping, root / "state/research/target_portfolios",
                mode=tp.MODE_HISTORICAL, now=lambda: fixtures.IN_WINDOW)
            entries = load_target_packages(root, "2026Q2")
            self.assertEqual(len(entries), 1)
            self.assertFalse(entries[0]["handoff_allowed"])
            self.assertIn("equal_weight.csv", entries[0]["downloads"])
            self.assertEqual(load_target_packages(root, "2026Q3"), [])
            payload = target_package_download(root, package.name, "equal_weight.csv")
            self.assertEqual(payload, (package / "equal_weight.csv").read_bytes())
            self.assertFalse(list(root.rglob("*.claim.json")))
            for key, name in (("../escape", "manifest.json"), (package.name, "../map.json"),
                              (package.name, "signal_result.json")):
                with self.assertRaises(ValueError):
                    target_package_download(root, key, name)
            (package / "equal_weight.csv").write_bytes(payload + b"tampered")
            with self.assertRaises(ValueError):
                target_package_download(root, package.name, "equal_weight.csv")
            damaged = load_target_packages(root, "2026Q2")[0]
            self.assertEqual(damaged["integrity"], "INVALID")
            self.assertEqual(damaged["downloads"], [])

    def test_diagnostic_pointer_and_contents_are_hash_checked_and_cycle_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "state/research/diagnostics/run"
            (directory / "primary").mkdir(parents=True)
            document = {"cycle_key": "2026Q2", "executable": False, "handoff_eligible": False,
                        "status": "NUMERICAL_DIAGNOSTIC_NOT_HISTORICAL_SIGNAL"}
            data = json.dumps(document).encode()
            (directory / "primary/diagnostic.json").write_bytes(data)
            manifest = directory / "artifact_manifest.json"
            manifest.write_text(json.dumps({"primary/diagnostic.json": hashlib.sha256(data).hexdigest()}))
            pointer = directory.parent / "latest.json"
            pointer.write_text(json.dumps({
                "path": str(directory.relative_to(root)),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))
            self.assertTrue(load_numeric_diagnostic(root, "2026Q2")["available"])
            self.assertFalse(load_numeric_diagnostic(root, "2026Q3")["available"])
            with self.assertRaises(ValueError):
                diagnostic_file(root, "../../secret")
            (directory / "primary/diagnostic.json").write_bytes(b"{}")
            self.assertIn("mismatch", load_numeric_diagnostic(root, "2026Q2")["error"])
            pointer.write_text(json.dumps({"path": "../escape", "sha256": "0" * 64}))
            self.assertFalse(load_numeric_diagnostic(root, None)["available"])
