import json
import tempfile
import unittest
from pathlib import Path

from pipeline.paper_manual_run import run
from pipeline.paper_signal_execution import FILES
from pipeline.return_comparison import sha256
from tests.test_paper_history_execution import synthetic_bundle


class ManualRunTests(unittest.TestCase):
    def test_real_clock_refuses_historical_cycle_as_current_before_acquisition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"format_version": "manual-paper-source-plan-v1", "cycle_key": "2026Q2"}))
            with self.assertRaisesRegex(ValueError, "actual execution window"):
                run(plan, root / "out", root, cycle="2026Q2")
            self.assertFalse((root / "out").exists())

    def test_acquisition_cannot_promote_empty_evidence_to_real_signal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            synthetic_bundle(root / "sources")
            files = {}
            for name in FILES:
                path = root / "sources" / name
                files[name] = {"cache_path": str(path.relative_to(root)), "sha256": sha256(path)}
            for name, value in (
                    ("publication_evidence.json", {"format_version": "publication-provenance-v1", "records": []}),
                    ("instrument_map.json", {"format_version": "paper-instrument-map-v1", "instruments": []})):
                path = root / "sources" / name
                path.write_text(json.dumps(value))
                files[name] = {"cache_path": str(path.relative_to(root)), "sha256": sha256(path)}
            plan = root / "plan.json"
            plan.write_text(json.dumps({"format_version": "manual-paper-source-plan-v1",
                                        "cycle_key": "2026Q2", "files": files}))
            before = {name: sha256(root / spec["cache_path"]) for name, spec in files.items()}
            with self.assertRaisesRegex(ValueError, "Nonempty versioned"):
                run(plan, root / "out", root, cycle="2026Q2", historical=True)
            status = json.loads((root / "out/workflow.json").read_text())
            self.assertEqual(status["status"], "BLOCKED_BEFORE_MODEL")
            self.assertEqual(status["failed_stage"], "QUALIFYING")
            self.assertFalse(status["model_started"])
            self.assertFalse((root / "out/signal_result.json").exists())
            self.assertFalse((root / "out/packages").exists())
            self.assertEqual(len(json.loads((root / "out/acquisition/acquisition.json").read_text())), 7)
            self.assertEqual(before, {name: sha256(root / spec["cache_path"]) for name, spec in files.items()})

    def test_source_digest_change_persists_acquisition_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {name: {"cache_path": "missing", "sha256": "0" * 64}
                     for name in FILES | {"publication_evidence.json", "instrument_map.json"}}
            plan = root / "plan.json"
            plan.write_text(json.dumps({"format_version": "manual-paper-source-plan-v1",
                                        "cycle_key": "2026Q2", "files": files}))
            with self.assertRaises(OSError):
                run(plan, root / "out", root, cycle="2026Q2", historical=True)
            result = json.loads((root / "out/workflow.json").read_text())
            self.assertEqual(result["failed_stage"], "ACQUIRING")
            self.assertFalse(result["model_started"])
