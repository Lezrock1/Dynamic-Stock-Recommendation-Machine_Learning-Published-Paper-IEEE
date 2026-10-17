from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduce.py")
assert SPEC is not None and SPEC.loader is not None
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class ReproductionTests(unittest.TestCase):
    def test_commands_have_only_relative_data_paths(self) -> None:
        for action in ("test", "demo", "dashboard"):
            args = reproduce.command(action)
            self.assertFalse(any(Path(arg).is_absolute() for arg in args[1:]))
        self.assertIn("synthetic", reproduce.command("demo"))
        with self.assertRaises(ValueError):
            reproduce.command("live-orders")

    def test_main_runs_in_checkout_not_callers_working_directory(self) -> None:
        with patch.object(reproduce, "dependency_errors", return_value=[]), \
                patch.object(reproduce.sys, "argv", ["reproduce.py", "test"]), \
                patch.object(reproduce.subprocess, "run") as run:
            run.return_value.returncode = 7
            self.assertEqual(reproduce.main(), 7)
            self.assertEqual(run.call_args.kwargs["cwd"], ROOT)

    def test_dependency_mismatches_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pipeline").mkdir()
            (root / "pipeline/requirements.txt").write_text("numpy==1.0\n")
            with patch.object(reproduce, "version", return_value="2.0"):
                self.assertEqual(reproduce.dependency_errors(root),
                                 ["numpy: installed 2.0, required 1.0"])
