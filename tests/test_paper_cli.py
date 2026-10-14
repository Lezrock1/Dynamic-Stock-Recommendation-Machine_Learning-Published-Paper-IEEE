from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from pipeline.paper_portfolio import run_cli
from pipeline.paper_store import PaperStore, StoreNotFoundError


class PaperPortfolioCliTests(unittest.TestCase):
    def test_initialization_is_repeatable_and_backup_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "nested" / "paper.sqlite3"
            output = io.StringIO()
            with redirect_stdout(output):
                run_cli(["--db", str(database), "init", "--initial-capital", "1000"])
                run_cli(["--db", str(database), "init", "--initial-capital", "1000"])
            store = PaperStore(database)
            for portfolio_id in ("equal_weight", "mean_var", "min_var"):
                self.assertEqual(store.cash_balance(portfolio_id), 1000.0)

            backup = root / "backups" / "paper-copy.sqlite3"
            with redirect_stdout(output):
                run_cli(["--db", str(database), "backup", "--output", str(backup)])
            self.assertEqual(
                PaperStore(backup).cash_balance("equal_weight"),
                1000.0,
            )

    def test_dashboard_and_status_refuse_to_initialize_missing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.sqlite3"
            with self.assertRaises(StoreNotFoundError):
                run_cli(["--db", str(missing), "status"])
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
