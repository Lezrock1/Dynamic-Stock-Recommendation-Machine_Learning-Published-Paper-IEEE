from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pipeline import dashboard_view, q2_research


def _submissions(*rows: tuple[str, str, str]) -> dict:
    fields = ("form", "reportDate", "accessionNumber")
    values = [dict(zip(fields, row)) for row in rows]
    recent = {
        "form": [row["form"] for row in values],
        "reportDate": [row["reportDate"] for row in values],
        "accessionNumber": [row["accessionNumber"] for row in values],
        "filingDate": [row["reportDate"] for row in values],
        "acceptanceDateTime": [row["reportDate"] + "T20:00:00.000Z" for row in values],
        "primaryDocument": ["report.htm"] * len(values),
    }
    return {"filings": {"recent": recent}}


class Q2PeriodTests(unittest.TestCase):
    def test_observed_52_week_report_period_is_used(self) -> None:
        submissions = _submissions(("10-Q", "2026-05-09", "q2-52-week"))
        self.assertEqual(
            q2_research._q2_report(submissions),
            {
                "period_end": "2026-05-09",
                "form": "10-Q",
                "derivation": "observed_report_in_q2_window",
            },
        )

    def test_q3_report_never_substitutes_for_missing_q2_report(self) -> None:
        submissions = _submissions(
            ("10-Q", "2026-09-30", "q3"),
            ("10-Q", "2026-03-31", "q1"),
            ("10-Q", "2025-12-31", "q4"),
            ("10-K", "2025-09-30", "fy"),
            ("10-Q", "2025-06-30", "prior-q2"),
        )
        expected = q2_research._q2_report(submissions)
        self.assertEqual(expected["period_end"], "2026-06-30")
        self.assertEqual(expected["form"], "10-Q")
        self.assertIsNone(q2_research._latest_q2_filing(submissions, expected))

    def test_sector_uses_a_current_ticker_after_historical_only_tickers(self) -> None:
        self.assertEqual(
            q2_research._sector_for_model_tickers(["OLD", "NEW"], {"NEW": 35}),
            35,
        )
        self.assertIsNone(q2_research._sector_for_model_tickers(["OLD"], {"NEW": 35}))


class DashboardQuarterTests(unittest.TestCase):
    def test_stock_highlight_uses_selected_strategy_weights(self) -> None:
        html = dashboard_view.HTML
        self.assertIn("Object.entries(selectedWeights()||{})", html)
        self.assertIn("canonical(ticker)===canonical(symbol)&&weight>1e-9", html)
        self.assertIn('class="${isMlSelected(r.symbol)?', html)
        self.assertIn("#stock-table .ml-selected .symbol{color:#187343}", html)
        self.assertIn("Grün = im ausgewählten ML-Lauf", html)

    def _write_run(self, root: Path, run_id: str, cycle_key: str) -> None:
        run_dir = root / "state" / "diagnostics" / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run_header.json").write_text(
            json.dumps({"cycle_key": cycle_key, "status": "COMPLETE_OFFLINE"}),
            encoding="utf-8",
        )
        model_path = run_dir / "model_run_t50_diagnostic.json"
        model_path.write_text(
            json.dumps({
                "target_weights": {"equal_weight_v1": {}},
                "forecasts": {},
                "data_snapshot": {},
            }),
            encoding="utf-8",
        )
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        (run_dir / "artifact_manifest.json").write_text(
            json.dumps({"artifacts": {model_path.name: {"sha256": digest}}}),
            encoding="utf-8",
        )

    def test_dashboard_can_select_an_archived_quarter(self) -> None:
        self.assertIn('id="quarter"', dashboard_view.HTML)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "newer-q3", "2026Q3")
            self._write_run(root, "older-q2", "2026Q2")
            selected = dashboard_view.load_research(root, "2026Q2")
            latest = dashboard_view.load_research(root)
        self.assertEqual(selected["cycle_key"], "2026Q2")
        self.assertEqual(selected["available_quarters"], ["2026Q3", "2026Q2"])
        self.assertEqual(latest["cycle_key"], "2026Q3")
        self.assertFalse(selected["executable"])


if __name__ == "__main__":
    unittest.main()
