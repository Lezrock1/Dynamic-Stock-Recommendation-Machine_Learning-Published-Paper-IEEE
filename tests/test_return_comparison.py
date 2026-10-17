from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.return_comparison import (
    compare, import_series, load_registry, search_instruments, valid_isin, validate_metadata,
)


def metadata() -> dict:
    return {
        "instrument_id": "fixture-v1", "name": "Synthetic test, not real performance",
        "instrument_type": "etf", "identifiers": {"wkn": ["A0TEST"], "ticker": ["TEST"],
                                                "isin": ["US0378331005"]},
        "share_class": "synthetic", "exchange": "test", "currency": "USD",
        "return_type": "etf_total_return", "dividend_tax_basis": "before_personal_tax",
        "reinvestment_rule": "ex_date", "cost_basis": "synthetic test only",
        "source": "constructed unit test", "source_url": "https://example.org/definition",
        "valuation_convention": "NYSE_close_UTC", "calendar": "NYSE",
        "review_status": "SOURCE_REVIEWED", "observed_at": "2026-10-01T12:00:00Z",
        "evidence_files": [{"path": "definition.txt", "sha256": hashlib.sha256(b"test").hexdigest()}],
    }


def instrument(identifier: str, values: list[float], start: int = 0) -> dict:
    days = ["2026-09-01", "2026-09-02", "2026-09-03"][start:]
    return {**metadata(), "instrument_id": identifier, "points": [
        {"date": d, "valued_at": d + "T20:00:00+00:00", "value": v}
        for d, v in zip(days, values, strict=True)
    ]}


class ReturnComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "definition.txt").write_text("test")
        self.csv_path = self.root / "levels.csv"
        self.meta_path = self.root / "metadata.json"
        self.meta_path.write_text(json.dumps(metadata()))
        with self.csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["date", "value", "valued_at"])
            writer.writerows([["2026-09-01", 100, "2026-09-01T20:00:00Z"],
                             ["2026-09-02", 101, "2026-09-02T20:00:00Z"],
                             ["2026-09-03", 99, "2026-09-03T20:00:00Z"]])

    def test_isin_checksum_and_no_heuristic_wkn_resolution(self) -> None:
        self.assertTrue(valid_isin("US0378331005"))
        self.assertFalse(valid_isin("US0378331006"))
        registry = {"instruments": [metadata(), {**metadata(), "instrument_id": "class-two"}]}
        self.assertEqual(len(search_instruments(registry, "A0TEST")), 2)
        self.assertEqual(search_instruments(registry, "A0TESX"), [])

    def test_csv_import_hash_and_definitions(self) -> None:
        imported = import_series(self.root, self.csv_path, self.meta_path)
        registry = load_registry(self.root)
        self.assertEqual(len(registry["instruments"]), 1)
        self.assertEqual(registry["instruments"][0]["return_type"], "etf_total_return")
        with self.assertRaises(FileExistsError):
            import_series(self.root, self.csv_path, self.meta_path)
        imported.write_text(imported.read_text() + " ")
        self.assertIn("hash mismatch", load_registry(self.root)["errors"][0])

    def test_undocumented_adjusted_close_and_price_index_rejected(self) -> None:
        for kind in ("price_return", "adj_close", "unknown"):
            doc = metadata()
            doc["return_type"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "not verified total returns"):
                validate_metadata(doc)

    def test_gap_is_not_zero_return_or_forward_filled(self) -> None:
        rows = self.csv_path.read_text().splitlines()
        self.csv_path.write_text("\n".join([rows[0], rows[1], rows[3]]) + "\n")
        with self.assertRaisesRegex(ValueError, "Missing/non-session"):
            import_series(self.root, self.csv_path, self.meta_path)

    def test_eur_rejected_without_historical_fx(self) -> None:
        doc = metadata()
        doc["currency"] = "EUR"
        with self.assertRaisesRegex(ValueError, "historical FX"):
            validate_metadata(doc)

    def test_source_evidence_hash_required(self) -> None:
        (self.root / "definition.txt").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Source evidence hash mismatch"):
            import_series(self.root, self.csv_path, self.meta_path)

    def test_declared_close_requires_actual_nyse_close(self) -> None:
        self.csv_path.write_text(self.csv_path.read_text().replace("T20:00:00Z", "T21:00:00Z"))
        with self.assertRaisesRegex(ValueError, "actual session closes"):
            import_series(self.root, self.csv_path, self.meta_path)

    def test_common_start_normalization_and_display_window_drawdown(self) -> None:
        a = instrument("a", [100, 110, 99])
        b = instrument("b", [200, 210], start=1)
        result = compare([a, b], "2026-09-01", "2026-09-03")
        self.assertEqual(result["start"], "2026-09-02")
        self.assertEqual(result["excluded_noncommon_dates"], 1)
        for series in result["series"]:
            self.assertEqual(series["points"][0]["index100"], 100)
        self.assertAlmostEqual(result["series"][0]["metrics"]["return"], -0.1)
        self.assertAlmostEqual(result["series"][0]["metrics"]["maximum_drawdown"], -0.1)

    def test_dividend_reinvestment_and_close_mismatch_rejected(self) -> None:
        a = instrument("a", [100, 101, 102])
        b = copy.deepcopy(a)
        b["reinvestment_rule"] = "pay_date"
        with self.assertRaisesRegex(ValueError, "reinvestment_rule"):
            compare([a, b], "2026-09-01", "2026-09-03")
        b = copy.deepcopy(a)
        b["points"][1]["valued_at"] = "2026-09-02T21:00:00Z"
        with self.assertRaisesRegex(ValueError, "valuation timestamps"):
            compare([a, b], "2026-09-01", "2026-09-03")

    def test_missing_registry_is_explicit(self) -> None:
        result = load_registry(self.root)
        self.assertEqual(result["instruments"], [])
        self.assertIn("no verified", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
