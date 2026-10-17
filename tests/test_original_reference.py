from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.original_reference import check_rows, load_originals, source_paths
from pipeline.run_model import PAPER_FEATURES


class OriginalReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = pd.DataFrame([{
            "tic": "A", "datadate": "2015-03-31", "tradedate": "2015-06-01",
            "gsector": 10, "y_return": -.2, **{f: float(i) for i, f in enumerate(PAPER_FEATURES)},
        }])
        self.rows = self.original.assign(
            data_source="WRDS_Compustat_original", security_id="A", cycle_key="2015Q1")
        self.source = {"sha256": "a" * 64}

    def test_byte_bound_original_row_and_target(self) -> None:
        result = check_rows(self.rows, self.original, self.source)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "ORIGINAL_REFERENCE_MATCHED_NOT_PIT_CERTIFIED")
        self.assertEqual(result[0]["original_excel_row"], 2)
        self.assertEqual(result[0]["checked_features"], 20)
        self.assertTrue(result[0]["checked_target"])

    def test_no_blanket_sec_assumption(self) -> None:
        self.rows["data_source"] = "SEC_free_reconstructed"
        self.assertEqual(check_rows(self.rows, self.original, self.source), [])

    def test_revised_value_or_target_rejected(self) -> None:
        for field in ("X2_EPS", "y_return"):
            with self.subTest(field=field):
                data = self.rows.copy()
                data.loc[0, field] += .001
                with self.assertRaisesRegex(ValueError, "content mismatch"):
                    check_rows(data, self.original, self.source)

    def test_missing_and_ambiguous_identity_rejected(self) -> None:
        for original in (self.original.iloc[:0], pd.concat([self.original, self.original], ignore_index=True)):
            with self.assertRaisesRegex(ValueError, "Missing or ambiguous"):
                check_rows(self.rows, original, self.source)

    def test_nan_and_csv_roundtrip_noise_not_imputed(self) -> None:
        self.original.loc[0, "X2_EPS"] = np.nan
        self.rows.loc[0, "X2_EPS"] = np.nan
        self.rows.loc[0, "y_return"] = np.nextafter(-.2, 0)
        self.assertEqual(len(check_rows(self.rows, self.original, self.source)), 1)
        self.rows.loc[0, "X2_EPS"] = 0
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            check_rows(self.rows, self.original, self.source)

    def test_absent_original_columns_remain_unknown_and_are_documented(self) -> None:
        frame = self.original.drop(columns=["X2_EPS"]).copy()
        frame["datadate"] = 20150331
        frame["tradedate"] = 20150601
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for path in source_paths(root):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original")
            with patch("pipeline.original_reference.subprocess.check_output", return_value=b"original"), \
                    patch("pipeline.original_reference.pd.read_excel", side_effect=lambda _: frame.copy()):
                originals, policy = load_originals(root)
            self.assertTrue(originals[10]["X2_EPS"].isna().all())
            self.assertEqual(policy["files"][0]["absent_original_feature_columns_preserved_as_nan"], ["X2_EPS"])
            with patch("pipeline.original_reference.subprocess.check_output", return_value=b"changed"):
                with self.assertRaisesRegex(ValueError, "Not the accepted"):
                    load_originals(root)
