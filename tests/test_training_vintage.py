from __future__ import annotations

import unittest

import pandas as pd

from pipeline.paper_fixed_lag import schedule, window_frames
from pipeline.training_vintage import AT_FIT, STRICT, feature_limits


class TrainingVintageTests(unittest.TestCase):
    def panel(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "cycle_key": str(p), "y_return": .01, "reconstruction_status": "VERIFIED",
            "feature_available_at": schedule(str(p))["information_cutoff"],
            "label_known_at": schedule(str(p + 1))["execution_at"],
        } for p in pd.period_range("2015Q1", "2025Q4", freq="Q")])

    def test_versions_available_at_fit_are_not_required_to_exist_in_2015(self) -> None:
        data = self.panel()
        data.loc[:8, "feature_available_at"] = "2018-06-06T14:30:57Z"
        strict_train, _ = window_frames(data, "2026Q2")
        revised_train, validation = window_frames(data, "2026Q2", training_vintage_policy=AT_FIT)
        self.assertEqual(len(strict_train), 31)
        self.assertEqual(len(revised_train), 40)
        self.assertEqual(validation.cycle_key.tolist(), ["2025Q1", "2025Q2", "2025Q3", "2025Q4"])

    def test_fit_time_cannot_admit_post_cutoff_revision(self) -> None:
        data = self.panel()
        data.loc[0, "feature_available_at"] = "2026-09-01T00:00:00Z"
        train, _ = window_frames(data, "2026Q2", training_vintage_policy=AT_FIT)
        self.assertNotIn("2015Q1", train.cycle_key.tolist())

    def test_validation_and_prediction_keep_their_own_cutoffs(self) -> None:
        cycles = pd.Series(["2015Q1", "2024Q4", "2025Q1", "2025Q4", "2026Q2"])
        limits = feature_limits(cycles, "2026Q2", AT_FIT)
        self.assertEqual(limits.iloc[0], pd.Timestamp(schedule("2026Q2")["information_cutoff"]))
        self.assertEqual(limits.iloc[1], limits.iloc[0])
        for index in (2, 3, 4):
            self.assertEqual(limits.iloc[index], pd.Timestamp(schedule(cycles.iloc[index])["information_cutoff"]))
        data = self.panel()
        data.loc[data.cycle_key.eq("2025Q1"), "feature_available_at"] = "2026-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "Missing one of four"):
            window_frames(data, "2026Q2", training_vintage_policy=AT_FIT)

    def test_policy_does_not_make_unverified_versions_verified(self) -> None:
        data = self.panel()
        data.loc[0, "reconstruction_status"] = "ORIGINAL_REFERENCE_ASSUMPTION"
        train, _ = window_frames(data, "2026Q2", training_vintage_policy=AT_FIT)
        self.assertNotIn("2015Q1", train.cycle_key.tolist())

    def test_strict_default_and_unknown_policy(self) -> None:
        cycles = pd.Series(["2015Q1"])
        self.assertTrue(feature_limits(cycles).equals(feature_limits(cycles, "2026Q2", STRICT)))
        with self.assertRaisesRegex(ValueError, "Unknown training"):
            feature_limits(cycles, "2026Q2", "allow_all")
        with self.assertRaisesRegex(ValueError, "explicit target"):
            feature_limits(cycles, policy=AT_FIT)
