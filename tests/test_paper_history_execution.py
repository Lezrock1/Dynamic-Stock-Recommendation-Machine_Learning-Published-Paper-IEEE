from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.fixed_lag_research import write_json
from pipeline.market_calendar import nyse_sessions
from pipeline.paper_fixed_lag import schedule
from pipeline.paper_history_execution import (
    REQUIRED_FILES, execute, publish_execution, validate_panel, verify_bundle,
)
from pipeline.return_comparison import compare, load_registry, sha256
from pipeline.run_model import PAPER_FEATURES


def synthetic_bundle(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Constructed engine fixture, never a real-data or historical-validity claim."""
    root.mkdir()
    target = pd.Period("2026Q2", "Q")
    periods = list(pd.period_range(target - 21, target - 2, freq="Q")) + [target]
    securities = [f"SYNTH{i:03}" for i in range(100)]
    rows, memberships, lineage = [], [], {}
    for period in periods:
        timing = schedule(str(period))
        for i, security in enumerate(securities):
            rows.append({
                "security_id": security, "cycle_key": str(period), "gsector": 45,
                "feature_available_at": timing["information_cutoff"],
                "reconstruction_status": "VERIFIED",
                "y_return": (i + 1) / 1000 if period != target else np.nan,
                "label_known_at": schedule(str(period + 1))["execution_at"] if period != target else None,
                "label_source_ref": "SYNTHETIC_TEST_ONLY",
                "reconstructed_at": "2026-10-01T12:00:00Z", "acquired_at": "2026-10-01T12:00:00Z",
                "period_end": period.end_time.date().isoformat(),
                "expected_period": period.end_time.date().isoformat(),
                "period_rule": "calendar-quarter-v2", "filing_form": "10-Q",
                "facts_ready_at": None, **{f: (i + 1) * (j + 1) / 100 for j, f in enumerate(PAPER_FEATURES)},
            })
            memberships.append({"security_id": security, "cycle_key": str(period), "gsector": 45})
            lineage[f"{period}:{security}"] = {f: [{
                "accession": "SYNTHETIC_TEST_ONLY", "value_version": "test-v1",
                "source_url": "https://example.org/synthetic-fixture",
                "published_at": timing["information_cutoff"],
            }] for f in PAPER_FEATURES}
    panel, members = pd.DataFrame(rows), pd.DataFrame(memberships)
    panel.to_csv(root / "panel.csv", index=False)
    members.to_csv(root / "membership.csv", index=False)
    days = nyse_sessions("2025-07-01", "2026-09-10")
    market = pd.DataFrame({s: 100 * np.exp(np.arange(len(days)) * 0.0001 +
                                          np.sin(np.arange(len(days)) / (i + 4)) * 0.001)
                           for i, s in enumerate(securities)}, index=days)
    market.rename_axis("date").to_csv(root / "raw_prices.csv")
    market.rename_axis("date").to_csv(root / "covariance_levels.csv")
    pd.DataFrame(columns=["action_id", "ticker", "ex_date", "kind", "value", "pay_date"]).to_csv(
        root / "corporate_actions.csv", index=False)
    write_json(root / "lineage.json", lineage)
    write_json(root / "definitions.json", {
        "fixture_classification": "SYNTHETIC_TEST_ONLY_NOT_HISTORICAL_EVIDENCE",
        "currency": "USD", "raw_price_basis": "unadjusted_as_traded",
        "action_coverage": "splits_dividends_delistings_verified",
        "target_definition": "forward_execution_split_adjusted_price_log_return",
        "covariance_definition": "split_adjusted_price_log",
        "historical_membership": "quarter_end_verified", "historical_sector": "point_in_time_verified",
        "identity_scope": "security_share_class_time_versioned",
        "source_transition": "factor_definitions_overlap_reviewed",
        "simulation_dividends": "receivable_ex_date_cash_pay_date_until_rebalance",
        "sources": ["https://example.org/synthetic-fixture"],
    })
    write_json(root / "bundle_manifest.json", {
        "schema_version": 1, "review_status": "SOURCE_REVIEWED",
        "files": {name: sha256(root / name) for name in REQUIRED_FILES},
    })
    return panel, members, lineage


class HistoricalExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.bundle = cls.root / "bundle"
        cls.panel, cls.members, cls.lineage = synthetic_bundle(cls.bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_source_and_future_guards(self) -> None:
        verify_bundle(self.bundle)
        panel = self.panel.copy()
        panel.loc[0, "feature_available_at"] = "2026-10-01T12:00:00Z"
        with self.assertRaisesRegex(ValueError, "Future information"):
            validate_panel(panel, self.members, self.lineage)
        members = self.members.copy()
        members.loc[0, "gsector"] = 10
        with self.assertRaisesRegex(ValueError, "historical membership/sector"):
            validate_panel(self.panel, members, self.lineage)
        panel = self.panel.copy()
        panel.loc[0, "period_end"] = "2015-12-31"
        with self.assertRaisesRegex(ValueError, "Wrong fiscal report period"):
            validate_panel(panel, self.members, self.lineage)
        panel = self.panel.copy()
        panel.loc[0, "label_known_at"] = panel.loc[0, "feature_available_at"]
        with self.assertRaisesRegex(ValueError, "Label maturity"):
            validate_panel(panel, self.members, self.lineage)

    def test_membership_lookup_still_rejects_missing_duplicate_and_null_keys(self) -> None:
        cases = [
            (self.members.iloc[1:].copy(), "No matching historical membership/sector"),
            (pd.concat([self.members, self.members.iloc[:1]], ignore_index=True),
             "Duplicate historical membership row"),
            (self.members.assign(security_id=self.members.security_id.mask(
                self.members.index == 0)), "No matching historical membership/sector"),
        ]
        for members, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    validate_panel(self.panel, members, self.lineage)

    def test_validation_does_not_reuse_membership_or_publication_across_calls(self) -> None:
        validate_panel(self.panel, self.members, self.lineage)
        members = self.members.copy()
        members.loc[0, "gsector"] = 10
        with self.assertRaisesRegex(ValueError, "No matching historical membership/sector"):
            validate_panel(self.panel, members, self.lineage)
        lineage = json.loads(json.dumps(self.lineage))
        row = self.panel.iloc[0]
        key = f"{row.cycle_key}:{row.security_id}"
        lineage[key][PAPER_FEATURES[0]][0]["published_at"] = "2026-10-01T12:00:00Z"
        with self.assertRaisesRegex(ValueError, "Feature value published after declared availability"):
            validate_panel(self.panel, self.members, lineage)
        validate_panel(self.panel, self.members, self.lineage)

    def test_full_synthetic_engine_and_independent_replay_and_import(self) -> None:
        output = self.root / "synthetic-run"
        result = execute(self.bundle, output, "2026Q2", "2026Q2")
        self.assertEqual(result["status"], "COMPLETED_NOT_VALIDATED")
        self.assertEqual(len(result["simulated_strategies"]), 3)
        self.assertEqual(result["network_attempts"], [])
        self.assertTrue(result["model_replay_identical"])
        primary = output / "primary"
        quarter = json.loads((primary / "2026Q2.json").read_text())
        model = quarter["sector_models"][0]
        self.assertEqual(len(model["training_periods"]), 16)
        self.assertEqual(len(model["validation_periods"]), 4)
        self.assertEqual(len(model["forecasts"]), 100)
        self.assertEqual(len(model["selected"]), 20)
        self.assertEqual(len(model["mse"]), 5)
        gross = pd.read_csv(primary / "equal_weight/gross/comparison.csv")
        net = pd.read_csv(primary / "equal_weight/net/comparison.csv")
        self.assertEqual(gross["date"].iloc[0], "2026-08-31")
        self.assertEqual(gross["value"].iloc[0], net["value"].iloc[0])
        self.assertLess(net["value"].iloc[1], gross["value"].iloc[1])
        weights = json.loads((primary / "equal_weight/net/weight_history.json").read_text())
        self.assertEqual(weights[0]["positions"], [])
        self.assertEqual(len(weights[1]["positions"]), 100)
        self.assertTrue(all(p["benchmark_weight"] is None for p in weights[1]["positions"]))
        published = publish_execution(self.root, output)
        self.assertEqual(len(published), 6)
        registry = load_registry(self.root)
        self.assertEqual(registry["errors"], [])
        comparison = compare(registry["instruments"], "2026-08-31", "2026-09-10")
        self.assertEqual(len(comparison["series"]), 6)
        self.assertTrue(all(s["historical_strategy_validation"] == "NOT_VALIDATED"
                            for s in registry["instruments"]))

    def test_failed_run_records_explicit_status(self) -> None:
        output = self.root / "failed-run"
        with self.assertRaises(FileNotFoundError):
            execute(self.root / "not-present", output, "2026Q2", "2026Q2")
        status = json.loads((output / "execution_summary.json").read_text())
        self.assertEqual(status["status"], "FAILED")
        self.assertFalse(status["model_replay_identical"])

    def test_seven_reviewed_rule_exclusions_do_not_require_all_members_ready(self) -> None:
        bundle = self.root / "seven-exclusions-bundle"
        shutil.copytree(self.bundle, bundle)
        members = self.members.copy()
        extras = pd.DataFrame([{
            "security_id": f"LATE{i}", "cycle_key": "2026Q2", "gsector": 45,
            "exclusion_status": "NOT_PUBLIC_BY_CUTOFF",
            "exclusion_reason": "SYNTHETIC reviewed paper-rule nonpublication exclusion",
        } for i in range(7)])
        pd.concat([members, extras], ignore_index=True).to_csv(bundle / "membership.csv", index=False)
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        manifest["files"]["membership.csv"] = sha256(bundle / "membership.csv")
        write_json(bundle / "bundle_manifest.json", manifest)
        output = self.root / "seven-exclusions-run"
        result = execute(bundle, output, "2026Q2", "2026Q2")
        self.assertEqual(result["status"], "COMPLETED_NOT_VALIDATED")
        self.assertTrue(result["model_replay_identical"])
        excluded = pd.read_csv(output / "primary/2026Q2_source_exclusions.csv")
        self.assertEqual(len(excluded), 7)
        quarter = json.loads((output / "primary/2026Q2.json").read_text())
        self.assertEqual(len(quarter["sector_models"][0]["forecasts"]), 100)
        members = pd.read_csv(bundle / "membership.csv")
        members.loc[members.security_id == "LATE0", "exclusion_status"] = "ACQUISITION_REQUIRED"
        members.to_csv(bundle / "membership.csv", index=False)
        manifest["files"]["membership.csv"] = sha256(bundle / "membership.csv")
        write_json(bundle / "bundle_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Unresolved source gaps cannot be historical exclusions"):
            execute(bundle, self.root / "cache-gap-not-excluded-run", "2026Q2", "2026Q2")

    def test_never_extends_last_selection_past_next_required_rebalance(self) -> None:
        bundle = self.root / "long-market-bundle"
        shutil.copytree(self.bundle, bundle)
        market = pd.read_csv(bundle / "raw_prices.csv", index_col="date", parse_dates=True)
        market = market.reindex(nyse_sessions(market.index[0], "2026-12-10"))
        market.loc[market.index > "2026-09-10"] = 110.0
        market.to_csv(bundle / "raw_prices.csv", index_label="date")
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        manifest["files"]["raw_prices.csv"] = sha256(bundle / "raw_prices.csv")
        write_json(bundle / "bundle_manifest.json", manifest)

        class LaterClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2027, 1, 1, tzinfo=timezone.utc)

        output = self.root / "bounded-horizon"
        with patch("pipeline.paper_history_execution.datetime", LaterClock):
            execute(bundle, output, "2026Q2", "2026Q2")
        daily = pd.read_csv(output / "primary/equal_weight/net/comparison.csv")
        self.assertEqual(daily["date"].iloc[-1], "2026-12-01")
        definition = json.loads((output / "primary/equal_weight/net/definition.json").read_text())
        self.assertEqual(definition["period"], "COMPLETED_HOLDING_PERIOD")


if __name__ == "__main__":
    unittest.main()
