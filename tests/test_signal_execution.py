import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fixed_lag_research import write_json
from pipeline.market_source_evidence import parse_chart
from pipeline.paper_signal_execution import FILES, execute_signal, strategy_results
from pipeline.return_comparison import sha256
from pipeline.training_vintage import AT_FIT
from pipeline.spy_primary_comparison import reconstruct
from tests.test_paper_history_execution import synthetic_bundle
from tests import test_target_portfolio as export_fixtures
from pipeline.target_portfolio import (
    export_package, inspect_package, signal_result_from_signal_execution, MODE_HISTORICAL,
)


class SignalStageTests(unittest.TestCase):
    def test_explicit_fit_time_policy_reaches_validation_and_fitting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            panel, _, lineage = synthetic_bundle(bundle)
            first = panel.cycle_key.iloc[0]
            publication = "2026-08-31T20:00:00Z"
            panel.loc[panel.cycle_key.eq(first), "feature_available_at"] = publication
            for key, features in lineage.items():
                if key.startswith(first + ":"):
                    for parts in features.values():
                        for part in parts:
                            part["published_at"] = publication
            panel.to_csv(bundle / "panel.csv", index=False)
            write_json(bundle / "lineage.json", lineage)
            for name in ("raw_prices.csv", "corporate_actions.csv"):
                (bundle / name).unlink()
            def seal():
                write_json(bundle / "bundle_manifest.json", {
                    "schema_version": 1, "review_status": "SOURCE_REVIEWED", "scope": "SIGNAL",
                    "files": {name: sha256(bundle / name) for name in FILES}})
            seal()
            with self.assertRaisesRegex(ValueError, "Future information"):
                execute_signal(bundle, root / "strict", "2026Q2")
            definitions = json.loads((bundle / "definitions.json").read_text())
            definitions["training_vintage_policy"] = AT_FIT
            write_json(bundle / "definitions.json", definitions)
            seal()
            result = execute_signal(bundle, root / "at-fit", "2026Q2")
            self.assertEqual(result["status"], "COMPLETED_NOT_VALIDATED")
            self.assertTrue(result["model_replay_identical"])
            signal = json.loads((root / "at-fit/primary/signal.json").read_text())
            self.assertEqual(signal["training_vintage_policy"], AT_FIT)
            self.assertEqual(len(signal["sector_models"][0]["training_periods"]), 16)

    def test_missing_covariance_never_blocks_equal_weight_or_invents_optimizer_results(self):
        selected = pd.Series(0.01, index=[f"S{i}" for i in range(25)])
        result = strategy_results(selected, pd.DataFrame(), "2026-08-31T20:00:00Z")
        self.assertEqual(result["equal_weight"]["status"], "SUCCESS")
        self.assertAlmostEqual(sum(result["equal_weight"]["weights"].values()), 1)
        for name in ("min_variance", "mean_variance"):
            self.assertEqual(result[name]["status"], "ERROR")
            self.assertEqual(result[name]["weights"], {})
            self.assertIn("253 complete", result[name]["error"])

    def test_signal_replay_does_not_require_prices_cash_pay_dates_or_valuation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
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
            started_at = datetime.now(timezone.utc)
            result = execute_signal(bundle, root / "run", "2026Q2")
            self.assertLessEqual(started_at, datetime.fromisoformat(result["computed_at_utc"]))
            self.assertLessEqual(datetime.fromisoformat(result["computed_at_utc"]), datetime.now(timezone.utc))
            self.assertLessEqual(result["pass_completed_at_utc"]["primary"], result["pass_completed_at_utc"]["replay"])
            self.assertTrue(result["synthetic_fixture"])
            self.assertEqual(result["status"], "COMPLETED_NOT_VALIDATED")
            self.assertEqual(len(result["successful_strategies"]), 3)
            self.assertTrue(result["model_replay_identical"])
            self.assertEqual(result["valuation_status"], "NOT_REQUESTED_SIGNAL_ONLY")
            self.assertEqual(result["network_attempts"], [])
            adapted = signal_result_from_signal_execution(root / "run")
            self.assertTrue(adapted["historical_only"])
            identities = export_fixtures.instrument_map([f"SYNTH{i:03}" for i in range(100)])
            mapping = root / "instrument_map.json"
            write_json(mapping, identities)
            package = export_package(adapted, mapping, root / "packages", mode=MODE_HISTORICAL)
            imported = inspect_package(package)
            self.assertFalse(imported.as_dict()["handoff_allowed"])
            self.assertEqual(imported.manifest["model"]["optimization_version"],
                             adapted["model_rules"]["optimization_version"])
            for name in ("equal_weight", "min_variance", "mean_variance"):
                self.assertEqual(imported.strategies[name].status, "COMPLETE")
                self.assertTrue((package / f"{name}.csv").is_file())

    def test_historical_nonpublication_rule_still_does_not_allow_cache_exclusion(self):
        from pipeline.paper_signal_execution import fit_prediction_cycle
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            panel, members, _ = synthetic_bundle(root / "bundle")
            extra = pd.DataFrame([{
                "security_id": "MISSING", "cycle_key": "2026Q2", "gsector": 45,
                "exclusion_status": "ACQUISITION_REQUIRED", "exclusion_reason": "No cached original"}])
            with self.assertRaisesRegex(ValueError, "Unresolved source gaps"):
                fit_prediction_cycle(panel, pd.concat([members, extra]), "2026Q2", root)

    def test_source_empty_quotes_are_explicit_schema_error_not_key_error(self):
        with self.assertRaisesRegex(ValueError, "incomplete daily close"):
            parse_chart({"chart": {"result": [{"meta": {"symbol": "TEST", "currency": "USD"},
                                              "timestamp": [], "indicators": {"quote": [{}]}}]}}, "TEST")

    def test_original_benchmark_simple_return_uses_previous_value(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "fundamental_back_testing.ipynb").read_text())
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if "spx_return =" in "".join(c.get("source", [])))
        expression = next(line.split("=", 1)[1].strip() for line in source.splitlines()
                          if line.startswith("spx_return ="))
        values = pd.Series([100.0, 110.0, 99.0])
        result = eval(expression, {"__builtins__": {}}, {"spx": values})
        self.assertAlmostEqual(result.iloc[1], 0.1)
        self.assertAlmostEqual(result.iloc[2], -0.1)

    def test_primary_nav_non_session_filter_is_explicit_and_never_fills(self):
        nav = pd.Series([100.0, 100.0, 102.0],
                        index=pd.to_datetime(["2014-04-17", "2014-04-18", "2014-04-21"]))
        events = pd.DataFrame({"EX-DATE": pd.to_datetime([]), "cash_distribution": []})
        with self.assertRaisesRegex(ValueError, "non-sessions"):
            reconstruct(nav, events, "2014-04-17")
        points, review = reconstruct(nav, events, "2014-04-17", nyse_only=True)
        self.assertEqual(len(points), 2)
        self.assertEqual(review["excluded_non_nyse_source_valuations"], {"2014-04-18": 100.0})
        self.assertAlmostEqual(points.value.iloc[-1], 102)
        events = pd.DataFrame({"EX-DATE": pd.to_datetime(["2014-04-18"]), "cash_distribution": [1.0]})
        with self.assertRaisesRegex(ValueError, "distribution falls"):
            reconstruct(nav, events, "2014-04-17", nyse_only=True)
