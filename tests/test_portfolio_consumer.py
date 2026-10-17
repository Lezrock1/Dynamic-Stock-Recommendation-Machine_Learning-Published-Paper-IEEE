"""Consumer tests use only synthetic packages and an in-memory execution simulator."""
from __future__ import annotations

import shutil
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest import mock

from pipeline import portfolio_consumer as pc
from pipeline import target_portfolio as tp
from tests.test_target_portfolio import IN_WINDOW, instrument_map, signal


class SimulatedCrash(BaseException):
    pass


class Simulator:
    supports_query = True
    supports_idempotency = True

    def __init__(self):
        self.orders = {}
        self.positions = {}
        self.fills = []
        self.events = []
        self.crash = False
        self.hide = False
        self.cancel_immediately = True

    def snapshot(self, account, strategy):
        self.events.append("snapshot")
        orders = () if self.hide else tuple(
            o for o in self.orders.values() if o.status in {"accepted", "partially_filled"})
        return pc.ExecutionSnapshot(account, strategy, dict(self.positions), orders, tuple(self.fills))

    def lookup(self, account, strategy, key):
        self.events.append("lookup")
        return None if self.hide else self.orders.get(key)

    def submit(self, account, strategy, key, intent):
        self.events.append("submit")
        if key not in self.orders:
            self.orders[key] = pc.OrderReport(key, f"sim-{len(self.orders)}", "accepted")
        if self.crash:
            raise SimulatedCrash()
        return self.orders[key]

    def cancel(self, account, strategy, order_id):
        self.events.append("cancel")
        if self.cancel_immediately:
            for key, order in self.orders.items():
                if order.order_id == order_id:
                    self.orders[key] = replace(order, status="cancelled")

    def fill(self, key, quantity=Decimal("2"), *, complete=True):
        order = self.orders[key]
        self.orders[key] = replace(order, status="filled" if complete else "partially_filled",
                                   filled_quantity=quantity)
        self.positions["SYNTH00"] = quantity
        self.fills = [pc.ConfirmedFill("fill-1", key, order.order_id, quantity)]


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        # Project-local fixtures only: never system temporary directories.
        self.root = Path(__file__).resolve().parent / (".consumer-fixture-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        map_path = self.root / "map.json"
        map_path.write_text(pc._json(instrument_map()), encoding="utf-8")
        self.package = tp.export_package(signal(), map_path, self.root / "packages",
                                         mode=tp.MODE_REGULAR, now=lambda: IN_WINDOW)
        self.ledger = pc.ConsumerLedger(self.root / "consumer.sqlite")
        self.provider = Simulator()
        self.planner = mock.Mock(return_value=[pc.OrderIntent("SYNTH00", "buy", Decimal("2"))])
        self.account = "SIM-DEPOT-1"
        self.strategy = "equal_weight"

    def accept(self, **kwargs):
        return self.ledger.accept_package(self.package, self.strategy, self.account,
                                          now=IN_WINDOW, research_simulation=True, **kwargs)

    def process(self, record):
        return self.ledger.process(self.account, self.strategy, record["allocation_id"],
                                   self.provider, self.planner)

    def replacement(self):
        map_path = self.root / "map.json"
        self.package = tp.export_package(signal(code_id="replacement-code"), map_path,
                                         self.root / "replacement-packages", mode=tp.MODE_REGULAR,
                                         now=lambda: IN_WINDOW)

    def test_default_rejects_unvalidated_model_and_simulation_is_explicit(self):
        with self.assertRaises(tp.TargetPortfolioError):
            self.ledger.accept_package(self.package, self.strategy, self.account, now=IN_WINDOW)
        record = self.accept()
        self.assertEqual(record["status"], "accepted")
        self.assertTrue(record["research_simulation"])
        self.assertTrue(record["handoff_blockers"])
        self.assertEqual(self.provider.events, [])

    def test_duplicate_accept_resume_partial_and_complete_do_not_execute_twice(self):
        record = self.accept()
        self.assertEqual(self.accept(), record)
        record = self.process(record)
        self.assertEqual(record["status"], "in_progress")
        key = record["orders"][0]["client_key"]
        self.provider.fill(key, Decimal("1"), complete=False)
        self.ledger = pc.ConsumerLedger(self.ledger.path)
        record = self.process(record)
        self.assertEqual(record["status"], "partially_executed")
        self.provider.fill(key)
        record = self.process(record)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(self.process(record), record)
        self.assertEqual(self.provider.events.count("submit"), 1)
        self.planner.assert_called_once()
        events = self.ledger.audit(self.account, self.strategy, record["allocation_id"])
        self.assertIn("reconciled_snapshot", [e["event"] for e in events])
        self.assertEqual([e["sequence"] for e in events], sorted(e["sequence"] for e in events))

    def test_crash_after_external_acceptance_queries_before_any_resubmit(self):
        record = self.accept()
        self.provider.crash = True
        with self.assertRaises(SimulatedCrash):
            self.process(record)
        persisted = self.ledger.get(self.account, self.strategy, record["allocation_id"])
        self.assertEqual(persisted["orders"][0]["status"], "submitting")
        self.ledger = pc.ConsumerLedger(self.ledger.path)
        self.provider.crash = False
        self.provider.events.clear()
        resumed = self.process(record)
        self.assertEqual(resumed["status"], "in_progress")
        self.assertEqual(self.provider.events, ["snapshot", "lookup"])
        self.assertEqual(len(self.provider.orders), 1)
        self.planner.assert_called_once()

    def test_ambiguous_crash_blocks_without_query_or_idempotency(self):
        record = self.accept()
        self.provider.crash = True
        with self.assertRaises(SimulatedCrash):
            self.process(record)
        self.provider.crash = False
        self.provider.supports_query = self.provider.supports_idempotency = False
        self.provider.hide = True
        self.provider.events.clear()
        blocked = self.process(record)
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("Ambiguous", blocked["blocked_reason"])
        self.assertEqual(self.provider.events, ["snapshot"])
        self.process(record)
        self.assertNotIn("submit", self.provider.events)
        self.provider.hide = False
        self.assertEqual(self.process(record)["status"], "in_progress")

    def test_idempotent_retry_only_after_fresh_snapshot(self):
        record = self.accept()
        self.provider.crash = True
        with self.assertRaises(SimulatedCrash):
            self.process(record)
        self.provider.crash = False
        self.provider.supports_query = False
        self.provider.hide = True
        self.provider.events.clear()
        self.process(record)
        self.assertEqual(self.provider.events, ["snapshot", "submit"])
        self.assertEqual(len(self.provider.orders), 1)

    def test_confirmed_fill_recovers_crash_without_resubmit_or_query(self):
        record = self.accept()
        self.provider.crash = True
        with self.assertRaises(SimulatedCrash):
            self.process(record)
        self.provider.fill(next(iter(self.provider.orders)))
        self.provider.supports_query = self.provider.supports_idempotency = False
        self.provider.events.clear()
        self.assertEqual(self.process(record)["status"], "completed")
        self.assertEqual(self.provider.events, ["snapshot"])

    def test_authoritative_nonacceptance_allows_retry_after_query(self):
        record = self.accept()
        self.provider.crash = True
        with self.assertRaises(SimulatedCrash):
            self.process(record)
        self.provider.orders.clear()  # authoritative simulator evidence of no external acceptance
        self.provider.crash = False
        self.provider.supports_idempotency = False
        self.provider.events.clear()
        self.process(record)
        self.assertEqual(self.provider.events, ["snapshot", "lookup", "submit"])

    def test_replacement_requires_reason_cancels_residuals_then_plans_from_live_state(self):
        old = self.process(self.accept())
        self.provider.fill(old["orders"][0]["client_key"], Decimal("1"), complete=False)
        self.replacement()
        with self.assertRaisesRegex(pc.ConsumerError, "explicit replacement"):
            self.accept()
        with self.assertRaisesRegex(pc.ConsumerError, "explicit replacement"):
            self.accept(replaces=old["allocation_id"], replacement_reason=" ")
        new = self.accept(replaces=old["allocation_id"], replacement_reason="Corrected allocation")
        with self.assertRaisesRegex(pc.ConsumerError, "Superseded"):
            self.process(old)
        self.provider.cancel_immediately = False
        blocked = self.process(new)
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(self.provider.events.count("submit"), 1)
        self.provider.cancel_immediately = True
        self.provider.events.clear()
        new = self.process(new)
        self.assertLess(self.provider.events.index("cancel"), self.provider.events.index("submit"))
        self.assertEqual(self.planner.call_args.args[1].positions, {"SYNTH00": Decimal("1")})
        self.assertEqual(self.ledger.get(self.account, self.strategy, old["allocation_id"])["status"],
                         "replaced")
        self.assertEqual(new["replaces"], old["allocation_id"])
        audit = self.ledger.audit(self.account, self.strategy, old["allocation_id"])
        self.assertEqual(audit[-1]["details"]["new"], new["allocation_id"])

    def test_replacement_does_not_resubmit_unknown_old_order(self):
        old = self.accept()
        self.provider.crash = True
        with self.assertRaises(SimulatedCrash):
            self.process(old)
        self.replacement()
        new = self.accept(replaces=old["allocation_id"], replacement_reason="Correction")
        self.provider.supports_query = self.provider.supports_idempotency = False
        self.provider.hide = True
        self.provider.events.clear()
        self.assertEqual(self.process(new)["status"], "blocked")
        self.assertEqual(self.provider.events, ["snapshot"])

    def test_bindings_separate_accounts_and_strategies_and_reject_wrong_resume(self):
        record = self.accept()
        other = self.ledger.accept_package(self.package, self.strategy, "SIM-DEPOT-2",
                                            now=IN_WINDOW, research_simulation=True)
        self.assertEqual(other["status"], "accepted")
        with self.assertRaises(pc.ConsumerError):
            self.ledger.process(self.account, "mean_variance", record["allocation_id"],
                                self.provider, self.planner)
        self.assertEqual(self.provider.events, [])

    def test_untracked_orders_prevent_new_plan_and_wrong_snapshot_fails(self):
        record = self.accept()
        self.provider.orders["foreign"] = pc.OrderReport("foreign", "manual-order", "accepted")
        self.assertEqual(self.process(record)["status"], "blocked")
        self.planner.assert_not_called()
        self.provider.snapshot = lambda a, s: pc.ExecutionSnapshot("WRONG-DEPOT", s, {})
        with self.assertRaisesRegex(pc.ConsumerError, "mismatch"):
            self.process(record)

    def test_invalid_plan_and_regressing_report_never_silently_accepted(self):
        record = self.accept()
        self.planner.return_value = [pc.OrderIntent("SYNTH00", "buy", Decimal("NaN"))]
        with self.assertRaises(pc.ConsumerError):
            self.process(record)
        self.assertNotIn("submit", self.provider.events)
        self.planner.return_value = [pc.OrderIntent("SYNTH00", "buy", Decimal("2"))]
        record = self.process(record)
        key = record["orders"][0]["client_key"]
        self.provider.fill(key, Decimal("1"), complete=False)
        self.process(record)
        self.provider.orders[key] = replace(self.provider.orders[key], status="accepted",
                                            filled_quantity=Decimal(0))
        with self.assertRaises(pc.ConsumerError):
            self.process(record)

    def test_parallel_consumers_share_durable_intent_and_submit_once(self):
        record = self.accept()
        other = pc.ConsumerLedger(self.ledger.path)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(ledger.process, self.account, self.strategy,
                                       record["allocation_id"], self.provider, self.planner)
                       for ledger in (self.ledger, other)]
            self.assertTrue(all(f.result()["status"] == "in_progress" for f in futures))
        self.assertEqual(self.provider.events.count("submit"), 1)
        self.planner.assert_called_once()

    def test_rejected_intent_blocks_remaining_batch(self):
        record = self.accept()
        self.planner.return_value = [pc.OrderIntent("SYNTH00", "buy", Decimal("2")),
                                     pc.OrderIntent("SYNTH01", "buy", Decimal("2"))]
        original = self.provider.submit

        def reject(*args):
            report = replace(original(*args), status="rejected")
            self.provider.orders[report.client_key] = report
            return report

        self.provider.submit = reject
        self.assertEqual(self.process(record)["status"], "blocked")
        self.assertEqual(self.provider.events.count("submit"), 1)
        self.assertEqual(self.process(record)["status"], "blocked")
        self.assertEqual(self.provider.events.count("submit"), 1)

    def test_planner_cannot_substitute_securities_or_invent_sell_positions(self):
        record = self.accept()
        for intent in [pc.OrderIntent("UNRELATED", "buy", Decimal("1")),
                       pc.OrderIntent("SYNTH00", "sell", Decimal("1"))]:
            with self.subTest(intent=intent), self.assertRaises(pc.ConsumerError):
                self.planner.return_value = [intent]
                self.process(record)
        self.assertNotIn("submit", self.provider.events)

    def test_replacement_cannot_discard_idempotent_but_unqueryable_uncertainty(self):
        old = self.accept()
        self.provider.crash = True
        with self.assertRaises(SimulatedCrash):
            self.process(old)
        self.replacement()
        new = self.accept(replaces=old["allocation_id"], replacement_reason="Correction")
        self.provider.supports_query = False
        self.provider.hide = True
        self.provider.events.clear()
        self.assertEqual(self.process(new)["status"], "blocked")
        self.assertEqual(self.provider.events, ["snapshot"])

    def test_nested_replacement_cannot_skip_unresolved_predecessor(self):
        old = self.process(self.accept())
        self.replacement()
        new = self.accept(replaces=old["allocation_id"], replacement_reason="Correction")
        self.package = tp.export_package(signal(code_id="third-code"), self.root / "map.json",
                                         self.root / "third-packages", mode=tp.MODE_REGULAR,
                                         now=lambda: IN_WINDOW)
        with self.assertRaisesRegex(pc.ConsumerError, "outstanding replacement"):
            self.accept(replaces=new["allocation_id"], replacement_reason="Another correction")

    def test_crash_between_retry_authorization_and_resubmit_preserves_uncertainty(self):
        old = self.accept()
        self.provider.crash = True
        with self.assertRaises(SimulatedCrash):
            self.process(old)
        self.provider.supports_query = False
        self.provider.hide = True
        original = self.ledger._save

        def crash_after_commit(db, record, event, *args, **kwargs):
            original(db, record, event, *args, **kwargs)
            if event == "retry_authorized_after_reconciliation":
                raise SimulatedCrash()

        with mock.patch.object(self.ledger, "_save", side_effect=crash_after_commit):
            with self.assertRaises(SimulatedCrash):
                self.process(old)
        self.ledger = pc.ConsumerLedger(self.ledger.path)
        self.replacement()
        new = self.accept(replaces=old["allocation_id"], replacement_reason="Correction")
        self.provider.events.clear()
        self.assertEqual(self.process(new)["status"], "blocked")
        self.assertEqual(self.provider.events, ["snapshot"])

    def test_empty_plan_completes_without_invented_quantities(self):
        record = self.accept()
        self.planner.return_value = []
        self.assertEqual(self.process(record)["status"], "completed")
        target, snapshot = self.planner.call_args.args
        self.assertEqual(snapshot.positions, {})
        self.assertNotIn("reference_price", target["positions"][0])
        self.assertNotIn("submit", self.provider.events)


if __name__ == "__main__":
    unittest.main()
