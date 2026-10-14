import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from pipeline import ixbrl_fallback as fb
from pipeline import sec_edgar as se

CACHE = Path(__file__).parents[1] / "cache" / "filings"
FILINGS = {  # cik -> (accession, document, report date, acceptance)
    40533: ("0000040533-26-000032", "gd-20260705.htm", "2026-07-05", "2026-07-22T12:00:00Z"),
    922621: ("0001628280-26-051079", "erie-20260630.htm", "2026-06-30", "2026-07-30T12:00:00Z"),
    1701605: ("0001701605-26-000023", "bkr-20260630.htm", "2026-06-30", "2026-07-22T12:00:00Z"),
    1067983: ("0001193125-26-341032", "brka-20260630.htm", "2026-06-30", "2026-08-01T12:00:00Z"),
    1841666: ("0001841666-26-000053", "apa-20260630.htm", "2026-06-30", "2026-08-06T12:00:00Z"),
    1176948: ("0001628280-26-054538", "ares-20260630.htm", "2026-06-30", "2026-08-06T12:00:00Z"),
}
CLASS = {"922621": "us-gaap:CommonClassAMember", "1701605": "us-gaap:CommonClassAMember",
         "1067983": "brka:EquivalentClassBMember"}
TAGS = {**{k: (v, False) for k, v in se.DURATION_TAGS.items()}, **{k: (v, True) for k, v in se.INSTANT_TAGS.items()}}


def fake_recent(cik, cutoff, count):
    accn, doc, report, accepted = FILINGS[int(cik)]
    row = {"form": "10-Q", "accessionNumber": accn, "reportDate": report, "filingDate": accepted[:10],
           "primaryDocument": doc, "accepted": pd.Timestamp(accepted)}
    return [row] if cutoff is None or row["accepted"] <= cutoff else []


@unittest.skipUnless((CACHE / "922621").exists(), "stored filings not present")
class StoredFilingFallback(unittest.TestCase):
    def entries(self, cik, as_of=None):
        cfg = {"enabled": True, "ciks": list(FILINGS), "recent_filings": 1, "traded_class_member": CLASS}
        with mock.patch.object(se, "IXBRL_FALLBACK", cfg), mock.patch.object(fb, "recent_filings", fake_recent):
            return fb.fallback_entries(cik, TAGS, as_of)

    def test_gd_revenue_comes_from_original_filing_with_provenance(self) -> None:
        e = self.entries(40533)
        revenue = e["Revenue"][0]
        self.assertAlmostEqual(revenue["val"], 14.094e9, delta=1e6)
        prov = revenue["_prov"]
        self.assertEqual(prov["accession"], "0000040533-26-000032")
        self.assertEqual(len(prov["document_sha256"]), 64)
        self.assertEqual(prov["period_end"], "2026-07-05")
        self.assertNotIn("EPS_Basic", e)  # GD publishes diluted EPS only; no guess

    def test_share_class_values_follow_configured_class(self) -> None:
        self.assertAlmostEqual(self.entries(922621)["EPS_Basic"][0]["val"], 3.87)
        self.assertEqual(self.entries(922621)["EPS_Basic"][0]["_prov"]["dimensions"],
                         [["us-gaap:StatementClassOfStockAxis", "us-gaap:CommonClassAMember"]])
        self.assertAlmostEqual(self.entries(1701605)["EPS_Basic"][0]["val"], 0.69)

    def test_ambiguous_or_dimension_only_revenue_stays_empty(self) -> None:
        self.assertNotIn("Revenue", self.entries(1841666))
        self.assertNotIn("Revenue", self.entries(1176948))

    def test_no_lookahead_before_acceptance(self) -> None:
        self.assertEqual(self.entries(40533, as_of="2026-07-21T00:00:00Z"), {})
        self.assertIn("Revenue", self.entries(40533, as_of="2026-07-23T00:00:00Z"))

    def test_disabled_without_registration(self) -> None:
        with mock.patch.object(se, "IXBRL_FALLBACK", {}):
            self.assertEqual(fb.fallback_entries(40533, TAGS), {})


if __name__ == "__main__":
    unittest.main()
