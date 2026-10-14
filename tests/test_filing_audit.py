import unittest
from pathlib import Path

from pipeline import filing_audit as fa
from pipeline import ixbrl

CACHE = Path(__file__).parents[1] / "cache" / "filings"


@unittest.skipUnless((CACHE / "922621").exists(), "stored filings not present")
class StoredFilingAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {c[0]: c for c in fa.CASES}

    def facts(self, ticker):
        _t, cik, accession, document, *_ = self.cases[ticker]
        return ixbrl.filing_facts(cik, accession, document)

    def classify(self, ticker, feature):
        _t, _c, _a, _d, period_end, days = self.cases[ticker]
        tags, instant, keyword = fa.FEATURES[feature]
        return fa.classify(self.facts(ticker), tags, instant, keyword, period_end, days)

    def test_erie_revenue_is_consolidated_but_eps_exists_only_per_share_class(self) -> None:
        self.assertEqual(self.classify("ERIE", "Revenue")["classification"], "PUBLISHED_UNDIMENSIONED")
        eps = self.classify("ERIE", "EPS_Basic")
        self.assertEqual(eps["classification"], "CLASS_DIMENSION_ONLY")
        self.assertFalse(eps["fallback_candidate"])
        self.assertGreaterEqual(len({f["dimensions"] for f in eps["evidence"]}), 2)

    def test_apa_total_revenue_is_not_a_standard_undimensioned_fact(self) -> None:
        self.assertEqual(self.classify("APA", "Revenue")["classification"], "OTHER_DIMENSION_ONLY")

    def test_conflicting_revenue_definitions_are_not_merged(self) -> None:
        for ticker in ("ARES", "BRK-B"):
            row = self.classify(ticker, "Revenue")
            self.assertEqual(row["classification"], "AMBIGUOUS_DEFINITIONS")
            self.assertFalse(row["fallback_candidate"])

    def test_evidence_carries_provenance(self) -> None:
        fact = self.classify("FDXF", "Revenue")["evidence"][0]
        for key in ("tag", "unit", "period_start", "period_end", "context_id", "accession", "document_sha256"):
            self.assertTrue(fact[key])


if __name__ == "__main__":
    unittest.main()
