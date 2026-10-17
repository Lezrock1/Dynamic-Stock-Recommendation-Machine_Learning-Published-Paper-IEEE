import unittest

from pipeline.filing_case_review import disposition


class FilingCaseTests(unittest.TestCase):
    cutoff = "2026-08-31T20:00:00Z"
    expected = {"period_end": "2026-06-30", "filing_form": "10-Q"}
    identity = {"ticker": "TEST", "cik": 1}

    def filing(self, acceptance):
        return {"reportDate": "2026-06-30", "form": "10-Q", "accessionNumber": "acc",
                "acceptanceDateTime": acceptance}

    def test_cache_absence_is_acquisition_not_nonpublication(self):
        result = disposition(self.identity, [self.filing("2026-08-01T12:00:00Z")],
                             self.expected, self.cutoff)
        self.assertEqual(result["category"], "ACQUISITION_REQUIRED")
        self.assertIsNone(result["historical_exclusion_status"])

    def test_late_filing_requires_rule_and_earlier_release_review(self):
        result = disposition(self.identity, [self.filing("2026-08-31T22:21:23Z")],
                             self.expected, self.cutoff)
        self.assertEqual(result["category"], "LATE_EXPECTED_FILING_RULE_REVIEW")
        self.assertIsNone(result["historical_exclusion_status"])

    def test_no_record_does_not_prove_late_publication(self):
        result = disposition(self.identity, [], self.expected, self.cutoff)
        self.assertEqual(result["category"], "TARGETED_REVIEW_REQUIRED")

    def test_acquired_identity_gap_is_targeted_review(self):
        identity = {**self.identity, "accession": "acc", "source_file": "original.htm",
                    "status": "MISSING_OR_AMBIGUOUS_COVER_IDENTITY"}
        result = disposition(identity, [self.filing("2026-08-01T12:00:00Z")],
                             self.expected, self.cutoff)
        self.assertEqual(result["category"], "TARGETED_REVIEW_REQUIRED")
        self.assertIsNone(result["historical_exclusion_status"])

    def test_verified_identity_not_full_feature_readiness(self):
        identity = {**self.identity, "accession": "acc", "source_file": "original.htm",
                    "status": "ORIGINAL_COVER_TICKER_ENTITY_MATCH"}
        result = disposition(identity, [self.filing(self.cutoff)], self.expected, self.cutoff)
        self.assertEqual(result["category"], "FILING_IDENTITY_EVIDENCED")
        self.assertNotIn("READY", result.values())

    def test_wrong_period_is_not_matching_report(self):
        filing = {**self.filing("2026-08-01T12:00:00Z"), "reportDate": "2026-03-31"}
        result = disposition(self.identity, [filing], self.expected, self.cutoff)
        self.assertEqual(result["category"], "TARGETED_REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
