import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.publication_provenance import (
    OBSERVED, ORIGINAL, VERSION, verify_publication_evidence, verify_signal_availability,
)


class PublicationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def store(self, name, data):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        (self.root / name).write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    def original(self):
        html = b"""<html xmlns:ix="ix" xmlns:xbrli="xbrli"><body>
        <xbrli:context id="end"><xbrli:entity><xbrli:identifier>1234</xbrli:identifier></xbrli:entity>
        <xbrli:period><xbrli:instant>2026-06-30</xbrli:instant></xbrli:period></xbrli:context>
        <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
        <ix:nonfraction name="us-gaap:Assets" contextref="end" unitref="usd" scale="3">125</ix:nonfraction>
        </body></html>"""
        accession = "0000001234-26-000001"
        proof = {"accessionNumber": [accession], "acceptanceDateTime": ["2026-08-05T12:00:00Z"]}
        result = {
            **self.store("filing.htm", html), "evidence_id": "original-assets",
            "availability_class": ORIGINAL, "accession": accession, "cik": 1234,
            "published_at_utc": "2026-08-05T12:00:00Z", "retrieved_at_utc": "2026-10-01T12:00:00Z",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1234/000000123426000001/filing.htm",
            "publication": {**self.store("submission.json", proof),
                            "source_url": "https://data.sec.gov/submissions/CIK0000001234.json"},
            "facts": [{"tag": "us-gaap:Assets", "unit": "iso4217:USD", "value": 125000.,
                       "period_start": None, "period_end": "2026-06-30", "context_id": "end"}],
        }
        result["capture"] = self.store("original-receipt.json", {
            "source_url": result["source_url"], "sha256": result["sha256"],
            "acquired_at": result["retrieved_at_utc"], "http_status": 200})
        return result

    def verify(self, record):
        return verify_publication_evidence(
            self.root, {"format_version": VERSION, "records": [record]}, "2026-08-31T20:00:00Z")

    def test_late_download_of_exact_original_is_allowed(self):
        result = self.verify(self.original())
        self.assertEqual(result["classes"], [ORIGINAL])
        self.assertEqual(result["model_qualification"], "NOT_ESTABLISHED_BY_PUBLICATION_CHECK_ALONE")

    def test_later_values_wrong_accession_time_and_companyfacts_are_rejected(self):
        original = self.original()
        changes = [
            {"published_at_utc": "2026-09-01T12:00:00Z"},
            {"published_at_utc": "2026-08-04T12:00:00Z"},
            {"source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000001234.json"},
            {"accession": "0000001234-26-000002"},
            {"publication": {**original["publication"],
                             "source_url": "https://data.sec.gov/submissions/CIK0000009999.json"}},
        ]
        wrong_fact = copy.deepcopy(original)
        wrong_fact["facts"][0]["value"] = 126000
        for record in [{**original, **change} for change in changes] + [wrong_fact]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.verify(record)

    def test_observed_snapshot_requires_receipt_and_real_pre_cutoff_capture(self):
        source = self.store("snapshot.json", {"value": 125})
        record = {**source, "evidence_id": "snapshot", "availability_class": OBSERVED,
                  "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000001234.json",
                  "published_at_utc": "2026-08-05T12:00:00Z",
                  "retrieved_at_utc": "2026-08-31T19:00:00Z"}
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.verify(record)
        record["capture"] = self.store("receipt.json", {
            "source_url": record["source_url"], "sha256": source["sha256"],
            "acquired_at": record["retrieved_at_utc"], "http_status": 200})
        self.assertEqual(self.verify(record)["classes"], [OBSERVED])
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.verify({**record, "retrieved_at_utc": "2026-09-01T12:00:00Z"})
        (self.root / "snapshot.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.verify(record)

    def test_fact_proof_cannot_certify_unrelated_market_or_membership_data(self):
        doc = {"format_version": VERSION, "records": [self.original()],
               "artifact_bindings": {"membership.csv": "original-assets"}}
        self.store("publication_evidence.json", doc)
        with self.assertRaisesRegex(ValueError, "membership.csv"):
            verify_signal_availability(self.root, "2026-08-31T20:00:00Z")

    def test_symlink_and_traversal_proofs_rejected(self):
        record = self.original()
        for path in ("../escape", "/absolute"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.verify({**record, "path": path})
        (self.root / "alias.htm").symlink_to(self.root / "filing.htm")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.verify({**record, "path": "alias.htm"})

    def test_mixed_late_original_and_observed_inputs_have_actual_coverage(self):
        original = self.original()
        records = [original]
        bindings = {}
        for name in ("membership.csv", "covariance_levels.csv"):
            source = self.store(name, b"synthetic source table bytes")
            url = "https://example.org/" + name
            retrieved = "2026-08-31T19:00:00Z"
            records.append({
                **source, "evidence_id": name, "availability_class": OBSERVED,
                "source_url": url, "retrieved_at_utc": retrieved, "published_at_utc": retrieved,
                "capture": self.store(name + ".receipt.json", {
                    "source_url": url, "sha256": source["sha256"], "acquired_at": retrieved,
                    "http_status": 200}),
            })
            bindings[name] = name
        self.store("panel.csv", b"synthetic calculated features")
        lineage = {"2026Q2:SYNTH": {"SYNTHETIC_FEATURE": [{
            **original["facts"][0], "evidence_id": original["evidence_id"],
            "accession": original["accession"], "source_url": original["source_url"],
        }]}}
        self.store("lineage.json", lineage)
        self.store("publication_evidence.json", {
            "format_version": VERSION, "records": records, "artifact_bindings": bindings})
        result = verify_signal_availability(self.root, "2026-08-31T20:00:00Z")
        self.assertEqual(result["bound_feature_components"], 1)
        self.assertEqual(set(result["classes"]), {OBSERVED, ORIGINAL})
        self.assertEqual(result["availability_coverage"], "SOURCE_BYTES_AND_LINEAGE_VERIFIED")
        lineage["2015Q2:SYNTH"] = lineage.pop("2026Q2:SYNTH")
        self.store("lineage.json", lineage)
        with self.assertRaisesRegex(ValueError, "own feature cutoff"):
            verify_signal_availability(self.root, "2026-08-31T20:00:00Z")
