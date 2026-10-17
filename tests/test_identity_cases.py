"""Regression cases grounded in the Q2 2026 original cover structures."""
import json
import unittest
from pathlib import Path

from pipeline.filing_identity_evidence import ROOT, cover_identity
from pipeline.identity_cases import referenced_exhibit_identity, release_evidence
from pipeline.smci_release_supplement import review


def context(key, dimensions="", cik=31791, end="2026-07-05"):
    return (f'<xbrli:context id="{key}"><xbrli:entity>'
            f'<xbrli:identifier>{cik}</xbrli:identifier>{dimensions}</xbrli:entity>'
            '<xbrli:period><xbrli:startdate>2025-12-29</xbrli:startdate>'
            f'<xbrli:enddate>{end}</xbrli:enddate></xbrli:period></xbrli:context>')


def tag(name, key, value):
    return f'<ix:nonNumeric name="dei:{name}" contextRef="{key}">{value}</ix:nonNumeric>'


class IdentityCasesTests(unittest.TestCase):
    def row_case(self, title_context=None):
        title_context = title_context or context("title")
        return (title_context + context("symbol", '<xbrldi:explicitMember '
                'dimension="revv:TradingSymbolAxis">revv:RVTYMember</xbrldi:explicitMember>')
                + context("exchange") + "<table><tr><td>"
                + tag("Security12bTitle", "title", "Common stock, $1 par value per share")
                + "</td><td>" + tag("TradingSymbol", "symbol", "RVTY")
                + "</td><td>" + tag("SecurityExchangeName", "exchange", "The New York Stock Exchange")
                + "</td></tr></table>").encode()

    def test_exact_row_across_contexts(self):
        result = cover_identity(self.row_case(), {}, "RVTY", 31791)
        self.assertEqual(result["status"], "ORIGINAL_COVER_TICKER_ENTITY_MATCH")
        self.assertEqual(result["equity_matches"][0]["bound_context_ids"],
                         ["title", "symbol", "exchange"])

    def test_different_entity_does_not_bind(self):
        self.assertFalse(cover_identity(self.row_case(context("title", cik=1)),
                                        {}, "RVTY", 31791)["equity_matches"])

    def test_different_period_does_not_bind(self):
        self.assertFalse(cover_identity(self.row_case(context("title", end="2025-07-05")),
                                        {}, "RVTY", 31791)["equity_matches"])

    def test_typed_dimension_does_not_bind(self):
        typed = '<xbrldi:typedMember dimension="x:Class"><x:Value>A</x:Value></xbrldi:typedMember>'
        self.assertFalse(cover_identity(self.row_case(context("title", typed)),
                                        {}, "RVTY", 31791)["equity_matches"])

    def test_separate_rows_do_not_bind(self):
        data = self.row_case().replace(b"</td><td>", b"</td></tr><tr><td>")
        self.assertFalse(cover_identity(data, {}, "RVTY", 31791)["equity_matches"])

    def test_distinct_classes_remain_ambiguous(self):
        data = ""
        for key in ["A", "B"]:
            dim = (f'<xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">'
                   f'x:Class{key}Member</xbrldi:explicitMember>')
            data += context(key, dim) + tag("TradingSymbol", key, "RVTY")
            data += tag("Security12bTitle", key, "Common stock")
        self.assertEqual(cover_identity(data.encode(), {}, "RVTY", 31791)["status"],
                         "MISSING_OR_AMBIGUOUS_COVER_IDENTITY")

    def test_mmm_exchange_only_dimension_is_not_a_new_share_class(self):
        dims = ('<xbrldi:explicitMember dimension="dei:EntityListingsExchangeAxis">'
                'exch:XNYS</xbrldi:explicitMember>')
        common = ('<xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">'
                  'us-gaap:CommonStockMember</xbrldi:explicitMember>')
        data = context("nyse", dims + common, cik=66740) + context("texas", dims, cik=66740)
        for key in ["nyse", "texas"]:
            data += tag("TradingSymbol", key, "MMM")
            data += tag("Security12bTitle", key, "Common Stock, Par Value $.01 Per Share")
        self.assertEqual(cover_identity(data.encode(), {}, "MMM", 66740)["status"],
                         "ORIGINAL_COVER_TICKER_ENTITY_MATCH")

    def test_ndaq_original_exchange_named_members_only(self):
        data = ""
        for key, member in [
            ("nasdaq", "CommonStock.01ParValuePerShareTheNasdaqStockMarketMember"),
            ("texas", "CommonStock.01ParValuePerShareNadasqTexasLLCMember"),
        ]:
            dimension = ('<xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">'
                         f'ndaq:{member}</xbrldi:explicitMember>')
            data += context(key, dimension, cik=1120193)
            data += tag("TradingSymbol", key, "NDAQ")
            data += tag("Security12bTitle", key, "Common Stock, $0.01 par value per share")
        self.assertEqual(cover_identity(data.encode(), {}, "NDAQ", 1120193)["status"],
                         "ORIGINAL_COVER_TICKER_ENTITY_MATCH")
        changed = data.replace("NadasqTexasLLCMember", "UnknownClassMember")
        self.assertEqual(cover_identity(changed.encode(), {}, "NDAQ", 1120193)["status"],
                         "MISSING_OR_AMBIGUOUS_COVER_IDENTITY")

    def test_cached_original_cases(self):
        source = ROOT / "state/research/acquisition/20261016-q2-identity-review-final/security_identities.json"
        if not source.exists():
            self.skipTest("Local research originals not distributed with code")
        rows = json.loads(source.read_text())
        for ticker in ["C", "MMM", "NDAQ", "RVTY"]:
            with self.subTest(ticker=ticker):
                row = next(r for r in rows if r["ticker"] == ticker)
                parsed = cover_identity((ROOT / row["source_file"]).read_bytes(), {}, ticker, row["cik"])
                if ticker == "C":
                    self.assertEqual(parsed["status"], "MISSING_OR_AMBIGUOUS_COVER_IDENTITY")
                    self.assertEqual(parsed["registration_exhibit_references"][0]["href"], "c-20260630_d2.htm")
                else:
                    self.assertEqual(parsed["status"], "ORIGINAL_COVER_TICKER_ENTITY_MATCH")
                    self.assertTrue(all("notes" not in title.lower()
                                        for m in parsed["equity_matches"] for title in m["security_titles"]))

    def test_smci_exact_original_release_periods_and_signs(self):
        source = ROOT / "state/research/acquisition/20261016-q2-primary-cases-01/sources/smci/fy2026-release.html"
        if not source.exists():
            self.skipTest("Local release original not distributed with code")
        result = release_evidence(source.read_bytes(), {"sha256": "test"})
        facts = {(r["field"], r["period_kind"], r["period_end"]): r for r in result["facts"]}
        self.assertEqual(facts["Revenue", "quarter", "2026-06-30"]["value"], 11119777000)
        self.assertEqual(facts["Revenue", "annual", "2026-06-30"]["value"], 39063072000)
        self.assertEqual(facts["CFO", "annual", "2026-06-30"]["value"], -6809886000)
        self.assertEqual(facts["EPS_Basic", "annual", "2026-06-30"]["value"], 3.65)
        self.assertEqual(facts["EPS_Basic", "annual", "2026-06-30"]["unit"], "USD/share")
        self.assertEqual(facts["StockholdersEquity", "instant", "2026-06-30"]["value"], 14479452000)
        self.assertEqual(result["publication_date"], "2026-08-11")
        self.assertIsNone(result["published_at"])
        self.assertIn("SharesOutstanding", result["missing_required_fields"])
        self.assertNotIn(("SharesOutstanding", "instant", "2026-06-30"), facts)

    def test_c_original_contextless_exhibit_uses_same_accession_main(self):
        source = ROOT / "state/research/acquisition/20261016-q2-identity-review-final/security_identities.json"
        exhibit = ROOT / "state/research/acquisition/20261016-c-original-exhibit/sources/c-20260630_d2.htm"
        if not source.exists() or not exhibit.exists():
            self.skipTest("Local original document set not distributed with code")
        row = next(r for r in json.loads(source.read_text()) if r["ticker"] == "C")
        main_data = (ROOT / row["source_file"]).read_bytes()
        exhibit_data = exhibit.read_bytes()
        self.assertFalse(cover_identity(exhibit_data, {}, "C", 831001)["matches"])
        main_provenance = {k: row[k] for k in ("accession", "source_url", "sha256", "published_at")}
        provenance = {**main_provenance,
                      "source_url": row["source_url"].rsplit("/", 1)[0] + "/c-20260630_d2.htm"}
        result = referenced_exhibit_identity(main_data, exhibit_data, main_provenance, provenance)
        self.assertEqual(result["status"], "ORIGINAL_COVER_TICKER_ENTITY_MATCH")
        self.assertEqual(result["equity_matches"][0]["context_id"], "c-2856")
        self.assertEqual(result["equity_matches"][0]["symbols"], ["C"])
        self.assertEqual(result["equity_matches"][0]["security_titles"],
                         ["Common Stock, par value $.01 per share"])
        self.assertEqual(result["equity_matches"][0]["exchange"], ["New York Stock Exchange"])
        self.assertEqual(len(result["equity_matches"]), 1)
        with self.assertRaisesRegex(ValueError, "same original accession"):
            referenced_exhibit_identity(main_data, exhibit_data, main_provenance,
                                        {**provenance, "accession": "0000831001-25-000045"})
        with self.assertRaisesRegex(ValueError, "does not reference"):
            referenced_exhibit_identity(main_data.replace(b"c-20260630_d2.htm", b"unrelated.htm"),
                                        exhibit_data, main_provenance, provenance)

    def test_smci_release_supplement_does_not_invent_end_shares_or_debt_tags(self):
        source = ROOT / "state/research/acquisition/20261016-q2-primary-cases-01/sources/smci/fy2026-release.html"
        if not source.exists():
            self.skipTest("Local release original not distributed with code")
        result = review(source.read_bytes())
        self.assertEqual(result["SharesOutstanding"]["status"], "NOT_DISCLOSED_IN_THIS_ORIGINAL_RELEASE")
        self.assertIsNone(result["SharesOutstanding"]["value"])
        self.assertIsNone(result["debt"]["LongTermDebt"])
        self.assertIsNone(result["identity"]["isin"])
        self.assertEqual(result["identity"]["exact_text"], "Super Micro Computer, Inc. (NASDAQ: SMCI)")
        values = {r["original_label"]: r["value"] for r in result["balance_sheet_observations"]
                  if r["period_end"] == "2026-06-30"}
        self.assertEqual(values["Common stock and additional paid-in capital"], 4600893000)
        self.assertEqual(values["Lines of credit and term loans, current"], 2039774000)
        self.assertEqual(values["Lines of credit and term loans, non-current"], 2016374000)
        self.assertEqual(values["Convertible notes"], 4664139000)


if __name__ == "__main__":
    unittest.main()
