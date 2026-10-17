"""Verify security identities against original historical filing cover-page contexts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd

from .fixed_lag_research import offline_guard, write_csv, write_json
from .historical_evidence import (
    EvidenceStore, MANIFEST, ROOT, acceptance_map, evidence_for_period, fact_catalog, filings,
    now, period_expectation,
)
from .ixbrl import parse_facts
from .paper_fixed_lag import continuation_cycles, schedule
from .return_comparison import sha256


def cover_identity(data: bytes, provenance: dict, ticker: str, cik: int) -> dict:
    soup = BeautifulSoup(data, "lxml")
    contexts = {}
    for context in soup.find_all("xbrli:context"):
        entity = context.find("xbrli:identifier")
        contexts[context.get("id")] = {
            "entity": entity.get_text(strip=True) if entity else None,
            "period": tuple(node.get_text(strip=True) for node in context.find_all(
                ["xbrli:startdate", "xbrli:enddate", "xbrli:instant"])),
            "dimensions": [(node.get("dimension"), node.get_text(strip=True))
                           for node in context.find_all("xbrldi:explicitmember")],
            "typed_dimensions": [node.get("dimension") for node in context.find_all("xbrldi:typedmember")],
        }
    tags = {}
    for node in soup.find_all("ix:nonnumeric"):
        tag = str(node.get("name", ""))
        if not tag.startswith("dei:"):
            continue
        tags.setdefault(node.get("contextref"), {}).setdefault(tag.split(":")[-1], []).append(
            "".join(node.strings).strip())
    matches = []
    for context_id, values in tags.items():
        symbols = values.get("TradingSymbol", [])
        normalized = [re.sub(r"\s+\((NYSE|NASDAQ)\)$", "", s.strip('\u201c\u201d"')).replace(".", "-")
                      for s in symbols]
        class_b_alias = (ticker == "BF-B" and "BFB" in symbols and
                         any(re.search(r"\bClass B\b", title, re.I)
                             for title in values.get("Security12bTitle", [])))
        if ticker not in normalized and not class_b_alias:
            continue
        context = contexts.get(context_id, {})
        entity = context.get("entity")
        if entity is None or not entity.isdigit() or int(entity) != cik:
            continue
        if context.get("typed_dimensions"):
            continue
        matches.append({
            "ticker": ticker, "cik": cik, "context_id": context_id,
            "dimensions": context.get("dimensions"), "symbols": symbols,
            "security_titles": values.get("Security12bTitle", []),
            "exchange": values.get("SecurityExchangeName", []),
            "registrant_names": values.get("EntityRegistrantName", []),
        })
    # RVTY's cover binds title, symbol and exchange in one row, not one context.
    for symbol in soup.find_all("ix:nonnumeric", attrs={"name": "dei:TradingSymbol"}):
        if symbol.get_text(strip=True) != ticker:
            continue
        row = symbol.find_parent("tr")
        if row is None or any(m["security_titles"] for m in matches):
            continue
        nodes = [n for n in row.find_all("ix:nonnumeric")
                 if n.get("name") in {"dei:TradingSymbol", "dei:Security12bTitle",
                                      "dei:SecurityExchangeName"}]
        titles = [n.get_text(strip=True) for n in nodes if n.get("name") == "dei:Security12bTitle"]
        symbols = [n.get_text(strip=True) for n in nodes if n.get("name") == "dei:TradingSymbol"]
        exchanges = [n.get_text(strip=True) for n in nodes if n.get("name") == "dei:SecurityExchangeName"]
        ids = list(dict.fromkeys(n.get("contextref") for n in nodes))
        bound = [contexts.get(key, {}) for key in ids]
        if len(titles) != 1 or symbols != [ticker] or len(exchanges) != 1:
            continue
        if not bound or any(not (c.get("entity") or "").isdigit() or int(c["entity"]) != cik
                            or c.get("typed_dimensions") or not c.get("period") for c in bound):
            continue
        if len({c["period"] for c in bound}) != 1:
            continue
        dimensions = list(dict.fromkeys(d for c in bound for d in c["dimensions"]))
        if any(axis not in {"dei:EntityListingsExchangeAxis", "revv:TradingSymbolAxis"}
               for axis, _ in dimensions):
            continue
        if any(axis == "revv:TradingSymbolAxis" and
               (ticker != "RVTY" or member != "revv:RVTYMember") for axis, member in dimensions):
            continue
        matches = [m for m in matches if m["context_id"] != symbol.get("contextref")]
        matches.append({
            "ticker": ticker, "cik": cik, "context_id": symbol.get("contextref"),
            "bound_context_ids": ids, "binding": "EXACT_COVER_TABLE_ROW_SAME_ENTITY_PERIOD",
            "dimensions": dimensions, "symbols": symbols, "security_titles": titles,
            "exchange": exchanges, "registrant_names": [],
        })
    equity = [m for m in matches if any(re.search(
        r"\b(common|ordinary|capital stock|beneficial interest|registered shares)\b", title, flags=re.I)
        and not re.search(r"\b(notes?|bonds?|debentures?)\b", title, flags=re.I)
        for title in m["security_titles"])]
    listing_class_members = {
        "us-gaap:CommonStockMember",
        "ndaq:CommonStock.01ParValuePerShareTheNasdaqStockMarketMember",
        "ndaq:CommonStock.01ParValuePerShareNadasqTexasLLCMember",
    }
    class_keys = {
        (tuple(m["security_titles"]),
         tuple(sorted(tuple(d) for d in m["dimensions"]
                      if d[0] != "dei:EntityListingsExchangeAxis"
                      and not (d[0] == "us-gaap:StatementClassOfStockAxis"
                               and d[1] in listing_class_members
                               and (d[1] == "us-gaap:CommonStockMember" or ticker == "NDAQ")))))
        for m in equity
    }
    status = ("ORIGINAL_COVER_TICKER_ENTITY_MATCH" if equity and len(class_keys) == 1
              else "MISSING_OR_AMBIGUOUS_COVER_IDENTITY")
    registration_exhibits = [
        {"title": a.get_text(" ", strip=True), "href": a.get("href")}
        for a in soup.find_all("a", href=True)
        if re.fullmatch(r"99\.01\+?", a.get_text(" ", strip=True))
        and "See Exhibit 99.01" in soup.get_text(" ", strip=True)
    ] if not equity else []
    return {"status": status, "ticker": ticker, "cik": cik, "matches": matches,
            **provenance, "verified_at": now(),
            "validity": "Evidence at this filing only, not a historical interval or a current-identity substitution",
            "historical_sector": None, "sector_note": "SEC SIC is not historical GICS",
            "equity_matches": equity,
            "registration_exhibit_references": registration_exhibits,
            "numeric_context_entities": {key: value["entity"] for key, value in contexts.items()},
            "explicit_alias": "BF-B identifier matched to original BFB Class B cover title"
                              if ticker == "BF-B" and equity else None,
            "share_class_note": "Exact identical equity titles retained across listing-only contexts; only explicit CommonStockMember and NDAQ original exchange-named listing members normalized. Other class dimensions remain distinct. Debt never selected. No class conversion."}


def collect(source_run: Path, supplemental: Path, output: Path) -> dict:
    store = EvidenceStore(output)
    manifest = json.loads((ROOT / MANIFEST).read_text())
    candidates = {"AVB": 915912, "EA": 712515, "EQR": 906107}
    cutoff = pd.Timestamp(schedule("2026Q2")["information_cutoff"])
    rows, issuer_cache = [], {}
    for security in manifest["securities"]:
        ticker = security["ticker"]
        cik = int(security.get("cik") or candidates.get(ticker) or 0)
        if not cik:
            rows.append({"ticker": ticker, "status": "CIK_UNRESOLVED"})
            continue
        if cik not in issuer_cache:
            sub_path = source_run / "sources/secsubmissions" / f"{cik}.json"
            if not sub_path.exists():
                sub_path = supplemental / "sources/secsubmissions" / f"{cik}.json"
            if not sub_path.exists():
                sub_path = store.fetch(f"secsubmissions/{cik}.json",
                                       f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            subs = json.loads(sub_path.read_text())
            if int(subs["cik"]) != cik:
                raise ValueError("Issuer/submissions identity mismatch")
            records = filings(subs)
            expected = period_expectation(records, "2026Q2")
            acceptance = acceptance_map(records)
            matching = [r for r in records if r["reportDate"] == expected.get("period_end")
                        and r["form"] == expected.get("filing_form")
                        and r["accessionNumber"] in acceptance
                        and acceptance[r["accessionNumber"]] <= cutoff]
            if not matching:
                issuer_cache[cik] = {"status": "NO_EXPECTED_ORIGINAL_FILING_BEFORE_CUTOFF",
                                     "period_expectation": expected}
            else:
                filing = min(matching, key=lambda r: acceptance[r["accessionNumber"]])
                accession, document = filing["accessionNumber"], filing["primaryDocument"]
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{document}"
                paths = [ROOT / "cache/filings" / str(cik) / accession / document,
                         supplemental / "sources/filings" / str(cik) / accession / document]
                cached = next((p for p in paths if p.is_file()), None)
                try:
                    name = f"filings/{cik}/{accession}/{document}"
                    path = store.copy_cache(cached, name) if cached else store.fetch(name, url)
                    provenance = {"accession": accession, "source_url": url, "sha256": sha256(path),
                                  "published_at": acceptance[accession].isoformat(),
                                  "period_end": expected["period_end"], "source_file": str(path)}
                    issuer_cache[cik] = {"status": "ORIGINAL_ACQUIRED", "path": path, "provenance": provenance}
                    numeric = parse_facts(path.read_bytes(), {"accession": accession, "url": url,
                                                             "sha256": sha256(path)})
                    write_json(output / f"numeric_facts/{cik}.json", numeric)
                except RuntimeError as exc:
                    issuer_cache[cik] = {"status": "ORIGINAL_ACQUISITION_FAILED", "error": str(exc)}
            if cik in candidates.values():
                fact_path = store.fetch(f"secfacts/{cik}.json",
                                        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
                doc = json.loads(fact_path.read_text())
                catalog, counts = fact_catalog(doc, acceptance, cutoff.isoformat(), cik, sha256(fact_path))
                write_json(output / f"catalogs/{cik}.json", catalog)
                write_json(output / f"issuer_values/{cik}.json", {
                    "cik": cik, "version_selection": counts, "expected": expected,
                    "period_evidence": evidence_for_period(catalog, expected["period_end"])
                    if expected.get("period_end") else None,
                })
        result = issuer_cache[cik]
        if result["status"] != "ORIGINAL_ACQUIRED":
            row = {"ticker": ticker, "cik": cik, **result}
        else:
            row = cover_identity(result["path"].read_bytes(), result["provenance"], ticker, cik)
            write_json(output / f"identities/{ticker}.json", row)
        rows.append(row)
        write_csv(output / "security_identities.csv", rows)
    write_json(output / "security_identities.json", rows)
    summary = {"status": "COMPLETED_ORIGINAL_IDENTITY_EVIDENCE", "securities": len(rows),
               "cover_matches": sum(r["status"] == "ORIGINAL_COVER_TICKER_ENTITY_MATCH" for r in rows),
               "unresolved": [{"ticker": r["ticker"], "status": r["status"]}
                              for r in rows if r["status"] != "ORIGINAL_COVER_TICKER_ENTITY_MATCH"],
               "historical_gics_intervals_proven": 0, "historical_validation": "NOT_VALIDATED",
               "candidate_recoveries_require_cover_match": candidates, "contact_stored": False}
    write_json(output / "summary.json", summary)
    return summary


def membership_audit(root: Path, output: Path, identity_rows: list[dict]) -> dict:
    snapshots = pd.read_csv(root / "cache/sp500_historical_snapshots.csv", parse_dates=["date"])
    source_hash = sha256(root / "cache/sp500_historical_snapshots.csv")
    q2 = {r["ticker"]: r for r in identity_rows if r["status"] == "ORIGINAL_COVER_TICKER_ENTITY_MATCH"}
    rows = []
    for cycle in continuation_cycles():
        end = pd.Period(cycle, "Q").end_time.normalize()
        past = snapshots[snapshots["date"] <= end].sort_values("date")
        if past.empty:
            raise ValueError(f"No local membership snapshot before {end}")
        selected = past.iloc[-1]
        for symbol in selected["tickers"].split(","):
            symbol = symbol.replace(".", "-")
            identity = q2.get(symbol) if cycle == "2026Q2" else None
            rows.append({"cycle_key": cycle, "security_id": symbol,
                         "membership_snapshot": selected["date"].date().isoformat(),
                         "membership_source_sha256": source_hash,
                         "membership_proof": "dated local secondary-source snapshot; not original S&P constituent licence",
                         "cik": identity["cik"] if identity else None,
                         "identity_status": identity["status"] if identity else "HISTORICAL_INTERVAL_UNRESOLVED",
                         "identity_accession": identity.get("accession") if identity else None,
                         "gsector": None, "historical_sector_status": "UNRESOLVED_NOT_FILLED_FROM_CURRENT_SECTOR"})
    write_csv(output / "37_quarter_security_membership.csv", rows)
    return {"rows": len(rows), "quarters": len(continuation_cycles()),
            "membership_source_sha256": source_hash,
            "historical_gics_intervals_proven": 0,
            "note": "Q2 cover identities are deliberately not carried backwards to earlier decisions"}


def review_originals(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    identity_path = source / "security_identities.json"
    rows = json.loads(identity_path.read_text())
    inputs = {str(identity_path): sha256(identity_path), str(Path(__file__)): sha256(Path(__file__))}
    attempts, updated = [], []
    with offline_guard(attempts):
        for row in rows:
            if "source_file" in row:
                path = ROOT / row["source_file"]
                if sha256(path) != row["sha256"]:
                    raise ValueError(f"Original filing hash mismatch for {row['ticker']}")
                inputs[str(path)] = row["sha256"]
                provenance = {key: row[key] for key in (
                    "accession", "source_url", "sha256", "published_at", "period_end", "source_file")}
                row = cover_identity(path.read_bytes(), provenance, row["ticker"], row["cik"])
            updated.append(row)
    if inputs != {path: sha256(Path(path)) for path in inputs}:
        raise ValueError("An original identity input changed during review")
    write_json(output / "security_identities.json", updated)
    write_csv(output / "security_identities.csv", [
        {key: value for key, value in row.items() if key != "numeric_context_entities"} for row in updated])
    summary = {
        "securities": len(updated),
        "cover_matches": sum(r["status"] == "ORIGINAL_COVER_TICKER_ENTITY_MATCH" for r in updated),
        "unresolved": [{"ticker": r["ticker"], "status": r["status"]} for r in updated
                       if r["status"] != "ORIGINAL_COVER_TICKER_ENTITY_MATCH"],
        "membership": membership_audit(ROOT, output, updated),
        "historical_validation": "NOT_VALIDATED", "network_attempts": attempts,
    }
    write_json(output / "input_manifest.json", inputs)
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--supplemental", type=Path)
    parser.add_argument("--review-originals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.review_originals:
        print(json.dumps(review_originals(args.review_originals, args.output), indent=2))
        return
    if not args.source_run or not args.supplemental:
        parser.error("--source-run and --supplemental are required for acquisition")
    summary = collect(args.source_run, args.supplemental, args.output)
    summary["membership"] = membership_audit(
        ROOT, args.output, json.loads((args.output / "security_identities.json").read_text()))
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "artifact_manifest.json", {
        str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob("*") if p.is_file()
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
