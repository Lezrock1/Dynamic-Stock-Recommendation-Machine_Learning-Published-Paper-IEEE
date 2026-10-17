"""Immutable source acquisition and exact accession-based historical fact evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests

from .fixed_lag_research import write_csv, write_json
from .paper_fixed_lag import schedule
from .paper_live import PERIOD_RULE_V2, period_window
from .return_comparison import sha256
from .sec_edgar import ADDITIVE_FIELDS, DURATION_TAGS, INSTANT_TAGS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path("state/expected_filings_q2_2026Q2/2026Q2.json")
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceStore:
    """No credentials, cache fallback, overwritten sources or invented fetch dates."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.records: list[dict] = []
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "dynamic-stock-recommendation-research/1.0"
        self.rate_limited_hosts: set[str] = set()

    def _record(self, record: dict) -> None:
        self.records.append(record)
        write_json(self.directory / "acquisition.json", self.records)

    def fetch(self, name: str, url: str) -> Path:
        if urlsplit(url).scheme != "https":
            raise ValueError("Source downloads must use HTTPS")
        path = self.directory / "sources" / name
        if path.resolve().is_relative_to((self.directory / "sources").resolve()) is False:
            raise ValueError("Invalid evidence filename")
        if path.exists():
            raise FileExistsError(path)
        host = urlsplit(url).hostname or ""
        if host in self.rate_limited_hosts:
            raise RuntimeError(f"Further requests stopped after rate limit: {host}")
        headers = {}
        if host == "sec.gov" or host.endswith(".sec.gov"):
            agent = os.environ.get("DYNAMIC_STOCK_USER_AGENT", "")
            if "@" not in agent or "configure" in agent:
                raise ValueError("A genuine session-only DYNAMIC_STOCK_USER_AGENT is required")
            headers["User-Agent"] = agent
        record = {"source_url": url, "file": name, "requested_at": now(), "attempts": []}
        for attempt in range(3):
            time.sleep(max(0, 0.22 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.session.get(url, headers=headers, timeout=60)
            except requests.RequestException as exc:
                record["attempts"].append({"error_type": type(exc).__name__, "checked_at": now()})
                if attempt == 2:
                    record["status"] = "NETWORK_FAILED"
                    self._record(record)
                    raise RuntimeError(f"Source request failed: {url}") from exc
                time.sleep(2 ** attempt)
                continue
            record["attempts"].append({"http_status": response.status_code, "checked_at": now()})
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
                    if delay > 120:
                        self.rate_limited_hosts.add(host)
                        record["status"] = "RATE_LIMIT_RETRY_DELAY_EXCEEDS_THIS_RUN"
                        self._record(record)
                        raise RuntimeError(f"Source asks for Retry-After={delay}s; stopping requests to {host}")
                    time.sleep(delay)
                    continue
            record["http_status"] = response.status_code
            record["acquired_at"] = now()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(response.content)
            record["sha256"] = sha256(path)
            record["size"] = path.stat().st_size
            record["status"] = "ACQUIRED" if response.ok else "HTTP_FAILED"
            self._record(record)
            if not response.ok:
                if response.status_code == 429:
                    self.rate_limited_hosts.add(host)
                raise RuntimeError(f"HTTP {response.status_code}: {url}")
            return path
        raise RuntimeError(f"Retry loop exhausted: {url}")

    def copy_cache(self, source: Path, name: str) -> Path:
        target = self.directory / "sources" / name
        if not target.resolve().is_relative_to((self.directory / "sources").resolve()):
            raise ValueError("Invalid evidence filename")
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        if sha256(source) != before or sha256(target) != before:
            raise RuntimeError(f"Cache changed while copying: {source.name}")
        original = source.resolve()
        self._record({"file": name, "original_cache": str(original.relative_to(ROOT))
                      if original.is_relative_to(ROOT) else str(original),
                      "sha256": before, "copied_at": now(), "acquired_at": None,
                      "status": "EXISTING_CACHE_COPIED_FETCH_TIME_UNKNOWN"})
        return target

    def json_source(self, cache: Path, name: str, url: str) -> dict:
        path = self.copy_cache(cache, name) if cache.is_file() else self.fetch(name, url)
        return json.loads(path.read_text())


def filings(document: dict) -> list[dict]:
    body = document.get("filings", {}).get("recent", document)
    accessions = body.get("accessionNumber", [])
    fields = ("accessionNumber", "form", "reportDate", "filingDate", "acceptanceDateTime", "primaryDocument")
    return [{field: body.get(field, [])[i] if i < len(body.get(field, [])) else None
             for field in fields} for i in range(len(accessions))]


def acceptance_map(rows: list[dict]) -> dict[str, pd.Timestamp]:
    result = {}
    for row in rows:
        if not row.get("acceptanceDateTime"):
            continue
        stamp = pd.Timestamp(row["acceptanceDateTime"])
        if stamp.tzinfo is None:
            raise ValueError("SEC acceptance timestamp lacks timezone; cannot invent one")
        accession = row["accessionNumber"]
        if accession in result and result[accession] != stamp:
            raise ValueError(f"Conflicting acceptance timestamps for {accession}")
        result[accession] = stamp.tz_convert("UTC")
    return result


def period_expectation(rows: list[dict], cycle: str) -> dict:
    low, high = period_window(pd.Period(cycle, "Q").end_time.normalize())
    candidates = [r for r in rows if r.get("form") in {"10-Q", "10-K"} and r.get("reportDate")
                  and low < pd.Timestamp(r["reportDate"]) <= high]
    periods = {(r["reportDate"], r["form"]) for r in candidates}
    if len(periods) != 1:
        return {"status": "MISSING_OR_AMBIGUOUS_PERIOD", "candidates": sorted(periods)}
    period, form = next(iter(periods))
    return {"status": "OBSERVED_EXPECTED_PERIOD", "period_end": period, "filing_form": form,
            "period_rule": PERIOD_RULE_V2,
            "derivation": "actual report period in versioned calendar window; not fiscal quarter number"}


def source_atom(namespace: str, tag: str, unit: str, entry: dict, published: pd.Timestamp,
                facts_hash: str, cik: int) -> dict:
    canonical = json.dumps({"namespace": namespace, "tag": tag, "unit": unit, "entry": entry},
                           sort_keys=True, allow_nan=False)
    return {
        "namespace": namespace, "tag": tag, "unit": unit, "value": float(entry["val"]),
        "period_start": entry.get("start"), "period_end": entry["end"],
        "accession": entry["accn"], "form": entry["form"],
        "published_at": published.isoformat(), "filed_date_not_used_as_timestamp": entry.get("filed"),
        "value_version": hashlib.sha256(canonical.encode()).hexdigest(),
        "raw_companyfacts_sha256": facts_hash, "raw_entry": entry,
        "source_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        "filing_source_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{entry['accn'].replace('-', '')}/",
        "proof": "SEC companyfacts accession-attributed value joined to actual SEC acceptance timestamp",
    }


def fact_catalog(document: dict, accepted: dict, cutoff: str, cik: int,
                 facts_hash: str) -> tuple[dict, dict]:
    cutoff_stamp = pd.Timestamp(cutoff)
    if cutoff_stamp.tzinfo is None:
        raise ValueError("Historical cutoff must have a timezone")
    catalog, counts = {}, {"later_versions_excluded": 0, "unknown_acceptance_excluded": 0,
                           "invalid_values_excluded": 0, "ambiguous_values_excluded": 0}
    for field, tags in {**DURATION_TAGS, **INSTANT_TAGS}.items():
        is_duration = field in DURATION_TAGS
        preferred_unit = "USD/shares" if field == "EPS_Basic" else "shares" if field == "SharesOutstanding" else "USD"
        by_period = {}
        alternatives = [("us-gaap", tag) for tag in tags]
        if field == "SharesOutstanding":
            alternatives.append(("dei", "EntityCommonStockSharesOutstanding"))
        for namespace, tag in alternatives:
            entries = document.get("facts", {}).get(namespace, {}).get(tag, {}).get("units", {}).get(
                preferred_unit, [])
            versions = {}
            for entry in entries:
                if entry.get("form") not in FORMS or not entry.get("end"):
                    continue
                stamp = accepted.get(entry.get("accn"))
                if stamp is None:
                    counts["unknown_acceptance_excluded"] += 1
                    continue
                if stamp > cutoff_stamp:
                    counts["later_versions_excluded"] += 1
                    continue
                value = entry.get("val")
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    counts["invalid_values_excluded"] += 1
                    continue
                if is_duration and not entry.get("start"):
                    continue
                key = (entry.get("start"), entry["end"]) if is_duration else (None, entry["end"])
                versions.setdefault(key, []).append((stamp, entry))
            selected = {}
            for key, values in versions.items():
                latest = max(stamp for stamp, _ in values)
                newest = [(stamp, entry) for stamp, entry in values if stamp == latest]
                if len({entry["val"] for _, entry in newest}) != 1:
                    counts["ambiguous_values_excluded"] += 1
                    continue
                stamp, entry = sorted(newest, key=lambda v: v[1]["accn"])[-1]
                atom = source_atom(namespace, tag, preferred_unit, entry, stamp, facts_hash, cik)
                selected[key] = {"value": float(entry["val"]), "components": [atom],
                                 "published_at": stamp.isoformat(), "operation": "direct"}
            direct = {}
            for (start, end), value in selected.items():
                if not is_duration or 55 <= (pd.Timestamp(end) - pd.Timestamp(start)).days <= 115:
                    direct.setdefault(end, []).append(value)
            for end, options in direct.items():
                stamp = max(v["published_at"] for v in options)
                latest = [v for v in options if v["published_at"] == stamp]
                if len({v["value"] for v in latest}) > 1:
                    counts["ambiguous_values_excluded"] += 1
                    continue
                by_period.setdefault(end, latest[0])
            if field in ADDITIVE_FIELDS:
                for (start, end), current in selected.items():
                    if not 115 < (pd.Timestamp(end) - pd.Timestamp(start)).days <= 390:
                        continue
                    prior = [(p_end, previous) for (p_start, p_end), previous in selected.items()
                             if p_start == start and 55 <= (pd.Timestamp(end) - pd.Timestamp(p_end)).days <= 115]
                    if not prior or end in by_period:
                        continue
                    _, previous = max(prior, key=lambda item: item[0])
                    by_period[end] = {
                        "value": current["value"] - previous["value"],
                        "components": current["components"] + previous["components"],
                        "published_at": max(current["published_at"], previous["published_at"]),
                        "operation": "current_cumulative_minus_prior_same_tag_unit_fiscal_start",
                        "scope_comparability": "REQUIRES_REVIEW_NO_AUTOMATIC_CIK_SUCCESSION",
                    }
        catalog[field] = by_period
    return catalog, counts


def evidence_for_period(catalog: dict, period: str) -> dict:
    end = pd.Timestamp(period)
    output = {}
    for field, values in catalog.items():
        output[field] = values.get(period)
    ttm = {}
    for field in ("Revenue", "CFO", "EPS_Basic", "OperatingIncome", "DepreciationAmortization"):
        periods = sorted(p for p in catalog[field] if pd.Timestamp(p) <= end)[-4:]
        dates = [pd.Timestamp(p) for p in periods]
        complete = (len(dates) == 4 and dates[-1] == end
                    and all(55 <= (b - a).days <= 115 for a, b in zip(dates, dates[1:]))
                    and 250 <= (dates[-1] - dates[0]).days <= 390)
        ttm[field] = {
            "status": "COMPONENT_VERSIONS_EVIDENCED" if complete else "MISSING_CONSECUTIVE_QUARTERS",
            "periods": periods,
            "components": [catalog[field][p] for p in periods],
            "value": sum(catalog[field][p]["value"] for p in periods) if complete else None,
        }
    revenue = catalog["Revenue"]
    prior = sorted(p for p in revenue if 330 <= (end - pd.Timestamp(p)).days <= 400)
    prior_value = revenue[prior[-1]] if prior else None
    return {"period_values": output, "ttm": ttm, "prior_year_revenue": prior_value,
            "component_evidence_not_feature_certification": True,
            "open_reviews": ["factor definitions/source overlap", "share-class and accounting scope",
                             "price/split basis", "historical sector/security identity"]}


def collect_q2(output: Path, refresh: bool = False) -> dict:
    store = EvidenceStore(output)
    manifest = json.loads((ROOT / MANIFEST).read_text())
    cutoff = schedule("2026Q2")["information_cutoff"]
    ticker_path = store.fetch("sec_company_tickers_exchange.json",
                             "https://www.sec.gov/files/company_tickers_exchange.json")
    mapping = json.loads(ticker_path.read_text())
    tickers = {str(row[mapping["fields"].index("ticker")]).replace(".", "-"): row
               for row in mapping["data"]}
    expected = {str(row["company_id"]): row for row in manifest["expected"]}
    issuer_results, rows = {}, []
    for security in manifest["securities"]:
        ticker = security["ticker"]
        match = tickers.get(ticker)
        candidate_cik = int(match[mapping["fields"].index("cik")]) if match else None
        cik = security.get("cik") or candidate_cik
        row = {**security, "candidate_current_sec_cik": candidate_cik,
               "cik_used_for_source_research": cik, "cycle_key": "2026Q2",
               "information_cutoff": cutoff, "local_checked_at": now(),
               "operational_facts_ready_at": None,
               "historical_identity_status": "NOT_CERTIFIED_CURRENT_MAPPING_IS_CANDIDATE_ONLY",
               "historical_sector_status": "NOT_TIME_VERSIONED",
               "historical_membership_source": manifest["universe_source"]}
        if not cik:
            rows.append({**row, "status": "CIK_UNRESOLVED"})
            continue
        cik = int(cik)
        if cik not in issuer_results:
            sources, errors = {}, []
            for kind, url in (
                ("secsubmissions", f"https://data.sec.gov/submissions/CIK{cik:010d}.json"),
                ("secfacts", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"),
            ):
                try:
                    cache = ROOT / "cache" / kind / f"{cik}.json"
                    if refresh:
                        source = store.fetch(f"{kind}/{cik}.json", url)
                        sources[kind] = json.loads(source.read_text())
                    else:
                        sources[kind] = store.json_source(cache, f"{kind}/{cik}.json", url)
                except (OSError, ValueError, RuntimeError) as exc:
                    errors.append({"source": kind, "error": str(exc)})
            if len(sources) != 2:
                issuer_results[cik] = {"status": "SOURCE_ACQUISITION_FAILED", "errors": errors}
            else:
                subs, facts = sources["secsubmissions"], sources["secfacts"]
                if int(subs["cik"]) != cik or int(facts["cik"]) != cik:
                    raise ValueError(f"Downloaded issuer mismatch: {cik}")
                filing_rows = filings(subs)
                accepted = acceptance_map(filing_rows)
                expectation = period_expectation(filing_rows, "2026Q2")
                facts_hash = sha256(output / "sources/secfacts" / f"{cik}.json")
                catalog, counts = fact_catalog(facts, accepted, cutoff, cik, facts_hash)
                result = {"cik": cik, "name": subs.get("name"), "current_tickers": subs.get("tickers"),
                          "former_names": subs.get("formerNames"), "sic_not_gics": subs.get("sic"),
                          "period_expectation": expectation, "version_selection": counts,
                          "acceptance_timestamps": len(accepted),
                          "earlier_submission_pages_not_yet_loaded": subs.get("filings", {}).get("files", []),
                          "status": "EXACT_VALUE_VERSION_CATALOG_NOT_FULL_MODEL_CERTIFICATION",
                          "original_expected": expected.get(str(cik)),
                          "availability_assumption": "reconstructed publication availability, not historic adapter readiness"}
                if expectation.get("period_end"):
                    result["period_evidence"] = evidence_for_period(catalog, expectation["period_end"])
                write_json(output / f"issuers/{cik}.json", result)
                write_json(output / f"catalogs/{cik}.json", catalog)
                issuer_results[cik] = result
        result = issuer_results[cik]
        expectation = result.get("period_expectation", {})
        values = result.get("period_evidence", {}).get("period_values", {})
        rows.append({**row, "status": result["status"], "expected_period": expectation.get("period_end"),
                     "expected_form": expectation.get("filing_form"),
                     "metric_versions_evidenced": sum(v is not None for v in values.values()),
                     "missing_period_metrics": [k for k, v in values.items() if v is None],
                     "errors": result.get("errors", [])})
    write_csv(output / "q2_security_evidence.csv", rows)
    write_json(output / "q2_security_evidence.json", rows)
    summary = {"status": "COMPLETED_SOURCE_AUDIT_NOT_MODEL_RUN", "cycle_key": "2026Q2",
               "information_cutoff": cutoff, "execution_date": "2026-09-01",
               "issuers_researched": len(issuer_results), "securities": len(rows),
               "acquisition_failures": sum(r["status"] == "SOURCE_ACQUISITION_FAILED" for r in issuer_results.values()),
               "complete_feature_datasets": 0, "historical_validation": "NOT_VALIDATED",
               "contact_stored": False, "model_started": False, "scheduler_enabled": False,
               "protected_existing_inputs": "read-only; new acquisition directory only"}
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_manifest.json", {
        str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="Download fresh SEC data rather than copy existing cache")
    args = parser.parse_args()
    print(json.dumps(collect_q2(args.output, args.refresh), indent=2))


if __name__ == "__main__":
    main()
