"""Offline publication-delay histograms; never infer earnings dates or market caps."""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = Path("state/diagnostics/publication_timing.json")
EVIDENCE_PATH = Path("state/sources/publication_timing_evidence.json")
SCHEMA_VERSION = 1
NY = ZoneInfo("America/New_York")
BUCKETS = [
    {"label": "< 0", "low": None, "high": -1},
    *({"label": f"{n}–{n + 6}", "low": n, "high": n + 6} for n in range(0, 91, 7)),
    {"label": "≥ 91", "low": 91, "high": None},
]


def quarter_end(cycle: str) -> date:
    if not re.fullmatch(r"\d{4}Q[1-4]", cycle):
        raise ValueError(f"Invalid quarter: {cycle}")
    year, month = int(cycle[:4]), int(cycle[-1]) * 3
    return date(year, month, calendar.monthrange(year, month)[1])


def cycle_offset(cycle: str, offset: int) -> str:
    quarter_end(cycle)
    value = int(cycle[:4]) * 4 + int(cycle[-1]) - 1 + offset
    year, quarter = divmod(value, 4)
    return f"{year}Q{quarter + 1}"


def _canonical(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def _stamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError(f"Timezone missing in publication evidence: {value}")
    return stamp


def _read(path: Path, sources: dict) -> dict:
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def bucket_index(days: int) -> int:
    return next(i for i, b in enumerate(BUCKETS)
                if (b["low"] is None or days >= b["low"])
                and (b["high"] is None or days <= b["high"]))


def aggregate(rows: list[dict], event: str, historical: bool = False) -> dict:
    """History counts issuer-quarter observations, not distinct companies."""
    counts = [0] * len(BUCKETS)
    cap_values = [0.0] * len(BUCKETS)
    known_dates = known_caps = joint = 0
    total_cap = dated_cap = 0.0
    for row in rows:
        cap = row.get("market_cap_usd")
        lag = row.get(event + "_days")
        if cap is not None:
            known_caps += 1
            total_cap += cap
        if lag is not None:
            known_dates += 1
            idx = bucket_index(lag)
            counts[idx] += 1
            if cap is not None:
                joint += 1
                dated_cap += cap
                cap_values[idx] += cap
    return {
        "expected": len(rows), "known_dates": known_dates,
        "missing_dates": len(rows) - known_dates,
        "known_caps": known_caps, "missing_caps": len(rows) - known_caps,
        "joint_observations": joint,
        "known_cap_usd": total_cap, "dated_cap_usd": dated_cap,
        "full_cap_coverage": known_caps == len(rows) and bool(rows),
        "counts": counts, "market_cap_usd": cap_values,
        "historical": historical,
        "unit": "issuer-quarter observations" if historical else "issuers",
    }


def _evidence(root: Path, sources: dict, now: datetime) -> tuple[dict, dict]:
    path = root / EVIDENCE_PATH
    if not path.exists():
        return {}, {}
    doc = _read(path, sources)
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported publication timing evidence schema")
    earnings, caps = {}, {}
    for kind, target in (("earnings", earnings), ("market_caps", caps)):
        for row in doc.get(kind, []):
            key = (row["cycle_key"], str(int(row["company_id"])))
            end = quarter_end(key[0])
            if not row.get("source_url", "").startswith("https://"):
                raise ValueError(f"{kind} {key}: HTTPS source_url required")
            if _stamp(row["observed_at"]) > now:
                raise ValueError(f"{kind} {key}: observation time is in the future")
            if key in target:
                raise ValueError(f"Duplicate {kind} evidence for {key}")
            if kind == "earnings":
                date.fromisoformat(row["published_on"])
                date.fromisoformat(row["period_end"])
                if row.get("first_publication_verified") is not True:
                    raise ValueError(f"Earnings {key}: first publication is not verified")
            else:
                value = row["value_usd"]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"Invalid USD market cap for {key}")
                if row.get("scope") != "issuer_all_share_classes" or row["as_of"] != end.isoformat():
                    raise ValueError(f"Market cap {key}: issuer scope and quarter-end date required")
            target[key] = row
    return earnings, caps


def build_report(root: Path, now: datetime, history_quarters: int = 12) -> dict:
    if now.tzinfo is None:
        raise ValueError("Observation time must be timezone-aware")
    if history_quarters < 1 or history_quarters > 40:
        raise ValueError("history_quarters must be between 1 and 40")
    sources: dict[str, str] = {}
    manifests = {}
    for folder in ("state/expected_filings", "state/expected_filings_q2_2026Q2"):
        for path in sorted((root / folder).glob("*.json")):
            if not re.fullmatch(r"\d{4}Q[1-4]", path.stem):
                continue
            doc = _read(path, sources)
            if doc.get("period_rule") != "calendar-quarter-v2":
                continue
            manifests[path.stem] = doc
    if not manifests:
        raise ValueError("No versioned quarterly report manifests available")
    latest = max(manifests)
    earliest = min(manifests)
    span = (int(latest[:4]) - int(earliest[:4])) * 4 + int(latest[-1]) - int(earliest[-1])
    requested_cycles = {cycle_offset(latest, -i) for i in range(history_quarters + span + 1)}
    requested_cycles.update(manifests)
    membership_path = root / "cache/sp500_historical_snapshots.csv"
    universe_path = root / "cache/sp500_universe.csv"
    members_raw, universe_raw = membership_path.read_bytes(), universe_path.read_bytes()
    sources[str(membership_path)] = hashlib.sha256(members_raw).hexdigest()
    sources[str(universe_path)] = hashlib.sha256(universe_raw).hexdigest()
    memberships = list(csv.DictReader(members_raw.decode().splitlines()))
    mappings = {}
    for row in csv.DictReader(universe_raw.decode().splitlines()):
        ticker = _canonical(row["tic"])
        cik = row.get("cik", "")
        if cik and cik != "nan":
            mappings[ticker] = str(int(float(cik)))
    earnings, caps = _evidence(root, sources, now)
    submissions: dict[str, dict] = {}
    rows_by_cycle, quarters = {}, {}
    for cycle in sorted(requested_cycles):
        end = quarter_end(cycle)
        if end > now.astimezone(NY).date():
            continue
        manifest = manifests.get(cycle)
        expected: dict[str, dict] = {}
        if manifest:
            for entry in manifest["expected"]:
                cik = str(int(entry["company_id"]))
                info = expected.setdefault(cik, {"company_id": cik, "tickers": [],
                                                  "period": entry["expected_period"],
                                                  "form": entry["filing_type"]})
                info["tickers"].extend(entry.get("issuer_tickers") or [entry["security_id"]])
            for entry in manifest.get("securities", []):
                cik = str(int(entry["cik"])) if entry.get("cik") is not None else None
                key = cik or "unresolved:" + entry["ticker"]
                info = expected.setdefault(key, {"company_id": cik, "tickers": []})
                info["tickers"].append(entry["ticker"])
            universe_source = manifest.get("universe_source")
            membership_status = "versioned_manifest"
        else:
            available = [m for m in memberships if date.fromisoformat(m["date"]) <= end]
            if not available:
                continue
            membership = max(available, key=lambda m: m["date"])
            for ticker in membership["tickers"].split(","):
                ticker = _canonical(ticker)
                cik = mappings.get(ticker)
                key = cik or "unresolved:" + ticker
                expected.setdefault(key, {"company_id": cik, "tickers": []})["tickers"].append(ticker)
            universe_source = {"file": str(membership_path.relative_to(root)),
                               "snapshot_date": membership["date"]}
            membership_status = "historical_membership_local_cik_mapping_not_time_validated"
        result_rows = []
        for key, info in expected.items():
            cik = info["company_id"]
            row = {"cycle_key": cycle, "company_id": cik, "tickers": sorted(set(info["tickers"])),
                   "sec_days": None, "earnings_days": None, "market_cap_usd": None,
                   "sec_status": "UNRESOLVED_CIK" if cik is None else "NO_MATCHING_FILING",
                   "earnings_status": "NO_VERIFIED_FIRST_RELEASE",
                   "market_cap_status": "NO_QUARTER_END_ISSUER_CAP"}
            if cik is not None:
                if cik not in submissions:
                    path = root / "cache/secsubmissions" / f"{cik}.json"
                    submissions[cik] = _read(path, sources) if path.exists() else {}
                recent = submissions[cik].get("filings", {}).get("recent", {})
                names = ("form", "reportDate", "accessionNumber", "acceptanceDateTime", "primaryDocument")
                filings = [dict(zip(names, values)) for values in
                           zip(*(recent.get(name, []) for name in names))]
                previous_end = quarter_end(cycle_offset(cycle, -1))
                candidates = []
                for filing in filings:
                    if filing["form"] not in {"10-Q", "10-K"} or not filing["reportDate"]:
                        continue
                    period = date.fromisoformat(filing["reportDate"])
                    if not previous_end + timedelta(days=7) < period <= end + timedelta(days=7):
                        continue
                    if info.get("period") and (filing["reportDate"] != info["period"]
                                              or filing["form"] != info["form"]):
                        continue
                    if not filing["acceptanceDateTime"]:
                        continue
                    accepted = _stamp(filing["acceptanceDateTime"])
                    if accepted <= now and filing["accessionNumber"]:
                        candidates.append((accepted, filing))
                periods = {(f["reportDate"], f["form"]) for _, f in candidates}
                if len(periods) > 1:
                    row["sec_status"] = "AMBIGUOUS_REPORT_PERIOD"
                elif candidates:
                    accepted, filing = min(candidates, key=lambda pair: pair[0])
                    row.update({
                        "sec_status": "OBSERVED_IN_SUBMISSIONS",
                        "sec_days": (accepted.astimezone(NY).date() - end).days,
                        "sec_published_at": accepted.isoformat(),
                        "period_end": filing["reportDate"], "form": filing["form"],
                        "accession": filing["accessionNumber"],
                        "sec_source_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                        f"{filing['accessionNumber'].replace('-', '')}/{filing['primaryDocument']}",
                    })
                evidence_key = (cycle, cik)
                release = earnings.get(evidence_key)
                period = info.get("period") or row.get("period_end")
                if release:
                    if release["period_end"] != period:
                        row["earnings_status"] = "EVIDENCE_PERIOD_MISMATCH"
                    elif date.fromisoformat(release["published_on"]) <= now.astimezone(NY).date():
                        row.update({"earnings_status": "VERIFIED_FIRST_RELEASE",
                                    "earnings_days": (date.fromisoformat(release["published_on"]) - end).days,
                                    "earnings_source": release})
                    else:
                        row["earnings_status"] = "FUTURE_PUBLICATION_EXCLUDED"
                cap = caps.get(evidence_key)
                if cap:
                    row.update({"market_cap_usd": cap["value_usd"],
                                "market_cap_status": "SOURCED_QUARTER_END_ISSUER_CAP",
                                "market_cap_source": cap})
            result_rows.append(row)
        rows_by_cycle[cycle] = result_rows
        quarters[cycle] = {
            "quarter_end": end.isoformat(), "universe_source": universe_source,
            "membership_status": membership_status,
            "observation_days_after_end": (now.astimezone(NY).date() - end).days,
            "sec": aggregate(result_rows, "sec"), "earnings": aggregate(result_rows, "earnings"),
            "rows": result_rows,
        }
    history = {}
    for selected in quarters:
        cycles = sorted(c for c in quarters if cycle_offset(selected, -history_quarters) <= c < selected)
        rows = [r for c in cycles for r in rows_by_cycle[c]]
        history[selected] = {"cycles": cycles, "sec": aggregate(rows, "sec", True),
                             "earnings": aggregate(rows, "earnings", True)}
    return {
        "schema_version": SCHEMA_VERSION, "generated_at": now.isoformat(),
        "buckets": BUCKETS, "quarters": quarters, "history": history,
        "sources": sources, "network_requests": 0,
        "notes": [
            "SEC = earliest cached original 10-Q/10-K acceptance, not the first earnings announcement.",
            "Calendar-day lag uses America/New_York and official calendar quarter-end, not fiscal period-end.",
            "Negative delays are retained for fiscal periods ending before calendar quarter-end.",
            "Missing dates remain unknown, not zero; recent quarters are right-censored at generated_at.",
            "Historical counts are issuer-quarter observations with changing quarterly membership.",
            "Historical local CIK mappings are not a fully time-versioned issuer identity history.",
            "No 8-K reportDate, estimated earnings calendar, SPY weight or adjusted-price/share product is substituted.",
            "Market caps require a sourced USD issuer total covering all share classes at quarter-end.",
            "Local SEC submissions may omit older filings; no missing filing is inferred from absence.",
        ],
    }


def load_report(root: Path, cycle: str | None) -> dict:
    path = root / REPORT_PATH
    if not path.exists():
        return {"available": False, "error": "Publikationsauswertung wurde noch nicht erzeugt."}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported publication timing report schema")
        selected = cycle or max(doc["quarters"])
        available = sorted(doc["quarters"], reverse=True)
        if selected not in doc["quarters"]:
            return {"available": False, "available_quarters": available,
                    "error": f"Keine Publikationsauswertung für {selected}."}
        quarter = {k: v for k, v in doc["quarters"][selected].items() if k != "rows"}
        return {"available": True, "cycle_key": selected, "generated_at": doc["generated_at"],
                "available_quarters": available, "buckets": doc["buckets"], "quarter": quarter,
                "history": doc["history"][selected], "notes": doc["notes"]}
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return {"available": False, "error": f"Publikationsauswertung ungültig: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-quarters", type=int, default=12)
    args = parser.parse_args()
    report = build_report(ROOT, datetime.now(timezone.utc), args.history_quarters)
    path = ROOT / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"Saved {path.relative_to(ROOT)}; {len(report['quarters'])} quarters; no network access")
    for cycle, item in report["quarters"].items():
        print(cycle, "SEC", item["sec"]["known_dates"], "/", item["sec"]["expected"],
              "earnings", item["earnings"]["known_dates"], "caps", item["sec"]["known_caps"])


if __name__ == "__main__":
    main()
