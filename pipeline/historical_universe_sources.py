"""Dated public Git snapshots of sector, CIK and constituent metadata; no current backfill."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .fixed_lag_research import write_csv, write_json
from .historical_evidence import EvidenceStore, ROOT
from .paper_fixed_lag import continuation_cycles
from .return_comparison import sha256

SECTORS = {
    "Energy": 10, "Materials": 15, "Industrials": 20, "Consumer Discretionary": 25,
    "Consumer Staples": 30, "Health Care": 35, "Financials": 40,
    "Information Technology": 45, "Telecommunication Services": 50,
    "Telecommunications Services": 50, "Communication Services": 50,
    "Utilities": 55, "Real Estate": 60,
}


def parse_constituents(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"CIK": str})
    # Historical versions of this public dataset used these exact older headers.
    frame = frame.rename(columns={"Name": "Security", "Sector": "GICS Sector"})
    if not {"Symbol", "Security", "GICS Sector"}.issubset(frame):
        raise ValueError(f"Historical constituent schema changed: {list(frame.columns)}")
    frame["security_id"] = frame["Symbol"].str.strip().str.replace(".", "-", regex=False)
    if frame["security_id"].duplicated().any():
        raise ValueError("Historical share classes must remain distinct unique securities")
    unknown = set(frame["GICS Sector"]) - set(SECTORS)
    if unknown:
        raise ValueError(f"Unknown dated sector labels: {sorted(unknown)}")
    frame["gsector"] = frame["GICS Sector"].map(SECTORS)
    return frame


def collect(output: Path, start: str, end: str) -> dict:
    store = EvidenceStore(output)
    source = ROOT / "cache/sp500_historical_snapshots.csv"
    memberships = pd.read_csv(source, parse_dates=["date"])
    rows, quarters = [], []
    for cycle in continuation_cycles(start, end):
        period_end = pd.Period(cycle, "Q").end_time.normalize()
        until = period_end.date().isoformat() + "T23:59:59Z"
        url = ("https://api.github.com/repos/datasets/s-and-p-500-companies/commits"
               f"?path=data%2Fconstituents.csv&until={until}&per_page=1")
        try:
            metadata = store.fetch(f"commits/{cycle}.json", url)
            commits = json.loads(metadata.read_text())
            if not isinstance(commits, list) or not commits:
                raise ValueError(f"No dated public sector snapshot for {cycle}")
            commit = commits[0]
            sha, stamp = commit["sha"], commit["commit"]["committer"]["date"]
            if pd.Timestamp(stamp) > pd.Timestamp(until):
                raise ValueError("Future Git revision cannot supply historical sectors")
            csv_url = f"https://raw.githubusercontent.com/datasets/s-and-p-500-companies/{sha}/data/constituents.csv"
            path = store.fetch(f"constituents/{cycle}.csv", csv_url)
            frame = parse_constituents(path)
            members = memberships[memberships["date"] <= period_end].sort_values("date").iloc[-1]
            expected = {ticker.replace(".", "-") for ticker in members["tickers"].split(",")}
            observed = set(frame["security_id"])
            for member in sorted(expected):
                match = frame[frame["security_id"] == member]
                record = match.iloc[0] if len(match) == 1 else None
                cik = record.get("CIK") if record is not None else None
                rows.append({
                    "cycle_key": cycle, "security_id": member,
                    "gsector": int(record["gsector"]) if record is not None else None,
                    "sector_name": record["GICS Sector"] if record is not None else None,
                    "company_name": record["Security"] if record is not None else None,
                    "cik": int(cik) if pd.notna(cik) and str(cik).strip().isdigit() else None,
                    "git_commit": sha, "git_commit_at": stamp, "source_url": csv_url,
                    "source_sha256": sha256(path),
                    "membership_snapshot": members["date"].date().isoformat(),
                    "membership_source_sha256": sha256(source),
                    "status": "DATED_SECONDARY_SECTOR_IDENTITY_OBSERVATION"
                              if record is not None else "MEMBERSHIP_SNAPSHOT_DISAGREEMENT",
                    "provenance_limit": "Not an authoritative effective-date interval; disagreements unresolved, no current backfill",
                })
            quarters.append({
                "cycle_key": cycle, "git_commit": sha, "git_commit_at": stamp,
                "metadata_count": len(frame), "membership_count": len(expected),
                "matching_securities": len(expected & observed),
                "missing_from_dated_metadata": sorted(expected - observed),
                "extra_in_dated_metadata": sorted(observed - expected),
                "status": "DATED_SOURCE_ACQUIRED_RECONCILIATION_REQUIRED",
            })
        except (RuntimeError, ValueError) as exc:
            quarters.append({"cycle_key": cycle, "status": "SOURCE_UNAVAILABLE_OR_INVALID", "error": str(exc)})
        write_csv(output / "quarter_source_status.csv", quarters)
        write_csv(output / "historical_security_sector_candidates.csv", rows)
    summary = {"quarters_requested": len(quarters),
               "quarters_acquired": sum(q["status"] != "SOURCE_UNAVAILABLE_OR_INVALID" for q in quarters),
               "dated_security_sector_observations": sum(r["gsector"] is not None for r in rows),
               "membership_disagreements": sum(r["gsector"] is None for r in rows),
               "authoritative_effective_date_intervals_claimed": False,
               "historical_validation": "NOT_VALIDATED"}
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2017Q2")
    parser.add_argument("--end", default="2026Q2")
    args = parser.parse_args()
    print(json.dumps(collect(args.output, args.start, args.end), indent=2))


if __name__ == "__main__":
    main()
