"""Reconcile a dated constituent snapshot using explicitly dated primary index changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd

from .fixed_lag_research import offline_guard, write_csv, write_json
from .historical_universe_sources import parse_constituents, SECTORS
from .return_comparison import sha256


def changes(document: bytes) -> list[dict]:
    rows = []
    for table in BeautifulSoup(document, "lxml").find_all("table"):
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td", recursive=False)]
            if len(cells) != 6 or cells[1] != "S&P 500":
                continue
            if cells[2] not in {"Addition", "Deletion"} or cells[5] not in SECTORS:
                raise ValueError(f"Unsupported primary index change: {cells}")
            rows.append(dict(zip(("effective_date", "index", "action", "company_name", "ticker", "sector"), cells)))
    if not rows:
        raise ValueError("No explicit S&P 500 change table in primary announcement")
    for row in rows:
        row["effective_date"] = pd.Timestamp(row["effective_date"]).date().isoformat()
    return sorted(rows, key=lambda r: (r["effective_date"], r["ticker"]))


def reconcile(source: Path, announcement: Path, identities: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    files = [source / "historical_security_sector_candidates.csv", source / "sources/constituents/2026Q2.csv",
             source / "sources/commits/2026Q2.json", announcement, identities / "security_identities.json",
             Path(__file__)]
    hashes = {str(path): sha256(path) for path in files}
    attempts = []
    with offline_guard(attempts):
        candidates = pd.read_csv(files[0])
        candidates = candidates[candidates["cycle_key"] == "2026Q2"].copy()
        original = parse_constituents(files[1])
        metadata = json.loads(files[2].read_text())[0]
        observed = set(original["security_id"])
        identity_rows = json.loads(files[4].read_text())
        applied = []
        for event in changes(announcement.read_bytes()):
            date = pd.Timestamp(event["effective_date"])
            if not pd.Timestamp(metadata["commit"]["committer"]["date"]).tz_localize(None).normalize() < date <= pd.Timestamp("2026-06-30"):
                continue
            ticker = event["ticker"]
            if event["action"] == "Deletion":
                if ticker not in observed:
                    raise ValueError(f"Primary deletion absent from earlier constituent snapshot: {ticker}")
                observed.remove(ticker)
            else:
                if ticker in observed:
                    raise ValueError(f"Primary addition already in earlier constituent snapshot: {ticker}")
                observed.add(ticker)
                match = candidates["security_id"] == ticker
                if match.sum() != 1:
                    raise ValueError(f"Primary addition not uniquely expected at Q2 cutoff: {ticker}")
                candidates.loc[match, ["gsector", "sector_name", "company_name", "status"]] = [
                    SECTORS[event["sector"]], event["sector"], event["company_name"],
                    "PRIMARY_DATED_INDEX_ADDITION"]
                issuer = [r for r in identity_rows if r["ticker"] == ticker
                          and r["status"] == "ORIGINAL_COVER_TICKER_ENTITY_MATCH"]
                if len(issuer) == 1:
                    candidates.loc[match, "cik"] = issuer[0]["cik"]
                candidates.loc[match, "source_sha256"] = sha256(announcement)
                candidates.loc[match, "source_url"] = "https://press.spglobal.com/2026-06-23-Honeywell-Aerospace-Set-to-Join-S-P-500-S-P-100-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600"
            applied.append(event)
        if observed != set(candidates["security_id"]) or candidates["gsector"].isna().any():
            raise ValueError("Primary changes do not fully reconcile the expected Q2 security set")
        summary = {
            "cycle_key": "2026Q2", "universe_as_of": "2026-06-30", "securities": len(candidates),
            "dated_sector_observations": int(candidates["gsector"].notna().sum()),
            "applied_primary_changes": applied, "all_historical_intervals_verified": False,
            "note": "Q2 observation only; original snapshot sectors retain secondary-source status. No Q3 overrides or backward identity fill.",
            "network_attempts": attempts, "historical_validation": "NOT_VALIDATED",
        }
        if hashes != {str(path): sha256(path) for path in files}:
            raise ValueError("Index reconciliation input changed during calculation")
        write_csv(output / "q2_security_sector_evidence.csv", candidates.to_dict("records"))
        write_json(output / "summary.json", summary)
        write_json(output / "input_manifest.json", hashes)
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "announcement", "identities", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.source, args.announcement, args.identities, args.output), indent=2))


if __name__ == "__main__":
    main()
