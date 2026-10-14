"""Reconstruct S&P 500 membership for a past reference date from dated, cited change records.

Starting point is the current constituent list. Every change dated after ``as_of`` is
reversed (additions removed, removals restored). The change log is the public
"Historical components of the S&P 500" table; its wikitext hash is returned as evidence.
Tickers that cannot be mapped to a CIK and sector raise instead of being dropped.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from . import universe as uv
except ImportError:  # pragma: no cover - script usage
    import universe as uv  # type: ignore

CHANGES_API = ("https://en.wikipedia.org/w/api.php?action=parse&page=Historical_components_of_the_S%26P_500"
               "&prop=wikitext&format=json")
CACHE = Path(__file__).resolve().parent.parent / "cache" / "sp500_changes.json"
YAHOO_TO_GICS = {
    "Basic Materials": 15, "Communication Services": 50, "Consumer Cyclical": 25, "Consumer Defensive": 30,
    "Energy": 10, "Financial Services": 40, "Healthcare": 35, "Industrials": 20, "Real Estate": 60,
    "Technology": 45, "Utilities": 55,
}
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")


def fetch_changes_wikitext(max_age_seconds: float = 6 * 3600) -> str:
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < max_age_seconds:
        return json.loads(CACHE.read_text())["wikitext"]
    try:
        response = requests.get(CHANGES_API, headers=uv.HEADERS, timeout=60)
        response.raise_for_status()
        text = response.json()["parse"]["wikitext"]["*"]
    except (requests.RequestException, KeyError, ValueError):
        if CACHE.exists():
            return json.loads(CACHE.read_text())["wikitext"]
        raise
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"wikitext": text}))
    return text


def parse_changes(wikitext: str) -> pd.DataFrame:
    """Rows: date, added, removed (empty string if none)."""
    rows = []
    for chunk in wikitext.split("\n|-"):
        cells = [c.strip() for c in re.split(r"\s*\|\|\s*|\n\s*\|(?!\|)", chunk.strip().lstrip("|"))]
        if len(cells) < 5:
            continue
        date = pd.to_datetime(cells[0], format="%B %d, %Y", errors="coerce")
        if pd.isna(date):
            continue
        added, removed = cells[1], cells[3]
        rows.append({
            "date": date,
            "added": added if _TICKER.match(added) else "",
            "removed": removed if _TICKER.match(removed) else "",
        })
    return pd.DataFrame(rows, columns=["date", "added", "removed"])


def members_as_of(as_of: str | pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Members on ``as_of`` (inclusive of changes effective on that day) plus evidence."""
    day = pd.Timestamp(as_of).normalize()
    universe = uv.get_sp500_universe()
    current = universe[universe["is_current"].astype(bool)].copy()
    wikitext = fetch_changes_wikitext()
    changes = parse_changes(wikitext)
    later = changes[changes["date"] > day]
    added_after = {t for t in later["added"] if t}
    removed_after = {t for t in later["removed"] if t}

    keep = current[~current["tic"].isin(added_after)].copy()
    dated_after = current[pd.to_datetime(current["date_added"], errors="coerce") > day]["tic"]
    unexplained = sorted(set(dated_after) - added_after)
    if unexplained:
        raise ValueError(f"members added after {day.date()} without a change record: {unexplained}")
    keep = keep[~keep["tic"].isin(dated_after)]

    restored, unmapped = [], []
    sec_map = uv._get_sec_ticker_map()
    former = universe[~universe["is_current"].astype(bool)].set_index("tic")
    for ticker in sorted(removed_after - set(keep["tic"])):
        if ticker in former.index:
            row = former.loc[ticker]
            restored.append({**row.to_dict(), "tic": ticker, "is_current": True,
                             "data_source": "restored_from_change_log"})
        elif ticker in sec_map:
            unmapped.append(ticker)
        else:
            unmapped.append(ticker)
    if unmapped:
        # Restored members need a sector; take it from the historical Compustat sector table.
        sectors = pd.read_excel(Path(uv.__file__).resolve().parent.parent / "Data" / "fundamental_final_table.xlsx",
                                usecols=["tic", "gsector", "conm"])
        sectors["tic"] = sectors["tic"].map(uv._normalize_ticker)
        for ticker in list(unmapped):
            info = sec_map.get(ticker)
            match = sectors[sectors["tic"] == ticker].dropna(subset=["gsector"])
            if info is None:
                continue
            if match.empty:
                import yfinance as yf
                code = YAHOO_TO_GICS.get(str(yf.Ticker(ticker).info.get("sector")))
                source = "restored_from_change_log;sector=yahoo_mapped_approximation"
            else:
                code, source = int(match["gsector"].mode().iloc[0]), "restored_from_change_log"
            if code is None:
                continue
            restored.append({
                "tic": ticker, "conm": info["title"], "gsector": code,
                "cik": int(info["cik_str"]), "date_added": pd.NA, "is_current": True,
                "data_source": source,
            })
            unmapped.remove(ticker)
    if unmapped:
        raise ValueError(f"removed after {day.date()} but no CIK/sector mapping: {unmapped}")
    result = pd.concat([keep, pd.DataFrame(restored)], ignore_index=True) if restored else keep
    evidence = {
        "as_of": day.date().isoformat(),
        "change_log": CHANGES_API,
        "change_log_sha256": hashlib.sha256(wikitext.encode()).hexdigest(),
        "reversed_additions": sorted(added_after & set(current["tic"])),
        "restored_removals": sorted(r["tic"] for r in restored),
        "member_count": int(len(result)),
    }
    return result.reset_index(drop=True), evidence
