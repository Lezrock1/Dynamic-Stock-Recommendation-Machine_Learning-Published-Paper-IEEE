"""Archive the official SPY daily holdings file (State Street) with hash and parse per-stock weights.

    python benchmark_weights.py fetch        # downloads, archives xlsx + json under state/sources/
The weights are the fund's holdings on the file's as-of date, not necessarily the portfolio start date.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import requests
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

URL = ("https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/"
       "holdings-daily-us-en-spy.xlsx")
SOURCES = Path(__file__).resolve().parent.parent / "state" / "sources"
EQUITY_TICKER = re.compile(r"^[A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)*$")


def _position(row: pd.Series) -> dict:
    currency = row.get("Local Currency")
    return {
        "name": str(row["Name"]).strip(),
        "ticker": str(row["Ticker"]).strip(),
        "identifier": str(row["Identifier"]).strip() if pd.notna(row["Identifier"]) else None,
        "weight": float(row["Weight"]) / 100.0,
        "currency": str(currency).strip() if pd.notna(currency) else None,
    }


def parse_holdings(content: bytes) -> dict:
    raw = pd.read_excel(io.BytesIO(content), header=None)
    as_of = datetime.strptime(str(raw.iloc[2, 1]).replace("As of", "").strip(), "%d-%b-%Y").date().isoformat()
    header = raw.index[raw[0] == "Name"][0]
    table = raw.iloc[header + 1:].copy()
    table.columns = raw.iloc[header].tolist()
    table = table[table["Ticker"].notna() & pd.to_numeric(table["Weight"], errors="coerce").notna()]
    positions = [_position(row) for _, row in table.iterrows()]
    equity = [p for p in positions if EQUITY_TICKER.fullmatch(p["ticker"])]
    non_equity = [p for p in positions if not EQUITY_TICKER.fullmatch(p["ticker"])]
    weights = {p["ticker"]: p["weight"] for p in equity}
    return {
        "as_of": as_of,
        "weights": weights,
        "sum": sum(p["weight"] for p in positions),
        "equity_weight_sum": sum(weights.values()),
        "non_equity_positions": non_equity,
        "non_equity_weight_sum": sum(p["weight"] for p in non_equity),
        "holdings_count": len(positions),
    }


def fetch(destination: Path = SOURCES) -> Path:
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    content = response.content
    parsed = parse_holdings(content)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"spy_holdings_{parsed['as_of']}.xlsx").write_bytes(content)
    meta = {**parsed, "source_url": URL, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
            "fetched_at": datetime.now(timezone.utc).isoformat(), "kind": "SPY ETF holdings (not index constituents)"}
    path = destination / f"spy_holdings_{parsed['as_of']}.json"
    path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return path


def latest(destination: Path = SOURCES) -> dict | None:
    files = sorted(destination.glob("spy_holdings_*.json"))
    if not files:
        return None
    result = json.loads(files[-1].read_text())
    # Older archived metadata included cash and other instruments in ``weights``.
    if "non_equity_positions" not in result:
        raw_weights = result.get("weights", {})
        result["weights"] = {
            ticker: weight for ticker, weight in raw_weights.items()
            if EQUITY_TICKER.fullmatch(str(ticker))
        }
        result["non_equity_positions"] = [
            {"ticker": str(ticker), "weight": float(weight), "name": None}
            for ticker, weight in raw_weights.items()
            if not EQUITY_TICKER.fullmatch(str(ticker))
        ]
        result["equity_weight_sum"] = sum(result["weights"].values())
        result["non_equity_weight_sum"] = sum(
            item["weight"] for item in result["non_equity_positions"]
        )
    return result


def fetch_if_stale(destination: Path = SOURCES, max_age_seconds: int = 86400,
                   now: datetime | None = None) -> dict:
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")
    current = now or datetime.now(timezone.utc)
    existing = latest(destination)
    if existing and existing.get("fetched_at"):
        fetched_at = datetime.fromisoformat(str(existing["fetched_at"]))
        if fetched_at.tzinfo is None:
            raise ValueError("SPY holdings fetched_at must include a timezone")
        age = (current.astimezone(timezone.utc) - fetched_at.astimezone(timezone.utc)).total_seconds()
        if 0 <= age < max_age_seconds:
            return existing
    fetch(destination)
    refreshed = latest(destination)
    if refreshed is None:
        raise RuntimeError("SPY holdings refresh completed without publishing metadata")
    return refreshed


if __name__ == "__main__":
    print(fetch() if sys.argv[1:] == ["fetch"] else __doc__)
