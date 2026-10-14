"""Archive the official SPY daily holdings file (State Street) with hash and parse per-stock weights.

    python benchmark_weights.py fetch        # downloads, archives xlsx + json under state/sources/
The weights are the fund's holdings on the file's as-of date, not necessarily the portfolio start date.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import requests
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

URL = ("https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/"
       "holdings-daily-us-en-spy.xlsx")
SOURCES = Path(__file__).resolve().parent.parent / "state" / "sources"


def parse_holdings(content: bytes) -> dict:
    raw = pd.read_excel(io.BytesIO(content), header=None)
    as_of = datetime.strptime(str(raw.iloc[2, 1]).replace("As of", "").strip(), "%d-%b-%Y").date().isoformat()
    header = raw.index[raw[0] == "Name"][0]
    table = raw.iloc[header + 1:].copy()
    table.columns = raw.iloc[header].tolist()
    table = table[table["Ticker"].notna() & pd.to_numeric(table["Weight"], errors="coerce").notna()]
    weights = {str(r.Ticker).strip(): float(r.Weight) / 100.0 for r in table.itertuples()}
    return {"as_of": as_of, "weights": weights, "sum": sum(weights.values())}


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
    return json.loads(files[-1].read_text()) if files else None


if __name__ == "__main__":
    print(fetch() if sys.argv[1:] == ["fetch"] else __doc__)
