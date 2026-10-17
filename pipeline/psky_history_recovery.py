"""Isolated, read-only public-source acquisition for PSKY history recovery."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from pipeline.market_calendar import nyse_sessions


SOURCES = {
    "company-merger.html": (
        "https://ir.paramount.com/news-releases/news-release-details/"
        "skydance-media-and-paramount-global-complete-merger-creating"
    ),
    "company-closing.pdf": "https://ir.paramount.com/node/71456/pdf",
    "nasdaq-identity.html": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2025-414",
    "yahoo-psky-max.json": (
        "https://query1.finance.yahoo.com/v8/finance/chart/PSKY"
        "?range=max&interval=1d&events=div%2Csplits"
    ),
    "yahoo-para-window.json": (
        "https://query1.finance.yahoo.com/v8/finance/chart/PARA"
        "?period1=1754524800&period2=1788220800&interval=1d&events=div%2Csplits"
    ),
    "stooq-psky.csv": "https://stooq.com/q/d/l/?s=psky.us&i=d&d1=20250807&d2=20260831",
    "stockanalysis-history.html": "https://stockanalysis.com/stocks/psky/history/?range=max",
    "company-stock.html": "https://ir.paramount.com/stock-information",
    "company-current.html": "https://ir.skydance.com/",
    "nasdaq-ticker-change.html": "https://www.nasdaqtrader.com/TraderNews.aspx?id=2026-697",
    "yahoo-skyd-window.json": (
        "https://query1.finance.yahoo.com/v8/finance/chart/SKYD"
        "?period1=1754524800&period2=1788220800&interval=1d&events=div%2Csplits"
    ),
    "company-merger-distribution.html": (
        "https://www.prnewswire.com/news-releases/skydance-media-and-paramount-global-complete-"
        "merger-creating-next-generation-media-company-302524343.html"
    ),
    "nasdaq-ticker-change-detail.html": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-710",
    "occ-ticker-change.html": "https://infomemo.theocc.com/infomemos?number=59894",
    "company-current-fwp.pdf": (
        "https://ir.skydance.com/static-files/bafd3d44-1454-440f-8e0a-94f58e6347f3"
    ),
    "company-new-merger-distribution.html": (
        "https://www.prnewswire.com/news-releases/paramount-skydance-and-warner-bros-"
        "discovery-announce-anticipated-closing-date-of-paramount-merger-302895095.html"
    ),
    "company-wbd-merger.html": (
        "https://ir.wbd.com/news-and-events/financial-news/financial-news-details/2026/"
        "Paramount-Skydance-and-Warner-Bros--Discovery-Announce-Anticipated-Closing-Date-"
        "of-Paramount-Merger/default.aspx"
    ),
    "company-warrants.html": (
        "https://www.skydance.com/news/press/skydance-provides-details-on-separate-"
        "trading-of-skyd-class-b-shares-and-warrants"
    ),
    "company-name-change-filing.html": "https://ir.paramount.com/node/73326/html",
    "occ-ticker-change.pdf": "https://infomemo.theocc.com/infomemo/search-memo?memo=59894",
    "company-warrants-distribution.html": (
        "https://www.prnewswire.com/news-releases/skydance-provides-details-on-separate-"
        "trading-of-skyd-class-b-shares-and-warrants-302909067.html"
    ),
    "company-name-change-quotemedia.html": (
        "https://app.quotemedia.com/data/downloadFiling?webmasterId=90423&ref=320369478"
        "&type=HTML&symbol=PSKY&cdn=1f376ba09808362368ecfd506e465c90"
        "&companyName=Paramount+Skydance+Corporation&formType=8-K"
        "&formDescription=Current+report+pursuant+to+Section+13+or+15%28d%29"
        "&dateFiled=2026-09-25"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def acquire(destination: Path, name: str, url: str) -> dict:
    """Preserve response bytes and timestamps, including HTTP error responses."""
    requested_at = utc_now()
    body_path = destination / name
    header_path = destination / (name + ".headers")
    try:
        response = subprocess.run(
            [
                "curl", "--silent", "--show-error", "--location",
                "--proto", "=https", "--proto-redir", "=https",
                "--max-time", "45", "--user-agent", "Mozilla/5.0",
                "--dump-header", str(header_path), "--output", str(body_path),
                "--write-out", "%{json}", url,
            ],
            capture_output=True, text=True, check=False, timeout=50,
        )
        details = json.loads(response.stdout) if response.stdout else {}
        body = body_path.read_bytes() if body_path.exists() else b""
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {
            "file": name,
            "source_url": url,
            "requested_at_utc": requested_at,
            "acquired_at_utc": utc_now(),
            "error": str(exc),
        }
    acquired_at = utc_now()
    return {
        "file": name,
        "source_url": url,
        "final_url": details.get("url_effective"),
        "requested_at_utc": requested_at,
        "acquired_at_utc": acquired_at,
        "http_status": details.get("http_code"),
        "curl_exit_code": response.returncode,
        "error": response.stderr,
        "response_headers_file": header_path.name,
        "response_headers_sha256": (
            hashlib.sha256(header_path.read_bytes()).hexdigest() if header_path.exists() else None
        ),
        "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def parse_chart(payload: dict) -> tuple[dict[str, float], dict]:
    """Extract only daily, same-provider close; reject malformed or split-bearing data."""
    chart = payload["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError("provider chart error or empty result")
    result = chart["result"][0]
    meta = result["meta"]
    if meta["symbol"] != "SKYD" or meta["currency"] != "USD":
        raise ValueError("unexpected symbol or currency")
    if meta["dataGranularity"] != "1d":
        raise ValueError("provider did not return daily observations")
    if meta.get("instrumentType") != "EQUITY":
        raise ValueError("unexpected instrument type")
    events = result.get("events", {})
    if events.get("splits"):
        raise ValueError("split events require an explicit reviewed adjustment rule")
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    if len(timestamps) != len(closes):
        raise ValueError("timestamp/close array lengths differ")
    prices = {}
    for stamp, close in zip(timestamps, closes):
        day = datetime.fromtimestamp(stamp, ZoneInfo("America/New_York")).date().isoformat()
        if day in prices:
            raise ValueError(f"duplicate session: {day}")
        if close is None or not isinstance(close, (int, float)) or isinstance(close, bool):
            raise ValueError(f"missing or nonnumeric close: {day}")
        if not math.isfinite(close) or close <= 0:
            raise ValueError(f"invalid close: {day}")
        prices[day] = close
    if list(prices) != sorted(prices):
        raise ValueError("unsorted provider timestamps")
    return prices, events


def exact_levels(prices: dict[str, float], cutoff: str = "2026-08-31") -> list[tuple[str, float]]:
    sessions = nyse_sessions("2025-08-07", cutoff)
    if len(sessions) < 253:
        raise ValueError("calendar window has fewer than 253 sessions")
    expected = [day.date().isoformat() for day in sessions[-253:]]
    missing = [day for day in expected if day not in prices]
    if missing:
        raise ValueError(f"missing required closes: {missing}")
    allowed = {day.date().isoformat() for day in sessions}
    if any(day <= cutoff and day not in allowed for day in prices):
        raise ValueError("off-calendar or pre-listing observations")
    return [(day, prices[day]) for day in expected]


def build_bundle(inputs: list[Path], output: Path) -> None:
    """Reproduce a research bundle from hash-verified acquired evidence, offline."""
    required = {
        "yahoo-skyd-window.json", "nasdaq-identity.html", "nasdaq-ticker-change.html",
        "company-merger-distribution.html", "company-warrants-distribution.html",
        "company-name-change-quotemedia.html",
    }
    available = {}
    for directory in inputs:
        manifest = json.loads((directory / "acquisition.json").read_text())
        records = manifest["sources"] if isinstance(manifest, dict) else manifest
        for record in records:
            name = record["file"]
            if name not in required or record.get("http_status") != 200:
                continue
            source = directory / "sources" / name
            body = source.read_bytes()
            if hashlib.sha256(body).hexdigest() != record["sha256"]:
                raise ValueError(f"source hash mismatch: {source}")
            header_name = record.get("response_headers_file")
            header_body = None
            if header_name:
                header_file = directory / "sources" / header_name
                if header_file.exists():
                    header_body = header_file.read_bytes()
                    if hashlib.sha256(header_body).hexdigest() != record["response_headers_sha256"]:
                        raise ValueError(f"response header hash mismatch: {header_file}")
            available[name] = (record, body, header_body)
    if required - available.keys():
        raise ValueError(f"missing acquired evidence: {sorted(required - available.keys())}")
    prices, events = parse_chart(json.loads(available["yahoo-skyd-window.json"][1]))
    levels = exact_levels(prices)
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    records = []
    for name, (record, body, header_body) in sorted(available.items()):
        (sources / name).write_bytes(body)
        if header_body is not None:
            (sources / record["response_headers_file"]).write_bytes(header_body)
        records.append(record)
    (output / "acquisition.json").write_text(json.dumps(records, indent=2) + "\n")
    with (output / "PSKY_253_provider_close.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["date", "provider_close"])
        writer.writerows(levels)
    event_rows = []
    for kind, values in events.items():
        for item in values.values():
            day = datetime.fromtimestamp(
                item["date"], ZoneInfo("America/New_York")
            ).date().isoformat()
            event_rows.append({"kind": kind, "date": day, "provider_event": item})
    (output / "source_events.json").write_text(json.dumps({
        "provider_symbol": "SKYD", "historical_security_symbol": "PSKY",
        "events": event_rows, "splits_reported": False,
        "independent_split_absence_verification": False,
        "later_event_excluded": {
            "event": "warrant distribution", "record_date": "2026-10-05",
            "distribution_date": "2026-10-13", "ex_date": "2026-10-14",
            "company_disclosed_price_adjustment": 0.19368,
            "source": "company-warrants-distribution.html",
            "reason": "after 2026-08-31 cutoff; no adjustment applied to provider_close",
        },
    }, indent=2) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir()) if path.is_file()
    }
    (output / "reproduction.json").write_text(json.dumps({
        "computed_at_utc": utc_now(),
        "levels": len(levels), "returns": len(levels) - 1,
        "start": levels[0][0], "end": levels[-1][0],
        "calendar": "NYSE", "provider": "Yahoo Finance",
        "provider_query_symbol": "SKYD",
        "historical_symbol": "PSKY",
        "class": "Class B common stock", "cusip": "69932A204",
        "basis": "indicators.quote[0].close; NOT indicators.adjclose",
        "concatenation": False, "filling": False, "truncation_of_required_window": False,
        "source_files_sha256": {r["file"]: r["sha256"] for r in records},
        "files_sha256": hashes,
        "reproduce_command": (
            ".venv/bin/python -m pipeline.psky_history_recovery --build-from "
            f"{output} --output NEW_DIRECTORY"
        ),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", choices=sorted(SOURCES))
    parser.add_argument("--build-from", action="append", type=Path)
    args = parser.parse_args()
    if args.build_from:
        build_bundle(args.build_from, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output / "sources"
    sources.mkdir()
    with ThreadPoolExecutor(max_workers=8) as pool:
        selected = {key: SOURCES[key] for key in (args.source or SOURCES)}
        records = list(pool.map(lambda item: acquire(sources, *item), selected.items()))
    manifest = {"computed_at_utc": utc_now(), "sources": records}
    (args.output / "acquisition.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for record in records:
        print(record["file"], record.get("http_status"), record.get("size"), record.get("error", ""))


if __name__ == "__main__":
    main()
