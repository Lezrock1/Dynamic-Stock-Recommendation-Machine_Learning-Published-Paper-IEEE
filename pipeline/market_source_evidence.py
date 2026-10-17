"""Acquire source-labelled daily closes and actions without certifying provider completeness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from .fixed_lag_research import write_csv, write_json
from .historical_evidence import EvidenceStore, ROOT, MANIFEST, now
from .market_calendar import nyse_sessions
from .return_comparison import sha256


def parse_chart(document: dict, ticker: str) -> tuple[pd.DataFrame, list[dict], dict]:
    chart = document.get("chart", {})
    if chart.get("error") or len(chart.get("result") or []) != 1:
        raise ValueError(f"{ticker}: chart API error or ambiguous response")
    result = chart["result"][0]
    meta = result.get("meta", {})
    if meta.get("symbol", "").replace(".", "-").upper() != ticker.upper():
        raise ValueError(f"{ticker}: market response security mismatch")
    if meta.get("currency") != "USD":
        raise ValueError(f"{ticker}: unsupported market currency {meta.get('currency')}")
    stamps = pd.to_datetime(result.get("timestamp", []), unit="s", utc=True)
    dates = stamps.tz_convert("America/New_York").tz_localize(None).normalize()
    quotes = result.get("indicators", {}).get("quote", [])
    if (len(dates) == 0 or len(quotes) != 1 or not isinstance(quotes[0], dict)
            or "close" not in quotes[0] or len(quotes[0]["close"]) != len(dates)):
        raise ValueError(f"{ticker}: incomplete daily close response")
    frame = pd.DataFrame({"provider_close": quotes[0]["close"]}, index=dates).sort_index()
    frame.index.name = "date"
    if frame.index.has_duplicates or frame.empty:
        raise ValueError(f"{ticker}: duplicate/empty session dates")
    close = pd.to_numeric(frame["provider_close"], errors="raise")
    if np.isinf(close).any() or (close.dropna() <= 0).any():
        raise ValueError(f"{ticker}: invalid provider closes")
    frame["provider_close"] = close
    actions = []
    for kind, events in result.get("events", {}).items():
        for identifier, event in events.items():
            stamp = pd.Timestamp(event["date"], unit="s", tz="UTC")
            actions.append({"source_event_id": identifier, "kind": kind, "ticker": ticker,
                            "date": stamp.tz_convert("America/New_York").date().isoformat(),
                            "raw_event": event,
                            "pay_date": None,
                            "status": "PROVIDER_EVENT_NOT_COMPLETE_CORPORATE_ACTION_CERTIFICATION"})
    expected = nyse_sessions(frame.index[0], frame.index[-1])
    summary = {"ticker": ticker, "currency": meta["currency"], "source_symbol": meta.get("symbol"),
               "first_date": frame.index[0].date().isoformat(), "last_date": frame.index[-1].date().isoformat(),
               "observations": len(frame), "null_closes": int(close.isna().sum()),
               "missing_sessions": [d.date().isoformat() for d in expected.difference(frame.index)],
               "non_sessions": [d.date().isoformat() for d in frame.index.difference(expected)],
               "dividend_events": sum(a["kind"] == "dividends" for a in actions),
               "split_events": sum(a["kind"] == "splits" for a in actions),
               "other_events": sum(a["kind"] not in {"dividends", "splits"} for a in actions),
               "source_review_status": "ACQUIRED_NOT_RAW_AS_TRADED_OR_ACTION_COVERAGE_CERTIFIED",
               "unresolved": ["historical share-class/security identifier intervals",
                              "provider close split basis and nonstandard distributions/spinoffs",
                              "complete pay dates, merger proceeds and delisting settlements",
                              "independent provider completeness/return-definition reconciliation"]}
    return frame, actions, summary


def collect(output: Path, tickers: list[str], start: str = "2005-01-01") -> dict:
    store = EvidenceStore(output)
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int(pd.Timestamp.now(tz="UTC").timestamp())
    results, all_actions = [], []
    for ticker in tickers:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}"
               f"?period1={period1}&period2={period2}&interval=1d&events=div%2Csplits%2CcapitalGains")
        try:
            source = store.fetch(f"charts/{ticker}.json", url)
            frame, actions, summary = parse_chart(json.loads(source.read_text()), ticker)
            # Today's incomplete bar is not an eligible completed valuation.
            today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
            frame = frame[frame.index < today]
            if frame.empty:
                raise ValueError(f"{ticker}: no completed daily price")
            summary["source_observations_including_incomplete"] = summary["observations"]
            summary["source_last_date_including_incomplete"] = summary["last_date"]
            expected = nyse_sessions(frame.index[0], frame.index[-1])
            summary.update({
                "first_date": frame.index[0].date().isoformat(),
                "last_date": frame.index[-1].date().isoformat(),
                "observations": len(frame),
                "null_closes": int(frame["provider_close"].isna().sum()),
                "missing_sessions": [d.date().isoformat() for d in expected.difference(frame.index)],
                "non_sessions": [d.date().isoformat() for d in frame.index.difference(expected)],
            })
            (output / "prices").mkdir(exist_ok=True)
            frame.to_csv(output / "prices" / f"{ticker}.csv")
            summary["last_completed_date"] = frame.index[-1].date().isoformat()
            summary["source_sha256"] = sha256(source)
            summary["observed_at"] = now()
            all_actions.extend(actions)
        except (ValueError, RuntimeError) as exc:
            summary = {"ticker": ticker, "source_review_status": "ACQUISITION_OR_SCHEMA_FAILED", "error": str(exc)}
        results.append(summary)
        write_csv(output / "security_market_coverage.csv", results)
    write_json(output / "security_market_coverage.json", results)
    write_json(output / "provider_action_events.json", all_actions)
    summary = {"status": "SOURCE_COLLECTION_NOT_MARKET_CERTIFICATION", "securities_requested": len(tickers),
               "acquired": sum(r["source_review_status"] != "ACQUISITION_OR_SCHEMA_FAILED" for r in results),
               "failed": [r for r in results if r["source_review_status"] == "ACQUISITION_OR_SCHEMA_FAILED"],
               "action_events": len(all_actions), "historical_validation": "NOT_VALIDATED",
               "raw_price_and_full_action_bundles_certified": 0}
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--tickers", nargs="+")
    args = parser.parse_args()
    tickers = args.tickers or [
        s["ticker"] for s in json.loads((ROOT / MANIFEST).read_text())["securities"]]
    print(json.dumps(collect(args.output, tickers, args.start), indent=2))


if __name__ == "__main__":
    main()
