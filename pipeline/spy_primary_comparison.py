"""Reconstruct an explicitly labelled SPY NAV total-return proxy from issuer sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fixed_lag_research import write_json
from .historical_evidence import EvidenceStore, ROOT, now
from .market_calendar import nyse_sessions, session_close_time
from .return_comparison import import_series, sha256

BASE = "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
PAGE = "https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy"
NAV_FILE = "navhist-us-en-spy.xlsx"
DIV_FILE = "spdr-etf-historical-distributions.xlsx"


def nav_levels(path: Path) -> pd.Series:
    frame = pd.read_excel(path, sheet_name="navhist", header=3)
    if not {"Date", "NAV"}.issubset(frame):
        raise ValueError("Issuer NAV workbook schema changed")
    dated = frame["Date"].astype(str).str.fullmatch(r"\d{2}-[A-Za-z]{3}-\d{4}")
    frame = frame.loc[dated, ["Date", "NAV"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], format="%d-%b-%Y", errors="raise")
    frame["NAV"] = pd.to_numeric(frame["NAV"], errors="raise")
    if frame["Date"].duplicated().any() or frame.empty:
        raise ValueError("Duplicate or empty issuer NAV history")
    if not np.isfinite(frame["NAV"]).all() or (frame["NAV"] <= 0).any():
        raise ValueError("Issuer NAV levels must be positive and finite")
    return frame.set_index("Date")["NAV"].sort_index()


def distributions(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="dividend")
    required = {"TICKER", "CUSIP", "EX-DATE", "RECORD DATE", "PAYABLE DATE", "DIVIDEND ($)",
                "SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)"}
    if not required.issubset(frame):
        raise ValueError("Issuer distribution workbook schema changed")
    frame = frame.loc[frame["TICKER"].str.strip() == "SPY"].copy()
    if frame.empty or set(frame["CUSIP"].astype(str)) != {"78462F103"}:
        raise ValueError("SPY issuer/security identity not proven in distribution workbook")
    for field in ("EX-DATE", "RECORD DATE", "PAYABLE DATE"):
        frame[field] = pd.to_datetime(frame[field], format="%m/%d/%Y", errors="raise")
    if frame["EX-DATE"].duplicated().any():
        raise ValueError("Duplicate distribution ex-date; review rather than double count")
    if (frame["PAYABLE DATE"] < frame["EX-DATE"]).any():
        raise ValueError("Pay date precedes ex date")
    fields = ["DIVIDEND ($)", "SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)"]
    for field in fields:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    if frame["DIVIDEND ($)"].isna().any():
        raise ValueError("Unknown distribution amounts may not become zero")
    # Blank capital-gain columns in this issuer table mean no capital-gain distribution.
    frame["cash_distribution"] = frame[fields].fillna(0).sum(axis=1)
    if not np.isfinite(frame["cash_distribution"]).all() or (frame["cash_distribution"] < 0).any():
        raise ValueError("Invalid cash distributions")
    return frame.sort_values("EX-DATE").reset_index(drop=True)


def reconstruct(nav: pd.Series, events: pd.DataFrame, start: str,
                nyse_only: bool = False) -> tuple[pd.DataFrame, dict]:
    last_completed = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    nav = nav[(nav.index >= pd.Timestamp(start)) & (nav.index < last_completed)]
    if len(nav) < 2:
        raise ValueError("Insufficient fully completed NAV sessions")
    expected = nyse_sessions(nav.index[0], nav.index[-1])
    excluded = nav.loc[nav.index.difference(expected)]
    if nyse_only:
        if events["EX-DATE"].isin(excluded.index).any():
            raise ValueError("A distribution falls on an excluded non-NYSE valuation; source review required")
        nav = nav.loc[nav.index.intersection(expected)]
    if not nav.index.equals(expected):
        missing = expected.difference(nav.index)
        extra = nav.index.difference(expected)
        raise ValueError(f"Issuer NAV session gaps/non-sessions: missing={list(missing[:8])}, extra={list(extra[:8])}")
    applicable = events[(events["EX-DATE"] > nav.index[0]) & (events["EX-DATE"] <= nav.index[-1])]
    if not applicable["EX-DATE"].isin(nav.index).all():
        raise ValueError("A distribution ex-date has no NAV; cannot reconstruct total return")
    # SPY is quarterly. Missing quarters in the source must not silently imply zero dividends.
    additional_distributions = {}
    for period in pd.period_range(nav.index[0], nav.index[-1], freq="Q"):
        if period.start_time < nav.index[0] or period.end_time.normalize() > nav.index[-1]:
            continue
        count = applicable["EX-DATE"].dt.to_period("Q").eq(period).sum()
        if count == 0:
            raise ValueError(f"SPY quarterly distribution coverage unresolved: {period}, count={count}")
        if count > 1:
            additional_distributions[str(period)] = int(count)
    cash = applicable.set_index("EX-DATE")["cash_distribution"].reindex(nav.index, fill_value=0)
    factors = (nav + cash).div(nav.shift(1))
    factors.iloc[0] = 1.0
    levels = factors.cumprod() * 100
    points = pd.DataFrame({
        "date": nav.index.strftime("%Y-%m-%d"), "value": levels.to_numpy(),
        "valued_at": [session_close_time(day).tz_convert("UTC").isoformat() for day in nav.index],
    })
    review = {
        "method": "NAV ex-date analytical reinvestment: L[t]=L[t-1]*(NAV[t]+cash_distribution[t])/NAV[t-1]",
        "not_adjusted_close": True, "not_official_published_total_return_index": True,
        "not_exchange_execution_price": True,
        "source": "State Street issuer NAV and explicit distribution records including capital gains",
        "start": points["date"].iloc[0], "end": points["date"].iloc[-1],
        "observations": len(points), "included_distributions": len(applicable),
        "quarters_with_additional_source_distributions": additional_distributions,
        "calendar_filter": "Explicit NYSE-only sampling of issuer NAV; no filling or fabricated levels"
                           if nyse_only else "Exact source/session match required",
        "excluded_non_nyse_source_valuations": {
            day.date().isoformat(): float(value) for day, value in excluded.items()} if nyse_only else {},
        "nav_unit_actions": "No automatic split inference. Any unit discontinuity requires separate source review.",
        "fee_treatment": "Fund fees/internal fund effects already in NAV; no additional deduction. Before personal tax.",
        "tax": "Gross distributions before individual investor withholding/taxes; not a tax-free claim",
        "reinvestment": "Analytical at ex-date NAV; not the actual pay-date cash ledger and not an investable trade claim",
        "historical_strategy_validation": "NOT_VALIDATED",
    }
    return points, review


def acquire(output: Path, start: str = "2017-08-31", publish: bool = False,
            cached_source: Path | None = None, nyse_only: bool = False) -> dict:
    store = EvidenceStore(output)
    sources = [
        (NAV_FILE, BASE + NAV_FILE), (DIV_FILE, BASE + DIV_FILE),
        ("spy_issuer_page.html", PAGE),
        ("issuer_distribution_page.html",
         "https://www.ssga.com/us/en/intermediary/resources/documents/etf-dividend-distributions"),
    ]
    acquired = []
    if cached_source is not None:
        manifest = json.loads((cached_source / "artifact_manifest.json").read_text())
        for name, _ in sources:
            source = cached_source / "sources" / name
            if manifest.get("sources/" + name) != sha256(source):
                raise ValueError(f"Cached primary source hash mismatch: {name}")
            acquired.append(store.copy_cache(source, name))
        write_json(output / "inherited_source_manifest.json", {
            "manifest_sha256": sha256(cached_source / "artifact_manifest.json"),
            "acquisition_journal_sha256": sha256(cached_source / "acquisition.json"),
            "original_acquisition": json.loads((cached_source / "acquisition.json").read_text()),
            "local_recheck_at": now(), "not_a_new_issuer_download": True,
        })
    else:
        acquired = [store.fetch(name, url) for name, url in sources]
    nav_path, div_path, page, dividend_page = acquired
    nav = nav_levels(nav_path)
    events = distributions(div_path)
    points, review = reconstruct(nav, events, start, nyse_only)
    jumps = nav.pct_change().abs()
    review["nav_large_moves_requiring_review"] = {
        day.date().isoformat(): float(change) for day, change in jumps[jumps > 0.30].items()
    }
    if review["nav_large_moves_requiring_review"]:
        write_json(output / "review.json", review)
        raise ValueError("NAV unit/action discontinuity requires review before publication")
    points.to_csv(output / "levels.csv", index=False)
    events.to_csv(output / "issuer_spy_distributions.csv", index=False)
    write_json(output / "review.json", review)
    identifier = "spy-issuer-nav-tr-" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    evidence = [nav_path, div_path, page, dividend_page, output / "review.json"]
    if cached_source is not None:
        evidence.append(output / "inherited_source_manifest.json")
    metadata = {
        "instrument_id": identifier,
        "name": "SPY ETF-NAV Total Return · S&P-500-Proxy · Ex-Tag-Rekonstruktion",
        "instrument_type": "etf", "identifiers": {"ticker": ["SPY"], "isin": ["US78462F1030"]},
        "share_class": "SPDR S&P 500 ETF Trust; CUSIP 78462F103; NAV not exchange execution",
        "exchange": "NYSE Arca instrument; issuer NAV at US equity close",
        "currency": "USD", "return_type": "etf_total_return",
        "dividend_tax_basis": "before_personal_tax", "reinvestment_rule": "ex_date",
        "cost_basis": "Fund expenses/internal effects embedded in NAV. No simulated investor trade costs.",
        "source": "State Street primary issuer NAV and dividend/capital distribution history; reconstructed analytic TR",
        "source_url": PAGE, "valuation_convention": "NYSE_close_UTC", "calendar": "NYSE",
        "review_status": "SOURCE_REVIEWED", "observed_at": now(),
        "evidence_files": [{"path": str(p.relative_to(output)), "sha256": sha256(p)} for p in evidence],
        "source_review": review,
    }
    write_json(output / "metadata.json", metadata)
    result = {"status": "PRIMARY_SOURCE_TR_RECONSTRUCTED", **review, "instrument_id": identifier}
    if publish:
        result["registry_file"] = str(import_series(ROOT, output / "levels.csv", output / "metadata.json"))
    write_json(output / "summary.json", result)
    write_json(output / "artifact_manifest.json", {
        str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2017-08-31")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--cached-source", type=Path)
    parser.add_argument("--nyse-only", action="store_true",
                        help="Explicitly sample issuer NAV only on NYSE sessions; list all excluded source valuations")
    args = parser.parse_args()
    print(json.dumps(acquire(args.output, args.start, args.publish, args.cached_source, args.nyse_only), indent=2))


if __name__ == "__main__":
    main()
