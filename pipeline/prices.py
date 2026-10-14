"""Raw close and dividend-adjusted close via yfinance.

Raw closes support valuation and share-count based market capitalization.
Adjusted closes are reserved for return labels and must not be booked as cash.
"""
from pathlib import Path
import time
import warnings

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "prices"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


MIN_RAW_CLOSE_COVERAGE = 0.9


def get_price_history(ticker: str, start: str = "2005-01-01") -> pd.DataFrame | None:
    """Return `close` and `adj_close`; force a refresh for obsolete cache schema."""
    cache_file = CACHE_DIR / f"{ticker}.csv"
    full_refresh_marker = CACHE_DIR / f"{ticker}.full_refresh"
    cached = None
    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        cache_age = time.time() - cache_file.stat().st_mtime
        if not full_refresh_marker.exists():
            full_refresh_marker.touch()
        full_refresh_due = time.time() - full_refresh_marker.stat().st_mtime >= 30 * 24 * 60 * 60
        if "close" not in cached.columns or cached["close"].notna().mean() < MIN_RAW_CLOSE_COVERAGE:
            # Caches written before the raw close was stored hold NaN for old sessions; the
            # incremental 7-day refresh would never repair them, so force a full refresh.
            full_refresh_due = True
        if (
            cache_age < 24 * 60 * 60
            and not full_refresh_due
            and not cached.empty
            and {"close", "adj_close"}.issubset(cached.columns)
        ):
            latest_date = pd.Timestamp(cached.index.max()).normalize()
            if latest_date >= pd.Timestamp.today().normalize() - pd.Timedelta(days=4):
                return cached
        refresh_start = start if full_refresh_due else max(
            pd.Timestamp(start), pd.Timestamp(cached.index.max()).normalize() - pd.Timedelta(days=7)
        ).strftime("%Y-%m-%d")
    else:
        refresh_start = start
        full_refresh_due = True

    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker, start=refresh_start, end=end, auto_adjust=False, progress=False, threads=False
    )
    if raw is None or raw.empty:
        if cached is not None and not cached.empty:
            warnings.warn(f"{ticker}: price refresh returned no rows; using cached data", RuntimeWarning)
            return cached
        return None

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    adj_close = raw["Adj Close"]
    if isinstance(adj_close, pd.DataFrame):
        adj_close = adj_close.iloc[:, 0]
    fresh = pd.concat([
        close.rename("close"),
        adj_close.rename("adj_close"),
    ], axis=1)
    fresh.index = pd.to_datetime(fresh.index)
    # Fresh rows take precedence (an unstable sort of concatenated duplicates once kept stale NaN rows).
    df = fresh if cached is None else fresh.combine_first(cached)
    df = df.sort_index()[["close", "adj_close"]]
    df.to_csv(cache_file)
    if full_refresh_due:
        full_refresh_marker.touch()
    return df


def price_on_or_before(price_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """Latest available adjusted close on or before `date` (falls back a few days for holidays)."""
    window = price_df.loc[:date]
    if window.empty:
        return None
    return float(window["adj_close"].iloc[-1])
