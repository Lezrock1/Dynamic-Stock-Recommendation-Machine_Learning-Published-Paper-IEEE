"""Daily adjusted-close price history via yfinance (free, deep history)."""
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "prices"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_price_history(ticker: str, start: str = "2005-01-01") -> pd.DataFrame | None:
    """Return a DataFrame with a single 'adj_close' column indexed by date."""
    cache_file = CACHE_DIR / f"{ticker}.csv"
    full_refresh_marker = CACHE_DIR / f"{ticker}.full_refresh"
    cached = None
    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        cache_age = time.time() - cache_file.stat().st_mtime
        if not full_refresh_marker.exists():
            full_refresh_marker.touch()
        full_refresh_due = time.time() - full_refresh_marker.stat().st_mtime >= 30 * 24 * 60 * 60
        if cache_age < 24 * 60 * 60 and not full_refresh_due and not cached.empty:
            latest_date = pd.Timestamp(cached.index.max()).normalize()
            if latest_date >= pd.Timestamp.today().normalize() - pd.Timedelta(days=4):
                return cached
        refresh_start = start if full_refresh_due else max(
            pd.Timestamp(start), pd.Timestamp(cached.index.max()).normalize() - pd.Timedelta(days=7)
        ).strftime("%Y-%m-%d")
    else:
        refresh_start = start
        full_refresh_due = True

    try:
        end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(ticker, start=refresh_start, end=end, auto_adjust=True, progress=False, threads=False)
    except Exception:
        return cached
    if raw is None or raw.empty:
        return cached

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    fresh = close.rename("adj_close").to_frame()
    fresh.index = pd.to_datetime(fresh.index)
    df = fresh if cached is None else pd.concat([cached, fresh]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
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
