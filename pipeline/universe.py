"""Current S&P 500 constituent list (free, from Wikipedia) mapped to GICS sector
codes that match the sector numbering used in the original paper's data files
(sector10=Energy ... sector60=Real Estate).
"""
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HISTORICAL_MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
CACHE_FILE = Path(__file__).resolve().parent.parent / "cache" / "sp500_universe.csv"
HISTORICAL_MEMBERSHIP_CACHE = Path(__file__).resolve().parent.parent / "cache" / "sp500_historical_snapshots.csv"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_TICKERS_CACHE = Path(__file__).resolve().parent.parent / "cache" / "sec_company_tickers.json"
# Use the same configurable identification for Wikipedia and SEC requests.
USER_AGENT = os.getenv(
    "DYNAMIC_STOCK_USER_AGENT",
    "dynamic-stock-recommendation-research/1.0 (configure DYNAMIC_STOCK_USER_AGENT)",
)
HEADERS = {"User-Agent": USER_AGENT}

# GICS sector name -> 2-digit sector code, matching Data/1-focasting_data/sectorXX_clean.xlsx
GICS_SECTOR_CODE = {
    "Energy": 10,
    "Materials": 15,
    "Industrials": 20,
    "Consumer Discretionary": 25,
    "Consumer Staples": 30,
    "Health Care": 35,
    "Financials": 40,
    "Information Technology": 45,
    "Communication Services": 50,  # formerly "Telecommunication Services"
    "Utilities": 55,
    "Real Estate": 60,
}

GICS_SECTOR_NAME = {value: key for key, value in GICS_SECTOR_CODE.items()}


def _normalize_ticker(value: str) -> str:
    return str(value).strip().upper().replace(".", "-")


def _get_sec_ticker_map(force_refresh: bool = False) -> dict[str, dict]:
    fresh = SEC_TICKERS_CACHE.exists() and (
        pd.Timestamp.now().timestamp() - SEC_TICKERS_CACHE.stat().st_mtime < 7 * 24 * 60 * 60
    )
    if force_refresh or not fresh:
        try:
            response = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=30)
            response.raise_for_status()
            SEC_TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            SEC_TICKERS_CACHE.write_text(response.text)
        except requests.RequestException:
            if not SEC_TICKERS_CACHE.exists():
                raise
    records = json.loads(SEC_TICKERS_CACHE.read_text()).values()
    return {_normalize_ticker(row["ticker"]): row for row in records}


def get_sp500_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Current S&P constituents plus SEC-mappable former constituents since 2017."""
    if CACHE_FILE.exists() and not force_refresh and pd.Timestamp.now().timestamp() - CACHE_FILE.stat().st_mtime < 7 * 24 * 60 * 60:
        cached = pd.read_csv(CACHE_FILE)
        if "is_current" in cached.columns and "data_source" in cached.columns:
            return cached

    try:
        resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        if CACHE_FILE.exists():
            return pd.read_csv(CACHE_FILE)
        raise
    table = pd.read_html(io.StringIO(resp.text))[0]

    df = table.rename(columns={
        "Symbol": "tic",
        "Security": "conm",
        "GICS Sector": "gsector_name",
        "CIK": "cik",
        "Date added": "date_added",
    })[["tic", "conm", "gsector_name", "cik", "date_added"]]

    df["tic"] = df["tic"].map(_normalize_ticker)
    df["gsector"] = df["gsector_name"].map(GICS_SECTOR_CODE)
    df["cik"] = df["cik"].astype(int)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["is_current"] = True
    df["data_source"] = "current_wikipedia"
    df = df.dropna(subset=["gsector"]).reset_index(drop=True)

    current_tickers = set(df["tic"])
    membership = get_historical_membership_snapshots()
    post_2017 = membership[membership["date"] >= pd.Timestamp("2017-09-01")]
    historical_tickers = {
        _normalize_ticker(ticker)
        for row in post_2017["tickers"].dropna()
        for ticker in str(row).split(",")
        if str(ticker).strip()
    } - current_tickers

    historical_table = Path(__file__).resolve().parent.parent / "Data" / "fundamental_final_table.xlsx"
    historical_sectors = pd.read_excel(historical_table, usecols=["tic", "gsector"])
    historical_sectors["tic"] = historical_sectors["tic"].map(_normalize_ticker)
    sector_by_ticker = historical_sectors.groupby("tic")["gsector"].agg(
        lambda values: values.dropna().mode().iloc[0] if not values.dropna().mode().empty else pd.NA
    ).to_dict()
    sec_by_ticker = _get_sec_ticker_map()
    former_rows = []
    for ticker in sorted(historical_tickers):
        filing_entity = sec_by_ticker.get(ticker)
        sector_code = sector_by_ticker.get(ticker)
        if filing_entity is None or pd.isna(sector_code) or int(sector_code) not in GICS_SECTOR_NAME:
            continue
        former_rows.append({
            "tic": ticker,
            "conm": filing_entity["title"],
            "gsector_name": GICS_SECTOR_NAME[int(sector_code)],
            "gsector": int(sector_code),
            "cik": int(filing_entity["cik_str"]),
            "date_added": pd.NA,
            "is_current": False,
            "data_source": "historical_snapshots_sec_cik_compustat_gics",
        })
    if former_rows:
        df = pd.concat([df, pd.DataFrame(former_rows)], ignore_index=True)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
    return df


def get_historical_membership_snapshots(force_refresh: bool = False) -> pd.DataFrame:
    """Get dated S&P constituent snapshots; this public history starts in 1996.

    Its maintainer documents incomplete early history and occasional membership gaps;
    callers should preserve the snapshot date and audit unmatched symbols.
    """
    cache_fresh = (
        HISTORICAL_MEMBERSHIP_CACHE.exists()
        and pd.Timestamp.now().timestamp() - HISTORICAL_MEMBERSHIP_CACHE.stat().st_mtime < 7 * 24 * 60 * 60
    )
    if force_refresh or not cache_fresh:
        try:
            response = requests.get(HISTORICAL_MEMBERSHIP_URL, headers=HEADERS, timeout=60)
            response.raise_for_status()
            HISTORICAL_MEMBERSHIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
            HISTORICAL_MEMBERSHIP_CACHE.write_bytes(response.content)
        except requests.RequestException:
            if not HISTORICAL_MEMBERSHIP_CACHE.exists():
                raise

    snapshots = pd.read_csv(HISTORICAL_MEMBERSHIP_CACHE, dtype={"date": str, "tickers": str})
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce")
    snapshots = snapshots.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return snapshots.reset_index(drop=True)


def apply_historical_membership_filter(
    frame: pd.DataFrame,
    trade_date_column: str = "tradedate",
    ticker_column: str = "tic",
    cutoff: str = "2017-09-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the post-Cutoff extension to constituents active on each trade date."""
    snapshots = get_historical_membership_snapshots()
    snapshot_dates = pd.DatetimeIndex(snapshots["date"])
    snapshot_symbols = [
        frozenset(symbol.strip().replace(".", "-") for symbol in str(value).split(",") if symbol.strip())
        for value in snapshots["tickers"]
    ]
    cutoff_date = pd.Timestamp(cutoff)
    kept_frames = []
    audit_rows = []

    for trade_date, group in frame.groupby(trade_date_column, sort=True):
        trade_date = pd.Timestamp(trade_date)
        if trade_date < cutoff_date:
            kept_frames.append(group)
            continue

        snapshot_index = snapshot_dates.searchsorted(trade_date, side="right") - 1
        if snapshot_index < 0:
            members = frozenset()
            snapshot_date = pd.NaT
        else:
            members = snapshot_symbols[snapshot_index]
            snapshot_date = snapshot_dates[snapshot_index]
        normalized_tickers = group[ticker_column].astype(str).str.strip().str.replace(".", "-", regex=False)
        mask = normalized_tickers.isin(members)
        retained = group.loc[mask]
        excluded_tickers = sorted(set(normalized_tickers.loc[~mask]))
        audit_rows.append({
            "trade_date": trade_date,
            "membership_snapshot_date": snapshot_date,
            "snapshot_age_days": (trade_date - snapshot_date).days if pd.notna(snapshot_date) else pd.NA,
            "tickers_before_filter": int(normalized_tickers.nunique()),
            "tickers_in_snapshot": len(members),
            "tickers_retained": int(normalized_tickers.loc[mask].nunique()),
            "tickers_excluded": int(normalized_tickers.loc[~mask].nunique()),
            "excluded_ticker_symbols": ",".join(excluded_tickers),
        })
        if not retained.empty:
            kept_frames.append(retained)

    filtered = pd.concat(kept_frames, ignore_index=True) if kept_frames else frame.iloc[0:0].copy()
    return filtered, pd.DataFrame(audit_rows)
