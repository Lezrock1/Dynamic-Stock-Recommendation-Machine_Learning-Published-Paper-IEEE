"""Build point-in-time fundamental features and fixed-quarter total-return labels.

The SEC fundamentals input must retain filed/available timestamps. The Yahoo
price input separates raw close (valuation/features) from adjusted close
(return labels); adjusted prices are never treated as tradable share prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .market_calendar import (
        first_session_on_or_after,
        previous_nyse_session,
        session_close_time,
    )
except ImportError:
    from market_calendar import (  # type: ignore[no-redef]
        first_session_on_or_after,
        previous_nyse_session,
        session_close_time,
    )

REBALANCE_MONTHS = [3, 6, 9, 12]


def _next_rebalance_date(available_date: pd.Timestamp) -> pd.Timestamp:
    """First fixed quarterly execution date whose prior-session cutoff includes data."""
    available_date = pd.Timestamp(available_date).normalize()
    for year_offset in range(3):
        year = available_date.year + year_offset
        for month in REBALANCE_MONTHS:
            nominal = pd.Timestamp(year=year, month=month, day=1)
            if nominal < available_date:
                continue
            execution_date = first_session_on_or_after(nominal)
            signal_date = previous_nyse_session(execution_date)
            if signal_date >= available_date:
                return execution_date
    raise ValueError(f"could not find a quarterly session after {available_date.date()}")


def _next_quarterly_execution_date(execution_date: pd.Timestamp) -> pd.Timestamp:
    execution_date = pd.Timestamp(execution_date).normalize()
    for year_offset in range(3):
        year = execution_date.year + year_offset
        for month in REBALANCE_MONTHS:
            nominal = pd.Timestamp(year=year, month=month, day=1)
            if nominal <= execution_date:
                continue
            candidate = first_session_on_or_after(nominal)
            if candidate > execution_date:
                return candidate
    raise ValueError(f"could not find the next quarterly execution date after {execution_date.date()}")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(float("nan"), index=df.index)


def _consecutive_quarter_roll(series: pd.Series, period_ends: pd.DatetimeIndex) -> pd.Series:
    result = series.rolling(4, min_periods=4).sum()
    dates = pd.DatetimeIndex(period_ends)
    for index in range(3, len(result)):
        quarter_gaps = [(dates[i] - dates[i - 1]).days for i in range(index - 2, index + 1)]
        total_span = (dates[index] - dates[index - 3]).days
        if not all(70 <= gap <= 120 for gap in quarter_gaps) or not 250 <= total_span <= 390:
            result.iloc[index] = np.nan
    return result


def _year_over_year_growth(series: pd.Series, period_ends: pd.DatetimeIndex) -> pd.Series:
    prior = series.shift(4)
    growth = series / prior - 1.0
    dates = pd.DatetimeIndex(period_ends)
    for index in range(4, len(growth)):
        if not 330 <= (dates[index] - dates[index - 4]).days <= 400:
            growth.iloc[index] = np.nan
        else:
            gaps = [(dates[i] - dates[i - 1]).days for i in range(index - 2, index + 1)]
            if not all(70 <= gap <= 120 for gap in gaps):
                growth.iloc[index] = np.nan
    return growth


def _exact_price(prices: pd.DataFrame, column: str, date: pd.Timestamp) -> float:
    normalized = pd.DatetimeIndex(prices.index).normalize()
    rows = prices.loc[normalized == pd.Timestamp(date).normalize(), column]
    if rows.empty or pd.isna(rows.iloc[-1]):
        return float("nan")
    value = float(rows.iloc[-1])
    return value if np.isfinite(value) and value > 0 else float("nan")


def compute_ratios(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    tic: str,
    filing_lag_days: int = 0,
    price_session: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build model features without using data unavailable at the signal cutoff.

    ``available_at`` must be the latest conservative availability date for the
    facts in each row. Reports arriving after the preceding session's close
    move to the next fixed quarter; no quarter-end-plus-two-month proxy is used.
    ``filing_lag_days`` is an optional additional delay, never an availability
    shortcut.
    """
    if filing_lag_days < 0:
        raise ValueError("filing_lag_days must be non-negative")
    if fundamentals.empty:
        return pd.DataFrame()
    if "available_at" not in fundamentals.columns:
        raise ValueError("fundamentals must retain point-in-time `available_at` metadata")
    if not {"close", "adj_close"}.issubset(prices.columns):
        raise ValueError("prices must contain separate raw `close` and total-return `adj_close` columns")

    df = fundamentals.sort_index().copy()
    # Revenue and share counts are not universal: a missing value stays NaN (never 0) and only
    # the features that need it become NaN. Industry-specific applicability is decided upstream.
    required = ["NetIncome", "Assets", "StockholdersEquity"]
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise ValueError(f"missing required columns: {missing_required}")
    period_ends = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()

    # Shares are not carried indefinitely from an older filing. EPS-derived
    # shares are used only when both period values are present and non-zero.
    shares = _col(df, "SharesOutstanding")
    if "EPS_Basic" in df.columns:
        implied = df["NetIncome"] / df["EPS_Basic"].replace(0, np.nan)
        shares = shares.where(shares.notna(), implied)
    df["SharesOutstanding"] = shares

    available_at = pd.to_datetime(df["available_at"], errors="coerce")
    if available_at.isna().any():
        raise ValueError("fundamental rows contain missing or invalid availability dates")
    if filing_lag_days:
        available_at = available_at + pd.to_timedelta(filing_lag_days, unit="D")

    trade_dates = [_next_rebalance_date(day) for day in available_at]
    signal_dates = [previous_nyse_session(day) for day in trade_dates]
    if price_session is not None:
        # Pending cycle: the natural signal day lies after the snapshot, so the
        # latest closed session is used and no later price is read.
        cap = pd.Timestamp(price_session).normalize()
        signal_dates = [min(day, cap) for day in signal_dates]
    df["price"] = [_exact_price(prices, "close", day) for day in signal_dates]
    df["market_cap"] = df["price"] * df["SharesOutstanding"]

    out = pd.DataFrame(index=df.index)
    out["tic"] = tic
    out["datadate"] = period_ends
    out["available_at"] = available_at.to_numpy()
    out["signal_as_of"] = [session_close_time(day) for day in signal_dates]
    out["signal_time"] = out["signal_as_of"]
    out["execution_time"] = [session_close_time(day) for day in trade_dates]
    out["tradedate"] = trade_dates
    out["expected_period"] = [
        f"{pd.Timestamp(period_end).year}-period-ending-{pd.Timestamp(period_end).date()}"
        for period_end in period_ends
    ]

    revenue = _col(df, "Revenue")
    cogs = _col(df, "COGS")
    long_term_debt = _col(df, "LongTermDebt")
    cash = _col(df, "CashAndEquivalents")
    inventory = _col(df, "InventoryNet")
    accounts_payable = _col(df, "AccountsPayableCurrent")

    out["X1_REVGH"] = _year_over_year_growth(revenue, period_ends)
    out["X2_EPS"] = _col(df, "EPS_Basic")
    out["X2_EPS"] = out["X2_EPS"].where(
        out["X2_EPS"].notna(), df["NetIncome"] / df["SharesOutstanding"]
    )
    out["X3_ROA"] = df["NetIncome"] / df["Assets"]
    out["X4_ROE"] = df["NetIncome"] / df["StockholdersEquity"]
    ttm_eps = _consecutive_quarter_roll(out["X2_EPS"], period_ends)
    ttm_revenue = _consecutive_quarter_roll(revenue, period_ends)
    ttm_cfo = _consecutive_quarter_roll(_col(df, "CFO"), period_ends)
    ttm_ebitda = _consecutive_quarter_roll(
        _col(df, "OperatingIncome") + _col(df, "DepreciationAmortization"),
        period_ends,
    )
    out["X5_PE"] = df["price"] / ttm_eps
    out["X6_PS"] = df["market_cap"] / ttm_revenue
    out["X7_NPM"] = df["NetIncome"] / revenue
    out["X8_GPM"] = (revenue - cogs) / revenue
    out["X9_OM"] = _col(df, "OperatingIncome") / revenue
    out["X10_PB"] = df["market_cap"] / df["StockholdersEquity"]
    out["X11_PCFO"] = df["market_cap"] / ttm_cfo
    out["X12_CR"] = _col(df, "AssetsCurrent") / _col(df, "LiabilitiesCurrent")
    ev = df["market_cap"] + long_term_debt - cash
    out["X13_EM"] = ev / ttm_ebitda
    out["X14_EVCFO"] = ev / ttm_cfo
    out["X15_LTDTA"] = long_term_debt / df["Assets"]
    out["X16_WCR"] = (_col(df, "AssetsCurrent") - _col(df, "LiabilitiesCurrent")) / df["Assets"]
    out["X17_DE"] = _col(df, "Liabilities") / df["StockholdersEquity"]
    out["X18_QR"] = (_col(df, "AssetsCurrent") - inventory) / _col(df, "LiabilitiesCurrent")
    out["X19_DSI"] = inventory * 91 / cogs
    out["X20_DPO"] = accounts_payable * 91 / cogs

    execution_prices = [_exact_price(prices, "adj_close", date) for date in trade_dates]
    label_end_dates = [_next_quarterly_execution_date(date) for date in trade_dates]
    label_end_prices = [_exact_price(prices, "adj_close", date) for date in label_end_dates]
    start_prices = pd.Series(execution_prices, index=out.index, dtype=float)
    end_prices = pd.Series(label_end_prices, index=out.index, dtype=float)
    valid_labels = start_prices.gt(0) & end_prices.gt(0)
    out["y_return"] = np.nan
    out.loc[valid_labels, "y_return"] = (
        end_prices.loc[valid_labels] / start_prices.loc[valid_labels] - 1.0
    )
    out["label_end_time"] = [session_close_time(date) for date in label_end_dates]
    out["label_known_at"] = out["label_end_time"]
    out["label_return_type"] = "simple_adjusted_close_total_return_proxy"
    return out.reset_index(drop=True)
