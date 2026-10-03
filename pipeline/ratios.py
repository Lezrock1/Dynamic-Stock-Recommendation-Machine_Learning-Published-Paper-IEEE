"""Compute the paper's 20 fundamental ratios (X1..X20) + forward return label
from raw SEC EDGAR line items and price history.

NOTE: the original paper sourced pre-computed ratios directly from WRDS/Compustat;
the exact proprietary formulas were never published. The definitions below are
standard, commonly used equivalents for each ratio name and are a best-effort
reconstruction, not a guaranteed byte-for-byte match to the original methodology.
"""
import numpy as np
import pandas as pd

REBALANCE_MONTHS = [3, 6, 9, 12]


def _next_rebalance_date(available_date: pd.Timestamp) -> pd.Timestamp:
    for year_offset in (0, 1):
        for month in REBALANCE_MONTHS:
            candidate = pd.Timestamp(year=available_date.year + year_offset, month=month, day=1)
            if candidate > available_date:
                return candidate
    raise RuntimeError("unreachable")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Some sectors (financials, REITs, utilities) never file certain line items
    (e.g. no InventoryNet/COGS) - return an all-NaN series instead of KeyError
    so the ratios that don't apply to that sector just come out NaN."""
    if name in df.columns:
        return df[name]
    return pd.Series(float("nan"), index=df.index)


def compute_ratios(fundamentals: pd.DataFrame, prices: pd.DataFrame, tic: str,
                    filing_lag_days: int = 45) -> pd.DataFrame:
    """fundamentals: output of sec_edgar.get_quarterly_fundamentals (indexed by period_end).
    prices: output of prices.get_price_history (indexed by date, column adj_close).
    Returns one row per quarter with tic, tradedate, X1_REVGH..X20_DPO, y_return.
    """
    df = fundamentals.sort_index().copy()
    required = ["Revenue", "NetIncome", "Assets", "StockholdersEquity"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"missing required columns: {missing_required}")

    # SEC's "company facts" endpoint only exposes a sparse subset of the
    # CommonStockSharesOutstanding/EntityCommonStockSharesOutstanding history for
    # some companies (e.g. multi-share-class issuers). Fill gaps by carrying the
    # last known share count forward/back, and as a last resort imply share count
    # from NetIncome / EPS (both far more reliably reported every quarter).
    shares = _col(df, "SharesOutstanding").ffill()
    if "EPS_Basic" in df.columns:
        implied = df["NetIncome"] / df["EPS_Basic"].replace(0, float("nan"))
        shares = shares.fillna(implied)
    df["SharesOutstanding"] = shares
    if df["SharesOutstanding"].isna().all():
        raise ValueError("missing required columns: ['SharesOutstanding']")

    price_at_end = []
    approx_filed = df.index + pd.DateOffset(months=2)
    df["tradedate"] = [_next_rebalance_date(d) for d in approx_filed]
    for trade_date in df["tradedate"]:
        window = prices.loc[:trade_date]
        price_at_end.append(float(window["adj_close"].iloc[-1]) if not window.empty else None)
    df["price"] = price_at_end
    df["market_cap"] = df["price"] * df["SharesOutstanding"]

    out = pd.DataFrame(index=df.index)
    out["tic"] = tic
    out["datadate"] = df.index
    out["tradedate"] = df["tradedate"]

    revenue = df["Revenue"]
    cogs = _col(df, "COGS")
    long_term_debt = _col(df, "LongTermDebt").fillna(0)
    cash = _col(df, "CashAndEquivalents").fillna(0)
    inventory = _col(df, "InventoryNet").fillna(0)
    accounts_payable = _col(df, "AccountsPayableCurrent").fillna(0)

    out["X1_REVGH"] = revenue.pct_change(4)
    out["X2_EPS"] = _col(df, "EPS_Basic")
    out["X2_EPS"] = out["X2_EPS"].fillna(df["NetIncome"] / df["SharesOutstanding"])
    out["X3_ROA"] = df["NetIncome"] / df["Assets"]
    out["X4_ROE"] = df["NetIncome"] / df["StockholdersEquity"]
    ttm_eps = out["X2_EPS"].rolling(4, min_periods=4).sum()
    ttm_revenue = revenue.rolling(4, min_periods=4).sum()
    ttm_cfo = _col(df, "CFO").rolling(4, min_periods=4).sum()
    ttm_ebitda = (_col(df, "OperatingIncome") + _col(df, "DepreciationAmortization")).rolling(4, min_periods=4).sum()
    out["X5_PE"] = df["price"] / ttm_eps
    out["X6_PS"] = df["market_cap"] / ttm_revenue
    out["X7_NPM"] = df["NetIncome"] / revenue
    out["X8_GPM"] = (revenue - cogs) / revenue
    out["X9_OM"] = _col(df, "OperatingIncome") / revenue
    out["X10_PB"] = df["market_cap"] / df["StockholdersEquity"]
    out["X11_PCFO"] = df["market_cap"] / ttm_cfo
    out["X12_CR"] = cash / _col(df, "LiabilitiesCurrent")
    ev = df["market_cap"] + long_term_debt - cash
    out["X13_EM"] = ev / ttm_ebitda
    out["X14_EVCFO"] = ev / ttm_cfo
    out["X15_LTDTA"] = long_term_debt / df["Assets"]
    out["X16_WCR"] = (_col(df, "AssetsCurrent") - _col(df, "LiabilitiesCurrent")) / df["Assets"]
    out["X17_DE"] = _col(df, "Liabilities") / df["StockholdersEquity"]
    out["X18_QR"] = (_col(df, "AssetsCurrent") - inventory) / _col(df, "LiabilitiesCurrent")
    out["X19_DSI"] = inventory * 91 / cogs
    out["X20_DPO"] = accounts_payable * 91 / cogs

    price_by_tradedate = [
        float(prices.loc[:d, "adj_close"].iloc[-1]) if not prices.loc[:d].empty else None
        for d in out["tradedate"]
    ]
    out["_trade_price"] = price_by_tradedate
    next_trade_price = pd.Series(out["_trade_price"]).shift(-1)
    ratio = next_trade_price / out["_trade_price"]
    out["y_return"] = np.log(ratio.where(ratio > 0))
    out = out.drop(columns=["_trade_price"])

    return out.reset_index(drop=True)
