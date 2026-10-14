"""Self-financing long-only portfolio simulation on raw, unadjusted prices.

Prices must use a consistent split-adjusted share basis and must not include
dividend adjustments. Corporate actions are supplied separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class BacktestDataError(ValueError):
    """Raised when the supplied market data cannot support a valid valuation."""


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost_bps_per_side: float = 10.0
    spread_slippage_bps_per_side: float = 0.0
    fractional_shares: bool = False
    cash_rate_annual: float = 0.0
    max_weight_per_asset: float | None = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        for name in ("transaction_cost_bps_per_side", "spread_slippage_bps_per_side"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.cash_rate_annual) or self.cash_rate_annual <= -1:
            raise ValueError("cash_rate_annual must be finite and greater than -1")
        if self.max_weight_per_asset is not None and not 0 < self.max_weight_per_asset <= 1:
            raise ValueError("max_weight_per_asset must be in (0, 1]")


@dataclass(frozen=True)
class CorporateAction:
    action_id: str
    ticker: str
    ex_date: str | pd.Timestamp
    kind: str
    value: float
    pay_date: str | pd.Timestamp | None = None


@dataclass
class BacktestResult:
    nav: pd.Series
    daily_returns: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    dividend_receivables: pd.Series
    initial_capital: float
    valid: bool = True


def simulate_portfolio(
    prices: pd.DataFrame,
    target_weights: Mapping[str | pd.Timestamp, Mapping[str, float]],
    *,
    config: BacktestConfig | None = None,
    corporate_actions: Sequence[CorporateAction] = (),
) -> BacktestResult:
    """Simulate cash, shares, actions, fees, and daily NAV without borrowing.

    ``prices`` contains raw close prices. A missing price for a held security is
    a hard error; prices are never forward-filled. Target weights are executed
    at that date's close and are long-only. Costs include both the configured
    transaction fee and the separately configured spread/slippage estimate.
    """
    config = config or BacktestConfig()
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise BacktestDataError("prices must have a non-empty DatetimeIndex")
    if prices.index.has_duplicates:
        raise BacktestDataError("price dates must be unique")
    if prices.columns.has_duplicates:
        raise BacktestDataError("price ticker columns must be unique")
    prices = prices.sort_index().astype(float)
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    if prices.index.has_duplicates:
        raise BacktestDataError("price dates must remain unique after date normalization")
    if (prices.where(prices.notna()) <= 0).any().any():
        raise BacktestDataError("all supplied prices must be positive when present")
    if np.isinf(prices.to_numpy()).any():
        raise BacktestDataError("prices cannot contain infinite values")

    normalized_targets: dict[pd.Timestamp, dict[str, float]] = {}
    for raw_date, weights in target_weights.items():
        rebalance_date = pd.Timestamp(raw_date).normalize()
        if rebalance_date in normalized_targets:
            raise BacktestDataError(f"duplicate target date: {rebalance_date.date()}")
        if rebalance_date not in prices.index:
            raise BacktestDataError(f"target date is not a supplied trading date: {rebalance_date.date()}")
        normalized: dict[str, float] = {}
        for ticker, weight in weights.items():
            ticker = str(ticker)
            value = float(weight)
            if not math.isfinite(value) or value < 0:
                raise BacktestDataError(f"invalid target weight for {ticker}: {weight}")
            if config.max_weight_per_asset is not None and value > config.max_weight_per_asset + 1e-12:
                raise BacktestDataError(f"{ticker} target weight exceeds configured maximum")
            if value > 0 and ticker not in prices.columns:
                raise BacktestDataError(f"no price series supplied for target security {ticker}")
            normalized[ticker] = value
        if sum(normalized.values()) > 1.0 + 1e-10:
            raise BacktestDataError(f"target weights exceed 100% on {rebalance_date.date()}")
        normalized_targets[rebalance_date] = normalized

    action_ids: set[str] = set()
    actions_by_date: dict[pd.Timestamp, list[CorporateAction]] = {}
    for action in corporate_actions:
        if not action.action_id or action.action_id in action_ids:
            raise BacktestDataError(f"empty or duplicate corporate action id: {action.action_id!r}")
        action_ids.add(action.action_id)
        if action.kind not in {"split", "dividend", "cash_delisting"}:
            raise BacktestDataError(f"unsupported corporate action kind: {action.kind}")
        if not math.isfinite(action.value) or action.value <= 0:
            raise BacktestDataError(f"corporate action value must be positive: {action.action_id}")
        if action.ticker not in prices.columns:
            raise BacktestDataError(f"no price series supplied for corporate action security {action.ticker}")
        ex_date = pd.Timestamp(action.ex_date).normalize()
        if ex_date not in prices.index:
            raise BacktestDataError(f"corporate action date is not a supplied trading date: {ex_date.date()}")
        if action.kind == "dividend" and action.pay_date is None:
            raise BacktestDataError(f"dividend {action.action_id} needs a pay_date")
        if action.kind == "cash_delisting" and action.pay_date is not None:
            raise BacktestDataError("cash_delisting value is settled on ex_date; omit pay_date")
        actions_by_date.setdefault(ex_date, []).append(action)

    positions: dict[str, float] = {}
    cash = config.initial_capital
    receivables: dict[str, tuple[str, pd.Timestamp, float]] = {}
    nav_values: list[float] = []
    cash_values: list[float] = []
    receivable_values: list[float] = []
    position_rows: list[dict[str, float]] = []
    trade_rows: list[dict[str, object]] = []
    last_date: pd.Timestamp | None = None
    total_cost_rate = (
        config.transaction_cost_bps_per_side + config.spread_slippage_bps_per_side
    ) / 10_000.0

    def trade(date_value: pd.Timestamp, ticker: str, delta: float, price: float) -> None:
        nonlocal cash
        if abs(delta) < 1e-12:
            return
        notional = abs(delta) * price
        fee = notional * total_cost_rate
        if delta < 0:
            if -delta > positions.get(ticker, 0.0) + 1e-10:
                raise BacktestDataError(f"trade would create a short position in {ticker}")
            cash += notional - fee
        else:
            required = notional + fee
            if required > cash + 1e-8:
                raise BacktestDataError(f"trade would make cash negative while buying {ticker}")
            cash -= required
        positions[ticker] = positions.get(ticker, 0.0) + delta
        if abs(positions[ticker]) < 1e-10:
            positions.pop(ticker, None)
        trade_rows.append({
            "date": date_value,
            "ticker": ticker,
            "side": "BUY" if delta > 0 else "SELL",
            "shares": abs(delta),
            "execution_price": price,
            "notional": notional,
            "transaction_cost": fee,
            "transaction_cost_bps_per_side": config.transaction_cost_bps_per_side,
            "spread_slippage_bps_per_side": config.spread_slippage_bps_per_side,
        })

    for current_date, row in prices.iterrows():
        current_date = pd.Timestamp(current_date).normalize()
        if last_date is not None and config.cash_rate_annual:
            elapsed_days = (current_date - last_date).days
            cash *= (1.0 + config.cash_rate_annual) ** (elapsed_days / 365.0)

        todays_actions = actions_by_date.get(current_date, [])
        for action in sorted(todays_actions, key=lambda item: (item.kind != "split", item.action_id)):
            shares = positions.get(action.ticker, 0.0)
            if shares <= 0:
                continue
            if action.kind == "split":
                positions[action.ticker] = shares * action.value
            elif action.kind == "dividend":
                receivable_id = action.action_id
                if receivable_id in receivables:
                    raise BacktestDataError(f"duplicate dividend accrual: {receivable_id}")
                pay_date = pd.Timestamp(action.pay_date).normalize()
                if pay_date < current_date:
                    raise BacktestDataError(f"dividend pay_date precedes ex_date: {receivable_id}")
                receivables[receivable_id] = (
                    action.ticker, pay_date, shares * action.value
                )
            elif action.kind == "cash_delisting":
                proceeds = shares * action.value
                cash += proceeds
                positions.pop(action.ticker, None)

        for action_id, (_, pay_date, amount) in list(receivables.items()):
            if pay_date == current_date:
                cash += amount
                del receivables[action_id]

        if current_date in normalized_targets:
            targets = normalized_targets[current_date]
            tickers = set(positions) | set(targets)
            nav_before_trade = cash + sum(
                positions.get(ticker, 0.0) * float(row[ticker])
                for ticker in positions
                if pd.notna(row.get(ticker, np.nan))
            ) + sum(amount for _, _, amount in receivables.values())
            current_prices: dict[str, float] = {}
            for ticker in tickers:
                price = row.get(ticker, np.nan)
                if pd.isna(price) or not math.isfinite(float(price)) or float(price) <= 0:
                    raise BacktestDataError(
                        f"missing/invalid execution price for {ticker} on {current_date.date()}"
                    )
                current_prices[ticker] = float(price)
            desired_shares: dict[str, float] = {}
            for ticker in tickers:
                target_value = nav_before_trade * targets.get(ticker, 0.0)
                raw_shares = target_value / current_prices[ticker]
                desired_shares[ticker] = (
                    raw_shares if config.fractional_shares else float(math.floor(raw_shares + 1e-12))
                )
            deltas = {
                ticker: desired_shares.get(ticker, 0.0) - positions.get(ticker, 0.0)
                for ticker in tickers
            }
            for ticker in sorted(tickers):
                if deltas[ticker] < 0:
                    trade(current_date, ticker, deltas[ticker], current_prices[ticker])

            buy_tickers = [ticker for ticker in sorted(tickers) if deltas[ticker] > 0]
            requested_cost = sum(
                deltas[ticker] * current_prices[ticker] * (1.0 + total_cost_rate)
                for ticker in buy_tickers
            )
            scale = min(1.0, cash / requested_cost) if requested_cost > 0 else 1.0
            for ticker in buy_tickers:
                shares_to_buy = deltas[ticker] * scale
                if not config.fractional_shares:
                    shares_to_buy = float(math.floor(shares_to_buy + 1e-12))
                if shares_to_buy > 0:
                    trade(current_date, ticker, shares_to_buy, current_prices[ticker])

        held_market_value = 0.0
        row_positions: dict[str, float] = {}
        for ticker, shares in positions.items():
            price = row.get(ticker, np.nan)
            if pd.isna(price) or not math.isfinite(float(price)) or float(price) <= 0:
                raise BacktestDataError(
                    f"missing/invalid valuation price for held security {ticker} on {current_date.date()}"
                )
            held_market_value += shares * float(price)
            row_positions[ticker] = shares
        receivable_total = sum(amount for _, _, amount in receivables.values())
        nav = cash + held_market_value + receivable_total
        if cash < -1e-7:
            raise BacktestDataError(f"cash became negative on {current_date.date()}: {cash}")
        if nav <= 0 or not math.isfinite(nav):
            raise BacktestDataError(f"invalid portfolio NAV on {current_date.date()}: {nav}")
        nav_values.append(nav)
        cash_values.append(max(cash, 0.0))
        receivable_values.append(receivable_total)
        position_rows.append(row_positions)
        last_date = current_date

    index = prices.index
    nav_series = pd.Series(nav_values, index=index, name="nav", dtype=float)
    daily_returns = nav_series.pct_change(fill_method=None).rename("daily_return")
    if len(daily_returns):
        daily_returns.iloc[0] = nav_series.iloc[0] / config.initial_capital - 1.0
    trades = pd.DataFrame(trade_rows, columns=[
        "date", "ticker", "side", "shares", "execution_price", "notional",
        "transaction_cost", "transaction_cost_bps_per_side",
        "spread_slippage_bps_per_side",
    ])
    if not trades.empty:
        trades = trades.sort_values(["date", "ticker", "side"]).reset_index(drop=True)
    return BacktestResult(
        nav=nav_series,
        daily_returns=daily_returns,
        cash=pd.Series(cash_values, index=index, name="cash", dtype=float),
        positions=pd.DataFrame(position_rows, index=index).fillna(0.0),
        trades=trades,
        dividend_receivables=pd.Series(
            receivable_values, index=index, name="dividend_receivables", dtype=float
        ),
        initial_capital=config.initial_capital,
    )


def total_return_metrics(
    nav: pd.Series,
    *,
    initial_capital: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Daily-NAV metrics; the initial missing percentage-change is excluded."""
    clean = nav.dropna().astype(float).sort_index()
    if len(clean) < 2:
        raise ValueError("at least two valid NAV observations are required")
    if clean.index.has_duplicates or (clean <= 0).any():
        raise ValueError("NAV index must be unique and values must be positive")
    returns = clean.pct_change(fill_method=None).dropna()
    years = (clean.index[-1] - clean.index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    starting_value = float(clean.iloc[0] if initial_capital is None else initial_capital)
    if not math.isfinite(starting_value) or starting_value <= 0:
        raise ValueError("initial_capital must be finite and positive")
    if initial_capital is not None:
        returns = pd.concat([
            pd.Series([clean.iloc[0] / starting_value - 1.0], index=[clean.index[0]]),
            returns,
        ])
    total_return = clean.iloc[-1] / starting_value - 1.0
    cagr = (clean.iloc[-1] / starting_value) ** (1 / years) - 1 if years > 0 else float("nan")
    volatility = returns.std(ddof=1) * math.sqrt(periods_per_year) if len(returns) > 1 else float("nan")
    drawdown = clean / clean.cummax().clip(lower=starting_value) - 1.0
    return {
        "start_value": starting_value,
        "end_value": float(clean.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "maximum_drawdown": float(drawdown.min()),
    }
