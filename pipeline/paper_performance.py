"""Rebuild paper performance outputs through 2017 and extend the strategy to today.

Historical returns use the repository's original portfolio weights and price archives.
The extension uses the live model's top-20%-per-sector predictions and cached Yahoo prices.

Those legacy calculations are retained only behind --legacy-reproduction. The normal
runner uses the raw-price self-financing engine and an explicitly verified benchmark.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import uuid
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .backtest import BacktestConfig, CorporateAction, simulate_portfolio, total_return_metrics
    from .benchmark import BenchmarkDataError, BenchmarkSeries
    from .market_calendar import nyse_sessions
    from .portfolio_weights import capped_equal_weights, optimize_weights
except ImportError:
    from backtest import BacktestConfig, CorporateAction, simulate_portfolio, total_return_metrics
    from benchmark import BenchmarkDataError, BenchmarkSeries
    from market_calendar import nyse_sessions
    from portfolio_weights import capped_equal_weights, optimize_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data"
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "paper_extension"
LEGACY_OUT_DIR = RESULTS_DIR / "paper_extension_legacy_reproduction"
OUT_DIR = LEGACY_OUT_DIR
INITIAL_VALUE = 1_000_000.0
TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.015
METHODS = {
    "Mean-Var": "mean_weighted_user8.xlsx",
    "Equally": "equally_weighted_user8.xlsx",
    "Min-Var": "minimum_weighted_user8.xlsx",
}


def date_from_yyyymmdd(value):
    return pd.to_datetime(str(int(value)), format="%Y%m%d")


def load_historical_prices(tickers):
    archive = DATA_DIR / "1-sp500_adj_price.csv.zip"
    with zipfile.ZipFile(archive) as zf:
        csv_name = next(name for name in zf.namelist()
                        if name.endswith(".csv") and not name.startswith("__MACOSX"))
        prices = pd.read_csv(zf.open(csv_name), usecols=["datadate", "tic", "adj_price"],
                             dtype={"datadate": str, "tic": str})
    prices = prices[prices["tic"].isin(tickers)]
    prices["date"] = pd.to_datetime(prices["datadate"], format="%Y%m%d")
    return prices.pivot_table(index="date", columns="tic", values="adj_price", aggfunc="last").sort_index()


def load_historical_spx():
    data = pd.read_excel(DATA_DIR / "1-spx_price.xlsx")
    data["date"] = pd.to_datetime(data["datadate"].astype(str), format="%Y%m%d")
    return data.set_index("date")["AdjClose"].sort_index()


def get_snapshot(panel, date):
    available = panel.loc[:date]
    if available.empty:
        return pd.Series(dtype=float)
    return available.iloc[-1]


def simulate_original_weights(weight_table, marks, prices):
    dates = sorted(weight_table["trade_date"].unique())
    by_date = {int(date): frame.set_index("tic")["weights"]
               for date, frame in weight_table.groupby("trade_date")}
    mark_dates = [date_from_yyyymmdd(d) for d in dates] + [date_from_yyyymmdd(marks[-1])]
    snapshots = prices.reindex(pd.DatetimeIndex(mark_dates), method="ffill")
    tickers = sorted(weight_table["tic"].unique())
    gross_values = pd.Series(np.nan, index=pd.DatetimeIndex(mark_dates), dtype=float)
    costs = pd.Series(0.0, index=gross_values.index)
    gross_values.iloc[0] = INITIAL_VALUE
    previous_shares = pd.Series(0, index=tickers, dtype=np.int64)

    for index, date in enumerate(dates):
        current_date, next_date = mark_dates[index], mark_dates[index + 1]
        capital = gross_values.loc[current_date]
        weights = by_date[int(date)].reindex(tickers).fillna(0)
        current_prices = snapshots.loc[current_date]
        target_shares = pd.Series(0, index=tickers, dtype=np.int64)
        for ticker, weight in weights.items():
            price = current_prices.get(ticker, np.nan)
            if pd.notna(price) and price > 0 and weight > 0:
                target_shares[ticker] = int(capital * weight / price)

        traded = (target_shares - previous_shares).abs()
        costs.loc[current_date] = sum(
            traded[ticker] * current_prices.get(ticker, 0)
            for ticker in tickers
            if pd.notna(current_prices.get(ticker, np.nan))
        ) * TRANSACTION_COST

        next_prices = snapshots.loc[next_date]
        gross_values.loc[next_date] = sum(
            target_shares[ticker] * next_prices.get(ticker, 0)
            for ticker in tickers
            if pd.notna(next_prices.get(ticker, np.nan))
        )
        previous_shares = target_shares

    return (gross_values - costs).rename("portfolio_value")


def load_live_targets(today):
    by_date = {}
    for path in sorted(RESULTS_DIR.glob("sector*/df_predict_best.csv")):
        prediction = pd.read_csv(path, index_col=0, parse_dates=True)
        prediction = prediction.loc[prediction.index <= today]
        sector = path.parent.name
        for date, row in prediction.iterrows():
            available = row.dropna().astype(float)
            if available.empty:
                continue
            cutoff = available.quantile(0.8)
            selected = available[available >= cutoff]
            by_date.setdefault(pd.Timestamp(date), []).extend(
                {"tic": ticker, "predicted_return": value, "sector": sector}
                for ticker, value in selected.items()
            )
    return {date: pd.DataFrame(rows).drop_duplicates("tic", keep="last")
            for date, rows in by_date.items()}


def load_live_prices(tickers):
    price_dir = REPO_ROOT / "cache" / "prices"
    series = []
    for ticker in tickers:
        path = price_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path, index_col=0, parse_dates=True)
        if "adj_close" in data.columns:
            series.append(data["adj_close"].rename(ticker))
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]


def build_live_weights(targets, live_prices):
    weight_sets = {method: {} for method in METHODS}
    for date, frame in sorted(targets.items()):
        selected = frame.set_index("tic")["predicted_return"]
        eligible = [ticker for ticker in selected.index if ticker in live_prices.columns]
        selected = selected.reindex(eligible).dropna()
        if selected.empty:
            continue
        weight_sets["Equally"][date] = capped_equal_weights(selected.index.tolist())
        weight_sets["Mean-Var"][date] = optimize_weights(selected, live_prices, date)
        weight_sets["Min-Var"][date] = optimize_weights(selected, live_prices, date, minimum_variance=True)
    return weight_sets


def simulate_live_extension(weight_dates, live_prices, anchor_value, anchor_date, prior_weights, today):
    dates = sorted(date for date in weight_dates if date >= anchor_date)
    if not dates or dates[0] != anchor_date:
        raise ValueError(f"Live predictions need a rebalance at historical endpoint {anchor_date.date()}")
    marks = dates + ([today] if dates[-1] < today else [])
    values = [anchor_value]
    previous_weights = prior_weights
    returns = []

    for index, date in enumerate(dates):
        next_date = marks[index + 1]
        weights = weight_dates[date]
        start_prices = get_snapshot(live_prices, date)
        end_prices = get_snapshot(live_prices, next_date)
        available = [ticker for ticker in weights.index
                     if ticker in start_prices.index and ticker in end_prices.index
                     and pd.notna(start_prices[ticker]) and pd.notna(end_prices[ticker])
                     and start_prices[ticker] > 0]
        active_weights = weights.reindex(available).fillna(0)
        gross_return = sum(active_weights[ticker] * (end_prices[ticker] / start_prices[ticker] - 1)
                           for ticker in available)
        turnover = sum(abs(weights.get(ticker, 0) - previous_weights.get(ticker, 0))
                       for ticker in set(weights.index) | set(previous_weights.index))
        net_return = gross_return - TRANSACTION_COST * turnover
        values.append(max(values[-1] * (1 + net_return), 0.0))
        returns.append(net_return)
        previous_weights = weights

    return (pd.Series(values, index=pd.DatetimeIndex(marks)),
            pd.Series(returns, index=pd.DatetimeIndex(marks[1:])))


def performance_metrics(values, returns=None, completed_only=False):
    values = values.dropna().sort_index()
    if returns is None:
        returns = values.pct_change().dropna()
    else:
        returns = returns.dropna()
    if completed_only and len(values.index) > 1:
        durations = values.index.to_series().diff().dt.days.reindex(returns.index)
        returns = returns[durations >= 60]
    annual_return = returns.mean() * 4 if len(returns) else np.nan
    annual_std = returns.std(ddof=1) * 2 if len(returns) > 1 else np.nan
    sharpe = (annual_return - RISK_FREE_RATE) / annual_std if annual_std and np.isfinite(annual_std) else np.nan
    max_drawdown = (values / values.cummax() - 1).min()
    return {
        "Start Value (million)": values.iloc[0] / 1e6,
        "End Value (million)": values.iloc[-1] / 1e6,
        "Total Return (%)": (values.iloc[-1] / values.iloc[0] - 1) * 100,
        "Maximum Drawdown (%)": max_drawdown * 100,
        "Annualized Return (%)": annual_return * 100,
        "Annualized Std (%)": annual_std * 100,
        "Sharpe Ratio": sharpe,
    }


def paper_published_tables():
    table_vii = {
        "Mean-Var": {"Annualized Return (%)": 13.17, "Annualized Std (%)": 17.0, "Sharpe Ratio": 0.687},
        "Equally": {"Annualized Return (%)": 16.12, "Annualized Std (%)": 16.4, "Sharpe Ratio": 0.887},
        "Min-Var": {"Annualized Return (%)": 13.29, "Annualized Std (%)": 12.9, "Sharpe Ratio": 0.917},
        "S&P 500": {"Annualized Return (%)": 7.12, "Annualized Std (%)": 13.8, "Sharpe Ratio": 0.406},
    }
    table_viii = {
        "Mean-Var": {"Start Value (million)": 1, "End Value (million)": 10.9917, "Total Return (%)": 999.17,
                     "Maximum Drawdown (%)": -56.89, "Annualized Return (%)": 8.29, "Annualized Std (%)": 23.6, "Sharpe Ratio": 0.287},
        "Equally": {"Start Value (million)": 1, "End Value (million)": 22.1383, "Total Return (%)": 2113.83,
                    "Maximum Drawdown (%)": -57.63, "Annualized Return (%)": 10.77, "Annualized Std (%)": 26.4, "Sharpe Ratio": 0.351},
        "Min-Var": {"Start Value (million)": 1, "End Value (million)": 12.81498, "Total Return (%)": 1181.50,
                    "Maximum Drawdown (%)": -46.30, "Annualized Return (%)": 9.87, "Annualized Std (%)": 18.1, "Sharpe Ratio": 0.462},
        "S&P 500": {"Start Value (million)": 1, "End Value (million)": 1.933153, "Total Return (%)": 93.32,
                    "Maximum Drawdown (%)": -66.73, "Annualized Return (%)": 5.22, "Annualized Std (%)": 19.1, "Sharpe Ratio": 0.195},
    }
    return table_vii, table_viii


def write_table_comparison(path, published, reconstructed, extended=None):
    rows = []
    for method, metrics in published.items():
        for metric, paper_value in metrics.items():
            row = {"Strategy": method, "Metric": metric, "Paper Published": paper_value,
                   "Original-data reconstruction": reconstructed.get(method, {}).get(metric, np.nan)}
            row["Reconstruction minus Paper"] = row["Original-data reconstruction"] - paper_value
            if extended is not None:
                row["Hybrid extension through today"] = extended.get(method, {}).get(metric, np.nan)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def legacy_reproduction_main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    weights = {method: pd.read_excel(DATA_DIR / "2-portfolio_data" / filename)
               for method, filename in METHODS.items()}
    tickers = sorted(set().union(*(set(frame["tic"]) for frame in weights.values())))
    historical_prices = load_historical_prices(tickers)
    historical_spx = load_historical_spx()
    trade_dates = sorted(weights["Equally"]["trade_date"].unique())
    historical_marks = trade_dates + [20170901]

    historical_values = {
        method: simulate_original_weights(frame, historical_marks, historical_prices)
        for method, frame in weights.items()
    }
    spx_marks = pd.DatetimeIndex([date_from_yyyymmdd(d) for d in historical_marks])
    spx_at_marks = historical_spx.reindex(spx_marks, method="ffill")
    paper_spx_returns = (spx_at_marks - spx_at_marks.shift(1)) / spx_at_marks
    paper_spx_returns.iloc[0] = 0
    historical_values["S&P 500"] = INITIAL_VALUE * (1 + paper_spx_returns).cumprod()
    historical_values["S&P 500 standard"] = INITIAL_VALUE * spx_at_marks / spx_at_marks.iloc[0]
    anchor_value = pd.Timestamp("2017-09-01")
    legacy_final_weights = {method: frame[frame.trade_date == trade_dates[-1]].set_index("tic")["weights"]
                            for method, frame in weights.items()}

    targets = load_live_targets(today)
    targets = {date: frame for date, frame in targets.items() if date >= anchor_value}
    all_tickers = sorted(set(ticker for frame in targets.values() for ticker in frame["tic"]))
    all_tickers.append("^GSPC")
    live_prices = load_live_prices(all_tickers)
    if "^GSPC" not in live_prices.columns:
        import yfinance as yf
        benchmark = yf.download("^GSPC", start="2017-06-01", auto_adjust=True, progress=False)
        live_prices = pd.concat([live_prices, benchmark["Close"].squeeze().rename("^GSPC")], axis=1)
    if anchor_value not in targets:
        raise ValueError("No live model predictions exist on the 2017-09-01 historical handoff date")

    live_weights = build_live_weights(targets, live_prices.drop(columns=["^GSPC"], errors="ignore"))
    values = {}
    live_returns = {}
    for method in METHODS:
        extension, returns = simulate_live_extension(
            live_weights[method], live_prices.drop(columns=["^GSPC"], errors="ignore"),
            historical_values[method].iloc[-1], anchor_value,
            legacy_final_weights[method], today,
        )
        values[method] = pd.concat([historical_values[method].iloc[:-1], extension]).sort_index()
        values[method] = values[method][~values[method].index.duplicated(keep="last")]
        live_returns[method] = returns

    benchmark_history = historical_values["S&P 500"]
    benchmark_standard_history = historical_values["S&P 500 standard"]
    benchmark_daily = live_prices["^GSPC"].dropna()
    benchmark_marks = sorted({date for frame in values.values() for date in frame.index
                              if date >= anchor_value} | {today})
    benchmark_live = benchmark_daily.reindex(pd.DatetimeIndex(benchmark_marks), method="ffill")
    paper_benchmark_returns = (benchmark_live - benchmark_live.shift(1)) / benchmark_live
    paper_benchmark_returns.iloc[0] = 0
    paper_benchmark_live = benchmark_history.iloc[-1] * (1 + paper_benchmark_returns).cumprod()
    standard_benchmark_live = benchmark_standard_history.iloc[-1] * benchmark_live / benchmark_live.loc[anchor_value]
    values["S&P 500"] = pd.concat([benchmark_history.iloc[:-1], paper_benchmark_live]).sort_index()
    values["S&P 500"] = values["S&P 500"][~values["S&P 500"].index.duplicated(keep="last")]
    values["S&P 500 standard"] = pd.concat([benchmark_standard_history.iloc[:-1], standard_benchmark_live]).sort_index()
    values["S&P 500 standard"] = values["S&P 500 standard"][~values["S&P 500 standard"].index.duplicated(keep="last")]

    performance = pd.DataFrame(values).sort_index()
    performance.to_csv(OUT_DIR / "quarterly_performance.csv", index_label="date")
    quarterly_returns = performance.pct_change(fill_method=None)
    cumulative_pnl = quarterly_returns.fillna(0).cumsum()
    cumulative_pnl.to_csv(OUT_DIR / "cumulative_pnl.csv", index_label="date")

    plt.figure(figsize=(11, 6))
    plot_labels = {"S&P 500": "S&P 500 (paper formula)",
                   "S&P 500 standard": "S&P 500 (standard return)"}
    for method in ["Mean-Var", "Min-Var", "Equally", "S&P 500", "S&P 500 standard"]:
        plt.plot(cumulative_pnl.index, cumulative_pnl[method], label=plot_labels.get(method, method), linewidth=1.5)
    plt.title("Cumulative P&L Quarterly (with transaction cost)")
    plt.xlabel("Quarterly Trade Date")
    plt.ylabel("Cumulative P&L (sum of quarterly returns)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_pnl.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for method in ["Mean-Var", "Equally", "Min-Var", "S&P 500", "S&P 500 standard"]:
        plt.plot(performance.index, performance[method] / 1e6, label=plot_labels.get(method, method), linewidth=1.6)
    plt.title("Portfolio Value Performance Quarterly")
    plt.xlabel("Quarterly Trade Date")
    plt.ylabel("Portfolio Value (million USD)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_portfolio_value.png", dpi=180)
    plt.close()

    published_vii, published_viii = paper_published_tables()
    in_sample_end = pd.Timestamp("2007-12-31")
    reconstructed_vii = {}
    reconstructed_viii = {}
    extended_viii = {}
    for method, series in historical_values.items():
        final_series = values[method]
        returns = series.pct_change().dropna()
        in_sample = returns.loc[(returns.index > pd.Timestamp("1995-06-01")) &
                                (returns.index <= in_sample_end)]
        reconstructed_vii[method] = performance_metrics(series.loc[in_sample.index.union([series.index[0]])], in_sample)
        reconstructed_viii[method] = performance_metrics(series)
        extended_viii[method] = performance_metrics(final_series, completed_only=True)

    write_table_comparison(OUT_DIR / "table_vii_comparison.csv", published_vii, reconstructed_vii)
    write_table_comparison(OUT_DIR / "table_viii_comparison.csv", published_viii, reconstructed_viii, extended_viii)

    benchmark_rows = []
    for table_name, published, reproduced, standard in [
        ("VII", published_vii["S&P 500"], reconstructed_vii["S&P 500"], reconstructed_vii["S&P 500 standard"]),
        ("VIII", published_viii["S&P 500"], reconstructed_viii["S&P 500"], reconstructed_viii["S&P 500 standard"]),
    ]:
        for metric, paper_value in published.items():
            benchmark_rows.append({"Table": table_name, "Metric": metric, "Paper Published": paper_value,
                                   "Notebook formula": reproduced.get(metric, np.nan),
                                   "Standard price return": standard.get(metric, np.nan)})
    pd.DataFrame(benchmark_rows).to_csv(OUT_DIR / "spx_formula_comparison.csv", index=False)

    diagnostics = [
        "Paper source: WRDS/Compustat, roughly 1,142 historical S&P 500 constituents, and quarterly membership updates; extension source: SEC companyfacts plus Yahoo Finance.",
        "Historical features are the repository's supplied WRDS-derived tables/weights through 2017-06; extension features are standard-formula proxies, not the unpublished Compustat formulas.",
        "The 1995-2017 performance leg reuses archived strategy weights; it does not retrain the original R model on reconstructed point-in-time historical data.",
        "Historical leg: original repository portfolio weights, adjusted stock-price archive, and SPX through 2017-09-01.",
        "The original backtest notebook loads balance_daily_user8.xlsx, which is absent from the repository; this reconstruction rebuilds its price matrix from the archived CSV and therefore cannot guarantee identical stock prices/portfolio paths.",
        "The post-2017 extension uses dated historical membership snapshots from fja05680/sp500 (MIT), not the current list alone; its maintainer documents early-history gaps and event corrections, and our snapshots are cached with age recorded in Data/live/sp500_membership_coverage.csv.",
        "The expanded fetch universe combines current constituents with historical tickers mapped to SEC CIKs and historical Compustat GICS; not every historical symbol maps (the present build contains 536 of 558 candidates with usable features/prices). This is still incomplete versus the paper's 1,142 historical components.",
        "Approximate feature definitions: quarterly revenue growth uses four-quarter change; P/E, P/S, price/CFO and EV multiples use trailing-quarter aggregates; enterprise multiple is EV/EBITDA; debt/equity uses total liabilities/equity. Exact paper/WRDS construction is not disclosed.",
        "SEC availability is approximated as period-end plus two months and the next Mar/Jun/Sep/Dec rebalance; actual filing dates are not used, so some observations may be look-ahead biased or delayed.",
        "The SEC parser currently keeps the latest available restatement for each historical fact, not the filing version known on that historical trade date; this is a material point-in-time look-ahead risk.",
        "Yahoo auto-adjusted historical prices can incorporate later corporate actions; this is convenient for returns but can distort historical price-multiple features.",
        "The paper's 5% missing-feature/stock cleaning is approximated by removing sector features above 5% missingness and dropping rows missing retained factors; this is not the authors' exact multi-stage cleaning procedure.",
        "The original backtest notebook computes S&P 500 returns as (P_t-P_{t-1})/P_t, not the standard denominator P_{t-1}. This reproduces its Table VIII SPX end value (1.933153m); Fig. 4/5 include both that paper formula and the standard price-ratio benchmark.",
        "Model set matches the paper's five labels, but implementations differ: sklearn versions are used; stepwise is a custom forward/backward AIC search; Ridge/RF/GBM hyperparameters are selected by 3-fold GridSearchCV rather than the paper's R defaults. GridSearchCV is not time-ordered.",
        "The model training rows are assembled ticker-by-ticker before fitting; default 3-fold GridSearchCV therefore does not represent chronological validation and can leak across time regimes.",
        "Rolling windows now use 16-40 training quarters plus 4 testing quarters, but free-data model history starts around 2009, so earliest train/test dates and available sample sizes differ materially from the 1990-start paper.",
        "Historical Table VII/VIII reconstruction does not match the paper from the archived inputs. The comparison CSV records measured differences; do not label these outputs an exact replication.",
        "Paper Table VIII's end values and annualized returns are different metrics: annualized return is arithmetic mean quarterly return times four, not CAGR; the table should not be expected to compound to its end value.",
        "The model handoff is at 2017-09-01: original Compustat features/memberships are used before that date, and reconstructed SEC features with dated S&P membership snapshots after it. It remains a hybrid, not a full original-data replication.",
        "Historical simulation uses the notebook's rebalance share rounding and 0.1% traded-notional costs; extension turnover costs use 0.1% of weight turnover rather than exact share-level trades.",
        "Table VII/ VIII annualized return uses mean quarterly return times four; volatility uses quarterly sample standard deviation times two; Sharpe uses 1.5% risk-free rate.",
        "The final observation may be a partial quarter through today; it is included in ending value/drawdown but excluded from annualized extension metrics when under 60 days.",
        "Predicted returns are selected using the paper's >= 80th percentile rule, so ties can select more than 20%.",
        "Mean/minimum-variance extension weights use long-only bounds 0-5%, one-year daily covariance, full investment when at least 20 eligible names are available, otherwise residual cash.",
        "Missing model outputs or Yahoo prices can reduce the effective universe and portfolio exposure.",
        "Original paper returns are log returns for ranking; portfolio performance uses simple price returns, as in the paper's backtest.",
    ]
    (OUT_DIR / "methodology_limitations.txt").write_text("\n".join(diagnostics) + "\n")

    print(f"Figures/tables saved under {OUT_DIR}")
    print(f"Historical handoff: {anchor_value.date()}, extension valuation: {today.date()}")
    print("Table VII reproduced metrics:")
    print(pd.DataFrame(reconstructed_vii).round(4).to_string())
    print("Table VIII hybrid extension metrics:")
    print(pd.DataFrame(extended_viii).round(4).to_string())


def run_performance(
    prices: pd.DataFrame,
    targets: dict[str, dict[str | pd.Timestamp, dict[str, float]]],
    benchmark: BenchmarkSeries,
    *,
    output_dir: str | Path,
    corporate_actions: list[CorporateAction],
    config: BacktestConfig | None = None,
    validation_evidence: dict[str, object] | None = None,
) -> Path:
    """Run each target set independently and write an auditable daily report."""
    config = config or BacktestConfig()
    output_dir = Path(output_dir)
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("daily raw-close price panel is required")
    prices = prices.sort_index().copy()
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    if prices.index.has_duplicates:
        raise ValueError("price panel dates must be unique")
    expected_sessions = nyse_sessions(prices.index[0], prices.index[-1])
    missing_sessions = expected_sessions.difference(prices.index)
    unexpected_dates = prices.index.difference(expected_sessions)
    if len(missing_sessions) or len(unexpected_dates):
        raise ValueError(
            f"price panel does not match the NYSE calendar: "
            f"{len(missing_sessions)} missing sessions, {len(unexpected_dates)} non-sessions"
        )
    if len(targets) != 3:
        raise ValueError("provide exactly three isolated strategy target sets")
    if not targets:
        raise ValueError("no strategy target weights were provided")

    benchmark_nav = benchmark.nav(config.initial_capital, prices.index)
    nav_columns: dict[str, pd.Series] = {str(benchmark.metadata["display_name"]): benchmark_nav}
    cash_columns: dict[str, pd.Series] = {}
    result_by_strategy = {}
    for strategy, strategy_targets in sorted(targets.items()):
        if not strategy_targets:
            raise ValueError(f"strategy {strategy!r} has no target-weight dates")
        result = simulate_portfolio(
            prices,
            strategy_targets,
            config=config,
            corporate_actions=corporate_actions,
        )
        result_by_strategy[strategy] = result
        nav_columns[strategy] = result.nav
        cash_columns[strategy] = result.cash

    nav_frame = pd.DataFrame(nav_columns).sort_index()
    cash_frame = pd.DataFrame(cash_columns, index=prices.index).fillna(config.initial_capital)
    output_dir.mkdir(parents=True, exist_ok=False)
    nav_frame.to_csv(Path(output_dir) / "daily_nav.csv", index_label="date")
    cash_frame.to_csv(Path(output_dir) / "daily_cash.csv", index_label="date")
    trade_frames = []
    metrics = []
    risk_free_rate = RISK_FREE_RATE
    elapsed_days = prices.index.to_series().diff().dt.days.fillna(0)
    interval_risk_free = (1.0 + risk_free_rate) ** (elapsed_days / 365.0) - 1.0
    benchmark_returns = benchmark.returns().reindex(prices.index)
    metrics_by_strategy = {}

    for name, nav in nav_columns.items():
        if name == str(benchmark.metadata["display_name"]):
            returns = benchmark_returns
            strategy_result = None
        else:
            strategy_result = result_by_strategy[name]
            returns = strategy_result.daily_returns.reindex(prices.index)
            trades = strategy_result.trades.copy()
            if not trades.empty:
                trades.insert(0, "strategy", name)
                trade_frames.append(trades)
        summary = total_return_metrics(
            nav, initial_capital=config.initial_capital
        )
        valid_returns = returns.dropna()
        excess = (returns - interval_risk_free).dropna()
        ann_vol = excess.std(ddof=1) * np.sqrt(252) if len(excess) > 1 else np.nan
        sharpe = (
            excess.mean() / excess.std(ddof=1) * np.sqrt(252)
            if len(excess) > 1 and excess.std(ddof=1) > 0 else np.nan
        )
        metric_row = {
            "series": name,
            **summary,
            "annualized_excess_volatility": float(ann_vol) if pd.notna(ann_vol) else np.nan,
            "sharpe_daily": float(sharpe) if pd.notna(sharpe) else np.nan,
            "annual_risk_free_rate": risk_free_rate,
        }
        if strategy_result is not None:
            metric_row["transaction_costs"] = float(strategy_result.trades["transaction_cost"].sum())
            metric_row["turnover_notional"] = float(strategy_result.trades["notional"].sum())
            active = (strategy_result.daily_returns - benchmark_returns).dropna()
            tracking_error = active.std(ddof=1) * np.sqrt(252) if len(active) > 1 else np.nan
            metric_row["tracking_error"] = float(tracking_error) if pd.notna(tracking_error) else np.nan
            metric_row["information_ratio"] = (
                float(active.mean() * 252 / tracking_error)
                if pd.notna(tracking_error) and tracking_error > 0 else np.nan
            )
            metric_row["relative_wealth_end"] = float(
                strategy_result.nav.iloc[-1] / benchmark_nav.iloc[-1] - 1.0
            )
        metrics.append(metric_row)
        metrics_by_strategy[name] = metric_row

    pd.DataFrame(metrics).to_csv(Path(output_dir) / "daily_metrics.csv", index=False)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    all_trades.to_csv(Path(output_dir) / "transaction_journal.csv", index=False)
    relative_wealth = pd.DataFrame({
        strategy: result.nav / benchmark_nav - 1.0
        for strategy, result in result_by_strategy.items()
    }, index=prices.index)
    relative_wealth.to_csv(Path(output_dir) / "relative_wealth.csv", index_label="date")

    evidence = validation_evidence or {}
    required_evidence = {
        "raw_price_basis",
        "corporate_actions",
        "historical_membership",
        "point_in_time_signal_and_labels",
        "benchmark_provenance",
    }
    evidence_complete = all(
        isinstance(evidence.get(key), dict)
        and evidence[key].get("verified") is True
        and isinstance(evidence[key].get("evidence_uri"), str)
        and evidence[key]["evidence_uri"].strip()
        for key in required_evidence
    )
    run_id = output_dir.name
    run_manifest = {
        "run_id": run_id,
        "mode": "historical_backtest",
        "currency": "USD",
        "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "as_of": prices.index[-1].date().isoformat(),
        "as_of_calendar": "NYSE; schedule includes holidays and shortened sessions",
        "git_commit": _git_commit(),
        "configuration": {
            "initial_capital": config.initial_capital,
            "transaction_cost_bps_per_side": config.transaction_cost_bps_per_side,
            "spread_slippage_bps_per_side": config.spread_slippage_bps_per_side,
            "fractional_shares": config.fractional_shares,
            "cash_rate_annual": config.cash_rate_annual,
            "max_weight_per_asset": config.max_weight_per_asset,
            "benchmark": benchmark.metadata,
            "risk_free_rate_for_metrics_only": risk_free_rate,
        },
        "input_sha256": {
            "raw_prices": hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes()).hexdigest(),
            "targets": hashlib.sha256(
                json.dumps(
                    {
                        strategy: {
                            pd.Timestamp(date).date().isoformat(): dict(weights)
                            for date, weights in strategy_targets.items()
                        }
                        for strategy, strategy_targets in targets.items()
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
            "corporate_actions": hashlib.sha256(
                json.dumps([action.__dict__ for action in corporate_actions], sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "benchmark_values": hashlib.sha256(
                pd.util.hash_pandas_object(benchmark.values, index=True).values.tobytes()
            ).hexdigest(),
            "benchmark_metadata": hashlib.sha256(
                json.dumps(benchmark.metadata, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
        },
        "benchmark_start": benchmark_nav.index[0].date().isoformat(),
        "benchmark_end": benchmark_nav.index[-1].date().isoformat(),
        "strategy_names": sorted(result_by_strategy),
        "implementation_checks": "PASSED",
        "historical_data_validation": "VALIDATED" if evidence_complete else "NOT_VALIDATED",
        "validation_evidence": evidence,
        "limitations": [
            "Validation claims require reviewed source evidence for each listed category.",
            "Prices are simulated close executions plus configured costs, not confirmed exchange fills.",
            "The official index is a gross total-return series; its values do not include investor trading costs.",
            "SPY must be selected by an explicit metadata type and is always labelled an ETF proxy.",
        ],
        "library_versions": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "pandas-market-calendars")
        },
    }
    with (Path(output_dir) / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(run_manifest, stream, indent=2, sort_keys=True, default=str)
    return Path(output_dir)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _load_price_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "close"}
    if not required.issubset(frame.columns):
        raise ValueError("price CSV must have date,ticker,close (raw unadjusted close) columns")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("price CSV has duplicate date/ticker observations")
    return frame.pivot(index="date", columns="ticker", values="close").sort_index()


def _load_targets(path: Path) -> dict[str, dict[pd.Timestamp, dict[str, float]]]:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"strategy", "date", "ticker", "target_weight"}
    if not required.issubset(frame.columns):
        raise ValueError("target CSV must have strategy,date,ticker,target_weight columns")
    if frame.duplicated(["strategy", "date", "ticker"]).any():
        raise ValueError("target CSV has duplicate strategy/date/ticker rows")
    result: dict[str, dict[pd.Timestamp, dict[str, float]]] = {}
    for (strategy, day), group in frame.groupby(["strategy", "date"], sort=True):
        weights = dict(zip(group["ticker"].astype(str), group["target_weight"].astype(float)))
        result.setdefault(str(strategy), {})[pd.Timestamp(day)] = weights
    return result


def _load_actions(path: Path) -> list[CorporateAction]:
    frame = pd.read_csv(path)
    required = {"action_id", "ticker", "ex_date", "kind", "value", "pay_date"}
    if not required.issubset(frame.columns):
        raise ValueError("actions CSV must have action_id,ticker,ex_date,kind,value,pay_date columns")
    return [
        CorporateAction(
            str(row.action_id),
            str(row.ticker),
            pd.Timestamp(row.ex_date),
            str(row.kind),
            float(row.value),
            None if pd.isna(row.pay_date) else pd.Timestamp(row.pay_date),
        )
        for row in frame.itertuples(index=False)
    ]


def _synthetic_demo(output_root: Path) -> Path:
    dates = nyse_sessions("2025-01-02", "2025-01-31")
    aaa = 100 * np.cumprod(np.repeat(1.002, len(dates)))
    dividend_index = 8
    split_index = 15
    aaa[dividend_index:] -= 1.0
    aaa[split_index:] /= 2.0
    prices = pd.DataFrame(
        {
            "AAA": aaa,
            "BBB": 75 * np.cumprod(np.repeat(1.001, len(dates))),
            "CCC": 50 * np.cumprod(np.repeat(1.0005, len(dates))),
        },
        index=dates,
    )
    benchmark = BenchmarkSeries(
        values=pd.Series(100 * np.cumprod(np.repeat(1.001, len(dates))), index=dates),
        metadata={
            "type": "synthetic_test_fixture",
            "display_name": "Synthetic benchmark (not S&P 500)",
            "source": "local deterministic fixture",
        },
    )
    execution_dates = [dates[0], dates[len(dates) // 2], dates[-1]]
    targets = {
        "Equal Weight": {
            day: {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2} for day in execution_dates
        },
        "Mean-Var": {
            day: {"AAA": 0.2, "BBB": 0.4, "CCC": 0.3} for day in execution_dates
        },
        "Min-Var": {
            day: {"AAA": 0.3, "BBB": 0.2, "CCC": 0.4} for day in execution_dates
        },
    }
    out = output_root / f"synthetic-{uuid.uuid4().hex}"
    return run_performance(
        prices,
        targets,
        benchmark,
        output_dir=out,
        corporate_actions=[
            CorporateAction(
                "synthetic-dividend", "AAA", dates[dividend_index], "dividend", 1.0, dates[10]
            ),
            CorporateAction(
                "synthetic-split", "AAA", dates[split_index], "split", 2.0
            ),
        ],
        config=BacktestConfig(
            initial_capital=1_000_000,
            transaction_cost_bps_per_side=10,
            spread_slippage_bps_per_side=0,
            fractional_shares=True,
            max_weight_per_asset=None,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a daily, self-financing paper backtest from explicitly sourced inputs."
    )
    parser.add_argument("--prices", type=Path, help="CSV: date,ticker,close (raw unadjusted closes)")
    parser.add_argument("--targets", type=Path, help="CSV: strategy,date,ticker,target_weight")
    parser.add_argument("--actions", type=Path, help="CSV: action_id,ticker,ex_date,kind,value,pay_date")
    parser.add_argument("--benchmark-csv", type=Path, help="CSV: date,total_return_index")
    parser.add_argument("--benchmark-metadata", type=Path, help="JSON benchmark provenance manifest")
    parser.add_argument("--validation-evidence", type=Path, help="reviewed evidence manifest JSON")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "paper_extension")
    parser.add_argument("--demo", action="store_true", help="run a deterministic synthetic demo")
    parser.add_argument(
        "--legacy-reproduction", action="store_true",
        help="run old, non-validating paper reproduction in a separate legacy output folder",
    )
    args = parser.parse_args()
    if args.legacy_reproduction:
        legacy_reproduction_main()
        return
    if args.demo:
        output = _synthetic_demo(args.output_dir)
        print(f"Synthetic-only demonstration written to {output}")
        print("No real benchmark or historical validation is represented.")
        return
    required = (
        args.prices, args.targets, args.actions, args.benchmark_csv, args.benchmark_metadata
    )
    if any(value is None for value in required):
        parser.error(
            "normal execution requires --prices, --targets, --actions, --benchmark-csv, "
            "and --benchmark-metadata; the repository's adjusted-price/SPX archives are "
            "not sufficient for a validated self-financing total-return comparison"
        )

    price_panel = _load_price_panel(args.prices)
    calendar = nyse_sessions(price_panel.index[0], price_panel.index[-1])
    if not calendar.equals(price_panel.index):
        raise ValueError("raw price CSV must contain every NYSE session exactly once")
    benchmark = BenchmarkSeries.from_files(
        args.benchmark_csv, args.benchmark_metadata, expected_trading_dates=calendar
    )
    output = args.output_dir / uuid.uuid4().hex
    evidence = (
        json.loads(args.validation_evidence.read_text(encoding="utf-8"))
        if args.validation_evidence else None
    )
    run_performance(
        price_panel,
        _load_targets(args.targets),
        benchmark,
        output_dir=output,
        corporate_actions=_load_actions(args.actions),
        validation_evidence=evidence,
    )
    print(f"Daily results written to {output}")


if __name__ == "__main__":
    main()