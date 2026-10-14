"""Target-weight calculations of the three strategies (shared by backtest and live tracking).

equal_weight: min(1/n, 5 %) per name, remainder stays cash.
mean_var:     maximise the Sharpe ratio of predicted returns over a one-year daily covariance.
min_var:      minimise the portfolio variance over the same covariance.
Both optimised variants use long-only bounds 0-5 % and require at least 20 usable names for full investment.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RISK_FREE_RATE = 0.015
WEIGHT_CAP = 0.05
MIN_HISTORY_DAYS = 126
STRATEGY_NAMES = ("equal_weight", "mean_var", "min_var")


class OptimizationError(RuntimeError):
    """The optimiser did not converge; a different weighting is never substituted silently."""


def capped_equal_weights(tickers):
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(min(1.0 / len(tickers), WEIGHT_CAP), index=tickers, dtype=float)


def optimize_weights(selected, price_panel, date, minimum_variance=False, fallback_to_equal=True):
    """``selected``: predicted returns indexed by ticker; ``price_panel``: adjusted closes.

    The optimized strategies target full investment. With the 5% per-name cap, fewer than
    ``ceil(1 / WEIGHT_CAP)`` usable names cannot satisfy that policy and fail explicitly.
    Equal weighting is distinct: it deliberately leaves the uncapped remainder in cash.
    """
    tickers = [ticker for ticker in selected.index if ticker in price_panel.columns]
    if not tickers:
        return pd.Series(dtype=float)
    history = price_panel.loc[:date, tickers].tail(253).pct_change().dropna(how="all")
    coverage = history.count()
    tickers = coverage[coverage >= MIN_HISTORY_DAYS].index.tolist()
    if not tickers:
        return pd.Series(dtype=float)
    min_names_for_full_investment = math.ceil(1.0 / WEIGHT_CAP)
    if len(tickers) < min_names_for_full_investment:
        raise OptimizationError(
            f"full investment requires at least {min_names_for_full_investment} usable names "
            f"under the {WEIGHT_CAP:.0%} position cap; found {len(tickers)}"
        )
    history = history[tickers]
    covariance = history.cov().to_numpy(dtype=float) * 252
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T

    initial = np.full(len(tickers), min(WEIGHT_CAP, 1 / len(tickers)))
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
    if minimum_variance:
        objective = lambda weights: float(weights @ covariance @ weights)
    else:
        expected = selected.reindex(tickers).to_numpy(dtype=float) * 4
        objective = lambda weights: -((expected @ weights - RISK_FREE_RATE) /
                                      np.sqrt(max(float(weights @ covariance @ weights), 1e-12)))
    result = minimize(objective, initial, method="SLSQP", bounds=[(0, WEIGHT_CAP)] * len(tickers),
                      constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
    if not result.success:
        if fallback_to_equal:
            return capped_equal_weights(tickers)
        raise OptimizationError(f"SLSQP failed ({result.message})")
    weights = pd.Series(np.clip(result.x, 0.0, WEIGHT_CAP), index=tickers)
    return weights
