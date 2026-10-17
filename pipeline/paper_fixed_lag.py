"""Causal, versioned paper research rules. Not an adapter for live orders."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .market_calendar import (
    first_session_on_or_after, nyse_sessions, previous_nyse_session, session_close_time,
)
from .run_model import PAPER_FEATURES, ml_model
from .portfolio_weights import OptimizationError
from .training_vintage import STRICT, feature_limits

RULE_VERSION = "paper_fixed_lag-v1"
TIE_RULE = "ceil-20pct-descending-prediction-security-id-v1"
OPTIMIZATION_VERSION = "annual-log-max-sharpe-v1"


@dataclass(frozen=True)
class FixedLagRules:
    version: str = RULE_VERSION
    minimum_training_quarters: int = 16
    maximum_training_quarters: int = 40
    validation_quarters: int = 4
    feature_missing_threshold: float = 0.05
    selected_fraction: float = 0.20
    cost_per_side: float = 0.001
    weight_cap: float = 0.05
    risk_free_annual: float = 0.015
    tie_rule: str = TIE_RULE
    cutoff_convention: str = "previous_NYSE_session_close"
    execution_convention: str = "fixed_date_next_NYSE_session_close"
    preprocessing: str = "training_only_5pct_feature_filter_then_complete_cases_no_imputation"
    status: str = "NOT_VALIDATED"
    optimization_version: str = OPTIMIZATION_VERSION


RULES = FixedLagRules()


def schedule(cycle: str) -> dict:
    return _schedule_cached(str(pd.Period(cycle, freq="Q-DEC"))).copy()


@lru_cache(maxsize=512)
def _schedule_cached(cycle: str) -> dict:
    period = pd.Period(cycle, freq="Q-DEC")
    year = period.year + (period.quarter == 4)
    month = {1: 6, 2: 9, 3: 12, 4: 3}[period.quarter]
    nominal = pd.Timestamp(year=year, month=month, day=1)
    execution = first_session_on_or_after(nominal)
    cutoff = session_close_time(previous_nyse_session(execution)).tz_convert("UTC")
    return {
        "cycle_key": str(period), "rule_version": RULE_VERSION,
        "quarter_end": period.end_time.date().isoformat(),
        "nominal_execution_date": nominal.date().isoformat(),
        "information_cutoff": cutoff.isoformat(),
        "signal_calculation": "offline reconstruction after cutoff; never claimed as an observed historical run",
        "execution_date": execution.date().isoformat(),
        "execution_at": session_close_time(execution).tz_convert("UTC").isoformat(),
        "cutoff_assumption": RULES.cutoff_convention,
        "execution_assumption": RULES.execution_convention,
        "operational_facts_ready_at": None,
    }


def continuation_cycles(start: str = "2017Q2", end: str = "2026Q2") -> list[str]:
    if pd.Period(start, "Q") > pd.Period(end, "Q"):
        raise ValueError("Continuation start is after end")
    return [str(p) for p in pd.period_range(start, end, freq="Q")]


def select_top_fraction(predictions: pd.Series) -> tuple[pd.Series, dict]:
    if predictions.empty or predictions.index.has_duplicates:
        raise ValueError("Predictions must be nonempty with unique security IDs")
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must be finite")
    count = math.ceil(len(predictions) * RULES.selected_fraction)
    ordered = pd.DataFrame({"security_id": predictions.index.astype(str),
                            "prediction": predictions.to_numpy(dtype=float)})
    ordered = ordered.sort_values(["prediction", "security_id"], ascending=[False, True],
                                  kind="stable")
    selected = ordered.head(count).set_index("security_id")["prediction"]
    cutoff = predictions.quantile(0.8)
    return selected, {
        "rule": TIE_RULE, "candidates": len(predictions), "selected": count,
        "original_quantile_ge_selected": int(predictions.ge(cutoff).sum()),
        "unique_predictions": int(predictions.nunique()),
        "constant_prediction": predictions.nunique() == 1,
        "constant_prediction_explanation": (
            "Inspect fitted estimator coefficients/tree leaves; a regularized or intercept-only "
            "estimator can produce ties. Ranking does not manufacture predictive information."
        ),
    }


def window_frames(frame: pd.DataFrame, target_cycle: str, *,
                  training_vintage_policy: str = STRICT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require four contiguous, completely matured validation decision quarters."""
    required = {"cycle_key", "y_return", "label_known_at", "feature_available_at",
                "reconstruction_status"}
    if not required.issubset(frame):
        raise ValueError(f"Panel missing {sorted(required - set(frame.columns))}")
    target = pd.Period(target_cycle, freq="Q")
    cutoff = pd.Timestamp(schedule(target_cycle)["information_cutoff"])
    data = frame.copy()
    periods = data["cycle_key"].map(lambda value: pd.Period(value, freq="Q"))
    known = pd.to_datetime(data["label_known_at"], utc=True, errors="coerce")
    available = pd.to_datetime(data["feature_available_at"], utc=True, errors="coerce")
    limits = feature_limits(data["cycle_key"], target_cycle, training_vintage_policy)
    feature_safe = available.notna() & available.le(limits)
    verified = data["reconstruction_status"].eq("VERIFIED")
    matured = known.notna() & known.le(cutoff) & data["y_return"].notna()
    data = data[feature_safe & verified & matured & (periods < target)].copy()
    if data.empty:
        raise ValueError("No source-verified rows with causal features and completed targets")
    periods = data["cycle_key"].map(lambda value: pd.Period(value, "Q"))
    last = max(periods)
    expected_last = target - 2
    if last != expected_last:
        raise ValueError(f"Latest completed label quarter must be {expected_last}; found {last}")
    validation_periods = [last - i for i in reversed(range(RULES.validation_quarters))]
    training_end = validation_periods[0] - 1
    available_training = sorted(set(p for p in periods if p <= training_end))
    training_periods = available_training[-RULES.maximum_training_quarters:]
    if len(training_periods) < RULES.minimum_training_quarters:
        raise ValueError(f"Need 16 complete training quarters; found {len(training_periods)}")
    expected_training = list(pd.period_range(training_periods[0], training_end, freq="Q"))
    if training_periods != expected_training:
        raise ValueError("Gap in training quarter sequence")
    if any(p not in set(periods) for p in validation_periods):
        raise ValueError("Missing one of four required validation quarters")
    train = data[periods.isin(training_periods)].copy()
    validation = data[periods.isin(validation_periods)].copy()
    return train, validation


def clean_windows(train: pd.DataFrame, validation: pd.DataFrame,
                  trade: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for frame in (train, validation, trade):
        if not set(PAPER_FEATURES).issubset(frame):
            raise ValueError("All 20 input feature columns are required; unknown values must be NaN")
    train = train.copy()
    train[PAPER_FEATURES] = train[PAPER_FEATURES].replace([np.inf, -np.inf], np.nan)
    features = train[PAPER_FEATURES].isna().mean().loc[lambda values: values <= 0.05].index.tolist()
    if not features:
        raise ValueError("No features remain under the training-only 5% missingness rule")
    cleaned = []
    for frame in (train, validation, trade):
        frame = frame.copy()
        frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)
        cleaned.append(frame.dropna(subset=features))
    for name, frame, minimum in (("training", cleaned[0], 16), ("validation", cleaned[1], 4)):
        actual = frame["cycle_key"].nunique()
        if actual < minimum:
            raise ValueError(f"{name} has only {actual} quarters after complete-case filtering")
        periods = sorted({pd.Period(value, "Q") for value in frame["cycle_key"]})
        original = train if name == "training" else validation
        expected_end = max(pd.Period(value, "Q") for value in original["cycle_key"])
        if periods != list(pd.period_range(periods[0], expected_end, freq="Q")):
            raise ValueError(f"{name} quarter gap after complete-case filtering")
    if cleaned[2].empty:
        raise ValueError("No prediction rows remain after complete-case filtering")
    return features, *cleaned


def fit_sector(panel: pd.DataFrame, trade: pd.DataFrame, cycle: str, *,
               training_vintage_policy: str = STRICT) -> dict:
    """Five families; no full-history feature selection, imputation or test refitting."""
    train, validation = window_frames(panel, cycle, training_vintage_policy=training_vintage_policy)
    features, train, validation, trade = clean_windows(train, validation, trade)
    cutoff = pd.Timestamp(schedule(cycle)["information_cutoff"])
    available = pd.to_datetime(trade["feature_available_at"], utc=True, errors="raise")
    if (available > cutoff).any() or not trade["reconstruction_status"].eq("VERIFIED").all():
        raise ValueError("Trade rows are unverified or contain future information")
    return fit_cleaned_sector(features, train, validation, trade)


def fit_cleaned_sector(features: list[str], train: pd.DataFrame, validation: pd.DataFrame,
                       trade: pd.DataFrame) -> dict:
    """Numerical fitting only; this function does not certify historical source availability."""
    if not features or any(frame.empty for frame in (train, validation, trade)):
        raise ValueError("Numerical fit needs nonempty cleaned train, validation and prediction data")
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "gbm": GradientBoostingRegressor(random_state=42),
        "stepwise_aic": Pipeline([("scale", StandardScaler()), ("model", ml_model.StepwiseAICRegressor())]),
    }
    scores, forecasts = {}, {}
    for name, model in models.items():
        model.fit(train[features], train["y_return"])
        scores[name] = float(mean_squared_error(validation["y_return"], model.predict(validation[features])))
        forecasts[name] = model.predict(trade[features])
    winner = min(scores, key=scores.get)
    predicted = pd.Series(forecasts[winner], index=trade["security_id"])
    chosen, ties = select_top_fraction(predicted)
    return {
        "features": features, "train": train, "validation": validation,
        "forecasts": predicted.to_dict(), "selected": chosen.to_dict(),
        "mse": scores, "winner": winner, "ties": ties,
        "deviation": "Deterministic sklearn defaults, not identical to original R caret tuning; NOT_VALIDATED",
    }


def optimization_inputs(selected: pd.Series, covariance_levels: pd.DataFrame,
                        cutoff: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep all three inputs in annual log units; never feed log means to simple covariance."""
    cutoff_day = pd.Timestamp(cutoff).tz_convert("America/New_York").tz_localize(None).normalize()
    history = covariance_levels.loc[:cutoff_day].tail(253).reindex(columns=selected.index)
    if len(history) != 253 or history.isna().any().any():
        missing = history.columns[history.isna().any()].tolist()
        raise OptimizationError(f"Need 253 complete, verified daily covariance levels; missing securities={missing}")
    sessions = nyse_sessions(cutoff_day - pd.Timedelta(days=400), cutoff_day)[-253:]
    if not history.index.equals(sessions):
        raise OptimizationError("Covariance history must contain the last 253 actual NYSE sessions through cutoff")
    if (history <= 0).any().any() or not np.isfinite(history).all().all():
        raise OptimizationError("Invalid source-defined covariance levels")
    returns = np.log(history / history.shift(1)).iloc[1:]
    covariance = returns.cov().to_numpy() * 252
    covariance = (covariance + covariance.T) / 2
    expected = selected.to_numpy(dtype=float) * 4
    risk_free = math.log1p(RULES.risk_free_annual)
    return expected, covariance, risk_free


def optimization_definition() -> dict:
    return {
        "version": OPTIMIZATION_VERSION,
        "mean_variance_label": "Rendite/Risiko - Max-Sharpe",
        "expected_return": "4 * predicted quarterly log return",
        "covariance": "252 * sample covariance of 252 daily log returns from 253 completed NYSE levels",
        "risk_free": "log1p(0.015), annual log risk-free rate",
        "objective": "(annual_log_mean @ weights - log1p(0.015)) / sqrt(weights @ annual_log_covariance @ weights)",
        "return_basis_constraint": "Targets and covariance must use the same price-only or gross-TR level definition",
        "portfolio_approximation": "Linear weighted asset log means and log covariance approximate the arithmetic-weight portfolio's log mean/volatility. Not exact expected simple-return Sharpe; no exp(log_mean)-1 expectation claim.",
        "horizon_assumption": "Four-quarter mean scaling and 252-day covariance scaling assume stable mean and negligible daily serial covariance; not inferred from realized backtest results.",
        "fixed_risk_free_assumption": "1.5% effective annual rate is fixed model policy, not a proven paper optimizer parameter or historical yield curve",
        "matlab_equivalence": "NOT_PROVEN: exact MATLAB frontier point, optimizer settings, risk-free rate and annualization conventions unavailable",
        "constraints": {"long_only": True, "maximum_stock_weight": 0.05,
                        "fully_invested": True, "leverage": False},
        "minimum_variance": "Minimize the same annual log covariance quadratic form; forecast means do not enter this objective",
    }


def research_weights(predicted_quarter_log: pd.Series, covariance_levels: pd.DataFrame,
                     cutoff: str) -> tuple[dict, dict]:
    """Versioned annual-log Max-Sharpe approximation; not conventional simple-return Sharpe."""
    selected = predicted_quarter_log
    if selected.empty:
        raise ValueError("Empty selection")
    if len(selected) < math.ceil(1 / RULES.weight_cap):
        raise OptimizationError("Full investment under 5% cap needs at least 20 selected securities")
    if selected.index.has_duplicates or not np.isfinite(selected).all():
        raise ValueError("Invalid predicted returns")
    expected, covariance, risk_free = optimization_inputs(selected, covariance_levels, cutoff)
    bounds = [(0, RULES.weight_cap)] * len(selected)
    initial = np.ones(len(selected)) / len(selected)
    results = {"equal_weight": dict(zip(selected.index, initial))}
    errors = {}
    for strategy in ("mean_var", "min_var"):
        if strategy == "mean_var" and np.max(np.diag(covariance)) <= 1e-16:
            errors[strategy] = "OPTIMIZATION_FAILED: Zero log covariance; Max-Sharpe ratio undefined"
            continue
        def objective(weights):
            variance = float(weights @ covariance @ weights)
            if strategy == "min_var":
                return variance
            return -float((expected @ weights - risk_free) / math.sqrt(max(variance, 1e-16)))
        fit = minimize(objective, initial, bounds=bounds,
                       constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
                       method="SLSQP", options={"maxiter": 1000, "ftol": 1e-10})
        if (not fit.success or not np.isfinite(fit.x).all()
                or abs(fit.x.sum() - 1) > 1e-7 or (fit.x < -1e-8).any()
                or (fit.x > RULES.weight_cap + 1e-8).any()):
            errors[strategy] = f"OPTIMIZATION_FAILED: {fit.message}"
        elif strategy == "mean_var" and float(fit.x @ covariance @ fit.x) <= 1e-16:
            errors[strategy] = "OPTIMIZATION_FAILED: Zero portfolio log variance; Max-Sharpe ratio undefined"
        else:
            results[strategy] = dict(zip(selected.index, fit.x))
    return results, errors


def rule_manifest() -> dict:
    result = asdict(RULES)
    result["optimization_definition"] = optimization_definition()
    result["assumptions"] = [
        "Paper does not fix intraday signal/execution times: prior close cutoff, next fixed session close execution.",
        "Four fully matured validation quarters require one decision-quarter maturity gap.",
        "Ambiguous stock-deletion rule is replaced by training-only feature filtering and complete cases; not exact replication.",
        "Ceiling(20% of eligible securities) with security-ID tie-break replaces quantile >= 0.8.",
        "Annual log expected returns, annual log covariance and log(1+annual risk-free rate); historical RF not supplied.",
        "Target and covariance must share the reviewed return definition: split-adjusted price or gross TR. Gross-TR targets are an explicit deviation from the paper price target.",
        "Historical sector/identity, factor definitions and TR evidence are prerequisites, not inferred from current caches.",
    ]
    return result
