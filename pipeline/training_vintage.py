"""Separate a row's economic period from its admissible input-version date."""
from __future__ import annotations

import pandas as pd

STRICT = "own-decision-vintage-v1"
AT_FIT = "published-version-at-training-v1"


def feature_limits(cycles: pd.Series, target_cycle: str | None = None,
                   policy: str = STRICT) -> pd.Series:
    from .paper_fixed_lag import RULES, schedule

    if policy not in {STRICT, AT_FIT}:
        raise ValueError(f"Unknown training vintage policy: {policy}")
    limits = pd.to_datetime(
        cycles.map(lambda c: schedule(str(c))["information_cutoff"]), utc=True)
    if policy == STRICT:
        return limits
    if target_cycle is None:
        raise ValueError("Training-time vintage policy requires an explicit target cycle")
    target = pd.Period(target_cycle, "Q")
    # The last matured quarter is target-2; all four validation decisions stay causal.
    first_validation = target - (RULES.validation_quarters + 1)
    training = cycles.map(lambda c: pd.Period(c, "Q") < first_validation)
    limits.loc[training] = pd.Timestamp(schedule(target_cycle)["information_cutoff"])
    return limits
