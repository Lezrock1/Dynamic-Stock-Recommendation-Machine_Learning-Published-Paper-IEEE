"""Deterministic SYNTHETIC adapters for tests and the offline demo. Not real market data."""
from __future__ import annotations

import hashlib
import math
from datetime import date
from typing import Mapping, Sequence

try:
    from .paper_runner import (
        Adapters, CorporateAction, FilingObservation, MarketData, SignalResult,
    )
except ImportError:  # pragma: no cover
    from paper_runner import (  # type: ignore
        Adapters, CorporateAction, FilingObservation, MarketData, SignalResult,
    )

SYNTHETIC_ORIGIN = "SYNTHETIC_DEMO"
STRATEGIES = ("equal_weight_v1", "mean_var_v1", "min_var_v1")


class SyntheticFilings:
    def __init__(self) -> None:
        self.arrived: dict[str, str] = {}

    def arrive(self, filing_key: str, filed_at: str) -> None:
        self.arrived[filing_key] = filed_at

    def observe(self, expected: Sequence[Mapping[str, str]]) -> dict[str, FilingObservation]:
        return {
            e["filing_key"]: FilingObservation(
                True, True, True, self.arrived[e["filing_key"]], SYNTHETIC_ORIGIN)
            for e in expected if e["filing_key"] in self.arrived
        }


class SyntheticSignals:
    def __init__(self, symbols: Sequence[str], sectors: int = 5) -> None:
        self.symbols, self.sectors, self.calls = list(symbols), sectors, []

    def predict_as_of(self, signal_as_of: str) -> SignalResult:
        self.calls.append(signal_as_of)
        weight = min(0.05, 0.9 / len(self.symbols))
        return SignalResult(
            signal_as_of=signal_as_of,
            sector_as_of={f"sector_{i}": signal_as_of for i in range(self.sectors)},
            data_snapshot={"origin": SYNTHETIC_ORIGIN, "symbols": len(self.symbols),
                           "eligibility_enforced": True},
            forecasts={s: {"expected_return": 0.01} for s in self.symbols},
            target_weights={name: {s: weight for s in self.symbols} for name in STRATEGIES},
        )


class SyntheticMarket:
    def __init__(self, actions: Mapping[str, Sequence[CorporateAction]] | None = None) -> None:
        self.actions = dict(actions or {})

    def close(self, symbol: str, session: str) -> float | None:
        base = 20 + int(hashlib.sha256(symbol.encode()).hexdigest()[:4], 16) % 80
        days = (date.fromisoformat(session) - date(2020, 1, 1)).days
        return round(base * (1 + 0.0004 * days + 0.01 * math.sin(days / 5)), 4)

    def benchmark(self, session: str) -> float | None:
        return 1000.0 * (1 + 0.0003 * (date.fromisoformat(session) - date(2020, 1, 1)).days)

    def corporate_actions(self, session: str, symbols: Sequence[str] = ()) -> Sequence[CorporateAction]:
        return tuple(a for a in self.actions.get(session, ()) if not symbols or a.symbol in symbols)


def synthetic_adapters(symbols: Sequence[str] = ("AAA", "BBB", "CCC", "DDD")) -> Adapters:
    return Adapters(SyntheticFilings(), SyntheticSignals(symbols), SyntheticMarket(), SYNTHETIC_ORIGIN)


def run_demo(database, symbol_count: int = 4) -> dict[str, object]:
    """Run the full synthetic flow on a NEW database with a simulated clock."""
    from datetime import datetime, timezone
    from pathlib import Path

    try:
        from .paper_dashboard import dashboard_snapshot
        from .paper_runner import PaperRunner
        from .paper_store import PaperStore
    except ImportError:  # pragma: no cover
        from paper_dashboard import dashboard_snapshot
        from paper_runner import PaperRunner
        from paper_store import PaperStore

    if Path(database).exists():
        raise FileExistsError(f"demo requires a new database path: {database}")
    symbols = [f"SYN{i}" for i in range(symbol_count)]
    keys = [f"co{i}-2025Q4" for i in range(10)]
    adapters = synthetic_adapters(symbols)
    store = PaperStore.create(database, "synthetic")
    for name in STRATEGIES:
        store.create_portfolio(name, name, name, 1_000_000.0, created_at="2026-01-05T15:00:00+00:00")
        cycle = store.create_cycle(name, "2025Q4", name, started_at="2026-01-05T15:00:00+00:00")
        store.register_expected_filings(cycle, keys, filing_metadata={
            k: {"company_id": k.split("-")[0], "expected_period": "2025Q4", "filing_type": "10-K"}
            for k in keys})
    clock = {"now": datetime(2026, 1, 5, 16, tzinfo=timezone.utc)}
    runner = PaperRunner(store, adapters, clock=lambda: clock["now"])
    for key in keys[:-1]:
        adapters.filings.arrive(key, "2026-01-05T12:00:00+00:00")
    runner.tick()
    adapters.filings.arrive(keys[-1], "2026-01-06T12:00:00+00:00")
    for stamp in ("2026-01-06T15:00:00", "2026-01-07T22:00:00", "2026-01-08T22:00:00"):
        clock["now"] = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
        runner.tick()
    snapshot = dashboard_snapshot(database)
    return {
        "WARNING": "SYNTHETIC DEMO - generated prices and reports, not real data",
        "model_runs": len(adapters.signals.calls),
        "live_paper_depot": snapshot["live_paper_depot"],
        "historical_backtest_validation": snapshot["historical_backtest_validation"],
        "latest_marks": {p["portfolio_id"]: p["latest_mark"] for p in snapshot["portfolios"]},
    }
