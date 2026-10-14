"""Explicitly sourced S&P 500 gross total-return / SPY-proxy adapters."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


class BenchmarkDataError(ValueError):
    """Raised if benchmark identity, provenance, or observations are invalid."""


@dataclass(frozen=True)
class BenchmarkSeries:
    values: pd.Series
    metadata: dict[str, object]

    @classmethod
    def from_files(
        cls,
        csv_path: str | Path,
        metadata_path: str | Path,
        *,
        expected_trading_dates: Sequence[str | pd.Timestamp] | None = None,
    ) -> "BenchmarkSeries":
        """Load `date,total_return_index` plus an explicit provenance manifest.

        A CSV column name is not proof of index identity. The manifest must
        state the identity, source, dividend convention, dates and verification
        evidence; it must be reviewed and supplied by the caller.
        """
        csv_path = Path(csv_path)
        metadata_path = Path(metadata_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkDataError(f"cannot read benchmark metadata {metadata_path}: {exc}") from exc
        required = {
            "type", "source", "index_identity", "currency", "dividend_treatment",
            "start_date", "end_date", "retrieved_at", "verified",
            "verification_method", "evidence_uri",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise BenchmarkDataError(f"benchmark metadata is missing: {', '.join(missing)}")
        if metadata["verified"] is not True:
            raise BenchmarkDataError("benchmark metadata must be explicitly verified")
        for field in ("source", "index_identity", "verification_method", "evidence_uri"):
            if not isinstance(metadata[field], str) or not metadata[field].strip():
                raise BenchmarkDataError(f"benchmark metadata {field} must be a non-empty string")
        if metadata["currency"] != "USD":
            raise BenchmarkDataError("benchmark currency must be USD")
        kind = metadata["type"]
        if kind == "sp500_gross_total_return":
            if "S&P 500" not in str(metadata["index_identity"]):
                raise BenchmarkDataError("primary benchmark identity must identify the S&P 500")
            if metadata["dividend_treatment"] != "gross_dividends_reinvested_before_withholding":
                raise BenchmarkDataError("primary benchmark must reinvest gross dividends before withholding")
            display_name = "S&P 500 Gross Total Return (USD)"
        elif kind == "spy_total_return_proxy":
            if "SPY" not in str(metadata["index_identity"]):
                raise BenchmarkDataError("SPY alternative must explicitly identify SPY")
            if metadata["dividend_treatment"] != "cash_distributions_reinvested_total_return":
                raise BenchmarkDataError("SPY proxy must explicitly reinvest cash distributions")
            if metadata.get("fund_expenses_embedded") is not True:
                raise BenchmarkDataError("SPY proxy metadata must state fund expenses are already embedded")
            display_name = "SPY ETF proxy (distributions reinvested)"
        else:
            raise BenchmarkDataError(
                "benchmark type must be sp500_gross_total_return or spy_total_return_proxy"
            )

        try:
            observations = pd.read_csv(csv_path, usecols=["date", "total_return_index"])
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise BenchmarkDataError(f"cannot read benchmark data {csv_path}: {exc}") from exc
        observations["date"] = pd.to_datetime(observations["date"], errors="coerce").dt.normalize()
        observations["total_return_index"] = pd.to_numeric(
            observations["total_return_index"], errors="coerce"
        )
        if observations["date"].isna().any() or observations["total_return_index"].isna().any():
            raise BenchmarkDataError("benchmark contains invalid dates or total_return_index values")
        if observations["date"].duplicated().any():
            raise BenchmarkDataError("benchmark dates must be unique")
        series = observations.set_index("date")["total_return_index"].sort_index()
        if series.empty or not (series > 0).all():
            raise BenchmarkDataError("benchmark total-return index must be non-empty and positive")
        if series.index[0] != pd.Timestamp(metadata["start_date"]).normalize():
            raise BenchmarkDataError("benchmark start_date does not match the first observation")
        if series.index[-1] != pd.Timestamp(metadata["end_date"]).normalize():
            raise BenchmarkDataError("benchmark end_date does not match the last observation")
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise BenchmarkDataError("benchmark dates must be unique and increasing")

        if expected_trading_dates is not None:
            expected = pd.DatetimeIndex(pd.to_datetime(list(expected_trading_dates))).normalize().unique().sort_values()
            missing_dates = expected.difference(series.index)
            if len(missing_dates):
                first_missing = ", ".join(d.strftime("%Y-%m-%d") for d in missing_dates[:10])
                raise BenchmarkDataError(
                    f"benchmark is missing {len(missing_dates)} expected trading dates: {first_missing}"
                )
            unexpected = series.index.difference(expected)
            if len(unexpected):
                raise BenchmarkDataError(
                    f"benchmark has {len(unexpected)} dates outside the expected trading calendar"
                )

        metadata = dict(metadata)
        metadata["display_name"] = display_name
        metadata["csv_path"] = str(csv_path)
        metadata["metadata_path"] = str(metadata_path)
        return cls(values=series.rename(display_name), metadata=metadata)

    def returns(self) -> pd.Series:
        """Return daily index changes; the first missing value is intentionally NaN."""
        return self.values.pct_change(fill_method=None).rename("benchmark_return")

    def nav(self, initial_capital: float, dates: Sequence[str | pd.Timestamp] | None = None) -> pd.Series:
        """Normalize the total-return index to the supplied starting capital."""
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        values = self.values
        if dates is not None:
            requested = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
            missing = requested.difference(values.index)
            if len(missing):
                raise BenchmarkDataError(f"benchmark does not cover requested date {missing[0].date()}")
            values = values.reindex(requested)
        return (initial_capital * values / values.iloc[0]).rename("benchmark_nav")
