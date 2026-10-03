"""Summarize the live model output: which stocks the model currently recommends
(top 20% predicted return per sector, at the most recent trade date) and how
often each stock has historically been picked.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_sector_predictions() -> dict[str, pd.DataFrame]:
    """sector_name -> DataFrame indexed by trade_date, columns = tickers, predicted next-quarter return."""
    out = {}
    for sector_dir in sorted(RESULTS_DIR.glob("sector*")):
        f = sector_dir / "df_predict_best.csv"
        if f.exists():
            out[sector_dir.name] = pd.read_csv(f, index_col=0, parse_dates=True)
    return out


def top20pct_per_quarter(df_predict: pd.DataFrame) -> pd.DataFrame:
    """Boolean DataFrame, same shape as df_predict: True where a ticker was in
    the top 20% of predicted returns for that sector in that quarter."""
    threshold = df_predict.quantile(0.8, axis=1)
    return df_predict.ge(threshold, axis=0) & df_predict.notna()


def main():
    sector_predictions = load_sector_predictions()
    if not sector_predictions:
        print(f"No results found under {RESULTS_DIR}. Run pipeline/run_model.py first.")
        return

    latest_picks = []
    history_rows = []
    for sector, df_predict in sector_predictions.items():
        picks = top20pct_per_quarter(df_predict)
        pick_rate = picks.mean(axis=0)  # fraction of quarters each ticker was picked
        n_quarters = df_predict.notna().sum(axis=0)  # quarters a ticker had a prediction at all
        for tic in df_predict.columns:
            history_rows.append({
                "sector": sector, "tic": tic,
                "quarters_covered": int(n_quarters[tic]),
                "times_picked": int(picks[tic].sum()),
                "pick_rate": float(pick_rate[tic]) if n_quarters[tic] > 0 else float("nan"),
            })

        # Different companies report on different fiscal calendars, so the most
        # recent rebalance quarter is naturally only partially populated; also
        # exclude synthetic future rebalance dates (next quarter hasn't started yet).
        today = pd.Timestamp.today().normalize()
        past_dates = df_predict.index[df_predict.index <= today]
        if len(past_dates) == 0:
            continue
        latest_date = past_dates.max()
        latest_row = df_predict.loc[latest_date].dropna()
        latest_picked = picks.loc[latest_date]
        for tic, pred_return in latest_row.items():
            latest_picks.append({
                "sector": sector, "tic": tic, "trade_date": latest_date,
                "predicted_return": pred_return, "picked": bool(latest_picked.get(tic, False)),
            })

    history_df = pd.DataFrame(history_rows).sort_values(["sector", "pick_rate"], ascending=[True, False])
    latest_df = pd.DataFrame(latest_picks).sort_values(["sector", "predicted_return"], ascending=[True, False])

    history_df.to_csv(RESULTS_DIR / "pick_history_summary.csv", index=False)
    latest_df.to_csv(RESULTS_DIR / "latest_quarter_predictions.csv", index=False)

    latest_selected = latest_df[latest_df["picked"]].sort_values("predicted_return", ascending=False)
    latest_selected.to_csv(RESULTS_DIR / "latest_picks.csv", index=False)

    never_picked = history_df[(history_df["quarters_covered"] > 0) & (history_df["times_picked"] == 0)]
    always_picked = history_df[(history_df["quarters_covered"] > 4) & (history_df["pick_rate"] >= 0.95)]

    print(f"Most recent trade date across sectors: {latest_df['trade_date'].max()}")
    print(f"\nCurrently selected (top 20% per sector), {len(latest_selected)} stocks:")
    print(latest_selected[["sector", "tic", "predicted_return"]].to_string(index=False))
    print(f"\nNever picked in any quarter ({len(never_picked)} stocks) -> see {RESULTS_DIR / 'pick_history_summary.csv'}")
    print(f"Almost always picked (>=95% of covered quarters, {len(always_picked)} stocks):")
    print(always_picked[["sector", "tic", "quarters_covered", "pick_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
