"""Run the paper's rolling-window model-selection pipeline (code/ml_model.py)
on the free-data tables built by build_dataset.py, one GICS sector at a time.

Safe to re-run: just re-run build_dataset.py first to refresh data, then this.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))
import ml_model  # noqa: E402

DATA_DIR = REPO_ROOT / "Data" / "live"
# ml_model.save_model_result() writes to a hardcoded 'results/<sector>/' path
# relative to the current working directory, so run this script from REPO_ROOT.

NO_FEATURE_COLUMNS = ["tic", "conm", "datadate", "tradedate", "gsector", "y_return", "data_source", "in_current_sp500"]
PAPER_FEATURES = [
    "X1_REVGH", "X2_EPS", "X3_ROA", "X4_ROE", "X5_PE", "X6_PS", "X7_NPM",
    "X8_GPM", "X9_OM", "X10_PB", "X11_PCFO", "X12_CR", "X13_EM", "X14_EVCFO",
    "X15_LTDTA", "X16_WCR", "X17_DE", "X18_QR", "X19_DSI", "X20_DPO",
]


def run_sector(path: Path, first_trade_date_index: int, testing_windows: int,
               max_rolling_window_index: int, features_only: bool = False,
               trade_start_date: pd.Timestamp | None = None):
    sector_data = pd.read_csv(path, parse_dates=["tradedate"])
    sector_name = path.stem.replace("_hybrid", "")

    sector_data = sector_data.sort_values(["tic", "tradedate", "datadate"])
    sector_data = sector_data.drop_duplicates(["tic", "tradedate"], keep="last")
    sector_data = sector_data[sector_data["tradedate"] <= pd.Timestamp.today().normalize()]
    historical_source = REPO_ROOT / "Data" / "1-focasting_data" / f"{sector_name}_clean.xlsx"
    live_source = DATA_DIR / f"{sector_name}_live.csv"
    compustat_columns = set(pd.read_excel(historical_source, nrows=0).columns)
    sec_columns = set(pd.read_csv(live_source, nrows=0).columns)
    present_features = [name for name in PAPER_FEATURES
                        if name in sector_data.columns and name in compustat_columns and name in sec_columns]
    if not present_features:
        print(f"{sector_name}: no indicators are shared by Compustat and SEC, skipping")
        return None
    sector_data[present_features] = sector_data[present_features].replace([np.inf, -np.inf], np.nan)
    if trade_start_date is not None:
        feature_reference = sector_data[sector_data["tradedate"] >= trade_start_date]
    else:
        feature_reference = sector_data
    missing_rate = feature_reference[present_features].isna().mean()
    features_column = missing_rate[missing_rate <= 0.05].index.tolist()
    if not features_column:
        print(f"{sector_name}: no features remain after 5% missing-data filter, skipping")
        return None
    feature_complete = sector_data[features_column].notna().all(axis=1)
    output_dir = REPO_ROOT / "results" / sector_name
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = len(sector_data)
    compustat_rows = sector_data[sector_data["data_source"] == "WRDS_Compustat_original"]
    sec_rows = sector_data[sector_data["data_source"] == "SEC_free_reconstructed"]
    manifest = pd.DataFrame({
        "feature": PAPER_FEATURES,
        "present_in_compustat_source": [name in compustat_columns for name in PAPER_FEATURES],
        "compustat_missing_rate": [float(compustat_rows[name].isna().mean()) if name in compustat_columns else 1.0
                                    for name in PAPER_FEATURES],
        "present_in_sec_source": [name in sec_columns for name in PAPER_FEATURES],
        "sec_missing_rate": [float(sec_rows[name].isna().mean()) if name in sec_columns and len(sec_rows) else 1.0
                             for name in PAPER_FEATURES],
        "missing_count": [int(sector_data[name].isna().sum()) if name in present_features else row_count
                          for name in PAPER_FEATURES],
        "live_reference_missing_rate": [float(missing_rate[name]) if name in present_features else 1.0
                         for name in PAPER_FEATURES],
        "included_in_training": [name in features_column for name in PAPER_FEATURES],
        "reason_excluded": ["" if name in features_column else
                            "source_feature_absent" if name not in compustat_columns or name not in sec_columns else
                            "missing_rate_above_5_percent" for name in PAPER_FEATURES],
        "complete_rows_after_filter": [int(feature_complete.sum()) if name in features_column else 0
                                       for name in PAPER_FEATURES],
    })
    manifest.insert(0, "sector", sector_name)
    manifest.to_csv(output_dir / "feature_manifest.csv", index=False)

    coverage_rows = []
    for trade_date, period in sector_data.groupby("tradedate", sort=True):
        expected_tickers = int(period["tic"].nunique())
        for name in PAPER_FEATURES:
            observed = int(period[name].notna().sum()) if name in present_features else 0
            coverage_rows.append({
                "sector": sector_name,
                "trade_date": trade_date,
                "feature": name,
                "tickers_expected": expected_tickers,
                "tickers_with_value": observed,
                "coverage_rate": observed / expected_tickers if expected_tickers else np.nan,
                "included_in_training": name in features_column,
            })
    pd.DataFrame(coverage_rows).to_csv(output_dir / "feature_quarter_coverage.csv", index=False)

    if features_only:
        print(f"{sector_name}: {len(features_column)}/20 features selected; audit written")
        return None

    sector_data = sector_data.dropna(subset=features_column)

    # y_return is only genuinely unknown for the latest (current) quarter - a stock's
    # ragged/shorter free-data history can otherwise leave NaN labels mid-history
    # (e.g. no next-quarter price available yet), which breaks model fitting if left in
    # the training/testing window. Drop those, but always keep the current quarter's
    # label-less rows since that's exactly what we want predictions for.
    latest_date = sector_data["tradedate"].max()
    sector_data = sector_data[sector_data["y_return"].notna() | (sector_data["tradedate"] == latest_date)]

    unique_datetime = sorted(sector_data["tradedate"].unique())
    unique_ticker = sorted(sector_data["tic"].unique())

    if len(unique_datetime) <= first_trade_date_index:
        print(f"{sector_name}: not enough history ({len(unique_datetime)} quarters), skipping")
        return None

    trade_date = [date for date in unique_datetime[first_trade_date_index:]
                  if trade_start_date is None or date >= trade_start_date]
    if not trade_date:
        print(f"{sector_name}: no trade dates on/after {trade_start_date}, skipping")
        return None

    start = time.time()
    result = ml_model.run_4model(
        sector_data,
        features_column,
        "y_return",
        "tradedate",
        "tic",
        unique_ticker,
        unique_datetime,
        trade_date,
        first_trade_date_index,
        testing_windows,
        max_rolling_window_index,
        trade_start_date=trade_start_date,
    )
    print(f"{sector_name}: done in {(time.time() - start) / 60:.1f} min")
    ml_model.save_model_result(result, sector_name)
    return result


def main(first_trade_date_index: int, testing_windows: int, max_rolling_window_index: int,
         sector: str | None, features_only: bool, trade_start_date: pd.Timestamp | None):
    files = sorted(DATA_DIR.glob("sector*_hybrid.csv"))
    if sector:
        files = [f for f in files if sector in f.stem]
    if not files:
        print(f"No hybrid sector files found in {DATA_DIR}. Run build_dataset.py first.")
        return
    for f in files:
        run_sector(f, first_trade_date_index, testing_windows, max_rolling_window_index,
               features_only, trade_start_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_trade_index", type=int, default=20, help="16 training + 4 testing quarters before first trade")
    parser.add_argument("--testing_window", type=int, default=4, help="quarters held out for model validation (default 1 year)")
    parser.add_argument("--max_rolling_window", type=int, default=44, help="40 training quarters plus 4 testing quarters")
    parser.add_argument("--sector", type=str, default=None, help="only run one sector, e.g. sector45")
    parser.add_argument("--features-only", action="store_true", help="write feature coverage manifests without fitting models")
    parser.add_argument("--trade-start-date", type=pd.Timestamp, default=pd.Timestamp("2017-09-01"),
                        help="emit predictions from this date while retaining older rows for training")
    args = parser.parse_args()
    main(args.first_trade_index, args.testing_window, args.max_rolling_window, args.sector,
         args.features_only, args.trade_start_date)
