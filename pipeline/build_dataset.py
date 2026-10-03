"""Bulk-fetch fundamentals + prices for the full free-data universe and assemble
one fundamentals table per GICS sector, in the schema the original ml_model.py
pipeline expects (tic, tradedate, gsector, X1_REVGH..X20_DPO, y_return).

Safe to re-run any time: SEC/price data are cached on disk, so re-running just
picks up newly available quarters.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prices as pr
import ratios as ra
import sec_edgar as se
import universe as uv

OUT_DIR = Path(__file__).resolve().parent.parent / "Data" / "live"
HISTORICAL_DIR = Path(__file__).resolve().parent.parent / "Data" / "1-focasting_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_FEATURES = [
    "X1_REVGH", "X2_EPS", "X3_ROA", "X4_ROE", "X5_PE", "X6_PS", "X7_NPM",
    "X8_GPM", "X9_OM", "X10_PB", "X11_PCFO", "X12_CR", "X13_EM", "X14_EVCFO",
    "X15_LTDTA", "X16_WCR", "X17_DE", "X18_QR", "X19_DSI", "X20_DPO",
]
HISTORICAL_CUTOFF = pd.Timestamp("2017-09-01")


def build_for_ticker(tic: str, cik: int) -> pd.DataFrame | None:
    try:
        fund = se.get_quarterly_fundamentals(cik)
    except Exception as exc:
        print(f"  fundamentals fetch failed for {tic}: {exc}")
        return None
    if fund is None or fund.empty:
        return None
    price_df = pr.get_price_history(tic)
    if price_df is None or price_df.empty:
        return None
    required = ["Revenue", "NetIncome", "Assets", "StockholdersEquity"]
    fund = fund.dropna(subset=[c for c in required if c in fund.columns])
    if fund.empty:
        return None
    try:
        return ra.compute_ratios(fund, price_df, tic)
    except Exception as exc:  # keep going for the rest of the universe
        print(f"  ratio computation failed for {tic}: {exc}")
        return None


def build_hybrid_sector(sector_name: str, live_sector: pd.DataFrame) -> pd.DataFrame:
    historical_path = HISTORICAL_DIR / f"{sector_name}_clean.xlsx"
    historical = pd.read_excel(historical_path)
    historical["tradedate"] = pd.to_datetime(
        historical["tradedate"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    if "datadate" in historical.columns:
        historical["datadate"] = pd.to_datetime(
            historical["datadate"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
        )
    historical = historical[historical["tradedate"] < HISTORICAL_CUTOFF].copy()

    live = live_sector.copy()
    live["tradedate"] = pd.to_datetime(live["tradedate"], errors="coerce")
    live = live[live["tradedate"] >= HISTORICAL_CUTOFF].copy()

    metadata = ["tic", "datadate", "tradedate", "conm", "gsector", "y_return", "in_current_sp500"]
    for frame in (historical, live):
        for name in metadata:
            if name not in frame.columns:
                frame[name] = pd.NA
        for name in PAPER_FEATURES:
            if name not in frame.columns:
                frame[name] = np.nan
    historical = historical[metadata + PAPER_FEATURES].copy()
    live = live[metadata + PAPER_FEATURES].copy()
    historical["data_source"] = "WRDS_Compustat_original"
    live["data_source"] = "SEC_free_reconstructed"
    combined = pd.concat([historical, live], ignore_index=True)
    combined = combined.sort_values(["tic", "tradedate", "datadate"])
    combined = combined.drop_duplicates(["tic", "tradedate"], keep="last")
    return combined.reset_index(drop=True)


def main(limit: int | None = None):
    uni = uv.get_sp500_universe()
    if limit:
        uni = uni.head(limit)

    all_rows = []
    failures = []
    start = time.time()
    for i, row in uni.iterrows():
        tic, cik, gsector = row["tic"], row["cik"], row["gsector"]
        print(f"[{i + 1}/{len(uni)}] {tic} (sector {gsector}) ...")
        try:
            df = build_for_ticker(tic, cik)
        except Exception as exc:
            print(f"  unexpected failure for {tic}: {exc}")
            df = None
        if df is None:
            failures.append(tic)
            continue
        df["gsector"] = gsector
        df["conm"] = row["conm"]
        df["in_current_sp500"] = bool(row["is_current"])
        all_rows.append(df)

    if not all_rows:
        print("No data collected.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    full, membership_audit = uv.apply_historical_membership_filter(full)
    membership_audit.to_csv(OUT_DIR / "sp500_membership_coverage.csv", index=False)
    full.to_csv(OUT_DIR / "fundamental_live_table.csv", index=False)

    hybrid_tables = []
    for sector, group in full.groupby("gsector"):
        sector_name = f"sector{int(sector)}"
        group.to_csv(OUT_DIR / f"{sector_name}_live.csv", index=False)
        hybrid = build_hybrid_sector(sector_name, group)
        hybrid.to_csv(OUT_DIR / f"{sector_name}_hybrid.csv", index=False)
        hybrid_tables.append(hybrid)

    pd.concat(hybrid_tables, ignore_index=True).to_csv(OUT_DIR / "fundamental_hybrid_table.csv", index=False)

    print(f"\nDone in {(time.time() - start) / 60:.1f} min.")
    print(f"Tickers with data: {full['tic'].nunique()} / {len(uni)}")
    print(f"Membership-filtered rows: {len(full)}")
    if not membership_audit.empty:
        print(f"Membership snapshots: {membership_audit.membership_snapshot_date.min().date()} to {membership_audit.membership_snapshot_date.max().date()}")
        print(f"Oldest snapshot age: {membership_audit.snapshot_age_days.max()} days")
    print(f"Failed / skipped: {len(failures)} -> {failures}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N tickers (for testing)")
    args = parser.parse_args()
    main(limit=args.limit)
