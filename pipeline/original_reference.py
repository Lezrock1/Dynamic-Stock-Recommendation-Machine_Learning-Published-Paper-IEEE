"""Byte-bind the expressly accepted original files, without claiming PIT certification."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .run_model import PAPER_FEATURES

VERSION = "original-repository-reference-assumption-v1"
COMMIT = "0f29f6027fe4061a1484fb7529402706b3903aca"
UPSTREAM = "https://github.com/AI4Finance-Foundation/Dynamic-Stock-Recommendation-Machine_Learning-Published-Paper-IEEE"
SECTORS = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)


def source_paths(root: Path) -> list[Path]:
    return [root / f"Data/1-focasting_data/sector{s}_clean.xlsx" for s in SECTORS]


def load_originals(root: Path) -> tuple[dict[int, pd.DataFrame], dict]:
    frames, sources = {}, []
    for sector, path in zip(SECTORS, source_paths(root)):
        relative = str(path.relative_to(root))
        original = subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=root)
        actual = path.read_bytes()
        if actual != original:
            raise ValueError(f"Not the accepted byte-identical original reference: {relative}")
        frame = pd.read_excel(path)
        absent_features = [field for field in PAPER_FEATURES if field not in frame]
        for field in absent_features:
            frame[field] = np.nan
        for field in ("datadate", "tradedate"):
            frame[field] = pd.to_datetime(
                frame[field].astype("Int64").astype(str), format="%Y%m%d", errors="raise"
            ).dt.strftime("%Y-%m-%d")
        frames[sector] = frame
        sources.append({
            "path": relative, "sha256": hashlib.sha256(actual).hexdigest(),
            "original_commit": COMMIT, "original_commit_timestamp": "2018-06-06T14:30:57Z",
            "source_url": f"{UPSTREAM}/blob/{COMMIT}/{relative}",
            "byte_identical_to_original_git_blob": True,
            "public_availability_at_commit_time_independently_proven": False,
            "absent_original_feature_columns_preserved_as_nan": absent_features,
            "rows": len(frame),
        })
    return frames, {
        "version": VERSION, "files": sources,
        "availability_class": "USER_ACCEPTED_ORIGINAL_REFERENCE_ASSUMPTION",
        "scope": "Only byte-bound original reference features and targets, not SEC continuations, identity or membership",
        "availability_note": "The commit timestamp is not proof of the public push time. No availability timestamp is fabricated.",
        "historical_validation": "NOT_VALIDATED",
    }


def check_rows(rows: pd.DataFrame, original: pd.DataFrame, source: dict) -> list[dict]:
    """Allow only CSV floating-point round-trip noise, never a revised value."""
    result = []
    fields = PAPER_FEATURES + ["y_return"]
    selected = rows.loc[rows["data_source"].eq("WRDS_Compustat_original")]
    original_keys = original.groupby(["tic", "datadate", "tradedate"], dropna=False).indices
    for row in selected.itertuples(index=False):
        key = (row.tic, str(row.datadate)[:10], str(row.tradedate)[:10])
        positions = original_keys.get(key)
        if positions is None or len(positions) != 1:
            raise ValueError(f"Missing or ambiguous original reference row: {key}")
        candidate = original.iloc[positions[0]]
        expected = candidate[fields].to_numpy(dtype=float)
        actual = np.array([getattr(row, field) for field in fields], dtype=float)
        tolerance = 32 * np.finfo(float).eps * np.maximum(1, np.abs(expected))
        equal = (np.isnan(expected) & np.isnan(actual)) | (
            np.isfinite(expected) & np.isfinite(actual) & (np.abs(expected - actual) <= tolerance))
        if not equal.all() or int(row.gsector) != int(candidate.gsector):
            changed = [name for name, same in zip(fields, equal) if not same]
            raise ValueError(f"Original reference content mismatch: {key}; fields={changed}")
        result.append({
            "security_id": row.security_id, "cycle_key": row.cycle_key,
            "gsector": int(row.gsector), "source_sha256": source["sha256"],
            "original_excel_row": int(positions[0]) + 2,
            "checked_features": len(PAPER_FEATURES), "checked_target": True,
            "status": "ORIGINAL_REFERENCE_MATCHED_NOT_PIT_CERTIFIED",
            "floating_comparison": "32_machine_epsilon_scaled_for_csv_roundtrip_only",
        })
    return result
