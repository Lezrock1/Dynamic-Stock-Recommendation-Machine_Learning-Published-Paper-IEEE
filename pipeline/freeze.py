"""Immutable input snapshot (file manifest with SHA-256) and an offline guard for diagnostic runs.

`NetworkGuard` turns every network path into a counted failure and makes the price/SEC readers
pure cache readers (no refresh, no marker or mtime updates), so a run provably uses the manifest's bytes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

try:
    from . import paper_live as pl, prices as pr, sec_edgar as se, universe as uv, universe_asof as ua
except ImportError:  # pragma: no cover
    import paper_live as pl, prices as pr, sec_edgar as se, universe as uv, universe_asof as ua  # type: ignore

import pandas as pd
import requests

INPUT_DIRS = ("secfacts", "secsubmissions", "prices", "filings", "nasdaq_dividends")
INPUT_FILES_GLOB = ("sector_feature_sets.json", "sp500_*", "sec_company_tickers.json", "universe_sector_lookup.json")


class NetworkBlocked(requests.RequestException):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_files(config: Mapping) -> list[Path]:
    root = pl.resolve_path(config, "database").parent.parent
    cache = root / "cache"
    files: list[Path] = []
    for sub in INPUT_DIRS:
        if (cache / sub).exists():
            files += [f for f in (cache / sub).rglob("*") if f.is_file() and f.name != ".DS_Store"]
    for pattern in INPUT_FILES_GLOB:
        files += [f for f in cache.glob(pattern) if f.is_file()]
    for sub in ("Data", "code", "pipeline", "tests"):
        directory = root / sub
        if directory.exists():
            files += [f for f in directory.rglob("*")
                      if f.is_file() and f.name != ".DS_Store"
                      and (sub not in ("code", "pipeline") or f.suffix == ".py"
                           or f.name == "requirements.txt")]
    manifest_dir = pl.resolve_path(config, "manifest_dir")
    files += [f for f in manifest_dir.glob("*.json") if f.is_file()]
    source_dir = root / "state" / "sources"
    if source_dir.exists():
        files += [f for f in source_dir.rglob("*") if f.is_file() and f.name != ".DS_Store"]
    database = pl.resolve_path(config, "database")
    if database.is_file():
        files.append(database)
        for suffix in ("-wal", "-journal"):
            sidecar = Path(str(database) + suffix)
            if sidecar.is_file():
                files.append(sidecar)
    config_path = Path(str(config.get("_path") or root / "config" / "real_paper.json"))
    if config_path.is_file():
        files.append(config_path)
    requirements = root / "pipeline" / "requirements.txt"
    if requirements.is_file():
        files.append(requirements)
    return sorted(set(files))


def build_file_manifest(config: Mapping) -> dict:
    root = pl.resolve_path(config, "database").parent.parent
    entries = {}
    for f in input_files(config):
        entries[f.relative_to(root).as_posix()] = {"sha256": _sha256(f), "size": f.stat().st_size}
    combined = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"file_count": len(entries), "combined_sha256": combined, "files": entries}


def compare_manifests(before: Mapping, after: Mapping) -> dict:
    a, b = before["files"], after["files"]
    return {"unchanged": a == b, "added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)),
            "modified": sorted(k for k in set(a) & set(b) if a[k] != b[k])}


def create_input_snapshot(config: Mapping, destination: str | Path) -> Path:
    """Copy every declared data, code, config, and run-state input; SQLite is copied consistently."""
    root = pl.resolve_path(config, "database").parent.parent
    database = pl.resolve_path(config, "database")
    if not database.is_file():
        raise FileNotFoundError(f"required run database is missing: {database}")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for source in input_files(config):
        if source == database or source in {
            Path(str(database) + "-wal"), Path(str(database) + "-journal")
        }:
            continue
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    target_database = destination / database.relative_to(root)
    target_database.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=30)
    target_connection = sqlite3.connect(target_database)
    try:
        source_connection.backup(target_connection)
        if target_connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise sqlite3.DatabaseError("run database snapshot failed its integrity check")
    finally:
        target_connection.close()
        source_connection.close()
    return target_database


def config_for_snapshot(config: Mapping, snapshot_root: str | Path) -> dict:
    """Point the existing real config at its corresponding paths within the frozen copy."""
    source_root = pl.resolve_path(config, "database").parent.parent
    snapshot_root = Path(snapshot_root).resolve()
    result = dict(config)
    result["database"] = str(snapshot_root / pl.resolve_path(config, "database").relative_to(source_root))
    result["manifest_dir"] = str(
        snapshot_root / pl.resolve_path(config, "manifest_dir").relative_to(source_root)
    )
    config_path = Path(str(config.get("_path") or source_root / "config" / "real_paper.json"))
    result["_path"] = str(snapshot_root / config_path.relative_to(source_root))
    return result


@contextmanager
def snapshot_context(root: str | Path):
    """Redirect data/model imports to the frozen input tree for an offline run."""
    root = Path(root).resolve()
    previous = (
        pl.REPO_ROOT, pr.CACHE_DIR, se.CACHE_DIR, se.SUBMISSIONS_DIR,
        uv.CACHE_FILE, uv.HISTORICAL_MEMBERSHIP_CACHE, uv.SEC_TICKERS_CACHE,
        ua.CACHE, ua.SECTOR_LOOKUP_CACHE, list(sys.path),
    )
    prior_ml_model = sys.modules.pop("ml_model", None)
    pl.REPO_ROOT = root
    pr.CACHE_DIR = root / "cache" / "prices"
    se.CACHE_DIR = root / "cache" / "secfacts"
    se.SUBMISSIONS_DIR = root / "cache" / "secsubmissions"
    uv.CACHE_FILE = root / "cache" / "sp500_universe.csv"
    uv.HISTORICAL_MEMBERSHIP_CACHE = root / "cache" / "sp500_historical_snapshots.csv"
    uv.SEC_TICKERS_CACHE = root / "cache" / "sec_company_tickers.json"
    ua.CACHE = root / "cache" / "sp500_changes.json"
    ua.SECTOR_LOOKUP_CACHE = root / "cache" / "universe_sector_lookup.json"
    sys.path.insert(0, str(root / "code"))
    try:
        yield
    finally:
        (pl.REPO_ROOT, pr.CACHE_DIR, se.CACHE_DIR, se.SUBMISSIONS_DIR,
         uv.CACHE_FILE, uv.HISTORICAL_MEMBERSHIP_CACHE, uv.SEC_TICKERS_CACHE,
         ua.CACHE, ua.SECTOR_LOOKUP_CACHE, original_path) = previous
        sys.path[:] = original_path
        sys.modules.pop("ml_model", None)
        if prior_ml_model is not None:
            sys.modules["ml_model"] = prior_ml_model


@contextmanager
def network_guard():
    """Yield a list that records every blocked network attempt."""
    attempts: list[str] = []

    def blocked_request(self, method, url, *args, **kwargs):
        attempts.append(f"{method} {url}")
        raise NetworkBlocked(f"offline run: {method} {url}")

    def cache_only_prices(ticker: str, start: str = "2005-01-01"):
        cache_file = pr.CACHE_DIR / f"{ticker}.csv"
        if not cache_file.exists():
            attempts.append(f"price cache missing: {ticker}")
            return None
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    def blocked_yf(*args, **kwargs):
        attempts.append("yfinance")
        raise NetworkBlocked("offline run: yfinance")

    def cached(fn):
        return lambda cik, max_age_seconds=0: fn(cik, 10 ** 12)

    def cached_universe(force_refresh=False):
        if not uv.CACHE_FILE.is_file():
            raise FileNotFoundError(f"frozen universe cache missing: {uv.CACHE_FILE}")
        return pd.read_csv(uv.CACHE_FILE)

    def cached_membership(force_refresh=False):
        if not uv.HISTORICAL_MEMBERSHIP_CACHE.is_file():
            raise FileNotFoundError(f"frozen historical membership cache missing: {uv.HISTORICAL_MEMBERSHIP_CACHE}")
        return pd.read_csv(uv.HISTORICAL_MEMBERSHIP_CACHE, dtype={"date": str, "tickers": str})

    def cached_sec_tickers(force_refresh=False):
        if not uv.SEC_TICKERS_CACHE.is_file():
            raise FileNotFoundError(f"frozen SEC ticker cache missing: {uv.SEC_TICKERS_CACHE}")
        records = json.loads(uv.SEC_TICKERS_CACHE.read_text(encoding="utf-8")).values()
        return {uv._normalize_ticker(row["ticker"]): row for row in records}

    def cached_changes(max_age_seconds=0):
        if not ua.CACHE.is_file():
            raise FileNotFoundError(f"frozen S&P changes cache missing: {ua.CACHE}")
        return json.loads(ua.CACHE.read_text(encoding="utf-8"))["wikitext"]

    originals = (requests.sessions.Session.request, pr.get_price_history, pr.yf.download, pr.yf.Ticker,
                 se._fetch_company_facts, se._fetch_submissions, uv.get_sp500_universe,
                 uv.get_historical_membership_snapshots, uv._get_sec_ticker_map, ua.fetch_changes_wikitext)
    requests.sessions.Session.request = blocked_request
    pr.get_price_history = cache_only_prices
    pr.yf.download = blocked_yf
    pr.yf.Ticker = blocked_yf
    se._fetch_company_facts = cached(originals[4])
    se._fetch_submissions = cached(originals[5])
    uv.get_sp500_universe = cached_universe
    uv.get_historical_membership_snapshots = cached_membership
    uv._get_sec_ticker_map = cached_sec_tickers
    ua.fetch_changes_wikitext = cached_changes
    try:
        yield attempts
    finally:
        (requests.sessions.Session.request, pr.get_price_history, pr.yf.download, pr.yf.Ticker,
         se._fetch_company_facts, se._fetch_submissions, uv.get_sp500_universe,
         uv.get_historical_membership_snapshots, uv._get_sec_ticker_map, ua.fetch_changes_wikitext) = originals
