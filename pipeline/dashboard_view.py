"""Read-only presentation enrichment. No database writes, downloads or model calls."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

if __package__:
    from .dashboard_html import HTML
    from .publication_timing import load_report
    from .publication_timing_view import extend_html
    from .performance_view import extend_html as extend_performance
    from .return_comparison import load_registry
    from .fixed_lag_research import load_summary
    from .target_portfolio import list_packages, inspect_package, TargetPortfolioError
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.dashboard_html import HTML
    from pipeline.publication_timing import load_report
    from pipeline.publication_timing_view import extend_html
    from pipeline.performance_view import extend_html as extend_performance
    from pipeline.return_comparison import load_registry
    from pipeline.fixed_lag_research import load_summary
    from pipeline.target_portfolio import list_packages, inspect_package, TargetPortfolioError

_quarter_control = (
    '<label for="strategy">Strategie</label>',
    '<label for="quarter">Quartal</label><select id="quarter" aria-label="Quartal auswählen">'
    '<option value="">Neuester verfügbarer Lauf</option></select>'
    '<label for="strategy">Strategie</label>',
)
if HTML.count(_quarter_control[0]) != 1 or HTML.count("<script>") != 1:
    raise RuntimeError("Dashboard template no longer has the expected research controls")
HTML = HTML.replace(_quarter_control[0], _quarter_control[1], 1)
HTML = HTML.replace(
    "<script>",
    """<script>
const quarterControl=document.getElementById('quarter');
const dashboardFetch=window.fetch.bind(window);
window.fetch=async(input,init)=>{
  const raw=typeof input==='string'?input:input.url;
  const url=new URL(raw,location.href);
  if(url.pathname==='/api/status'){
    const requested=new URLSearchParams(location.search).get('cycle_key');
    if(requested)url.searchParams.set('cycle_key',requested);
    else url.searchParams.delete('cycle_key');
    const response=await dashboardFetch(url,init);
    const payload=await response.clone().json();
    const research=payload.dashboard_research||{};
    const quarters=[...new Set([...(research.available_quarters||[]),
      ...(payload.publication_timing?.available_quarters||[]),
      ...(payload.fixed_lag_research?.quarters||[]).map(q=>q.cycle_key)])].sort().reverse();
    const options=[new Option('Neuester verfügbarer Lauf',''),
      ...quarters.map(value=>new Option(value,value))];
    const selected=research.cycle_key||'';
    if(selected&&!quarters.includes(selected))
      options.push(new Option(selected+' · nicht verfügbar',selected));
    quarterControl.replaceChildren(...options);
    quarterControl.value=selected;
    return response;
  }
  return dashboardFetch(input,init);
};
quarterControl.addEventListener('change',()=>{
  const next=new URL(location.href);
  if(quarterControl.value)next.searchParams.set('cycle_key',quarterControl.value);
  else next.searchParams.delete('cycle_key');
  location.assign(next);
});
""",
    1,
)

_stock_highlight = (
    'visible.map(r=>`<tr><td><div class="company"',
    'visible.map(r=>`<tr><td class="${isMlSelected(r.symbol)?'
    '\'ml-selected\':\'\'}" title="${isMlSelected(r.symbol)?'
    '\'Im ausgewählten ML-Lauf und in der gewählten Strategie gewichtet\':\'\'}">'
    '<div class="company"',
)
if HTML.count(_stock_highlight[0]) != 1:
    raise RuntimeError("Dashboard template no longer has the expected stock table")
HTML = HTML.replace(_stock_highlight[0], _stock_highlight[1], 1)
HTML = HTML.replace(
    "</style>",
    "#stock-table .ml-selected .company,#stock-table .ml-selected .symbol"
    "{color:#187343}\n</style>",
    1,
)
HTML = HTML.replace(
    "function selectedWeights(){",
    """function isMlSelected(symbol){
  return Object.entries(selectedWeights()||{}).some(
    ([ticker,weight])=>canonical(ticker)===canonical(symbol)&&weight>1e-9);
}
function selectedWeights(){""",
    1,
)
HTML = HTML.replace(
    "$('#stocks-sub').textContent=isR?",
    "$('#stocks-sub').textContent='Grün = im ausgewählten ML-Lauf und in der "
    "gewählten Strategie gewichtet. '+(isR?",
    1,
).replace(
    "'Ziel = freigegebenes Signal · Ist = bewerteter Depotbestand';",
    "'Ziel = freigegebenes Signal · Ist = bewerteter Depotbestand');",
    1,
)

STRATEGIES = ("equal_weight_v1", "mean_var_v1", "min_var_v1")
HTML = extend_html(HTML)
HTML = extend_performance(HTML)
VARIANTS = ("model_run_t50_enforced.json", "model_run_t50_diagnostic.json",
            "model_run_t75_diagnostic.json", "model_run_t100_diagnostic.json")


def _read(path: Path):
    if path.stat().st_size > 25_000_000:
        raise ValueError("Dashboard input exceeds 25 MB")
    raw = path.read_bytes()
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))), raw


def _verified(path: Path, manifest: dict):
    expected = manifest.get("artifacts", {}).get(path.name, {}).get("sha256")
    if not expected:
        raise ValueError(f"Missing artifact hash: {path.name}")
    data, raw = _read(path)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError(f"Artifact hash mismatch: {path.name}")
    return data, digest


def load_research(root: Path, cycle_key: str | None = None) -> dict:
    """Only complete, hash-matching archived suites; never promote them to signals."""
    result = {"available": False, "variants": [], "warnings": [], "latest_attempt": None,
              "executable": False, "label": "Archivierte ML-Analyse · kein Depot-Signal"}
    runs = root / "state" / "diagnostics" / "runs"
    if not runs.is_dir():
        result["available_quarters"] = []
        if cycle_key:
            result["cycle_key"] = cycle_key
            result["warnings"].append(f"Kein vollständiger Archivlauf für {cycle_key} vorhanden.")
        return result
    by_cycle: dict[str, dict] = {}
    for directory in sorted((p for p in runs.iterdir() if p.is_dir()), reverse=True):
        try:
            header, _ = _read(directory / "run_header.json")
            if result["latest_attempt"] is None:
                result["latest_attempt"] = {"run_id": directory.name, "status": header.get("status", "UNKNOWN"),
                                            "time": header.get("evaluation_time")}
            if header.get("status") not in {"COMPLETE_OFFLINE", "COMPLETE"}:
                continue
            manifest, _ = _read(directory / "artifact_manifest.json")
            variants = []
            for filename in VARIANTS:
                if not (directory / filename).is_file():
                    continue
                model, digest = _verified(directory / filename, manifest)
                weights = model.get("target_weights", {})
                for strategy, items in weights.items():
                    if strategy not in STRATEGIES or not isinstance(items, dict):
                        raise ValueError("Unexpected archived weights schema")
                    if any(isinstance(w, bool) or not isinstance(w, (int, float))
                           or not math.isfinite(w) or w < 0 for w in items.values()):
                        raise ValueError("Invalid archived weight")
                    if sum(items.values()) > 1 + 1e-6:
                        raise ValueError("Archived long-only weights exceed 100%")
                snapshot = model.get("data_snapshot", {})
                variants.append({"id": filename.removesuffix(".json"), "sha256": digest,
                                 "threshold": model.get("threshold"),
                                 "eligibility_enforced": model.get("enforce_eligibility") is True,
                                 "signal_as_of": model.get("signal_as_of"),
                                 "target_weights": weights, "forecasts": model.get("forecasts", {}),
                                 "forecast_records": snapshot.get("forecast_records", []),
                                 "optimisation": snapshot.get("portfolio_optimisation", {}),
                                 "sector_models": snapshot.get("sector_models", []),
                                 "price_session": snapshot.get("price_session"),
                                 "universe_as_of": snapshot.get("universe_as_of"), "executable": False})
            if not variants:
                continue
            run_cycle = header.get("cycle_key")
            if not isinstance(run_cycle, str) or not run_cycle:
                result["warnings"].append(f"{directory.name}: cycle_key fehlt.")
                continue
            if run_cycle in by_cycle:
                continue
            selected = {
                "available": True, "variants": variants, "warnings": [],
                "latest_attempt": result["latest_attempt"], "executable": False,
                "label": "Archivierte ML-Analyse · kein Depot-Signal",
                "run_id": directory.name,
                "code_sha256": header.get("code", {}).get("combined_sha256"),
                "cycle_key": run_cycle, "time": header.get("evaluation_time"),
            }
            # Optional issuer labels from the same archive; no network lookup.
            if (directory / "coverage.json").is_file():
                try:
                    coverage, _ = _verified(directory / "coverage.json", manifest)
                    selected["issuer_tickers"] = {}
                    for row in coverage.get("tickers", []):
                        cik, ticker = row.get("cik"), row.get("ticker") or row.get("security_id")
                        if cik is not None and ticker:
                            selected["issuer_tickers"].setdefault(str(int(cik)), []).append(ticker)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    selected["warnings"].append(f"Issuer labels unavailable: {exc}")
            by_cycle[run_cycle] = selected
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result["warnings"].append(f"{directory.name}: {exc}")
    available_quarters = sorted(by_cycle, reverse=True)
    selected_cycle = cycle_key if cycle_key is not None else (
        available_quarters[0] if available_quarters else None
    )
    if selected_cycle in by_cycle:
        result = by_cycle[selected_cycle]
    elif cycle_key is not None:
        result["cycle_key"] = cycle_key
        result["warnings"].append(f"Kein vollständiger Archivlauf für {cycle_key} vorhanden.")
    result["available_quarters"] = available_quarters
    result["requested_cycle"] = cycle_key
    return result


def enhance_snapshot(snapshot: dict, root: Path, cycle_key: str | None = None) -> dict:
    """Keep all original status/target/actual fields intact. Research is a separate namespace."""
    result = copy.deepcopy(snapshot)
    if snapshot.get("environment") == "real":
        result["dashboard_research"] = load_research(Path(root), cycle_key)
        result["publication_timing"] = load_report(
            Path(root), cycle_key or result["dashboard_research"].get("cycle_key")
        )
        result["return_comparison"] = load_registry(Path(root))
        result["fixed_lag_research"] = load_summary(Path(root))
        result["historical_source_progress"] = load_source_progress(Path(root))
        result["numeric_q2_diagnostic"] = load_numeric_diagnostic(Path(root), cycle_key)
        result["target_portfolio_packages"] = load_target_packages(Path(root), cycle_key)
    else:
        result["dashboard_research"] = {"available": False, "variants": [], "executable": False,
                                        "warnings": [], "latest_attempt": None, "available_quarters": [],
                                        "cycle_key": cycle_key}
    result["dashboard_view_version"] = "2026-10-15.2"
    return result


def load_source_progress(root: Path) -> dict:
    pointer = root / "state/research/acquisition/latest.json"
    if not pointer.is_file():
        return {"available": False}
    try:
        reference, _ = _read(pointer)
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to((root / "state/research/acquisition").resolve()):
            raise ValueError("Historical source summary outside acquisition directory")
        summary, raw = _read(path)
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("Historical source summary hash mismatch")
        counts = ("cover_identity_matches", "expected_securities", "original_value_entity_matches",
                  "market_sources_acquired")
        if (any(type(summary.get(key)) is not int or summary[key] < 0 for key in counts)
                or any(not isinstance(summary.get(key), str)
                       for key in ("as_of", "model_run_status", "historical_validation"))
                or not isinstance(summary.get("remaining_blockers"), list)
                or not all(isinstance(value, str) for value in summary["remaining_blockers"])):
            raise ValueError("Historical source summary schema invalid")
        return {"available": True, **summary}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"available": False, "error": f"Quellenfortschritt nicht lesbar: {exc}"}


DIAGNOSTIC_DOWNLOADS = {
    "diagnostic.json", "forecasts_UNVERIFIED.csv", "original_reference_rows.csv",
    *{f"{strategy}_DIAGNOSTIC_UNVERIFIED.csv" for strategy in ("equal_weight", "min_variance", "mean_variance")},
    *{f"{strategy}_DIAGNOSTIC_FAILED.json" for strategy in ("equal_weight", "min_variance", "mean_variance")},
}


def diagnostic_file(root: Path, name: str) -> Path:
    allowed = DIAGNOSTIC_DOWNLOADS
    if name not in allowed:
        raise ValueError("Unknown diagnostic artifact")
    pointer = root / "state/research/diagnostics/latest.json"
    if pointer.is_symlink():
        raise ValueError("Diagnostic pointer must not be a symlink")
    reference, _ = _read(pointer)
    directory = root / reference["path"]
    if directory.is_symlink():
        raise ValueError("Diagnostic directory must not be a symlink")
    directory = directory.resolve()
    if not directory.is_relative_to((root / "state/research/diagnostics").resolve()):
        raise ValueError("Diagnostic path outside research directory")
    manifest_path = directory / "artifact_manifest.json"
    if manifest_path.is_symlink() or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("Diagnostic manifest hash mismatch")
    manifest, _ = _read(manifest_path)
    path = directory / "primary" / name
    if (path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(directory)
            or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("primary/" + name)):
        raise ValueError("Diagnostic file hash/path mismatch")
    return path


def load_numeric_diagnostic(root: Path, cycle_key: str | None) -> dict:
    if cycle_key not in {None, "2026Q2"} or not (root / "state/research/diagnostics/latest.json").is_file():
        return {"available": False}
    try:
        path = diagnostic_file(root, "diagnostic.json")
        result, _ = _read(path)
        if result.get("status") != "NUMERICAL_DIAGNOSTIC_NOT_HISTORICAL_SIGNAL" or result.get("executable") is not False:
            raise ValueError("Unexpected diagnostic execution classification")
        reference, _ = _read(root / "state/research/diagnostics/latest.json")
        manifest, _ = _read(root / reference["path"] / "artifact_manifest.json")
        downloads = sorted(name for name in DIAGNOSTIC_DOWNLOADS if "primary/" + name in manifest)
        return {"available": True, **result, "downloads": downloads}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"available": False, "error": f"Numerische Diagnose nicht lesbar: {exc}"}


def load_target_packages(root: Path, cycle_key: str | None) -> list[dict]:
    entries = list_packages(root / "state/research/target_portfolios")
    results = []
    for item in entries:
        if cycle_key and item.get("cycle_key") not in {None, cycle_key}:
            continue
        entry = {**item, "package_key": Path(item["path"]).name, "downloads": []}
        if item["integrity"] == "VERIFIED":
            try:
                report = inspect_package(Path(item["path"]))
                filenames = ["manifest.json", "zusammenfassung.md"]
                for value in report.manifest["strategies"].values():
                    filenames.append(value["file"] or value["failure_file"])
                entry["downloads"] = filenames
            except (TargetPortfolioError, OSError, ValueError) as exc:
                entry.update(integrity="INVALID", error=str(exc), handoff_allowed=False)
        results.append(entry)
    return results


def target_package_download(root: Path, package_key: str, name: str) -> bytes:
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", package_key)
            or "/" in name or "\\" in name):
        raise ValueError("Invalid target-package download path")
    base = (root / "state/research/target_portfolios").resolve()
    directory = base / package_key
    if directory.is_symlink() or not directory.resolve().is_relative_to(base):
        raise ValueError("Invalid target-package directory")
    report = inspect_package(directory)
    allowed = {"manifest.json", "zusammenfassung.md"}
    allowed.update(value["file"] or value["failure_file"]
                   for value in report.manifest["strategies"].values())
    if name not in allowed:
        raise ValueError("Unknown target-package download")
    data = (directory / name).read_bytes()
    if name == "manifest.json":
        if json.loads(data) != report.manifest:
            raise ValueError("Target-package manifest changed during download")
    else:
        if hashlib.sha256(data).hexdigest() != report.manifest["files"][name]["sha256"]:
            raise ValueError("Target-package download hash mismatch")
    return data
