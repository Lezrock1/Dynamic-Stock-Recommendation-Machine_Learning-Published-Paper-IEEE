"""Inline-XBRL fallback: read the facts of one original SEC filing document.

SEC companyfacts is an aggregation that omits dimensioned facts and company-specific extension
concepts. This module reads the primary iXBRL document of a single accepted filing and returns every
numeric fact with its tag, context (period + explicit dimensions), unit, accession, source URL and the
document hash. It never merges share classes or revenue definitions: a caller must select a fact by
tag, period and an explicit dimension rule, and every selected fact keeps its full provenance.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from . import sec_edgar as se
except ImportError:  # pragma: no cover - script execution
    import sec_edgar as se  # type: ignore

FILING_CACHE = Path(__file__).resolve().parent.parent / "cache" / "filings"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{document}"
_MIN_INTERVAL_SECONDS = 0.2
_last_request = 0.0


class IxbrlError(RuntimeError):
    pass


def document_url(cik: int, accession: str, document: str) -> str:
    return ARCHIVE_URL.format(cik=int(cik), nodash=accession.replace("-", ""), document=document)


def fetch_document(cik: int, accession: str, document: str) -> tuple[bytes, dict]:
    """Return the filing document and its provenance; cached on disk with a SHA-256 sidecar."""
    global _last_request
    folder = FILING_CACHE / str(int(cik)) / accession
    path = folder / document
    meta_path = folder / (document + ".meta.json")
    if path.exists() and meta_path.exists():
        data = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise IxbrlError(f"cached filing {path} does not match its recorded hash")
        return data, meta
    se.require_user_agent()
    wait = _MIN_INTERVAL_SECONDS - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    url = document_url(cik, accession, document)
    response = requests.get(url, headers=se.HEADERS, timeout=60)
    _last_request = time.time()
    if response.status_code != 200:
        raise IxbrlError(f"{url} returned HTTP {response.status_code}")
    data = response.content
    meta = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(), "accession": accession, "cik": int(cik)}
    folder.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    meta_path.write_text(json.dumps(meta, indent=1))
    return data, meta


def _number(text: str, fmt: str, scale: int, sign: str | None) -> float | None:
    fmt = (fmt or "").split(":")[-1].lower()
    cleaned = text.strip()
    if fmt in ("fixed-zero", "zerodash", "fixedzero") or cleaned in ("-", "—", "–"):
        value = 0.0
    else:
        if fmt in ("num-comma-decimal", "numcommadecimal"):
            cleaned = cleaned.replace(".", "").replace(" ", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "").replace(" ", "")
        cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
        if not cleaned or cleaned in ("-", "."):
            return None
        try:
            value = float(cleaned)
        except ValueError:
            return None
    value *= 10 ** scale
    return -value if sign == "-" else value


def parse_facts(data: bytes, provenance: Mapping[str, object]) -> list[dict]:
    """All numeric ix:nonFraction facts of one document with context, dimensions and unit."""
    soup = BeautifulSoup(data, "lxml")
    contexts: dict[str, dict] = {}
    for ctx in soup.find_all("xbrli:context"):
        period = ctx.find("xbrli:period")
        if period is None:
            continue
        instant = period.find("xbrli:instant")
        start, end = period.find("xbrli:startdate"), period.find("xbrli:enddate")
        dims = tuple(sorted((m.get("dimension", ""), m.get_text(strip=True))
                            for m in ctx.find_all("xbrldi:explicitmember")))
        typed = tuple(sorted(m.get("dimension", "") for m in ctx.find_all("xbrldi:typedmember")))
        contexts[ctx.get("id", "")] = {
            "start": start.get_text(strip=True) if start is not None else None,
            "end": (end.get_text(strip=True) if end is not None else
                    instant.get_text(strip=True) if instant is not None else None),
            "instant": instant is not None, "dimensions": dims, "typed_dimensions": typed,
        }
    units: dict[str, str] = {}
    for unit in soup.find_all("xbrli:unit"):
        divide = unit.find("xbrli:divide")
        if divide is not None:
            parts = [m.get_text(strip=True) for m in divide.find_all("xbrli:measure")]
            units[unit.get("id", "")] = "/".join(parts)
        else:
            measure = unit.find("xbrli:measure")
            units[unit.get("id", "")] = measure.get_text(strip=True) if measure is not None else ""
    facts = []
    for node in soup.find_all("ix:nonfraction"):
        context = contexts.get(node.get("contextref", ""))
        if context is None or node.get("xsi:nil") == "true":
            continue
        value = _number(node.get_text(), node.get("format", ""), int(node.get("scale", "0") or 0), node.get("sign"))
        if value is None:
            continue
        facts.append({
            "tag": node.get("name", ""), "value": value, "unit": units.get(node.get("unitref", ""), ""),
            "period_start": context["start"], "period_end": context["end"], "instant": context["instant"],
            "dimensions": context["dimensions"], "typed_dimensions": context["typed_dimensions"],
            "context_id": node.get("contextref", ""), "decimals": node.get("decimals"),
            "accession": provenance["accession"], "source_url": provenance["url"],
            "document_sha256": provenance["sha256"],
        })
    return facts


def filing_facts(cik: int, accession: str, document: str) -> list[dict]:
    data, meta = fetch_document(cik, accession, document)
    return parse_facts(data, meta)


def select(facts: Iterable[Mapping], tags: Iterable[str], *, period_end: str, days: tuple[int, int] | None = None,
           instant: bool = False, dimensions: tuple = (), unit_contains: str | None = None) -> list[dict]:
    """Facts with an exact tag/period/dimension match; no class or definition merging."""
    wanted = set(tags)
    out = []
    for fact in facts:
        if fact["tag"] not in wanted or fact["period_end"] != period_end or fact["instant"] != instant:
            continue
        if tuple(fact["dimensions"]) != tuple(sorted(dimensions)) or fact["typed_dimensions"]:
            continue
        if unit_contains and unit_contains not in fact["unit"]:
            continue
        if days is not None and not instant:
            length = (pd.Timestamp(fact["period_end"]) - pd.Timestamp(fact["period_start"])).days
            if not days[0] <= length <= days[1]:
                continue
        out.append(dict(fact))
    return out
