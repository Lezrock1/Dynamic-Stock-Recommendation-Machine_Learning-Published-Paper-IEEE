"""Check source bytes, original publication versions and actual retrieval separately."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from bs4 import BeautifulSoup

from .historical_evidence import filings
from .ixbrl import parse_facts
from .paper_fixed_lag import schedule
from .training_vintage import STRICT, feature_limits

OBSERVED = "OBSERVED_BEFORE_CUTOFF"
ORIGINAL = "ORIGINAL_PUBLICATION_RETRIEVED_LATER"
VERSION = "publication-provenance-v1"


def timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Evidence timestamp must be an explicit timezone-aware string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Evidence timestamp requires a timezone")
    return result.astimezone(timezone.utc)


def checked_bytes(root: Path, record: dict) -> bytes:
    name = record.get("path")
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Evidence path must be relative and contained")
    path = root / name
    components = [root.joinpath(*Path(name).parts[:i]) for i in range(1, len(Path(name).parts) + 1)]
    if (any(part.is_symlink() for part in components)
            or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("Evidence path must not escape its root or use symlinks")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record.get("sha256"):
        raise ValueError(f"Evidence hash mismatch: {name}")
    return data


def _https(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Evidence source must be a credential-free HTTPS URL")
    return parsed.hostname


def verify_publication_evidence(root: Path, doc: dict, cutoff: str, *,
                                now: datetime | None = None) -> dict:
    """Verify evidence, not an assertion of model/feature correctness.

    Late retrieval is allowed only for an original SEC archive version, bound to its accession
    in a separately hashed submissions response. Current Companyfacts is never that original.
    Each original record also re-parses its referenced numeric facts from the original bytes.
    """
    moment = now or datetime.now(timezone.utc)
    limit = timestamp(cutoff)
    if doc.get("format_version") != VERSION or not isinstance(doc.get("records"), list) or not doc["records"]:
        raise ValueError("Nonempty versioned publication evidence required")
    verified = {}
    for record in doc["records"]:
        identity = record.get("evidence_id")
        if not isinstance(identity, str) or not identity or identity in verified:
            raise ValueError("Publication evidence IDs must be nonempty and unique")
        data = checked_bytes(root, record)
        source = record.get("source_url", "")
        _https(source)
        retrieved = timestamp(record.get("retrieved_at_utc"))
        published = timestamp(record.get("published_at_utc"))
        if retrieved > moment or published > limit or published > retrieved:
            raise ValueError(f"Invalid publication/retrieval chronology: {identity}")
        kind = record.get("availability_class")
        capture = record.get("capture")
        if not isinstance(capture, dict):
            raise ValueError(f"Missing actual acquisition receipt: {identity}")
        receipt = json.loads(checked_bytes(root, capture))
        if "record_index" in capture:
            index = capture["record_index"]
            if type(index) is not int or not isinstance(receipt, list) or not 0 <= index < len(receipt):
                raise ValueError("Invalid acquisition receipt record selector")
            receipt = receipt[index]
        if not isinstance(receipt, dict):
            raise ValueError("Acquisition receipt must select one actual record")
        if (receipt.get("source_url") != source or receipt.get("sha256") != record["sha256"]
                or timestamp(receipt.get("acquired_at")) != retrieved
                or receipt.get("http_status") != 200):
            raise ValueError(f"Acquisition receipt does not bind these source bytes: {identity}")
        if kind == OBSERVED:
            if retrieved > limit:
                raise ValueError(f"Pre-cutoff observation actually retrieved later: {identity}")
        elif kind == ORIGINAL:
            accession, cik = record.get("accession"), record.get("cik")
            if (not isinstance(accession, str) or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
                    or type(cik) is not int or cik <= 0):
                raise ValueError(f"Original SEC identity/accession missing: {identity}")
            parsed = urlsplit(source)
            prefix = f"/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
            if (parsed.hostname != "www.sec.gov" or not parsed.path.startswith(prefix)
                    or not parsed.path[len(prefix):] or "/" in parsed.path[len(prefix):]
                    or parsed.query or parsed.fragment):
                raise ValueError(f"Original source URL does not bind accession/entity: {identity}")
            publication = record.get("publication")
            if not isinstance(publication, dict):
                raise ValueError(f"Original publication proof missing: {identity}")
            if _https(publication.get("source_url", "")) != "data.sec.gov":
                raise ValueError("Publication proof must be the original SEC submissions source")
            proof_url = urlsplit(publication["source_url"])
            if not re.fullmatch(rf"/submissions/CIK{cik:010d}(?:-submissions-\d+)?\.json", proof_url.path):
                raise ValueError("SEC publication proof URL belongs to a different entity")
            proof = json.loads(checked_bytes(root, publication))
            accepted = [row for row in filings(proof) if row["accessionNumber"] == accession]
            if len(accepted) != 1 or timestamp(accepted[0]["acceptanceDateTime"]) != published:
                raise ValueError(f"Publication time not present in original accession record: {identity}")
            facts = parse_facts(data, {"accession": accession, "url": source, "sha256": record["sha256"]})
            soup = BeautifulSoup(data, "lxml")
            entities = {}
            for context in soup.find_all("xbrli:context"):
                identifier = context.find("xbrli:identifier")
                text = identifier.get_text(strip=True) if identifier else ""
                entities[context.get("id")] = int(text) if text.isdigit() else None
            claims = record.get("facts")
            if not isinstance(claims, list) or not claims:
                raise ValueError(f"Original numeric version bindings missing: {identity}")
            for claim in claims:
                required = {"tag", "unit", "value", "period_start", "period_end", "context_id"}
                if not required.issubset(claim):
                    raise ValueError(f"Incomplete original fact binding: {identity}")
                matches = [fact for fact in facts if all(fact.get(key) == claim[key] for key in required)
                           and not fact["dimensions"] and not fact["typed_dimensions"]
                           and entities.get(fact["context_id"]) == cik]
                if len(matches) != 1:
                    raise ValueError(f"Exact original fact/version not found: {identity}:{claim.get('tag')}")
        else:
            raise ValueError(f"Unknown availability class: {kind}")
        verified[identity] = {
            "availability_class": kind, "published_at_utc": published.isoformat(),
            "retrieved_at_utc": retrieved.isoformat(), "sha256": record["sha256"],
            "source_url": source,
        }
    return {"format_version": VERSION, "information_cutoff_utc": limit.isoformat(),
            "verified_records": verified, "classes": sorted({r["availability_class"] for r in verified.values()}),
            "model_qualification": "NOT_ESTABLISHED_BY_PUBLICATION_CHECK_ALONE"}


def verify_signal_availability(root: Path, cutoff: str, *,
                               training_vintage_policy: str = STRICT) -> dict:
    """Require actual evidence coverage, not merely a nonempty list or provenance marker.

    Membership and covariance must currently have byte-identical pre-cutoff source captures.
    Late-retrieved SEC numeric originals may supply feature components, but cannot certify
    unrelated identity tables, transformed prices, label horizons or formula equivalence.
    """
    doc = json.loads((root / "publication_evidence.json").read_text())
    result = verify_publication_evidence(root, doc, cutoff)
    records = {r["evidence_id"]: r for r in doc["records"]}
    verified = result["verified_records"]
    bindings = doc.get("artifact_bindings", {})
    for name in ("membership.csv", "covariance_levels.csv"):
        evidence_id = bindings.get(name)
        record = records.get(evidence_id)
        if (record is None or record["availability_class"] != OBSERVED
                or hashlib.sha256((root / name).read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"No byte-bound pre-cutoff source for {name}; original fact proof cannot certify it")
    lineage = json.loads((root / "lineage.json").read_text())
    if not isinstance(lineage, dict) or not lineage:
        raise ValueError("No feature lineage to bind publication evidence")
    component_count = 0
    panel_sha256 = hashlib.sha256((root / "panel.csv").read_bytes()).hexdigest()
    target = None
    if training_vintage_policy != STRICT:
        target = str(pd.Period(pd.Timestamp(cutoff).tz_convert("UTC").tz_localize(None) -
                               pd.DateOffset(months=2), "Q"))
        if timestamp(schedule(target)["information_cutoff"]) != timestamp(cutoff):
            raise ValueError("Publication cutoff does not match a fixed-lag target cycle")
    for row_key, features in lineage.items():
        limits = feature_limits(
            pd.Series([row_key.split(":", 1)[0]]), target, training_vintage_policy)
        own_cutoff = limits.iloc[0].to_pydatetime()
        for feature, parts in features.items():
            for part in parts:
                evidence_id = part.get("evidence_id")
                if evidence_id not in records:
                    raise ValueError(f"Unbound source component: {row_key}:{feature}")
                record = records[evidence_id]
                if timestamp(verified[evidence_id]["published_at_utc"]) > own_cutoff:
                    raise ValueError(f"Component not public at its own feature cutoff: {row_key}:{feature}")
                if record["availability_class"] == ORIGINAL:
                    fields = ("tag", "unit", "value", "period_start", "period_end", "context_id")
                    if (part.get("accession") != record["accession"]
                            or part.get("source_url") != record["source_url"]
                            or any(key not in part for key in fields)
                            or not any(all(part[key] == fact[key] for key in fields)
                                       for fact in record["facts"])):
                        raise ValueError(f"Lineage does not bind exact parsed original version: {row_key}:{feature}")
                elif timestamp(record["retrieved_at_utc"]) > own_cutoff:
                    raise ValueError(f"Snapshot was not observed at its own feature cutoff: {row_key}:{feature}")
                elif panel_sha256 != record["sha256"]:
                    raise ValueError(f"Observed component does not bind the source panel bytes: {row_key}:{feature}")
                component_count += 1
    result["bound_feature_components"] = component_count
    result["availability_coverage"] = "SOURCE_BYTES_AND_LINEAGE_VERIFIED"
    return result
