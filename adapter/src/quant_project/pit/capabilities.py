"""Derive release capability from stored rows, verified manifests, and coverage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from decimal import Decimal, InvalidOperation
from datetime import timezone

import duckdb

from .ingestion import _canonical, _digest


def _timestamp_iso(value):
    return value.astimezone(timezone.utc).isoformat()


def _manifest(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or not value.get("manifest_sha256"):
        return None
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return value if _digest(_canonical(body)) == value["manifest_sha256"] else None


def _verify_batch_artifacts(root, batch, dataset, connection):
    batch_id = batch["batch_id"]
    directory = Path(root) / "manifests" / "batches" / batch_id
    raw_manifest = _manifest(directory / "raw-manifest.json")
    normalized_manifest = _manifest(directory / "normalized-manifest.json")
    if not raw_manifest or not normalized_manifest:
        return None
    if raw_manifest.get("manifest_sha256") != batch["raw_manifest_sha256"]:
        return None
    for artifact in raw_manifest.get("files", []):
        relative = Path(artifact.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts \
                or not relative.as_posix().startswith(f"raw/original/{batch_id}/"):
            return None
        path = Path(root) / relative
        if not path.is_file() or _sha256_file(path) != artifact.get("sha256"):
            return None
    relative = normalized_manifest.get("path")
    expected_relative = f"normalized/{dataset}/{batch_id}.jsonl"
    if relative != expected_relative:
        return None
    normalized_path = Path(root) / Path(relative or "")
    if not normalized_path.is_file() or _sha256_file(normalized_path) != batch["normalized_sha256"]:
        return None
    if normalized_manifest.get("sha256") != batch["normalized_sha256"]:
        return None
    try:
        lines = [json.loads(line) for line in normalized_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError):
        return None
    if len(lines) != int(batch["record_count"]):
        return None
    table = "fundamentals_pit" if dataset == "fundamentals" else "industries_pit"
    if dataset == "fundamentals":
        columns = ("record_id,instrument,report_period,announcement_at,announcement_precision,available_at,"
                   "availability_policy,revision_id,revision_sequence,supersedes_record_id,source,source_record_id,"
                   "source_version,source_priority,financial_fields,raw_payload,raw_content_sha256,"
                   "normalized_content_sha256,ingestion_batch_id,net_profit_yi,debt_pct,"
                   "deducted_profit_yoy_pct,roe_pct")
    else:
        columns = ("record_id,instrument,classification,industry_code,industry_name,effective_from,effective_to,"
                   "available_at,availability_policy,version_id,revision_sequence,supersedes_record_id,source,"
                   "source_record_id,source_version,source_priority,raw_payload,raw_content_sha256,"
                   "normalized_content_sha256,ingestion_batch_id")
    stored = {row[0]: row[1:] for row in connection.execute(f"SELECT {columns} FROM {table}").fetchall()}
    verified_ids = set()
    for item in lines:
        normalized_content = item.get("normalized_content")
        expected_hash = item.get("normalized_content_sha256")
        if not isinstance(normalized_content, dict) or _digest(_canonical(normalized_content)) != expected_hash:
            return None
        record_id = item.get("record_id")
        row = stored.get(record_id)
        if item.get("dataset") != dataset or row is None:
            return None
        if dataset == "fundamentals":
            (instrument, report_period, announcement_at, precision, available_at, policy, revision_id,
             sequence, supersedes, source, source_record_id, source_version, source_priority,
             financial_fields, raw_payload, raw_hash, normalized_hash, ingestion_batch_id,
             net_profit_yi, debt_pct, deducted_profit_yoy_pct, roe_pct) = row
            actual = {"instrument": instrument, "report_period": report_period.isoformat(),
                "announcement_at": _timestamp_iso(announcement_at), "announcement_precision": precision,
                "available_at": _timestamp_iso(available_at), "availability_policy": policy,
                "revision_id": revision_id, "revision_sequence": sequence,
                "supersedes_record_id": supersedes, "source": source, "source_record_id": source_record_id,
                "source_version": source_version, "source_priority": source_priority,
                "financial_fields": json.loads(financial_fields) if isinstance(financial_fields, str) else financial_fields,
                "raw_payload": json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload}
            fields = actual["financial_fields"]
            try:
                typed_values_match = all(
                    value == (Decimal(fields.get(primary, fields.get(alias)))
                              if fields.get(primary, fields.get(alias)) is not None else None)
                    for value, primary, alias in ((net_profit_yi, "npYi", "net_profit_yi"),
                        (debt_pct, "debt", "debt_pct"),
                        (deducted_profit_yoy_pct, "dedYoy", "deducted_profit_yoy_pct"),
                        (roe_pct, "roe", "roe_pct")))
            except (InvalidOperation, TypeError):
                typed_values_match = False
            if not typed_values_match:
                return None
        else:
            (instrument, classification, industry_code, industry_name, effective_from, effective_to,
             available_at, policy, version_id, sequence, supersedes, source, source_record_id,
             source_version, source_priority, raw_payload, raw_hash, normalized_hash, ingestion_batch_id) = row
            actual = {"instrument": instrument, "classification": classification, "industry_code": industry_code,
                "industry_name": industry_name, "effective_from": _timestamp_iso(effective_from),
                "effective_to": _timestamp_iso(effective_to) if effective_to is not None else None,
                "available_at": _timestamp_iso(available_at), "availability_policy": policy,
                "version_id": version_id, "revision_sequence": sequence,
                "supersedes_record_id": supersedes, "source": source, "source_record_id": source_record_id,
                "source_version": source_version, "source_priority": source_priority,
                "raw_payload": json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload}
        if (item.get("raw_content_sha256") != raw_hash or expected_hash != normalized_hash
                or batch["source"] != source
                or batch["source_version"] != source_version
                or _canonical(actual) != _canonical(normalized_content)):
            return None
        verified_ids.add(record_id)
    return verified_ids


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_capability(dataset, reason):
    return {"available": False, "point_in_time": False, "data_mode": None,
            "version": None, "source": [], "coverage": {}, "reason": reason}


def derive_release_capabilities(database, *, artifacts_root=None, source_descriptor=None):
    """Inspect a database copy; caller booleans and filesystem timestamps are ignored."""
    path = Path(database).expanduser().resolve()
    capabilities = {name: _empty_capability(name, "dataset unavailable")
                    for name in ("market_data", "fundamentals", "industry")}
    versions, sources, coverage = {}, {}, {}
    try:
        connection = duckdb.connect(str(path), read_only=True)
    except Exception:
        return {"capabilities": capabilities, "versions": versions,
                "sources": sources, "coverage": coverage, "known_biases": ["database_not_readable"]}
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if "daily_effective" in tables or "daily" in tables:
            table = "daily_effective" if "daily_effective" in tables else "daily"
            count, instruments, first, last = connection.execute(
                f"SELECT count(*), count(DISTINCT instrument), min(date), max(date) FROM {table}"
            ).fetchone()
            available = bool(count)
            descriptor = source_descriptor or {}
            market_version = (descriptor.get("data_revision") or descriptor.get("baseline_id")
                              or descriptor.get("provider_version"))
            capabilities["market_data"] = {
                "available": available, "point_in_time": False,
                "data_mode": "snapshot_compatible" if available else None,
                "version": market_version, "source": ["market database"],
                "coverage": {"rows": int(count), "instruments": int(instruments),
                             "from": first.isoformat() if first else None,
                             "to": last.isoformat() if last else None},
                "reason": "historical row revisions and availability manifests are not present",
            }
            if available:
                versions["market_data"] = market_version
                sources["market_data"] = ["market database"]
                coverage["market_data"] = capabilities["market_data"]["coverage"]

        reference_names, reference_payloads = {}, {}
        if "reference_documents" in tables:
            for name, payload in connection.execute(
                    "SELECT name,payload FROM reference_documents ORDER BY name").fetchall():
                reference_names[name] = payload
                try:
                    reference_payloads[name] = json.loads(payload) if payload is not None else None
                except ValueError:
                    reference_payloads[name] = None
        for dataset, document_name, instrument_key in (
                ("fundamentals", "fullmarket_f10.json", "code"),
                ("industry", "fullmarket_ind.json", None)):
            payload = reference_payloads.get(document_name)
            exists = document_name in reference_names and payload is not None
            if exists:
                if isinstance(payload, dict):
                    instrument_count = len(payload)
                elif isinstance(payload, list):
                    instrument_count = len({str(row.get(instrument_key)) for row in payload
                                            if isinstance(row, dict) and row.get(instrument_key)})
                else:
                    instrument_count = 0
                digest = _digest(str(reference_names[document_name]).encode("utf-8"))
                capabilities[dataset] = {
                    "available": True, "point_in_time": False, "data_mode": "latest_snapshot",
                    "version": digest, "source": ["reference_documents." + document_name],
                    "coverage": {"instruments": instrument_count},
                    "reason": "current snapshot has no trusted historical availability/effective time",
                }
                versions[dataset] = digest
                sources[dataset] = ["reference_documents." + document_name]
                coverage[dataset] = capabilities[dataset]["coverage"]

        biases = []
        for dataset, table in (("fundamentals", "fundamentals_pit"), ("industry", "industries_pit")):
            if table not in tables or "ingestion_batches" not in tables:
                if capabilities[dataset]["available"]:
                    biases.append(f"{dataset}_current_snapshot_lookahead" if dataset == "fundamentals"
                                  else "industry_snapshot_drift")
                continue
            row_count, instrument_count = connection.execute(
                f"SELECT count(*), count(DISTINCT instrument) FROM {table}"
            ).fetchone()
            batches = connection.execute(
                "SELECT batch_id,dataset,source,source_version,record_count,raw_manifest_sha256,normalized_sha256,status "
                "FROM ingestion_batches WHERE dataset=? ORDER BY batch_id", [dataset]
            ).fetchall()
            batch_rows = [dict(zip(("batch_id", "dataset", "source", "source_version", "record_count",
                                    "raw_manifest_sha256", "normalized_sha256", "status"), row))
                          for row in batches]
            verified_ids = set()
            verified_by_batch = {}
            verified = bool(row_count and batch_rows and artifacts_root)
            if verified:
                for batch_row in batch_rows:
                    if batch_row["status"] != "complete":
                        verified = False
                        break
                    batch_ids = _verify_batch_artifacts(artifacts_root, batch_row, dataset, connection)
                    if batch_ids is None:
                        verified = False
                        break
                    verified_ids.update(batch_ids)
                    verified_by_batch[batch_row["batch_id"]] = batch_ids
                if verified:
                    stored_rows = connection.execute(
                        f"SELECT record_id,ingestion_batch_id FROM {table}").fetchall()
                    stored_ids = {row[0] for row in stored_rows}
                    verified = (verified_ids == stored_ids and all(
                        record_id in verified_by_batch.get(origin_batch, set())
                        for record_id, origin_batch in stored_rows))
            if verified:
                version_set = sorted({row["source_version"] for row in batch_rows})
                source_set = sorted({row["source"] for row in batch_rows})
                if dataset == "fundamentals":
                    minimum, maximum = connection.execute(
                        "SELECT min(report_period),max(report_period) FROM fundamentals_pit").fetchone()
                    covered = {"rows": int(row_count), "instruments": int(instrument_count),
                               "report_period_from": minimum.isoformat() if minimum else None,
                               "report_period_to": maximum.isoformat() if maximum else None}
                else:
                    minimum, maximum = connection.execute(
                        "SELECT min(effective_from),max(effective_to) FROM industries_pit").fetchone()
                    classifications = [row[0] for row in connection.execute(
                        "SELECT DISTINCT classification FROM industries_pit ORDER BY classification").fetchall()]
                    covered = {"rows": int(row_count), "instruments": int(instrument_count),
                               "classifications": classifications,
                               "effective_from_min": minimum.isoformat() if minimum else None,
                               "effective_to_max": maximum.isoformat() if maximum else None}
                capabilities[dataset] = {"available": True, "point_in_time": True,
                    "data_mode": "point_in_time", "version": version_set,
                    "source": source_set, "coverage": covered, "reason": None}
                versions[dataset], sources[dataset], coverage[dataset] = version_set, source_set, covered
            elif capabilities[dataset]["available"]:
                biases.append(f"{dataset}_current_snapshot_lookahead" if dataset == "fundamentals"
                              else "industry_snapshot_drift")
                # PIT rows without their complete evidence are not promoted to
                # a capability. Preserve only non-PIT snapshot availability.
                capabilities[dataset]["point_in_time"] = False
            elif row_count:
                capabilities[dataset] = {"available": True, "point_in_time": False,
                    "data_mode": "snapshot_compatible", "version": None, "source": [],
                    "coverage": {"rows": int(row_count), "instruments": int(instrument_count)},
                    "reason": "PIT rows are missing verified ingestion manifests"}
        return {"capabilities": capabilities, "versions": versions, "sources": sources,
                "coverage": coverage, "known_biases": sorted(set(biases))}
    finally:
        connection.close()


def validate_release_dependencies(release, dependencies, *, data_mode, as_of=None, as_of_policy=None):
    """Require data presence for each dependency and PIT proof only in PIT mode."""
    from .contracts import normalize_dependencies, normalize_data_mode
    from .time_policy import normalize_as_of

    deps = normalize_dependencies(dependencies)
    mode = normalize_data_mode(data_mode)
    if mode == "point_in_time":
        normalize_as_of(as_of)
        if not as_of_policy:
            raise ValueError("point_in_time research requires an explicit as_of_policy")
    caps = release.get("capabilities") or {}
    errors = []
    for dataset, needed in deps.items():
        if not needed:
            continue
        capability = caps.get(dataset) or {}
        if not capability.get("available"):
            errors.append(f"{dataset} data are not available in Research Release")
        elif mode == "point_in_time" and not capability.get("point_in_time"):
            errors.append(f"{dataset} lacks verified point-in-time capability")
    if errors:
        raise ValueError("; ".join(errors))
    return {dataset: caps.get(dataset, {"available": False, "point_in_time": False})
            for dataset in deps}
