"""Vendor-neutral, append-only PIT ingestion with validation and rollback evidence."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol, runtime_checkable

import duckdb

from .repositories import normalize_instrument
from .schema import assert_research_database
from .time_policy import (AVAILABLE_AT_POLICIES, SHANGHAI, PITTimeError,
                          available_at_for_announcement, parse_announcement)


class IngestionError(RuntimeError):
    pass


class PITQualityError(IngestionError):
    def __init__(self, report):
        self.report = report
        messages = "; ".join(f"{item['code']}: {item['message']}" for item in report.get("issues", []))
        super().__init__(messages or "PIT data quality validation failed")


@runtime_checkable
class FundamentalSourceAdapter(Protocol):
    source: str
    source_version: str

    def iter_records(self) -> Iterable[Mapping]: ...


@runtime_checkable
class IndustrySourceAdapter(Protocol):
    source: str
    source_version: str

    def iter_records(self) -> Iterable[Mapping]: ...


def _encode(value):
    if isinstance(value, Decimal):
        return _decimal_string(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _canonical(value) -> bytes:
    return json.dumps(_encode(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(value).hexdigest()


def _decimal_string(value) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid decimal value: {value!r}") from error
    if not number.is_finite():
        raise ValueError(f"financial values must be finite decimals: {value!r}")
    text = format(number.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _aware(value, field):
    if value is None or value == "":
        raise ValueError(f"{field} is required")
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an ISO datetime with timezone") from error
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return stamp.astimezone(timezone.utc)


def _date(value, field):
    try:
        return value if isinstance(value, date) and not isinstance(value, datetime) else date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an ISO date") from error


def _raw_bytes(record):
    raw = record.get("raw_content")
    if raw is None:
        raise ValueError("raw_content bytes are required to preserve the original source")
    if isinstance(raw, str):
        return raw.encode("utf-8")
    if not isinstance(raw, bytes):
        raise ValueError("raw_content must be bytes or text")
    return raw


def _safe_suffix(filename):
    suffix = Path(str(filename or "").replace("\\", "/")).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) else ".bin"


def _record_id(dataset, identity):
    return _digest(_canonical({"schema": "pit-record-id-v1", "dataset": dataset, **identity}))


def _financial_fields(value):
    if not isinstance(value, Mapping):
        raise ValueError("financial_fields must be an object")
    return {str(key): _decimal_string(item) if item is not None else None
            for key, item in sorted(value.items())}


def _optional_decimal(fields, *names):
    for name in names:
        if name in fields and fields[name] is not None:
            return Decimal(fields[name])
    return None


def _classify_issue(code, message, record_id=None):
    issue = {"severity": "error", "code": code, "message": str(message)}
    if record_id:
        issue["record_id"] = record_id
    return issue


def _fetch_mappings(connection, sql, parameters=None):
    cursor = connection.execute(sql, parameters or [])
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


class PITIngestionService:
    """Write only to a disposable/staging DB and preserve a report for every batch."""

    def __init__(self, database, *, trading_sessions=(), artifacts_root=None,
                 known_classifications=(), calendar_version=None, transaction_hook=None):
        self.database = assert_research_database(database)
        if not self.database.is_file():
            raise IngestionError(f"PIT database does not exist: {self.database}")
        self.trading_sessions = tuple(trading_sessions)
        self.artifacts_root = Path(artifacts_root or self.database.parent / "pit-ingestion").expanduser().resolve()
        self.known_classifications = set(map(str, known_classifications))
        self.calendar_version = str(calendar_version or "unversioned-calendar")
        self.transaction_hook = transaction_hook

    def ingest_fundamentals(self, adapter: FundamentalSourceAdapter, *, dry_run=False, batch_id=None):
        return self._ingest("fundamentals", adapter, dry_run=dry_run, batch_id=batch_id)

    def ingest_industries(self, adapter: IndustrySourceAdapter, *, dry_run=False, batch_id=None):
        return self._ingest("industry", adapter, dry_run=dry_run, batch_id=batch_id)

    def _adapter_values(self, adapter):
        source = str(getattr(adapter, "source", "") or "")
        source_version = str(getattr(adapter, "source_version", "") or "")
        method = getattr(adapter, "iter_records", None)
        if not source or not source_version or not callable(method):
            raise IngestionError("source adapter must provide source, source_version, and iter_records()")
        rows = list(method())
        if not all(isinstance(row, Mapping) for row in rows):
            raise IngestionError("source adapter records must be objects")
        return source, source_version, rows

    def _ingest(self, dataset, adapter, *, dry_run, batch_id):
        source, source_version, raw_records = self._adapter_values(adapter)
        batch_id = str(batch_id or uuid.uuid4().hex)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", batch_id):
            raise IngestionError(f"invalid batch_id: {batch_id!r}")
        started = datetime.now(timezone.utc)
        issues = []
        normalized = []
        for index, raw in enumerate(raw_records):
            try:
                normalized.append(self._normalize_record(dataset, raw, source, source_version))
            except Exception as error:
                code = getattr(error, "quality_code", None) or self._issue_code(error, dataset)
                issues.append(_classify_issue(code, f"record[{index}]: {error}"))
        duplicate_count, new_rows = self._collapse_and_compare(dataset, normalized, issues)
        if dataset == "industry":
            self._check_industry_overlaps(normalized, issues)
        raw_entries, normalized_path, raw_manifest_hash, normalized_hash = self._persist_artifacts(
            dataset, batch_id, raw_records, normalized)
        report = {"schema": "quant-project-pit-quality-report-v1", "batch_id": batch_id,
                  "dataset": dataset, "source": source, "source_version": source_version,
                  "checked_at": started.isoformat(), "record_count": len(raw_records),
                  "accepted_count": len(normalized), "new_count": len(new_rows),
                  "duplicate_count": duplicate_count, "issues": issues,
                  "status": "failed" if issues else ("dry_run" if dry_run else "ready")}
        batch_artifacts = self.artifacts_root / "manifests" / "batches" / batch_id
        batch_artifacts.mkdir(parents=True, exist_ok=True)
        (batch_artifacts / "quality-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if issues:
            if not dry_run:
                self._record_failure(batch_id, dataset, source, source_version, started,
                                     len(raw_records), raw_manifest_hash, normalized_hash, report)
            raise PITQualityError(report)
        result = {**report, "raw_manifest_sha256": raw_manifest_hash,
                  "normalized_sha256": normalized_hash, "normalized_path": str(normalized_path),
                  "raw_artifacts": raw_entries}
        if dry_run:
            return result
        return self._commit(dataset, batch_id, source, source_version, started,
                            normalized, new_rows, duplicate_count, raw_manifest_hash,
                            normalized_hash, result)

    def _normalize_record(self, dataset, source_record, source, source_version):
        raw = dict(source_record)
        instrument = normalize_instrument(raw.get("instrument"))
        raw_bytes = _raw_bytes(raw)
        raw_sha = _digest(raw_bytes)
        if raw.get("raw_content_sha256") and raw["raw_content_sha256"] != raw_sha:
            raise ValueError("raw hash does not match source bytes")
        if not isinstance(raw.get("raw_payload"), Mapping):
            raise ValueError("raw_payload must preserve the parsed supplier record")
        raw_payload = _encode(raw["raw_payload"])
        try:
            collected_at = _aware(raw.get("collected_at"), "collected_at")
        except ValueError as error:
            raise ValueError(str(error)) from error
        source_record_id = str(raw.get("source_record_id") or "")
        if not source_record_id:
            raise ValueError("source_record_id is required")
        source_priority = int(raw.get("source_priority", 0))
        if source_priority < 0:
            raise ValueError("source_priority must be non-negative")

        if dataset == "fundamentals":
            announcement_precision = str(raw.get("announcement_precision") or "")
            try:
                announcement = parse_announcement(raw.get("announcement_at"), announcement_precision)
            except PITTimeError as error:
                raise ValueError(str(error)) from error
            report_period = _date(raw.get("report_period"), "report_period")
            if announcement.date() < report_period:
                raise ValueError("announcement_at is earlier than report_period")
            policy = str(raw.get("availability_policy") or "conservative_next_session_v1")
            try:
                available = available_at_for_announcement(
                    raw.get("announcement_at"), precision=announcement_precision,
                    trading_sessions=self.trading_sessions, policy=policy)
            except PITTimeError as error:
                raise ValueError(str(error)) from error
            supplied_available = raw.get("available_at")
            if supplied_available is not None and _aware(supplied_available, "available_at") != available.astimezone(timezone.utc):
                raise ValueError("available_at does not match the versioned trading-calendar policy")
            revision_id = str(raw.get("revision_id") or "")
            sequence = int(raw.get("revision_sequence", 0))
            if not revision_id or sequence < 1:
                raise ValueError("revision_id and positive revision_sequence are required")
            fields = _financial_fields(raw.get("financial_fields", {}))
            identity = {"instrument": instrument, "report_period": report_period.isoformat(),
                        "source": source, "revision_id": revision_id,
                        "source_record_id": source_record_id, "source_version": source_version}
            business = {"instrument": instrument, "report_period": report_period.isoformat(),
                        "announcement_at": announcement.astimezone(timezone.utc).isoformat(),
                        "announcement_precision": announcement_precision,
                        "available_at": available.astimezone(timezone.utc).isoformat(),
                        "availability_policy": policy, "revision_id": revision_id,
                        "revision_sequence": sequence, "supersedes_record_id": raw.get("supersedes_record_id"),
                        "source": source, "source_record_id": source_record_id,
                        "source_version": source_version, "source_priority": source_priority,
                        "financial_fields": fields, "raw_payload": raw_payload}
            record_id = _record_id(dataset, identity)
            record = {**business, "record_id": record_id, "collected_at": collected_at,
                      "raw_content_sha256": raw_sha,
                      "normalized_content_sha256": _digest(_canonical(business)),
                      "raw_content": raw_bytes, "raw_filename": raw.get("raw_filename"),
                      "_normalized_business": business}
            if raw.get("record_id") and raw["record_id"] != record_id:
                raise ValueError("record_id is not the deterministic PIT record id")
            if raw.get("normalized_content_sha256") and raw["normalized_content_sha256"] != record["normalized_content_sha256"]:
                raise ValueError("normalized hash does not match normalized financial record")
            record["net_profit_yi"] = _optional_decimal(fields, "npYi", "net_profit_yi")
            record["debt_pct"] = _optional_decimal(fields, "debt", "debt_pct")
            record["deducted_profit_yoy_pct"] = _optional_decimal(fields, "dedYoy", "deducted_profit_yoy_pct")
            record["roe_pct"] = _optional_decimal(fields, "roe", "roe_pct")
            record["supersedes_record_id"] = raw.get("supersedes_record_id")
            record["report_period"] = report_period
            record["announcement_at"] = announcement.astimezone(timezone.utc)
            record["available_at"] = available.astimezone(timezone.utc)
        else:
            classification = str(raw.get("classification") or "")
            if not classification:
                raise ValueError("classification is required")
            if classification not in self.known_classifications:
                error = ValueError(f"unknown industry classification system: {classification}")
                error.quality_code = "unknown_industry_classification"
                raise error
            effective_from = _aware(raw.get("effective_from"), "effective_from")
            effective_to = (_aware(raw["effective_to"], "effective_to")
                            if raw.get("effective_to") is not None else None)
            if effective_to is not None and effective_to <= effective_from:
                raise ValueError("effective_to must be later than effective_from")
            available = _aware(raw.get("available_at"), "available_at")
            policy = str(raw.get("availability_policy") or "source_available_at_v1")
            version_id = str(raw.get("version_id") or "")
            sequence = int(raw.get("revision_sequence", 0))
            industry_code, industry_name = str(raw.get("industry_code") or ""), str(raw.get("industry_name") or "")
            if not version_id or sequence < 1 or not industry_code or not industry_name:
                raise ValueError("version_id, positive revision_sequence, industry_code, and industry_name are required")
            identity = {"instrument": instrument, "classification": classification,
                        "source": source, "version_id": version_id,
                        "source_record_id": source_record_id, "source_version": source_version}
            business = {**identity, "industry_code": industry_code, "industry_name": industry_name,
                        "effective_from": effective_from.isoformat(),
                        "effective_to": effective_to.isoformat() if effective_to else None,
                        "available_at": available.isoformat(), "availability_policy": policy,
                        "revision_sequence": sequence, "supersedes_record_id": raw.get("supersedes_record_id"),
                        "source_priority": source_priority, "raw_payload": raw_payload}
            record_id = _record_id(dataset, identity)
            record = {**business, "record_id": record_id, "collected_at": collected_at,
                      "raw_content_sha256": raw_sha,
                      "normalized_content_sha256": _digest(_canonical(business)),
                      "raw_content": raw_bytes, "raw_filename": raw.get("raw_filename"),
                      "_normalized_business": business,
                      "effective_from": effective_from, "effective_to": effective_to,
                      "available_at": available}
            if raw.get("record_id") and raw["record_id"] != record_id:
                raise ValueError("record_id is not the deterministic PIT record id")
            if raw.get("normalized_content_sha256") and raw["normalized_content_sha256"] != record["normalized_content_sha256"]:
                raise ValueError("normalized hash does not match normalized industry record")
        return record

    @staticmethod
    def _issue_code(error, dataset):
        message = str(error).lower()
        if "announcement_at is required" in message or "announcement_at" in message and "required" in message:
            return "missing_announcement_time"
        if "effective_from is required" in message:
            return "missing_effective_time"
        if "timezone" in message or "offset" in message:
            return "timestamp_missing_timezone"
        if "raw hash" in message:
            return "raw_hash_mismatch"
        if "normalized hash" in message:
            return "normalized_hash_mismatch"
        if "announcement_at is earlier" in message:
            return "announcement_before_report_period"
        if "available_at" in message and "policy" in message:
            return "available_at_policy_mismatch"
        if "effective_to" in message:
            return "invalid_effective_interval"
        if "security code" in message:
            return "unparseable_security_code"
        if "classification" in message:
            return "unknown_industry_classification" if dataset == "industry" else "invalid_classification"
        if "source_record_id" in message and "hash" in message:
            return "source_record_conflict"
        return "schema_validation_error"

    def _collapse_and_compare(self, dataset, normalized, issues):
        table = "fundamentals_pit" if dataset == "fundamentals" else "industries_pit"
        identity_fields = (("source", "source_record_id", "source_version"),
                           ("instrument", "report_period", "source", "revision_id")) if dataset == "fundamentals" else (
                           ("source", "source_record_id", "source_version"),
                           ("instrument", "classification", "source", "version_id"))
        unique = {}
        collapsed = {}
        for record in normalized:
            old = collapsed.get(record["record_id"])
            if old:
                if old["normalized_content_sha256"] != record["normalized_content_sha256"]:
                    issues.append(_classify_issue("source_record_conflict", "same source record id has different content",
                                                   record["record_id"]))
                continue
            collapsed[record["record_id"]] = record
            for fields in identity_fields:
                key = tuple(record[field] for field in fields)
                if key in unique and unique[key]["normalized_content_sha256"] != record["normalized_content_sha256"]:
                    issues.append(_classify_issue("revision_content_conflict", f"unique revision key conflicts: {key}",
                                                   record["record_id"]))
                unique[key] = record
        connection = duckdb.connect(str(self.database), read_only=True)
        try:
            existing = _fetch_mappings(connection, f"SELECT * FROM {table}")
        finally:
            connection.close()
        existing_by_id = {row["record_id"]: row for row in existing}
        all_ids = set(existing_by_id) | set(collapsed)
        for record in collapsed.values():
            supersedes = record.get("supersedes_record_id")
            if not supersedes:
                continue
            if supersedes == record["record_id"] or supersedes not in all_ids:
                issues.append(_classify_issue("invalid_supersedes_reference",
                    f"supersedes_record_id does not identify an existing PIT record: {supersedes}",
                    record["record_id"]))
                continue
            parent = collapsed.get(supersedes) or existing_by_id.get(supersedes)
            if dataset == "fundamentals":
                same_group = (record["instrument"], record["report_period"]) == (
                    parent["instrument"], parent["report_period"])
            else:
                same_group = (record["instrument"], record["classification"]) == (
                    parent["instrument"], parent["classification"])
            if not same_group:
                issues.append(_classify_issue("invalid_supersedes_reference",
                    "supersedes_record_id must refer to the same security and dataset key",
                    record["record_id"]))
        new_rows = []
        duplicate_count = 0
        for record in collapsed.values():
            old = existing_by_id.get(record["record_id"])
            if old:
                if old["normalized_content_sha256"] != record["normalized_content_sha256"] \
                        or old["raw_content_sha256"] != record["raw_content_sha256"]:
                    issues.append(_classify_issue("source_record_conflict", "existing source record has conflicting content",
                                                   record["record_id"]))
                else:
                    duplicate_count += 1
                continue
            for existing_row in existing:
                for fields in identity_fields:
                    if all(record[field] == existing_row[field] for field in fields) \
                            and record["normalized_content_sha256"] != existing_row["normalized_content_sha256"]:
                        code = "revision_content_conflict" if dataset == "fundamentals" else "industry_record_conflict"
                        issues.append(_classify_issue(code, f"existing unique source identity has conflicting content: {fields}",
                                                       record["record_id"]))
            new_rows.append(record)
        return duplicate_count, new_rows

    def _check_industry_overlaps(self, records, issues):
        all_rows = list(records)
        connection = duckdb.connect(str(self.database), read_only=True)
        try:
            all_rows.extend(_fetch_mappings(connection, "SELECT * FROM industries_pit"))
        finally:
            connection.close()
        grouped = {}
        for row in all_rows:
            key = (str(row["instrument"]), str(row["classification"]))
            grouped.setdefault(key, []).append(row)
        seen = set()
        for key, rows in grouped.items():
            for index, left in enumerate(rows):
                left_start, left_end = left["effective_from"], left.get("effective_to")
                for right in rows[index + 1:]:
                    if left.get("record_id") == right.get("record_id"):
                        continue
                    right_start, right_end = right["effective_from"], right.get("effective_to")
                    overlaps = (left_end is None or right_start < left_end) and (right_end is None or left_start < right_end)
                    if not overlaps or left["industry_code"] == right["industry_code"]:
                        continue
                    pair = tuple(sorted((str(left.get("record_id", "")), str(right.get("record_id", "")))))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    supersedes = {left.get("supersedes_record_id"), right.get("supersedes_record_id")}
                    if left.get("record_id") in supersedes or right.get("record_id") in supersedes:
                        continue
                    if int(left.get("source_priority") or 0) != int(right.get("source_priority") or 0):
                        continue
                    issues.append(_classify_issue(
                        "industry_valid_interval_overlap",
                        f"conflicting valid intervals overlap for {key[0]} / {key[1]}",
                        str(right.get("record_id", "")) or None,
                    ))

    def _persist_artifacts(self, dataset, batch_id, raw_records, normalized):
        raw_dir = self.artifacts_root / "raw" / "original" / batch_id
        normalized_dir = self.artifacts_root / "normalized" / dataset
        manifest_dir = self.artifacts_root / "manifests" / "batches" / batch_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        raw_entries, raw_by_key = [], {}
        for source_record in raw_records:
            try:
                original_bytes = _raw_bytes(source_record)
            except (TypeError, ValueError):
                continue
            key = _digest(original_bytes)
            if key in raw_by_key:
                continue
            suffix = _safe_suffix(source_record.get("raw_filename"))
            filename = f"{key}{suffix}"
            target = raw_dir / filename
            if not target.exists():
                target.write_bytes(original_bytes)
            raw_by_key[key] = filename
            raw_entries.append({"path": f"raw/original/{batch_id}/{filename}",
                                "sha256": key, "size_bytes": target.stat().st_size})
        raw_body = {"schema": "quant-project-pit-raw-manifest-v1", "batch_id": batch_id,
                    "files": sorted(raw_entries, key=lambda item: item["path"])}
        raw_hash = _digest(_canonical(raw_body))
        raw_manifest = raw_body | {"manifest_sha256": raw_hash}
        (manifest_dir / "raw-manifest.json").write_text(
            json.dumps(raw_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        normalized_rows = []
        for record in normalized:
            normalized_rows.append({"dataset": dataset, "record_id": record["record_id"],
                                    "raw_content_sha256": record["raw_content_sha256"],
                                    "normalized_content_sha256": record["normalized_content_sha256"],
                                    "normalized_content": record["_normalized_business"]})
        normalized_bytes = b"".join(_canonical(row) + b"\n" for row in normalized_rows)
        normalized_path = normalized_dir / f"{batch_id}.jsonl"
        normalized_path.write_bytes(normalized_bytes)
        normalized_hash = _digest(normalized_bytes)
        normalized_body = {"schema": "quant-project-pit-normalized-manifest-v1", "batch_id": batch_id,
                           "dataset": dataset, "path": f"normalized/{dataset}/{batch_id}.jsonl",
                           "sha256": normalized_hash, "record_count": len(normalized_rows)}
        normalized_body["manifest_sha256"] = _digest(_canonical({key: value for key, value in normalized_body.items()
                                                                 if key != "manifest_sha256"}))
        (manifest_dir / "normalized-manifest.json").write_text(
            json.dumps(normalized_body, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return raw_entries, normalized_path, raw_hash, normalized_hash

    def _record_failure(self, batch_id, dataset, source, source_version, started,
                        record_count, raw_hash, normalized_hash, report):
        connection = duckdb.connect(str(self.database))
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """INSERT INTO ingestion_batches(batch_id,dataset,source,source_version,started_at,completed_at,
                   record_count,raw_manifest_sha256,normalized_sha256,status,details_json)
                   VALUES (?,?,?,?,?,current_timestamp,?,?,?, ?,?::JSON) ON CONFLICT(batch_id) DO NOTHING""",
                [batch_id, dataset, source, source_version, started, record_count, raw_hash, normalized_hash,
                 "failed/rolled_back", json.dumps(report, ensure_ascii=False, sort_keys=True)],
            )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        finally:
            connection.close()

    def _commit(self, dataset, batch_id, source, source_version, started,
                normalized, new_rows, duplicate_count, raw_hash, normalized_hash, report):
        table = "fundamentals_pit" if dataset == "fundamentals" else "industries_pit"
        connection = duckdb.connect(str(self.database))
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """INSERT INTO ingestion_batches(batch_id,dataset,source,source_version,started_at,completed_at,
                   record_count,raw_manifest_sha256,normalized_sha256,status,details_json)
                   VALUES (?,?,?,?,?,NULL,?,?,?,'running',?::JSON)""",
                [batch_id, dataset, source, source_version, started, len(normalized), raw_hash, normalized_hash,
                 json.dumps({"accepted_count": len(normalized), "inserted_count": len(new_rows),
                             "duplicate_count": duplicate_count}, ensure_ascii=False, sort_keys=True)],
            )
            if dataset == "fundamentals":
                sql = """INSERT INTO fundamentals_pit(record_id,instrument,report_period,announcement_at,
                    announcement_precision,available_at,availability_policy,revision_id,revision_sequence,
                    supersedes_record_id,source,source_record_id,source_version,source_priority,collected_at,
                    ingestion_batch_id,raw_content_sha256,normalized_content_sha256,raw_payload,financial_fields,
                    net_profit_yi,debt_pct,deducted_profit_yoy_pct,roe_pct)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?::JSON,?::JSON,?,?,?,?)"""
                for row in new_rows:
                    connection.execute(sql, [row["record_id"], row["instrument"], row["report_period"],
                        row["announcement_at"], row["announcement_precision"], row["available_at"],
                        row["availability_policy"], row["revision_id"], row["revision_sequence"],
                        row.get("supersedes_record_id"), row["source"], row["source_record_id"],
                        row["source_version"], row["source_priority"], row["collected_at"], batch_id,
                        row["raw_content_sha256"], row["normalized_content_sha256"],
                        json.dumps(row["raw_payload"], ensure_ascii=False, sort_keys=True),
                        json.dumps(row["financial_fields"], ensure_ascii=False, sort_keys=True),
                        row["net_profit_yi"], row["debt_pct"], row["deducted_profit_yoy_pct"], row["roe_pct"]])
            else:
                sql = """INSERT INTO industries_pit(record_id,instrument,classification,industry_code,industry_name,
                    effective_from,effective_to,available_at,availability_policy,version_id,revision_sequence,
                    supersedes_record_id,source,source_record_id,source_version,source_priority,collected_at,
                    ingestion_batch_id,raw_content_sha256,normalized_content_sha256,raw_payload)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?::JSON)"""
                for row in new_rows:
                    connection.execute(sql, [row["record_id"], row["instrument"], row["classification"],
                        row["industry_code"], row["industry_name"], row["effective_from"], row["effective_to"],
                        row["available_at"], row["availability_policy"], row["version_id"],
                        row["revision_sequence"], row.get("supersedes_record_id"), row["source"],
                        row["source_record_id"], row["source_version"], row["source_priority"],
                        row["collected_at"], batch_id, row["raw_content_sha256"],
                        row["normalized_content_sha256"],
                        json.dumps(row["raw_payload"], ensure_ascii=False, sort_keys=True)])
            if self.transaction_hook:
                self.transaction_hook(connection, report)
            final_details = {"accepted_count": len(normalized), "inserted_count": len(new_rows),
                             "duplicate_count": duplicate_count}
            connection.execute(
                "UPDATE ingestion_batches SET completed_at=current_timestamp, record_count=?, status='complete', details_json=?::JSON WHERE batch_id=?",
                [len(normalized), json.dumps(final_details, sort_keys=True), batch_id],
            )
            connection.execute("COMMIT")
        except Exception as error:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
            connection.close()
            failed_report = report | {"status": "failed/rolled_back",
                                      "failure": f"{type(error).__name__}: {error}"}
            report_path = self.artifacts_root / "manifests" / "batches" / batch_id / "quality-report.json"
            report_path.write_text(json.dumps(failed_report, ensure_ascii=False, indent=2,
                                              sort_keys=True) + "\n", encoding="utf-8")
            self._record_failure(batch_id, dataset, source, source_version, started,
                                 len(normalized), raw_hash, normalized_hash,
                                 failed_report)
            raise IngestionError(f"ingestion batch {batch_id} rolled back: {error}") from error
        else:
            connection.close()
        return report | {"status": "complete", "inserted_count": len(new_rows),
                         "duplicate_count": duplicate_count,
                         "raw_manifest_sha256": raw_hash, "normalized_sha256": normalized_hash}
