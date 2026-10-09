"""Read-only repositories for financial and industry PIT facts."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb

from .time_policy import normalize_as_of


class PITRepositoryError(RuntimeError):
    pass


class PITConflictError(PITRepositoryError):
    pass


def normalize_instrument(value) -> str:
    raw = str(value or "").strip()
    compact = raw.replace(".", "").replace("_", "")
    if re.fullmatch(r"(?i)(sh|sz|bj)\d{6}", compact):
        return compact.lower()
    if re.fullmatch(r"\d{6}", compact):
        if compact.startswith("6"):
            return "sh" + compact
        if compact.startswith(("0", "3")):
            return "sz" + compact
        if compact.startswith(("4", "8")):
            return "bj" + compact
    raise PITRepositoryError(f"security code cannot be parsed: {value!r}")


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _connection(database):
    if hasattr(database, "execute"):
        return database, False
    path = Path(database).expanduser().resolve()
    if not path.is_file():
        raise PITRepositoryError(f"PIT database does not exist: {path}")
    try:
        connection = duckdb.connect(str(path), read_only=True)
    except Exception as error:
        raise PITRepositoryError(f"cannot open PIT database read-only: {path}") from error
    return connection, True


class _PITRepository:
    table: str

    def __init__(self, database):
        self.database = database

    def _rows(self, sql, parameters):
        connection, owned = _connection(self.database)
        try:
            tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
            if self.table not in tables:
                raise PITRepositoryError(f"PIT table is unavailable: {self.table}")
            cursor = connection.execute(sql, parameters)
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except PITRepositoryError:
            raise
        except Exception as error:
            raise PITRepositoryError(f"PIT query failed for {self.table}: {error}") from error
        finally:
            if owned:
                connection.close()

    @staticmethod
    def _winner(rows, key, *, dataset):
        grouped = defaultdict(list)
        for row in rows:
            grouped[key(row)].append(row)
        winners = []
        for group_key, candidates in grouped.items():
            ids = {row["record_id"] for row in candidates}
            superseded = {row.get("supersedes_record_id") for row in candidates
                          if row.get("supersedes_record_id") in ids}
            candidates = [row for row in candidates if row["record_id"] not in superseded]
            if not candidates:
                raise PITConflictError(f"{dataset} revision chain has no current record for {group_key}")
            priority = max(int(row.get("source_priority") or 0) for row in candidates)
            candidates = [row for row in candidates if int(row.get("source_priority") or 0) == priority]
            sequence = max(int(row["revision_sequence"]) for row in candidates)
            candidates = [row for row in candidates if int(row["revision_sequence"]) == sequence]
            hashes = {row["normalized_content_sha256"] for row in candidates}
            if len(hashes) > 1:
                raise PITConflictError(
                    f"unresolved {dataset} conflict at equal source priority/revision sequence for {group_key}"
                )
            # Equal normalized payloads are the same business fact. Use source
            # identity only to keep the returned metadata stable, never to rank
            # conflicting business values.
            winners.append(sorted(candidates, key=lambda row: (row["source"], row["source_record_id"],
                                                               row["source_version"]))[0])
        return winners


class FundamentalPITRepository(_PITRepository):
    table = "fundamentals_pit"

    def get_fundamentals(self, instruments, fields, as_of, report_period=None):
        """Return the latest legally visible revision per instrument/report period."""
        codes = sorted({normalize_instrument(value) for value in instruments})
        if not codes:
            return []
        if not isinstance(fields, (list, tuple, set)) or not all(isinstance(x, str) and x for x in fields):
            raise PITRepositoryError("fields must be a sequence of non-empty names")
        moment = normalize_as_of(as_of)
        sql = """SELECT record_id,instrument,report_period,announcement_at,announcement_precision,
                    available_at,availability_policy,revision_id,revision_sequence,supersedes_record_id,
                    source,source_record_id,source_version,source_priority,collected_at,ingestion_batch_id,
                    raw_content_sha256,normalized_content_sha256,financial_fields,net_profit_yi,debt_pct,
                    deducted_profit_yoy_pct,roe_pct
                 FROM fundamentals_pit WHERE instrument=ANY(?) AND available_at<=?"""
        parameters = [codes, moment]
        if report_period is not None:
            try:
                report_period = date.fromisoformat(str(report_period)[:10])
            except ValueError as error:
                raise PITRepositoryError(f"invalid report_period: {report_period!r}") from error
            sql += " AND report_period=?"
            parameters.append(report_period)
        rows = self._rows(sql, parameters)
        rows = self._winner(rows, lambda row: (row["instrument"], row["report_period"]),
                            dataset="fundamentals")
        result = []
        for row in sorted(rows, key=lambda item: (item["instrument"], item["report_period"])):
            values = _json(row.pop("financial_fields")) or {}
            payload = {**row, **values}
            missing = [field for field in fields if field not in payload]
            if missing:
                raise PITRepositoryError(f"requested unknown financial fields: {', '.join(missing)}")
            result.append({key: payload.get(key) for key in (
                "record_id", "instrument", "report_period", "announcement_at", "available_at",
                "availability_policy", "revision_id", "revision_sequence", "source", "source_version",
                "ingestion_batch_id", *fields,
            )})
        return result


class IndustryPITRepository(_PITRepository):
    table = "industries_pit"

    def get_industry(self, instruments, classification, as_of):
        """Resolve industry using both valid time and knowledge time."""
        codes = sorted({normalize_instrument(value) for value in instruments})
        if not codes:
            return []
        if not classification:
            raise PITRepositoryError("classification is required")
        moment = normalize_as_of(as_of)
        rows = self._rows(
            """SELECT record_id,instrument,classification,industry_code,industry_name,effective_from,
                      effective_to,available_at,availability_policy,version_id,revision_sequence,
                      supersedes_record_id,source,source_record_id,source_version,source_priority,
                      collected_at,ingestion_batch_id,raw_content_sha256,normalized_content_sha256
               FROM industries_pit WHERE instrument=ANY(?) AND classification=? AND available_at<=?
                 AND effective_from<=? AND (effective_to IS NULL OR effective_to>?)""",
            [codes, str(classification), moment, moment, moment],
        )
        rows = self._winner(rows, lambda row: (row["instrument"], row["classification"]), dataset="industry")
        return sorted(rows, key=lambda row: row["instrument"])
