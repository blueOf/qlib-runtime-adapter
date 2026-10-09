"""DuckDB PIT schemas. Schema installation is intentionally blocked on the active DB."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

PIT_SCHEMA_VERSION = "quant-project-pit-v1"


class PITSchemaError(RuntimeError):
    pass


def _protected_databases() -> set[Path]:
    from ..paths import FORMAL_DATA_ROOT

    protected = {(FORMAL_DATA_ROOT / "market.duckdb").resolve()}
    pointer = FORMAL_DATA_ROOT / "active-release.json"
    if pointer.is_file():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8-sig"))
            if payload.get("database"):
                protected.add(Path(payload["database"]).expanduser().resolve())
        except (OSError, ValueError):
            raise PITSchemaError(f"cannot verify active database pointer: {pointer}")
    return protected


def assert_research_database(database) -> Path:
    path = Path(database).expanduser().resolve()
    if path in _protected_databases():
        raise PITSchemaError(f"refusing PIT schema/data writes to protected active database: {path}")
    return path


def initialize_pit_schema(database) -> Path:
    """Install PIT tables in a disposable, staging, or Research Release database."""
    path = assert_research_database(database)
    if not path.is_file():
        raise PITSchemaError(f"PIT schema requires an existing fixture/staging database: {path}")
    connection = duckdb.connect(str(path))
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS pit_schema_metadata (
                schema_version VARCHAR PRIMARY KEY,
                installed_at TIMESTAMPTZ NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals_pit (
                record_id VARCHAR PRIMARY KEY,
                instrument VARCHAR NOT NULL,
                report_period DATE NOT NULL,
                announcement_at TIMESTAMPTZ NOT NULL,
                announcement_precision VARCHAR NOT NULL CHECK (announcement_precision IN ('timestamp','date')),
                available_at TIMESTAMPTZ NOT NULL,
                availability_policy VARCHAR NOT NULL,
                revision_id VARCHAR NOT NULL,
                revision_sequence INTEGER NOT NULL CHECK (revision_sequence >= 1),
                supersedes_record_id VARCHAR,
                source VARCHAR NOT NULL,
                source_record_id VARCHAR NOT NULL,
                source_version VARCHAR NOT NULL,
                source_priority INTEGER NOT NULL DEFAULT 0,
                collected_at TIMESTAMPTZ NOT NULL,
                ingestion_batch_id VARCHAR NOT NULL,
                raw_content_sha256 VARCHAR NOT NULL,
                normalized_content_sha256 VARCHAR NOT NULL,
                raw_payload JSON NOT NULL,
                financial_fields JSON NOT NULL,
                net_profit_yi DECIMAL(38,12),
                debt_pct DECIMAL(38,12),
                deducted_profit_yoy_pct DECIMAL(38,12),
                roe_pct DECIMAL(38,12),
                UNIQUE (source, source_record_id, source_version),
                UNIQUE (instrument, report_period, source, revision_id)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS industries_pit (
                record_id VARCHAR PRIMARY KEY,
                instrument VARCHAR NOT NULL,
                classification VARCHAR NOT NULL,
                industry_code VARCHAR NOT NULL,
                industry_name VARCHAR NOT NULL,
                effective_from TIMESTAMPTZ NOT NULL,
                effective_to TIMESTAMPTZ,
                available_at TIMESTAMPTZ NOT NULL,
                availability_policy VARCHAR NOT NULL,
                version_id VARCHAR NOT NULL,
                revision_sequence INTEGER NOT NULL CHECK (revision_sequence >= 1),
                supersedes_record_id VARCHAR,
                source VARCHAR NOT NULL,
                source_record_id VARCHAR NOT NULL,
                source_version VARCHAR NOT NULL,
                source_priority INTEGER NOT NULL DEFAULT 0,
                collected_at TIMESTAMPTZ NOT NULL,
                ingestion_batch_id VARCHAR NOT NULL,
                raw_content_sha256 VARCHAR NOT NULL,
                normalized_content_sha256 VARCHAR NOT NULL,
                raw_payload JSON NOT NULL,
                UNIQUE (source, source_record_id, source_version),
                UNIQUE (instrument, classification, source, version_id)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_batches (
                batch_id VARCHAR PRIMARY KEY,
                dataset VARCHAR NOT NULL CHECK (dataset IN ('fundamentals','industry')),
                source VARCHAR NOT NULL,
                source_version VARCHAR NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                record_count BIGINT NOT NULL DEFAULT 0,
                raw_manifest_sha256 VARCHAR,
                normalized_sha256 VARCHAR,
                status VARCHAR NOT NULL,
                details_json JSON NOT NULL
            )
        """)
        connection.execute(
            "INSERT INTO pit_schema_metadata VALUES (?, current_timestamp) ON CONFLICT DO NOTHING",
            [PIT_SCHEMA_VERSION],
        )
    except Exception as error:
        raise PITSchemaError(f"failed to install PIT schema in {path}: {error}") from error
    finally:
        connection.close()
    return path
