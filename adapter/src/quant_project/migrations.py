"""Explicit, checksum-verified v1-to-v2 migrations for staging copies only.

Nothing in this module activates a database or changes the live provider.  Its
only mutable target is a caller-supplied staging directory.  DB3 must provide a
separate, user-approved activation operation.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .audit import audit_database
from .market import connect, database_path


V1_LAYOUT = "market-database-v1"
V2_LAYOUT = "market-database-v2"
MIGRATION_ID = "0001_market_database_v2_core"


class MigrationError(RuntimeError):
    """A staging-only migration error; the source database is untouched."""


@dataclass(frozen=True)
class Migration:
    migration_id: str
    description: str
    checksum: str


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# The manifest content is deliberately stable and checked before a copy is made.
_CORE_MANIFEST = """0001_market_database_v2_core
schema_migrations ingestion_runs data_revisions migration_quarantine
daily_final daily_intraday_overlay daily_effective minute5_final
"""
MIGRATIONS = (Migration(MIGRATION_ID, "Split v1 final and intraday daily data in a staging copy",
                        _checksum(_CORE_MANIFEST)),)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_migration_registry(migrations=MIGRATIONS) -> list[dict]:
    ids = [migration.migration_id for migration in migrations]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise MigrationError("migration ids must be unique and lexicographically sortable")
    for migration in migrations:
        if len(migration.checksum) != 64 or any(char not in "0123456789abcdef" for char in migration.checksum):
            raise MigrationError(f"migration {migration.migration_id} has an invalid checksum")
    return [{"id": item.migration_id, "description": item.description, "checksum": item.checksum}
            for item in migrations]


def _preflight(source: Path, staging_root: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    staging_root.mkdir(parents=True, exist_ok=True)
    # Copying a DuckDB database needs one source-sized file; validation and the
    # provider build need headroom but never write beside the source.
    required = source.stat().st_size * 2 + 64 * 1024 * 1024
    available = shutil.disk_usage(staging_root).free
    return {"sourceBytes": source.stat().st_size, "requiredBytes": required,
            "availableBytes": available, "passed": available >= required}


def _tables(database) -> set[str]:
    return {row[0] for row in database.execute("SHOW TABLES").fetchall()}


def _columns(database, table: str) -> list[str]:
    return [row[1] for row in database.execute(f"PRAGMA table_info('{table}')").fetchall()]


def _require_v1_source(source: Path) -> None:
    with connect(source, read_only=True) as database:
        tables = _tables(database)
        required = {"daily", "minute5", "securities", "trading_calendar", "reference_documents", "store_metadata"}
        missing = sorted(required - tables)
        if missing:
            raise MigrationError("source is missing required v1 tables: " + ", ".join(missing))
        metadata = dict(database.execute("SELECT key,value FROM store_metadata").fetchall())
        layout = metadata.get("layout")
        if layout not in (None, V1_LAYOUT, V2_LAYOUT):
            raise MigrationError(f"unsupported source layout: {layout}")


def _create_v2_schema(database) -> None:
    database.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations(
          migration_id VARCHAR PRIMARY KEY, checksum VARCHAR NOT NULL,
          applied_at TIMESTAMPTZ NOT NULL, description VARCHAR NOT NULL
        )
    """)
    database.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_runs(
          run_id VARCHAR PRIMARY KEY, operation VARCHAR NOT NULL, status VARCHAR NOT NULL,
          started_at TIMESTAMPTZ NOT NULL, completed_at TIMESTAMPTZ,
          source_database VARCHAR NOT NULL, manifest_sha256 VARCHAR, details_json VARCHAR,
          error VARCHAR
        )
    """)
    database.execute("""
        CREATE TABLE IF NOT EXISTS data_revisions(
          revision_id VARCHAR PRIMARY KEY, parent_revision_id VARCHAR,
          operation VARCHAR NOT NULL, manifest_sha256 VARCHAR NOT NULL,
          created_at TIMESTAMPTZ NOT NULL, details_json VARCHAR NOT NULL
        )
    """)
    database.execute("""
        CREATE TABLE IF NOT EXISTS migration_quarantine(
          source_table VARCHAR NOT NULL, instrument VARCHAR, date DATE,
          reason VARCHAR NOT NULL, detected_at TIMESTAMPTZ NOT NULL,
          migration_id VARCHAR NOT NULL
        )
    """)


def _copy_v1_market_data(database) -> dict:
    tables = _tables(database)
    if "daily_final" in tables and "daily_intraday_overlay" in tables and "minute5_final" in tables:
        return {"alreadyPrepared": True, "finalRows": 0, "minute5Rows": 0,
                "overlayRows": 0, "quarantinedRows": 0, "overlayDate": None}
    columns = ",".join(_columns(database, "daily"))
    minute_columns = ",".join(_columns(database, "minute5"))
    database.execute("CREATE TABLE daily_final AS SELECT * FROM daily WHERE false")
    database.execute("CREATE TABLE daily_intraday_overlay AS SELECT * FROM daily WHERE false")
    database.execute("CREATE TABLE minute5_final AS SELECT * FROM minute5 WHERE false")
    database.execute("INSERT INTO daily_final SELECT * FROM daily WHERE bar_state='final'")
    database.execute("INSERT INTO minute5_final SELECT * FROM minute5")
    last_complete = database.execute("SELECT max(date) FROM daily_final").fetchone()[0]
    calendar = {row[0] for row in database.execute("SELECT date FROM trading_calendar WHERE is_trading_day").fetchall()}
    provisional_dates = [row[0] for row in database.execute(
        "SELECT DISTINCT date FROM daily WHERE bar_state='provisional' ORDER BY date"
    ).fetchall()]
    allowed = [day for day in provisional_dates if day > last_complete and day in calendar]
    selected_day = allowed[-1] if allowed else None
    quarantined = 0
    for day in provisional_dates:
        if day <= last_complete:
            reason = "stale_provisional_at_or_before_last_complete"
        elif day not in calendar:
            reason = "provisional_on_non_trading_day"
        elif day != selected_day:
            reason = "superseded_by_newer_intraday_overlay"
        else:
            continue
        count = database.execute("SELECT count(*) FROM daily WHERE bar_state='provisional' AND date=?", [day]).fetchone()[0]
        database.execute("""
            INSERT INTO migration_quarantine
            SELECT 'daily',instrument,date,?,current_timestamp,?
            FROM daily WHERE bar_state='provisional' AND date=?
        """, [reason, MIGRATION_ID, day])
        quarantined += int(count)
    if selected_day is not None:
        database.execute("INSERT INTO daily_intraday_overlay SELECT * FROM daily WHERE bar_state='provisional' AND date=?",
                         [selected_day])
    database.execute("""
        CREATE OR REPLACE VIEW daily_effective AS
        SELECT * FROM daily_intraday_overlay
        UNION ALL
        SELECT f.* FROM daily_final f
        WHERE NOT EXISTS (
          SELECT 1 FROM daily_intraday_overlay o
          WHERE o.instrument=f.instrument AND o.date=f.date
        )
    """)
    for table, key in (("daily_final", "instrument,date"), ("daily_intraday_overlay", "instrument,date"),
                       ("minute5_final", "instrument,datetime")):
        if database.execute(f"SELECT 1 FROM {table} GROUP BY {key} HAVING count(*)>1 LIMIT 1").fetchone():
            raise MigrationError(f"{table} has duplicate keys after migration")
    final_rows = database.execute("SELECT count(*) FROM daily_final").fetchone()[0]
    minute_rows = database.execute("SELECT count(*) FROM minute5_final").fetchone()[0]
    overlay_rows = database.execute("SELECT count(*) FROM daily_intraday_overlay").fetchone()[0]
    return {"alreadyPrepared": False, "finalRows": int(final_rows), "minute5Rows": int(minute_rows),
            "overlayRows": int(overlay_rows), "quarantinedRows": quarantined,
            "overlayDate": str(selected_day) if selected_day else None, "dailyColumns": columns,
            "minuteColumns": minute_columns}


def _validate_stage(path: Path) -> dict:
    audit = audit_database(path)
    summaries = audit["tableSummaries"]
    required = {"daily_final", "daily_intraday_overlay", "daily_effective", "minute5_final",
                "schema_migrations", "ingestion_runs", "data_revisions"}
    missing = sorted(required - set(audit["tables"]))
    if missing:
        raise MigrationError("staging database is missing v2 objects: " + ", ".join(missing))
    overlay = summaries["daily_intraday_overlay"]
    with connect(path, read_only=True) as database:
        dates = database.execute("SELECT DISTINCT date FROM daily_intraday_overlay ORDER BY date").fetchall()
        duplicates = database.execute("""
            SELECT 1 FROM daily_effective GROUP BY instrument,date HAVING count(*)>1 LIMIT 1
        """).fetchone()
        stale = database.execute("""
            SELECT count(*) FROM daily_intraday_overlay
            WHERE date <= (SELECT max(date) FROM daily_final)
        """).fetchone()[0]
    if len(dates) > 1:
        raise MigrationError("v2 staging overlay spans more than one trade date")
    if stale:
        raise MigrationError("v2 staging contains a stale overlay")
    if duplicates:
        raise MigrationError("daily_effective has duplicate instrument/date keys")
    return {"audit": audit, "overlayDates": [str(row[0]) for row in dates],
            "overlayRows": overlay["rows"], "passed": True}


def rebuild_provider(database_path_value, output_root, *, profile=None, source_state=None,
                     generation_id=None, release_id=None) -> dict:
    """Rebuild a v2 provider into ``output_root`` from one database revision.

    The descriptor is written last and carries the revision the provider was
    built for, so a reader can always detect a provider that lags its database.
    """
    database = database_path(database_path_value)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from .storage import MarketStore
    store = MarketStore(database)
    if not store.is_v2:
        raise MigrationError("v2 provider rebuild requires a v2 database")
    profile = profile or "staging-v2"
    source_state = source_state or {"state": "staging_only"}
    data = store.describe()
    with connect(database, read_only=True) as db:
        first, last = db.execute("SELECT min(date),max(date) FROM daily_effective").fetchone()
        complete = db.execute("SELECT max(date) FROM daily_final").fetchone()[0]
        calendar = [str(row[0]) for row in db.execute(
            "SELECT date FROM trading_calendar WHERE is_trading_day ORDER BY date"
        ).fetchall()]
        instruments = db.execute("""
            SELECT instrument,min(date),max(date) FROM daily_effective
            GROUP BY instrument ORDER BY instrument
        """).fetchall()
        minute_first, minute_last = db.execute("SELECT min(date),max(date) FROM minute5_final").fetchone()
        revisions = db.execute("SELECT revision_id FROM data_revisions ORDER BY created_at DESC LIMIT 1").fetchone()
    if first is None or last is None:
        raise MigrationError("cannot build a provider from an empty staging database")
    revision = store.revision() or (revisions[0] if revisions else None)
    content = {"database": str(database), "firstDate": str(first), "lastDate": str(last),
               "lastCompleteDate": str(complete) if complete else None, "revision": revision,
               "instruments": [(item, str(begin), str(end)) for item, begin, end in instruments]}
    version = _json_hash(content)[:16]
    (root / "calendars").mkdir(exist_ok=True)
    (root / "instruments").mkdir(exist_ok=True)
    (root / "calendars/day.txt").write_text("\n".join(day for day in calendar if str(first) <= day <= str(last)) + "\n", encoding="utf-8")
    clocks = []
    for hour, minute in ((9, 30), (13, 0)):
        for index in range(1, 25):
            total = hour * 60 + minute + index * 5
            clocks.append(f"{total // 60:02d}:{total % 60:02d}:00")
    minute_days = [day for day in calendar if minute_first and str(minute_first) <= day <= str(minute_last)]
    (root / "calendars/5min.txt").write_text("\n".join(
        f"{day} {clock}" for day in minute_days for clock in clocks
    ) + "\n", encoding="utf-8")
    (root / "instruments/all.txt").write_text("\n".join(
        f"{item}\t{begin}\t{end} 15:00:00" for item, begin, end in content["instruments"]
    ) + "\n", encoding="utf-8")
    descriptor = {"schema_version": "stock-qlib-provider-v2", "profile": profile,
                  "layout": V2_LAYOUT, "database": str(database), "provider_version": version,
                  "data_revision": revision, "first_date": str(first), "last_date": str(last),
                  "last_complete_date": str(complete) if complete else None,
                  "data_version": data["metadata"].get("data_version"),
                  "baseline_id": profile,
                  "timezone": data["metadata"].get("timezone", "Asia/Shanghai"),
                  "numeric_type": "float64", "amount_unit": "CNY", "volume_unit": "shares",
                  "adjustment_policy": "none; raw CNY prices and share volume",
                  "frequencies": ["day", "5min"], "counts": data["counts"],
                  "instruments": data["instruments"], "requested_universe": data["requested_universe"],
                  "references": data["references"],
                  "generation_id": generation_id, "release_id": release_id,
                  "source_state": dict(source_state)}
    # Written last and atomically: a reader either sees the previous revision or
    # the new one, never a descriptor that announces data it cannot serve.
    temporary = root / ".descriptor.json.tmp"
    temporary.write_text(json.dumps(descriptor, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temporary, root / "descriptor.json")
    return {"root": str(root), "descriptor": descriptor,
            "validation": verify_provider(database, root,
                                          expected_state=source_state.get("state"))}


def rebuild_staging_provider(database_path_value, output_root) -> dict:
    """Rebuild a disposable provider beside a staging database, never live provider."""
    return rebuild_provider(database_path_value, output_root, profile="staging-v2",
                            source_state={"state": "staging_only"})


def verify_provider(database_path_value, root, *, expected_state=None) -> dict:
    """Verify that a v2 provider cites precisely its database and revision."""
    database = database_path(database_path_value)
    root = Path(root).resolve()
    descriptor_path = root / "descriptor.json"
    required = [descriptor_path, root / "calendars/day.txt", root / "calendars/5min.txt", root / "instruments/all.txt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise MigrationError("provider is incomplete: " + ", ".join(missing))
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("database") != str(database) or descriptor.get("layout") != V2_LAYOUT:
        raise MigrationError("provider does not point at its v2 database")
    if expected_state is not None and descriptor.get("source_state", {}).get("state") != expected_state:
        raise MigrationError(f"provider is not marked {expected_state}")
    with connect(database, read_only=True) as db:
        metadata = dict(db.execute("SELECT key,value FROM store_metadata").fetchall())
    revision = metadata.get("overlay_revision") or metadata.get("data_revision")
    if revision is not None and descriptor.get("data_revision") != revision:
        raise MigrationError("provider revision does not match its database revision")
    return {"passed": True, "descriptor": str(descriptor_path), "database": str(database),
            "providerVersion": descriptor.get("provider_version"),
            "revision": descriptor.get("data_revision")}


def verify_staging_provider(database_path_value, root) -> dict:
    """Verify that a staging provider cites precisely the staging database."""
    return verify_provider(database_path_value, root, expected_state="staging_only")


def migrate_v1_copy(source, staging_root, *, dry_run: bool = False, run_id: str | None = None) -> dict:
    """Create and validate a v2 staging copy; it intentionally has no activate flag."""
    source_path = database_path(source)
    staging_root = Path(staging_root).resolve()
    run_id = run_id or f"migration-{uuid4().hex}"
    registry = verify_migration_registry()
    preflight = _preflight(source_path, staging_root)
    _require_v1_source(source_path)
    before = audit_database(source_path)
    report = {"schema": "stock-market-migration-report-v1", "runId": run_id,
              "source": str(source_path), "sourceSha256": _file_hash(source_path),
              "targetLayout": V2_LAYOUT, "migrations": registry, "preflight": preflight,
              "before": before, "activation": {"attempted": False, "activated": False},
              "rollback": {"sourceDatabase": "unchanged_by_design", "liveProvider": "unchanged_by_design"}}
    if not preflight["passed"]:
        raise MigrationError("insufficient free disk space for staging migration")
    if dry_run:
        report.update({"status": "dry_run", "plannedQuarantine": before["staleProvisional"],
                       "stagingDatabase": None, "provider": None})
        (staging_root / f"{run_id}.dry-run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    staging = staging_root / f"{run_id}.duckdb"
    if staging.exists():
        raise MigrationError(f"staging target already exists: {staging}")
    shutil.copy2(source_path, staging)
    report["stagingDatabase"] = str(staging)
    try:
        with connect(staging) as database:
            database.execute("BEGIN TRANSACTION")
            _create_v2_schema(database)
            existing = dict(database.execute("SELECT migration_id,checksum FROM schema_migrations").fetchall())
            expected = MIGRATIONS[0]
            if expected.migration_id in existing:
                if existing[expected.migration_id] != expected.checksum:
                    raise MigrationError("applied migration checksum does not match the registered checksum")
                data_report = {"alreadyApplied": True}
            else:
                data_report = _copy_v1_market_data(database)
                database.execute("INSERT INTO schema_migrations VALUES(?,?,current_timestamp,?)",
                                 [expected.migration_id, expected.checksum, expected.description])
            database.execute("INSERT INTO ingestion_runs VALUES(?,?,?,current_timestamp,NULL,?,?,?,NULL)",
                             [run_id, "schema_migration", "running", str(source_path), None,
                              json.dumps({"migration": expected.migration_id}, ensure_ascii=False)])
            database.execute("INSERT OR REPLACE INTO store_metadata VALUES('layout',?)", [V2_LAYOUT])
            database.execute("COMMIT")
            database.execute("CHECKPOINT")
        validation = _validate_stage(staging)
        report["dataMigration"] = data_report
        report["validation"] = validation
        manifest_hash = _json_hash({key: value for key, value in report.items() if key not in ("provider", "after")})
        with connect(staging) as database:
            parent = database.execute("SELECT revision_id FROM data_revisions ORDER BY created_at DESC LIMIT 1").fetchone()
            revision_id = _json_hash({"parent": parent[0] if parent else None, "operation": "schema_migration",
                                      "manifest": manifest_hash})[:32]
            database.execute("INSERT OR REPLACE INTO data_revisions VALUES(?,?,?,?,current_timestamp,?)",
                             [revision_id, parent[0] if parent else None, "schema_migration", manifest_hash,
                              json.dumps({"runId": run_id, "migration": MIGRATION_ID}, ensure_ascii=False)])
            database.execute("UPDATE ingestion_runs SET status='success',completed_at=current_timestamp,manifest_sha256=?,details_json=? WHERE run_id=?",
                             [manifest_hash, json.dumps({"revisionId": revision_id}, ensure_ascii=False), run_id])
            database.execute("CHECKPOINT")
        provider = rebuild_staging_provider(staging, staging_root / f"{run_id}.provider")
        report.update({"status": "success", "after": audit_database(staging), "provider": provider,
                       "revisionId": revision_id, "manifestSha256": manifest_hash,
                       "stagingSha256": _file_hash(staging)})
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "provider": None})
        failure = staging_root / f"{run_id}.failed.json"
        failure.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise MigrationError(report["error"]) from error
    (staging_root / f"{run_id}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
