"""One mutable DuckDB for daily bars, five-minute bars and required references."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import duckdb

from .common import DATA_ROOT, utc_now

ROOT = DATA_ROOT
# ``DATABASE`` is the bootstrap v1 file.  It is deliberately *not* the default
# for production entry points any more: pass ``None`` (the default of every
# helper below) to follow the pinned active market context.  Only migration and
# bootstrap code should name this constant directly.
DATABASE = ROOT / "market.duckdb"
BOOTSTRAP_DATABASE = DATABASE
LAYOUT = "market-database-v1"
SHANGHAI = ZoneInfo("Asia/Shanghai")
FIELDS = ["exchange", "code", "instrument", "date", "datetime", "freq", "open", "high", "low", "close",
          "volume", "amount", "is_st", "trade_status", "zero_volume", "quality_ohlc_ok", "quality_time_ok"]
DAILY_COLUMNS = FIELDS + ["eligible", "bar_state", "snapshot_at"]
MINUTE_COLUMNS = FIELDS + ["minute_day_ok", "eligible"]
DAILY_MARKET_METRIC_COLUMNS = [
    "turnover_rate", "pe_ttm", "total_market_cap", "float_market_cap",
    "pb", "previous_close", "close_limit_status",
]


def sql_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def active_database():
    """The database the pinned active market context points at."""
    from .active_market import active_market_context

    return active_market_context().database


def database_path(value=None):
    """Resolve a database path; ``None`` means "whatever is active now"."""
    if value is None:
        return active_database()
    path = Path(value).resolve()
    return path if path.suffix == ".duckdb" else path / "market.duckdb"


def connect(value=None, read_only=False):
    database = duckdb.connect(str(database_path(value)), read_only=read_only)
    database.execute("SET TimeZone='Asia/Shanghai'")
    database.execute("SET threads=2")
    return database


def create_daily_effective_view(database):
    """Expose the effective daily bar and optional point-in-time market metrics.

    The market metrics live in their own keyed daily table so adding valuation
    data never rewrites the multi-million-row OHLCV tables.  Older/staging v2
    databases that have not installed the table keep the original bar-only
    view.
    """
    tables = {row[0] for row in database.execute("SHOW TABLES").fetchall()}
    if "daily_market_metrics" not in tables:
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
        return
    metric_columns = {
        row[0] for row in database.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='daily_market_metrics'"
        ).fetchall()
    }
    close_limit_status = (
        "m.close_limit_status" if "close_limit_status" in metric_columns
        else "NULL::TINYINT AS close_limit_status"
    )
    database.execute(f"""
        CREATE OR REPLACE VIEW daily_effective AS
        WITH daily_base AS (
          SELECT * FROM daily_intraday_overlay
          UNION ALL
          SELECT f.* FROM daily_final f
          WHERE NOT EXISTS (
            SELECT 1 FROM daily_intraday_overlay o
            WHERE o.instrument=f.instrument AND o.date=f.date
          )
        )
        SELECT
          b.*,
          m.turnover_rate,
          m.pe_ttm,
          m.total_market_cap,
          m.float_market_cap,
          m.pb,
          m.previous_close,
          {close_limit_status},
          m.source AS market_metrics_source,
          m.snapshot_at AS market_metrics_snapshot_at
        FROM daily_base b
        LEFT JOIN daily_market_metrics m
          ON m.instrument=b.instrument AND m.date=b.date
    """)


@contextmanager
def database_update_copy(value=None):
    """Update a same-directory copy and atomically replace the live database."""
    target = database_path(value)
    if not target.is_file():
        raise FileNotFoundError(target)
    with connect(target) as database:
        database.execute("CHECKPOINT")
    required = target.stat().st_size + 64 * 1024 * 1024
    free = shutil.disk_usage(target.parent).free
    if free < required:
        raise OSError(f"数据库副本需要至少 {required} 字节可用空间，当前只有 {free} 字节")
    staging = target.with_name(f".{target.stem}.update-{uuid4().hex}.duckdb")
    shutil.copy2(target, staging)
    try:
        yield staging
        with connect(staging) as database:
            required_tables = {
                "daily", "minute5", "securities", "trading_calendar",
                "reference_documents", "store_metadata",
            }
            actual = {row[0] for row in database.execute("SHOW TABLES").fetchall()}
            missing = sorted(required_tables - actual)
            if missing:
                raise ValueError(f"数据库副本缺少必要表：{', '.join(missing)}")
            database.execute("CHECKPOINT")
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)
        staging.with_suffix(staging.suffix + ".wal").unlink(missing_ok=True)


def touch(database):
    database.execute("INSERT OR REPLACE INTO store_metadata VALUES('data_version',?)", [utc_now()])


def reference(name, default=None, value=None):
    with connect(value, read_only=True) as database:
        row = database.execute("SELECT payload FROM reference_documents WHERE name=?", [name]).fetchone()
    return json.loads(row[0]) if row else default


def write_reference(name, payload, value=None):
    with connect(value) as database:
        database.execute("INSERT OR REPLACE INTO reference_documents VALUES(?,?,current_timestamp)",
                         [name, json.dumps(payload, ensure_ascii=False, allow_nan=False)])
        touch(database)
        database.execute("CHECKPOINT")


def describe(value=None):
    from .storage import MarketStore
    return MarketStore(value).describe()


def write_quotes(packet, as_of, value=None):
    """Overwrite today's row with the scan node; the close update overwrites it again."""
    from .storage import MarketStore
    if MarketStore(value).is_v2:
        raise ValueError("v2 intraday writes require overlay preflight and commit_overlay")
    if packet.get("units") != {"price": "CNY", "volume": "share", "amount": "CNY"}:
        raise ValueError("盘中报价必须使用元、股、元")
    stamp = datetime.fromisoformat(as_of.replace("Z", "+00:00")) if isinstance(as_of, str) else as_of
    if stamp.tzinfo is None:
        raise ValueError("扫描时间必须带时区")
    stamp = stamp.astimezone(SHANGHAI)
    rows, rejected, seen = [], [], set()
    for quote in packet.get("quotes", []):
        code = str(quote.get("code", ""))
        try:
            if len(code) != 6 or not code.isdigit() or code in seen:
                raise ValueError("股票代码无效或重复")
            seen.add(code)
            numbers = [quote[key] for key in ("open", "high", "low", "price", "volume", "amount")]
            if any(not isinstance(number, (float, int)) or not math.isfinite(number) or number < 0 for number in numbers):
                raise ValueError("价量额无效")
            o, h, l, c, volume, amount = numbers
            if min(o, h, l, c) <= 0 or l > min(o, c) or h < max(o, c):
                raise ValueError("OHLC关系无效")
            supplier = datetime.fromisoformat(quote["timestamp"].replace("Z", "+00:00")).astimezone(SHANGHAI)
            if supplier.date() != stamp.date():
                raise ValueError("报价不是扫描当天")
            exchange = "sh" if code.startswith("6") else "sz"
            name = str(quote.get("name", "")).replace(" ", "").upper()
            is_st = name.startswith(("ST", "*ST", "SST", "S*ST"))
            status = 1 if volume > 0 else 0
            rows.append((exchange, code, exchange + code, stamp.date(), stamp, "1d", o, h, l, c, volume, amount,
                         is_st, status, volume == 0, True, True, not is_st and status == 1, "provisional", supplier))
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"code": code, "reason": str(error)})
    if not rows:
        return {"date": stamp.date().isoformat(), "written": 0, "rejected": rejected}
    with connect(value) as database:
        database.execute("CREATE TEMP TABLE incoming_quotes AS SELECT * FROM daily LIMIT 0")
        database.executemany(f"INSERT INTO incoming_quotes({','.join(DAILY_COLUMNS)}) VALUES({','.join('?' for _ in DAILY_COLUMNS)})", rows)
        database.execute("BEGIN TRANSACTION")
        database.execute("DELETE FROM daily USING incoming_quotes q WHERE daily.instrument=q.instrument AND daily.date=q.date")
        database.execute(f"INSERT INTO daily({','.join(DAILY_COLUMNS)}) SELECT {','.join(DAILY_COLUMNS)} FROM incoming_quotes")
        database.execute("INSERT INTO trading_calendar(date,is_trading_day) SELECT CAST(? AS DATE),true WHERE NOT EXISTS(SELECT 1 FROM trading_calendar WHERE date=CAST(? AS DATE))",
                         [stamp.date().isoformat(), stamp.date().isoformat()])
        touch(database)
        database.execute("COMMIT")
        database.execute("CHECKPOINT")
    return {"date": stamp.date().isoformat(), "written": len(rows), "rejected": rejected,
            "barState": "provisional", "asOf": stamp.isoformat()}


def build(history, output=None, merge_existing=True):
    """Overwrite observed daily keys and whole five-minute security/day groups.

    ``output=None`` publishes into the pinned active database, so a close update
    follows the DB3 pointer instead of the bootstrap v1 file.
    """
    source = database_path(history)
    destination = database_path(output)
    if source == destination:
        raise ValueError("采集批次不能是正式数据库本身")
    from .storage import MarketStore
    if MarketStore(destination).is_v2:
        return _build_v2(history, destination)
    with connect(destination) as database:
        database.execute(f"ATTACH {sql_quote(source)} AS capture (READ_ONLY)")
        fields = ','.join('r.' + name for name in FIELDS)
        eligible = "coalesce(r.is_st=false AND r.trade_status=1 AND r.quality_ohlc_ok AND r.quality_time_ok,false)"
        database.execute(f"CREATE TABLE IF NOT EXISTS daily AS SELECT {fields},{eligible} eligible,'final'::VARCHAR bar_state,NULL::TIMESTAMPTZ snapshot_at FROM capture.daily r WHERE false")
        database.execute(f"CREATE TABLE IF NOT EXISTS minute5 AS SELECT {fields},r.minute_day_ok,{eligible} AND coalesce(r.minute_day_ok,false) eligible FROM capture.minute5 r WHERE false")
        for table in ("securities", "trading_calendar"):
            database.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM capture.{table} WHERE false")
        database.execute("CREATE TABLE IF NOT EXISTS reference_documents(name VARCHAR PRIMARY KEY,payload VARCHAR,updated_at TIMESTAMPTZ)")
        database.execute("CREATE TABLE IF NOT EXISTS store_metadata(key VARCHAR PRIMARY KEY,value VARCHAR)")
        database.execute("INSERT OR REPLACE INTO store_metadata VALUES('layout',?)", [LAYOUT])
        database.execute(f"CREATE TEMP TABLE incoming_daily AS SELECT {fields},{eligible} eligible,'final'::VARCHAR bar_state,NULL::TIMESTAMPTZ snapshot_at FROM capture.daily r")
        database.execute(f"CREATE TEMP TABLE incoming_minute AS SELECT {fields},r.minute_day_ok,{eligible} AND coalesce(r.minute_day_ok,false) eligible FROM capture.minute5 r")
        for table, key in (("incoming_daily", "instrument,date"), ("incoming_minute", "instrument,datetime")):
            if database.execute(f"SELECT 1 FROM {table} GROUP BY {key} HAVING count(*)>1 LIMIT 1").fetchone():
                raise ValueError(f"{table} 有重复数据键")
        incoming = database.execute("SELECT count(*),min(date),max(date) FROM incoming_daily").fetchone()
        if not incoming[0]:
            raise ValueError("采集批次没有日线记录")
        database.execute("BEGIN TRANSACTION")
        database.execute("DELETE FROM daily WHERE bar_state='provisional' AND date IN (SELECT DISTINCT date FROM incoming_daily)")
        database.execute("DELETE FROM daily USING incoming_daily n WHERE daily.instrument=n.instrument AND daily.date=n.date")
        database.execute(f"INSERT INTO daily({','.join(DAILY_COLUMNS)}) SELECT {','.join(DAILY_COLUMNS)} FROM incoming_daily")
        database.execute("DELETE FROM minute5 USING (SELECT DISTINCT instrument,date FROM incoming_minute) n WHERE minute5.instrument=n.instrument AND minute5.date=n.date")
        database.execute(f"INSERT INTO minute5({','.join(MINUTE_COLUMNS)}) SELECT {','.join(MINUTE_COLUMNS)} FROM incoming_minute")
        for table, key in (("securities", "instrument"), ("trading_calendar", "date")):
            database.execute(f"DELETE FROM {table} USING capture.{table} n WHERE {table}.{key}=n.{key}")
            database.execute(f"INSERT INTO {table} BY NAME SELECT * FROM capture.{table}")
        touch(database)
        database.execute("CREATE OR REPLACE VIEW eligible_daily AS SELECT * FROM daily WHERE eligible")
        database.execute("CREATE OR REPLACE VIEW eligible_minute5 AS SELECT * FROM minute5 WHERE eligible")
        database.execute("COMMIT")
        database.execute("CHECKPOINT")
    return destination


def _build_v2(history, destination):
    """Publish a validated capture batch into v2 final tables only.

    The legacy ``daily`` table may still exist in a migrated file for audit
    provenance, but it is never read or overwritten by this path.  When the
    destination is an active generation, the generation is marked ``pending``
    first and stays that way until the provider rebuild records the new
    revision; readers fail closed in between instead of mixing two revisions.
    """
    source = database_path(history)
    from .migrations import _json_hash
    from .active_market import mark_generation_pending

    mark_generation_pending(destination, "final_data_update")
    with connect(destination) as database:
        database.execute(f"ATTACH {sql_quote(source)} AS capture (READ_ONLY)")
        eligible_daily = "coalesce(r.is_st=false AND r.trade_status=1 AND r.quality_ohlc_ok AND r.quality_time_ok,false)"
        eligible_minute = eligible_daily + " AND coalesce(r.minute_day_ok,false)"
        fields = ",".join(FIELDS)
        database.execute(f"CREATE TEMP TABLE incoming_daily AS SELECT {fields},{eligible_daily} eligible,'final'::VARCHAR bar_state,NULL::TIMESTAMPTZ snapshot_at FROM capture.daily r")
        database.execute(f"CREATE TEMP TABLE incoming_minute AS SELECT {fields},r.minute_day_ok,{eligible_minute} eligible FROM capture.minute5 r")
        for table, key in (("incoming_daily", "instrument,date"), ("incoming_minute", "instrument,datetime")):
            if database.execute(f"SELECT 1 FROM {table} GROUP BY {key} HAVING count(*)>1 LIMIT 1").fetchone():
                raise ValueError(f"{table} 有重复数据键")
        if not database.execute("SELECT 1 FROM incoming_daily LIMIT 1").fetchone():
            raise ValueError("采集批次没有日线记录")
        dates = [str(row[0]) for row in database.execute("SELECT DISTINCT date FROM incoming_daily ORDER BY date").fetchall()]
        parent = database.execute("SELECT revision_id FROM data_revisions ORDER BY created_at DESC LIMIT 1").fetchone()
        manifest = {"operation": "final_data_update", "dates": dates,
                    "dailyRows": int(database.execute("SELECT count(*) FROM incoming_daily").fetchone()[0]),
                    "minuteRows": int(database.execute("SELECT count(*) FROM incoming_minute").fetchone()[0])}
        revision = _json_hash({"parent": parent[0] if parent else None, **manifest})[:32]
        database.execute("BEGIN TRANSACTION")
        database.execute("DELETE FROM daily_final USING incoming_daily n WHERE daily_final.instrument=n.instrument AND daily_final.date=n.date")
        database.execute(f"INSERT INTO daily_final({','.join(DAILY_COLUMNS)}) SELECT {','.join(DAILY_COLUMNS)} FROM incoming_daily")
        database.execute("DELETE FROM minute5_final USING (SELECT DISTINCT instrument,date FROM incoming_minute) n WHERE minute5_final.instrument=n.instrument AND minute5_final.date=n.date")
        database.execute(f"INSERT INTO minute5_final({','.join(MINUTE_COLUMNS)}) SELECT {','.join(MINUTE_COLUMNS)} FROM incoming_minute")
        for table, key in (("securities", "instrument"), ("trading_calendar", "date")):
            database.execute(f"DELETE FROM {table} USING capture.{table} n WHERE {table}.{key}=n.{key}")
            database.execute(f"INSERT INTO {table} BY NAME SELECT * FROM capture.{table}")
        database.execute(f"DELETE FROM daily_intraday_overlay WHERE date<=CAST(? AS DATE)", [max(dates)])
        database.execute("INSERT OR REPLACE INTO data_revisions VALUES(?,?,?,?,current_timestamp,?)",
                         [revision, parent[0] if parent else None, "final_data_update", _json_hash(manifest), json.dumps(manifest, ensure_ascii=False, sort_keys=True)])
        database.execute("INSERT OR REPLACE INTO store_metadata VALUES('layout',?)", ["market-database-v2"])
        database.execute("INSERT OR REPLACE INTO store_metadata VALUES('data_revision',?)", [revision])
        database.execute("INSERT OR REPLACE INTO store_metadata VALUES('data_version',?)", [utc_now()])
        # A final update supersedes any intraday overlay revision: the current
        # revision of this database is the final one until the next overlay.
        database.execute("DELETE FROM store_metadata WHERE key IN ('overlay_quote_snapshot_sha256','overlay_revision')")
        create_daily_effective_view(database)
        database.execute("COMMIT")
        database.execute("CHECKPOINT")
    return destination


def main():
    parser = argparse.ArgumentParser(description="把已采集的日线和5分线覆盖更新到唯一数据库")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="目标数据库；缺省时使用当前固定的活动市场")
    args = parser.parse_args()
    print(json.dumps({"database": str(build(args.history, args.output))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
