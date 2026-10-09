"""Read-only database evidence used by migration and C0 contract tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .market import connect, database_path
from .storage import MarketStore, V2_LAYOUT


def _table_columns(database, table):
    return [row[1] for row in database.execute(f"PRAGMA table_info('{table}')").fetchall()]


def _table_summary(database, table, *, state_column=None):
    rows, first, last = database.execute(
        f"SELECT count(*), min(date), max(date) FROM {table}"
    ).fetchone()
    summary = {"rows": int(rows), "firstDate": str(first) if first else None,
               "lastDate": str(last) if last else None, "columns": _table_columns(database, table)}
    if state_column and state_column in summary["columns"]:
        summary["stateDistribution"] = {
            str(state): int(count) for state, count in database.execute(
                f"SELECT {state_column}, count(*) FROM {table} GROUP BY {state_column} ORDER BY {state_column}"
            ).fetchall()
        }
    return summary


def audit_database(value) -> dict:
    """Return structure, ranges, states, references and stale provisional rows.

    This function always opens the target read-only and never rebuilds a
    provider.  It understands both the v1 source tables and a v2 staging copy.
    """
    store = MarketStore(value)
    path = store.database
    with connect(path, read_only=True) as database:
        tables = {row[0] for row in database.execute("SHOW TABLES").fetchall()}
        table_summaries = {}
        for table, state_column in (("daily", "bar_state"), ("minute5", None),
                                    ("daily_final", "bar_state"),
                                    ("daily_intraday_overlay", "bar_state"),
                                    ("minute5_final", None)):
            if table in tables:
                table_summaries[table] = _table_summary(database, table, state_column=state_column)
        metadata = (dict(database.execute("SELECT key,value FROM store_metadata").fetchall())
                    if "store_metadata" in tables else {})
        references = []
        if "reference_documents" in tables:
            for name, payload, updated_at in database.execute(
                "SELECT name,payload,updated_at FROM reference_documents ORDER BY name"
            ).fetchall():
                text = payload or ""
                references.append({"name": name, "bytes": len(text.encode("utf-8")),
                                   "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                   "updatedAt": str(updated_at) if updated_at else None})
        final_table = store.table("daily_final")
        final_filter = "WHERE bar_state='final'" if "bar_state" in _table_columns(database, final_table) else ""
        last_complete = database.execute(f"SELECT max(date) FROM {final_table} {final_filter}").fetchone()[0]
        provisional = []
        quarantine = []
        quarantine_details = []
        if store.is_v2:
            overlay = store.table("daily_overlay")
            provisional = [(str(day), int(count)) for day, count in database.execute(
                f"SELECT date,count(*) FROM {overlay} GROUP BY date ORDER BY date"
            ).fetchall()]
            stale = [(str(day), int(count)) for day, count in database.execute(
                f"SELECT date,count(*) FROM {overlay} WHERE date<=? GROUP BY date ORDER BY date",
                [last_complete],
            ).fetchall()] if last_complete is not None else []
            if "migration_quarantine" in tables:
                quarantine = [{"date": str(day) if day else None, "reason": reason, "rows": int(rows)}
                              for day, reason, rows in database.execute(
                                  "SELECT date,reason,count(*) FROM migration_quarantine GROUP BY date,reason ORDER BY date,reason"
                              ).fetchall()]
                quarantine_details = [{"date": str(day) if day else None, "code": instrument,
                                       "state": "provisional", "reason": reason, "rows": int(rows)}
                                      for day, instrument, reason, rows in database.execute(
                                          "SELECT date,instrument,reason,count(*) FROM migration_quarantine "
                                          "GROUP BY date,instrument,reason ORDER BY date,instrument,reason"
                                      ).fetchall()]
        else:
            if "daily" in tables and "bar_state" in _table_columns(database, "daily"):
                provisional = [(str(day), int(count)) for day, count in database.execute(
                    "SELECT date,count(*) FROM daily WHERE bar_state='provisional' GROUP BY date ORDER BY date"
                ).fetchall()]
            stale = [(str(day), int(count)) for day, count in database.execute(
                "SELECT date,count(*) FROM daily WHERE bar_state='provisional' AND date<=? GROUP BY date ORDER BY date",
                [last_complete],
            ).fetchall()] if last_complete is not None and "daily" in tables else []
    return {"database": str(path), "tables": sorted(tables), "tableSummaries": table_summaries,
            "metadata": metadata, "lastCompleteDate": str(last_complete) if last_complete else None,
            "provisional": [{"date": day, "rows": rows} for day, rows in provisional],
            "staleProvisional": [{"date": day, "rows": rows} for day, rows in stale],
            "quarantine": quarantine,
            "quarantineStats": {
                "byDate": [{"date": day, "rows": sum(row["rows"] for row in quarantine_details if row["date"] == day)}
                           for day in sorted({row["date"] for row in quarantine_details})],
                "byReason": [{"reason": reason, "rows": sum(row["rows"] for row in quarantine_details if row["reason"] == reason)}
                             for reason in sorted({row["reason"] for row in quarantine_details})],
                "byState": [{"state": state, "rows": sum(row["rows"] for row in quarantine_details if row["state"] == state)}
                            for state in sorted({row["state"] for row in quarantine_details})],
                "byCode": [{"code": code, "rows": sum(row["rows"] for row in quarantine_details if row["code"] == code)}
                           for code in sorted({row["code"] for row in quarantine_details})],
                "details": quarantine_details,
            },
            "layout": V2_LAYOUT if store.is_v2 else store.layout,
            "references": references}
