"""Layout-aware access to the market database.

The application used to know that v1 meant ``daily``/``minute5``.  Keeping
that knowledge in every reader made the v2 staging copy unsafe: a caller could
silently read the legacy mixed table or write an intraday row into final data.
This module is the small routing boundary shared by readers, writers,
publication and audit code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .market import database_path


V1_LAYOUT = "market-database-v1"
V2_LAYOUT = "market-database-v2"


@dataclass(frozen=True)
class LayoutDescriptor:
    layout: str
    database: Path
    daily_effective: str
    daily_final: str
    daily_overlay: str | None
    minute5_final: str
    is_v2: bool

    @property
    def schema_version(self):
        return self.layout


def _descriptor_value(value) -> tuple[dict, Path]:
    if isinstance(value, dict):
        meta = dict(value)
        return meta, database_path(meta["database"])
    path = Path(value).resolve()
    if path.name == "descriptor.json":
        meta = json.loads(path.read_text(encoding="utf-8"))
        return meta, database_path(meta["database"])
    database = database_path(path)
    return {}, database


def _tables(database: Path) -> set[str]:
    from .market import connect

    with connect(database, read_only=True) as db:
        return {row[0] for row in db.execute("SHOW TABLES").fetchall()}


def resolve_layout(value) -> LayoutDescriptor:
    """Resolve a descriptor or database path without requiring manual switches."""
    meta, database = _descriptor_value(value)
    layout = meta.get("layout") or meta.get("schema_version")
    if layout not in (V1_LAYOUT, V2_LAYOUT):
        if not database.exists():
            layout = V1_LAYOUT
        else:
            tables = _tables(database)
            layout = V2_LAYOUT if {"daily_final", "daily_intraday_overlay", "minute5_final"} <= tables else V1_LAYOUT
    if layout == V2_LAYOUT:
        return LayoutDescriptor(layout, database, "daily_effective", "daily_final",
                                "daily_intraday_overlay", "minute5_final", True)
    return LayoutDescriptor(layout, database, "daily", "daily", None, "minute5", False)


class MarketStore:
    """A read/write-safe semantic view over either database layout."""

    def __init__(self, value=None):
        if value is None:
            from .active_market import active_market_context

            value = active_market_context().as_descriptor()
        self.descriptor = resolve_layout(value)

    @property
    def database(self) -> Path:
        return self.descriptor.database

    @property
    def layout(self) -> str:
        return self.descriptor.layout

    @property
    def is_v2(self) -> bool:
        return self.descriptor.is_v2

    def table(self, semantic: str) -> str:
        routes = {
            "daily_effective": self.descriptor.daily_effective,
            "daily_final": self.descriptor.daily_final,
            "daily_overlay": self.descriptor.daily_overlay,
            "minute5_final": self.descriptor.minute5_final,
        }
        if semantic not in routes or routes[semantic] is None:
            raise ValueError(f"semantic table is unavailable in {self.layout}: {semantic}")
        return routes[semantic]

    def metadata(self) -> dict[str, str]:
        from .market import connect

        with connect(self.database, read_only=True) as db:
            return dict(db.execute("SELECT key,value FROM store_metadata").fetchall())

    def revision(self) -> str | None:
        metadata = self.metadata()
        return metadata.get("overlay_revision") or metadata.get("data_revision")

    def describe(self) -> dict:
        """Return semantic counts; v2 never reports legacy mixed rows as effective."""
        from .market import connect

        daily = self.table("daily_effective")
        minute = self.table("minute5_final")
        with connect(self.database, read_only=True) as db:
            metadata = dict(db.execute("SELECT key,value FROM store_metadata").fetchall())
            counts, spans = {}, {}
            for table, freq in ((daily, "day"), (minute, "5min")):
                rows, first, last = db.execute(
                    f"SELECT count(*),min(date),max(date) FROM {table}"
                ).fetchone()
                counts[freq] = {"rows": int(rows), "first_date": str(first) if first else None,
                                "last_date": str(last) if last else None}
                for inst, first, last in db.execute(
                    f"SELECT instrument,min(date),max(date) FROM {table} GROUP BY instrument"
                ).fetchall():
                    spans.setdefault(inst, {})[freq] = [str(first), str(last)]
            final = self.table("daily_final")
            last_complete = db.execute(
                f"SELECT max(date) FROM {final} WHERE bar_state='final'"
            ).fetchone()[0]
            calendar = [str(row[0]) for row in db.execute(
                "SELECT DISTINCT date FROM trading_calendar WHERE is_trading_day ORDER BY date"
            ).fetchall()]
            securities = [row[0] for row in db.execute(
                "SELECT DISTINCT instrument FROM securities ORDER BY instrument"
            ).fetchall()]
            references = [row[0] for row in db.execute(
                "SELECT name FROM reference_documents ORDER BY name"
            ).fetchall()]
            final_rows = db.execute(f"SELECT count(*) FROM {final}").fetchone()[0]
            overlay_rows = (db.execute(
                f"SELECT count(*) FROM {self.table('daily_overlay')}"
            ).fetchone()[0] if self.is_v2 else 0)
        return {
            "layout": self.layout, "database": str(self.database), "metadata": metadata,
            "counts": counts, "finalCounts": {"day": int(final_rows)},
            "overlayCount": int(overlay_rows), "instruments": spans, "calendar": calendar,
            "requested_universe": securities, "last_complete_date": str(last_complete) if last_complete else None,
            "references": references, "data_revision": self.revision(),
        }


def route_descriptor(value) -> dict:
    """Expose the resolved routes for manifests and diagnostics."""
    store = MarketStore(value)
    return {"layout": store.layout, "database": str(store.database),
            "tables": {name: store.table(name) for name in
                       ("daily_effective", "daily_final", "minute5_final")},
            "daily_overlay": store.descriptor.daily_overlay}
