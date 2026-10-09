"""The single resolver for the production market database and provider.

Production entry points used to depend on two import-time constants:
``market.DATABASE`` (``data/market.duckdb``) and the fixed provider root
``data/provider``.  That made a DB3 pointer switch a no-op for every default
entry: only a caller that explicitly asked for ``active`` followed the pointer.

This module replaces both with one *pinned* context resolved once per command:

* no active pointer exists  -> ``market``/``history`` resolve to the bootstrap
  v1 database and provider (byte-for-byte the historical behaviour);
* a valid active pointer exists -> they resolve to the pointer's active
  generation;
* a pointer exists but cannot be validated -> :class:`MarketContextError`.
  A broken pointer must never be silently ignored in favour of v1, because that
  would silently resume writing to the database the switch moved away from.

Explicit absolute paths keep working for tests, research and migration tools;
they simply bypass the pointer.  See ``docs/README.md`` for the current data boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .paths import (ENV_DATA_ROOT, FORMAL_DATA_ROOT, LEGACY_ENV_DATA_ROOT,
                    resolve_workspace_path)


POINTER_SCHEMA = "stock-db3-active-pointer-v1"
POINTER_SCHEMA_V2 = "stock-db3-active-pointer-v2"
POINTER_SCHEMAS = (POINTER_SCHEMA, POINTER_SCHEMA_V2)
POINTER_NAME = "active-release.json"
ENV_POINTER = "STOCK_QLIB_ACTIVE_POINTER"

BOOTSTRAP_SOURCE = "bootstrap-v1"
POINTER_SOURCE = "active-pointer"
EXPLICIT_SOURCE = "explicit-path"

V1_LAYOUT = "market-database-v1"
V2_LAYOUT = "market-database-v2"

ACTIVATION_MANIFEST = "activation_manifest.json"
CURRENT_STATE = "current_state.json"

CONSISTENT = "consistent"
PENDING = "pending"


class MarketContextError(RuntimeError):
    """The active market cannot be resolved safely; callers must stop."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def data_root() -> Path:
    """The one data root: env override (tests/rehearsal) or the formal project data dir."""
    override = os.environ.get(ENV_DATA_ROOT) or os.environ.get(LEGACY_ENV_DATA_ROOT)
    if override:
        return Path(override).expanduser().resolve()
    from . import common

    return Path(common.DATA_ROOT)


def bootstrap_database() -> Path:
    return data_root() / "market.duckdb"


def bootstrap_provider() -> Path:
    return data_root() / "provider"


def production_data_root() -> Path:
    """The real stack data directory, *regardless* of the environment override.

    ``data_root()`` is overridable so tests and rehearsals can run a whole
    command chain against a disposable copy.  That override must not be able to
    disguise a change to production as something else, so the few checks that
    ask "is this production?" use this function instead.
    """
    return FORMAL_DATA_ROOT.resolve()


def production_pointer() -> Path:
    """The production active pointer -- the one file a real switch writes."""
    return production_data_root() / POINTER_NAME


def pointer_path(pointer=None) -> Path:
    if pointer is not None:
        return Path(pointer).expanduser().resolve()
    override = os.environ.get(ENV_POINTER)
    if override:
        return Path(override).expanduser().resolve()
    return data_root() / POINTER_NAME


def pointer_digest(pointer=None) -> str | None:
    path = pointer_path(pointer)
    return sha256_file(path) if path.is_file() else None


def _provider_root(value) -> Path:
    path = Path(value).expanduser().resolve()
    return path.parent if path.name == "descriptor.json" else path


@dataclass(frozen=True)
class ActiveMarketContext:
    """One pinned answer to "which database and provider is production?"."""

    layout: str
    database: Path
    provider: Path
    descriptor: Path
    resolution_source: str
    pointer: Path
    pointer_digest: str | None = None
    pointer_payload: dict = field(default_factory=dict)
    release_id: str | None = None
    generation_id: str | None = None
    revision: str | None = None
    pointer_revision: str | None = None
    provider_version: str | None = None
    generation_root: Path | None = None
    update_state: str | None = None
    resolved_at: str = field(default_factory=_now)

    @property
    def is_bootstrap(self) -> bool:
        return self.resolution_source == BOOTSTRAP_SOURCE

    @property
    def is_active_pointer(self) -> bool:
        return self.resolution_source == POINTER_SOURCE

    @property
    def is_explicit(self) -> bool:
        return self.resolution_source == EXPLICIT_SOURCE

    def as_descriptor(self) -> dict:
        """A descriptor dict accepted by :class:`quant_project.storage.MarketStore`."""
        return {"layout": self.layout, "database": str(self.database),
                "data_revision": self.revision, "provider": str(self.provider)}

    def manifest(self) -> dict:
        """The JSON-safe record every run result and manifest must carry."""
        return {
            "resolutionSource": self.resolution_source,
            "layout": self.layout,
            "database": str(self.database),
            "provider": str(self.provider),
            "descriptor": str(self.descriptor),
            "releaseId": self.release_id,
            "generationId": self.generation_id,
            "revision": self.revision,
            "pointerRevision": self.pointer_revision,
            "providerVersion": self.provider_version,
            "pointer": str(self.pointer),
            "pointerDigest": self.pointer_digest,
            "isBootstrap": self.is_bootstrap,
            "updateState": self.update_state,
            "resolvedAt": self.resolved_at,
        }


def _fail(message, pointer=None, **details) -> "MarketContextError":
    payload = {"pointer": str(pointer if pointer is not None else pointer_path()), **details}
    return MarketContextError(f"{message} ({json.dumps(payload, ensure_ascii=False, sort_keys=True)})")


def _check_generation(root: Path, descriptor: dict, revision: str | None,
                      payload: dict | None = None) -> dict:
    """Fail closed when an active generation's database and provider disagree.

    A generation is written by two steps (database update, then provider
    rebuild).  ``current_state.json`` is marked ``pending`` before the first
    step and ``consistent`` after the second, so a crash in between is visible
    to every later reader instead of serving a half-updated pair.

    Two different revisions are in play and they are deliberately not compared
    with each other:

    * the *activated* revision -- frozen when the operator switched, recorded in
      the pointer and in the generation's ``activated_revision``;
    * the *current* revision -- moved by every legitimate intraday overlay write
      and close update, tracked by ``current_state.json``, the provider
      descriptor and the database together.

    Requiring the pointer's revision to equal the live one would fail closed
    after an ordinary intraday write, which is exactly what an active generation
    is allowed to receive.
    """
    root = Path(root)
    manifest_path = root / ACTIVATION_MANIFEST
    state_path = root / CURRENT_STATE
    if not manifest_path.is_file():
        if payload is not None:
            # The pointer names a target whose activation record is gone: the
            # release identity it claims can no longer be verified.
            raise _fail("active pointer does not point at a complete generation",
                        generation=str(root), manifest=str(manifest_path))
        # A plain staging database is not an active generation.
        return {}
    if not state_path.is_file():
        raise _fail("active generation is missing current_state.json", generation=str(root))
    try:
        state = read_json(state_path)
    except (OSError, ValueError) as error:
        raise _fail(f"active generation state is unreadable: {error}", generation=str(root))
    if state.get("schema") != "stock-db3-current-state-v1":
        raise _fail("active generation state has an unexpected schema", generation=str(root))
    update_state = state.get("update_state")
    if update_state != CONSISTENT:
        raise _fail("active generation is not in a consistent state",
                    generation=str(root), updateState=update_state,
                    pendingOperation=state.get("pending_operation"))
    activated = state.get("activated_revision") or state.get("data_revision")
    if payload is not None:
        if payload.get("release_id") and state.get("release_id") \
                and payload["release_id"] != state["release_id"]:
            raise _fail("active pointer release id disagrees with the generation",
                        generation=str(root), pointerRelease=payload.get("release_id"),
                        generationRelease=state.get("release_id"))
        if payload.get("revision") and activated and payload["revision"] != activated:
            raise _fail("active pointer revision is not the generation's activated revision",
                        generation=str(root), pointerRevision=payload.get("revision"),
                        activatedRevision=activated)
    recorded = state.get("data_revision")
    provider_revision = descriptor.get("data_revision")
    if recorded != provider_revision:
        raise _fail("active generation provider does not match its recorded revision",
                    generation=str(root), stateRevision=recorded, providerRevision=provider_revision)
    if revision is not None and recorded != revision:
        raise _fail("active generation database revision is ahead of its provider",
                    generation=str(root), databaseRevision=revision, providerRevision=recorded)
    generation_id = state.get("generation_id") or Path(root).name
    return {"generationRoot": str(root), "generationId": generation_id, "updateState": update_state,
            "releaseId": state.get("release_id"), "dataRevision": recorded,
            "activatedRevision": activated, "pointerPayload": state}


def _database_revision(database: Path) -> str | None:
    from .market import connect

    with connect(database, read_only=True) as db:
        row = db.execute("SELECT key,value FROM store_metadata").fetchall()
    metadata = {key: value for key, value in row}
    return metadata.get("overlay_revision") or metadata.get("data_revision")


def _layout_of(database: Path) -> str:
    from .market import connect

    with connect(database, read_only=True) as db:
        tables = {row[0] for row in db.execute("SHOW TABLES").fetchall()}
    return V2_LAYOUT if {"daily_final", "daily_intraday_overlay", "minute5_final"} <= tables else V1_LAYOUT


def _bootstrap_context() -> ActiveMarketContext:
    database = bootstrap_database()
    provider = bootstrap_provider()
    layout = _layout_of(database) if database.is_file() else V1_LAYOUT
    return ActiveMarketContext(
        layout=layout, database=database, provider=provider, descriptor=provider / "descriptor.json",
        resolution_source=BOOTSTRAP_SOURCE, pointer=pointer_path(), pointer_digest=None,
        revision=_database_revision(database) if database.is_file() else None,
    )


def resolve_active_market(pointer=None, *, verify: bool = True) -> ActiveMarketContext:
    """Resolve the active market once.  Raises instead of falling back to v1."""
    path = pointer_path(pointer)
    if not path.is_file():
        if pointer is not None or os.environ.get(ENV_POINTER):
            raise _fail("active pointer does not exist", pointer=path)
        return _bootstrap_context()
    digest = sha256_file(path)
    try:
        payload = read_json(path)
    except (OSError, ValueError) as error:
        raise _fail(f"active pointer is unreadable: {error}", pointer=path, digest=digest)
    if not isinstance(payload, dict) or payload.get("schema") not in POINTER_SCHEMAS:
        raise _fail("active pointer schema is not recognised", pointer=path, digest=digest,
                    schema=(payload or {}).get("schema") if isinstance(payload, dict) else None)
    for key in ("database", "provider", "release_root"):
        if not payload.get(key):
            raise _fail(f"active pointer is missing {key}", pointer=path, digest=digest)
    database = Path(payload["database"]).expanduser().resolve()
    provider = _provider_root(payload["provider"])
    descriptor = Path(payload.get("descriptor") or (provider / "descriptor.json")).expanduser().resolve()
    if not database.is_file():
        raise _fail("active pointer target database is missing", pointer=path, digest=digest,
                    database=str(database))
    if not descriptor.is_file():
        raise _fail("active pointer target provider descriptor is missing", pointer=path, digest=digest,
                    descriptor=str(descriptor))
    try:
        meta = read_json(descriptor)
    except (OSError, ValueError) as error:
        raise _fail(f"active provider descriptor is unreadable: {error}", pointer=path, digest=digest)
    if Path(meta.get("database", "")).expanduser().resolve() != database:
        raise _fail("active provider does not cite the pointer's database", pointer=path, digest=digest,
                    database=str(database), providerDatabase=meta.get("database"))
    layout = meta.get("layout")
    if layout != V2_LAYOUT:
        raise _fail("active pointer must reference a market-database-v2 provider", pointer=path,
                    digest=digest, layout=layout)
    generation_root = Path(payload.get("release_root") or database.parent).expanduser().resolve()
    live_revision = _database_revision(database) if verify else None
    extra = {}
    if verify:
        # The comparison that matters is between the *live* pair -- database,
        # provider descriptor and current_state -- so that is what is read here.
        # The pointer's own revision is the frozen activation record, checked
        # against the generation instead of against the moving revision.
        extra = _check_generation(generation_root, meta, live_revision, payload)
    revision = extra.get("dataRevision") or live_revision or payload.get("revision") or meta.get("data_revision")
    return ActiveMarketContext(
        layout=layout, database=database, provider=provider, descriptor=descriptor,
        resolution_source=POINTER_SOURCE, pointer=path, pointer_digest=digest, pointer_payload=payload,
        release_id=payload.get("release_id") or extra.get("releaseId"),
        generation_id=extra.get("generationId") or generation_root.name,
        revision=revision, pointer_revision=payload.get("revision"),
        provider_version=meta.get("provider_version"),
        generation_root=generation_root if extra else None,
        update_state=extra.get("updateState"),
    )


def explicit_market_context(database, provider=None, *, verify: bool = False) -> ActiveMarketContext:
    """Bind an explicit database/provider pair for tests, research and migration."""
    from .market import database_path

    path = database_path(database)
    root = _provider_root(provider) if provider is not None else path.parent / "provider"
    layout = _layout_of(path) if path.is_file() else V1_LAYOUT
    if verify:
        revision = _database_revision(path)
        meta = read_json(root / "descriptor.json") if (root / "descriptor.json").is_file() else {}
        extra = _check_generation(generation_root_for(path) or path.parent, meta, revision)
    else:
        extra = {}
    return ActiveMarketContext(
        layout=layout, database=path, provider=root, descriptor=root / "descriptor.json",
        resolution_source=EXPLICIT_SOURCE, pointer=pointer_path(),
        generation_root=Path(path.parent) if extra else None, update_state=extra.get("updateState"),
    )


def _configured_path(value) -> Path:
    """Interpret a path written in a config file the way the repo means it.

    Configs are relative to the workspace root (``quant_project/data/market.duckdb``
    from ``stock-workspace``), not to whatever directory a command happens to be
    started from, so a relative literal is resolved against the workspace first
    and only then against the working directory.
    """
    return resolve_workspace_path(value)


def resolve_configured_database(value=None) -> Path:
    """Resolve a *configured* database reference through the active market.

    Workflows and config files name the market database as the literal path
    ``quant_project/data/market.duckdb``.  That name is a symbol for "the market
    database", not a research pin: once a DB3 pointer exists, the file it names
    is the bootstrap v1 copy the switch moved away from, so resolving it
    literally would silently read an abandoned database.  ``None``, ``market``,
    ``history``, ``active`` and the bootstrap path itself therefore all resolve
    to the pinned active database; every other path stays explicit for tests,
    research snapshots and migration tools.
    """
    from .market import database_path

    if value in (None, "market", "history", "active"):
        return active_market_context().database
    path = database_path(_configured_path(value))
    return active_market_context().database if path == bootstrap_database() else path


def resolve_configured_provider(value=None) -> Path:
    """The provider counterpart of :func:`resolve_configured_database`."""
    if value in (None, "market", "history", "active"):
        return active_market_context().provider
    path = _provider_root(_configured_path(value))
    return active_market_context().provider if path == bootstrap_provider() else path


_PINNED: ActiveMarketContext | None = None


def pin_active_market(pointer=None, **kwargs) -> ActiveMarketContext:
    """Resolve once and reuse the answer for the rest of the command."""
    global _PINNED
    if _PINNED is None or pointer is not None or kwargs:
        _PINNED = resolve_active_market(pointer, **kwargs)
    return _PINNED


def pinned_active_market() -> ActiveMarketContext | None:
    return _PINNED


def active_market_context() -> ActiveMarketContext:
    """The pinned context, resolved on first use.

    A command pins its generation at start-up; a later pointer switch therefore
    cannot make one command mix two releases.
    """
    return _PINNED if _PINNED is not None else pin_active_market()


def unpin_active_market() -> None:
    global _PINNED
    _PINNED = None


def generation_root_for(database) -> Path | None:
    """The generation directory that owns ``database``, when it is one.

    A generation keeps its database next to its provider (``<root>/database/``,
    ``<root>/provider/``), so the manifest is one or two levels up from the
    database file.  Both shapes are checked: a staging database may also sit
    directly in its own directory.
    """
    from .market import database_path

    path = database_path(database)
    for candidate in (path.parent, path.parent.parent):
        if (candidate / ACTIVATION_MANIFEST).is_file():
            return candidate
    return None


def current_state(database_or_root) -> dict | None:
    root = generation_root_for(database_or_root) or Path(database_or_root)
    path = Path(root) / CURRENT_STATE
    return read_json(path) if path.is_file() else None


def mark_generation_pending(database, operation: str) -> dict | None:
    """Declare that the generation's database is about to move past its provider."""
    root = generation_root_for(database)
    if root is None:
        return None
    state = current_state(root) or {"schema": "stock-db3-current-state-v1",
                                    "generation_id": root.name,
                                    "data_revision": None, "provider_revision": None}
    state.update(update_state=PENDING, pending_operation=operation, pending_since=_now())
    write_json_atomic(root / CURRENT_STATE, state)
    return state


def mark_generation_consistent(database, *, provider=None, operation: str,
                               provider_version: str | None = None) -> dict | None:
    """Declare the database and provider consistent again after a rebuild."""
    from .market import database_path

    root = generation_root_for(database)
    if root is None:
        return None
    path = database_path(database)
    state = current_state(root) or {"schema": "stock-db3-current-state-v1", "generation_id": root.name}
    revision = _database_revision(path)
    meta = {}
    descriptor = (Path(provider) / "descriptor.json") if provider is not None else root / "provider/descriptor.json"
    if descriptor.is_file():
        meta = read_json(descriptor)
    state.update(update_state=CONSISTENT, data_revision=revision,
                 provider_revision=meta.get("data_revision", revision),
                 provider_version=provider_version or meta.get("provider_version"),
                 provider_manifest_sha256=sha256_tree(root / "provider") if (root / "provider").is_dir() else None,
                 last_successful_update={"operation": operation, "at": _now(), "revision": revision},
                 pending_operation=None, pending_since=None, updated_at=_now())
    write_json_atomic(root / CURRENT_STATE, state)
    return state
