"""Immutable research releases for reproducible experiments.

An operating market generation is mutable by design.  A research release is a
separate database/provider copy with a sealed manifest, explicit point-in-time
capabilities, known biases, and content hashes.  No experiment may turn a
mutable operational path into a release merely by naming a version string.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .active_market import read_json, sha256_file, sha256_tree
from .common import atomic_json
from .pit.capabilities import derive_release_capabilities
from .pit.schema import initialize_pit_schema

SCHEMA = "quant-project-research-release-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ResearchReleaseError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _provider_root(value):
    path = Path(value).expanduser().resolve()
    return path.parent if path.name == "descriptor.json" else path


def _canonical_hash(value):
    import hashlib

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _release_file_manifest(root, folder, schema):
    directory = Path(root) / folder
    entries = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        entries.append({"path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    body = {"schema": schema, "files": entries}
    digest = _canonical_hash(body)
    atomic_json(Path(root) / "manifests" / f"{folder.split('/')[0]}-manifest.json",
                body | {"manifest_sha256": digest})
    return digest


def _copy_pit_artifacts(source_root, target_root):
    source = Path(source_root).expanduser().resolve() if source_root else None
    if source is not None and source.is_dir():
        for relative in ("raw/original", "normalized/fundamentals", "normalized/industries",
                         "manifests/batches"):
            from_path, to_path = source / relative, Path(target_root) / relative
            if from_path.is_dir():
                shutil.copytree(from_path, to_path, dirs_exist_ok=True)
    (Path(target_root) / "raw/original").mkdir(parents=True, exist_ok=True)
    (Path(target_root) / "normalized/fundamentals").mkdir(parents=True, exist_ok=True)
    (Path(target_root) / "normalized/industries").mkdir(parents=True, exist_ok=True)
    (Path(target_root) / "manifests/batches").mkdir(parents=True, exist_ok=True)


def build_research_release(*, database, provider, output_root, release_id,
                           point_in_time=False, known_biases=None,
                           reference_version=None, corporate_action_version=None,
                           calendar_version=None, artifacts_root=None) -> dict:
    """Freeze a database/provider pair; never overwrite an existing release."""
    if not SAFE_ID.fullmatch(str(release_id)):
        raise ResearchReleaseError(f"research release id 无效：{release_id!r}")
    source_database = Path(database).expanduser().resolve()
    source_provider = _provider_root(provider)
    descriptor_path = source_provider / "descriptor.json"
    if not source_database.is_file() or not descriptor_path.is_file():
        raise ResearchReleaseError("research release 需要完整的 database/provider pair")
    descriptor = read_json(descriptor_path)
    if Path(descriptor.get("database", "")).expanduser().resolve() != source_database:
        raise ResearchReleaseError("provider descriptor 没有指向所请求的数据库")
    root = Path(output_root).expanduser().resolve()
    target = root / release_id
    if target.exists():
        raise ResearchReleaseError(f"research release 已存在，不覆盖：{target}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{release_id}.prepare-{uuid4().hex}"
    try:
        (temporary / "database").mkdir(parents=True)
        shutil.copy2(source_database, temporary / "database/research.duckdb")
        shutil.copytree(source_provider, temporary / "provider/market-data")
        frozen_database = target / "database/research.duckdb"
        frozen_provider = target / "provider"
        qlib_descriptor_path = temporary / "provider/market-data/descriptor.json"
        qlib_descriptor = read_json(qlib_descriptor_path)
        qlib_descriptor.update({
            "database": str(frozen_database.resolve()), "profile": "research-release",
            "research_release": release_id,
            "source_state": {"state": "immutable_research_release"},
        })
        qlib_descriptor_path.write_text(
            json.dumps(qlib_descriptor, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _copy_pit_artifacts(artifacts_root, temporary)
        # Installing schema on this new copy cannot mutate the source/active DB.
        try:
            initialize_pit_schema(temporary / "database/research.duckdb")
        except Exception as error:
            # Older release unit fixtures intentionally use a non-DuckDB byte
            # file; retain that compatibility without declaring PIT capability.
            if "not a valid DuckDB database" not in str(error) and "not a valid DuckDB" not in str(error):
                try:
                    import duckdb
                    probe = duckdb.connect(str(temporary / "database/research.duckdb"), read_only=True)
                    probe.close()
                except Exception:
                    pass
                else:
                    raise
        raw_manifest_hash = _release_file_manifest(
            temporary, "raw/original", "quant-project-release-raw-manifest-v1")
        normalized_manifest_hash = _release_file_manifest(
            temporary, "normalized", "quant-project-release-normalized-manifest-v1")
        derived = derive_release_capabilities(
            temporary / "database/research.duckdb", artifacts_root=temporary,
            source_descriptor=qlib_descriptor)
        capabilities = derived["capabilities"]
        aggregate_pit = all(capabilities[name]["point_in_time"]
                            for name in ("market_data", "fundamentals", "industry"))
        versions = {
            "reference": reference_version,
            "corporate_actions": corporate_action_version,
            "calendar": calendar_version,
            **derived["versions"],
        }
        biases = sorted(set(list(known_biases or []) + derived["known_biases"]))
        provider_descriptor = {
            "schema": "quant-project-research-provider-v1",
            "research_release_id": release_id,
            "database": str(frozen_database.resolve()),
            "dataset_versions": versions,
            "capabilities": capabilities,
            "manifest_hashes": {"raw": raw_manifest_hash, "normalized": normalized_manifest_hash},
            "coverage": derived["coverage"],
        }
        atomic_json(temporary / "provider/descriptor.json", provider_descriptor)
        quality_report = {"schema": "quant-project-pit-quality-report-v1",
                          "status": "no_historical_source" if not any(
                              capabilities[name]["point_in_time"] for name in ("fundamentals", "industry"))
                          else "verified",
                          "issues": [],
                          "reason": (None if any(capabilities[name]["point_in_time"]
                                                 for name in ("fundamentals", "industry")) else
                                     "trusted historical announcement/effective-date source unavailable")}
        atomic_json(temporary / "manifests/quality-report.json", quality_report)
        manifest = {
            "schema": SCHEMA,
            "release_id": release_id,
            "sealed": True,
            "created_at": _now(),
            "database": str(frozen_database.resolve()),
            "provider": str(frozen_provider.resolve()),
            "layout": qlib_descriptor.get("layout"),
            "data_revision": qlib_descriptor.get("data_revision")
            or qlib_descriptor.get("baseline_id"),
            "provider_version": qlib_descriptor.get("provider_version"),
            "database_sha256": sha256_file(temporary / "database/research.duckdb"),
            "provider_sha256": sha256_tree(temporary / "provider"),
            "raw_manifest_sha256": raw_manifest_hash,
            "normalized_manifest_sha256": normalized_manifest_hash,
            "source": {
                "database": str(source_database),
                "provider": str(source_provider),
                "database_sha256": sha256_file(source_database),
                "provider_sha256": sha256_tree(source_provider),
            },
            "versions": versions,
            "capabilities": {**capabilities, "point_in_time": aggregate_pit},
            "sources": derived["sources"],
            "coverage": derived["coverage"],
            "known_biases": biases,
        }
        atomic_json(temporary / "release.json", manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return manifest | {"root": str(target), "manifest": str(target / "release.json")}


def verify_research_release(release_root) -> dict:
    root = Path(release_root).expanduser().resolve()
    manifest_path = root / "release.json"
    if not manifest_path.is_file():
        raise ResearchReleaseError(f"research release 缺少 manifest：{manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != SCHEMA or not manifest.get("sealed"):
        raise ResearchReleaseError(f"research release schema/state 无效：{manifest_path}")
    database, provider = root / "database/research.duckdb", root / "provider"
    if not database.is_file() or not (provider / "descriptor.json").is_file():
        raise ResearchReleaseError(f"research release 文件不完整：{root}")
    if sha256_file(database) != manifest.get("database_sha256"):
        raise ResearchReleaseError(f"research release 数据库已被改动：{root}")
    if sha256_tree(provider) != manifest.get("provider_sha256"):
        raise ResearchReleaseError(f"research release provider 已被改动：{root}")
    raw_manifest = root / "manifests/raw-manifest.json"
    normalized_manifest = root / "manifests/normalized-manifest.json"
    if not _verify_release_file_manifest(root, raw_manifest):
        raise ResearchReleaseError(f"research release raw manifest 缺失或无效：{root}")
    if not _verify_release_file_manifest(root, normalized_manifest):
        raise ResearchReleaseError(f"research release normalized manifest 缺失或无效：{root}")
    if _manifest_body_hash(raw_manifest) != manifest.get("raw_manifest_sha256"):
        raise ResearchReleaseError(f"research release raw manifest 已被改动：{root}")
    if _manifest_body_hash(normalized_manifest) != manifest.get("normalized_manifest_sha256"):
        raise ResearchReleaseError(f"research release normalized manifest 已被改动：{root}")
    descriptor = read_json(provider / "descriptor.json")
    if Path(descriptor.get("database", "")).expanduser().resolve() != database:
        raise ResearchReleaseError(f"research release provider/database 不匹配：{root}")
    if descriptor.get("research_release_id") != manifest.get("release_id"):
        raise ResearchReleaseError(f"research release provider id 不匹配：{root}")
    derived = derive_release_capabilities(database, artifacts_root=root,
                                          source_descriptor=read_json(provider / "market-data/descriptor.json"))
    capabilities = {**derived["capabilities"],
                    "point_in_time": all(derived["capabilities"][name]["point_in_time"]
                                         for name in ("market_data", "fundamentals", "industry"))}
    if capabilities != manifest.get("capabilities"):
        raise ResearchReleaseError(f"research release capability does not match its data/manifests: {root}")
    return manifest | {"root": str(root), "status": "verified"}


def _manifest_body_hash(path):
    payload = read_json(path)
    claimed = payload.pop("manifest_sha256", None)
    return _canonical_hash(payload) if claimed else None


def _verify_release_file_manifest(root, path):
    if not Path(path).is_file():
        return False
    manifest = read_json(path)
    folder = "raw/original" if Path(path).name == "raw-manifest.json" else "normalized"
    directory = Path(root) / folder
    actual = sorted(item.relative_to(root).as_posix()
                    for item in directory.rglob("*") if item.is_file()) if directory.is_dir() else []
    entries = manifest.get("files", [])
    declared = sorted(str(entry.get("path", "")) for entry in entries)
    if declared != actual:
        return False
    for entry in entries:
        candidate = Path(root) / Path(entry.get("path", ""))
        if (not candidate.is_file() or sha256_file(candidate) != entry.get("sha256")
                or candidate.stat().st_size != entry.get("size_bytes")):
            return False
    return _manifest_body_hash(path) == manifest.get("manifest_sha256")


def list_research_releases(output_root) -> list[dict]:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        return []
    result = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        try:
            release = verify_research_release(path)
            result.append({"release_id": release["release_id"], "root": release["root"],
                           "status": release["status"],
                           "point_in_time": release["capabilities"]["point_in_time"],
                           "dataset_capabilities": {name: release["capabilities"][name]
                                                     for name in ("market_data", "fundamentals", "industry")},
                           "data_revision": release.get("data_revision")})
        except ResearchReleaseError as error:
            result.append({"release_id": path.name, "root": str(path),
                           "status": "invalid", "error": str(error)})
    return result
