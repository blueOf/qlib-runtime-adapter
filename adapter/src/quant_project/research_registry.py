"""Atomic registry for completed, reproducible research runs."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

from .common import atomic_json, read_json, sha256, utc_now

SCHEMA = "quant-project-research-run-registry-v1"


class ResearchRegistryError(RuntimeError):
    pass


@contextmanager
def _registry_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ResearchRegistryError(f"Research Registry 正在由另一个进程更新：{path}") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at": utc_now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _artifact_record(run_root: Path, relative: str, role: str) -> dict:
    path = (run_root / relative).resolve()
    if not _within(path, run_root) or not path.is_file():
        raise ResearchRegistryError(f"Research run artifact 缺失或越界：{relative}")
    return {"path": relative, "role": role, "sha256": sha256(path),
            "bytes": path.stat().st_size}


def register_research_run(run_root, reports_root) -> dict:
    """Register one completed run without treating its output as the registry itself."""
    run_root = Path(run_root).expanduser().resolve()
    reports_root = Path(reports_root).expanduser().resolve()
    if not _within(run_root, reports_root):
        raise ResearchRegistryError(f"Research run 不在 reports root 中：{run_root}")
    summary_path = run_root / "summary.json"
    if not summary_path.is_file():
        raise ResearchRegistryError(f"Research run 缺少 summary.json：{run_root}")
    summary = read_json(summary_path)
    if summary.get("status") != "success":
        raise ResearchRegistryError("只有成功完成的 research run 可以注册")
    experiment_id, run_id = summary.get("experiment_id"), summary.get("run_id")
    if not experiment_id or not run_id:
        raise ResearchRegistryError("Research run summary 缺少 experiment_id/run_id")
    expected = (reports_root / str(experiment_id) / str(run_id)).resolve()
    if run_root != expected:
        raise ResearchRegistryError(f"Research run 路径与标识不一致：{run_root}")

    resolved_name = "resolved_config.yaml" if (run_root / "resolved_config.yaml").is_file() else "resolved_config.json"
    named = {"summary.json": "summary", resolved_name: "resolved_config"}
    for relative in (summary.get("reports") or {}).values():
        named.setdefault(str(relative), "report")
    for relative in (summary.get("run_evidence") or {}).values():
        role = ("model_score" if str(relative).startswith("model_scores/") else
                "model_artifact" if str(relative).startswith("model_artifacts/") else "evidence")
        named.setdefault(str(relative), role)
    artifacts = [_artifact_record(run_root, relative, named[relative])
                 for relative in sorted(named)]
    entry = {
        "key": f"{experiment_id}/{run_id}",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "stage": summary.get("stage"),
        "status": summary["status"],
        "registered_at": utc_now(),
        "run_root": run_root.relative_to(reports_root).as_posix(),
        "context_snapshot": summary.get("context_snapshot"),
        "artifacts": artifacts,
    }
    if isinstance(summary.get("experiment_name"), str) and summary["experiment_name"].strip():
        entry["experiment_name"] = summary["experiment_name"]
    for field in ("compiled_plan_sha256", "execution_status", "retention_class"):
        if field in summary:
            entry[field] = summary[field]
    registry_path = reports_root / "registry.json"
    with _registry_lock(reports_root / ".registry.lock"):
        if registry_path.is_file():
            registry = read_json(registry_path)
            if registry.get("schema") != SCHEMA or not isinstance(registry.get("entries"), list):
                raise ResearchRegistryError(f"Research Registry schema 无效：{registry_path}")
        else:
            registry = {"schema": SCHEMA, "entries": []}
        if any(item.get("key") == entry["key"] for item in registry["entries"]):
            raise ResearchRegistryError(f"Research run 已注册，不覆盖：{entry['key']}")
        registry["entries"].append(entry)
        registry["entries"].sort(key=lambda item: item["key"])
        registry["updated_at"] = utc_now()
        atomic_json(registry_path, registry)
    return entry | {"registry": str(registry_path)}


def verify_research_registry(registry_path, *, run_key=None) -> dict:
    registry_path = Path(registry_path).expanduser().resolve()
    if not registry_path.is_file():
        raise ResearchRegistryError(f"Research Registry 不存在：{registry_path}")
    registry = read_json(registry_path)
    entries = registry.get("entries")
    if registry.get("schema") != SCHEMA or not isinstance(entries, list):
        raise ResearchRegistryError(f"Research Registry schema 无效：{registry_path}")
    if run_key is not None:
        entries = [entry for entry in entries if entry.get("key") == run_key]
        if len(entries) != 1:
            raise ResearchRegistryError(f"Research Registry run key 缺失或重复：{run_key}")
    root, keys = registry_path.parent, set()
    for entry in entries:
        key = entry.get("key")
        if not key or key in keys:
            raise ResearchRegistryError(f"Research Registry key 缺失或重复：{key}")
        keys.add(key)
        run_root = (root / str(entry.get("run_root", ""))).resolve()
        if not _within(run_root, root) or run_root != (root / key).resolve():
            raise ResearchRegistryError(f"Research Registry run_root 无效：{key}")
        exempt = set()
        if entry.get("retention_tombstone"):
            tombstone = entry["retention_tombstone"]
            tombstone_path = (root / str(tombstone.get("path", ""))).resolve()
            if not _within(tombstone_path, root) or not tombstone_path.is_file() or sha256(tombstone_path) != tombstone.get("sha256"):
                raise ResearchRegistryError(f"Retention tombstone 缺失或改动：{key}")
            state = read_json(tombstone_path)
            exempt = {item["path"] for item in state.get("unavailable_artifacts", [])}
            if (state.get("run_key") != key or entry.get("execution_status") != "reference_only"
                    or entry.get("retention_class") != "reference_only"
                    or exempt != set(entry.get("unavailable_artifacts", []))
                    or state.get("unavailable_external_artifacts", []) != entry.get("unavailable_external_artifacts", [])):
                raise ResearchRegistryError(f"Retention downgrade 状态不一致：{key}")
        for artifact in entry.get("artifacts") or []:
            if artifact.get("path") in exempt:
                continue
            path = (run_root / str(artifact.get("path", ""))).resolve()
            if (not _within(path, run_root) or not path.is_file()
                    or path.stat().st_size != artifact.get("bytes")
                    or sha256(path) != artifact.get("sha256")):
                raise ResearchRegistryError(f"Research run artifact 已缺失或改动：{key}/{artifact.get('path')}")
    return {"schema": SCHEMA, "registry": str(registry_path),
            "entries": len(entries), "status": "verified"}


def list_research_runs(registry_path) -> list[dict]:
    registry_path = Path(registry_path).expanduser().resolve()
    if not registry_path.is_file():
        return []
    verify_research_registry(registry_path)
    return [{name: entry.get(name) for name in
             ("key", "experiment_id", "experiment_name", "run_id", "stage", "status", "registered_at",
              "execution_status", "retention_class", "retention_tombstone")
             if entry.get(name) is not None}
            for entry in read_json(registry_path)["entries"]]
