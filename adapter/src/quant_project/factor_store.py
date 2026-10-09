"""Content-addressed, immutable FactorScore datasets."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .active_market import read_json, sha256_file
from .common import atomic_json

SCHEMA = "quant-project-factor-dataset-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class FactorStoreError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _normalized_rows(rows, factor_id):
    result, keys = [], set()
    for value in rows:
        timestamp, symbol = str(value["timestamp"]), str(value["symbol"])
        key = (timestamp, symbol)
        if key in keys:
            raise FactorStoreError(f"FactorScore 键重复：{timestamp}|{symbol}")
        keys.add(key)
        score = float(value["score"])
        if not math.isfinite(score):
            raise FactorStoreError(f"FactorScore 不是有限数：{timestamp}|{symbol}")
        source = str(value.get("source_factor_id") or factor_id)
        if source != factor_id:
            raise FactorStoreError(f"FactorScore source_factor_id 不一致：{source}")
        result.append({"timestamp": timestamp, "symbol": symbol, "score": score,
                       "source_factor_id": factor_id})
    return sorted(result, key=lambda item: (item["timestamp"], item["symbol"]))


def publish_factor_dataset(*, dataset_id, factor_id, rows, output_root, context,
                           research_release=None, snapshot_compatible=False) -> dict:
    if not SAFE_ID.fullmatch(str(dataset_id)):
        raise FactorStoreError(f"Factor dataset id 无效：{dataset_id!r}")
    if not research_release and not snapshot_compatible:
        raise FactorStoreError("Factor dataset 需要 research_release 或 snapshot_compatible=true")
    release_identity = None
    if research_release:
        from .research_release import ResearchReleaseError, verify_research_release

        try:
            release = verify_research_release(research_release)
        except ResearchReleaseError as error:
            raise FactorStoreError(f"Factor dataset 的 research release 无效：{error}") from error
        release_identity = {"release_id": release["release_id"], "root": release["root"],
                            "database_sha256": release["database_sha256"],
                            "provider_sha256": release["provider_sha256"],
                            "point_in_time": release["capabilities"]["point_in_time"]}
    normalized = _normalized_rows(rows, factor_id)
    root = Path(output_root).expanduser().resolve()
    target = root / dataset_id
    if target.exists():
        raise FactorStoreError(f"Factor dataset 已存在，不覆盖：{target}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{dataset_id}.prepare-{uuid4().hex}"
    try:
        temporary.mkdir()
        score_path = temporary / "scores.jsonl"
        score_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                                 separators=(",", ":")) + "\n"
                                      for row in normalized), encoding="utf-8")
        timestamps = [row["timestamp"] for row in normalized]
        manifest = {
            "schema": SCHEMA,
            "dataset_id": dataset_id,
            "factor_id": factor_id,
            "sealed": True,
            "created_at": _now(),
            "research_release": release_identity,
            "snapshot_compatible": bool(snapshot_compatible),
            "context": dict(context),
            "rows": len(normalized),
            "symbols": len({row["symbol"] for row in normalized}),
            "first_timestamp": min(timestamps) if timestamps else None,
            "last_timestamp": max(timestamps) if timestamps else None,
            "artifacts": {"scores.jsonl": {"sha256": sha256_file(score_path),
                                             "bytes": score_path.stat().st_size}},
            "content_sha256": sha256_file(score_path),
        }
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return manifest | {"root": str(target), "manifest": str(target / "manifest.json")}


def verify_factor_dataset(dataset_root) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FactorStoreError(f"Factor dataset 缺少 manifest：{root}")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != SCHEMA or not manifest.get("sealed"):
        raise FactorStoreError(f"Factor dataset schema/state 无效：{root}")
    for name, recorded in (manifest.get("artifacts") or {}).items():
        path = root / name
        if not path.is_file() or path.stat().st_size != recorded.get("bytes") \
                or sha256_file(path) != recorded.get("sha256"):
            raise FactorStoreError(f"Factor dataset artifact 已缺失或改动：{name}")
    score_hash = (manifest.get("artifacts") or {}).get("scores.jsonl", {}).get("sha256")
    if not score_hash or manifest.get("content_sha256") != score_hash:
        raise FactorStoreError(f"Factor dataset content hash 无效：{root}")
    return manifest | {"root": str(root), "status": "verified"}


def load_factor_scores(dataset_root) -> list[dict]:
    verified = verify_factor_dataset(dataset_root)
    path = Path(verified["root"]) / "scores.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def list_factor_datasets(output_root) -> list[dict]:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        return []
    result = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        try:
            item = verify_factor_dataset(path)
            result.append({key: item.get(key) for key in
                           ("dataset_id", "factor_id", "rows", "first_timestamp",
                            "last_timestamp", "snapshot_compatible", "status")})
        except FactorStoreError as error:
            result.append({"dataset_id": path.name, "status": "invalid", "error": str(error)})
    return result
