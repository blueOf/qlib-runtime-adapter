"""Sealed model scores with stable row identities and input lineage."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .common import atomic_json, read_json, sha256
from .modeling.contracts import ArtifactError, DataContractError, sha256_json

SCHEMA = "quant-project-model-score-dataset-v1"


def validate_model_scores(rows):
    ids, keys = set(), set()
    for row in rows:
        key = (row["source_model_id"], row["timestamp"], row["symbol"])
        if key in keys or row["model_score_id"] in ids:
            raise DataContractError("duplicate ModelScore row identity")
        expected_id = sha256_json({name: row[name] for name in
                                  ("research_run_id", "fold_id", "timestamp", "symbol", "model_config_sha256")})
        if expected_id != row["model_score_id"] or not math.isfinite(float(row["score"])):
            raise DataContractError("ModelScore identity or value is invalid")
        if not all(row.get(name) for name in ("source_factor_set_id", "factor_set_config_sha256",
                                              "artifact_manifest_sha256", "source_factor_vector_id")):
            raise DataContractError("ModelScore requires complete FactorVector/model/artifact lineage")
        ids.add(row["model_score_id"])
        keys.add(key)


def publish_model_scores(directory, rows, *, run_id, compiled_plan_sha256):
    directory = Path(directory).resolve()
    rows = list(rows)
    validate_model_scores(rows)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "scores.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    manifest = {"schema": SCHEMA, "sealed": True, "run_id": run_id,
                "compiled_plan_sha256": compiled_plan_sha256, "rows": len(rows),
                "scores_file": "scores.jsonl", "scores_sha256": sha256(path)}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def verify_model_scores(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    path = directory / "scores.jsonl"
    if manifest.get("schema") != SCHEMA or manifest.get("sealed") is not True \
            or manifest.get("scores_file") != "scores.jsonl" or not path.is_file() \
            or sha256(path) != manifest.get("scores_sha256"):
        raise ArtifactError("model score dataset is missing or changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != manifest["rows"]:
        raise ArtifactError("model score dataset row count changed")
    validate_model_scores(rows)
    return manifest | {"status": "verified", "root": str(directory)}


def load_model_scores(directory):
    verify_model_scores(directory)
    return [json.loads(line) for line in (Path(directory) / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line]
