"""Read-only integrity and restoration audits for generic model runs."""
from __future__ import annotations

import math
from pathlib import Path

from ..common import read_json
from ..modeling.artifacts import ArtifactStore
from ..modeling.contracts import ArtifactError, FitContext, ModelInputBatch, ModelSpec, PredictContext, sha256_json
from ..modeling.runner import ModelRunner
from ..model_score_store import load_model_scores
from ..research_registry import ResearchRegistryError, verify_research_registry


def audit_model_run(run_root, *, roundtrip=False):
    run_root = Path(run_root).resolve()
    try:
        plan = read_json(run_root / "compiled_plan.json")
        bundle = read_json(run_root / "model_bundle.json")
        summary = read_json(run_root / "summary.json")
        report = read_json(run_root / "model_report.json")
        vectors = read_json(run_root / "factor_vectors.json")["rows"]
        scores = load_model_scores(run_root / "model_scores")
    except (OSError, ValueError, KeyError) as error:
        raise ArtifactError("model run evidence is unavailable") from error
    try:
        verify_research_registry(run_root.parent.parent / "registry.json",
                                 run_key=f"{summary['experiment_id']}/{summary['run_id']}")
    except (ResearchRegistryError, KeyError) as error:
        raise ArtifactError("registered model run evidence is missing or changed") from error
    entry = next(item for item in read_json(run_root.parent.parent / "registry.json")["entries"]
                 if item["key"] == f"{summary['experiment_id']}/{summary['run_id']}")
    downgraded = entry.get("execution_status") == "reference_only"
    if roundtrip and downgraded:
        raise ArtifactError("reference_only registered run cannot execute a restoration audit")
    claimed = plan["compiled_plan_sha256"]
    if sha256_json({key: value for key, value in plan.items() if key != "compiled_plan_sha256"}) != claimed:
        raise ArtifactError("compiled model plan hash mismatch")
    if any(value.get("compiled_plan_sha256") != claimed for value in (bundle, summary, report)):
        raise ArtifactError("model run artifacts refer to inconsistent compiled plans")
    if sha256_json({key: value for key, value in bundle.items() if key != "bundle_sha256"}) != bundle["bundle_sha256"]:
        raise ArtifactError("model bundle hash mismatch")
    definitions = {item["id"]: item for item in plan["models"]}
    members = []
    member_keys = set()
    for member in bundle["members"]:
        member_key = member["model_id"], member["fold_id"]
        if member_key in member_keys:
            raise ArtifactError("duplicate model bundle member")
        member_keys.add(member_key)
        path = member.get("path")
        if path:
            artifact_root = (run_root / path).resolve()
            if not artifact_root.is_relative_to(run_root / "model_artifacts"):
                raise ArtifactError("bundle member escapes its model artifact root")
        else:
            artifact_root = Path(member["external_root"]).resolve()
        store = ArtifactStore(artifact_root.parent)
        manifest = store.verify(artifact_root)
        if manifest["manifest_sha256"] != member["manifest_sha256"] or manifest["payload_sha256"] != member["payload_sha256"]:
            raise ArtifactError("bundle member hash mismatch")
        retained = [row for row in scores if row["source_model_id"] == member["model_id"] and row["fold_id"] == member["fold_id"]]
        if not retained or any(row["artifact_manifest_sha256"] != member["manifest_sha256"]
                               or row["model_payload_sha256"] != member["payload_sha256"] for row in retained):
            raise ArtifactError("ModelScore references an inconsistent or missing fold artifact")
        if roundtrip:
            definition = dict(definitions[member["model_id"]])
            definition.update(fit_policy="frozen_artifact", artifact={"root": str(artifact_root),
                              "manifest_sha256": member["manifest_sha256"]})
            spec = ModelSpec.from_mapping(definition)
            runner = ModelRunner(artifact_store=store)
            fitted = runner.fit(spec, None, FitContext(run_id="artifact-audit", feature_order=spec.input_factor_ids))
            expected_keys = {(row["timestamp"], row["symbol"]) for row in retained}
            batch = ModelInputBatch.from_rows([row for row in vectors if (row["timestamp"], row["symbol"]) in expected_keys], spec.input_factor_ids)
            restored = runner.predict(fitted, batch, PredictContext(run_id="artifact-audit", fold_id=member["fold_id"]))
            expected = {(row["timestamp"], row["symbol"]): row["score"] for row in retained}
            if len(restored) != len(expected) or any(
                    not math.isclose(row.score, expected[(row.timestamp, row.symbol)], rel_tol=1e-12, abs_tol=1e-12)
                    for row in restored):
                raise ArtifactError("restored scores do not match sealed ModelScore values")
        members.append({"model_id": member["model_id"], "fold_id": member["fold_id"], "status": "verified"})
    if member_keys != {(row["source_model_id"], row["fold_id"]) for row in scores}:
        raise ArtifactError("ModelScore fold coverage does not match its model bundle")
    return {"schema": "quant-project-model-audit-v1", "status": "verified", "run_id": summary["run_id"],
            "compiled_plan_sha256": claimed, "members": members, "score_rows": len(scores),
            "roundtrip": roundtrip, "retention_class": "reference_only" if downgraded else
            "executable_frozen" if roundtrip else "reproducible_research"}
