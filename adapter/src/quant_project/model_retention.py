"""Explicit executable-state downgrade when required evidence is unavailable."""
from __future__ import annotations

from pathlib import Path

from .common import atomic_json, read_json, sha256, utc_now
from .modeling.contracts import ArtifactError
from .modeling.artifacts import ArtifactStore
from .research_registry import _registry_lock


def record_model_retention_downgrade(run_root):
    run_root = Path(run_root).resolve()
    reports_root = run_root.parent.parent
    registry_path = reports_root / "registry.json"
    key = run_root.relative_to(reports_root).as_posix()
    with _registry_lock(reports_root / ".registry.lock"):
        registry = read_json(registry_path)
        entry = next((item for item in registry["entries"] if item["key"] == key
                      and any(record["path"] == "model_bundle.json" for record in item["artifacts"])), None)
        if entry is None:
            raise ArtifactError("model run is not registered")
        unavailable = []
        external_unavailable = []
        for artifact in entry["artifacts"]:
            path = (run_root / artifact["path"]).resolve()
            if not path.is_relative_to(run_root):
                raise ArtifactError("registered artifact escapes the model run")
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                unavailable.append({"path": artifact["path"], "expected_sha256": artifact["sha256"],
                                    "reason": "missing" if not path.is_file() else "changed"})
        bundle_record = next((item for item in entry["artifacts"] if item["path"] == "model_bundle.json"), None)
        bundle_path = run_root / "model_bundle.json"
        if bundle_record and bundle_path.is_file() and sha256(bundle_path) == bundle_record["sha256"]:
            for member in read_json(bundle_path)["members"]:
                if not member.get("external_root"):
                    continue
                try:
                    root = Path(member["external_root"]).resolve()
                    store = ArtifactStore(root.parent)
                    manifest = store.verify(root)
                    trust_path = root.parent / ".trusted" / f"{manifest['artifact_id']}.json"
                    if (manifest["manifest_sha256"] != member["manifest_sha256"]
                            or manifest["payload_sha256"] != member["payload_sha256"]
                            or not trust_path.is_file()
                            or read_json(trust_path).get("manifest_sha256") != member["manifest_sha256"]):
                        raise ArtifactError("external model artifact pin is missing or changed")
                except (ArtifactError, OSError, ValueError) as error:
                    external_unavailable.append({"external_root": member["external_root"],
                        "model_id": member["model_id"], "fold_id": member["fold_id"],
                        "expected_manifest_sha256": member["manifest_sha256"],
                        "expected_payload_sha256": member["payload_sha256"], "reason": str(error)})
        if not unavailable and not external_unavailable:
            raise ArtifactError("required evidence is intact; no retention downgrade is justified")
        if entry.get("retention_tombstone"):
            raise ArtifactError("retention downgrade is already recorded")
        tombstone = {"schema": "quant-project-model-retention-tombstone-v1", "run_key": key,
                     "recorded_at": utc_now(), "execution_status": "reference_only",
                     "retention_class": "reference_only", "unavailable_artifacts": unavailable,
                     "unavailable_external_artifacts": external_unavailable}
        path = reports_root / "model_tombstones" / run_root.parent.name / f"{run_root.name}.json"
        atomic_json(path, tombstone)
        entry.update(execution_status="reference_only", retention_class="reference_only",
                     unavailable_artifacts=[item["path"] for item in unavailable],
                     unavailable_external_artifacts=external_unavailable,
                     retention_tombstone={"path": path.relative_to(reports_root).as_posix(), "sha256": sha256(path)})
        registry["updated_at"] = utc_now()
        atomic_json(registry_path, registry)
    return tombstone | {"path": str(path)}
