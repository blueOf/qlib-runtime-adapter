"""Atomic, content-addressed model artifacts."""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .contracts import (ARTIFACT_SCHEMA, ArtifactError, ModelAdapter,
                        ModelSpec, sha256_json)
from .runtime import adapter_runtime


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def _inside_root(self, path: Path) -> Path:
        path = path.expanduser().resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ArtifactError(f"artifact path escapes approved root: {path}") from error
        return path

    @staticmethod
    def _safe_relative(path: Path, root: Path) -> str:
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise ArtifactError(f"artifact file escapes temporary root: {path}") from error
        if not relative or relative == "manifest.json" or relative.startswith("../"):
            raise ArtifactError(f"invalid artifact file path: {relative!r}")
        return relative

    @staticmethod
    def _identity_payload(manifest: dict) -> dict:
        return {key: manifest[key] for key in (
            "schema", "adapter_id", "adapter_version", "model_spec_sha256",
            "model_identity_sha256", "fit_policy",
            "files", "payload_sha256", "metadata")}

    def publish(self, *, model_spec: ModelSpec, adapter: ModelAdapter, handle,
                metadata: dict | None = None) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".model-artifact-{uuid4().hex}"
        try:
            temporary.mkdir()
            adapter_metadata = dict(adapter.dump(handle, temporary) or {})
            files: dict[str, dict] = {}
            for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
                relative = self._safe_relative(path, temporary)
                if relative == "manifest.json":
                    raise ArtifactError("adapter may not write manifest.json")
                files[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            combined_metadata = dict(metadata or {})
            combined_metadata["adapter_dump"] = adapter_metadata
            combined_metadata["runtime"] = adapter_runtime(adapter)
            payload_sha256 = sha256_json(files)
            base = {
                "schema": ARTIFACT_SCHEMA,
                "adapter_id": adapter.adapter_id,
                "adapter_version": adapter.adapter_version,
                "model_spec_sha256": model_spec.config_sha256,
                "model_identity_sha256": model_spec.model_identity_sha256,
                "fit_policy": model_spec.fit_policy,
                "files": files,
                "payload_sha256": payload_sha256,
                "metadata": combined_metadata,
            }
            artifact_id = sha256_json(base)
            manifest = base | {
                "artifact_id": artifact_id,
                "sealed": True,
                "created_at": _now(),
            }
            target = self._inside_root(self.root / artifact_id)
            if target.exists():
                raise ArtifactError(f"refusing to overwrite existing model artifact: {target}")
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)
            temporary = None
            verified = self.verify(target)
            trust_root = self.root / ".trusted"
            trust_root.mkdir(exist_ok=True)
            (trust_root / f"{artifact_id}.json").write_text(
                json.dumps({"manifest_sha256": verified["manifest_sha256"]}), encoding="utf-8")
            return verified
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def verify(self, artifact_root) -> dict:
        root = self._inside_root(Path(artifact_root))
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ArtifactError(f"model artifact missing manifest: {root}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ArtifactError(f"invalid model artifact manifest: {manifest_path}") from error
        if not isinstance(manifest, dict) or manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("sealed") is not True:
            raise ArtifactError(f"invalid model artifact schema/state: {root}")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ArtifactError("model artifact files must be a mapping")
        actual_files = {
            self._safe_relative(path, root)
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual_files != set(str(relative) for relative in files):
            raise ArtifactError("model artifact contains unlisted or missing payload files")
        for relative, recorded in files.items():
            if not isinstance(recorded, dict) or "sha256" not in recorded or "bytes" not in recorded:
                raise ArtifactError(f"invalid model artifact file record: {relative}")
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in str(relative):
                raise ArtifactError(f"invalid relative artifact path: {relative}")
            path = (root / relative_path).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ArtifactError("artifact member escapes its own directory") from error
            if not path.is_file() or path.stat().st_size != recorded["bytes"] \
                    or _sha256_file(path) != recorded["sha256"]:
                raise ArtifactError(f"model artifact file missing or changed: {relative}")
        if manifest.get("payload_sha256") != sha256_json(files):
            raise ArtifactError("model artifact payload hash mismatch")
        try:
            expected_id = sha256_json(self._identity_payload(manifest))
        except (KeyError, TypeError) as error:
            raise ArtifactError("artifact manifest missing identity fields") from error
        if manifest.get("artifact_id") != expected_id:
            raise ArtifactError("model artifact identity hash mismatch")
        return manifest | {
            "root": str(root),
            "manifest_sha256": _sha256_file(manifest_path),
            "status": "verified",
        }

    def load(self, *, artifact_ref: dict, adapter: ModelAdapter, model_spec: ModelSpec):
        if not isinstance(artifact_ref, dict):
            raise ArtifactError("artifact reference must be a mapping")
        if artifact_ref.get("root"):
            root = self._inside_root(Path(str(artifact_ref["root"])))
        elif artifact_ref.get("artifact_id"):
            artifact_id = str(artifact_ref["artifact_id"])
            if not SAFE_ID.fullmatch(artifact_id):
                raise ArtifactError("invalid artifact_id")
            root = self._inside_root(self.root / artifact_id)
        else:
            raise ArtifactError("artifact reference requires root or artifact_id")
        manifest = self.verify(root)
        expected_sha = artifact_ref.get("manifest_sha256")
        if expected_sha and expected_sha != manifest["manifest_sha256"]:
            raise ArtifactError("artifact manifest does not match its pinned reference")
        trust_path = self.root / ".trusted" / f"{manifest['artifact_id']}.json"
        if not trust_path.is_file():
            raise ArtifactError("artifact was not published by this approved store; deserialization denied")
        trusted = json.loads(trust_path.read_text(encoding="utf-8"))
        if trusted.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ArtifactError("artifact does not match the locally registered manifest")
        if manifest.get("adapter_id") != adapter.adapter_id:
            raise ArtifactError("artifact adapter id does not match ModelSpec")
        if manifest.get("adapter_version") != adapter.adapter_version:
            raise ArtifactError("artifact adapter version does not match registry")
        if (manifest.get("model_spec_sha256") != model_spec.config_sha256 and
                manifest.get("model_identity_sha256") != model_spec.model_identity_sha256):
            raise ArtifactError("artifact ModelSpec hash does not match requested ModelSpec")
        if manifest.get("metadata", {}).get("runtime") != adapter_runtime(adapter):
            raise ArtifactError("artifact dependency or implementation version has changed")
        handle = adapter.load(manifest, root)
        return handle, manifest
