"""Small helpers shared by model adapters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import (CapabilityError, ConfigError, DataContractError, FitContext,
                         ModelInputBatch, ModelSpec, PredictContext, canonical_json)


class AdapterBase:
    """Convenience base class; adapters remain plain Python objects."""

    adapter_id = "base"
    adapter_version = "1.0.0"

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "task_types": (),
            "fit_policies": (),
            "supports_missing": False,
            "supports_multi_output": False,
            "deterministic": True,
        }

    def validate(self, model_spec: ModelSpec, feature_ids: Sequence[str], target_field: str | None) -> None:
        capabilities = self.capabilities()
        requirements = model_spec.metadata.get("required_capabilities", {})
        if not isinstance(requirements, Mapping):
            raise ConfigError("required_capabilities must be a mapping")
        for name, required in requirements.items():
            if required is True and not capabilities.get(name, False):
                raise CapabilityError(f"adapter {self.adapter_id} does not provide required capability {name}")
        if model_spec.task_type not in set(capabilities.get("task_types", ())):
            raise CapabilityError(
                f"adapter {self.adapter_id} does not support task_type {model_spec.task_type!r}")
        if model_spec.fit_policy not in set(capabilities.get("fit_policies", ())):
            raise CapabilityError(
                f"adapter {self.adapter_id} does not support fit_policy {model_spec.fit_policy!r}")
        if model_spec.execution_status == "reference_only":
            raise CapabilityError("reference_only ModelSpec cannot be executed")
        if model_spec.input_factor_ids:
            missing = [item for item in model_spec.input_factor_ids if item not in feature_ids]
            if missing:
                raise DataContractError(f"ModelSpec features are missing from batch: {missing}")
        if model_spec.fit_policy == "train_per_fold" and not (target_field or model_spec.target_field):
            raise ConfigError("train_per_fold requires an explicit target field")

    @staticmethod
    def write_json(directory: Path, name: str, value: Mapping[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(
            canonical_json(value) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def read_json(directory: Path, name: str) -> dict[str, Any]:
        path = directory / name
        if not path.is_file():
            raise DataContractError(f"adapter state file is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise DataContractError(f"adapter state must be an object: {path}")
        return value

    @staticmethod
    def require_feature_ids(model_spec: ModelSpec, batch: ModelInputBatch) -> tuple[str, ...]:
        fields = model_spec.input_factor_ids or batch.feature_ids
        if not fields:
            raise ConfigError(f"ModelSpec {model_spec.id} has no input features")
        return tuple(fields)
