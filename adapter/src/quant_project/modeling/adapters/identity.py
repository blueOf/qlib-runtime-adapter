"""The explicit no-learning IdentityModel adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterBase
from ..contracts import (CapabilityError, ConfigError, DataContractError, FitContext,
                         ModelInputBatch, ModelSpec, PredictContext, finite_score)


class IdentityAdapter(AdapterBase):
    adapter_id = "identity.v1"
    adapter_version = "1.0.0"

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "task_types": ("identity", "transform"),
            "fit_policies": ("no_fit", "frozen_artifact"),
            "supports_missing": False,
            "supports_multi_output": False,
            "deterministic": True,
        }

    def validate(self, model_spec: ModelSpec, feature_ids: Sequence[str], target_field: str | None) -> None:
        super().validate(model_spec, feature_ids, target_field)
        fields = model_spec.input_factor_ids
        if len(fields) != 1:
            raise ConfigError("IdentityModel requires exactly one input factor")
        if model_spec.target_field or target_field:
            raise CapabilityError("IdentityModel cannot consume a target")
        if model_spec.params.get("higher_is_better") not in (True, False):
            raise ConfigError("IdentityModel requires params.higher_is_better")

    def fit(self, train_batch: ModelInputBatch, fit_context: FitContext) -> dict[str, Any]:
        spec = fit_context.metadata.get("model_spec") or {}
        fields = tuple(spec.get("input_factor_ids") or ())
        if len(fields) != 1:
            raise ConfigError("IdentityModel requires exactly one input factor")
        params = spec.get("params") or {}
        return {"factor_id": fields[0], "higher_is_better": bool(params["higher_is_better"])}

    def predict(self, fitted_handle: Mapping[str, Any], inference_batch: ModelInputBatch,
                predict_context: PredictContext) -> Sequence[Mapping[str, Any]]:
        factor_id = str(fitted_handle["factor_id"])
        sign = 1.0 if bool(fitted_handle["higher_is_better"]) else -1.0
        output = []
        for row in inference_batch.rows:
            key = f"{row['timestamp']}|{row['symbol']}"
            if row.get(factor_id) is None:
                raise DataContractError(f"IdentityModel missing value for {key}: {factor_id}")
            output.append({"timestamp": str(row["timestamp"]), "symbol": str(row["symbol"]),
                           "score": finite_score(sign * float(row[factor_id]), key=key)})
        return output

    def dump(self, fitted_handle: Mapping[str, Any], artifact_directory: Path) -> Mapping[str, Any]:
        self.write_json(artifact_directory, "adapter_state.json", dict(fitted_handle))
        return {"kind": "deterministic_state", "state_file": "adapter_state.json"}

    def load(self, artifact_manifest: Mapping[str, Any], artifact_directory: Path) -> Mapping[str, Any]:
        state = self.read_json(artifact_directory, "adapter_state.json")
        if not state.get("factor_id") or state.get("higher_is_better") not in (True, False):
            raise ConfigError("invalid IdentityModel artifact state")
        return state
