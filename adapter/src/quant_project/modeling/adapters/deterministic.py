"""Deterministic fixed-linear and cross-sectional rank-blend adapters."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterBase
from ..contracts import (ConfigError, DataContractError, FitContext, ModelInputBatch,
                         ModelSpec, PredictContext, finite_score)


def _average_ranks(values: Mapping[str, float | None]) -> dict[str, float]:
    """Return 1..N average ranks, with missing values excluded."""
    ordered = sorted(((key, value) for key, value in values.items() if value is not None),
                     key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for key, _ in ordered[index:end]:
            ranks[key] = rank
        index = end
    return ranks


class DeterministicAdapter(AdapterBase):
    adapter_version = "1.0.0"

    def __init__(self, adapter_id: str, *, transform: str):
        self.adapter_id = adapter_id
        self.transform = transform

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "task_types": ("transform",),
            "fit_policies": ("no_fit", "frozen_artifact"),
            "supports_missing": True,
            "supports_multi_output": False,
            "deterministic": True,
        }

    def validate(self, model_spec: ModelSpec, feature_ids: Sequence[str], target_field: str | None) -> None:
        super().validate(model_spec, feature_ids, target_field)
        if not model_spec.input_factor_ids:
            raise ConfigError(f"{self.adapter_id} requires input_factor_ids")
        weights = model_spec.params.get("weights")
        if not isinstance(weights, Mapping):
            raise ConfigError(f"{self.adapter_id} requires a weights mapping")
        missing = [field for field in model_spec.input_factor_ids if field not in weights]
        if missing:
            raise ConfigError(f"{self.adapter_id} weights omit features: {missing}")
        if any(not math.isfinite(float(weights[field])) for field in model_spec.input_factor_ids):
            raise ConfigError(f"{self.adapter_id} weights must be finite")
        directions = model_spec.params.get("directions", {})
        if not isinstance(directions, Mapping):
            raise ConfigError(f"{self.adapter_id} directions must be a mapping")
        if any(field in directions and directions[field] not in (True, False)
               for field in model_spec.input_factor_ids):
            raise ConfigError(f"{self.adapter_id} directions must contain booleans")
        missing_policy = str(model_spec.params.get("missing_policy", "fail"))
        if missing_policy not in {"fail", "worst_rank", "explicit_worst_rank"}:
            raise ConfigError(f"unsupported missing_policy: {missing_policy}")
        if self.transform == "rank" and model_spec.params.get("rank_method", "average") != "average":
            raise ConfigError("fixed rank blend currently supports only rank_method=average")

    def fit(self, train_batch: ModelInputBatch, fit_context: FitContext) -> Mapping[str, Any]:
        spec = fit_context.metadata.get("model_spec") or {}
        params = dict(spec.get("params") or {})
        return {
            "factor_ids": list(spec.get("input_factor_ids") or train_batch.feature_ids),
            "weights": {str(key): float(value) for key, value in (params.get("weights") or {}).items()},
            "directions": {str(key): bool(value) for key, value in (params.get("directions") or {}).items()},
            "missing_policy": str(params.get("missing_policy", "fail")),
            "transform": self.transform,
            "rank_method": str(params.get("rank_method", "average")),
        }

    def _values(self, handle: Mapping[str, Any], batch: ModelInputBatch) -> dict[str, dict[str, float | None]]:
        factor_ids = tuple(str(item) for item in handle["factor_ids"])
        directions = dict(handle.get("directions") or {})
        missing_policy = str(handle.get("missing_policy", "fail"))
        values: dict[str, dict[str, float | None]] = {field: {} for field in factor_ids}
        for row in batch.rows:
            key = f"{row['timestamp']}|{row['symbol']}"
            for field in factor_ids:
                raw = row.get(field)
                if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
                    if missing_policy == "fail":
                        raise DataContractError(f"missing/non-finite value for {key}: {field}")
                    values[field][key] = None
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    raise DataContractError(f"missing/non-finite value for {key}: {field}")
                if directions.get(field) is False:
                    value = -value
                values[field][key] = value
        return values

    def predict(self, fitted_handle: Mapping[str, Any], inference_batch: ModelInputBatch,
                predict_context: PredictContext) -> Sequence[Mapping[str, Any]]:
        factor_ids = tuple(str(item) for item in fitted_handle["factor_ids"])
        weights = {str(key): float(value) for key, value in fitted_handle["weights"].items()}
        values = self._values(fitted_handle, inference_batch)
        ranks: dict[str, dict[str, float]] = {}
        if self.transform == "rank":
            grouped_keys: dict[str, list[str]] = {}
            for row in inference_batch.rows:
                grouped_keys.setdefault(str(row["timestamp"]), []).append(
                    f"{row['timestamp']}|{row['symbol']}"
                )
            for field in factor_ids:
                ranks[field] = {}
                for timestamp, keys in grouped_keys.items():
                    del timestamp
                    raw = {key: values[field][key] for key in keys}
                    rank_values = _average_ranks(raw)
                    count = max(1, len(rank_values))
                    ranks[field].update({key: value / count for key, value in rank_values.items()})
                    for key in keys:
                        if key not in ranks[field]:
                            ranks[field][key] = 0.0
        output = []
        for row in inference_batch.rows:
            key = f"{row['timestamp']}|{row['symbol']}"
            score = 0.0
            for field in factor_ids:
                value = (ranks[field][key] if self.transform == "rank" else values[field][key])
                if value is None:
                    value = 0.0
                score += weights[field] * value
            output.append({"timestamp": str(row["timestamp"]), "symbol": str(row["symbol"]),
                           "score": finite_score(score, key=key)})
        return output

    def dump(self, fitted_handle: Mapping[str, Any], artifact_directory: Path) -> Mapping[str, Any]:
        self.write_json(artifact_directory, "adapter_state.json", dict(fitted_handle))
        return {"kind": "deterministic_state", "state_file": "adapter_state.json"}

    def load(self, artifact_manifest: Mapping[str, Any], artifact_directory: Path) -> Mapping[str, Any]:
        state = self.read_json(artifact_directory, "adapter_state.json")
        if state.get("transform") != self.transform:
            raise ConfigError(f"artifact transform does not match adapter {self.adapter_id}")
        return state
