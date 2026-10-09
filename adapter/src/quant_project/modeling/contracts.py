"""Versioned, dependency-light contracts for the model adapter layer.

The module deliberately does not import pandas, sklearn, Qlib, or any other
estimator library.  Adapters consume plain row mappings so the registry and
the Identity model remain usable in a minimal research environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


MODEL_SPEC_SCHEMA = "quant-project-model-spec-v1"
ARTIFACT_SCHEMA = "quant-project-model-artifact-manifest-v1"
FIT_POLICIES = frozenset({"no_fit", "train_per_fold", "frozen_artifact"})
EXECUTION_STATUSES = frozenset({"executable", "reference_only"})
MODEL_ID = re.compile(r"^MOD_[A-Z0-9_]+_V[1-9][0-9]*$")


class ModelingError(RuntimeError):
    """Base class for machine-readable model-layer failures."""

    code = "MODELING_ERROR"

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.details = dict(details or {})


class ConfigError(ModelingError):
    code = "CONFIG_INVALID"


class AdapterUnknownError(ModelingError):
    code = "ADAPTER_UNKNOWN"


class CapabilityError(ModelingError):
    code = "CAPABILITY_UNSUPPORTED"


class DependencyUnavailableError(ModelingError):
    code = "DEPENDENCY_UNAVAILABLE"


class DataContractError(ModelingError):
    code = "DATA_CONTRACT_INVALID"


class LeakageError(ModelingError):
    code = "LEAKAGE_DETECTED"


class FitError(ModelingError):
    code = "FIT_FAILED"


class PredictError(ModelingError):
    code = "PREDICT_FAILED"


class ArtifactError(ModelingError):
    code = "ARTIFACT_INVALID"


class ResourceBudgetError(ModelingError):
    code = "RESOURCE_BUDGET_EXCEEDED"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"value is not canonically serializable: {error}") from error


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{name} must be a sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ConfigError(f"{name} must contain unique non-empty strings")
    return result


@dataclass(frozen=True)
class ModelSpec:
    """A fully resolved model configuration.

    ``params`` must contain one concrete configuration.  Adapter-level
    hyperparameter grids are intentionally not supported here; comparison
    callers create separate ModelSpec/attempt identities instead.
    """

    id: str
    adapter: str
    input_factor_ids: tuple[str, ...] = ()
    factor_set_id: str | None = None
    factor_set_config_sha256: str | None = None
    fit_policy: str = "no_fit"
    execution_status: str = "executable"
    task_type: str = "transform"
    target_field: str | None = None
    target_contract: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    artifact: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_factor_ids", _string_tuple(self.input_factor_ids, "input_factor_ids"))
        for name in ("params", "resources", "metadata"):
            object.__setattr__(self, name, _freeze(_require_mapping(getattr(self, name), name)))
        if self.artifact is not None:
            object.__setattr__(self, "artifact", _freeze(_require_mapping(self.artifact, "artifact")))
        if not MODEL_ID.fullmatch(self.id):
            raise ConfigError(f"invalid ModelSpec id: {self.id!r}")
        if not self.adapter or any(char.isspace() for char in self.adapter):
            raise ConfigError("ModelSpec.adapter must be a non-empty registry key")
        if self.fit_policy not in FIT_POLICIES:
            raise ConfigError(f"unsupported fit_policy: {self.fit_policy!r}")
        if self.execution_status not in EXECUTION_STATUSES:
            raise ConfigError(f"unsupported execution_status: {self.execution_status!r}")
        if len(set(self.input_factor_ids)) != len(self.input_factor_ids):
            raise ConfigError("ModelSpec.input_factor_ids must be unique")
        if self.fit_policy == "frozen_artifact" and self.execution_status == "executable":
            if not self.artifact:
                raise ConfigError("executable frozen_artifact ModelSpec requires artifact reference")
        if self.fit_policy == "no_fit":
            provenance = str(self.metadata.get("parameter_provenance", "") or
                             self.params.get("parameter_provenance", ""))
            # Identity has no learned or user-selected parameters, so its
            # provenance is intrinsic to the registered adapter.  Other
            # no-fit models must explain where their fixed weights/rules came
            # from; this keeps a hand-tuned or label-derived model from being
            # silently reclassified as a harmless deterministic baseline.
            if not provenance and self.adapter != "identity.v1":
                raise ConfigError("no_fit ModelSpec requires parameter_provenance")
            normalized = provenance.lower().replace("-", "_")
            label_free = "without_label" in normalized or "no_label" in normalized
            label_derived = any(token in normalized for token in (
                "label_derived", "fitted_on_label", "fit_on_label",
                "trained_on_label", "learned_from_label", "supervised",
                "selected_with_label", "learned", "trained", "fitted",
                "selected",
            ))
            if "fit" in normalized and not ("without_label_fit" in normalized or
                                             normalized == "no_fit" or label_free):
                label_derived = True
            if "label" in normalized and not label_free:
                label_derived = True
            if label_derived:
                raise ConfigError("label-derived parameters must use frozen_artifact, not no_fit")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelSpec":
        data = _require_mapping(value, "ModelSpec")
        schema = data.get("schema_version", MODEL_SPEC_SCHEMA)
        if schema != MODEL_SPEC_SCHEMA:
            raise ConfigError(f"unsupported ModelSpec schema: {schema!r}")
        params = _require_mapping(data.get("params", data.get("parameters", {})), "ModelSpec.params")
        metadata = _require_mapping(data.get("metadata", {}), "ModelSpec.metadata")
        if "parameter_provenance" in data and "parameter_provenance" not in metadata:
            metadata["parameter_provenance"] = data["parameter_provenance"]
        return cls(
            id=str(data.get("id", "")),
            adapter=str(data.get("adapter", "")),
            input_factor_ids=_string_tuple(data.get("input_factor_ids", data.get("input", {}).get("factor_ids")
                                             if isinstance(data.get("input"), Mapping) else None),
                                           "ModelSpec.input_factor_ids"),
            factor_set_id=(str(data["factor_set_id"]) if data.get("factor_set_id") is not None else
                           (str(data["input"]["factor_set"]) if isinstance(data.get("input"), Mapping)
                            and data["input"].get("factor_set") is not None else None)),
            factor_set_config_sha256=(str(data["factor_set_config_sha256"])
                                      if data.get("factor_set_config_sha256") is not None else None),
            fit_policy=str(data.get("fit_policy", data.get("training", {}).get("fit_policy", "no_fit")
                             if isinstance(data.get("training"), Mapping) else "no_fit")),
            execution_status=str(data.get("execution_status", "executable")),
            task_type=str(data.get("task_type", "transform")),
            target_field=(str(data["target_field"]) if data.get("target_field") is not None else None),
            target_contract=(str(data["target_contract"]) if data.get("target_contract") is not None else
                             (str(data["training"]["target_contract"])
                              if isinstance(data.get("training"), Mapping)
                              and data["training"].get("target_contract") is not None else None)),
            params=params,
            resources=_require_mapping(data.get("resources", {}), "ModelSpec.resources"),
            artifact=(_require_mapping(data["artifact"], "ModelSpec.artifact")
                      if data.get("artifact") is not None else None),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SPEC_SCHEMA,
            "id": self.id,
            "adapter": self.adapter,
            "input_factor_ids": list(self.input_factor_ids),
            "factor_set_id": self.factor_set_id,
            "factor_set_config_sha256": self.factor_set_config_sha256,
            "fit_policy": self.fit_policy,
            "execution_status": self.execution_status,
            "task_type": self.task_type,
            "target_field": self.target_field,
            "target_contract": self.target_contract,
            "params": _jsonable(dict(self.params)),
            "resources": _jsonable(dict(self.resources)),
            "artifact": _jsonable(dict(self.artifact)) if self.artifact else None,
            "metadata": _jsonable(dict(self.metadata)),
        }

    @property
    def config_sha256(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def model_identity_sha256(self) -> str:
        """Hash of the model definition independent of execution policy.

        A trained artifact is commonly produced by a ``train_per_fold`` run
        and later referenced by an otherwise identical ``frozen_artifact``
        spec.  Including the fit policy or the artifact's own reference in
        this identity would make safe round-trip verification recursive.
        The full resolved spec hash remains available as ``config_sha256``
        for run/score lineage.
        """
        return sha256_json({
            "id": self.id,
            "adapter": self.adapter,
            "input_factor_ids": list(self.input_factor_ids),
            "factor_set_id": self.factor_set_id,
            "factor_set_config_sha256": self.factor_set_config_sha256,
            "task_type": self.task_type,
            "target_field": self.target_field,
            "target_contract": self.target_contract,
            "params": _jsonable(dict(self.params)),
            "resources": _jsonable(dict(self.resources)),
        })


@dataclass(frozen=True)
class ModelInputBatch:
    """Immutable view of feature rows handed to an adapter."""

    rows: tuple[Mapping[str, Any], ...]
    feature_ids: tuple[str, ...]

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]], feature_ids: Sequence[str]) -> "ModelInputBatch":
        copied = tuple(_freeze(dict(row)) for row in rows)
        result = cls(copied, _string_tuple(feature_ids, "feature_ids"))
        result.validate()
        return result

    def validate(self) -> None:
        seen: set[tuple[str, str]] = set()
        for index, row in enumerate(self.rows):
            if not isinstance(row, Mapping):
                raise DataContractError(f"row {index} is not a mapping")
            if row.get("timestamp") is None or row.get("symbol") is None:
                raise DataContractError(f"row {index} requires timestamp and symbol")
            key = (str(row["timestamp"]), str(row["symbol"]))
            if key in seen:
                raise DataContractError(f"duplicate input key: {key[0]}|{key[1]}")
            seen.add(key)
            missing = [field for field in self.feature_ids if field not in row]
            if missing:
                raise DataContractError(f"row {index} missing feature columns: {missing}")


@dataclass(frozen=True)
class FitContext:
    run_id: str
    research_release: str | None = None
    data_mode: str = "point_in_time"
    fold_id: str | None = None
    train_window: tuple[str, str] | None = None
    validation_window: tuple[str, str] | None = None
    inference_window: tuple[str, str] | None = None
    purge_sessions: int = 0
    embargo_sessions: int = 0
    random_seed: int | None = None
    feature_order: tuple[str, ...] = ()
    target_field: str | None = None
    input_data_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ConfigError("FitContext.run_id is required")
        if self.purge_sessions < 0 or self.embargo_sessions < 0:
            raise ConfigError("purge/embargo sessions cannot be negative")
        if self.random_seed is not None and type(self.random_seed) is not int:
            raise ConfigError("random_seed must be an integer or null")

    def with_runtime(self, *, model_spec: ModelSpec, registry: Any) -> "FitContext":
        metadata = dict(self.metadata)
        metadata["model_spec"] = model_spec.to_dict()
        metadata["model_spec_sha256"] = model_spec.config_sha256
        metadata["_registry"] = registry
        return FitContext(
            run_id=self.run_id, research_release=self.research_release, data_mode=self.data_mode,
            fold_id=self.fold_id, train_window=self.train_window,
            validation_window=self.validation_window, inference_window=self.inference_window,
            purge_sessions=self.purge_sessions, embargo_sessions=self.embargo_sessions,
            random_seed=self.random_seed, feature_order=self.feature_order,
            target_field=self.target_field, input_data_sha256=self.input_data_sha256,
            metadata=metadata,
        )


@dataclass(frozen=True)
class PredictContext:
    run_id: str
    data_mode: str = "point_in_time"
    fold_id: str | None = None
    inference_window: tuple[str, str] | None = None
    feature_order: tuple[str, ...] = ()
    input_data_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ConfigError("PredictContext.run_id is required")

    def with_runtime(self, *, model_spec: ModelSpec, registry: Any) -> "PredictContext":
        metadata = dict(self.metadata)
        metadata["model_spec"] = model_spec.to_dict()
        metadata["model_spec_sha256"] = model_spec.config_sha256
        metadata["_registry"] = registry
        return PredictContext(
            run_id=self.run_id, data_mode=self.data_mode, fold_id=self.fold_id,
            inference_window=self.inference_window, feature_order=self.feature_order,
            input_data_sha256=self.input_data_sha256, metadata=metadata,
        )


@dataclass(frozen=True)
class ArtifactManifest:
    schema: str
    artifact_id: str
    adapter_id: str
    adapter_version: str
    model_spec_sha256: str
    fit_policy: str
    files: Mapping[str, Mapping[str, Any]]
    model_identity_sha256: str | None = None
    manifest_sha256: str | None = None
    payload_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "model_spec_sha256": self.model_spec_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "fit_policy": self.fit_policy,
            "files": _jsonable(dict(self.files)),
            "manifest_sha256": self.manifest_sha256,
            "payload_sha256": self.payload_sha256,
            "metadata": _jsonable(dict(self.metadata)),
            "root": self.root,
        }


@dataclass(frozen=True)
class FittedModel:
    spec: ModelSpec
    adapter_id: str
    adapter_version: str
    handle: Any
    artifact: Mapping[str, Any]
    fit_context: FitContext | None = None


@dataclass(frozen=True)
class ModelScore:
    model_score_id: str
    research_run_id: str
    timestamp: str
    symbol: str
    score: float
    fold_id: str | None
    source_model_id: str
    source_factor_set_id: str | None
    model_config_sha256: str
    factor_set_config_sha256: str | None
    artifact_manifest_sha256: str
    model_payload_sha256: str | None = None
    source_factor_vector_id: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "model_score_id": self.model_score_id,
            "research_run_id": self.research_run_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "score": self.score,
            "fold_id": self.fold_id,
            "source_model_id": self.source_model_id,
            "source_factor_set_id": self.source_factor_set_id,
            "model_config_sha256": self.model_config_sha256,
            "factor_set_config_sha256": self.factor_set_config_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "model_payload_sha256": self.model_payload_sha256,
            "source_factor_vector_id": self.source_factor_vector_id,
        }
        if self.diagnostics:
            value["diagnostics"] = _jsonable(dict(self.diagnostics))
        return value


@runtime_checkable
class ModelAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def capabilities(self) -> Mapping[str, Any]: ...

    def validate(self, model_spec: ModelSpec, feature_ids: Sequence[str], target_field: str | None) -> None: ...

    def fit(self, train_batch: ModelInputBatch, fit_context: FitContext) -> Any: ...

    def predict(self, fitted_handle: Any, inference_batch: ModelInputBatch,
                predict_context: PredictContext) -> Sequence[Mapping[str, Any]]: ...

    def dump(self, fitted_handle: Any, artifact_directory: Any) -> Mapping[str, Any]: ...

    def load(self, artifact_manifest: Mapping[str, Any], artifact_directory: Any) -> Any: ...


def finite_score(value: Any, *, key: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise PredictError(f"score is not numeric for {key}") from error
    if not math.isfinite(score):
        raise PredictError(f"score is not finite for {key}")
    return score
