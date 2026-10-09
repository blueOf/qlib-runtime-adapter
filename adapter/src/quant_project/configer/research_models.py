"""Resolved, versioned inputs to the generic model research stage."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from ..modeling.contracts import ConfigError, ModelSpec, _freeze, _jsonable, sha256_json


def _id(value, prefix):
    if not isinstance(value, str) or not re.fullmatch(rf"{prefix}_[A-Z0-9_]+_V[1-9][0-9]*", value):
        raise ConfigError(f"invalid {prefix} id: {value!r}")
    return value


def _object(value, name):
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be an object")
    return dict(value)


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ConfigError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class FactorSetSpec:
    id: str
    factors: tuple[Mapping, ...]

    @classmethod
    def from_mapping(cls, value):
        value = _object(value, "FactorSet")
        factors = value.get("factors")
        if not isinstance(factors, list) or not factors:
            raise ConfigError("FactorSet requires ordered resolved factors")
        ids, result = set(), []
        for item in factors:
            item = _object(item, "resolved factor")
            factor_id = _id(item.get("id"), "FAC")
            if factor_id in ids:
                raise ConfigError("FactorSet contains duplicate factors")
            ids.add(factor_id)
            if type(item.get("higher_is_better")) is not bool:
                raise ConfigError("each resolved factor requires higher_is_better")
            if item.get("dtype") != "float64" or item.get("missing_policy") != "fail":
                raise ConfigError("v1 model research requires explicit float64/fail factor schema")
            if not re.fullmatch("[a-f0-9]{64}", str(item.get("config_sha256", ""))):
                raise ConfigError("each resolved factor requires config_sha256")
            result.append(_freeze(item))
        return cls(_id(value.get("id"), "FSET"), tuple(result))

    @property
    def feature_ids(self):
        return tuple(item["id"] for item in self.factors)

    def to_dict(self):
        return {"schema_version": "quant-project-factor-set-v1", "id": self.id,
                "factors": _jsonable(self.factors)}

    @property
    def config_sha256(self):
        return sha256_json(self.to_dict())

    def bind_model(self, value):
        data = dict(value)
        if data.get("factor_set_id", self.id) != self.id:
            raise ConfigError("ModelSpec FactorSet does not match research input")
        if data.get("factor_set_config_sha256", self.config_sha256) != self.config_sha256:
            raise ConfigError("ModelSpec FactorSet hash does not match resolved FactorSet")
        if tuple(data.get("input_factor_ids", self.feature_ids)) != self.feature_ids:
            raise ConfigError("ModelSpec input order must match resolved FactorSet")
        data.update(input_factor_ids=list(self.feature_ids), factor_set_id=self.id,
                    factor_set_config_sha256=self.config_sha256)
        return ModelSpec.from_mapping(data)


@dataclass(frozen=True)
class LabelSpec:
    id: str
    field: str
    horizon_sessions: int
    availability_field: str = "available_at"

    @classmethod
    def from_mapping(cls, value):
        value = _object(value, "LabelSpec")
        field = value.get("field")
        if not isinstance(field, str) or not field or field in {"timestamp", "symbol", "available_at"}:
            raise ConfigError("LabelSpec requires a distinct target field")
        if value.get("availability_field", "available_at") != "available_at":
            raise ConfigError("v1 labels require explicit available_at")
        return cls(_id(value.get("id"), "LBL"), field,
                   _nonnegative_int(value.get("horizon_sessions"), "horizon_sessions"))

    def to_dict(self):
        return {"schema_version": "quant-project-label-spec-v1", "id": self.id, "field": self.field,
                "horizon_sessions": self.horizon_sessions, "availability_field": self.availability_field}


@dataclass(frozen=True)
class SplitPolicySpec:
    id: str
    sessions: tuple[str, ...]
    folds: tuple[Mapping, ...]
    purge_sessions: int
    embargo_sessions: int

    @classmethod
    def from_mapping(cls, value):
        value = _object(value, "SplitPolicySpec")
        sessions = value.get("sessions")
        folds = value.get("folds")
        if not isinstance(sessions, list) or not sessions or sessions != sorted(set(sessions)):
            raise ConfigError("SplitPolicy requires unique ordered calendar sessions")
        if not isinstance(folds, list) or not folds:
            raise ConfigError("SplitPolicy requires preregistered folds (including SPLIT_NONE)")
        return cls(_id(value.get("id"), "SPLIT"), tuple(sessions),
                   tuple(_freeze(_object(item, "fold")) for item in folds),
                   _nonnegative_int(value.get("purge_sessions"), "purge_sessions"),
                   _nonnegative_int(value.get("embargo_sessions"), "embargo_sessions"))

    def to_dict(self):
        return {"schema_version": "quant-project-split-policy-v1", "id": self.id,
                "sessions": list(self.sessions), "folds": _jsonable(self.folds),
                "purge_sessions": self.purge_sessions, "embargo_sessions": self.embargo_sessions}


@dataclass(frozen=True)
class EvaluationPolicySpec:
    id: str
    metric: str
    minimum_label_coverage: float
    mode: str = "fixed_comparison"
    baseline_model_id: str | None = None

    @classmethod
    def from_mapping(cls, value):
        value = _object(value, "EvaluationPolicySpec")
        metric = value.get("metric", "rank_ic")
        if metric not in {"rank_ic", "mse", "log_loss"}:
            raise ConfigError("evaluation metric must be rank_ic, mse, or log_loss")
        coverage = value.get("minimum_label_coverage", 1.0)
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 < coverage <= 1:
            raise ConfigError("minimum_label_coverage must be in (0, 1]")
        mode = value.get("mode", "fixed_comparison")
        if mode not in {"fixed_comparison", "nested_walk_forward"}:
            raise ConfigError("unsupported model research evaluation mode")
        baseline = value.get("baseline_model_id")
        if baseline is not None:
            _id(baseline, "MOD")
        return cls(_id(value.get("id"), "EVAL"), metric, float(coverage), mode, baseline)

    def to_dict(self):
        return {"schema_version": "quant-project-evaluation-policy-v1", "id": self.id,
                "metric": self.metric, "minimum_label_coverage": self.minimum_label_coverage,
                "mode": self.mode, "baseline_model_id": self.baseline_model_id}
