"""Stable domain objects; they intentionally do not import Qlib."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..pit.contracts import DataModeError, merge_dependencies, normalize_data_mode, normalize_dependencies
from .research_models import FactorSetSpec, LabelSpec, SplitPolicySpec, EvaluationPolicySpec
from ..modeling.contracts import ModelSpec


_FACTOR = re.compile(r"^FAC_[A-Z0-9_]+_V[1-9][0-9]*$")
_SIGNAL = re.compile(r"^SIG_[A-Z0-9_]+_V[1-9][0-9]*$")
_STRATEGY = re.compile(r"^STR_[A-Z0-9_]+_V[1-9][0-9]*$")
_EXPERIMENT = re.compile(r"^EXP[0-9]+_V[1-9][0-9]*$")


class ConfigError(ValueError):
    pass


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _id(value, pattern, name):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ConfigError(f"{name} has an invalid ID: {value!r}")
    return value


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FactorSpec:
    id: str
    type: str
    implementation: str | None
    output_field: str
    higher_is_better: bool
    dependencies: dict[str, bool] = field(default_factory=lambda: {"market_data": True, "fundamentals": False, "industry": False})
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = _mapping(value, "FactorSpec")
        direction = _mapping(value.get("direction", {}), "FactorSpec.direction")
        metadata = _mapping(value.get("metadata", {}), "FactorSpec.metadata")
        try:
            dependencies = normalize_dependencies(value.get("dependencies", metadata.get("dependencies")),
                                                   defaults={"market_data": True})
        except DataModeError as error:
            raise ConfigError(str(error)) from error
        return cls(id=_id(value.get("id"), _FACTOR, "FactorSpec"),
                   type=str(value.get("type", "custom")), implementation=value.get("implementation"),
                   output_field=str(value.get("output_field", "factor_score")),
                   higher_is_better=bool(direction.get("higher_is_better", True)),
                   dependencies=dependencies, metadata=metadata)

    def to_dict(self):
        return {"id": self.id, "type": self.type, "implementation": self.implementation,
                "output_field": self.output_field,
                "direction": {"higher_is_better": self.higher_is_better},
                "dependencies": _jsonable(self.dependencies), "metadata": _jsonable(self.metadata)}


@dataclass(frozen=True)
class SignalSpec:
    id: str
    source_factor: str
    selection: dict[str, Any]
    ranking: dict[str, Any]
    direction: dict[str, Any]
    eligibility: dict[str, Any]
    dependencies: dict[str, bool] = field(default_factory=lambda: {"market_data": False, "fundamentals": False, "industry": False})
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = _mapping(value, "SignalSpec")
        source = _mapping(value.get("source"), "SignalSpec.source")
        eligibility = _mapping(value.get("eligibility", {}), "SignalSpec.eligibility")
        metadata = _mapping(value.get("metadata", {}), "SignalSpec.metadata")
        inferred_industry = bool(eligibility.get("known_industry_max_positions") or eligibility.get("max_per_industry"))
        try:
            dependencies = normalize_dependencies(value.get("dependencies", metadata.get("dependencies")),
                                                  defaults={"industry": inferred_industry})
        except DataModeError as error:
            raise ConfigError(str(error)) from error
        return cls(id=_id(value.get("id"), _SIGNAL, "SignalSpec"),
                   source_factor=_id(source.get("factor") or source.get("ref"), _FACTOR, "SignalSpec.source.factor"),
                   selection=_mapping(value.get("selection"), "SignalSpec.selection"),
                   ranking=_mapping(value.get("ranking", {}), "SignalSpec.ranking"),
                   direction=_mapping(value.get("direction", {}), "SignalSpec.direction"),
                   eligibility=eligibility, dependencies=dependencies, metadata=metadata)

    def to_dict(self):
        return {"id": self.id, "source": {"factor": self.source_factor}, "selection": _jsonable(self.selection),
                "ranking": _jsonable(self.ranking), "direction": _jsonable(self.direction),
                "eligibility": _jsonable(self.eligibility),
                "dependencies": _jsonable(self.dependencies), "metadata": _jsonable(self.metadata)}


@dataclass(frozen=True)
class StrategySpec:
    id: str
    portfolio: dict[str, Any]
    entry: dict[str, Any]
    exit: dict[str, Any]
    execution: dict[str, Any]
    accounting: dict[str, Any]
    dependencies: dict[str, bool] = field(default_factory=lambda: {"market_data": True, "fundamentals": False, "industry": False})
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = _mapping(value, "StrategySpec")
        metadata = _mapping(value.get("metadata", {}), "StrategySpec.metadata")
        try:
            dependencies = normalize_dependencies(value.get("dependencies", metadata.get("dependencies")),
                                                   defaults={"market_data": True})
        except DataModeError as error:
            raise ConfigError(str(error)) from error
        return cls(id=_id(value.get("id"), _STRATEGY, "StrategySpec"),
                   portfolio=_mapping(value.get("portfolio", {}), "StrategySpec.portfolio"),
                   entry=_mapping(value.get("entry", {}), "StrategySpec.entry"),
                   exit=_mapping(value.get("exit", {}), "StrategySpec.exit"),
                   execution=_mapping(value.get("execution", {}), "StrategySpec.execution"),
                   accounting=_mapping(value.get("accounting", {}), "StrategySpec.accounting"),
                   dependencies=dependencies, metadata=metadata)

    def to_dict(self):
        return {"id": self.id, "portfolio": _jsonable(self.portfolio), "entry": _jsonable(self.entry),
                "exit": _jsonable(self.exit), "execution": _jsonable(self.execution),
                "accounting": _jsonable(self.accounting), "dependencies": _jsonable(self.dependencies),
                "metadata": _jsonable(self.metadata)}


@dataclass(frozen=True)
class BacktestContext:
    market: str
    universe: str
    timezone: str
    start: str | None
    end: str | None
    research_release: str | None
    market_data_revision: str | None
    corporate_action_version: str | None
    calendar_version: str | None
    snapshot_compatible: bool = False
    data_mode: str = "snapshot_compatible"
    as_of: str | None = None
    as_of_policy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = _mapping(value, "BacktestContext")
        snapshot_compatible = bool(value.get("snapshot_compatible", False))
        try:
            data_mode = normalize_data_mode(value.get("data_mode", "snapshot_compatible"),
                                            snapshot_compatible=snapshot_compatible)
        except DataModeError as error:
            raise ConfigError(str(error)) from error
        snapshot_compatible = bool(value.get("snapshot_compatible", False))
        return cls(market=str(value.get("market", "CN_A")), universe=str(value.get("universe", "")),
                   timezone=str(value.get("timezone", "Asia/Shanghai")),
                   start=str(value["start"]) if value.get("start") is not None else None,
                   end=str(value["end"]) if value.get("end") is not None else None,
                   research_release=value.get("research_release"),
                   market_data_revision=value.get("market_data_revision"),
                   corporate_action_version=value.get("corporate_action_version"),
                   calendar_version=value.get("calendar_version"),
                   snapshot_compatible=snapshot_compatible, data_mode=data_mode,
                   as_of=str(value["as_of"]) if value.get("as_of") is not None else None,
                   as_of_policy=str(value["as_of_policy"]) if value.get("as_of_policy") is not None else None,
                   metadata=_mapping(value.get("metadata", {}), "BacktestContext.metadata"))

    def to_dict(self):
        return {"market": self.market, "universe": self.universe, "timezone": self.timezone,
                "start": self.start, "end": self.end, "research_release": self.research_release,
                "market_data_revision": self.market_data_revision,
                "corporate_action_version": self.corporate_action_version, "calendar_version": self.calendar_version,
                "snapshot_compatible": self.snapshot_compatible, "data_mode": self.data_mode,
                "as_of": self.as_of, "as_of_policy": self.as_of_policy,
                "metadata": _jsonable(self.metadata)}


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    factor: str
    signal: str
    strategy: str
    context: BacktestContext
    dependencies: dict[str, bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = _mapping(value, "ExperimentSpec")
        if value.get("schema_version", "stock-experiment-v1") != "stock-experiment-v1":
            raise ConfigError("ExperimentSpec has an unsupported schema_version")
        try:
            dependencies = (normalize_dependencies(value["dependencies"])
                            if value.get("dependencies") is not None else None)
        except DataModeError as error:
            raise ConfigError(str(error)) from error
        return cls(id=_id(value.get("id"), _EXPERIMENT, "ExperimentSpec"),
                   factor=_id(value.get("factor"), _FACTOR, "ExperimentSpec.factor"),
                   signal=_id(value.get("signal"), _SIGNAL, "ExperimentSpec.signal"),
                   strategy=_id(value.get("strategy"), _STRATEGY, "ExperimentSpec.strategy"),
                   context=BacktestContext.from_dict(value.get("context")),
                   dependencies=dependencies,
                   metadata=_mapping(value.get("metadata", {}), "ExperimentSpec.metadata"))

    def to_dict(self):
        return {"schema_version": "stock-experiment-v1", "id": self.id, "factor": self.factor,
                "signal": self.signal, "strategy": self.strategy, "context": self.context.to_dict(),
                "dependencies": _jsonable(self.dependencies),
                "metadata": _jsonable(self.metadata)}


@dataclass(frozen=True)
class ResolvedExperiment:
    experiment: ExperimentSpec
    factor: FactorSpec
    signal: SignalSpec
    strategy: StrategySpec
    context: BacktestContext
    source_hashes: dict[str, str] = field(default_factory=dict)

    def dependencies_for_stage(self, stage="full", *, direct_candidates=False) -> dict[str, bool]:
        if stage not in {"factor", "signal", "strategy", "full"}:
            raise ConfigError(f"unknown research stage: {stage!r}")
        parts = []
        if not (stage == "strategy" and direct_candidates):
            parts.append(self.factor.dependencies)
            if stage in {"signal", "full", "strategy"}:
                parts.append(self.signal.dependencies)
        if stage in {"strategy", "full"}:
            parts.append(self.strategy.dependencies)
        return merge_dependencies(*parts)

    @property
    def declared_dependencies(self):
        return merge_dependencies(self.factor.dependencies, self.signal.dependencies,
                                  self.strategy.dependencies, self.experiment.dependencies or {})

    def to_dict(self):
        return {"schema_version": "stock-resolved-experiment-v1", "experiment": self.experiment.to_dict(),
                "factor": self.factor.to_dict(), "signal": self.signal.to_dict(), "strategy": self.strategy.to_dict(),
                "context": self.context.to_dict(), "dependencies": self.declared_dependencies,
                "source_hashes": dict(sorted(self.source_hashes.items()))}

    @property
    def sha256(self):
        return canonical_hash(self.to_dict())
