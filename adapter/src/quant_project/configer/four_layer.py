"""V2 four-layer composition; no implicit conversion of legacy experiments."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .models import BacktestContext, ConfigError, StrategySpec, canonical_hash, _id, _mapping, _jsonable, _SIGNAL, _STRATEGY, _EXPERIMENT
from .research_models import FactorSetSpec
from ..modeling.contracts import ModelSpec
from ..pit.contracts import merge_dependencies, normalize_dependencies

_FACTOR_SET = re.compile(r"^FSET_[A-Z0-9_]+_V[1-9][0-9]*$")
_MODEL = re.compile(r"^MOD_[A-Z0-9_]+_V[1-9][0-9]*$")


@dataclass(frozen=True)
class ModelSignalSpec:
    id: str
    source_model: str
    selection: dict[str, Any]
    ranking: dict[str, Any]
    direction: dict[str, Any]
    eligibility: dict[str, Any]
    dependencies: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = _mapping(value, "ModelSignalSpec")
        source = _mapping(value.get("source"), "ModelSignalSpec.source")
        if set(source) != {"kind", "model", "field", "higher_is_better"} or (
                source.get("kind") != "model_score" or source.get("field") != "score"
                or source.get("higher_is_better") is not True):
            raise ConfigError("V2 Signal source must explicitly bind a ModelScore model, score field and higher_is_better=true")
        eligibility = _mapping(value.get("eligibility", {}), "ModelSignalSpec.eligibility")
        ranking = _mapping(value.get("ranking", {}), "ModelSignalSpec.ranking")
        if set(ranking) - {"order"}:
            raise ConfigError("unsupported V2 ranking options")
        if ranking.get("order", "descending") not in {"descending", "desc"}:
            raise ConfigError("ModelScore is already direction-normalized; V2 ranking must be descending")
        dependencies = normalize_dependencies(value.get("dependencies"), defaults={
            "industry": bool(eligibility.get("known_industry_max_positions") or eligibility.get("max_per_industry"))})
        selection = _mapping(value.get("selection"), "ModelSignalSpec.selection")
        _validate_selection(selection)
        if set(eligibility) - {"known_industry_max_positions", "missing_industry_is_independent"}:
            raise ConfigError("unsupported V2 eligibility options")
        direction = _mapping(value.get("direction", {}), "ModelSignalSpec.direction")
        if set(direction) - {"side"} or str(direction.get("side", "long")).upper() not in {"LONG", "SHORT"}:
            raise ConfigError("V2 signal direction supports only LONG or SHORT side")
        return cls(_id(value.get("id"), _SIGNAL, "ModelSignalSpec"),
                   _id(source.get("model"), _MODEL, "ModelSignalSpec.source.model"),
                   selection, ranking, direction,
                   eligibility, dependencies, _mapping(value.get("metadata", {}), "ModelSignalSpec.metadata"))

    def to_dict(self):
        return {"id": self.id, "source": {"kind": "model_score", "model": self.source_model,
                "field": "score", "higher_is_better": True}, "selection": _jsonable(self.selection),
                "ranking": _jsonable(self.ranking), "direction": _jsonable(self.direction),
                "eligibility": _jsonable(self.eligibility), "dependencies": _jsonable(self.dependencies),
                "metadata": _jsonable(self.metadata)}


def _validate_selection(selection):
    method = selection.get("method")
    options = {"top_k": {"top_k"}, "bottom_k": {"bottom_k"},
               "top_percentile": {"value", "top_percentile"}, "rank_range": {"min_rank", "max_rank"},
               "threshold": {"threshold", "operator"}, "dual_threshold": {"long_threshold", "short_threshold"}}
    if method not in options or set(selection) - {"method", *options[method]}:
        raise ConfigError("unsupported V2 signal selection method/options")
    if method in {"top_k", "bottom_k"}:
        if type(selection.get(method)) is not int or selection[method] < 1:
            raise ConfigError(f"V2 {method} requires a positive integer")
    elif method == "rank_range":
        minimum, maximum = selection.get("min_rank", 1), selection.get("max_rank")
        if type(minimum) is not int or minimum < 1 or type(maximum) is not int or maximum < minimum:
            raise ConfigError("V2 rank_range requires explicit 1 <= min_rank <= max_rank")
    else:
        fields = ("long_threshold", "short_threshold") if method == "dual_threshold" else ("threshold",) if method == "threshold" else ()
        values = [selection.get(name) for name in fields]
        if method == "top_percentile":
            if "value" in selection and "top_percentile" in selection:
                raise ConfigError("V2 top_percentile cannot specify conflicting aliases")
            values = [selection.get("value", selection.get("top_percentile"))]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ConfigError("V2 signal thresholds/percentiles must be finite numbers")
        if method == "top_percentile" and not 0 < values[0] <= 1:
            raise ConfigError("V2 top_percentile must be in (0, 1]")
        if method == "threshold" and selection.get("operator", "gte") not in {"gte", "lte"}:
            raise ConfigError("V2 threshold operator must be gte or lte")
        if method == "dual_threshold" and values[1] >= values[0]:
            raise ConfigError("V2 dual_threshold requires short_threshold < long_threshold")


@dataclass(frozen=True)
class ExperimentSpecV2:
    id: str
    factor_set: str
    model: str
    signal: str
    strategy: str
    context: BacktestContext
    dependencies: dict[str, bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = _mapping(value, "ExperimentSpecV2")
        if value.get("schema_version") != "stock-experiment-v2":
            raise ConfigError("ExperimentSpecV2 requires stock-experiment-v2")
        return cls(_id(value.get("id"), _EXPERIMENT, "ExperimentSpecV2"),
                   _id(value.get("factor_set"), _FACTOR_SET, "ExperimentSpecV2.factor_set"),
                   _id(value.get("model"), _MODEL, "ExperimentSpecV2.model"),
                   _id(value.get("signal"), _SIGNAL, "ExperimentSpecV2.signal"),
                   _id(value.get("strategy"), _STRATEGY, "ExperimentSpecV2.strategy"),
                   BacktestContext.from_dict(value.get("context")),
                   normalize_dependencies(value["dependencies"]) if value.get("dependencies") is not None else None,
                   _mapping(value.get("metadata", {}), "ExperimentSpecV2.metadata"))

    def to_dict(self):
        return {"schema_version": "stock-experiment-v2", "id": self.id, "factor_set": self.factor_set,
                "model": self.model, "signal": self.signal, "strategy": self.strategy,
                "context": self.context.to_dict(), "dependencies": _jsonable(self.dependencies),
                "metadata": _jsonable(self.metadata)}


@dataclass(frozen=True)
class ResolvedExperimentV2:
    experiment: ExperimentSpecV2
    factor_set: FactorSetSpec
    model: ModelSpec
    signal: ModelSignalSpec
    strategy: StrategySpec
    context: BacktestContext
    source_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def factor_dependencies(self):
        return merge_dependencies(*(normalize_dependencies(item.get("dependencies"), defaults={"market_data": True})
                                    for item in self.factor_set.factors))

    def dependencies_for_stage(self, stage="full", *, direct_candidates=False):
        if direct_candidates:
            raise ConfigError("V2 cannot bypass its pinned ModelScore/Signal chain with raw candidates")
        if stage not in {"factor", "model", "signal", "strategy", "full"}:
            raise ConfigError(f"unknown research stage: {stage!r}")
        parts = [self.factor_dependencies]
        if stage in {"signal", "strategy", "full"}:
            parts.append(self.signal.dependencies)
        if stage in {"strategy", "full"}:
            parts.append(self.strategy.dependencies)
        return merge_dependencies(*parts)

    @property
    def declared_dependencies(self):
        return merge_dependencies(self.factor_dependencies, self.signal.dependencies,
                                  self.strategy.dependencies, self.experiment.dependencies or {})

    def to_dict(self):
        return {"schema_version": "stock-resolved-experiment-v2", "experiment": self.experiment.to_dict(),
                "factor_set": self.factor_set.to_dict(), "model": self.model.to_dict(),
                "signal": self.signal.to_dict(), "strategy": self.strategy.to_dict(),
                "context": self.context.to_dict(), "dependencies": self.declared_dependencies,
                "source_hashes": dict(sorted(self.source_hashes.items()))}

    @property
    def sha256(self):
        return canonical_hash(self.to_dict())
