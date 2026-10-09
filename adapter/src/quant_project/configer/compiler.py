"""Compile a resolved experiment into a deterministic research execution plan."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from ..common import atomic_json
from .models import ResolvedExperiment

SCHEMA = "quant-project-compiled-experiment-v1"
SIGNAL_METHODS = {"top_k", "top_percentile", "bottom_k", "threshold",
                  "rank_range", "dual_threshold"}
ENTRY_TIMINGS = {"same_close", "next_open", "next_close",
                 "next_session_open", "next_session_close"}


class CompilerError(ValueError):
    pass


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _callable_reference(reference: str):
    module_name, separator, attribute = str(reference or "").rpartition(".")
    if not separator:
        raise CompilerError(f"custom factor implementation is not importable: {reference!r}")
    try:
        target = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise CompilerError(f"custom factor implementation cannot be resolved: {reference!r}") from error
    if not callable(target):
        raise CompilerError(f"custom factor implementation is not callable: {reference!r}")
    return reference


def compile_experiment(resolved: ResolvedExperiment, *, stage="full", direct_candidates=False) -> dict:
    from .four_layer import ResolvedExperimentV2
    if isinstance(resolved, ResolvedExperimentV2):
        if direct_candidates:
            raise CompilerError("V2 cannot bypass ModelScore with raw candidates")
        method, timing = resolved.signal.selection.get("method"), resolved.strategy.entry.get("timing")
        if method not in SIGNAL_METHODS or timing not in ENTRY_TIMINGS:
            raise CompilerError("unsupported V2 signal selection or execution timing")
        payload = {"schema": "quant-project-compiled-experiment-v2", "experiment_id": resolved.experiment.id,
                   "experiment_name": resolved.experiment.metadata.get("name"), "stage": stage,
                   "resolved_config_sha256": resolved.sha256, "factor_set": resolved.factor_set.to_dict(),
                   "model": resolved.model.to_dict(), "signal": resolved.signal.to_dict(),
                   "execution_strategy": resolved.strategy.to_dict(), "context": resolved.context.to_dict(),
                   "handoffs": ["FactorVector", "ModelScore", "ModelCandidateSignal", "Order/Fill/AccountSnapshot"],
                   "data_access": {"dependencies": resolved.declared_dependencies,
                                   "actual_dependencies": resolved.dependencies_for_stage(stage)}}
        payload["compiled_plan_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        return payload
    factor_type = resolved.factor.type
    if factor_type == "custom":
        factor_engine = {"kind": "python_callable",
                         "implementation": _callable_reference(resolved.factor.implementation)}
    elif factor_type == "qlib_expression":
        expression = resolved.factor.metadata.get("expression") or resolved.factor.implementation
        if not expression:
            raise CompilerError("qlib_expression factor requires an expression")
        factor_engine = {"kind": "qlib_expression", "expression": str(expression)}
    else:
        raise CompilerError(f"unsupported factor type: {factor_type}")
    method = resolved.signal.selection.get("method")
    if method not in SIGNAL_METHODS:
        raise CompilerError(f"unsupported signal selection method: {method}")
    timing = resolved.strategy.entry.get("timing", "same_close")
    if timing not in ENTRY_TIMINGS:
        raise CompilerError(f"unsupported strategy entry timing: {timing}")
    context = resolved.context.to_dict()
    runtime_metadata = resolved.context.metadata
    actual_dependencies = resolved.dependencies_for_stage(stage, direct_candidates=direct_candidates)
    payload = {
        "schema": SCHEMA,
        "experiment_id": resolved.experiment.id,
        "experiment_name": resolved.experiment.metadata.get("name"),
        "resolved_config_sha256": resolved.sha256,
        "factor": {"id": resolved.factor.id, "engine": factor_engine,
                   "output_field": resolved.factor.output_field,
                   "higher_is_better": resolved.factor.higher_is_better},
        "signal": {"id": resolved.signal.id, "source_factor": resolved.signal.source_factor,
                   "builder": method, "selection": resolved.signal.selection,
                   "ranking": resolved.signal.ranking,
                   "eligibility": resolved.signal.eligibility},
        "strategy": {"id": resolved.strategy.id, "engine": "deterministic_event_v1",
                     "entry_timing": timing, "portfolio": resolved.strategy.portfolio,
                     "exit": resolved.strategy.exit, "execution": resolved.strategy.execution,
                     "accounting": resolved.strategy.accounting},
        "context": context,
        "data_access": {
            "data_mode": resolved.context.data_mode,
            "as_of": resolved.context.as_of or resolved.context.end,
            "as_of_policy": resolved.context.as_of_policy,
            "research_release_id": resolved.context.research_release,
            "dependencies": resolved.declared_dependencies,
            "actual_dependencies": actual_dependencies,
            "capabilities": runtime_metadata.get("dataset_capabilities", {}),
            "versions": runtime_metadata.get("dataset_versions", {}),
            "sources": runtime_metadata.get("dataset_sources", {}),
            "coverage": runtime_metadata.get("dataset_coverage", {}),
            "known_biases": runtime_metadata.get("known_biases", []),
        },
    }
    payload["compiled_plan_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def persist_compiled_plan(resolved: ResolvedExperiment, output) -> Path:
    path = Path(output).resolve() / "compiled_plan.json"
    atomic_json(path, compile_experiment(resolved))
    return path
