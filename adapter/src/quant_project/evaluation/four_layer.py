"""Atomic four-layer research over explicit caller-supplied FactorVectors.

The model component is the existing bounded research service. Selection and
trading reuse the existing deterministic engines without a FactorScore shim.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from statistics import mean

from ..common import atomic_json, read_json, sha256
from ..configer.compiler import compile_experiment
from ..configer.four_layer import ResolvedExperimentV2
from ..configer.resolver import resolve_experiment
from ..modeling.attempts import ModelAttemptLedger
from ..modeling.contracts import ConfigError, CapabilityError, DataContractError, LeakageError, ModelingError, sha256_json
from ..model_score_store import load_model_scores
from ..paths import CONFIG_ROOT, RUNS_ROOT
from ..research_registry import register_research_run
from .model_research import ModelResearchService, _at, _rows_in, attempt_summary
from .research import (_has_industry_input, _point_in_time_industry_groups, _research_release_identity,
                       _warning_context, model_signal_eval, strategy_backtest)

INPUT_SCHEMA = "quant-project-four-layer-input-v2"
STAGES = {"factor", "model", "signal", "strategy", "full"}


def _schema(payload):
    from jsonschema import Draft202012Validator
    path = Path(__file__).resolve().parents[3] / "contracts/four-layer-input.schema.json"
    issue = next(Draft202012Validator(read_json(path)).iter_errors(payload), None)
    if issue:
        raise ConfigError(f"four-layer input violates its schema: {issue.message}")


class FourLayerResearchService:
    def __init__(self, *, config_root=None, reports_root=None):
        self.config_root = Path(config_root or CONFIG_ROOT).resolve()
        self.reports_root = Path(reports_root or RUNS_ROOT / "experiments").resolve()
        self.model_service = ModelResearchService(config_root=self.config_root, reports_root=self.reports_root)

    def compile(self, experiment_id, payload, *, run_id=None, stage="full"):
        try:
            return self._compile(experiment_id, payload, run_id=run_id, stage=stage)
        except ModelingError:
            raise
        except (ValueError, TypeError, OSError, KeyError) as error:
            raise ConfigError(str(error)) from error

    def _compile(self, experiment_id, payload, *, run_id=None, stage="full"):
        if stage not in STAGES:
            raise CapabilityError("unsupported four-layer stage")
        _schema(payload)
        try:
            resolved = resolve_experiment(experiment_id, self.config_root, context=payload["context"])
        except (ValueError, FileNotFoundError) as error:
            if isinstance(error, ModelingError):
                raise
            raise ConfigError(str(error)) from error
        if not isinstance(resolved, ResolvedExperimentV2):
            raise ConfigError("four-layer input requires an explicit V2 catalog composition; legacy experiments are not converted")
        context = resolved.context
        if resolved.model.metadata.get("compatibility_only") or resolved.signal.metadata.get("compatibility_only"):
            if context.metadata.get("research_mode") != "compatibility_only" or context.metadata.get("decisionEligible") is not False:
                raise CapabilityError("compatibility-only composition requires explicit research_mode=compatibility_only and decisionEligible=false")
        if not context.start or not context.end or not context.as_of:
            raise ConfigError("V2 requires context.start/end for OOS execution and an explicit context.as_of")
        if _at(context.end, end_of_day=True) > _at(context.as_of, end_of_day=True):
            raise LeakageError("execution window is after context.as_of")
        composition = compile_experiment(resolved, stage=stage)
        signal = resolved.signal.to_dict()
        # ModelResearchService's standalone comparison contract is model-agnostic.
        # Only this validated bridge removes the reference for component compile;
        # the owning plan restores and pins the exact V2 Signal/model binding.
        signal["source"] = {"kind": "model_score", "field": "score", "higher_is_better": True}
        model_input = {"schema": "quant-project-model-research-input-v1", "operation": payload.get("operation", "evaluate"),
                       "context": context.to_dict() | {"random_seed": payload["context"].get("random_seed", 0)},
                       "factor_set": resolved.factor_set.to_dict(), "models": [resolved.model.to_dict()],
                       "signal": signal, "execution_strategy": resolved.strategy.to_dict(),
                       "split_policy": payload["split_policy"], "budget": payload["budget"], "rows": payload["rows"]}
        for key in ("label", "labels", "evaluation_policy"):
            if key in payload:
                model_input[key] = payload[key]
        # The existing service pins the full release, using the same identity path
        # resolution as the deterministic engine (ID or explicit release directory).
        release = _research_release_identity(resolved)
        if context.data_mode == "point_in_time":
            model_input["context"]["research_release"] = release["root"]
            if any("available_at" not in row for row in payload["rows"]):
                raise LeakageError("PIT FactorVectors require explicit available_at for every observation")
            if resolved.factor_dependencies["fundamentals"]:
                raise CapabilityError("precomputed V2 inputs cannot verify fundamental PIT lineage; use a repository-backed factor adapter")
        compiled = self.model_service.compile(experiment_id, model_input, run_id=run_id)
        if compiled["evaluation"] and compiled["evaluation"].mode != "fixed_comparison":
            raise CapabilityError("each V2 composition pins one model; nested model selection belongs to the independent model research stage")
        if compiled["evaluation"] and compiled["evaluation"].baseline_model_id not in {None, resolved.model.id}:
            raise ConfigError("single-model V2 evaluation cannot claim an absent comparison baseline")
        oos = [row for fold in compiled["folds"] for row in _rows_in(compiled["rows"], fold["inference_window"])]
        for row in compiled["rows"]:
            try:
                canonical = date.fromisoformat(row["timestamp"]).isoformat()
            except ValueError as error:
                raise CapabilityError("V2 deterministic session execution currently requires daily ISO FactorVector timestamps") from error
            if canonical != row["timestamp"]:
                raise CapabilityError("V2 deterministic session execution currently requires daily ISO FactorVector timestamps")
        if any(not context.start <= row["timestamp"] <= context.end for row in oos):
            raise ConfigError("OOS fold observations must fit context.start/end; training rows are not clipped")
        prices = payload.get("prices", [])
        for row in prices:
            stamp = str(row.get("timestamp", row.get("date", "")))
            try:
                canonical = date.fromisoformat(stamp).isoformat()
            except ValueError as error:
                raise DataContractError("V2 market bars require daily ISO session timestamps") from error
            if canonical != stamp or stamp not in compiled["split"].sessions:
                raise DataContractError("market bar is outside the pinned SplitPolicy calendar")
        prices = [row for row in prices if context.start <= str(row.get("timestamp", row.get("date"))) <= context.end]
        actions = payload.get("corporate_actions", [])
        groups, regimes = payload.get("groups"), payload.get("regimes")
        dependencies = resolved.dependencies_for_stage(stage)
        if _has_industry_input(oos, groups):
            dependencies["industry"] = True
        if release:
            from ..pit.capabilities import validate_release_dependencies
            validate_release_dependencies(release, dependencies, data_mode=context.data_mode,
                                          as_of=context.as_of, as_of_policy=context.as_of_policy)
        if context.data_mode == "point_in_time" and dependencies["industry"]:
            groups = _point_in_time_industry_groups(resolved, release, oos, groups)
        if dependencies["industry"] and stage in {"signal", "strategy", "full"} and not _has_industry_input(oos, groups):
            raise DataContractError("V2 Signal industry dependency lacks observation-time groups")
        calendar = list(compiled["split"].sessions)
        if stage in {"strategy", "full"}:
            if not prices:
                raise DataContractError("four-layer execution requires non-empty market bars inside context.start/end")
            expected_sessions = {day for day in calendar if context.start <= day <= context.end}
            supplied_sessions = {str(row.get("timestamp", row.get("date"))) for row in prices}
            if supplied_sessions != expected_sessions:
                # The legacy event engine schedules on bar sessions. Missing an
                # entire session must not silently turn next_open into next_quote.
                raise DataContractError("V2 execution bars must cover every pinned session inside context.start/end")
            # Validate execution parameters, market rules and actions before fitting.
            try:
                strategy_backtest(resolved, [], prices, actions, trading_calendar=calendar)
            except (ValueError, TypeError, KeyError) as error:
                raise DataContractError(f"invalid execution input: {error}") from error
        plan = dict(compiled["plan"])
        plan.pop("compiled_plan_sha256")
        plan.update(schema="quant-project-compiled-four-layer-v2", stage=stage, composition=composition,
                    resolved_config_sha256=resolved.sha256, signal=resolved.signal.to_dict(),
                    context=model_input["context"], execution_engine="deterministic_event_v1",
                    actual_dependencies=dependencies, input_sha256=sha256_json(payload),
                    prices_sha256=sha256_json(prices), corporate_actions_sha256=sha256_json(actions),
                    groups_sha256=sha256_json(groups), regimes_sha256=sha256_json(regimes),
                    pipeline_runtime=self._pipeline_runtime())
        if stage == "factor":
            plan["planned_attempts_upper_bound"] = 0
        plan["compiled_plan_sha256"] = sha256_json(plan)
        compiled["plan"] = plan
        # Default run identity must include the entire four-layer/data plan.
        if run_id is None:
            compiled["run_id"] = f"four-{plan['compiled_plan_sha256'][:16]}"
            for row in compiled["rows"]:
                row["source_factor_vector_id"] = sha256_json({"research_run_id": compiled["run_id"],
                    "timestamp": row["timestamp"], "symbol": row["symbol"],
                    "factor_set_config_sha256": resolved.factor_set.config_sha256})
        compiled.update(resolved=resolved, prices=prices, corporate_actions=actions, groups=groups,
                        regimes=regimes, trading_calendar=calendar, oos=oos)
        return compiled

    @staticmethod
    def _pipeline_runtime():
        root = Path(__file__).resolve().parents[1]
        files = ["configer/models.py", "configer/four_layer.py", "configer/research_models.py", "configer/loader.py",
                 "configer/resolver.py", "configer/validator.py", "configer/compiler.py", "evaluation/contracts.py",
                 "evaluation/four_layer.py", "evaluation/model_research.py", "evaluation/research.py",
                 "model_score_store.py", "price_limits.py"]
        return {"files": {name: sha256(root / name) for name in files}}

    @staticmethod
    def _factor_report(compiled):
        factor_set = compiled["factor_set"]
        return {"schema": "quant-project-factor-vector-report-v1", "compiled_plan_sha256": compiled["plan"]["compiled_plan_sha256"],
                "factor_set": factor_set.to_dict(), "input_rows": len(compiled["rows"]), "oos_rows": len(compiled["oos"]),
                "feature_order": list(factor_set.feature_ids), "quality": {"missing": 0, "non_finite": 0, "duplicate_keys": 0},
                "values": {field: {"min": min(row[field] for row in compiled["rows"]),
                                   "max": max(row[field] for row in compiled["rows"]),
                                   "mean": mean(row[field] for row in compiled["rows"])} for field in factor_set.feature_ids},
                "note": "caller-supplied FactorVector schema/coverage report; not a factor recalculation or predictive-power conclusion"}

    def run(self, experiment_id, payload, *, run_id=None, stage="full", dry_run=False):
        compiled = self.compile(experiment_id, payload, run_id=run_id, stage=stage)
        plan, resolved, run_id = compiled["plan"], compiled["resolved"], compiled["run_id"]
        if dry_run:
            return {"status": "validated", "stage": stage, "run_id": run_id, "compiled_plan": plan}
        target = self.reports_root / experiment_id / run_id
        if target.exists():
            raise ConfigError("four-layer run exists and cannot be overwritten")
        prereg = self.reports_root / "four_layer_plans" / experiment_id / run_id
        prereg.mkdir(parents=True, exist_ok=False)
        atomic_json(prereg / "comparison_manifest.json", plan)
        ledger = ModelAttemptLedger(self.reports_root / "model_attempts.jsonl")
        with tempfile.TemporaryDirectory(prefix=".pending-four-layer-", dir=self.reports_root) as directory:
            pending = Path(directory)
            published = False
            try:
                atomic_json(pending / "compiled_plan.json", plan)
                atomic_json(pending / "comparison_manifest.json", plan)
                atomic_json(pending / "resolved_config.json", resolved.to_dict())
                atomic_json(pending / "research_input.json", payload)
                atomic_json(pending / "execution_input.json", {"prices": compiled["prices"],
                    "corporate_actions": compiled["corporate_actions"], "groups": compiled["groups"],
                    "regimes": compiled["regimes"], "trading_calendar": compiled["trading_calendar"]})
                reports = {"factor_report": "factor_report.json"}
                atomic_json(pending / "factor_report.json", self._factor_report(compiled))
                if stage == "factor":
                    self.model_service.persist_vectors(compiled, pending)
                else:
                    self.model_service.execute_into(compiled, pending, ledger)
                    reports["model_report"] = "model_report.json"
                if stage in {"signal", "strategy", "full"}:
                    scores = load_model_scores(pending / "model_scores")
                    label_map = ({f"{ts}|{symbol}": value for (ts, symbol), value in
                                 self.model_service._evaluation_labels(compiled["labels"], compiled["label"], compiled["as_of"]).items()}
                                 if compiled["label"] else None)
                    report, candidates = model_signal_eval(resolved, scores, label_map, groups=compiled["groups"],
                        regimes=compiled["regimes"], compiled_plan_sha256=plan["compiled_plan_sha256"])
                    report.update(experiment_id=experiment_id, run_id=run_id, compiled_plan_sha256=plan["compiled_plan_sha256"])
                    atomic_json(pending / "signal_report.json", report)
                    atomic_json(pending / "candidate_signals.json", [row.to_dict() for row in candidates])
                    reports["signal_report"] = "signal_report.json"
                if stage in {"strategy", "full"}:
                    report, evidence = strategy_backtest(resolved, candidates, compiled["prices"],
                        compiled["corporate_actions"], trading_calendar=compiled["trading_calendar"])
                    report.update(experiment_id=experiment_id, run_id=run_id, compiled_plan_sha256=plan["compiled_plan_sha256"])
                    atomic_json(pending / "strategy_report.json", report)
                    for name, value in evidence.items():
                        atomic_json(pending / f"{name}.json", value)
                    reports["strategy_report"] = "strategy_report.json"
                evidence = {str(index): path.relative_to(pending).as_posix() for index, path in
                            enumerate(sorted(pending.rglob("*"))) if path.is_file()}
                summary = {"schema": "quant-project-four-layer-summary-v2", "status": "success", "stage": stage,
                    "experiment_id": experiment_id, "run_id": run_id, "experiment_name": resolved.experiment.metadata.get("name"),
                    "compiled_plan_sha256": plan["compiled_plan_sha256"], "resolved_config_sha256": resolved.sha256,
                    "context_snapshot": plan["context"], "reports": reports, "run_evidence": evidence,
                    "composition": {"factor_set": resolved.factor_set.id, "model": resolved.model.id,
                                    "signal": resolved.signal.id, "strategy": resolved.strategy.id},
                    "execution": {"actual_engine": "deterministic_event_v1" if stage in {"strategy", "full"} else
                                  "factor_vector_validation_v1" if stage == "factor" else "bounded_model_research_v1",
                                  "qlib_executed": False},
                    "execution_status": "executable" if stage != "factor" else "not_executed",
                    "retention_class": "reproducible_research", "attempt_ledger": attempt_summary(ledger, run_id, plan["compiled_plan_sha256"]),
                    "warnings": _warning_context(resolved),
                    "note": "four-layer research only; historic training and H1/production state are not migrated"}
                if "decisionEligible" in resolved.context.metadata:
                    summary["decisionEligible"] = resolved.context.metadata["decisionEligible"]
                    for relative in reports.values():
                        report = read_json(pending / relative)
                        report["decisionEligible"] = summary["decisionEligible"]
                        report["research_mode"] = resolved.context.metadata.get("research_mode")
                        atomic_json(pending / relative, report)
                atomic_json(pending / "summary.json", summary)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(pending, target)
                published = True
                entry = register_research_run(target, self.reports_root)
                return {"status": "success", "stage": stage, "output": str(target), "summary": summary, "registry": entry}
            except Exception as error:
                if published:
                    os.replace(target, pending)
                failure = {"schema": "quant-project-four-layer-failure-v2", "status": "failed", "stage": stage,
                    "experiment_id": experiment_id, "run_id": run_id, "compiled_plan_sha256": plan["compiled_plan_sha256"],
                    "code": getattr(error, "code", "MODELING_ERROR"), "message": str(error),
                    "attempt_ledger": attempt_summary(ledger, run_id, plan["compiled_plan_sha256"])}
                atomic_json(prereg / "failure_result.json", failure)
                if isinstance(error, ModelingError):
                    error.details["failure_result"] = str(prereg / "failure_result.json")
                    raise
                raise ModelingError(str(error), details={"failure_result": str(prereg / "failure_result.json")}) from error
