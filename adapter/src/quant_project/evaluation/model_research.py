"""Generic model research service over preregistered data, models and folds.

Every estimator invocation is made by ModelRunner in a bounded worker.
Outer predictions are sealed before evaluation labels are joined. This module
owns split validation and comparison, not candidate signals or execution.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
from datetime import date, datetime, time, timezone
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from ..common import atomic_json, read_json
from ..configer.loader import ConfigLoader, resolve_research_component
from ..configer.research_models import FactorSetSpec, LabelSpec, SplitPolicySpec, EvaluationPolicySpec
from ..modeling.attempts import ModelAttemptLedger
from ..modeling.contracts import ConfigError, DataContractError, LeakageError, ModelingError, ResourceBudgetError, sha256_json
from ..modeling.isolation import run_isolated_fold
from ..modeling.registry import build_default_registry
from ..modeling.runtime import runtime_identity
from ..model_score_store import publish_model_scores, verify_model_scores
from ..paths import CONFIG_ROOT, RUNS_ROOT
from ..research_registry import register_research_run
from .model_metrics import score_metrics

INPUT_SCHEMA = "quant-project-model-research-input-v1"
REPORT_SCHEMA = "quant-project-model-report-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _key(row):
    return str(row["timestamp"]), str(row["symbol"])


def attempt_summary(ledger, run_id, compiled_plan_sha256):
    """Run IDs are scoped by Experiment directories, not globally unique."""
    events = [event for event in ledger.events() if event["identity"].get("run_id") == run_id
              and event["identity"].get("compiled_plan_sha256") == compiled_plan_sha256]
    latest = {event["attempt_id"]: event for event in events}
    return {"attempts": len(latest), "fit_attempts": sum(bool(event["identity"].get("counts_as_fit")) for event in latest.values()),
            "succeeded": sum(event["status"] == "succeeded" for event in latest.values()),
            "failed": sum(event["status"] == "failed" for event in latest.values()),
            "unfinished": sum(event["status"] == "started" for event in latest.values())}


def _at(value, *, end_of_day=False):
    try:
        text = str(value)
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if len(text) == 10 and end_of_day:
            result = datetime.combine(date.fromisoformat(text), time.max)
        return (result.replace(tzinfo=ZoneInfo("Asia/Shanghai")) if result.tzinfo is None else result).astimezone(timezone.utc)
    except (ValueError, TypeError) as error:
        raise DataContractError(f"invalid timestamp: {value!r}") from error


def _safe(value, name):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ConfigError(f"invalid {name}")
    return value


def _component(value, loader, method):
    return resolve_research_component(value, loader, method)


def _window(fold, name, sessions, *, optional=False):
    value = fold.get(name)
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"fold {name} must specify inclusive session endpoints")
    left, right = str(value[0]), str(value[1])
    if left not in sessions or right not in sessions or sessions.index(left) > sessions.index(right):
        raise ConfigError(f"fold {name} is outside its calendar")
    return left, right


def _rows_in(rows, window):
    if window is None:
        return []
    return [row for row in rows if window[0] <= row["timestamp"][:10] <= window[1]]


class ModelResearchService:
    def __init__(self, *, config_root=None, reports_root=None):
        self.config_root = Path(config_root or CONFIG_ROOT).resolve()
        self.reports_root = Path(reports_root or RUNS_ROOT / "experiments").resolve()
        self.registry = build_default_registry()

    def compile(self, experiment_id, payload, *, run_id=None):
        if not re.fullmatch(r"EXP[0-9]+_V[1-9][0-9]*", str(experiment_id)):
            raise ConfigError("model research requires a versioned Experiment ID")
        if not isinstance(payload, dict) or payload.get("schema") != INPUT_SCHEMA:
            raise ConfigError(f"model research input schema must be {INPUT_SCHEMA}")
        if not isinstance(payload.get("budget"), dict):
            raise ResourceBudgetError("model research requires an explicit resource budget")
        from jsonschema import Draft202012Validator
        schema_path = Path(__file__).resolve().parents[3] / "contracts/model-research-input.schema.json"
        issues = sorted(Draft202012Validator(read_json(schema_path)).iter_errors(payload), key=lambda item: str(item.path))
        if issues:
            raise ConfigError(f"model research input violates its schema: {issues[0].message}")
        loader = ConfigLoader(self.config_root)
        factor_set = FactorSetSpec.from_mapping(_component(payload.get("factor_set"), loader, "factor_set"))
        model_values = payload.get("models")
        if not isinstance(model_values, list) or not model_values:
            raise ConfigError("model research requires a preregistered models array")
        models = [factor_set.bind_model(_component(value, loader, "model")) for value in model_values]
        if len({model.id for model in models}) != len(models):
            raise ConfigError("model comparison cannot repeat Model IDs")
        context = payload.get("context")
        if not isinstance(context, dict) or context.get("data_mode") not in {"point_in_time", "snapshot_compatible"}:
            raise ConfigError("model research requires an explicit context.data_mode")
        as_of = _at(context.get("as_of"), end_of_day=True)
        operation = payload.get("operation", "evaluate")
        if operation not in {"evaluate", "inference"}:
            raise ConfigError("operation must be evaluate or inference")
        label = evaluation = None
        if operation == "evaluate":
            label = LabelSpec.from_mapping(_component(payload.get("label"), loader, "label"))
            if label.field in factor_set.feature_ids:
                raise LeakageError("target field cannot also be a FactorSet input")
            evaluation = EvaluationPolicySpec.from_mapping(_component(payload.get("evaluation_policy"), loader, "evaluation_policy"))
            if len(models) > 1 and evaluation.baseline_model_id not in {model.id for model in models}:
                raise ConfigError("a model comparison requires an explicit baseline from its model list")
        elif payload.get("labels") is not None or payload.get("label") is not None or payload.get("evaluation_policy") is not None:
            raise LeakageError("inference-only research cannot consume labels or evaluation configuration")
        split = SplitPolicySpec.from_mapping(_component(payload.get("split_policy"), loader, "split_policy"))
        for session in split.sessions:
            try:
                canonical = date.fromisoformat(session).isoformat()
            except (ValueError, TypeError) as error:
                raise ConfigError("SplitPolicy sessions must use canonical ISO dates") from error
            if canonical != session:
                raise ConfigError("SplitPolicy sessions must use canonical ISO dates")
        if label and split.purge_sessions < label.horizon_sessions:
            raise LeakageError("purge_sessions must cover the declared label horizon")
        budget = payload.get("budget")
        if not isinstance(budget, dict):
            raise ResourceBudgetError("model research requires an explicit resource budget")
        for name in ("max_threads", "max_train_rows", "max_attempts"):
            if type(budget.get(name)) is not int or budget[name] <= 0:
                raise ResourceBudgetError(f"budget.{name} must be a positive integer")
        if isinstance(budget.get("max_wall_seconds"), bool) or not isinstance(budget.get("max_wall_seconds"), (int, float)) \
                or not math.isfinite(budget["max_wall_seconds"]) or budget["max_wall_seconds"] <= 0:
            raise ResourceBudgetError("budget.max_wall_seconds must be positive and finite")
        signal = _component(payload.get("signal"), loader, "signal")
        execution = _component(payload.get("execution_strategy"), loader, "strategy")
        if not re.fullmatch(r"SIG_[A-Z0-9_]+_V[1-9][0-9]*", str(signal.get("id", ""))) \
                or signal.get("source") != {"kind": "model_score", "field": "score", "higher_is_better": True}:
            raise ConfigError("model research requires a versioned ModelScore Signal contract")
        if not re.fullmatch(r"STR_[A-Z0-9_]+_V[1-9][0-9]*", str(execution.get("id", ""))):
            raise ConfigError("model research requires a versioned Execution Strategy")
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise DataContractError("model research requires a non-empty FactorVector input")
        keys = set()
        resolved_rows = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("timestamp"), str) or not row.get("symbol"):
                raise DataContractError("factor rows require timestamp/symbol")
            key = _key(row)
            if key in keys:
                raise DataContractError("duplicate FactorVector observation")
            keys.add(key)
            _at(row["timestamp"])
            if row["timestamp"][:10] not in split.sessions:
                raise DataContractError("factor observation is outside SplitPolicy calendar")
            allowed = {"timestamp", "symbol", "factor_vector_id", "available_at", *factor_set.feature_ids}
            if set(row) - allowed or any(field not in row for field in factor_set.feature_ids):
                raise DataContractError("factor rows must match FactorSet exactly and cannot contain labels")
            if row.get("available_at") and _at(row["available_at"]) > _at(row["timestamp"], end_of_day=True):
                raise LeakageError("factor observation was unavailable at prediction time")
            clean = {"timestamp": key[0], "symbol": key[1]}
            for field in factor_set.feature_ids:
                if isinstance(row[field], bool) or not isinstance(row[field], (int, float)) or not math.isfinite(row[field]):
                    raise DataContractError("FactorSet float64/fail schema rejects missing or non-finite values")
                clean[field] = float(row[field])
            resolved_rows.append(clean)
        labels = {}
        for row in payload.get("labels", []) if operation == "evaluate" else []:
            if not isinstance(row, dict) or label.field not in row or "available_at" not in row:
                raise DataContractError("labels require timestamp/symbol, target and available_at")
            key = _key(row)
            if key in labels or key not in keys:
                raise DataContractError("labels must join FactorVector keys one-to-one")
            if not isinstance(row[label.field], (int, float)) or isinstance(row[label.field], bool) \
                    or not math.isfinite(row[label.field]):
                raise DataContractError("label must be finite")
            if _at(row["available_at"]) < _at(row["timestamp"]):
                raise LeakageError("label availability precedes observation")
            horizon_end = split.sessions.index(row["timestamp"][:10]) + label.horizon_sessions
            if horizon_end >= len(split.sessions) or _at(row["available_at"], end_of_day=True) < _at(split.sessions[horizon_end]):
                raise LeakageError("label availability precedes the declared horizon maturity")
            labels[key] = dict(row)
        folds, fold_ids, prediction_keys = [], set(), set()
        prediction_sessions = set()
        for fold in split.folds:
            fold_id = _safe(fold.get("id"), "fold id")
            if fold_id in fold_ids:
                raise ConfigError("duplicate fold id")
            fold_ids.add(fold_id)
            train = _window(fold, "train_window", split.sessions, optional=True)
            validation = _window(fold, "validation_window", split.sessions, optional=True)
            inference = _window(fold, "inference_window", split.sessions)
            if _at(inference[1], end_of_day=True) > as_of:
                raise LeakageError("inference window is after the research as_of cutoff")
            self._validate_gap(train, validation or inference, split)
            if validation:
                self._validate_gap(validation, inference, split)
            current_keys = {_key(row) for row in _rows_in(resolved_rows, inference)}
            current_sessions = set(split.sessions[split.sessions.index(inference[0]):split.sessions.index(inference[1]) + 1])
            if not current_keys or prediction_keys.intersection(current_keys) or prediction_sessions.intersection(current_sessions):
                raise LeakageError("OOS folds overlap or have no prediction observations")
            prediction_keys.update(current_keys)
            prediction_sessions.update(current_sessions)
            if any(model.fit_policy == "train_per_fold" for model in models) and train is None:
                raise ConfigError("learned models require an explicit train window for every fold")
            first_prediction = (validation or inference)[0]
            cutoff = str(fold.get("train_cutoff") or (split.sessions[split.sessions.index(first_prediction) - 1]
                                                       if train else inference[0]))
            if train and (cutoff[:10] < train[1] or cutoff[:10] >= first_prediction):
                raise LeakageError("train_cutoff must follow training observations and precede validation/inference")
            inner_folds = []
            if evaluation and evaluation.mode == "nested_walk_forward":
                inner = fold.get("inner_folds")
                if not isinstance(inner, (tuple, list)) or not inner or train is None:
                    raise ConfigError("nested walk-forward requires preregistered inner folds")
                inner_ids, inner_keys = set(), set()
                for item in inner:
                    iid = _safe(item.get("id"), "inner fold id")
                    if iid in inner_ids:
                        raise ConfigError("duplicate inner fold id")
                    inner_ids.add(iid)
                    itrain = _window(item, "train_window", split.sessions)
                    itest = _window(item, "inference_window", split.sessions)
                    if itrain[0] < train[0] or itest[1] > train[1]:
                        raise LeakageError("inner folds must stay within their outer training window")
                    self._validate_gap(itrain, itest, split)
                    ikeys = {_key(row) for row in _rows_in(resolved_rows, itest)}
                    if not ikeys or ikeys.intersection(inner_keys):
                        raise LeakageError("inner validation coverage overlaps or is empty")
                    inner_keys.update(ikeys)
                    inner_cutoff = str(item.get("train_cutoff") or split.sessions[split.sessions.index(itest[0]) - 1])
                    if inner_cutoff[:10] < itrain[1] or inner_cutoff[:10] >= itest[0]:
                        raise LeakageError("inner train_cutoff is outside its authorized interval")
                    inner_folds.append({"id": f"{fold_id}.{iid}", "train_window": itrain,
                                        "inference_window": itest, "validation_window": None,
                                        "train_cutoff": inner_cutoff})
            folds.append({"id": fold_id, "train_window": train, "validation_window": validation,
                          "inference_window": inference, "train_cutoff": cutoff, "inner_folds": inner_folds})
        attempt_count = len(models) * sum(len(fold["inner_folds"]) for fold in folds)
        attempt_count += (len(folds) * 2 if evaluation and evaluation.mode == "nested_walk_forward"
                          and evaluation.baseline_model_id else len(folds) if evaluation and evaluation.mode == "nested_walk_forward"
                          else len(models) * len(folds))
        if attempt_count > budget["max_attempts"]:
            raise ResourceBudgetError("preregistered attempts exceed budget.max_attempts")
        for model in models:
            adapter = self.registry.resolve(model.adapter)
            target = label.field if label and model.fit_policy == "train_per_fold" else model.target_field
            if model.fit_policy == "train_per_fold" and (not label or model.target_field != label.field
                                                        or model.target_contract != label.id):
                raise ConfigError("learned ModelSpec must bind an explicit target field and contract")
            for node in model.params.get("graph", {}).get("nodes", ()):
                if node.get("fit_policy", model.fit_policy) == "train_per_fold" and (
                        not label or node.get("target_field", model.target_field) != label.field
                        or node.get("target_contract", model.target_contract) != label.id):
                    raise ConfigError("v1 model research graph nodes must share the preregistered LabelSpec")
            adapter.validate(model, factor_set.feature_ids, target)
            if model.adapter == "identity.v1" and model.params["higher_is_better"] != factor_set.factors[0]["higher_is_better"]:
                raise ConfigError("Identity direction must match its resolved input factor")
            if model.adapter in {"fixed-linear.v1", "fixed-rank-blend.v1"}:
                directions = model.params.get("directions", {})
                if dict(directions) != {item["id"]: item["higher_is_better"] for item in factor_set.factors}:
                    raise ConfigError("fixed model directions must explicitly match every resolved input factor")
            if model.fit_policy == "frozen_artifact" and not model.artifact.get("manifest_sha256"):
                raise ConfigError("frozen model research requires a pinned artifact manifest SHA-256")
            parameter_sets = [model.params, *(node.get("params", {}) for node in model.params.get("graph", {}).get("nodes", ()))]
            for params in parameter_sets:
                n_jobs = params.get("n_jobs", 1)
                if type(n_jobs) is not int or not 1 <= n_jobs <= budget["max_threads"]:
                    raise ResourceBudgetError("ModelSpec or graph-node parallelism exceeds research budget")
            for fold in folds:
                train_rows = _rows_in(resolved_rows, fold["train_window"])
                row_cap = min(budget["max_train_rows"], int(model.resources.get("max_train_rows", budget["max_train_rows"])))
                if len(train_rows) > row_cap:
                    raise ResourceBudgetError("fold exceeds max_train_rows")
                if model.fit_policy == "train_per_fold":
                    if not train_rows:
                        raise DataContractError("training fold is empty")
                    self._training_labels(train_rows, labels, label, fold["train_cutoff"])
                    for inner in fold["inner_folds"]:
                        self._training_labels(_rows_in(resolved_rows, inner["train_window"]), labels, label, inner["train_cutoff"])
        release = context.get("research_release")
        release_identity = None
        if context["data_mode"] == "point_in_time":
            if not release:
                raise ConfigError("point_in_time model research requires a pinned Research Release")
            from ..research_release import verify_research_release
            release_identity = verify_research_release(release)
        else:
            release_identity = {"release_id": release, "data_mode": "snapshot_compatible"}
        optional_models = any(model.adapter.startswith("sklearn-") or any(
            str(node.get("adapter", "")).startswith("sklearn-") for node in model.params.get("graph", {}).get("nodes", ())) for model in models)
        runtime = runtime_identity(optional_models=optional_models)
        plan = {"schema": "quant-project-compiled-model-research-v1", "experiment_id": experiment_id,
                "operation": operation, "factor_set": factor_set.to_dict(), "models": [model.to_dict() for model in models],
                "label": label.to_dict() if label else None, "split_policy": split.to_dict(),
                "evaluation_policy": evaluation.to_dict() if evaluation else None,
                "signal": signal, "execution_strategy": execution, "context": context,
                "release_identity": release_identity, "runtime": runtime, "budget": budget,
                "data_sha256": sha256_json(resolved_rows), "labels_sha256": sha256_json(payload.get("labels", [])),
                "planned_attempts_upper_bound": attempt_count}
        plan["compiled_plan_sha256"] = sha256_json(plan)
        run_id = _safe(run_id or f"model-{plan['compiled_plan_sha256'][:16]}", "run id")
        factor_set_config_sha256 = factor_set.config_sha256
        for row in resolved_rows:
            row["source_factor_vector_id"] = sha256_json({"research_run_id": run_id, "timestamp": row["timestamp"],
                                                        "symbol": row["symbol"], "factor_set_config_sha256": factor_set_config_sha256})
        return {"plan": plan, "run_id": run_id, "factor_set": factor_set, "models": models,
                "label": label, "split": split, "evaluation": evaluation, "rows": resolved_rows,
                "labels": labels, "folds": folds, "as_of": as_of}

    @staticmethod
    def _validate_gap(train, inference, split):
        if train is None:
            return
        gap = split.sessions.index(inference[0]) - split.sessions.index(train[1]) - 1
        if gap < max(split.purge_sessions, split.embargo_sessions):
            raise LeakageError("fold train/validation/inference windows overlap or violate purge/embargo")

    @staticmethod
    def _training_labels(rows, labels, label, cutoff):
        for row in rows:
            value = labels.get(_key(row))
            if value is None or _at(value["available_at"]) > _at(cutoff, end_of_day=True):
                raise LeakageError("a training label is missing or unavailable at the fold cutoff")
        return [dict(row, **{label.field: labels[_key(row)][label.field]}) for row in rows]

    @staticmethod
    def _evaluation_labels(labels, label, cutoff):
        return {key: row[label.field] for key, row in labels.items() if _at(row["available_at"]) <= cutoff}

    def run(self, experiment_id, payload, *, run_id=None, dry_run=False):
        compiled = self.compile(experiment_id, payload, run_id=run_id)
        plan, run_id = compiled["plan"], compiled["run_id"]
        if dry_run:
            return {"status": "validated", "stage": "model", "run_id": run_id, "compiled_plan": plan}
        target = self.reports_root / experiment_id / run_id
        if target.exists():
            raise ConfigError("model research run exists and cannot be overwritten")
        plans_root = self.reports_root / "model_plans" / experiment_id / run_id
        plans_root.mkdir(parents=True, exist_ok=False)
        atomic_json(plans_root / "comparison_manifest.json", plan)
        ledger = ModelAttemptLedger(self.reports_root / "model_attempts.jsonl")
        with tempfile.TemporaryDirectory(prefix=".pending-model-", dir=self.reports_root) as pending:
            pending = Path(pending)
            try:
                result = self.execute_into(compiled, pending, ledger)
                atomic_json(pending / "compiled_plan.json", plan)
                atomic_json(pending / "comparison_manifest.json", plan)
                atomic_json(pending / "resolved_config.json", plan)
                run_evidence = {str(index): path.relative_to(pending).as_posix()
                                for index, path in enumerate(sorted(pending.rglob("*"))) if path.is_file()}
                summary = {"schema": "quant-project-model-run-summary-v1", "status": "success", "stage": "model",
                           "experiment_id": experiment_id, "run_id": run_id, "operation": plan["operation"],
                           "context_snapshot": plan["context"], "compiled_plan_sha256": plan["compiled_plan_sha256"],
                           "reports": {"model_report": "model_report.json"}, "run_evidence": run_evidence,
                           "execution_status": "executable", "retention_class": "reproducible_research",
                           "attempt_ledger": attempt_summary(ledger, run_id, plan["compiled_plan_sha256"])}
                atomic_json(pending / "summary.json", summary)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(pending, target)
                entry = register_research_run(target, self.reports_root)
                return {"status": "success", "stage": "model", "output": str(target), "summary": summary,
                        "registry": entry, "report": result["report"]}
            except Exception as error:
                if target.exists() and not (pending / "summary.json").exists():
                    # Publication is only successful after Registry accepts
                    # it. Put an unregistered directory back into the pending
                    # area so the failure cleanup cannot leave success output.
                    os.replace(target, pending)
                failure = {"schema": "quant-project-model-failure-v1", "status": "failed", "stage": "model",
                           "experiment_id": experiment_id, "run_id": run_id,
                           "compiled_plan_sha256": plan["compiled_plan_sha256"],
                           "code": getattr(error, "code", "MODELING_ERROR"), "message": str(error),
                           "attempt_ledger": attempt_summary(ledger, run_id, plan["compiled_plan_sha256"])}
                atomic_json(plans_root / "failure_result.json", failure)
                if isinstance(error, ModelingError):
                    error.details["failure_result"] = str(plans_root / "failure_result.json")
                    raise
                raise ModelingError(str(error), details={"failure_result": str(plans_root / "failure_result.json")}) from error

    @staticmethod
    def persist_vectors(compiled, pending):
        factor_set = compiled["factor_set"]
        factor_set_config_sha256 = factor_set.config_sha256
        atomic_json(Path(pending) / "factor_vectors.json", {"schema": "quant-project-factor-vector-dataset-v1",
                    "factor_set": compiled["plan"]["factor_set"], "rows": compiled["rows"],
                    "vectors": [{"factor_vector_id": row["source_factor_vector_id"], "timestamp": row["timestamp"],
                                 "symbol": row["symbol"], "factor_set_id": compiled["factor_set"].id,
                                 "factor_set_config_sha256": factor_set_config_sha256,
                                 "values": {field: row[field] for field in compiled["factor_set"].feature_ids}}
                                for row in compiled["rows"]]})

    def execute_into(self, compiled, pending, ledger):
        """Execute/seal the model component in an owning run's staging area.

        This does not publish or register an independent successful run. The
        caller owns atomic publication only after its downstream stages pass.
        """
        pending = Path(pending)
        result = self._execute(compiled, pending, ledger)
        atomic_json(pending / "model_report.json", result["report"])
        publish_model_scores(pending / "model_scores", result["scores"], run_id=compiled["run_id"],
                             compiled_plan_sha256=compiled["plan"]["compiled_plan_sha256"])
        verify_model_scores(pending / "model_scores")
        atomic_json(pending / "model_bundle.json", result["bundle"])
        self.persist_vectors(compiled, pending)
        return result

    def _execute(self, compiled, pending, ledger):
        plan = compiled["plan"]
        all_scores, observations, members, selection = [], [], [], []
        models = compiled["models"]
        evaluation, label, split = compiled["evaluation"], compiled["label"], compiled["split"]
        for outer in compiled["folds"]:
            outer_models = models
            if evaluation and evaluation.mode == "nested_walk_forward":
                candidate_values = {}
                for model in models:
                    values = []
                    for inner in outer["inner_folds"]:
                        result = self._invoke(compiled, model, inner, pending, ledger, purpose="inner_selection")
                        available = self._evaluation_labels(compiled["labels"], label, _at(outer["train_cutoff"], end_of_day=True))
                        metrics = score_metrics(result["scores"], available, evaluation.metric)
                        self._check_coverage(metrics, evaluation)
                        if metrics[evaluation.metric] is None:
                            raise DataContractError("inner selection metric is undefined")
                        values.append(metrics[evaluation.metric])
                    candidate_values[model.id] = mean(values)
                reverse = evaluation.metric == "rank_ic"
                chosen_id = sorted(candidate_values, key=lambda key: ((-1 if reverse else 1) * candidate_values[key], key))[0]
                selection.append({"outer_fold_id": outer["id"], "selected_model_id": chosen_id,
                                  "inner_metrics": candidate_values, "selection_cutoff": outer["train_cutoff"]})
                outer_models = [next(model for model in models if model.id == chosen_id)]
                if evaluation.baseline_model_id and evaluation.baseline_model_id != chosen_id:
                    outer_models.append(next(model for model in models if model.id == evaluation.baseline_model_id))
            for model in outer_models:
                result = self._invoke(compiled, model, outer, pending, ledger, purpose="outer_oos")
                all_scores.extend(result["scores"])
                artifact = result["artifact"]
                member = {"fold_id": outer["id"], "model_id": model.id,
                          "model_identity_sha256": model.model_identity_sha256,
                          "artifact_id": artifact["artifact_id"], "manifest_sha256": artifact["manifest_sha256"],
                          "payload_sha256": artifact["payload_sha256"],
                          "path": Path(artifact["root"]).relative_to(pending).as_posix()
                          if Path(artifact["root"]).is_relative_to(pending) else None,
                          "external_root": artifact["root"] if not Path(artifact["root"]).is_relative_to(pending) else None}
                members.append(member)
                if label:
                    available = self._evaluation_labels(compiled["labels"], label, compiled["as_of"])
                    metrics = score_metrics(result["scores"], available, evaluation.metric)
                    self._check_coverage(metrics, evaluation)
                else:
                    metrics = {"rows": len(result["scores"]), "evaluated_rows": 0}
                observations.append({"model_id": model.id, "fold_id": outer["id"], "metrics": metrics,
                                     "prediction_coverage": 1.0,
                                     "label_coverage": metrics["evaluated_rows"] / metrics["rows"],
                                     "wall_seconds": result["wall_seconds"]})
        by_model = {}
        for model in models:
            values = [row for row in observations if row["model_id"] == model.id]
            if not values:
                continue
            metric_values = [row["metrics"].get(evaluation.metric) for row in values] if evaluation else []
            defined = [value for value in metric_values if value is not None]
            by_model[model.id] = {"folds": len(values), "prediction_rows": sum(row["metrics"]["rows"] for row in values),
                                  "evaluated_rows": sum(row["metrics"]["evaluated_rows"] for row in values),
                                  "mean_fold_metric": mean(defined) if defined else None,
                                  "defined_metric_folds": len(defined)}
        comparison = []
        if evaluation and evaluation.baseline_model_id and evaluation.mode == "fixed_comparison":
            baseline = by_model.get(evaluation.baseline_model_id, {}).get("mean_fold_metric")
            for model_id, record in by_model.items():
                value = record["mean_fold_metric"]
                comparison.append({"model_id": model_id, "baseline_model_id": evaluation.baseline_model_id,
                                   "metric": evaluation.metric,
                                   "delta": value - baseline if value is not None and baseline is not None else None})
        selected_pipeline = None
        if selection:
            selected_by_fold = {row["outer_fold_id"]: row["selected_model_id"] for row in selection}
            selected_results = [row for row in observations if selected_by_fold[row["fold_id"]] == row["model_id"]]
            values = [row["metrics"][evaluation.metric] for row in selected_results
                      if row["metrics"][evaluation.metric] is not None]
            selected_pipeline = {"folds": len(selected_results), "mean_fold_metric": mean(values) if values else None,
                                 "prediction_rows": sum(row["metrics"]["rows"] for row in selected_results)}
            if evaluation.baseline_model_id:
                baseline = by_model[evaluation.baseline_model_id]["mean_fold_metric"]
                selected_mean = selected_pipeline["mean_fold_metric"]
                comparison.append({"model_id": "nested_selected_pipeline", "baseline_model_id": evaluation.baseline_model_id,
                                   "metric": evaluation.metric,
                                   "delta": selected_mean - baseline if selected_mean is not None and baseline is not None else None})
        report = {"schema": REPORT_SCHEMA, "status": "success", "experiment_id": plan["experiment_id"],
                  "run_id": compiled["run_id"], "operation": plan["operation"], "compiled_plan_sha256": plan["compiled_plan_sha256"],
                  "factor_set_id": compiled["factor_set"].id, "model_ids": [model.id for model in models],
                  "label": plan["label"], "split_policy": plan["split_policy"], "evaluation_policy": plan["evaluation_policy"],
                  "runtime": plan["runtime"], "data_sha256": plan["data_sha256"],
                  "fold_results": observations, "model_results": by_model, "comparisons": comparison,
                  "inner_selection": selection, "selected_pipeline": selected_pipeline,
                  "attempt_ledger": attempt_summary(ledger, compiled["run_id"], plan["compiled_plan_sha256"]),
                  "warnings": ([{"code": "SNAPSHOT_COMPATIBLE_ONLY"}] if plan["context"]["data_mode"] == "snapshot_compatible" else []),
                  "artifacts": {"model_scores": "model_scores/manifest.json", "bundle": "model_bundle.json"}}
        bundle = {"schema": "quant-project-model-artifact-bundle-v1", "sealed": True,
                  "run_id": compiled["run_id"], "compiled_plan_sha256": plan["compiled_plan_sha256"], "members": members,
                  "retention_class": "reproducible_research", "execution_status": "executable"}
        bundle["bundle_sha256"] = sha256_json(bundle)
        return {"scores": all_scores, "report": report, "bundle": bundle}

    @staticmethod
    def _check_coverage(metrics, evaluation):
        if not metrics["rows"] or metrics["evaluated_rows"] / metrics["rows"] < evaluation.minimum_label_coverage:
            raise DataContractError("mature evaluation label coverage is below the preregistered minimum")

    def _invoke(self, compiled, model, fold, pending, ledger, *, purpose):
        train = _rows_in(compiled["rows"], fold["train_window"])
        if model.fit_policy == "train_per_fold":
            train = self._training_labels(train, compiled["labels"], compiled["label"], fold["train_cutoff"])
        else:
            train = None
        inference = _rows_in(compiled["rows"], fold["inference_window"])
        context = {"run_id": compiled["run_id"], "fold_id": fold["id"],
                   "data_mode": compiled["plan"]["context"]["data_mode"],
                   "inference_window": fold["inference_window"], "feature_order": list(model.input_factor_ids),
                   "input_data_sha256": sha256_json(inference)}
        fit_context = context | {"train_window": fold["train_window"] if model.fit_policy != "no_fit" else None,
                                 "validation_window": fold["validation_window"],
                                 "purge_sessions": compiled["split"].purge_sessions,
                                 "embargo_sessions": compiled["split"].embargo_sessions,
                                 "random_seed": int(compiled["plan"]["context"].get("random_seed", 0)),
                                 "target_field": model.target_field,
                                 "metadata": {"compiled_plan_sha256": compiled["plan"]["compiled_plan_sha256"],
                                              "resource_budget": compiled["plan"]["budget"],
                                              "label": compiled["plan"]["label"],
                                              "split_policy_id": compiled["split"].id,
                                              "split_policy_sha256": sha256_json(compiled["plan"]["split_policy"]),
                                              "evaluation_policy": compiled["plan"]["evaluation_policy"]}}
        fit_context["input_data_sha256"] = sha256_json(train or [])
        predict_context = dict(context)
        if compiled["label"]:
            predict_context["metadata"] = {"target_fields": [compiled["label"].field]}
        store = pending / "model_artifacts"
        if model.fit_policy == "frozen_artifact":
            artifact_root = Path(str(model.artifact.get("root", ""))).resolve()
            store = Path(str(model.artifact.get("store_root", artifact_root.parent))).resolve()
        identity = {"run_id": compiled["run_id"], "compiled_plan_sha256": compiled["plan"]["compiled_plan_sha256"],
                    "model_id": model.id, "model_config_sha256": model.config_sha256, "fold_id": fold["id"],
                    "purpose": purpose, "train_sha256": sha256_json(train or []),
                    "inference_sha256": sha256_json(inference), "random_seed": fit_context["random_seed"],
                    "counts_as_fit": model.fit_policy == "train_per_fold"}
        event = ledger.append(identity, "started")
        request = {"model": model.to_dict(), "train_rows": train, "inference_rows": inference,
                   "fit_context": fit_context, "predict_context": predict_context, "artifact_store": str(store)}
        budget = compiled["plan"]["budget"]
        try:
            result = run_isolated_fold(request, pending / "attempts" / event["attempt_id"],
                                       max_wall_seconds=budget["max_wall_seconds"], max_threads=budget["max_threads"])
            prediction_hash = sha256_json(result["scores"])
            atomic_json(pending / "attempts" / event["attempt_id"] / "frozen_predictions.json",
                        {"sealed": True, "prediction_rows_sha256": prediction_hash, "rows": result["scores"]})
            ledger.append(identity, "succeeded", artifact_manifest_sha256=result["artifact"]["manifest_sha256"],
                          prediction_rows_sha256=prediction_hash)
            return result
        except BaseException as error:
            ledger.append(identity, "failed", code=getattr(error, "code", "MODELING_ERROR"), message=str(error))
            raise


def model_eval(experiment_id, payload, *, config_root=None, reports_root=None, run_id=None, dry_run=False):
    return ModelResearchService(config_root=config_root, reports_root=reports_root).run(
        experiment_id, payload, run_id=run_id, dry_run=dry_run)
