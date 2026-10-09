"""Restricted, declarative composite-model adapter.

The graph is intentionally small: registered child adapters produce scores and
the final node currently combines them with a finite weighted sum.  There is
no arbitrary Python expression or dynamic import path in the graph.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterBase
from ..contracts import (CapabilityError, ConfigError, FitContext, LeakageError, ModelInputBatch,
                         ModelSpec, PredictContext, PredictError, finite_score)


class CompositeGraphAdapter(AdapterBase):
    adapter_id = "composite-graph.v1"
    adapter_version = "1.0.0"

    def __init__(self, registry):
        self.registry = registry

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "task_types": ("transform", "regression", "binary_classification",
                           "multiclass_classification", "multi_head"),
            "fit_policies": ("no_fit", "train_per_fold", "frozen_artifact"),
            "supports_missing": False,
            "supports_multi_output": False,
            "deterministic": True,
        }

    @staticmethod
    def _graph(model_spec: ModelSpec) -> dict[str, Any]:
        graph = model_spec.params.get("graph")
        if not isinstance(graph, Mapping):
            raise ConfigError("composite-graph.v1 requires params.graph")
        result = dict(graph)
        if set(result) - {"nodes", "output_node", "combine"}:
            raise ConfigError("composite graph contains unsupported operations")
        nodes = result.get("nodes")
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)) or not nodes:
            raise ConfigError("composite graph requires a non-empty nodes list")
        return result

    def validate(self, model_spec: ModelSpec, feature_ids: Sequence[str], target_field: str | None) -> None:
        super().validate(model_spec, feature_ids, target_field)
        graph = self._graph(model_spec)
        node_ids = []
        safe_ids = set()
        for node in graph["nodes"]:
            if not isinstance(node, Mapping):
                raise ConfigError("composite graph nodes must be mappings")
            node_id = str(node.get("id", ""))
            adapter_id = str(node.get("adapter", ""))
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", node_id) or not adapter_id:
                raise ConfigError("composite graph node requires id and adapter")
            if node_id in node_ids:
                raise ConfigError(f"duplicate composite graph node: {node_id}")
            node_ids.append(node_id)
            safe_id = "MOD_NODE_" + "".join(
                char if char.isalnum() else "_" for char in node_id.upper()
            ) + "_V1"
            if safe_id in safe_ids:
                raise ConfigError("composite graph node ids collide after normalization")
            safe_ids.add(safe_id)
            if any(key in node for key in ("callable", "import_path", "expression")):
                raise ConfigError("composite graph does not allow arbitrary code")
            allowed = {"id", "adapter", "input_factor_ids", "params", "fit_policy", "task_type", "target_field",
                       "target_contract", "resources", "metadata", "artifact"}
            if set(node) - allowed:
                raise CapabilityError("composite node contains an unsupported input or operation")
            if adapter_id == self.adapter_id or any(key in node for key in ("depends_on", "upstream", "inputs")):
                raise CapabilityError("v1 composite supports parallel atomic nodes and one weighted-sum output only")
            if not self.registry.has(adapter_id):
                raise ConfigError(f"composite graph references unknown adapter: {adapter_id}")
            spec = self._child_spec(model_spec.to_dict(), node)
            if model_spec.fit_policy == "no_fit" and spec.fit_policy != "no_fit":
                raise CapabilityError("a no_fit graph cannot contain a learned or frozen child")
            self.registry.resolve(adapter_id).validate(spec, feature_ids, spec.target_field)
        output_node = str(graph.get("output_node", ""))
        if output_node != "score":
            raise ConfigError("composite graph output_node must be score")
        combine = graph.get("combine")
        if not isinstance(combine, Mapping) or combine.get("method") != "weighted_sum":
            raise ConfigError("composite graph currently requires combine.method=weighted_sum")
        if set(combine) != {"method", "weights"}:
            raise ConfigError("composite aggregation only accepts method and fixed weights")
        weights = combine.get("weights")
        if not isinstance(weights, Mapping) or set(weights) != set(node_ids):
            raise ConfigError("composite graph weights must name every node exactly once")
        for value in weights.values():
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ConfigError("composite graph weights must be numeric") from error
            if not math.isfinite(number):
                raise ConfigError("composite graph weights must be finite")

    def _child_spec(self, parent: Mapping[str, Any], node: Mapping[str, Any]) -> ModelSpec:
        node_id = str(node["id"])
        safe_id = "MOD_NODE_" + "".join(char if char.isalnum() else "_" for char in node_id.upper()) + "_V1"
        params = dict(node.get("params") or {})
        fit_policy = str(node.get("fit_policy", parent.get("fit_policy", "no_fit")))
        return ModelSpec(
            id=safe_id,
            adapter=str(node["adapter"]),
            input_factor_ids=tuple(str(value) for value in (node.get("input_factor_ids") or parent.get("input_factor_ids") or ())),
            factor_set_id=parent.get("factor_set_id"),
            factor_set_config_sha256=parent.get("factor_set_config_sha256"),
            fit_policy=fit_policy,
            execution_status="executable",
            task_type=str(node.get("task_type", "transform" if fit_policy == "no_fit" else
                                   parent.get("task_type", "transform"))),
            target_field=(None if fit_policy == "no_fit" else
                          str(node["target_field"]) if node.get("target_field") is not None else
                          (str(parent["target_field"]) if parent.get("target_field") is not None else None)),
            target_contract=(None if fit_policy == "no_fit" else
                             str(node["target_contract"]) if node.get("target_contract") is not None else
                             str(parent["target_contract"]) if parent.get("target_contract") is not None else None),
            params=params,
            resources=dict(node.get("resources") or parent.get("resources") or {}),
            artifact=(dict(node["artifact"]) if node.get("artifact") is not None else
                      ({"availability": "embedded"} if fit_policy == "frozen_artifact" else None)),
            metadata=dict(node.get("metadata") or {"parameter_provenance": "composite_graph"}),
        )

    @staticmethod
    def _context(context: FitContext, spec: ModelSpec) -> FitContext:
        return replace(context, target_field=spec.target_field, feature_order=spec.input_factor_ids).with_runtime(
            model_spec=spec, registry=context.metadata["_registry"])

    @staticmethod
    def _predict_context(context: PredictContext, spec: ModelSpec) -> PredictContext:
        return context.with_runtime(model_spec=spec, registry=context.metadata["_registry"])

    def fit(self, train_batch: ModelInputBatch, fit_context: FitContext) -> Mapping[str, Any]:
        parent = fit_context.metadata.get("model_spec") or {}
        graph = self._graph(ModelSpec.from_mapping(parent))
        registry = fit_context.metadata.get("_registry") or self.registry
        children = []
        for node in graph["nodes"]:
            spec = self._child_spec(parent, node)
            adapter = registry.resolve(spec.adapter)
            adapter.validate(spec, train_batch.feature_ids, spec.target_field)
            child_context = self._context(fit_context, spec)
            child_batch = train_batch
            if spec.fit_policy == "no_fit":
                child_context = replace(child_context, target_field=None, feature_order=spec.input_factor_ids)
                child_batch = ModelInputBatch.from_rows([], spec.input_factor_ids)
            from ..runner import ModelRunner
            ModelRunner._check_resources(spec, child_batch, spec.input_factor_ids)
            historical = None
            if spec.fit_policy == "frozen_artifact":
                from ..artifacts import ArtifactStore
                root = Path(str(spec.artifact.get("root", ""))).resolve()
                handle, manifest = ArtifactStore(root.parent).load(artifact_ref=dict(spec.artifact), adapter=adapter, model_spec=spec)
                historical = manifest.get("metadata", {}).get("fit_context")
            else:
                handle = adapter.fit(child_batch, child_context)
            children.append({"id": str(node["id"]), "spec": spec.to_dict(),
                             "adapter": adapter, "handle": handle, "historical_fit_context": historical})
        return {"graph": graph, "children": children}

    def predict(self, fitted_handle: Mapping[str, Any], inference_batch: ModelInputBatch,
                predict_context: PredictContext) -> Sequence[Mapping[str, Any]]:
        child_scores: dict[str, dict[tuple[str, str], float]] = {}
        for child in fitted_handle["children"]:
            spec = ModelSpec.from_mapping(child["spec"])
            historical = child.get("historical_fit_context") or {}
            train_window = historical.get("train_window")
            if train_window and any(str(row["timestamp"])[:10] <= str(train_window[1])[:10]
                                    for row in inference_batch.rows):
                raise LeakageError("composite frozen child inference must follow its historical training window")
            adapter = child["adapter"]
            context = self._predict_context(predict_context, spec)
            rows = adapter.predict(child["handle"], inference_batch, context)
            expected_keys = {(str(row["timestamp"]), str(row["symbol"])) for row in inference_batch.rows}
            current = {}
            for row in rows:
                key = str(row["timestamp"]), str(row["symbol"])
                if key in current or key not in expected_keys:
                    raise PredictError("composite child emitted duplicate or unexpected keys")
                current[key] = finite_score(row["score"], key=f"{key[0]}|{key[1]}")
            if set(current) != expected_keys:
                raise PredictError("composite child prediction coverage is incomplete")
            child_scores[child["id"]] = current
        weights = {str(key): float(value)
                   for key, value in (fitted_handle["graph"]["combine"]["weights"]).items()}
        output = []
        for row in inference_batch.rows:
            key = (str(row["timestamp"]), str(row["symbol"]))
            score = sum(weights[node_id] * child_scores[node_id][key] for node_id in child_scores)
            output.append({"timestamp": key[0], "symbol": key[1],
                           "score": finite_score(score, key=f"{key[0]}|{key[1]}"),
                           "diagnostics": {"components": {node_id: child_scores[node_id][key] for node_id in child_scores}}})
        return output

    def dump(self, fitted_handle: Mapping[str, Any], artifact_directory: Path) -> Mapping[str, Any]:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        node_records = []
        for child in fitted_handle["children"]:
            node_dir_name = f"nodes/{child['id']}"
            node_dir = artifact_directory / node_dir_name
            adapter_meta = child["adapter"].dump(child["handle"], node_dir)
            node_records.append({"id": child["id"], "spec": child["spec"],
                                 "adapter_id": child["adapter"].adapter_id,
                                 "adapter_version": child["adapter"].adapter_version,
                                 "directory": node_dir_name, "adapter_meta": dict(adapter_meta),
                                 "historical_fit_context": child.get("historical_fit_context")})
        self.write_json(artifact_directory, "graph_state.json", {
            "graph": fitted_handle["graph"], "nodes": node_records,
        })
        return {"kind": "composite_graph", "state_file": "graph_state.json"}

    def load(self, artifact_manifest: Mapping[str, Any], artifact_directory: Path) -> Mapping[str, Any]:
        state = self.read_json(artifact_directory, "graph_state.json")
        registry = self.registry
        children = []
        for record in state.get("nodes", []):
            adapter = registry.resolve(str(record["adapter_id"]))
            if str(record.get("adapter_version")) != adapter.adapter_version:
                raise ConfigError("composite child adapter version does not match registry")
            node_dir = artifact_directory / str(record["directory"])
            if not node_dir.resolve().is_relative_to(artifact_directory.resolve() / "nodes"):
                raise ConfigError("composite child artifact path escapes nodes directory")
            handle = adapter.load(artifact_manifest, node_dir)
            children.append({"id": str(record["id"]), "spec": dict(record["spec"]),
                             "adapter": adapter, "handle": handle,
                             "historical_fit_context": record.get("historical_fit_context")})
        return {"graph": dict(state["graph"]), "children": children}
